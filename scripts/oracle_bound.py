#!/usr/bin/env python
"""Oracle upper bound on n-best rescoring (Phase B analysis).

For the SAME seeded test split used by rescore_nbest.py, this computes the best
WER any rescorer could achieve from the cached n-best -- i.e. if an oracle always
picked the lowest-WER hypothesis in the list. Comparing

    baseline (top-1 acoustic)  <  GPT-2 adapted  <  oracle (best in n-best)

quantifies (a) how much head-room the n-best contains and (b) what fraction of
that head-room the text LM actually captures. It also reports the oracle as a
function of n-best depth N, showing where the recoverable error saturates.

Pure analysis: no audio, no LM, no GPU -- only the cached hypotheses + references.

Usage:
  python scripts/oracle_bound.py --dataset l2arctic_24spk
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.io.manifest import read_manifest                 # noqa: E402
from asr_l2.scoring.metrics import corpus_wer, score_pair    # noqa: E402


def reproduce_split(cfg, manifest: Path, nbest: dict, dev_frac: float,
                    min_dur: float, max_utts: int):
    """Identical construction to rescore_nbest.main(): manifest order, min-dur
    filter, cap, keep those with cached n-best, seeded shuffle, dev=first frac."""
    utts = list(read_manifest(manifest))
    if min_dur and min_dur > 0:
        utts = [u for u in utts if u.duration_s >= min_dur]
    if max_utts is not None:
        utts = utts[:max_utts]
    by_id = {u.utt_id: u for u in utts}
    ids = [u.utt_id for u in utts if u.utt_id in nbest]
    rng = random.Random(cfg.get("seed", 1234))
    rng.shuffle(ids)
    n_dev = max(1, int(len(ids) * dev_frac))
    return by_id, set(ids[:n_dev]), set(ids[n_dev:])


def oracle_pick(ref: str, hyps: list, depth: int) -> str:
    """Lowest-WER hypothesis within the top-`depth` (by acoustic score)."""
    ranked = sorted(hyps, key=lambda h: -h[1])[:depth]        # h = [text, ac]
    return min(ranked, key=lambda h: score_pair(ref, h[0])["wer"])[0]


def best_acoustic(hyps: list) -> str:
    return max(hyps, key=lambda h: h[1])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l2arctic_24spk")
    ap.add_argument("--dev-frac", type=float, default=0.34)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    rdir = Path(cfg["paths"]["results_root"]) / args.dataset
    manifest = data_root / "processed" / args.dataset / "manifest.jsonl"

    nbest = {}
    with open(rdir / "phaseB_nbest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            nbest[d["utt_id"]] = d["hyps"]
    max_n = max(len(v) for v in nbest.values())

    # use a cap large enough to include every clip (the 24-spk run used all)
    by_id, dev_ids, test_ids = reproduce_split(
        cfg, manifest, nbest, args.dev_frac,
        cfg["eval"].get("min_duration_s", 0.0), max_utts=10 ** 9)
    test = sorted(test_ids)

    refs = [by_id[u].text for u in test]
    base_hyps = [best_acoustic(nbest[u]) for u in test]
    base_wer = corpus_wer(refs, base_hyps)["wer"]

    # oracle vs n-best depth
    depth_curve = {}
    for depth in [1, 2, 3, 5, 10]:
        if depth > max_n:
            continue
        oh = [oracle_pick(by_id[u].text, nbest[u], depth) for u in test]
        depth_curve[depth] = corpus_wer(refs, oh)["wer"]
    oracle_wer = depth_curve[max(depth_curve)]

    # what fraction of the available head-room does GPT-2 capture?
    gpt2_path = rdir / "phaseB_summary_gpt2.json"
    gpt2_wer = None
    captured = None
    if gpt2_path.exists():
        gpt2_wer = json.loads(gpt2_path.read_text())["test_wer_adapted"]
        headroom = base_wer - oracle_wer
        captured = (base_wer - gpt2_wer) / headroom if headroom > 0 else None

    out = {
        "dataset": args.dataset,
        "n_test": len(test),
        "nbest_depth": max_n,
        "test_wer_baseline_top1": base_wer,
        "test_wer_oracle": oracle_wer,
        "oracle_by_depth": depth_curve,
        "test_wer_gpt2_adapted": gpt2_wer,
        "headroom_abs": (base_wer - oracle_wer),
        "headroom_rel_pct": 100 * (base_wer - oracle_wer) / base_wer if base_wer else None,
        "gpt2_fraction_of_headroom_captured": captured,
    }
    (rdir / "phaseB_oracle.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    # sanity: did we reproduce the rescore test split?
    if gpt2_path.exists():
        exp = json.loads(gpt2_path.read_text()).get("n_test")
        print(f"\n[split check] reproduced n_test={len(test)} vs gpt2 summary n_test={exp}",
              "OK" if exp == len(test) else "MISMATCH", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
