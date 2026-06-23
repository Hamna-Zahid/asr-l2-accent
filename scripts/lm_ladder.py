#!/usr/bin/env python
"""LM-strength ladder for audio-free n-best rescoring (Phase B analysis).

Runs a ladder of LMs of increasing strength over the SAME cached n-best and the
SAME seeded dev/test split used by rescore_nbest.py / oracle_bound.py, and asks:
does the rescoring gain scale with LM quality, and how close does it get to the
oracle ceiling (the recoverable head-room in the n-best)?

For each LM it reports:
  * ppl_word  -- word-level perplexity on the held-out TEST references, a metric
    comparable across n-gram and neural backends (both expose logprob_per_word).
    Lower = stronger. None of the LMs see the test references in training
    (n-gram corpus is leakage-guarded; neural LMs are zero-shot or fine-tuned on
    the same guarded corpus), so this is a fair, held-out strength axis.
  * best_lambda, test WER baseline/adapted, relative delta, and the fraction of
    the oracle head-room captured.

All text-only on the cached n-best -- no audio. Heavy only in neural LM forward
passes, so run the big rungs (gpt2-large, 1B+) on a GPU/Colab.

Usage (edit --lms or pass your own specs:labels):
  python scripts/lm_ladder.py --dataset l2arctic_24spk \
      --lms "models/lm/l2arctic_4gram.pkl=4-gram" \
            "neural:distilgpt2=distilGPT-2 (82M)" \
            "neural:gpt2=GPT-2 (124M)" \
            "neural:gpt2-large=GPT-2-large (774M)" \
            "neural:Qwen/Qwen2.5-1.5B=Qwen2.5 (1.5B)"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.lm.model import load_lm                          # noqa: E402
from asr_l2.scoring.metrics import corpus_wer                # noqa: E402
from oracle_bound import reproduce_split                     # noqa: E402

DEFAULT_LMS = [
    "models/lm/l2arctic_4gram.pkl=4-gram (weak)",
    "neural:distilgpt2=distilGPT-2 (82M)",
    "neural:gpt2=GPT-2 (124M)",
    "neural:gpt2-large=GPT-2-large (774M)",
    "neural:Qwen/Qwen2.5-1.5B=Qwen2.5 (1.5B)",
]


def word_ppl(lm, sentences) -> float:
    """exp(-mean per-word log-prob) over held-out sentences (word-level, so it is
    comparable between the n-gram and neural backends)."""
    vals = [lm.logprob_per_word(s) for s in sentences if s.strip()]
    return math.exp(-sum(vals) / len(vals)) if vals else float("nan")


def best_pick(scored, lam):
    bt, bs = scored[0][0], None
    for t, ac, ls in scored:
        s = ac + lam * ls
        if bs is None or s > bs:
            bs, bt = s, t
    return bt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="l2arctic_24spk")
    ap.add_argument("--dev-frac", type=float, default=0.34)
    ap.add_argument("--lms", nargs="*", default=DEFAULT_LMS,
                    help="spec=label items; spec is a load_lm spec")
    ap.add_argument("--grid", default="0.0,0.05,0.1,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = Path(cfg["paths"]["results_root"]) / args.dataset
    manifest = Path(cfg["paths"]["data_root"]) / "processed" / args.dataset / "manifest.jsonl"
    grid = [float(x) for x in args.grid.split(",")]

    nbest = {}
    with open(rdir / "phaseB_nbest.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            nbest[d["utt_id"]] = d["hyps"]

    by_id, dev_ids, test_ids = reproduce_split(
        cfg, manifest, nbest, args.dev_frac,
        cfg["eval"].get("min_duration_s", 0.0), max_utts=10 ** 9)
    ids = sorted(dev_ids | test_ids)
    test = sorted(test_ids)
    test_refs = [by_id[u].text for u in test]

    # baseline (top-1 acoustic) and oracle ceiling
    base_wer = corpus_wer(test_refs, [max(nbest[u], key=lambda h: h[1])[0] for u in test])["wer"]
    oracle = json.loads((rdir / "phaseB_oracle.json").read_text())["test_wer_oracle"] \
        if (rdir / "phaseB_oracle.json").exists() else None
    headroom = (base_wer - oracle) if oracle else None

    rungs = []
    for item in args.lms:
        spec, _, label = item.partition("=")
        label = label or spec
        print(f"\n>> {label}  ({spec})", flush=True)
        try:
            lm = load_lm(spec)
        except Exception as e:                       # noqa: BLE001
            print(f"   SKIP (load failed: {e})", file=sys.stderr)
            continue
        ppl = word_ppl(lm, test_refs)
        scored = {u: [(t, ac, lm.logprob_per_word(t)) for t, ac in nbest[u]] for u in ids}

        def wer_at(lam, idset):
            return corpus_wer([by_id[u].text for u in idset],
                              [best_pick(scored[u], lam) for u in idset])["wer"]

        dev_curve = {lam: wer_at(lam, sorted(dev_ids)) for lam in grid}
        best_lambda = min(dev_curve, key=lambda L: dev_curve[L])
        adapt = wer_at(best_lambda, test)
        delta_rel = 100 * (adapt - base_wer) / base_wer if base_wer else None
        cap = ((base_wer - adapt) / headroom) if headroom else None
        rung = {"label": label, "spec": spec, "ppl_word": ppl,
                "best_lambda": best_lambda, "test_wer_baseline": base_wer,
                "test_wer_adapted": adapt, "delta_rel_pct": delta_rel,
                "frac_headroom_captured": cap,
                "dev_wer_by_lambda": {str(k): v for k, v in dev_curve.items()}}
        rungs.append(rung)
        print(f"   ppl_word={ppl:.1f}  lambda*={best_lambda}  "
              f"WER {base_wer:.4f}->{adapt:.4f}  ({delta_rel:+.1f}%)  "
              f"headroom captured {cap*100:.0f}%" if cap is not None else "")

    out = {"dataset": args.dataset, "n_test": len(test),
           "test_wer_baseline": base_wer, "test_wer_oracle": oracle,
           "headroom_abs": headroom, "ladder": rungs}
    (rdir / "phaseB_lm_ladder.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote", rdir / "phaseB_lm_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
