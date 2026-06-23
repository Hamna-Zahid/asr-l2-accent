#!/usr/bin/env python
"""Phase B.3/B.4 — audio-free n-best rescoring with a domain LM.

For each eval clip we take the frozen decoder's top-N hypotheses (with acoustic
scores) and re-rank them by:

    combined(hyp) = acoustic_score(hyp) + lambda * lm_logprob_per_word(hyp)

lambda is tuned on a DEV split and the WER delta is reported on a disjoint TEST
split, both overall and stratified by accent / proficiency. No audio is touched
beyond the frozen forward pass — this is the audio-free adaptation.

N-best decoding (the expensive, GPU-friendly part) is cached to
`results/<ds>/phaseB_nbest.jsonl`, so re-running / re-tuning lambda is instant.

Example (run the decode on GPU/Colab, then this is cheap):
  python scripts/rescore_nbest.py --dataset l2arctic --lm models/lm/l2arctic_4gram.pkl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                       # noqa: E402
from asr_l2.io.manifest import read_manifest                 # noqa: E402
from asr_l2.lm.model import load_lm                          # noqa: E402
from asr_l2.scoring.metrics import corpus_wer                # noqa: E402


def _load_eval_utts(cfg, dataset, manifest, max_utts, min_dur):
    utts = list(read_manifest(manifest))
    if min_dur and min_dur > 0:
        utts = [u for u in utts if u.duration_s >= min_dur]
    if max_utts is not None:
        utts = utts[:max_utts]
    return utts


def _get_nbest_cache(cache_path: Path) -> dict[str, list]:
    if not cache_path.exists():
        return {}
    out = {}
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["utt_id"]] = d["hyps"]   # list of [text, acoustic_score]
    return out


def _decode_nbest(utts, cfg, n, cache_path: Path, existing: dict):
    """Decode missing n-best and append to cache. Needs the ASR engine (slow on CPU)."""
    from asr_l2.asr.nbest import NBestEngine
    from asr_l2.io.audio import load_audio
    todo = [u for u in utts if u.utt_id not in existing]
    if not todo:
        return existing
    print(f">> decoding n-best for {len(todo)} clips (cached: {len(existing)})")
    eng = NBestEngine(model=cfg["asr"].get("_model_override") or cfg["asr"]["model"],
                      device=cfg["asr"]["device"],
                      compute_type=cfg["asr"]["compute_type"],
                      cpu_threads=cfg["asr"]["cpu_threads"])
    print(f">> engine on {eng.device}")
    fh = open(cache_path, "a", encoding="utf-8")
    for i, u in enumerate(todo):
        hyps = eng.nbest(load_audio(u.wav_path), n=n)
        rec = [[h.text, h.acoustic_score] for h in hyps]
        existing[u.utt_id] = rec
        fh.write(json.dumps({"utt_id": u.utt_id, "hyps": rec}) + "\n")
        fh.flush()
        if (i + 1) % 10 == 0:
            print(f"   {i+1}/{len(todo)}", flush=True)
    fh.close()
    return existing


def _rescore_pick(scored_hyps: list, lam: float) -> str:
    """Pick best hyp from precomputed (text, acoustic, lm_score) tuples."""
    best_text, best_score = scored_hyps[0][0], None
    for text, ac, lmscore in scored_hyps:
        s = ac + lam * lmscore
        if best_score is None or s > best_score:
            best_score, best_text = s, text
    return best_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lm", required=True, help="LM file (.pkl or .arpa/.bin)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--nbest", type=int, default=None)
    ap.add_argument("--dev-frac", type=float, default=0.34,
                    help="fraction of eval clips used to tune lambda")
    ap.add_argument("--max-utterances", type=int, default=None)
    ap.add_argument("--lambda-grid", default=None,
                    help="comma-separated lambda values (override config grid)")
    ap.add_argument("--model", default=None,
                    help="ASR model override for n-best decoding (e.g. base.en, tiny.en)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["asr"]["_model_override"] = args.model
    data_root = Path(cfg["paths"]["data_root"])
    rdir = Path(cfg["paths"]["results_root"]) / args.dataset
    rdir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest) if args.manifest else \
        data_root / "processed" / args.dataset / "manifest.jsonl"
    max_utts = args.max_utterances if args.max_utterances is not None \
        else cfg["eval"]["max_utterances"]
    min_dur = cfg["eval"].get("min_duration_s", 0.0)
    n = args.nbest or cfg["lm"]["nbest"]

    utts = _load_eval_utts(cfg, args.dataset, manifest, max_utts, min_dur)
    by_id = {u.utt_id: u for u in utts}

    # Model-specific n-best cache so a model-size variant never reuses another
    # model's decoded hypotheses.
    _tag = ("_" + args.model.replace(".", "")) if args.model else ""
    cache_path = rdir / f"phaseB_nbest{_tag}.jsonl"
    nbest = _get_nbest_cache(cache_path)
    nbest = _decode_nbest(utts, cfg, n, cache_path, nbest)

    lm = load_lm(args.lm)

    # dev/test split (seeded, disjoint) — lambda tuned on dev, reported on test.
    ids = [u.utt_id for u in utts if u.utt_id in nbest]
    rng = random.Random(cfg.get("seed", 1234))
    rng.shuffle(ids)
    n_dev = max(1, int(len(ids) * args.dev_frac))
    dev_ids, test_ids = set(ids[:n_dev]), set(ids[n_dev:])

    # Precompute the LM score of every hypothesis ONCE (essential for neural LMs,
    # which are expensive; the lambda sweep below is then pure arithmetic).
    print(f">> scoring {sum(len(nbest[u]) for u in ids)} hypotheses with the LM ...")
    scored = {}
    for k, uid in enumerate(ids):
        scored[uid] = [(t, ac, lm.logprob_per_word(t)) for t, ac in nbest[uid]]
        if (k + 1) % 50 == 0:
            print(f"   scored {k+1}/{len(ids)} clips", flush=True)

    grid = [float(x) for x in args.lambda_grid.split(",")] if args.lambda_grid \
        else cfg["lm"]["lambda_grid"]

    def wer_at(lam, id_set):
        refs, hyps = [], []
        for uid in id_set:
            refs.append(by_id[uid].text)
            hyps.append(_rescore_pick(scored[uid], lam))
        return corpus_wer(refs, hyps)["wer"]

    dev_curve = {lam: wer_at(lam, dev_ids) for lam in grid}
    best_lambda = min(dev_curve, key=lambda L: dev_curve[L])

    # Report on TEST: baseline (lambda=0) vs adapted (best_lambda).
    rows = []
    for uid in sorted(test_ids):
        u = by_id[uid]
        base = _rescore_pick(scored[uid], 0.0)     # = top acoustic = Whisper top-1
        adapt = _rescore_pick(scored[uid], best_lambda)
        from asr_l2.scoring.metrics import score_pair
        rows.append({
            "utt_id": uid, "accent": u.accent, "cefr": u.cefr,
            "ref": u.text, "baseline_hyp": base, "adapted_hyp": adapt,
            "changed": base != adapt,
            "wer_baseline": score_pair(u.text, base)["wer"],
            "wer_adapted": score_pair(u.text, adapt)["wer"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(rdir / "phaseB_rescore.csv", index=False)

    base_overall = corpus_wer(df["ref"].tolist(), df["baseline_hyp"].tolist())
    adapt_overall = corpus_wer(df["ref"].tolist(), df["adapted_hyp"].tolist())

    # Stratified by accent (honest sub-group check).
    strat = []
    for acc, g in df.groupby(df["accent"].fillna("unknown")):
        b = corpus_wer(g["ref"].tolist(), g["baseline_hyp"].tolist())["wer"]
        a = corpus_wer(g["ref"].tolist(), g["adapted_hyp"].tolist())["wer"]
        strat.append({"accent": acc, "n": len(g), "wer_baseline": b,
                      "wer_adapted": a, "delta": (a - b) if (a and b) else None})

    summary = {
        "dataset": args.dataset, "lm": args.lm, "nbest": n,
        "n_eval": len(ids), "n_dev": len(dev_ids), "n_test": len(test_ids),
        "lambda_grid": grid,
        "dev_wer_by_lambda": {str(k): v for k, v in dev_curve.items()},
        "best_lambda": best_lambda,
        "test_wer_baseline": base_overall["wer"],
        "test_wer_adapted": adapt_overall["wer"],
        "test_wer_delta": (adapt_overall["wer"] - base_overall["wer"])
        if (adapt_overall["wer"] and base_overall["wer"]) else None,
        "n_changed": int(df["changed"].sum()),
        "stratified_by_accent": strat,
    }
    (rdir / "phaseB_summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    print(">> DONE")
    print(json.dumps(summary, indent=2))
    if summary["test_wer_delta"] is not None and summary["test_wer_delta"] > 0:
        print("\nNOTE: adaptation did not help on TEST (delta > 0). Report this "
              "honestly; try a larger/cleaner domain corpus or more n-best.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
