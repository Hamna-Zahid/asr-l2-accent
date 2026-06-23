#!/usr/bin/env python
"""Phase A.3 — cross-reference ASR errors against L2-Arctic phoneme annotations.

Question answered: when the ASR makes a substitution/deletion error on an
L2-Arctic utterance, does it coincide with a phoneme the human annotators
flagged as genuinely MISPRONOUNCED (accent-driven), or does it fall on speech
the annotators judged correctly produced (i.e. an ASR-model error, not an
accent effect)?

Inputs:
  data/processed/l2arctic/phone_annotations.jsonl   (from download_l2arctic.py)
  results/l2arctic/phaseA_error_annotation_offline.csv  (from build_error_csv.py)

Output:
  results/l2arctic/phaseA3_phoneme_join.csv     (per-error, with the flag)
  results/l2arctic/phaseA3_summary.json         (proportions)

Method: temporal overlap. Each ASR error has an approximate time; each annotated
phone has a [xmin,xmax] span. An ASR error "coincides with a mispronunciation"
if its time falls within (tol-padded) the span of a phone tagged s/d/a. Because
ASR word timestamps are approximate (and deletions inherit a neighbour's time),
this join is APPROXIMATE — documented as such; treat it as a strong hint, not a
hard label, and refine in the manual annotation pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tol", type=float, default=0.15,
                    help="time tolerance (s) padding each phone span (default 0.15)")
    ap.add_argument("--condition", default="offline")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    rdir = Path(cfg["paths"]["results_root"]) / "l2arctic"

    ann_path = data_root / "processed" / "l2arctic" / "phone_annotations.jsonl"
    err_path = rdir / f"phaseA_error_annotation_{args.condition}.csv"
    for p in (ann_path, err_path):
        if not p.exists():
            print(f"ERROR: required input not found: {p}\n"
                  f"       Run download_l2arctic.py, run_streaming_harness.py "
                  f"(--dataset l2arctic), and build_error_csv.py first.",
                  file=sys.stderr)
            return 2

    # Load annotations: utt_id -> list of mispronounced phone spans.
    mispron: dict[str, list[dict]] = {}
    with open(ann_path, "r", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            spans = [p for p in d["phones"] if p.get("err_type")]
            mispron[d["utt_id"]] = spans

    err = pd.read_csv(err_path)
    # Only utts that actually have manual annotations can be cross-referenced.
    err = err[err["utt_id"].isin(mispron.keys())].copy()
    if err.empty:
        print("WARNING: no ASR errors on annotated utterances to join.",
              file=sys.stderr)

    coincides, nearest, dist, classification = [], [], [], []
    for _, r in err.iterrows():
        t = r.get("approx_time_s")
        spans = mispron.get(r["utt_id"], [])
        best = None
        best_d = None
        for s in spans:
            if (s["xmin"] - args.tol) <= t <= (s["xmax"] + args.tol):
                d0 = 0.0
            else:
                d0 = min(abs(t - s["xmin"]), abs(t - s["xmax"]))
            if best_d is None or d0 < best_d:
                best_d, best = d0, s
        hit = best is not None and best_d == 0.0
        coincides.append(hit)
        nearest.append(f'{best["canonical"]}>{best["perceived"]}({best["err_type"]})'
                       if best else "")
        dist.append(round(best_d, 3) if best_d is not None else None)
        classification.append(
            "coincides_mispronunciation" if hit else "asr_error_on_correct_speech")

    err["coincides_mispronunciation"] = coincides
    err["nearest_mispron"] = nearest
    err["nearest_mispron_dist_s"] = dist
    err["phoneme_classification"] = classification

    out_csv = rdir / "phaseA3_phoneme_join.csv"
    err.to_csv(out_csv, index=False)

    n = len(err)
    n_coin = int(sum(coincides))
    by_op = (err.groupby("op")["coincides_mispronunciation"]
                .agg(["count", "sum"]).reset_index()
                .rename(columns={"count": "n_errors", "sum": "n_coincide"})
                .to_dict(orient="records"))
    summary = {
        "condition": args.condition, "tol_s": args.tol,
        "n_annotated_utts": len(mispron),
        "n_asr_errors_on_annotated_utts": n,
        "n_coincide_mispronunciation": n_coin,
        "frac_coincide": round(n_coin / n, 4) if n else None,
        "interpretation": ("frac_coincide ~ share of ASR errors that fall on a "
                           "genuinely mispronounced phone (accent-driven); the "
                           "remainder are ASR errors on correctly-produced speech."),
        "by_op": by_op,
    }
    (rdir / "phaseA3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(">> DONE")
    print(f"   join CSV: {out_csv}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
