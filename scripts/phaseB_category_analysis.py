#!/usr/bin/env python
"""Phase B.5b — which ERROR CATEGORIES did audio-free rescoring fix?

For every test clip, categorize the errors in the baseline (Whisper top-1)
hypothesis and in the adapted (LM-rescored) hypothesis, then report the net
change per category. This is the category-level before/after that the error
taxonomy (Phase A) was built to enable, and it pinpoints *what kind* of error
the text-only LM repairs.

Input : results/<ds>/phaseB_rescore.csv  (ref, baseline_hyp, adapted_hyp)
Output: results/<ds>/phaseB_category_delta.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.scoring.metrics import score_pair          # noqa: E402
from asr_l2.scoring.normalize import normalize_text     # noqa: E402
from asr_l2.errors.categorize import classify           # noqa: E402


def _categorized_errors(ref: str, hyp: str) -> Counter:
    """Category counts for the errors in one ref/hyp pair."""
    sc = score_pair(ref, hyp)
    rn, hn = normalize_text(ref), normalize_text(hyp)
    ref_trunc = bool(rn) and rn in hn and len(hn.split()) > len(rn.split())
    c = Counter()
    for e in sc["errors"]:
        c[classify(e.op, e.ref_word, e.hyp_word, "", False, ref_trunc)] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    rdir = repo / "results" / args.dataset
    rescore = rdir / "phaseB_rescore.csv"
    if not rescore.exists():
        print(f"ERROR: {rescore} not found.", file=sys.stderr)
        return 2

    base_tot, adapt_tot = Counter(), Counter()
    n_changed = 0
    for r in csv.DictReader(open(rescore, encoding="utf-8")):
        base_tot += _categorized_errors(r["ref"], r["baseline_hyp"])
        adapt_tot += _categorized_errors(r["ref"], r["adapted_hyp"])
        if r.get("changed", "").lower() == "true":
            n_changed += 1

    cats = sorted(set(base_tot) | set(adapt_tot))
    delta = {c: adapt_tot[c] - base_tot[c] for c in cats}   # negative = fixed
    summary = {
        "dataset": args.dataset, "n_changed_utts": n_changed,
        "baseline_errors_by_category": dict(base_tot),
        "adapted_errors_by_category": dict(adapt_tot),
        "net_change_by_category": delta,
        "fixed_by_category": {c: -d for c, d in delta.items() if d < 0},
        "regressed_by_category": {c: d for c, d in delta.items() if d > 0},
    }
    (rdir / "phaseB_category_delta.json").write_text(json.dumps(summary, indent=2),
                                                     encoding="utf-8")
    print(">> DONE", args.dataset)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
