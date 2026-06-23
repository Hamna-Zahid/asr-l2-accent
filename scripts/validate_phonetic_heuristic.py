#!/usr/bin/env python
"""Validate the phonetic accent heuristic against L2-ARCTIC phoneme ground truth.

The Svarah accent labels rely on a phonetic-similarity heuristic (metaphone /
Jaro-Winkler) because Svarah has no phoneme annotations. L2-ARCTIC DOES have
ground truth (the phoneme join), so we can measure how well the heuristic agrees
with it on substitution errors there, and report that as evidence the heuristic
is trustworthy when transferred to Svarah.

Treats the phoneme-join 'coincides_mispronunciation' as ground truth (accent),
and the phonetic heuristic prediction as the test, on L2-ARCTIC substitutions.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.errors.categorize import phonetic_similar   # noqa: E402

rdir = Path(__file__).resolve().parents[1] / "results" / "l2arctic"
rows = list(csv.DictReader(open(rdir / "phaseA3_phoneme_join.csv", encoding="utf-8")))

tp = fp = tn = fn = 0
for r in rows:
    if r["op"] != "substitute":
        continue
    gt = r.get("coincides_mispronunciation", "").lower() == "true"   # ground-truth accent
    pred = phonetic_similar(r["ref_word"], r["hyp_word"])            # heuristic
    if pred and gt:
        tp += 1
    elif pred and not gt:
        fp += 1
    elif not pred and gt:
        fn += 1
    else:
        tn += 1

n = tp + fp + tn + fn
agree = (tp + tn) / n if n else 0
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
# Cohen's kappa
po = agree
pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else 0
kappa = (po - pe) / (1 - pe) if (1 - pe) else 0

out = {
    "n_substitutions": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    "agreement": round(agree, 3), "precision": round(prec, 3),
    "recall": round(rec, 3), "f1": round(f1, 3), "cohen_kappa": round(kappa, 3),
}
(rdir / "phonetic_heuristic_validation.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
