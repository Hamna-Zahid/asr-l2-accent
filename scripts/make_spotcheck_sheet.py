#!/usr/bin/env python
"""Prepare a human listening spot-check sheet for L2-ARCTIC error labels, and
run an automated rule-faithfulness audit on the auto-assigned categories.

Two outputs:
  1. results/l2arctic/spotcheck_sheet.csv  -- a 50-error stratified sample with
     the wav path and seek time, ready for a human to listen and fill in
     `human_category`, `agree` (Y/N), and `notes`. The author then reports the
     resulting agreement % in Section 7 (audio-verified).
  2. console audit -- checks every label in the full 264-error set satisfies the
     definitional text rule for its category (a consistency check, NOT an
     audio-perceptual one).

Usage:  python scripts/make_spotcheck_sheet.py [--n 50] [--seed 20240623]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "l2arctic" / "phaseA_error_annotation_offline.annotated.csv"
MANIFEST = REPO / "data" / "processed" / "l2arctic" / "manifest.jsonl"
OUT = REPO / "results" / "l2arctic" / "spotcheck_sheet.csv"


def load_rows():
    with open(SRC, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wav_map():
    m = {}
    if MANIFEST.exists():
        for line in open(MANIFEST, encoding="utf-8"):
            d = json.loads(line)
            m[d["utt_id"]] = d.get("wav_path", "")
    return m


def audit(rows):
    """Rule-faithfulness audit: does each label satisfy its definitional rule?
    This is a text/rule consistency check, not an audio-perceptual judgment."""
    ok = defaultdict(int)
    bad = defaultdict(list)
    for r in rows:
        cat = r["why_category"]
        ref = (r["ref_word"] or "").lower()
        hyp = (r["hyp_word"] or "").lower()
        ctx = (r["ref_context"] or "").lower()
        passes = True
        if cat == "accent_phoneme":
            # accent labels are assigned only when the error coincides with a
            # phoneme annotation; we cannot re-derive that from text alone, so we
            # only check it is a real substitution/deletion on a content word.
            passes = r["op"] in ("substitute", "delete") and bool(ref)
        elif cat == "hallucination":
            # inserted/substituted text not licensed by the reference context
            passes = r["op"] in ("insert", "substitute") and hyp and hyp not in ctx
        elif cat == "normalization":
            # number/format variants: a digit on one side or a number word
            numwords = {"hundred", "thousand", "zero", "one", "two", "three",
                        "four", "five", "six", "seven", "eight", "nine", "ten"}
            passes = any(c.isdigit() for c in ref + hyp) or bool(
                ({ref, hyp} & numwords))
        elif cat == "model_lexical":
            passes = r["op"] in ("substitute", "delete", "insert")
        elif cat == "deletion_other":
            passes = r["op"] == "delete"
        elif cat == "reference_error":
            passes = True  # adjudicated by reference inspection; no text rule
        if passes:
            ok[cat] += 1
        else:
            bad[cat].append((r["utt_id"], r["op"], ref, hyp))
    print("=== rule-faithfulness audit (text-level, NOT audio) ===")
    total = len(rows)
    npass = sum(ok.values())
    for cat in sorted(set(r["why_category"] for r in rows)):
        n = sum(1 for r in rows if r["why_category"] == cat)
        print(f"  {cat:16s} {ok[cat]:3d}/{n:<3d} satisfy rule")
        for ex in bad[cat][:3]:
            print(f"        flag: {ex}")
    print(f"  OVERALL {npass}/{total} = {100*npass/total:.1f}% rule-consistent")


def make_sheet(rows, n, seed):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["why_category"]].append(r)
    cats = list(by_cat)
    # proportional allocation with a floor of 4 per category so rare categories
    # are represented, then top up the largest categories to reach n.
    alloc = {c: min(len(by_cat[c]), 4) for c in cats}
    while sum(alloc.values()) < n:
        # add to the category with the most remaining headroom
        c = max(cats, key=lambda c: len(by_cat[c]) - alloc[c])
        if len(by_cat[c]) - alloc[c] <= 0:
            break
        alloc[c] += 1
    sample = []
    for c in cats:
        sample += rng.sample(by_cat[c], alloc[c])
    rng.shuffle(sample)

    wm = wav_map()
    cols = ["idx", "utt_id", "wav_path", "seek_s", "op", "ref_word", "hyp_word",
            "ref_context", "hyp_sentence", "auto_category",
            "human_category", "agree_Y_N", "notes"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(sample, 1):
            w.writerow({
                "idx": i,
                "utt_id": r["utt_id"],
                "wav_path": wm.get(r["utt_id"], ""),
                "seek_s": r["approx_time_s"],
                "op": r["op"],
                "ref_word": r["ref_word"],
                "hyp_word": r["hyp_word"],
                "ref_context": r["ref_context"],
                "hyp_sentence": r["hyp_sentence"],
                "auto_category": r["why_category"],
                "human_category": "",
                "agree_Y_N": "",
                "notes": "",
            })
    print(f"\n=== listening sheet: {OUT} ===")
    print(f"  {len(sample)} errors, allocation: {dict(Counter(r['why_category'] for r in sample))}")
    print("  Fill human_category + agree_Y_N while listening; agreement % = mean(agree==Y).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20240623)
    a = ap.parse_args()
    rows = load_rows()
    audit(rows)
    make_sheet(rows, a.n, a.seed)


if __name__ == "__main__":
    main()
