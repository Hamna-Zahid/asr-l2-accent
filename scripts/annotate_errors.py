#!/usr/bin/env python
"""Critical semi-automated error categorization.

Goes beyond surface rules by using every signal available in the (now full-
sentence) error log to reason about *why* each error occurred:

  reference_error : the reference transcript is a TRUNCATION/variant of what the
                    ASR said (ref text is a substring of the hyp text, or a known
                    colloquial form). The model was effectively right.
  normalization   : orthographic-only difference -- British/American spelling,
                    digit<->word number forms, contraction expansion.
  accent_phoneme  : a sound-alike confusion. For L2-ARCTIC, confirmed by the
                    manual phoneme join; for both corpora, detected when ref and
                    hyp words are phonetically near (metaphone match / high
                    Jaro-Winkler) -- i.e. the model mis-heard a sound.
  disfluency      : an inserted/again word that repeats its neighbour.
  hallucination   : an inserted word with no reference and no truncation excuse.
  model_lexical   : a real-word substitution that is NOT a sound-alike (the model
                    chose a wrong, acoustically-distant word).
  deletion_other  : a dropped word not otherwise explained.

For L2-ARCTIC the accent label is anchored on human phoneme annotations; for
Svarah (no phoneme labels) it is a phonetic-similarity heuristic -- documented
as such in the paper. This is an analysis aid, not a substitute for listening.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import jellyfish

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.scoring.normalize import normalize_text, tokenize  # noqa: E402

BRIT_AMER = {("centre", "center"), ("centres", "centers"), ("colour", "color"),
    ("favour", "favor"), ("honour", "honor"), ("labour", "labor"),
    ("theatre", "theater"), ("metre", "meter"), ("litre", "liter"),
    ("organise", "organize"), ("realise", "realize"), ("recognise", "recognize"),
    ("travelled", "traveled"), ("grey", "gray"), ("defence", "defense"),
    ("licence", "license"), ("practise", "practice"), ("programme", "program")}
COLLOQUIAL = {("em", "them"), ("til", "until"), ("cause", "because"),
    ("gonna", "going"), ("wanna", "want"), ("kinda", "kind"), ("ok", "okay")}
NUM_WORDS = {"zero","one","two","three","four","five","six","seven","eight","nine",
    "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million","first","second","third",
    "fourth","fifth","sixth","seventh","eighth","ninth","tenth","twentieth"}
_DIGIT = re.compile(r"\d")


def _num(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    an = bool(_DIGIT.search(a)) or a in NUM_WORDS
    bn = bool(_DIGIT.search(b)) or b in NUM_WORDS
    return an and bn and a != b


def _spelling(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if (a, b) in BRIT_AMER or (b, a) in BRIT_AMER:
        return True
    for x, y in ((a, b), (b, a)):
        if len(x) > 3 and x[:-2] == y[:-2] and x[-2:] in {"re", "ur"}:
            return True
    return False


def _colloquial(a, b):
    a, b = (a or "").lower().strip("'"), (b or "").lower().strip("'")
    return (a, b) in COLLOQUIAL or (b, a) in COLLOQUIAL


def _phonetic_similar(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if not a or not b or a == b:
        return False
    try:
        if jellyfish.metaphone(a) == jellyfish.metaphone(b) and jellyfish.metaphone(a):
            return True
    except Exception:
        pass
    return jellyfish.jaro_winkler_similarity(a, b) >= 0.86


def classify(op, ref, hyp, ctx, coincides, ref_trunc):
    # 1. reference is a truncation/variant of what the ASR said -> model was right
    if op == "insert" and ref_trunc:
        return "reference_error"
    if op == "substitute" and (_colloquial(ref, hyp)):
        return "reference_error"
    # 2. orthographic-only differences
    if op == "substitute" and (_num(ref, hyp) or _spelling(ref, hyp)):
        return "normalization"
    # 3. accent: asserted ONLY from L2-ARCTIC phoneme ground truth. The text-only
    #    phonetic proxy was validated against this ground truth (Cohen's kappa ~ 0)
    #    and is therefore NOT used to label accent.
    if coincides:
        return "accent_phoneme"
    # 4. insertions: repetition -> disfluency, else hallucination
    if op == "insert":
        toks = re.findall(r"[a-z0-9']+", (ctx or "").lower())
        if hyp and toks.count((hyp or "").lower()) >= 1:
            return "disfluency"
        return "hallucination"
    # 5. real-word, acoustically-distant substitution
    if op == "substitute":
        return "model_lexical"
    if op == "delete":
        return "deletion_other"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    rdir = Path(args.results_dir) if args.results_dir else repo / "results" / args.dataset
    err_csv = rdir / "phaseA_error_annotation_offline.csv"
    join_csv = rdir / "phaseA3_phoneme_join.csv"

    coin = {}
    if join_csv.exists():
        for r in csv.DictReader(open(join_csv, encoding="utf-8")):
            coin[(r["utt_id"], r["op"], r["ref_word"], r["hyp_word"],
                  r["approx_time_s"])] = \
                r.get("coincides_mispronunciation", "").lower() == "true"

    rows = list(csv.DictReader(open(err_csv, encoding="utf-8")))
    # utterance-level signal: is the reference a substring (truncation) of the hyp?
    trunc = {}
    for r in rows:
        rn, hn = normalize_text(r.get("ref_sentence", "")), normalize_text(r.get("hyp_sentence", ""))
        trunc[r["utt_id"]] = bool(rn) and rn in hn and len(hn.split()) > len(rn.split())

    counts = {}
    for r in rows:
        key = (r["utt_id"], r["op"], r["ref_word"], r["hyp_word"], r["approx_time_s"])
        cat = classify(r["op"], r["ref_word"], r["hyp_word"], r["ref_context"],
                       coin.get(key, False), trunc.get(r["utt_id"], False))
        r["why_category"] = cat
        r["annotator_notes"] = "auto-critical"
        counts[cat] = counts.get(cat, 0) + 1

    out_csv = rdir / "phaseA_error_annotation_offline.annotated.csv"
    try:
        fh = open(out_csv, "w", newline="", encoding="utf-8")
    except PermissionError:
        out_csv = rdir / "phaseA_error_annotation_offline.annotated.NEW.csv"
        print(f"   (target locked/open; wrote {out_csv.name} instead)", file=sys.stderr)
        fh = open(out_csv, "w", newline="", encoding="utf-8")
    with fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    total = sum(counts.values()) or 1
    summary = {"dataset": args.dataset, "n_errors": total, "counts": counts,
               "proportions": {k: round(v / total, 4) for k, v in
                               sorted(counts.items(), key=lambda x: -x[1])}}
    (rdir / "phaseA_error_categories.json").write_text(json.dumps(summary, indent=2),
                                                       encoding="utf-8")
    print(">> DONE", args.dataset)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
