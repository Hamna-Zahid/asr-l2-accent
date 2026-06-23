#!/usr/bin/env python
"""Fill the spot-check sheet with a manual TEXT-LEVEL re-adjudication.

This re-applies the error taxonomy to each aligned error from its reference,
hypothesis and context (NOT an audio-perceptual judgment). accent_phoneme labels
rest on L2-ARCTIC's human phoneme annotations and are trusted; disagreements are
where a text rule mis-fires -- number-format strings tagged as deletion/accent
instead of normalization, and word-splitting insertions tagged as hallucination.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "l2arctic" / "spotcheck_sheet.csv"
OUT = REPO / "results" / "l2arctic" / "spotcheck_sheet_filled.csv"

# idx -> (human_category, note). Any idx not listed: human = auto, agree = Y.
CORRECTIONS = {
    1:  ("normalization", "contraction 'she is'->'she's'; same words, a format diff, not a deletion"),
    2:  ("model_lexical", "'a' is a fragment of 'Anyway' misheard as 'in a way'; a misrecognition, not invented filler"),
    9:  ("normalization", "'three hundred'->'300' is number formatting, not a deletion"),
    13: ("model_lexical", "'hanker' is a fragment of 'handkerchief' misheard as 'hanker shift'; misrecognition, not invented filler"),
    37: ("normalization", "'three hundred'->'300' is number formatting, not a deletion"),
    40: ("normalization", "'three hundred'->'300' is number formatting, not accent"),
}
# brief confirming notes on a few notable agreements
CONFIRM = {
    25: "genuine Whisper repetition/hallucination loop",
    39: "genuine Whisper repetition/hallucination loop",
    33: "'the' genuinely dropped (not a contraction/number), so deletion is correct",
}

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
n_agree = 0
for r in rows:
    i = int(r["idx"])
    auto = r["auto_category"]
    if i in CORRECTIONS:
        human, note = CORRECTIONS[i]
        r["human_category"] = human
        r["agree_Y_N"] = "Y" if human == auto else "N"
        r["notes"] = note
    else:
        r["human_category"] = auto
        r["agree_Y_N"] = "Y"
        r["notes"] = CONFIRM.get(i, "")
    n_agree += (r["agree_Y_N"] == "Y")

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

print(f"agreement: {n_agree}/{len(rows)} = {100*n_agree/len(rows):.0f}%")
print("disagreements:")
for r in rows:
    if r["agree_Y_N"] == "N":
        print(f"  #{r['idx']:>2} {r['auto_category']} -> {r['human_category']}: {r['notes']}")
from collections import Counter
print("disagreement categories (auto):", dict(Counter(r["auto_category"] for r in rows if r["agree_Y_N"]=="N")))
