"""Shared error-category rules (used by annotation and Phase B category analysis).

Categories and the signal each rests on are documented in scripts/annotate_errors.py.
Accent detection uses L2-ARCTIC phoneme ground truth where available, and a
phonetic-similarity heuristic (metaphone / Jaro-Winkler) otherwise.
"""
from __future__ import annotations

import re

try:
    import jellyfish
    _HAVE_JELLY = True
except Exception:  # pragma: no cover
    _HAVE_JELLY = False

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

CATEGORIES = ["accent_phoneme", "model_lexical", "hallucination",
              "normalization", "disfluency", "reference_error",
              "deletion_other", "other"]


def is_number_form(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    an = bool(_DIGIT.search(a)) or a in NUM_WORDS
    bn = bool(_DIGIT.search(b)) or b in NUM_WORDS
    return an and bn and a != b


def is_spelling(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if (a, b) in BRIT_AMER or (b, a) in BRIT_AMER:
        return True
    for x, y in ((a, b), (b, a)):
        if len(x) > 3 and x[:-2] == y[:-2] and x[-2:] in {"re", "ur"}:
            return True
    return False


def is_colloquial(a, b):
    a, b = (a or "").lower().strip("'"), (b or "").lower().strip("'")
    return (a, b) in COLLOQUIAL or (b, a) in COLLOQUIAL


def phonetic_similar(a, b):
    a, b = (a or "").lower(), (b or "").lower()
    if not a or not b or a == b or not _HAVE_JELLY:
        return False
    try:
        ma, mb = jellyfish.metaphone(a), jellyfish.metaphone(b)
        if ma and ma == mb:
            return True
        return jellyfish.jaro_winkler_similarity(a, b) >= 0.86
    except Exception:
        return False


def classify(op, ref, hyp, ctx, coincides=False, ref_trunc=False):
    if op == "insert" and ref_trunc:
        return "reference_error"
    if op == "substitute" and is_colloquial(ref, hyp):
        return "reference_error"
    if op == "substitute" and (is_number_form(ref, hyp) or is_spelling(ref, hyp)):
        return "normalization"
    # Accent is asserted ONLY when L2-ARCTIC phoneme ground truth confirms it.
    # A text-only phonetic-similarity proxy was validated against this ground
    # truth and found uncorrelated (Cohen's kappa ~ 0), so it is NOT used to
    # label accent; phonetically-similar substitutions fall through to
    # model_lexical (we do not claim accent without phoneme evidence).
    if coincides:
        return "accent_phoneme"
    if op == "insert":
        toks = re.findall(r"[a-z0-9']+", (ctx or "").lower())
        if hyp and toks.count((hyp or "").lower()) >= 1:
            return "disfluency"
        return "hallucination"
    if op == "substitute":
        return "model_lexical"
    if op == "delete":
        return "deletion_other"
    return "other"
