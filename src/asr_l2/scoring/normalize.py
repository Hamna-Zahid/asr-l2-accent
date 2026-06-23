"""Text normalization for fair WER/CER scoring.

Deliberately conservative: lowercase, strip punctuation, collapse whitespace,
and expand a small set of common contractions. We do NOT do aggressive number
or spelling normalization — for L2 accent research, over-normalizing can hide
real errors we want to measure. Document any change here in the paper's methods.
"""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation (keep intra-word apostrophes), collapse WS."""
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    # Drop apostrophes only when not between letters (keep "don't", drop "'cause").
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization of already-normalized text."""
    norm = normalize_text(text)
    return norm.split() if norm else []
