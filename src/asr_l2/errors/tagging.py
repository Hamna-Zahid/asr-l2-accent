"""Turn ref/hyp alignments into reviewable, human-annotatable error rows.

This is the Phase A.2 artifact generator. Each error becomes one row with:
  - op type (substitution / deletion / insertion),
  - the reference and hypothesis words,
  - local context (a few words either side, from the reference),
  - an approximate audio timestamp so the annotator can listen,
  - EMPTY columns the human fills in (why_category, notes).

We do NOT guess *why* an error happened — that is the manual annotation step.
We only provide the substrate (the auto-detected what/where).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..asr.engine import Word
from ..scoring.metrics import score_pair
from ..scoring.normalize import tokenize

# The taxonomy the HUMAN assigns later (kept here as documentation + a controlled
# vocabulary the annotation CSV references). Auto-tagging never fills this in.
WHY_CATEGORIES = [
    "accent_phoneme",   # accent-driven phoneme confusion
    "disfluency",       # filler / repetition / self-correction in the speech
    "vad_cutoff",       # word lost at a VAD / chunk boundary
    "hallucination",    # ASR invented text not in the audio
    "reference_error",  # the ground-truth transcript itself is wrong
    "normalization",    # spurious error caused by text normalization mismatch
    "other",
]


@dataclass
class ErrorRow:
    utt_id: str
    dataset: str
    condition: str
    op: str
    ref_word: str | None
    hyp_word: str | None
    ref_context: str          # "... w-2 w-1 [REFWORD] w+1 w+2 ..."
    ref_sentence: str         # full reference transcript (for the annotator)
    hyp_sentence: str         # full ASR hypothesis (so you can see what it said)
    approx_time_s: float | None
    speaker: str | None = None
    accent: str | None = None
    # Human-filled columns (left blank by the auto pass):
    why_category: str = ""
    annotator_notes: str = ""
    extra: dict = field(default_factory=dict)


def _context(ref_tokens: list[str], idx: int | None, window: int = 3) -> str:
    """Build a readable reference-context string around a reference index."""
    if idx is None or not ref_tokens:
        return ""
    lo = max(0, idx - window)
    hi = min(len(ref_tokens), idx + window + 1)
    parts = []
    for j in range(lo, hi):
        parts.append(f"[{ref_tokens[j]}]" if j == idx else ref_tokens[j])
    return " ".join(parts)


def _hyp_time(hyp_words: list[Word], hyp_idx: int | None) -> float | None:
    if hyp_idx is not None and 0 <= hyp_idx < len(hyp_words):
        return round(hyp_words[hyp_idx].start, 3)
    return None


def tag_utterance(utt_id: str, dataset: str, condition: str,
                  reference: str, hypothesis: str, hyp_words: list[Word],
                  speaker: str | None = None,
                  accent: str | None = None) -> list[ErrorRow]:
    """Produce one ErrorRow per substitution/deletion/insertion.

    ``hyp_words`` are the ASR word-timings (aligned positionally to the
    normalized hypothesis tokens) used to place an approximate timestamp.
    Deletions (no hyp word) inherit the timestamp of the nearest prior hyp word.
    """
    sc = score_pair(reference, hypothesis)
    ref_tokens = tokenize(reference)
    rows: list[ErrorRow] = []
    last_time = 0.0
    for e in sc["errors"]:
        t = _hyp_time(hyp_words, e.hyp_idx)
        if t is None:
            t = last_time              # deletion: approximate from prior emission
        else:
            last_time = t
        rows.append(ErrorRow(
            utt_id=utt_id, dataset=dataset, condition=condition, op=e.op,
            ref_word=e.ref_word, hyp_word=e.hyp_word,
            ref_context=_context(ref_tokens, e.ref_idx),
            ref_sentence=reference.strip(), hyp_sentence=hypothesis.strip(),
            approx_time_s=t, speaker=speaker, accent=accent,
        ))
    return rows
