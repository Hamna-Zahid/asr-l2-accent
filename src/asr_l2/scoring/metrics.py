"""WER / CER and word-level alignment via jiwer.

The alignment output here feeds the error-taxonomy CSV in Phase A, so we expose
the per-chunk substitution / insertion / deletion operations, not just scalars.
"""
from __future__ import annotations

from dataclasses import dataclass

import jiwer

from .normalize import normalize_text


@dataclass
class ErrorOp:
    """One alignment operation between reference and hypothesis word sequences."""

    op: str               # "substitute" | "insert" | "delete"
    ref_word: str | None  # None for insertions
    hyp_word: str | None  # None for deletions
    ref_idx: int | None   # word index in the reference (None for insertions)
    hyp_idx: int | None   # word index in the hypothesis (None for deletions)


def score_pair(reference: str, hypothesis: str) -> dict:
    """Return WER, CER, edit counts, and the list of ErrorOps for one pair.

    Both strings are normalized first. Empty references are reported with
    ``wer=None`` (cannot be defined) so they can be filtered, not silently
    counted as perfect or as total error.
    """
    ref_n = normalize_text(reference)
    hyp_n = normalize_text(hypothesis)

    if not ref_n:
        return {
            "wer": None, "cer": None,
            "n_ref_words": 0, "n_hyp_words": len(hyp_n.split()),
            "substitutions": 0, "insertions": len(hyp_n.split()), "deletions": 0,
            "errors": [],
        }

    out = jiwer.process_words(ref_n, hyp_n)
    cer = jiwer.cer(ref_n, hyp_n)

    ref_words = ref_n.split()
    hyp_words = hyp_n.split()
    errors: list[ErrorOp] = []
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            continue
        if chunk.type == "substitute":
            for r, h in zip(range(chunk.ref_start_idx, chunk.ref_end_idx),
                            range(chunk.hyp_start_idx, chunk.hyp_end_idx)):
                errors.append(ErrorOp("substitute", ref_words[r], hyp_words[h], r, h))
        elif chunk.type == "delete":
            for r in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors.append(ErrorOp("delete", ref_words[r], None, r, None))
        elif chunk.type == "insert":
            for h in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                errors.append(ErrorOp("insert", None, hyp_words[h], None, h))

    return {
        "wer": out.wer,
        "cer": cer,
        "n_ref_words": len(ref_words),
        "n_hyp_words": len(hyp_words),
        "substitutions": out.substitutions,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "errors": errors,
    }


def corpus_wer(references: list[str], hypotheses: list[str]) -> dict:
    """Aggregate (micro-averaged) WER/CER over a list of pairs."""
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]
    keep = [(r, h) for r, h in zip(refs, hyps) if r]
    if not keep:
        return {"wer": None, "cer": None, "n_pairs": 0}
    refs_k, hyps_k = zip(*keep)
    return {
        "wer": jiwer.wer(list(refs_k), list(hyps_k)),
        "cer": jiwer.cer(list(refs_k), list(hyps_k)),
        "n_pairs": len(keep),
    }
