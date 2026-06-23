"""Unified domain-LM interface used by the rescorer.

Two backends, one API (``logprob`` / ``logprob_per_word`` in *natural* log):
  - ``KenLMModel``  : loads a KenLM .arpa/.bin (preferred; built on Colab/Linux).
  - ``PyNGramLM``   : pure-Python fallback (.pkl) for the no-compiler box.

``load_lm(path)`` dispatches on file extension so the rescorer is backend-blind.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..scoring.normalize import normalize_text, tokenize
from .ngram import PyNGramLM

_LN10 = math.log(10.0)


class KenLMModel:
    """Wrapper around a KenLM model (scores are log10 -> convert to ln)."""

    def __init__(self, path: str | Path):
        import kenlm  # only needed when this backend is used
        self.model = kenlm.Model(str(path))
        self.path = str(path)

    def logprob(self, text) -> float:
        sent = " ".join(text) if isinstance(text, list) else normalize_text(text)
        return self.model.score(sent, bos=True, eos=True) * _LN10

    def logprob_per_word(self, text) -> float:
        toks = text if isinstance(text, list) else tokenize(text)
        n = len(toks) + 1
        return self.logprob(text) / n if n else 0.0


def load_lm(spec: str | Path):
    """Load an LM backend from a file path or a neural spec.

    Accepts: ``.pkl`` (PyNGramLM), ``.arpa``/``.bin`` (KenLM), or
    ``neural:<model_name_or_path>`` (e.g. ``neural:distilgpt2``, ``neural:gpt2``,
    or a local fine-tuned checkpoint dir) for a strong Transformer LM.
    """
    s = str(spec)
    if s.startswith("neural:"):
        from .neural import NeuralLM
        return NeuralLM(s.split(":", 1)[1])
    path = Path(spec)
    suf = path.suffix.lower()
    if suf in {".arpa", ".bin", ".klm", ".mmap"}:
        return KenLMModel(path)
    if suf in {".pkl", ".pickle"}:
        return PyNGramLM.load(path)
    raise ValueError(f"Unknown LM spec: {spec} (expected .pkl/.arpa or neural:<model>)")
