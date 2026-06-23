"""Pure-Python interpolated n-gram LM (no compiler needed).

This is the fallback used on the Windows/8GB box where KenLM cannot be built.
It trains modified-count interpolated n-grams with linear (Jelinek-Mercer)
interpolation across orders and an add-k unigram base, giving proper normalized
log-probabilities (so perplexity is well-defined and comparable to KenLM).

For the paper's main results we prefer KenLM (built on Colab/Linux); this model
exists so the full pipeline runs end-to-end anywhere and so results are
reproducible without a C++ toolchain. See ``model.py`` for the unified loader.
"""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

from ..scoring.normalize import tokenize

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class PyNGramLM:
    def __init__(self, order: int = 4, add_k: float = 0.1,
                 interp_lambda: float = 0.7):
        self.order = order
        self.add_k = add_k
        self.lam = interp_lambda                 # weight on the higher-order estimate
        # counts[k] maps a k-gram tuple -> count (k from 1..order)
        self.counts: list[dict] = [defaultdict(int) for _ in range(order + 1)]
        self.vocab: set[str] = set()
        self.total_unigrams = 0

    # ---- training ---------------------------------------------------------
    def _ngrams(self, tokens: list[str]):
        toks = [BOS] * (self.order - 1) + tokens + [EOS]
        for k in range(1, self.order + 1):
            for i in range(len(toks) - k + 1):
                yield k, tuple(toks[i:i + k])

    def train(self, sentences) -> "PyNGramLM":
        for sent in sentences:
            toks = sent if isinstance(sent, list) else tokenize(sent)
            if not toks:
                continue
            self.vocab.update(toks)
            for k, gram in self._ngrams(toks):
                self.counts[k][gram] += 1
        self.vocab.update([EOS, UNK])
        self.total_unigrams = sum(self.counts[1].values())
        return self

    # ---- scoring ----------------------------------------------------------
    def _p_unigram(self, w: str) -> float:
        c = self.counts[1].get((w,), 0)
        V = len(self.vocab)
        return (c + self.add_k) / (self.total_unigrams + self.add_k * V)

    def _p_interp(self, gram: tuple) -> float:
        """Interpolated P(w | history) for a full-order gram tuple."""
        w = gram[-1]
        p = self._p_unigram(w)
        # build up from order 2..len(gram), interpolating higher orders in
        for k in range(2, len(gram) + 1):
            sub = gram[-k:]
            hist = sub[:-1]
            hist_c = self.counts[k - 1].get(hist, 0)
            if hist_c > 0:
                p_ml = self.counts[k].get(sub, 0) / hist_c
                p = self.lam * p_ml + (1 - self.lam) * p
            else:
                p = (1 - self.lam) * p + self.lam * p  # == p (no higher evidence)
        return p

    def logprob(self, text) -> float:
        """Total natural-log probability of the text (with EOS)."""
        toks = text if isinstance(text, list) else tokenize(text)
        toks = [BOS] * (self.order - 1) + toks + [EOS]
        lp = 0.0
        for i in range(self.order - 1, len(toks)):
            gram = tuple(toks[i - self.order + 1:i + 1])
            p = self._p_interp(gram)
            lp += math.log(p if p > 0 else 1e-12)
        return lp

    def logprob_per_word(self, text) -> float:
        toks = text if isinstance(text, list) else tokenize(text)
        n = len(toks) + 1   # + EOS
        return self.logprob(toks) / n if n else 0.0

    def perplexity(self, sentences) -> float:
        total_lp, total_n = 0.0, 0
        for s in sentences:
            toks = s if isinstance(s, list) else tokenize(s)
            total_lp += self.logprob(toks)
            total_n += len(toks) + 1
        return math.exp(-total_lp / total_n) if total_n else float("inf")

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "PyNGramLM":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def stats(self) -> dict:
        return {"order": self.order, "vocab": len(self.vocab),
                "unigrams": self.total_unigrams,
                "ngrams": {k: len(self.counts[k]) for k in range(1, self.order + 1)}}
