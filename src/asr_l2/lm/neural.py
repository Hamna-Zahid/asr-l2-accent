"""Neural (causal Transformer) LM scorer for strong-LM n-best rescoring.

Addresses the obvious critique of the n-gram null result -- "your LM was too
weak." A pretrained GPT-2-class model is a strong, general English LM; if even
it cannot improve the ranking of the decoder's n-best, the negative result is
robust (the better hypothesis simply is not in the list, or differs acoustically
rather than lexically). Scores are natural-log token log-likelihoods.

Zero-shot by default (general English). Pass a locally fine-tuned model path to
test a *domain-adapted* neural LM. CPU-friendly with distilgpt2.
"""
from __future__ import annotations

from ..scoring.normalize import tokenize


class NeuralLM:
    def __init__(self, model_name: str = "distilgpt2", device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._torch = torch
        # Auto-pick GPU when available (Colab) so large ladder LMs (gpt2-large,
        # 1B+) run at usable speed; fall back to CPU transparently. fp16 on GPU
        # keeps the bigger models inside T4 memory.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device).eval()
        self.device = device
        self.name = model_name

    def logprob(self, text) -> float:
        """Total natural-log likelihood of the (raw) text under the LM."""
        s = " ".join(text) if isinstance(text, list) else str(text)
        s = s.strip()
        if not s:
            return 0.0
        ids = self.tok(s, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] < 2:
            return 0.0
        with self._torch.no_grad():
            out = self.model(ids, labels=ids)
        n_pred = ids.shape[1] - 1            # loss is mean NLL over predicted tokens
        return -float(out.loss.item()) * n_pred

    def logprob_per_word(self, text) -> float:
        toks = text if isinstance(text, list) else tokenize(text)
        n = len(toks) + 1
        return self.logprob(text) / n if n else 0.0
