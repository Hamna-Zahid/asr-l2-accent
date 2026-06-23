#!/usr/bin/env python
"""Fine-tune a small causal LM on the leakage-guarded in-domain corpus, turning
the zero-shot neural rescorer into a genuinely *domain-adapted* (still audio-free)
one. Phase B analysis.

The corpus (data/lm/l2arctic_corpus.txt) already excludes every sentence used in
evaluation; because all L2-ARCTIC speakers read the same prompts, this script
additionally ASSERTS zero normalized overlap with the eval manifest before
training, so the adaptation cannot memorize a test sentence. The LM instead
learns the read-prompt register from disjoint prompts -- any rescoring gain on
the held-out prompts is real domain adaptation.

Output: a HuggingFace checkpoint dir usable directly as a rescorer LM, e.g.
  python scripts/rescore_nbest.py --dataset l2arctic_24spk --lm neural:models/lm/ft_gpt2_l2arctic

Usage:
  python scripts/finetune_lm.py --base gpt2 --epochs 3
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from asr_l2.scoring.normalize import normalize_text          # noqa: E402
from asr_l2.io.manifest import read_manifest                 # noqa: E402


def assert_leakage_free(corpus_lines, eval_manifest: Path):
    corpus = {normalize_text(l) for l in corpus_lines if l.strip()}
    evalset = {normalize_text(u.text) for u in read_manifest(eval_manifest)}
    overlap = corpus & evalset
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} corpus sentences are eval "
                         f"references, e.g. {list(overlap)[:3]}. Aborting.")
    print(f"[leakage guard] 0 overlap with {eval_manifest.name} "
          f"({len(corpus)} corpus / {len(evalset)} eval sentences) OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2", help="HF base model to fine-tune")
    ap.add_argument("--corpus", default=str(REPO / "data/lm/l2arctic_corpus.txt"))
    ap.add_argument("--eval-manifest",
                    default=str(REPO / "data/processed/l2arctic_24spk/manifest.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lines = [l.strip() for l in open(args.corpus, encoding="utf-8") if l.strip()]
    assert_leakage_free(lines, Path(args.eval_manifest))

    out = Path(args.out) if args.out else REPO / "models/lm" / f"ft_{args.base.replace('/','_')}_l2arctic"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fine-tune] base={args.base} device={device} corpus={len(lines)} sents -> {out}")

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base).to(device).train()

    # pack the corpus into fixed-length blocks of token ids
    text = ("\n".join(lines))
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = (ids.numel() // args.block) * args.block
    blocks = ids[:n].view(-1, args.block)
    ds = TensorDataset(blocks)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        tot, k = 0.0, 0
        for (batch,) in dl:
            batch = batch.to(device)
            opt.zero_grad()
            loss = model(batch, labels=batch).loss
            loss.backward()
            opt.step()
            tot += float(loss.item()); k += 1
        print(f"  epoch {ep+1}/{args.epochs}  train_loss={tot/k:.4f}  ppl={math.exp(tot/k):.1f}")

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[fine-tune] saved -> {out}")
    print(f"  rescore with:  --lm neural:{out.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
