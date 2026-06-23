#!/usr/bin/env python
"""Phase B.2 — train the domain n-gram LM from a text corpus.

Primary backend is the pure-Python interpolated n-gram (PyNGramLM): no compiler,
runs identically on Windows/Colab, fully reproducible. If you have KenLM and a
corpus large enough to warrant it, build that separately with `lmplz` and point
the rescorer at the .arpa/.bin (the rescorer auto-detects the backend).

Reports model stats and held-out perplexity (90/10 split by default) so you can
sanity-check the LM before using it for rescoring.

Example:
  python scripts/train_lm.py --corpus data/lm/l2arctic_corpus.txt \
      --out models/lm/l2arctic_4gram.pkl --order 4
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config           # noqa: E402
from asr_l2.lm.ngram import PyNGramLM            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True, help="text file, one sentence/line")
    ap.add_argument("--out", required=True, help="output .pkl path")
    ap.add_argument("--config", default=None)
    ap.add_argument("--order", type=int, default=None, help="n-gram order (def: config)")
    ap.add_argument("--add-k", type=float, default=0.1)
    ap.add_argument("--interp-lambda", type=float, default=0.7)
    ap.add_argument("--held-out-frac", type=float, default=0.1)
    args = ap.parse_args()

    cfg = load_config(args.config)
    order = args.order or cfg["lm"]["ngram_order"]

    sentences = [l.strip() for l in open(args.corpus, encoding="utf-8") if l.strip()]
    if not sentences:
        print(f"ERROR: empty corpus: {args.corpus}", file=sys.stderr)
        return 2

    rng = random.Random(cfg.get("seed", 1234))
    rng.shuffle(sentences)
    n_held = max(1, int(len(sentences) * args.held_out_frac))
    held, train = sentences[:n_held], sentences[n_held:]

    lm = PyNGramLM(order=order, add_k=args.add_k, interp_lambda=args.interp_lambda)
    lm.train(train)
    lm.save(args.out)

    train_ppl = lm.perplexity(train[: min(500, len(train))])
    held_ppl = lm.perplexity(held)
    print(">> DONE")
    print(f"   corpus: {len(sentences)} sentences  (train {len(train)} / held-out {n_held})")
    print(f"   stats:  {lm.stats()}")
    print(f"   perplexity  train~{train_ppl:.1f}  held-out~{held_ppl:.1f}")
    print(f"   saved:  {args.out}")
    if held_ppl > 1e4:
        print("   NOTE: very high perplexity -> corpus likely too small/sparse.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
