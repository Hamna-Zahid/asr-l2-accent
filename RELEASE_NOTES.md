# Release v1.0.0 — Paper release

First public release accompanying the paper *"Where Cascaded ASR Pipelines Fail
on Accented Learner Speech, and What Audio-Free Adaptation Can (and Can't) Fix."*

This archive contains the complete, reproducible pipeline and all **derived**
results behind the paper. **No raw audio is included** — each corpus (Svarah,
L2-ARCTIC) is fetched from its official source by a provided script, under its
own license.

## What's included

- **Phase A** — streaming-vs-offline harness, error alignment and taxonomy,
  L2-ARCTIC phoneme cross-referencing, and the accent-vs-model decomposition.
- **Phase B** — audio-free n-best rescoring (weak 4-gram and neural LMs),
  leakage-guarded LM training, λ tuning, and utterance- and speaker-level
  cluster bootstrap significance testing.
- **Analysis package** — oracle upper bound, the LM-strength ladder
  (4-gram → GPT-2-xl + a fine-tuned in-domain LM), and the model-size study.
- **Figures** — all paper figures (audio-grounded streaming/rescoring
  illustrations, oracle head-room, LM-strength ladder) as 300-DPI PNG + vector
  PDF, regenerable from the released results.
- **Manuscript sources** under `paper/`, and the 50-error annotation spot-check
  sheet.
- Ready-to-run **Colab notebooks** for the GPU steps.

## Reproducibility

All subsampling, dev/test splits, and bootstrap resampling use fixed seeds
(`config/default.yaml`). Offline WER matched across CPU (int8) and GPU (float16)
decoding, confirming hardware-independence of the reported numbers. See
`README.md` for the end-to-end command sequence.

## Headline results

- Streaming penalty up to ~5× WER (offline → 320 ms chunks).
- ~51% of L2-ARCTIC errors are accent-driven (phoneme-confirmed).
- Strong-LM rescoring: −8.0% relative WER on L2-ARCTIC read speech
  (24 speakers, speaker-level *p* = 0.0003); no gain on spontaneous Svarah.
- Oracle shows ~50% of the error is recoverable from the n-best, but text-only
  rescoring saturates at ~a quarter of that ceiling — the rest is acoustic.

## License

MIT for the code; datasets retain their own licenses (see `DATA.md`).
