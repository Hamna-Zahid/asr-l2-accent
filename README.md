# Where Cascaded ASR Pipelines Fail on Accented Learner Speech — and What Audio-Free Adaptation Can (and Can't) Fix

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20817700.svg)](https://doi.org/10.5281/zenodo.20817700)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible, CPU-friendly research pipeline that **(A)** measures *where and why*
a streaming Whisper ASR pipeline fails on accented second-language (L2) English —
separating the streaming/chunking penalty from accent-driven and model errors —
and **(B)** tests whether an **audio-free, text-only** adaptation (n-best rescoring
on a frozen decoder) recovers any of that error, evaluated through the same error
taxonomy.

This repository accompanies the paper (see [Citation](#paper--citation)). It
contains all code, configuration, and **derived** results (WER/CER tables, error
taxonomies, rescoring outputs, significance estimates, figures). **No audio is
included** — each corpus is fetched from its official source by a provided script.

## Key findings

- **Streaming penalty is large and monotonic.** Going from offline to 320 ms
  chunks raises WER up to ~5× (4.8× on L2-ARCTIC, 5.3× on Svarah).
- **About half of L2-ARCTIC errors are accent-driven** (51% coincide with the
  corpus's human phoneme-mispronunciation annotations); the rest are model errors
  on correctly-produced speech.
- **Audio-free rescoring follows a clean 2×2** in LM strength × speech type. A
  weak n-gram never yields a test gain. A strong GPT-2 cuts L2-ARCTIC read-speech
  WER by **8.0% relative** across 24 speakers (speaker-level cluster bootstrap
  *p* = 0.0003), holds across ASR model sizes, but gives **nothing** on
  spontaneous Svarah.
- **An oracle shows half the error is recoverable from the n-best**, yet the gain
  saturates with LM strength at only ~a quarter of that head-room (a fine-tuned
  124 M LM matches a 1.5 B one) — the rest is acoustic/accent error invisible to
  any text model.

## Hardware assumptions

Built and tested on a deliberately constrained target:

- **8 GB RAM**, **CPU-only** (Intel i5-4300U class, no CUDA GPU) for Phase A and
  all text-only rescoring.
- ASR via **faster-whisper / CTranslate2** with `int8` quantization; primary
  model **`small.en`**, with `tiny.en`/`base.en` for the model-size study.
- A free **Colab T4 GPU** is used only for the heavier all-condition / n-best
  decoding and the larger LMs in the ladder; every step has a CPU fallback.

## Setup

```bash
# Python 3.11 (the ML stack lacks 3.12+ wheels for some pins as of 2026-06).
python3.11 -m venv .venv
. .venv/Scripts/activate            # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

KenLM is optional; without a C++ toolchain the pipeline falls back to a
pure-Python n-gram LM automatically.

## Datasets

See **[DATA.md](DATA.md)**. No audio is committed. **Svarah** (Indian-accented
English) is gated — put `HF_TOKEN=hf_...` in a repo-root `.env`. **L2-ARCTIC**
(phoneme annotations) is downloaded from its official source under its license.

## Reproducing the results

```bash
# --- Data ---
python scripts/download_svarah.py    --max-utterances 400
python scripts/download_l2arctic.py  --archive /path/to/extracted/l2arctic

# --- Phase A: streaming + error diagnostics ---
python scripts/run_streaming_harness.py --dataset svarah
python scripts/build_error_csv.py       --dataset l2arctic
python scripts/join_l2arctic_phonemes.py
python scripts/report_phase_a.py

# --- Phase B: audio-free n-best rescoring ---
python scripts/build_text_corpus.py
python scripts/train_lm.py
python scripts/rescore_nbest.py         --dataset l2arctic_24spk --lm models/lm/l2arctic_4gram.pkl
python scripts/rescore_nbest.py         --dataset l2arctic_24spk --lm neural:gpt2
python scripts/bootstrap_significance.py --dataset l2arctic_24spk --by-speaker
python scripts/report_phase_b.py

# --- Phase B analysis package (oracle ceiling, LM ladder, fine-tuned LM) ---
python scripts/oracle_bound.py          --dataset l2arctic_24spk
python scripts/finetune_lm.py           --base gpt2 --epochs 3
python scripts/lm_ladder.py             --dataset l2arctic_24spk

# --- Figures ---
python scripts/make_paper_figures.py
python scripts/make_streaming_figure.py
python scripts/make_hero_figure.py
python scripts/make_oracle_figure.py
python scripts/make_ladder_figure.py
```

The heavy GPU steps (all-condition / n-best decoding, large ladder LMs, fine-tune)
have ready-to-run Colab notebooks in **`colab/`** (`asr_l2_gpu.ipynb`,
`asr_l2_analysis.ipynb`). All random subsampling, dev/test splits, and bootstrap
resampling use fixed seeds (`config/default.yaml`).

## Annotation

Error-cause labels are produced by transparent text rules plus L2-ARCTIC's
phoneme ground truth; methodology and the `why_category` vocabulary are in
**[ANNOTATION.md](ANNOTATION.md)**. A 50-error stratified spot-check sheet is
released for independent validation (manual re-adjudication agreement: 88%).

## Repository layout

```
config/    default.yaml (all knobs)
src/asr_l2/  io, asr, vad, scoring, errors, lm, report
scripts/   standalone CLI entry points (one per pipeline stage)
colab/     GPU notebooks for the heavy decoding + analysis package
paper/     manuscript sources (LaTeX + docx generator) and figures
data/      git-ignored; created by download scripts (no audio committed)
results/   derived CSV/Parquet/JSON/plots (safe to release)
```

## Paper & Citation

Archived code & derived results: **Zenodo DOI [10.5281/zenodo.20817700](https://doi.org/10.5281/zenodo.20817700)**.

```bibtex
@misc{zahid2026asrl2,
  title  = {Where Cascaded ASR Pipelines Fail on Accented Learner Speech,
            and What Audio-Free Adaptation Can (and Can't) Fix},
  author = {Zahid, Hamna},
  year   = {2026},
  doi    = {10.5281/zenodo.20817700},
  note   = {Code and derived results: https://doi.org/10.5281/zenodo.20817700}
}
```

## License

Code: **MIT** (see [LICENSE](LICENSE)). Each dataset retains its own license
(see [DATA.md](DATA.md)); no audio is redistributed here.
