#!/usr/bin/env python
"""Generate colab/asr_l2_gpu.ipynb with a guaranteed-valid structure.

Hand-written / hand-edited .ipynb files are easy to corrupt (e.g. mixing string
and list `source` fields), which Colab rejects as a JSON/notebook error. This
builds the notebook with consistent fields and writes it deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS = [
    (MD, """# ASR L2-Accent - GPU runner (Colab)

Runs the heavy ASR sweeps (L2-Arctic harness, and later Phase B n-best
rescoring) on a free Colab T4 GPU - far faster than the 8 GB CPU box. The
pipeline code is identical; only `device` changes (AsrEngine auto-selects CUDA).

**Before running:** Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU.

You need two files made locally by `python scripts/make_upload_zips.py`
(written to the repository's parent folder): `asr-l2-accent-code.zip` and
`l2arctic_processed.zip`, plus your Hugging Face token (for Svarah)."""),

    (CODE, """# 1. Confirm GPU is attached
!nvidia-smi -L"""),

    (CODE, """# 2. Install deps  (jellyfish = phonetics; transformers = strong neural LM)
!pip install -q faster-whisper jiwer librosa soundfile pandas pyarrow pyyaml tqdm datasets jellyfish transformers"""),

    (CODE, """# 3. Upload the code zip (asr-l2-accent-code.zip) and unpack
import zipfile, os
from google.colab import files
up = files.upload()                      # pick asr-l2-accent-code.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('/content')
os.chdir('/content/asr-l2-accent')
print('cwd:', os.getcwd())
print(sorted(os.listdir()))"""),

    (CODE, """# 4. Hugging Face token (for Svarah). Prefer Colab Secrets (key icon) named HF_TOKEN.
import os
try:
    from google.colab import userdata
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
    print('HF_TOKEN loaded from Colab secrets')
except Exception:
    import getpass
    os.environ['HF_TOKEN'] = getpass.getpass('Paste HF token: ')"""),

    (MD, """## Data
Svarah is re-fetched from HF (gated, needs the token). L2-Arctic is uploaded as
the small processed set (300 clips) so you don't move the 7 GB of raw zips."""),

    (CODE, """# 5a. (Optional) Svarah on GPU - only if you also want to redo Svarah here
!python scripts/download_svarah.py --max-utterances 400"""),

    (CODE, """# 5b. L2-Arctic: upload l2arctic_processed.zip (zip of data/processed/l2arctic/)
import zipfile, os, json
from google.colab import files
os.makedirs('data/processed', exist_ok=True)
up = files.upload()                      # pick l2arctic_processed.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('data/processed')
# rewrite absolute wav_path in the manifest to the Colab location
mp = 'data/processed/l2arctic/manifest.jsonl'
rows = [json.loads(l) for l in open(mp)]
for r in rows:
    r['wav_path'] = os.path.abspath('data/processed/l2arctic/wav/%s.wav' % r['utt_id'])
open(mp, 'w').write('\\n'.join(json.dumps(r) for r in rows) + '\\n')
print('l2arctic clips:', len(rows))"""),

    (CODE, """# 6. Phase A on GPU - all chunk sizes incl 320ms (cheap on GPU) + error taxonomy
!python scripts/run_streaming_harness.py --dataset l2arctic --chunk-sizes 320,640,1280 --max-utterances 300
!python scripts/build_error_csv.py        --dataset l2arctic
!python scripts/join_l2arctic_phonemes.py
!python scripts/annotate_errors.py        --dataset l2arctic
!python scripts/report_phase_a.py         --dataset l2arctic"""),

    (CODE, """# 7. (Optional) Svarah sweep on GPU too - all conditions, full speed
# !python scripts/run_streaming_harness.py --dataset svarah --chunk-sizes 320,640,1280
# !python scripts/build_error_csv.py --dataset svarah
# !python scripts/report_phase_a.py --dataset svarah"""),

    (MD, """## Phase B - audio-free LM rescoring (frozen decoder)
Build a leakage-free domain LM (test sentences excluded), then re-rank the
decoder's n-best with it. n-best decoding is the GPU-heavy part."""),

    (CODE, """# B1. L2-Arctic: the full 2x2 -> weak 4-gram AND strong GPT-2, all 6 L1s.
!python scripts/build_text_corpus.py \\
    --prompts-sources data/processed/l2arctic/PROMPTS \\
    --exclude-manifest data/processed/l2arctic/manifest.jsonl \\
    --out data/lm/l2arctic_corpus.txt
!python scripts/train_lm.py --corpus data/lm/l2arctic_corpus.txt --out models/lm/l2arctic_4gram.pkl --order 4
# weak LM (4-gram); --max-utterances 300 = all 6 L1s, ~200 test clips for significance
!python scripts/rescore_nbest.py --dataset l2arctic --lm models/lm/l2arctic_4gram.pkl --max-utterances 300
!cp results/l2arctic/phaseB_summary.json results/l2arctic/phaseB_summary_ngram.json
!cp results/l2arctic/phaseB_rescore.csv  results/l2arctic/phaseB_rescore_ngram.csv
# strong LM (zero-shot GPT-2) on the SAME cached n-best; fine lambda grid
!python scripts/rescore_nbest.py --dataset l2arctic --lm neural:gpt2 --max-utterances 300 \\
    --lambda-grid 0,0.02,0.05,0.1,0.2,0.3,0.5,1.0
!cp results/l2arctic/phaseB_summary.json results/l2arctic/phaseB_summary_gpt2.json
!python scripts/phaseB_category_analysis.py --dataset l2arctic
!python scripts/bootstrap_significance.py   --dataset l2arctic
!python scripts/report_phase_b.py           --dataset l2arctic"""),

    (CODE, """# B2. Svarah: same 2x2 (weak 4-gram + strong GPT-2). Needs Svarah from cell 5a.
!python scripts/download_svarah.py --text-only data/lm/svarah_all_text.txt
!python scripts/build_text_corpus.py \\
    --text-sources data/lm/svarah_all_text.txt \\
    --exclude-manifest data/processed/svarah/manifest.jsonl \\
    --out data/lm/svarah_corpus.txt
!python scripts/train_lm.py --corpus data/lm/svarah_corpus.txt --out models/lm/svarah_4gram.pkl --order 4
!python scripts/rescore_nbest.py --dataset svarah --lm models/lm/svarah_4gram.pkl
!cp results/svarah/phaseB_summary.json results/svarah/phaseB_summary_ngram.json
!python scripts/rescore_nbest.py --dataset svarah --lm neural:gpt2 \\
    --lambda-grid 0,0.02,0.05,0.1,0.2,0.3,0.5,1.0
!cp results/svarah/phaseB_summary.json results/svarah/phaseB_summary_gpt2.json
!python scripts/bootstrap_significance.py   --dataset svarah
!python scripts/report_phase_b.py           --dataset svarah"""),

    (MD, """## Robustness experiments (reviewer-driven)
R1 breaks the single-speaker-per-L1 confound by using all 24 L2-ARCTIC speakers (4 per L1,
1200 clips) and adds a speaker-level cluster bootstrap. R2 checks the 2x2 holds across ASR
model size (tiny.en, base.en). Both are GPU-cheap."""),

    (CODE, """# R1. 24-SPEAKER robustness (Gaps 1 & 3) - PHASE B ONLY.
# NB: we intentionally SKIP the Phase A streaming sweep here - it takes ~80 min on
# 1200 clips and is NOT needed for the speaker-confound question. This keeps R1
# well inside the free-tier session, and we DOWNLOAD the result immediately so a
# later cell or a disconnect can never lose it.
import zipfile, os, json
from google.colab import files
up = files.upload()                          # pick l2arctic24_processed.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('data/processed')
mp = 'data/processed/l2arctic_24spk/manifest.jsonl'
rows = [json.loads(l) for l in open(mp)]
for r in rows:
    r['wav_path'] = os.path.abspath('data/processed/l2arctic_24spk/wav/%s.wav' % r['utt_id'])
open(mp, 'w').write('\\n'.join(json.dumps(r) for r in rows) + '\\n')
print('24-speaker clips:', len(rows))
!python scripts/build_text_corpus.py --prompts-sources data/processed/l2arctic_24spk/PROMPTS --exclude-manifest data/processed/l2arctic_24spk/manifest.jsonl --out data/lm/l2arctic24_corpus.txt
!python scripts/train_lm.py --corpus data/lm/l2arctic24_corpus.txt --out models/lm/l2arctic24_4gram.pkl --order 4
!python scripts/rescore_nbest.py --dataset l2arctic_24spk --lm models/lm/l2arctic24_4gram.pkl --max-utterances 1200
!cp results/l2arctic_24spk/phaseB_summary.json results/l2arctic_24spk/phaseB_summary_ngram.json
!python scripts/rescore_nbest.py --dataset l2arctic_24spk --lm neural:gpt2 --max-utterances 1200 --lambda-grid 0,0.02,0.05,0.1,0.2,0.3,0.5,1.0
!cp results/l2arctic_24spk/phaseB_summary.json results/l2arctic_24spk/phaseB_summary_gpt2.json
!python scripts/bootstrap_significance.py --dataset l2arctic_24spk
!python scripts/bootstrap_significance.py --dataset l2arctic_24spk --by-speaker
# --- save NOW (don't wait for the final cell) ---
!cd /content/asr-l2-accent && zip -qr r1_24spk.zip results/l2arctic_24spk
files.download('/content/asr-l2-accent/r1_24spk.zip')"""),

    (CODE, """# R2. Model-size robustness (Gap 2): tiny.en and base.en on L2-Arctic. Saves immediately.
import os; os.makedirs('results/model_robustness', exist_ok=True)
!python scripts/rescore_nbest.py --dataset l2arctic --model tiny.en --lm neural:gpt2 --max-utterances 300 --lambda-grid 0,0.02,0.05,0.1,0.2,0.3,0.5,1.0
!python scripts/bootstrap_significance.py --dataset l2arctic
!cp results/l2arctic/phaseB_summary.json results/model_robustness/summary_tiny.json
!cp results/l2arctic/phaseB_significance.json results/model_robustness/sig_tiny.json
!python scripts/rescore_nbest.py --dataset l2arctic --model base.en --lm neural:gpt2 --max-utterances 300 --lambda-grid 0,0.02,0.05,0.1,0.2,0.3,0.5,1.0
!python scripts/bootstrap_significance.py --dataset l2arctic
!cp results/l2arctic/phaseB_summary.json results/model_robustness/summary_base.json
!cp results/l2arctic/phaseB_significance.json results/model_robustness/sig_base.json
from google.colab import files
!cd /content/asr-l2-accent && zip -qr r2_models.zip results/model_robustness
files.download('/content/asr-l2-accent/r2_models.zip')"""),

    (CODE, """# 8. Zip results (CSV/Parquet/plots/reports - no audio) and download
!cd /content/asr-l2-accent && zip -qr results_gpu.zip results -x '*.wav'
from google.colab import files
files.download('/content/asr-l2-accent/results_gpu.zip')"""),
]


def _src_lines(text: str) -> list[str]:
    """nbformat source = list of lines, each ending in \\n except the last."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def main() -> None:
    cells = []
    for ctype, text in CELLS:
        cell = {"cell_type": ctype, "metadata": {}, "source": _src_lines(text)}
        if ctype == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = Path(__file__).resolve().parents[1] / "colab" / "asr_l2_gpu.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
