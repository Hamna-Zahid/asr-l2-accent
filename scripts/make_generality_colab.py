#!/usr/bin/env python
"""Build the Colab notebook for the second-ASR-family (wav2vec2) generality run.

Unlike the text-only analysis package, this needs AUDIO (wav2vec2 decodes the
clips), so it uploads the processed L2-ARCTIC 24-speaker audio. It then re-runs
the entire Phase B analysis on the wav2vec2 n-best so the 2x2 / oracle / ladder
can be compared against Whisper.

Outputs colab/asr_l2_generality.ipynb. Needs (made by make_upload_zips.py):
  asr-l2-accent-code.zip  and  l2arctic24_processed.zip
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# Second ASR family (wav2vec2) — generality run

Re-runs Phase B on a **wav2vec2-CTC** model (a different architecture from
Whisper) so we can show the LM-strength 2x2, the oracle ceiling, and the ladder
are not Whisper-specific. wav2vec2 decodes audio, so this uploads the processed
24-speaker L2-ARCTIC set.

**Runtime → T4 GPU.** You need `asr-l2-accent-code.zip` and
`l2arctic24_processed.zip` (from `python scripts/make_upload_zips.py`)."""),

    (CODE, "!nvidia-smi -L"),
    (CODE, "!pip install -q faster-whisper transformers pyctcdecode jiwer librosa soundfile pandas pyarrow pyyaml tqdm"),

    (CODE, """# Upload the code zip and unpack
import zipfile, os
from google.colab import files
up = files.upload()                      # asr-l2-accent-code.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('/content')
os.chdir('/content/asr-l2-accent'); print('cwd:', os.getcwd())"""),

    (CODE, """# Upload the 24-speaker processed audio and fix manifest paths
import zipfile, os, json
from google.colab import files
os.makedirs('data/processed', exist_ok=True)
up = files.upload()                      # l2arctic24_processed.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('data/processed')
mp = 'data/processed/l2arctic_24spk/manifest.jsonl'
rows = [json.loads(l) for l in open(mp)]
wav_dir = '/content/asr-l2-accent/data/processed/l2arctic_24spk/wav'
for r in rows:                            # rewrite wav_path to the Colab location
    fn = r['wav_path'].replace(chr(92), '/').split('/')[-1]   # chr(92)='\\' -> handle Windows paths
    r['wav_path'] = wav_dir + '/' + fn
open(mp, 'w').write('\\n'.join(json.dumps(r) for r in rows) + '\\n')
print('clips:', len(rows), '| sample wav exists:', os.path.exists(rows[0]['wav_path']))"""),

    (CODE, """# Safety net: make sure the weak-LM (4-gram) exists; rebuild from the
# leakage-guarded corpus if an older code zip omitted it.
import os
if not os.path.exists('models/lm/l2arctic_4gram.pkl'):
    os.makedirs('models/lm', exist_ok=True)
    !python scripts/train_lm.py --corpus data/lm/l2arctic_corpus.txt --out models/lm/l2arctic_4gram.pkl --order 4
print('4-gram LM present:', os.path.exists('models/lm/l2arctic_4gram.pkl'))"""),

    (MD, "## 1. Decode wav2vec2 n-best (GPU) — the only audio-heavy step"),
    (CODE, """!python scripts/decode_nbest.py --engine wav2vec2 \\
        --dataset l2arctic_24spk --out l2arctic_24spk_w2v2 --nbest 10"""),

    (MD, "## 2. Phase B on wav2vec2: rescore + significance (mirrors the Whisper run)"),
    (CODE, """# --no-decode = use ONLY the wav2vec2 cache (never fall back to Whisper);
# --max-utterances 1200 = all clips; wide lambda grid because CTC acoustic scores
# are scaled differently than Whisper's (the optimum sits well above 1.0).
GRID = "0.0,0.2,0.5,1.0,2.0,3.0,5.0,8.0,12.0"
!python scripts/rescore_nbest.py --dataset l2arctic_24spk_w2v2 --no-decode --max-utterances 1200 --lambda-grid $GRID --lm models/lm/l2arctic_4gram.pkl
import os; os.replace('results/l2arctic_24spk_w2v2/phaseB_summary.json', 'results/l2arctic_24spk_w2v2/phaseB_summary_ngram.json')
!python scripts/rescore_nbest.py --dataset l2arctic_24spk_w2v2 --no-decode --max-utterances 1200 --lambda-grid $GRID --lm neural:gpt2
import os; os.replace('results/l2arctic_24spk_w2v2/phaseB_summary.json', 'results/l2arctic_24spk_w2v2/phaseB_summary_gpt2.json')
!python scripts/bootstrap_significance.py --dataset l2arctic_24spk_w2v2 --by-speaker"""),

    (MD, "## 3. Oracle ceiling + LM-strength ladder on wav2vec2"),
    (CODE, "!python scripts/oracle_bound.py --dataset l2arctic_24spk_w2v2"),
    (CODE, """neural_args = [
  "models/lm/l2arctic_4gram.pkl=4-gram (weak)",
  "neural:distilgpt2=distilGPT-2 (82M)",
  "neural:gpt2=GPT-2 (124M)",
  "neural:gpt2-medium=GPT-2-medium (355M)",
  "neural:gpt2-large=GPT-2-large (774M)",
]
import subprocess, sys
subprocess.run([sys.executable, "scripts/lm_ladder.py", "--dataset", "l2arctic_24spk_w2v2",
                "--grid", "0.0,0.2,0.5,1.0,2.0,3.0,5.0,8.0,12.0", "--lms", *neural_args], check=True)"""),

    (CODE, """# Quick look: does the 2x2 hold on wav2vec2?
import json
for tag in ["ngram", "gpt2"]:
    s = json.load(open(f"results/l2arctic_24spk_w2v2/phaseB_summary_{tag}.json"))
    d = s["test_wer_delta"]; b = s["test_wer_baseline"]
    print(f"  {tag:6s}: WER {b:.4f} -> {s['test_wer_adapted']:.4f}  ({100*d/b:+.1f}%)  lambda*={s['best_lambda']}")
o = json.load(open("results/l2arctic_24spk_w2v2/phaseB_oracle.json"))
print(f"  oracle: baseline {o['test_wer_baseline_top1']:.4f} -> {o['test_wer_oracle']:.4f} "
      f"({o['headroom_rel_pct']:.0f}% recoverable)")"""),

    (MD, "## 4. Download results"),
    (CODE, """import zipfile, glob
with zipfile.ZipFile("phaseB_wav2vec2_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("results/l2arctic_24spk_w2v2/*.json") + \\
             glob.glob("results/l2arctic_24spk_w2v2/phaseB_nbest.jsonl") + \\
             glob.glob("results/l2arctic_24spk_w2v2/*.csv"):
        z.write(f)
from google.colab import files
files.download("phaseB_wav2vec2_results.zip")"""),
]


def main():
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"source": s.splitlines(keepends=True)} if t == MD else
                {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
            for t, s in CELLS
        ],
        "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    out = REPO / "colab" / "asr_l2_generality.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
