#!/usr/bin/env python
"""Build colab/asr_l2_edacc.ipynb: download EdAcc (2nd spontaneous corpus) and
test whether the "spontaneous speech defeats audio-free rescoring" finding (seen
on Svarah) replicates. Decodes Whisper n-best on EdAcc and rescores with GPT-2.

Needs the code zip (asr-l2-accent-code.zip from make_upload_zips.py). EdAcc is
public on HF (no token).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MD, CODE = "markdown", "code"

CELLS = [
    (MD, """# EdAcc — second spontaneous corpus (replication test)

Tests whether *spontaneous* speech defeats audio-free GPT-2 rescoring on an
**independent** corpus (EdAcc: Edinburgh International Accents of English), the
way it did on Svarah. EdAcc is public (CC-BY-SA, no token). **Runtime -> T4 GPU.**
You need `asr-l2-accent-code.zip` (from `make_upload_zips.py`)."""),

    (CODE, "!nvidia-smi -L"),
    (CODE, "!pip install -q datasets faster-whisper transformers jiwer librosa soundfile pandas pyarrow pyyaml tqdm"),

    (CODE, """import zipfile, os
from google.colab import files
up = files.upload()                      # asr-l2-accent-code.zip
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('/content')
os.chdir('/content/asr-l2-accent'); print('cwd:', os.getcwd())"""),

    (MD, """## 1. Download EdAcc (subsample, ~200 spontaneous clips)
The schema is auto-detected and printed. If the text/accent column mapping looks
wrong, re-run with the right names patched into `download_edacc.py`."""),
    (CODE, "!python scripts/download_edacc.py --split validation --max-utterances 200"),

    (MD, """## 2. Whisper n-best + GPT-2 rescoring on EdAcc
Same frozen Whisper-small.en, same audio-free GPT-2 rescoring as the main study.
The question: does GPT-2 help on spontaneous EdAcc (expected: little/none, as on Svarah)?"""),
    (CODE, "!python scripts/rescore_nbest.py --dataset edacc --lm neural:gpt2 --max-utterances 200"),
    (CODE, "!python scripts/bootstrap_significance.py --dataset edacc"),

    (CODE, """import json
s = json.load(open("results/edacc/phaseB_summary.json"))
b = s["test_wer_baseline"]
print(f"EdAcc GPT-2 rescoring: WER {b:.4f} -> {s['test_wer_adapted']:.4f} "
      f"({100*s['test_wer_delta']/b:+.1f}%)  lambda*={s['best_lambda']}")
print("=> spontaneous-speech null replicates" if s['test_wer_delta'] >= -0.001
      else "=> gain on EdAcc (differs from Svarah) -- worth a closer look")"""),

    (MD, "## 3. Download results"),
    (CODE, """import zipfile, glob
with zipfile.ZipFile("phaseB_edacc_results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("results/edacc/*.json") + glob.glob("results/edacc/*.csv") + \\
             glob.glob("data/processed/edacc/integrity.json"):
        z.write(f)
from google.colab import files
files.download("phaseB_edacc_results.zip")"""),
]


def main():
    nb = {"cells": [{"cell_type": t, "metadata": {},
                     **({"source": s.splitlines(keepends=True)} if t == MD else
                        {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
                    for t, s in CELLS],
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = REPO / "colab" / "asr_l2_edacc.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
