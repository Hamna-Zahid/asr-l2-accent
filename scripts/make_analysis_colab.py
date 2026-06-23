#!/usr/bin/env python
"""Build the Phase-B analysis-package Colab bundle + notebook.

The analysis package (oracle bound, LM-strength ladder, in-domain fine-tuned LM)
is entirely text-only on the cached n-best -- NO audio -- so the upload is tiny
(~2.5 MB) and the only reason for Colab is GPU for the big ladder rungs
(gpt2-large/xl) and the fine-tune.

Outputs:
  asr_l2_analysis_bundle.zip   (repo root; code + cached n-best + manifest + corpus + 4-gram)
  colab/asr_l2_analysis.ipynb
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = "asr-l2-accent"

# files/dirs to ship (all small, no audio)
DIRS = ["src", "scripts", "config"]
FILES = [
    "models/lm/l2arctic_4gram.pkl",
    "data/lm/l2arctic_corpus.txt",
    "data/processed/l2arctic_24spk/manifest.jsonl",
    "results/l2arctic_24spk/phaseB_nbest.jsonl",
    "results/l2arctic_24spk/phaseB_summary_gpt2.json",
    "results/l2arctic_24spk/phaseB_summary_ngram.json",
    "results/l2arctic_24spk/phaseB_oracle.json",
]


def build_bundle():
    out = REPO / "asr_l2_analysis_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in DIRS:
            for p in (REPO / d).rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    zf.write(p, f"{ROOT}/{p.relative_to(REPO).as_posix()}")
        for f in FILES:
            p = REPO / f
            if p.exists():
                zf.write(p, f"{ROOT}/{f}")
            else:
                print("  WARN missing:", f)
    print(f"wrote {out} ({out.stat().st_size//1024} KB)")


MD, CODE = "markdown", "code"
LADDER = (
    'neural_args = [\n'
    '  "models/lm/l2arctic_4gram.pkl=4-gram (weak)",\n'
    '  "neural:distilgpt2=distilGPT-2 (82M)",\n'
    '  "neural:gpt2=GPT-2 (124M)",\n'
    '  "neural:gpt2-medium=GPT-2-medium (355M)",\n'
    '  "neural:gpt2-large=GPT-2-large (774M)",\n'
    '  "neural:gpt2-xl=GPT-2-xl (1.5B)",\n'
    '  "neural:models/lm/ft_gpt2_l2arctic=GPT-2 fine-tuned (in-domain)",\n'
    ']'
)

CELLS = [
    (MD, """# ASR L2-Accent - Phase B analysis package (Colab)

Computes the **oracle head-room**, the **LM-strength ladder** (does the rescoring
gain scale with LM quality, and how close to the oracle ceiling?), and a
**genuinely fine-tuned in-domain LM** (audio-free domain adaptation).

All text-only on the cached n-best - no audio - so this is fast and the upload is
~2.5 MB. GPU is only needed for the big ladder rungs and the fine-tune.

**Before running:** Runtime -> Change runtime type -> T4 GPU.
You need `asr_l2_analysis_bundle.zip` (made locally by
`python scripts/make_analysis_colab.py`)."""),

    (CODE, "!nvidia-smi -L"),

    (CODE, "!pip install -q jiwer pandas pyarrow pyyaml tqdm transformers accelerate"),

    (CODE, """# Upload asr_l2_analysis_bundle.zip and unpack
import zipfile, os
from google.colab import files
up = files.upload()
with zipfile.ZipFile(next(iter(up))) as z:
    z.extractall('/content')
os.chdir('/content/asr-l2-accent')
print('cwd:', os.getcwd()); print(sorted(os.listdir()))"""),

    (MD, "## 1. Oracle upper bound (instant, no GPU)\nHow much of the error is recoverable from the n-best by reranking alone."),
    (CODE, "!python scripts/oracle_bound.py --dataset l2arctic_24spk"),

    (MD, """## 2. Fine-tune a small LM on the in-domain corpus (GPU)
Leakage-guarded (asserts zero overlap with the eval prompts). Turns the zero-shot
neural rescorer into a domain-adapted one."""),
    (CODE, "!python scripts/finetune_lm.py --base gpt2 --epochs 3"),

    (MD, """## 3. LM-strength ladder (GPU for the big rungs)
4-gram -> distilGPT-2 -> GPT-2 -> medium -> large -> xl (1.5B), plus the
fine-tuned in-domain GPT-2. Reports per-word perplexity (strength axis), test
WER delta, and fraction of the oracle head-room captured."""),
    (CODE, LADDER + """
import subprocess, sys
cmd = [sys.executable, "scripts/lm_ladder.py", "--dataset", "l2arctic_24spk", "--lms", *neural_args]
print(" ".join(cmd))
subprocess.run(cmd, check=True)"""),

    (CODE, """# Inspect the ladder
import json
d = json.load(open("results/l2arctic_24spk/phaseB_lm_ladder.json"))
print(f"baseline {d['test_wer_baseline']:.4f}  oracle {d['test_wer_oracle']:.4f}")
for r in d["ladder"]:
    cap = r["frac_headroom_captured"]
    print(f"  {r['label']:34s} ppl={r['ppl_word']:7.1f}  "
          f"WER->{r['test_wer_adapted']:.4f} ({r['delta_rel_pct']:+5.1f}%)  "
          f"headroom {None if cap is None else round(cap*100)}%")"""),

    (MD, "## 4. Download results"),
    (CODE, """import zipfile, glob
out = "phaseB_analysis_results.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob("results/l2arctic_24spk/phaseB_lm_ladder.json") + \\
             glob.glob("results/l2arctic_24spk/phaseB_oracle.json"):
        z.write(f)
    # ship the fine-tuned LM too (small) so the ladder is reproducible offline
    for f in glob.glob("models/lm/ft_gpt2_l2arctic/**/*", recursive=True):
        z.write(f)
from google.colab import files
files.download(out)
print("downloaded", out)"""),
]


def build_notebook():
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {},
             **({"source": s.splitlines(keepends=True)} if t == MD else
                {"source": s.splitlines(keepends=True), "outputs": [], "execution_count": None})}
            for t, s in CELLS
        ],
        "metadata": {"accelerator": "GPU",
                     "colab": {"provenance": []},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 0,
    }
    out = REPO / "colab" / "asr_l2_analysis.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    build_bundle()
    build_notebook()
