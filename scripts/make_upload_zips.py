#!/usr/bin/env python
"""Build the Colab upload artifacts with POSIX (forward-slash) paths.

PowerShell's Compress-Archive writes backslash separators that break extraction
on Linux/Colab, so we build the zips with Python's zipfile instead. Outputs to
the PARENT folder so the zips are never committed to the repo:
  ../asr-l2-accent-code.zip   (code: src, scripts, config, docs)
  ../l2arctic_processed.zip   (the small processed L2-Arctic set)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent

CODE_ITEMS = ["src", "scripts", "config", "requirements.txt", "README.md",
              "DATA.md", "ANNOTATION.md", "LICENSE", ".gitignore", "colab"]


def _add(zf: zipfile.ZipFile, path: Path, arc_root: str) -> int:
    n = 0
    if path.is_file():
        zf.write(path, arcname=f"{arc_root}/{path.name}")
        return 1
    for p in sorted(path.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if p.is_file():
            rel = p.relative_to(path.parent).as_posix()  # forward slashes
            zf.write(p, arcname=f"{arc_root}/{rel}" if arc_root else rel)
            n += 1
    return n


def main() -> None:
    code_zip = OUT / "asr-l2-accent-code.zip"
    with zipfile.ZipFile(code_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        total = 0
        for item in CODE_ITEMS:
            p = ROOT / item
            if not p.exists():
                continue
            # arcname root "asr-l2-accent" so it unzips to that folder
            if p.is_file():
                zf.write(p, arcname=f"asr-l2-accent/{item}")
                total += 1
            else:
                total += _add(zf, p, "asr-l2-accent")
    print(f"wrote {code_zip} ({code_zip.stat().st_size/1e6:.2f} MB, {total} files)")

    # Processed L2-Arctic sets (6-speaker default and optional 24-speaker).
    for sub, zipname in [("l2arctic", "l2arctic_processed.zip"),
                         ("l2arctic_24spk", "l2arctic24_processed.zip")]:
        l2 = ROOT / "data" / "processed" / sub
        if l2.exists():
            l2_zip = OUT / zipname
            with zipfile.ZipFile(l2_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                n = _add(zf, l2, "")   # unzips to <sub>/ under data/processed
            print(f"wrote {l2_zip} ({l2_zip.stat().st_size/1e6:.1f} MB, {n} files)")
        else:
            print(f"SKIP {zipname}: {l2} not found")


if __name__ == "__main__":
    main()
