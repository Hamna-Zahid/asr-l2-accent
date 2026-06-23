#!/usr/bin/env python
"""Phase B.1 — assemble the domain text corpus for LM training.

Pulls domain text from any mix of:
  --manifest-sources  one or more manifest.jsonl files (uses each row's `text`)
  --text-sources      one or more plain UTF-8 files (one sentence per line)
  --prompts-sources   L2-Arctic-style PROMPTS files ( "( id  \"sentence\" )" )

and writes a single normalized, de-duplicated corpus (one sentence per line)
plus a provenance report.

*** LEAKAGE GUARD (read this) ***
Pass --exclude-manifest pointing at your TEST manifest. Every sentence whose
normalized form matches a test reference is REMOVED from the corpus. This
matters enormously for L2-Arctic: all speakers read the same ARCTIC prompts, so
without this the LM would memorize the exact test sentences and rescoring would
be meaninglessly perfect. Always exclude the test set.

LICENSE: derived transcript text may be license-restricted (see DATA.md). The
corpus is written under data/ (git-ignored). Do NOT commit it to a public repo
unless every source's license permits redistributing derived text.

Examples:
  # L2-Arctic LM: train on the full ARCTIC prompt list minus the test sentences
  python scripts/build_text_corpus.py \
      --prompts-sources /path/to/l2arctic_release_v5.0/PROMPTS \
      --exclude-manifest data/processed/l2arctic/manifest.jsonl \
      --out data/lm/l2arctic_corpus.txt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                    # noqa: E402
from asr_l2.scoring.normalize import normalize_text       # noqa: E402

_PROMPT_RE = re.compile(r'\(\s*\S+\s+"(.+?)"\s*\)')


def _from_manifest(path: Path) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line).get("text", ""))
    return out


def _from_text(path: Path) -> list[str]:
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def _from_prompts(path: Path) -> list[str]:
    """Parse CMU/L2-Arctic PROMPTS lines like: ( arctic_a0001 \"text here\" )."""
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _PROMPT_RE.search(line)
        if m:
            out.append(m.group(1))
        elif line.strip():
            out.append(line.strip())   # tolerate plain-sentence prompt files
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest-sources", nargs="*", default=[])
    ap.add_argument("--text-sources", nargs="*", default=[])
    ap.add_argument("--prompts-sources", nargs="*", default=[])
    ap.add_argument("--exclude-manifest", default=None,
                    help="TEST manifest whose sentences are removed (leakage guard)")
    ap.add_argument("--min-words", type=int, default=2,
                    help="drop very short sentences below this word count")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    load_config(args.config)  # validates config presence; paths resolved by caller

    # Build the exclude set of normalized test references.
    exclude: set[str] = set()
    if args.exclude_manifest:
        for t in _from_manifest(Path(args.exclude_manifest)):
            n = normalize_text(t)
            if n:
                exclude.add(n)

    provenance = {"sources": [], "n_raw": 0, "n_excluded_leakage": 0,
                  "n_too_short": 0, "n_duplicate": 0, "n_final": 0,
                  "exclude_manifest": args.exclude_manifest,
                  "exclude_set_size": len(exclude)}

    seen: set[str] = set()
    kept: list[str] = []

    def ingest(sentences: list[str], label: str):
        n_src = 0
        for s in sentences:
            provenance["n_raw"] += 1
            norm = normalize_text(s)
            if not norm or len(norm.split()) < args.min_words:
                provenance["n_too_short"] += 1
                continue
            if norm in exclude:
                provenance["n_excluded_leakage"] += 1
                continue
            if norm in seen:
                provenance["n_duplicate"] += 1
                continue
            seen.add(norm)
            kept.append(norm)
            n_src += 1
        provenance["sources"].append({"source": label, "kept": n_src})

    for p in args.manifest_sources:
        ingest(_from_manifest(Path(p)), f"manifest:{p}")
    for p in args.prompts_sources:
        ingest(_from_prompts(Path(p)), f"prompts:{p}")
    for p in args.text_sources:
        ingest(_from_text(Path(p)), f"text:{p}")

    provenance["n_final"] = len(kept)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n", encoding="utf-8")
    (out.with_suffix(".provenance.json")).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    print(">> DONE")
    print(json.dumps(provenance, indent=2))
    if not args.exclude_manifest:
        print("\nWARNING: no --exclude-manifest given. If any source overlaps your "
              "TEST set, the LM will leak. Pass your test manifest to be safe.",
              file=sys.stderr)
    if len(kept) < 100:
        print(f"\nNOTE: only {len(kept)} sentences — a small LM. Add more domain "
              f"text (IELTS/TOEFL transcripts, in-domain corpora) for a stronger LM.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
