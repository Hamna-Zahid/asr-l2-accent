"""The unified utterance manifest shared by every dataset and pipeline stage.

A manifest is a JSON-Lines file: one ``Utterance`` per line. Keeping a single
flat schema means Phase A / Phase B scripts never need dataset-specific logic.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Utterance:
    """One evaluation utterance in the unified internal format."""

    utt_id: str                 # globally unique, e.g. "svarah_00012"
    dataset: str                # "svarah" | "l2arctic" | "speak_and_improve"
    wav_path: str               # absolute or repo-relative path to 16k mono WAV
    text: str                   # reference transcript (normalized at scoring time)
    duration_s: float
    speaker: str | None = None
    accent: str | None = None   # native language / accent label if known
    cefr: str | None = None     # proficiency level if known (Speak & Improve)
    extra: dict[str, Any] = field(default_factory=dict)  # dataset-specific fields


def write_manifest(path: str | Path, utts: Iterable[Utterance]) -> int:
    """Write utterances to a JSONL manifest. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for u in utts:
            fh.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> Iterator[Utterance]:
    """Yield ``Utterance`` objects from a JSONL manifest."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Utterance(**d)
