"""Minimal Praat TextGrid reader (dependency-free).

Handles both the long ("ooTextFile") and short TextGrid forms, extracting named
IntervalTiers as lists of (xmin, xmax, text). We only need interval tiers
("words", "phones") for L2-Arctic, so point/TextTier support is omitted.

L2-Arctic annotation phones tier encodes mispronunciations as "CPL,PPL,tag"
where tag in {s,d,a} (substitution / deletion / addition); correctly produced
phones carry just the phone label. See ``parse_l2arctic_phone``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass
class IntervalTier:
    name: str
    intervals: list[Interval]


def _read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_interval_tiers(path: str | Path) -> dict[str, IntervalTier]:
    """Return {tier_name: IntervalTier} for all interval tiers in the file."""
    text = _read_text(path)
    if "intervals [" in text or "intervals:" in text:
        return _parse_long(text)
    return _parse_short(text)


def _parse_long(text: str) -> dict[str, IntervalTier]:
    tiers: dict[str, IntervalTier] = {}
    # Split into per-tier blocks at "item [N]:".
    blocks = re.split(r"item\s*\[\d+\]\s*:", text)
    for blk in blocks[1:]:
        cls = re.search(r'class\s*=\s*"([^"]+)"', blk)
        if not cls or "IntervalTier" not in cls.group(1):
            continue
        name_m = re.search(r'name\s*=\s*"([^"]*)"', blk)
        name = name_m.group(1) if name_m else "unknown"
        intervals = []
        for m in re.finditer(
            r"intervals\s*\[\d+\]\s*:\s*"
            r"xmin\s*=\s*([\d.eE+-]+)\s*"
            r"xmax\s*=\s*([\d.eE+-]+)\s*"
            r'text\s*=\s*"((?:[^"]|"")*)"', blk):
            intervals.append(Interval(float(m.group(1)), float(m.group(2)),
                                      m.group(3).replace('""', '"').strip()))
        tiers[name] = IntervalTier(name, intervals)
    return tiers


def _parse_short(text: str) -> dict[str, IntervalTier]:
    # Short form: tokens are line-based. Walk lines pulling quoted/number tokens.
    lines = [ln.strip() for ln in text.splitlines()]
    tiers: dict[str, IntervalTier] = {}
    i = 0
    n = len(lines)

    def is_q(s: str) -> bool:
        return len(s) >= 2 and s.startswith('"') and s.endswith('"')

    while i < n:
        if lines[i] == '"IntervalTier"':
            name = lines[i + 1].strip('"') if i + 1 < n else "unknown"
            # i+2 xmin, i+3 xmax, i+4 size
            try:
                size = int(float(lines[i + 4]))
            except (ValueError, IndexError):
                i += 1
                continue
            intervals = []
            j = i + 5
            for _ in range(size):
                if j + 2 >= n:
                    break
                try:
                    xmin = float(lines[j]); xmax = float(lines[j + 1])
                except ValueError:
                    break
                txt = lines[j + 2]
                txt = txt.strip('"') if is_q(txt) else txt
                intervals.append(Interval(xmin, xmax, txt.strip()))
                j += 3
            tiers[name] = IntervalTier(name, intervals)
            i = j
        else:
            i += 1
    return tiers


# --- L2-Arctic specific -----------------------------------------------------

@dataclass
class PhoneError:
    xmin: float
    xmax: float
    canonical: str       # CPL: the phoneme that should be produced
    perceived: str       # PPL: what the annotator perceived (or "" )
    err_type: str        # "s" | "d" | "a" | "" (empty = correct)


def parse_l2arctic_phone(interval: Interval) -> PhoneError:
    """Decode one L2-Arctic 'phones' interval into a PhoneError.

    Annotated errors look like "AH,AA,s" (canonical, perceived, type). Correct
    phones are just "AH" (no commas) -> err_type == "".
    """
    parts = [p.strip() for p in interval.text.split(",")]
    if len(parts) >= 3:
        return PhoneError(interval.xmin, interval.xmax,
                          parts[0], parts[1], parts[2].lower())
    return PhoneError(interval.xmin, interval.xmax, interval.text.strip(), "", "")
