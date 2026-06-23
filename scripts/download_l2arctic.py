#!/usr/bin/env python
"""Phase A — prepare L2-Arctic into the unified format + extract phoneme errors.

L2-Arctic is NOT anonymously downloadable: you must accept its license and
download the archive yourself (see DATA.md / https://psi.engr.tamu.edu/
l2-arctic-corpus/). This script then converts your *locally extracted* copy.

Expected extracted layout (one folder per speaker):
  <root>/<SPK>/wav/arctic_*.wav
  <root>/<SPK>/transcript/arctic_*.txt
  <root>/<SPK>/annotation/arctic_*.TextGrid   (manual; ~150 utts/speaker)
  <root>/<SPK>/textgrid/arctic_*.TextGrid      (forced-aligned; optional)

Outputs:
  data/processed/l2arctic/wav/<utt_id>.wav            (16 kHz mono)
  data/processed/l2arctic/manifest.jsonl              (unified Utterance list)
  data/processed/l2arctic/phone_annotations.jsonl     (manual phoneme errors)

The phone_annotations file is the ground truth joined against ASR errors in
Phase A.3 (scripts/join_l2arctic_phonemes.py).

Example:
  python scripts/download_l2arctic.py --archive /path/to/l2arctic_release_v5
  python scripts/download_l2arctic.py --archive ... --annotated-only --max-utterances 300
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config                        # noqa: E402
from asr_l2.io.audio import convert_to_wav16k                # noqa: E402
from asr_l2.io.manifest import Utterance, write_manifest     # noqa: E402
from asr_l2.io.textgrid import (read_interval_tiers,         # noqa: E402
                                parse_l2arctic_phone)

# Map each L2-Arctic speaker to their native language (L1). From the corpus
# paper; used to populate the `accent` field. Unknown speakers -> None.
SPEAKER_L1 = {
    "ABA": "Arabic", "SKA": "Arabic", "YBAA": "Arabic", "ZHAA": "Arabic",
    "BWC": "Mandarin", "LXC": "Mandarin", "NCC": "Mandarin", "TXHC": "Mandarin",
    "ASI": "Hindi", "RRBI": "Hindi", "SVBI": "Hindi", "TNI": "Hindi",
    "HJK": "Korean", "HKK": "Korean", "YDCK": "Korean", "YKWK": "Korean",
    "EBVS": "Spanish", "ERMS": "Spanish", "MBMPS": "Spanish", "NJS": "Spanish",
    "HQTV": "Vietnamese", "PNV": "Vietnamese", "THV": "Vietnamese", "TLV": "Vietnamese",
}

# Default representative subset: one speaker per L1 (6 of 24). Keeps ASR runtime
# tractable on a slow CPU while covering every accent group for the A.3 analysis.
DEFAULT_SUBSET = ["ABA", "BWC", "ASI", "HJK", "EBVS", "HQTV"]


def _read_transcript(spk_dir: Path, stem: str) -> str | None:
    txt = spk_dir / "transcript" / f"{stem}.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8", errors="replace").strip()
    return None


def _extract_phone_errors(tg_path: Path) -> list[dict] | None:
    """Return the phones tier as a list of phone-error dicts, or None."""
    try:
        tiers = read_interval_tiers(tg_path)
    except Exception as e:  # noqa: BLE001 — be robust to odd TextGrids
        print(f"   WARN: failed to parse {tg_path.name}: {e}", file=sys.stderr)
        return None
    phones = tiers.get("phones") or tiers.get("phone")
    if phones is None:
        return None
    out = []
    for iv in phones.intervals:
        if not iv.text.strip():
            continue
        pe = parse_l2arctic_phone(iv)
        out.append({"xmin": round(pe.xmin, 4), "xmax": round(pe.xmax, 4),
                    "canonical": pe.canonical, "perceived": pe.perceived,
                    "err_type": pe.err_type})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True,
                    help="path to the extracted L2-Arctic root folder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-utterances", type=int, default=None,
                    help="global cap across all speakers")
    ap.add_argument("--max-per-speaker", type=int, default=None,
                    help="cap per speaker (spreads budget evenly across L1s)")
    ap.add_argument("--annotated-only", action="store_true",
                    help="keep only utterances that have a manual annotation")
    ap.add_argument("--speakers", default=None,
                    help="comma-separated speaker IDs (default: representative "
                         "6-speaker subset when zips are present)")
    ap.add_argument("--all-speakers", action="store_true",
                    help="process all 24 speakers (slow; extracts ~7GB)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.archive)
    if not root.exists():
        print(f"ERROR: archive path not found: {root}", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else \
        Path(cfg["paths"]["data_root"]) / "processed" / "l2arctic"
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Which speakers? Explicit --speakers wins; otherwise, if the folder holds
    # per-speaker zips (unextracted L2-Arctic), default to the representative
    # subset to keep runtime sane; if already-extracted folders exist, use all.
    if args.speakers:
        wanted = [s.strip() for s in args.speakers.split(",")]
    elif args.all_speakers:
        wanted = None
    else:
        zips_present = any((root / f"{s}.zip").exists() for s in SPEAKER_L1)
        wanted = DEFAULT_SUBSET if zips_present else None
        if zips_present:
            print(f">> no --speakers given; using default subset {DEFAULT_SUBSET} "
                  f"(pass --all-speakers or --speakers to change)")

    # Auto-extract any speaker zip whose folder isn't already present.
    targets = wanted if wanted is not None else [p.stem for p in root.glob("*.zip")
                                                 if p.stem in SPEAKER_L1]
    for spk in targets:
        spk_dir = root / spk
        spk_zip = root / f"{spk}.zip"
        if not (spk_dir / "wav").is_dir() and spk_zip.exists():
            print(f">> extracting {spk}.zip ...")
            with zipfile.ZipFile(spk_zip) as zf:
                zf.extractall(root)

    # Discover speaker folders: a dir with a wav/ subfolder.
    spk_dirs = sorted(p for p in root.iterdir()
                      if p.is_dir() and (p / "wav").is_dir()
                      and (wanted is None or p.name in wanted))
    if not spk_dirs:
        print(f"ERROR: no speaker folders with a wav/ subdir under {root}.\n"
              f"       Found zips: {[p.name for p in root.glob('*.zip')]}\n"
              f"       Is this the L2-Arctic root (per-speaker zips or folders)?",
              file=sys.stderr)
        return 3
    print(f">> using {len(spk_dirs)} speakers: {[p.name for p in spk_dirs]}")

    utts: list[Utterance] = []
    annotations: dict[str, list[dict]] = {}
    total_dur = 0.0
    n_annotated = 0

    for spk_dir in spk_dirs:
        spk = spk_dir.name
        l1 = SPEAKER_L1.get(spk)
        ann_dir = spk_dir / "annotation"
        spk_count = 0
        for wav in sorted((spk_dir / "wav").glob("*.wav")):
            if args.max_per_speaker and spk_count >= args.max_per_speaker:
                break  # spread the budget evenly across L1s
            stem = wav.stem
            text = _read_transcript(spk_dir, stem)
            if not text:
                continue
            tg = ann_dir / f"{stem}.TextGrid"
            phone_errs = _extract_phone_errors(tg) if tg.exists() else None
            has_ann = phone_errs is not None
            if args.annotated_only and not has_ann:
                continue

            utt_id = f"l2arctic_{spk}_{stem}"
            dst = wav_dir / f"{utt_id}.wav"
            try:
                dur = convert_to_wav16k(wav, dst)
            except Exception as e:  # noqa: BLE001
                print(f"   WARN: skip {wav.name}: {e}", file=sys.stderr)
                continue
            total_dur += dur
            if has_ann:
                annotations[utt_id] = phone_errs
                n_annotated += 1
            spk_count += 1
            utts.append(Utterance(
                utt_id=utt_id, dataset="l2arctic", wav_path=str(dst),
                text=text, duration_s=round(dur, 3), speaker=spk, accent=l1,
                extra={"has_phone_annotation": has_ann,
                       "n_phone_errors": sum(1 for p in (phone_errs or [])
                                             if p["err_type"])},
            ))
            if args.max_utterances and len(utts) >= args.max_utterances:
                break
        if args.max_utterances and len(utts) >= args.max_utterances:
            break

    write_manifest(out_dir / "manifest.jsonl", utts)
    with open(out_dir / "phone_annotations.jsonl", "w", encoding="utf-8") as fh:
        for utt_id, errs in annotations.items():
            fh.write(json.dumps({"utt_id": utt_id, "phones": errs},
                                ensure_ascii=False) + "\n")

    report = {
        "dataset": "l2arctic", "speakers": [p.name for p in spk_dirs],
        "utterances_written": len(utts),
        "utterances_with_manual_annotation": n_annotated,
        "total_duration_hours": round(total_dur / 3600, 3),
    }
    (out_dir / "integrity.json").write_text(json.dumps(report, indent=2))
    print(">> DONE")
    print(json.dumps(report, indent=2))
    if n_annotated == 0:
        print("WARNING: no manual annotations found — A.3 phoneme cross-ref will "
              "be empty. Check that <SPK>/annotation/*.TextGrid exist.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
