#!/usr/bin/env python
"""Download & prepare the Svarah (AI4Bharat) Indian-accented English set.

Svarah is GATED on Hugging Face. Before running this you must:
  1. Create a Hugging Face account.
  2. Visit https://huggingface.co/datasets/ai4bharat/Svarah and accept the
     access agreement.
  3. Create a token (https://huggingface.co/settings/tokens) and either pass
     it via --token or set the HF_TOKEN environment variable.

This script converts Svarah into the repo's unified internal format:
  data/processed/svarah/wav/<utt_id>.wav   (16 kHz mono PCM16)
  data/processed/svarah/manifest.jsonl     (one Utterance per line)

It also writes a small integrity report so a stranger can confirm the data
loaded correctly (counts, total duration, schema actually seen on HF).

No audio is committed to the repo; data/ is git-ignored.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from asr_l2.config import load_config, load_dotenv  # noqa: E402
from asr_l2.io.audio import TARGET_SR, save_wav  # noqa: E402
from asr_l2.io.manifest import Utterance, write_manifest  # noqa: E402

load_dotenv()  # pick up HF_TOKEN from repo-root .env if present

HF_ID = "ai4bharat/Svarah"

# Candidate column names; Svarah's exact schema is detected at runtime and the
# first match wins. The actual schema seen is printed + saved for transparency.
TEXT_KEYS = ["text", "transcript", "sentence", "transcription", "normalized_text"]
# Svarah's HF schema exposes the speaker's L1 as 'primary_language' (the key
# accent stratifier for analysis). Other candidates kept for portability.
ACCENT_KEYS = ["primary_language", "native_language", "l1", "mother_tongue",
               "language", "accent"]
SPEAKER_KEYS = ["speaker_id", "speaker", "spk_id", "spkid"]


def _first_present(keys: list[str], available: list[str]) -> str | None:
    for k in keys:
        if k in available:
            return k
    return None


def _decode_audio(audio_field) -> np.ndarray | None:
    """Decode an undecoded HF Audio field ({'bytes':..,'path':..}) to mono 16k.

    Uses soundfile (handles WAV/FLAC/OGG); falls back to librosa for anything
    soundfile can't read (e.g. MP3 via audioread). Returns float32 or None.
    """
    data, path = audio_field.get("bytes"), audio_field.get("path")
    try:
        if data is not None:
            arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        else:
            arr, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        try:
            arr, sr = librosa.load(path or io.BytesIO(data), sr=None, mono=True)
        except Exception:
            return None
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=sr,
                               target_sr=TARGET_SR)
    return arr.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config YAML")
    ap.add_argument("--token", default=None,
                    help="HF access token (else uses HF_TOKEN env var)")
    ap.add_argument("--hf-id", default=HF_ID, help="HuggingFace dataset id")
    ap.add_argument("--split", default="test", help="dataset split to load")
    ap.add_argument("--max-utterances", type=int, default=None,
                    help="cap number of utterances (default: all)")
    ap.add_argument("--out", default=None,
                    help="output dir (default: <data_root>/processed/svarah)")
    ap.add_argument("--text-only", default=None, metavar="TXT",
                    help="skip audio; dump ALL transcripts (one/line) to this file "
                         "for the Phase B domain corpus, then exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Svarah is gated. Provide --token or set HF_TOKEN.\n"
              "       Accept the agreement at "
              "https://huggingface.co/datasets/ai4bharat/Svarah first.",
              file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else \
        Path(cfg["paths"]["data_root"]) / "processed" / "svarah"
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. pip install datasets", file=sys.stderr)
        return 2

    print(f">> loading {args.hf_id} split={args.split} (this may take a while)")
    ds = load_dataset(args.hf_id, split=args.split, token=token)
    # Read rows through Arrow formatting so the Audio column comes back as the
    # raw {bytes, path} struct. datasets>=5 routes python/numpy audio decoding
    # through torchcodec (unreliable on Windows); Arrow format bypasses it and
    # we decode the bytes ourselves with soundfile in _decode_audio().
    ds_arrow = ds.with_format("arrow")

    available = list(ds.features.keys())
    print(">> dataset schema (columns):", available)
    text_key = _first_present(TEXT_KEYS, available)
    accent_key = _first_present(ACCENT_KEYS, available)
    speaker_key = _first_present(SPEAKER_KEYS, available)
    if text_key is None:
        print(f"ERROR: could not find a transcript column among {TEXT_KEYS}.\n"
              f"       Available columns: {available}\n"
              f"       Re-run with the correct column name patched in.",
              file=sys.stderr)
        return 3
    # The audio lives in whichever column is typed as datasets.Audio (in Svarah
    # this is the 'audio_filepath' column itself, not a column named 'audio').
    from datasets import Audio
    audio_key = next((k for k, v in ds.features.items() if isinstance(v, Audio)),
                     None)
    if audio_key is None:
        audio_key = _first_present(["audio", "audio_filepath"], available)
    if audio_key is None:
        print(f"ERROR: no Audio-typed column found. Columns: {available}",
              file=sys.stderr)
        return 3
    print(f">> mapped: text='{text_key}' accent='{accent_key}' "
          f"speaker='{speaker_key}' audio='{audio_key}'")

    # Text-only mode: dump every transcript (no audio decode) for the LM corpus.
    if args.text_only:
        texts = []
        for i in range(len(ds)):
            t = (ds_arrow[i].to_pylist()[0].get(text_key) or "").strip()
            if t:
                texts.append(t)
        out_txt = Path(args.text_only)
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        out_txt.write_text("\n".join(texts) + "\n", encoding="utf-8")
        print(f">> wrote {len(texts)} transcripts to {out_txt} "
              f"(exclude your test manifest when building the corpus!)")
        return 0

    # Choose which rows to extract. When capping, shuffle DETERMINISTICALLY so
    # the subset is representative across speakers/accents (Svarah may be
    # speaker-ordered), not just the first N. The seed makes it reproducible.
    import random
    total_rows = len(ds)
    if args.max_utterances is None or args.max_utterances >= total_rows:
        indices = list(range(total_rows))
    else:
        rng = random.Random(cfg.get("seed", 1234))
        indices = list(range(total_rows))
        rng.shuffle(indices)
        indices = sorted(indices[:args.max_utterances])
    print(f">> extracting {len(indices)} of {total_rows} utterances "
          f"(seed={cfg.get('seed', 1234)})")

    utts: list[Utterance] = []
    total_dur = 0.0
    skipped = 0

    for count, i in enumerate(indices):
        ex = ds_arrow[i].to_pylist()[0]
        text = (ex.get(text_key) or "").strip()
        if not text:
            skipped += 1
            continue
        arr = _decode_audio(ex[audio_key])
        if arr is None:
            skipped += 1
            continue
        utt_id = f"svarah_{i:05d}"
        wav_path = wav_dir / f"{utt_id}.wav"
        save_wav(wav_path, arr, TARGET_SR)
        dur = len(arr) / TARGET_SR
        total_dur += dur
        utts.append(Utterance(
            utt_id=utt_id, dataset="svarah", wav_path=str(wav_path),
            text=text, duration_s=round(dur, 3),
            speaker=str(ex.get(speaker_key)) if speaker_key else None,
            accent=str(ex.get(accent_key)) if accent_key else None,
            extra={k: ex[k] for k in available
                   if k not in {audio_key, text_key} and not _is_heavy(ex[k])},
        ))
        if (count + 1) % 25 == 0:
            print(f"   {count + 1}/{len(indices)} processed")

    manifest_path = out_dir / "manifest.jsonl"
    written = write_manifest(manifest_path, utts)

    report = {
        "dataset": "svarah", "hf_id": args.hf_id, "split": args.split,
        "utterances_written": written, "skipped_empty_text": skipped,
        "total_duration_hours": round(total_dur / 3600, 3),
        "schema_seen": available,
        "column_mapping": {"text": text_key, "accent": accent_key,
                           "speaker": speaker_key},
    }
    (out_dir / "integrity.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(">> DONE")
    print(json.dumps(report, indent=2))
    if written == 0:
        print("WARNING: 0 utterances written — check schema mapping.", file=sys.stderr)
    return 0


def _is_heavy(v) -> bool:
    """Skip large/binary fields when copying extras into the manifest."""
    if isinstance(v, (bytes, bytearray)):
        return True
    if isinstance(v, (list, dict)) and len(v) > 32:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
