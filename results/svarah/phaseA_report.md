# Phase A report — svarah

> **PRELIMINARY.** Error-type proportions below are automatic substitution/
> deletion/insertion counts only. The accent vs disfluency vs VAD-cutoff vs
> hallucination breakdown requires the manual annotation pass on the error CSV.

## Hardware / run config
- Model: `small.en`  | compute: `int8` | device: `auto`
- Chunk sizes (ms): [640, 1280]  | left context: 2000 ms
- Utterances/condition cap: 150

## Per-condition WER / CER / latency
(`mean_wer`/`mean_cer` are means of per-utterance scores; `mean_rtf` is mean
real-time factor; `alat_ms` is the algorithmic latency = chunk size.)

| condition | n | mean_wer | mean_cer | mean_rtf | mean_proc_s | alat_ms |
| --- | --- | --- | --- | --- | --- | --- |
| offline | 150 | 0.132 | 0.081 | 0.045 | 0.212 | nan |
| stream_320ms | 150 | 0.705 | 0.451 | 0.657 | 4.011 | 320.000 |
| stream_640ms | 150 | 0.477 | 0.291 | 0.312 | 1.920 | 640.000 |
| stream_1280ms | 150 | 0.364 | 0.211 | 0.169 | 1.044 | 1280.000 |

## Latency vs WER trade-off
![latency vs WER](phaseA_latency_wer_tradeoff.png)

## Error-type mix (PRELIMINARY, auto S/D/I)
![error mix](phaseA_error_mix.png)

Totals across all conditions:
- **substitutions**: 1420 (56.2%)
- **deletions**: 178 (7.0%)
- **insertions**: 927 (36.7%)

Error CSV: `phaseA_error_annotation_offline.csv` — 160 rows, 0 manually annotated with a why-category (0% done).

## What is automated vs. manual here
- **Automated:** all numbers and plots above (S/D/I counts, WER/CER, latency).
- **Manual (yours):** open the error CSV and fill `why_category` per error.
  Phase B's category-level deltas only become meaningful after that pass.
