# Model-Evaluation

Reproducible benchmark for **LatentSync, MuseTalk, Wav2Lip, and HeyGem** — combining system + quality metrics, with per-duration HTML reports showing one common input video, all 4 model outputs, and stats. Also includes OmniVoice/IndicF5 TTS benchmarks (Marathi + 13 languages) and a dedicated **Wav2Lip multilingual lip-sync** benchmark (Hindi · English · Marathi).

## What was benchmarked

- **Common input:** `marathi-demo-2.mp4` (15.02 s, 1280×720, 24 fps) — same file used for every duration
- **Driving audio:** trimmed/changed MP3/WAV only (5 s, 10 s, 20 s, 64.5 s)
- **Two modes:**
  - *Reconstruction* — video's own audio (SSIM/PSNR/LPIPS/LMD valid)
  - *Cross-audio / dubbing* — new voice track (no-reference metrics + LSE + CSIM only)
- **Models:** Wav2Lip, HeyGem, MuseTalk, LatentSync
- **TTS:** OmniVoice and IndicF5 — Marathi (4 prompts), and 13 Indic + English languages

### Wav2Lip multilingual (new)

Two portrait images + one video, lip-synced to OmniVoice audio in Hindi, English, and Marathi — 12 runs, timing only (no quality metrics).

| Run set | Visuals | Languages | Audio | Total |
|---------|---------|-----------|-------|-------|
| Multilingual | visual_01.jpg, visual_02.jpg, marathi-demo-2.mp4 | Hindi, English, Marathi | 14.20 / 14.22 / 14.28 s | 9 |
| Timing matrix | visual_01.jpg | Hindi | 5 s / 10 s / 20 s | 3 |

Findings: static image ~5× faster than video (8.01 s vs 40.92 s avg wall); image mode dominated by fixed overhead; video ~2.8× realtime on the MIG slice. See `wav2lip_report.html` and `docs/WAV2LIP.md`.

## Quality metrics

LSE-D, LSE-C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE/sharpness, manual MOS (1–5).

## Reports

| Report | Description |
|--------|-------------|
| `benchmark_report.html` | System metrics (wall time, RTF, VRAM, RAM) + video comparison |
| `quality_report.html` | Quality metrics per run + grouped comparison tables |
| `tts_report.html` | Marathi TTS — OmniVoice + IndicF5, WER/CER, CSIM, MOS, RTF |
| `tts_languages_report.html` | 13-language TTS — per-language WER/CER, CSIM, DNSMOS, RTF |
| `wav2lip_report.html` | Wav2Lip multilingual timing report (12 runs, hi/en/mr) |

The reports link directly to each other. All video references are relative and self-contained under `reports/videos/`.

## Documentation (`docs/`)

- `docs/WAV2LIP.md` — Wav2Lip deep dive: inputs, 12-run table, commands, findings, caveats
- `docs/PROJECT.md` — project documentation: repo layout, how to run, cluster, deployment

## Benchmark scripts (`hpc/`)

Working Slurm-based benchmark scripts:

- `hpc/benchmark.sbatch` — Slurm job script for benchmark runs
- `hpc/benchmark.py` / `hpc/benchmark_worker.py` — run engine, quality merge, failure guard
- `hpc/run_wav2lip_multilingual.py` — Wav2Lip wrapper over `benchmark_worker.py` (`--static`/`--fps`)
- `hpc/run_wav2lip_batch.sh` — batch runner for the 11 non-cold Wav2Lip runs
- `hpc/make_wav2lip_report.py` — builds `wav2lip_report.html`
- `hpc/make_quality_report.py` — quality-report HTML generator (grouped tables, sticky model column)
- `hpc/make_quality_report.sh` — cluster wrapper to regenerate quality HTML
- `hpc/quality_metrics.py` — quality engine and NIQE implementation
- `hpc/repair_niqe.py` — NIQE-only repair over output folders
- `hpc/fetch_chunks.py` — low-MTU chunked download helper
- `hpc/build_reports_folder.py` — builds self-contained `reports/` folder
- `hpc/check_all_report_refs.py` — verifies every report href/src resolves
- `hpc/restyle_benchmark_report.py` — restyles system report tables
- `hpc/tts_texts_lipsync_multilingual.json` — Hindi/English/Marathi OmniVoice scripts

## GPU / cluster

- GPU work only via Slurm on `aicoeserver05` (RTX PRO 6000 Blackwell MIG 1g.24gb)
- Login node for light work only
- SSH key auth only

## Notes

- MOS is a manual 1–5 rating (left blank by design).
- GPU utilization unavailable on MIG.
- Large MP4/MP3/WAV assets are gitignored to keep the repo small.
- Slurm on node05 was `DOWN+INVALID_REG` at run time; Wav2Lip jobs ran manually over SSH on one MIG slice.
