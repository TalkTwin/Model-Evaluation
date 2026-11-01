# Model-Evaluation

Reproducible benchmark for **LatentSync, MuseTalk, Wav2Lip, and HeyGem** — combining system + quality metrics, with per-duration HTML reports showing one common input video, all 4 model outputs, and stats.

## What was benchmarked

- **Common input:** `marathi-demo-2.mp4` (15.02 s, 1280×720, 24 fps) — same file used for every duration
- **Driving audio:** trimmed/changed MP3/WAV only (5 s, 10 s, 20 s, 64.5 s)
- **Two modes:**
  - *Reconstruction* — video's own audio (SSIM/PSNR/LPIPS/LMD valid)
  - *Cross-audio / dubbing* — new voice track (no-reference metrics + LSE + CSIM only)
- **Models:** Wav2Lip, HeyGem, MuseTalk, LatentSync

## Quality metrics

LSE-D, LSE-C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE/sharpness, manual MOS (1–5).

## Reports (`reports/`)

| Report | Description |
|--------|-------------|
| `benchmark_report.html` | System metrics (wall time, RTF, VRAM, RAM) + video comparison |
| `quality_report.html` | Quality metrics per run + grouped comparison tables |

The two reports link directly to each other. All video references are relative and self-contained under `reports/videos/`.

## Benchmark scripts (`hpc/`)

Working Slurm-based benchmark scripts:

- `hpc/benchmark.sbatch` — Slurm job script for benchmark runs
- `hpc/benchmark.py` / `hpc/benchmark_worker.py` — run engine, quality merge, failure guard
- `hpc/make_quality_report.py` — quality-report HTML generator (grouped tables, sticky model column)
- `hpc/make_quality_report.sh` — cluster wrapper to regenerate quality HTML
- `hpc/quality_metrics.py` — quality engine and NIQE implementation
- `hpc/repair_niqe.py` — NIQE-only repair over output folders
- `hpc/fetch_chunks.py` — low-MTU chunked download helper
- `hpc/build_reports_folder.py` — builds self-contained `reports/` folder
- `hpc/check_all_report_refs.py` — verifies every report href/src resolves
- `hpc/restyle_benchmark_report.py` — restyles system report tables

## GPU / cluster

- GPU work only via Slurm on `aicoeserver05` (RTX PRO 6000 Blackwell MIG 1g.24gb)
- Login node for light work only
- SSH key auth only

## Notes

- MOS is a manual 1–5 rating (left blank by design).
- GPU utilization unavailable on MIG.
- Large MP4/MP3/WAV assets are gitignored to keep the repo small.
