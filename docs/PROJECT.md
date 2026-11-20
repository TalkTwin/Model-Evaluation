# Project documentation — Model-Evaluation

Reproducible lip-sync and TTS benchmarking for **LatentSync, MuseTalk, Wav2Lip, HeyGem, OmniVoice, and IndicF5** — system metrics (wall time, RTF, VRAM, RAM), quality metrics (LSE-D/C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE), and per-language HTML reports.

## Benchmarks in this repo

| Benchmark | Report | Scope |
|-----------|--------|-------|
| System benchmark (4 models × 7 durations) | `benchmark_report.html` | wall time, RTF, VRAM, RAM, video comparison |
| Quality report | `quality_report.html` | LSE-D/C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE |
| Marathi TTS (OmniVoice + IndicF5) | `tts_report.html` | WER/CER, CSIM, predicted MOS, RTF, VRAM, audio samples |
| 13-language TTS | `tts_languages_report.html` | one sentence × 13 languages, 2 models, 26 samples |
| Wav2Lip multilingual (hi/en/mr) | `wav2lip_report.html` | 12 runs, 2 images + 1 video, OmniVoice audio, timing only |

## Repository layout

```
Model-Evaluation/
├── README.md                      # this file — overview + how to run
├── index.html                     # landing page with report cards
├── wav2lip_report.html            # dedicated Wav2Lip timing report (new)
├── benchmark_report.html          # system benchmark report
├── quality_report.html            # quality metrics report
├── tts_report.html                # Marathi TTS report
├── tts_languages_report.html      # 13-language TTS report
├── assets/lipsync_multilingual/   # visual_01.jpg, visual_02.jpg (source .jfif)
├── docs/
│   ├── WAV2LIP.md                 # Wav2Lip deep dive (this benchmark)
│   └── PROJECT.md                 # project documentation (this file)
├── hpc/                           # cluster runners + report generators
│   ├── benchmark_worker.py        # shared timing/measurement harness
│   ├── run_wav2lip_multilingual.py # Wav2Lip wrapper over benchmark_worker
│   ├── plan_wav2lip_multilingual.json # 12-run plan
│   ├── run_wav2lip_batch.sh       # batch runner for the 11 non-cold runs
│   ├── make_wav2lip_report.py     # builds wav2lip_report.html
│   ├── check_report.py            # sanity-checks report links/assets
│   ├── tts_texts_lipsync_multilingual.json # hi/en/mr OmniVoice scripts
│   └── … (quality, TTS Slurm scripts)
├── outputs/                       # per-benchmark run JSONs, logs, MP4s
│   └── bench-20261119-lipsync-multilingual/
├── videos/                        # web-referenced output MP4s
│   └── wav2lip_multilingual/
└── reports/                       # report index
    └── README.md
```

## How to run the Wav2Lip multilingual benchmark

1. Copy OmniVoice audio (`hi.wav`, `en.wav`, `mr.wav`, plus `hi_5s.wav`, `hi_10s.wav`, `hi_20s.wav`) into `inputs/lipsync_multilingual/audio/`.
2. Upload the plan and wrapper:
   ```bash
   scp hpc/plan_wav2lip_multilingual.json hpc/run_wav2lip_multilingual.py <node>:~/projects/
   ```
3. Run all 12 (or one at a time):
   ```bash
   cd ~/projects && export PATH=/usr/bin:$PATH
   CUDA_VISIBLE_DEVICES=MIG-60a375d5-8693-57ab-8a24-63f3be2d2782 \
     ~/envs/latentsync/bin/python run_wav2lip_multilingual.py \
     --plan plan_wav2lip_multilingual.json --all
   ```
   Or a single run: `--run-index 0`. First image run is a cold start (~47 s); rerun after warm cache for a fair number (~10 s).
4. Regenerate the HTML report:
   ```bash
   python hpc/make_wav2lip_report.py
   ```
5. Verify links/assets:
   ```bash
   python hpc/check_report.py
   ```

## Cluster

- GPU work via manual SSH on `aicoeserver05` → `node05` (Slurm GPU detection was `DOWN+INVALID_REG` at run time).
- MIG slice: `MIG-60a375d5-8693-57ab-8a24-63f3be2d2782` (1g.24gb), 2× RTX PRO 6000 Blackwell.
- Set `PATH=/usr/bin:$PATH`, `HF_HUB_OFFLINE=1`, `TORCH_COMPILE_DISABLE=1`.
- SSH key auth only; login node for light work.

## Deployment

Site: `https://talktwin.github.io/Model-Evaluation/`
Remote `main`: `f6a58a386a8338eb7aeceea928411091d74fe5a6`
Deploy via the GitHub API (not `git push`). **Ask before deploying** — one action per approval.

## Reports landing

Open `index.html` for the report cards, or go straight to any report HTML. `reports/README.md` lists every report and what it contains.
