# Wav2Lip — deep dive

Multilingual lip-sync benchmark for **Wav2Lip GAN** driven by OmniVoice audio across Hindi, English, and Marathi.

## What was benchmarked

| Item | Value |
|------|-------|
| Model | Wav2Lip GAN (`wav2lip_gan.pth`) |
| Hardware | node05 · RTX PRO 6000 Blackwell Server · MIG `1g.24gb` slice |
| GPU env | `~/envs/latentsync/bin/python` (torch 2.7.1+cu128, CUDA 12.8) |
| Audio | OmniVoice zero-shot, 24 kHz, same Marathi female reference |
| Languages | Hindi (हिन्दी), English, Marathi (मराठी) |
| Visuals | 2 still images + 1 video |
| Runs | 12 completed / 0 failed |
| Quality metrics | **None** (intentionally excluded per scope) |

### Inputs

- **visual_01.jpg** — 450×678 portrait (cropped 1 px from 451×679 so H.264/yuv420p gets even dimensions), JPEG
- **visual_02.jpg** — 554×554 square, JPEG
- **marathi-demo-2.mp4** — 15.02 s, 1280×720, 24 fps
- **Audio:** Hindi 14.20 s, English 14.22 s, Marathi 14.28 s (OmniVoice, Marathi female reference, shortened Marathi script so it fits the 15.02 s video)

### The 12 runs

| # | Tag | Mode | Visual | Language | Audio | Wall | RTF | VRAM | RSS |
|---|-----|------|--------|----------|-------|------|-----|------|-----|
| 0 | visual01-hi | static image | visual_01.jpg | Hindi | 14.20 s | 9.96 s | 0.701 | 4.15 GB | 2.31 GB |
| 1 | visual01-en | static image | visual_01.jpg | English | 14.22 s | 11.58 s | 0.814 | 4.15 GB | 2.09 GB |
| 2 | visual01-mr | static image | visual_01.jpg | Marathi | 14.28 s | 7.75 s | 0.543 | 4.15 GB | 2.09 GB |
| 3 | visual02-hi | static image | visual_02.jpg | Hindi | 14.20 s | 7.28 s | 0.513 | 4.15 GB | 2.23 GB |
| 4 | visual02-en | static image | visual_02.jpg | English | 14.22 s | 7.24 s | 0.509 | 4.15 GB | 2.24 GB |
| 5 | visual02-mr | static image | visual_02.jpg | Marathi | 14.28 s | 7.57 s | 0.53 | 4.15 GB | 2.24 GB |
| 6 | video-hi | video | marathi-demo-2.mp4 | Hindi | 14.20 s | 40.07 s | 2.822 | 18.9 GB | 3.90 GB |
| 7 | video-en | video | marathi-demo-2.mp4 | English | 14.22 s | 41.74 s | 2.936 | 18.9 GB | 3.89 GB |
| 8 | video-mr | video | marathi-demo-2.mp4 | Marathi | 14.28 s | 40.95 s | 2.868 | 18.9 GB | 3.89 GB |
| 9 | visual01-hi-5s | static image | visual_01.jpg | Hindi | 5.00 s | 6.25 s | 1.251 | 3.68 GB | 1.84 GB |
| 10 | visual01-hi-10s | static image | visual_01.jpg | Hindi | 10.00 s | 6.94 s | 0.694 | 4.15 GB | 2.08 GB |
| 11 | visual01-hi-20s | static image | visual_01.jpg | Hindi | 20.00 s | 7.51 s | 0.375 | 4.15 GB | 2.29 GB |

Output duration matches input audio (≤ 0.01 s difference) on all 12 runs.

### Command

```bash
# still image (static mode — face detected once)
~/envs/latentsync/bin/python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face .../visual_01.jpg \
  --audio .../hi.wav \
  --outfile .../visual01-hi.mp4 \
  --static True --fps 24

# video (fps taken from source)
~/envs/latentsync/bin/python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face .../marathi-demo-2.mp4 \
  --audio .../hi.wav \
  --outfile .../video-hi.mp4
```

Runs were driven through `hpc/run_wav2lip_multilingual.py`, which wraps the shared `hpc/benchmark_worker.py` and appends `--static`/`--fps` for Wav2Lip. Wall time includes Python startup, torch/CUDA init, checkpoint load, face detection, inference, and ffmpeg mux. RTF = wall ÷ audio duration (1.0 = realtime).

### Key findings

**Static image mode is ~5× faster than video mode** — average 8.01 s vs 40.92 s wall per run. Face detection runs once per image instead of on every frame.

**Image mode is dominated by fixed overhead.** A 5 s clip takes 6.25 s and a 20 s clip takes 7.51 s — model load and the single face detection cost more than the extra frames.

**Video mode runs at ~2.8× realtime** on this MIG slice (≈41 s wall for a 14.2 s clip) with 18.9 GB peak VRAM.

**Cold start matters.** The very first image run took **47.15 s**; once weights and the face detector were in page cache, the same run took **9.96 s**. The table uses warm runs.

### Outputs

All 12 MP4s are at `videos/wav2lip_multilingual/<tag>.mp4` and referenced relatively from `wav2lip_report.html`. JSON run records + logs live at `outputs/bench-20261119-lipsync-multilingual/wav2lip/`.

### Caveats

- Static-image outputs move **only the mouth**; head and background are frozen.
- Image resolution is native (450×678, 554×554) — no upscaling.
- The 20 s timing audio is a **looped extension** of the 14.2 s Hindi clip, used only for scaling; not a separate script.
- Slurm on node05 was invalid (`DOWN+INVALID_REG`), so jobs ran manually over SSH on one MIG `1g.24gb` slice. Timings are not directly comparable to full-GPU or H200 runs.
- Per scope, **no LSE-D/C, CSIM, or NIQE quality metrics** were run for these outputs.
