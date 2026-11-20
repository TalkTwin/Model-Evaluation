# Lip-Sync benchmark reports — Marathi demo

Open **benchmark_report.html** for system metrics (wall time, RTF, VRAM) and video comparison.  
Open **quality_report.html** for quality metrics (LSE-D/C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE).  
Open **wav2lip_report.html** for the dedicated Wav2Lip multilingual timing report (Hindi · English · Marathi, 12 runs, timing only — no quality metrics).  
Open **tts_report.html** (Marathi TTS) and **tts_languages_report.html** (13-language TTS) for voice benchmarks.

The reports link directly to each other — no separate landing page needed. See `index.html` for report cards and `docs/` for deep dives.

## Full report index

| Report | What it contains |
|--------|------------------|
| `benchmark_report.html` | System metrics + video comparison (4 models × 7 durations) |
| `quality_report.html` | Quality metrics per run + grouped comparison tables |
| `tts_report.html` | Marathi TTS — OmniVoice + IndicF5, WER/CER, CSIM, MOS, RTF |
| `tts_languages_report.html` | 13-language TTS — per-language WER/CER, CSIM, DNSMOS, RTF |
| `wav2lip_report.html` | Wav2Lip multilingual timing — 12 runs, hi/en/mr, image + video |

## Documentation

| Doc | What it contains |
|-----|------------------|
| `docs/WAV2LIP.md` | Wav2Lip deep dive — inputs, 12-run table, commands, findings, caveats |
| `docs/PROJECT.md` | Project documentation — repo layout, how to run, cluster, deployment |

## Videos

Videos are self-contained under `videos/`:

- `videos/inputs/marathi-demo-2.mp4` — common 15.02 s source video
- `videos/recon/<model>/recon-{5,10,15}s.mp4` — reconstruction (video's own audio)
- `videos/cross/<model>/marathi-demo-2-{5,10,20}s.mp4` — dubbing
- `videos/long/<model>/marathi-demo-2.mp4` — 64.5 s dubbing
- `videos/wav2lip_multilingual/<tag>.mp4` — 12 Wav2Lip multilingual outputs

Models: Wav2Lip, HeyGem, MuseTalk, LatentSync.  
MOS is a manual 1–5 rating and is intentionally left blank.
