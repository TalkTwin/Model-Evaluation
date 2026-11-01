# Lip-Sync benchmark reports — Marathi demo

Open **benchmark_report.html** for system metrics (wall time, RTF, VRAM) and video comparison.
Open **quality_report.html** for quality metrics (LSE-D/C, FID, FVD, SSIM, PSNR, LPIPS, CSIM, LMD, NIQE).

The two reports link directly to each other — no separate landing page needed.

Videos are self-contained under `videos/`:

- `videos/inputs/marathi-demo-2.mp4` — common 15.02 s source video
- `videos/recon/<model>/recon-{5,10,15}s.mp4` — reconstruction (video's own audio)
- `videos/cross/<model>/marathi-demo-2-{5,10,20}s.mp4` — dubbing
- `videos/long/<model>/marathi-demo-2.mp4` — 64.5 s dubbing

Models: Wav2Lip, HeyGem, MuseTalk, LatentSync.
MOS is a manual 1–5 rating and is intentionally left blank.
