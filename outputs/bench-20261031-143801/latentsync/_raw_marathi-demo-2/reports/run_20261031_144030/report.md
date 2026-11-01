# LatentSync benchmark report

- Generated: 2026-10-31T14:40:30
- Host: `aicoeserver05`  |  Slurm job: 1567
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
- Settings: steps=20, guidance=1.5, deepcache=True
- LatentSync commit: a229c39
- Jobs: 1/1 succeeded

## Summary

| Input | Status | Video len | Audio len | Output len | Inference | Total wall | RTF | Peak VRAM (device) | PyTorch peak alloc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| marathi-demo-2 | success | 15.0s | 64.5s | 64.6s | 1437.7s | 1443.2s | 22.3x | 19.73 GiB | 15.13 GiB |

RTF = compute seconds spent per second of output video (lower is faster).

## marathi-demo-2  (success)

- Input video: `/home/aryannavale/projects/inputs/marathi-demo-2/marathi-demo-2.mp4`: 15.02s, 1280x720 @ 24.0 fps, 360 frames, 13.34 MB
- Input audio: `/home/aryannavale/projects/inputs/marathi-demo-2/marathi-demo-voice-2.mp3`: 64.52s, 44100 Hz, 1.0 MB
- Output video: `/home/aryannavale/projects/outputs/bench-20261031-143801/latentsync/_raw_marathi-demo-2/marathi-demo-2-latentsync.mp4`: 64.60s, 1280x720 @ 25.0 fps, 1615 frames, 12.49 MB
- Timing: boot 3.6s, inference (model load + diffusion) 1437.7s, total wall 1443.2s
- Throughput: RTF 22.26x, 1.12 output frames per compute-second
- Length check: audio - output = -0.08s, video - output = -49.58s
- CPU RAM peak: 18785 MB

**GPU memory**

| Metric | MiB | Meaning |
|---|---:|---|
| GPU slice total | 24192 | memory of the MIG slice you were given |
| PyTorch peak allocated | 15489 | tensors actually in use (weights, latents, activations) |
| PyTorch peak reserved | 19656 | PyTorch caching allocator pool (allocated + cache) |
| Device used, start | 225 | before inference started (CUDA context) |
| Device used, peak | 20203 | everything on the slice, sampled every 0.5 s |
| Non-PyTorch overhead | 547 | device peak minus reserved: CUDA context, cuDNN workspaces, ONNX Runtime (InsightFace) |
| Headroom | 3989 | free memory at the peak (83.5% of slice used) |

![VRAM timeline](marathi-demo-2__marathi-demo-2.vram.png)
