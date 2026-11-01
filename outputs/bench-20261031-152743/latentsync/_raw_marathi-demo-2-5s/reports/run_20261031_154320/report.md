# LatentSync benchmark report

- Generated: 2026-10-31T15:43:20
- Host: `aicoeserver05`  |  Slurm job: 1568
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb
- Settings: steps=20, guidance=1.5, deepcache=True
- LatentSync commit: a229c39
- Jobs: 1/1 succeeded

## Summary

| Input | Status | Video len | Audio len | Output len | Inference | Total wall | RTF | Peak VRAM (device) | PyTorch peak alloc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| marathi-demo-2-5s | success | 15.0s | 5.0s | 5.1s | 140.2s | 143.0s | 27.6x | 16.73 GiB | 12.90 GiB |

RTF = compute seconds spent per second of output video (lower is faster).

## marathi-demo-2-5s  (success)

- Input video: `/home/aryannavale/projects/inputs/marathi-demo-2-5s/marathi-demo-2.mp4`: 15.02s, 1280x720 @ 24.0 fps, 360 frames, 13.34 MB
- Input audio: `/home/aryannavale/projects/inputs/marathi-demo-2-5s/marathi-demo-voice-2-5s.mp3`: 5.00s, 44100 Hz, 0.06 MB
- Output video: `/home/aryannavale/projects/outputs/bench-20261031-152743/latentsync/_raw_marathi-demo-2-5s/marathi-demo-2-latentsync.mp4`: 5.08s, 1280x720 @ 25.0 fps, 127 frames, 1.0 MB
- Timing: boot 1.1s, inference (model load + diffusion) 140.2s, total wall 143.0s
- Throughput: RTF 27.59x, 0.91 output frames per compute-second
- Length check: audio - output = -0.08s, video - output = 9.94s
- CPU RAM peak: 10826 MB

**GPU memory**

| Metric | MiB | Meaning |
|---|---:|---|
| GPU slice total | 24192 | memory of the MIG slice you were given |
| PyTorch peak allocated | 13206 | tensors actually in use (weights, latents, activations) |
| PyTorch peak reserved | 16584 | PyTorch caching allocator pool (allocated + cache) |
| Device used, start | 225 | before inference started (CUDA context) |
| Device used, peak | 17131 | everything on the slice, sampled every 0.5 s |
| Non-PyTorch overhead | 547 | device peak minus reserved: CUDA context, cuDNN workspaces, ONNX Runtime (InsightFace) |
| Headroom | 7061 | free memory at the peak (70.8% of slice used) |

![VRAM timeline](marathi-demo-2-5s__marathi-demo-2.vram.png)
