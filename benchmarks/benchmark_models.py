#!/usr/bin/env python3
"""
benchmark_models.py - unified batch runner + benchmark reporter for lip-sync models.

Models
  latentsync : ByteDance LatentSync        (module scripts.inference)
  musetalk   : MuseTalk v1.5               (module scripts.inference, YAML config)
  wav2lip    : Wav2Lip GAN                 (module inference)
  heygem     : HeyGem RTX50                (module run, fixed work id 1004)

Per input it measures
  - input video : duration, resolution, fps, frames, size
  - input audio : duration, sample rate, channels, size
  - output video: duration, resolution, fps, frames, size
  - boot time, inference time (model load + diffusion), total wall time
  - real-time factor (compute seconds per output second)
  - VRAM: PyTorch peak allocated/reserved, device peak (everything on the MIG
    slice), non-PyTorch overhead, headroom, 0.5 s timeline + PNG plot
  - peak CPU RAM of the inference process
  - warnings such as audio/video length mismatch

Reports (written after every single run, so a crash still leaves data)
  <outputs>/reports/<model>_run_<stamp>/report.json
                                         report.md
                                         results.csv
                                         <run>.log
                                         <run>.worker.json
                                         <run>.vram.png

Usage
  # GPU node, batch mode
  ~/envs/latentsync/bin/python benchmark_models.py --model latentsync --select all --yes --offline
  ~/envs/musetalk/bin/python   benchmark_models.py --model musetalk   --select 1,2 --yes
  ~/envs/latentsync/bin/python benchmark_models.py --model wav2lip    --select all --yes
  ~/envs/hygem-rtx50/bin/python benchmark_models.py --model heygem    --select all --yes

  # list inputs only (works on the login node, no GPU needed)
  python benchmark_models.py --list

  # merge every report under ~/projects/outputs/reports into one comparison
  python benchmark_models.py --compare
"""

import argparse
import csv
import json
import math
import os
import platform
import re
import runpy
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

HOME = Path.home()
DEFAULT_INPUTS = HOME / "projects" / "inputs"
DEFAULT_OUTPUTS = HOME / "projects" / "outputs"
DEFAULT_REPORTS = DEFAULT_OUTPUTS / "reports"

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"}
MB = 1024 ** 2

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "latentsync": {
        "title": "LatentSync",
        "repo": HOME / "projects" / "LatentSync",
        "python": HOME / "envs" / "latentsync" / "bin" / "python",
        "module": "scripts.inference",
        "required": [
            ("scripts/inference.py", 1),
            ("configs/unet/stage2_512.yaml", 1),
            ("checkpoints/latentsync_unet.pt", 1e9),
            ("checkpoints/whisper/tiny.pt", 1),
            ("checkpoints/auxiliary/models/buffalo_l", "dir"),
        ],
        "env": {},
    },
    "musetalk": {
        "title": "MuseTalk v1.5",
        "repo": HOME / "projects" / "MuseTalk",
        "python": HOME / "envs" / "musetalk" / "bin" / "python",
        "module": "scripts.inference",
        "required": [
            ("scripts/inference.py", 1),
            ("models/musetalkV15/unet.pth", 1e9),
            ("models/musetalkV15/musetalk.json", 1),
            ("models/whisper", "dir"),
            ("models/dwpose", "dir"),
        ],
        "env": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    },
    "wav2lip": {
        "title": "Wav2Lip GAN",
        "repo": HOME / "projects" / "Wav2Lip",
        "python": HOME / "envs" / "latentsync" / "bin" / "python",
        "module": "inference",
        "required": [
            ("inference.py", 1),
            ("checkpoints/wav2lip_gan.pth", 1e6),
            ("face_detection/detection/sfd/s3fd.pth", 1e6),
        ],
        "env": {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"},
    },
    "heygem": {
        "title": "HeyGem RTX50",
        "repo": HOME / "projects" / "hygem-rtx50",
        "python": HOME / "envs" / "hygem-rtx50" / "bin" / "python",
        "module": "run",
        "required": [
            ("run.py", 1),
            ("config/config.ini", 1),
            ("env.sh", 1),
        ],
        "env": {},
    },
}


# ---------------------------------------------------------------------------
# Worker mode: runs inside the child process and samples VRAM in-process
# ---------------------------------------------------------------------------
def worker_main(metrics_path, model, inf_args):
    import torch

    spec = MODELS[model]
    boot_epoch = time.time()
    cuda_ok = torch.cuda.is_available()
    samples, util = [], []
    util_ok = [True]
    stop = threading.Event()
    t_ref = time.perf_counter()

    def take():
        if not cuda_ok:
            return
        try:
            free, total = torch.cuda.mem_get_info()
            samples.append([
                round(time.perf_counter() - t_ref, 2),
                round((total - free) / MB, 1),
                round(torch.cuda.memory_allocated() / MB, 1),
                round(torch.cuda.memory_reserved() / MB, 1),
            ])
        except Exception:
            pass
        if util_ok[0]:
            try:
                util.append(torch.cuda.utilization())
            except Exception:
                util_ok[0] = False

    def loop():
        while not stop.wait(0.5):
            take()

    th = None
    if cuda_ok:
        torch.cuda.init()
        take()
        th = threading.Thread(target=loop, daemon=True)
        th.start()

    exit_code, err = 0, None
    t_run = time.perf_counter()
    try:
        os.chdir(spec["repo"])
        sys.path.insert(0, str(spec["repo"]))
        sys.argv = [spec["module"]] + list(inf_args)
        runpy.run_module(spec["module"], run_name="__main__", alter_sys=True)
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:
        exit_code = 1
        err = traceback.format_exc()
        traceback.print_exc()
    finally:
        run_s = time.perf_counter() - t_run
        stop.set()
        if th:
            th.join(timeout=3)
        try:
            if cuda_ok:
                torch.cuda.synchronize()
        except Exception:
            pass
        take()

        gpu = {"cuda_available": cuda_ok, "torch_version": torch.__version__,
               "cuda_version": torch.version.cuda}
        if cuda_ok:
            try:
                props = torch.cuda.get_device_properties(0)
                gpu.update({
                    "name": torch.cuda.get_device_name(0),
                    "total_mb": round(props.total_memory / MB, 1),
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / MB, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / MB, 1),
                    "device_used_start_mb": samples[0][1] if samples else None,
                    "device_used_peak_mb": max(s[1] for s in samples) if samples else None,
                    "samples": len(samples),
                    "util_avg_pct": round(sum(util) / len(util), 1) if util else None,
                    "util_peak_pct": max(util) if util else None,
                    "timeline": samples,
                })
            except Exception as e:
                gpu["metrics_error"] = str(e)
        try:
            import resource
            cpu_ram_peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        except Exception:
            cpu_ram_peak = None

        with open(metrics_path, "w") as f:
            json.dump({"boot_epoch": boot_epoch, "run_s": round(run_s, 3),
                       "exit_code": exit_code, "error": err,
                       "cpu_ram_peak_mb": cpu_ram_peak, "gpu": gpu}, f)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Media probing
# ---------------------------------------------------------------------------
def _fps(s):
    try:
        a, b = str(s).split("/")
        return round(float(a) / float(b), 3) if float(b) else None
    except Exception:
        try:
            return round(float(s), 3)
        except Exception:
            return None


def probe_media(path):
    path = Path(path)
    info = {"size_mb": round(path.stat().st_size / MB, 2) if path.exists() else None}
    if shutil.which("ffprobe") and path.exists():
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=60)
            data = json.loads(r.stdout)
            fmt = data.get("format", {})
            info["duration_s"] = round(float(fmt["duration"]), 3) if fmt.get("duration") else None
            for s in data.get("streams", []):
                if s.get("codec_type") == "video" and "width" not in info:
                    nb = s.get("nb_frames")
                    info.update(video_codec=s.get("codec_name"), width=s.get("width"),
                                height=s.get("height"), fps=_fps(s.get("avg_frame_rate", "")),
                                frames=int(nb) if nb and str(nb).isdigit() else None)
                elif s.get("codec_type") == "audio" and "audio_codec" not in info:
                    sr = s.get("sample_rate")
                    info.update(audio_codec=s.get("codec_name"),
                                sample_rate=int(sr) if sr else None,
                                channels=s.get("channels"))
        except Exception as e:
            info["probe_error"] = str(e)
    if info.get("duration_s") is None and path.exists():
        try:
            if path.suffix.lower() in VIDEO_EXT:
                import cv2
                cap = cv2.VideoCapture(str(path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                info.update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            fps=round(fps, 3), frames=int(n),
                            duration_s=round(n / fps, 3) if fps else None)
                cap.release()
            else:
                import librosa
                info["duration_s"] = round(float(librosa.get_duration(path=str(path))), 3)
        except Exception as e:
            info["probe_fallback_error"] = str(e)
    if info.get("frames") is None and info.get("fps") and info.get("duration_s") and "width" in info:
        info["frames"] = int(round(info["fps"] * info["duration_s"]))
    return info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def f2(x, nd=2, unit=""):
    return "-" if x is None else f"{x:.{nd}f}{unit}"


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return 1, str(e)


def discover(root):
    root = Path(root)
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if any(p.suffix.lower() in VIDEO_EXT | AUDIO_EXT for p in root.iterdir() if p.is_file()):
        folders.insert(0, root)
    items = []
    for fo in folders:
        files = [p for p in fo.iterdir() if p.is_file()]
        vids = sorted(p for p in files if p.suffix.lower() in VIDEO_EXT)
        auds = sorted(p for p in files if p.suffix.lower() in AUDIO_EXT)
        if vids and auds:
            items.append({"name": fo.name, "folder": fo, "video": vids[0], "audio": auds[0],
                          "extra_video": len(vids) - 1, "extra_audio": len(auds) - 1})
    return items


def parse_selection(text, n):
    text = text.strip().lower()
    if text in ("", "a", "all", "*"):
        return list(range(n))
    picked = set()
    for part in re.split(r"[,\s]+", text):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            picked.update(range(min(a, b), max(a, b) + 1))
        elif part.isdigit():
            picked.add(int(part))
        else:
            raise ValueError(f"cannot understand '{part}'")
    bad = [i for i in picked if i < 1 or i > n]
    if bad:
        raise ValueError(f"out of range: {sorted(bad)} (valid 1..{n})")
    return sorted(i - 1 for i in picked)


def print_listing(items):
    print(f"\n{'#':>3}  {'Folder':<22}{'Video':<28}{'Len':>8} {'Res':>10}  {'Audio':<28}{'Len':>8}")
    print("-" * 112)
    for i, it in enumerate(items, 1):
        vp, ap = it["vp"], it["ap"]
        res = f"{vp.get('width')}x{vp.get('height')}" if vp.get("width") else "-"
        print(f"{i:>3}  {it['name'][:21]:<22}{it['video'].name[:27]:<28}"
              f"{f2(vp.get('duration_s'), 1, 's'):>8} {res:>10}  "
              f"{it['audio'].name[:27]:<28}{f2(ap.get('duration_s'), 1, 's'):>8}")
        if it["extra_video"] or it["extra_audio"]:
            print("     note: folder has extra media files; using first video/audio only")
    print()


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def preflight(model, repo, spec):
    checks = []

    def add(ok, msg, fatal=True):
        checks.append((ok, fatal, msg))

    add(Path(repo).is_dir(), f"repo {repo}")
    add(Path(spec["python"]).exists(), f"python {spec['python']}")
    for rel, want in spec["required"]:
        p = Path(repo, rel)
        if want == "dir":
            add(p.is_dir(), f"{rel} (directory)")
        else:
            ok = p.exists() and p.stat().st_size >= want
            size = f" ({p.stat().st_size / 1e9:.1f} GB)" if p.exists() else ""
            add(ok, f"{rel}{size}")
    add(bool(shutil.which("ffmpeg")), "ffmpeg on PATH (needed to mux the output)")
    add(bool(shutil.which("ffprobe")), "ffprobe on PATH (nicer media info)", fatal=False)
    rc, out = sh(["nvidia-smi", "-L"])
    add(rc == 0 and "GPU" in out, f"GPU visible on {socket.gethostname()}")

    print(f"Preflight ({spec['title']})")
    fatal_fail = False
    for ok, fatal, msg in checks:
        print(f"  [{'OK' if ok else ('FAIL' if fatal else 'WARN')}] {msg}")
        fatal_fail |= (not ok and fatal)
    print()
    return not fatal_fail


# ---------------------------------------------------------------------------
# Per-model inference arguments
# ---------------------------------------------------------------------------
def build_inf_args(model, job, ctx, args, tag):
    if model == "latentsync":
        a = [
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--inference_steps", str(args.steps),
            "--guidance_scale", str(args.guidance),
            "--video_path", str(job["video"]),
            "--audio_path", str(job["audio"]),
            "--video_out_path", str(ctx["out"]),
        ]
        if args.deepcache:
            a.append("--enable_deepcache")
        return a

    if model == "musetalk":
        cfg_dir = ctx["run_dir"] / "musetalk_configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{tag}.json"   # JSON is valid YAML for OmegaConf
        cfg_path.write_text(json.dumps({
            "task_0": {
                "video_path": str(job["video"]),
                "audio_path": str(job["audio"]),
                "bbox_shift": 0,
            }
        }, indent=2))
        ctx["result_dir"] = ctx["run_dir"] / "musetalk_out" / tag
        ctx["result_dir"].mkdir(parents=True, exist_ok=True)
        return [
            "--inference_config", str(cfg_path),
            "--result_dir", str(ctx["result_dir"]),
            "--unet_model_path", "models/musetalkV15/unet.pth",
            "--unet_config", "models/musetalkV15/musetalk.json",
            "--whisper_dir", "models/whisper",
            "--version", "v15",
            "--use_float16",
            "--batch_size", str(args.batch_size),
        ]

    if model == "wav2lip":
        return [
            "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
            "--face", str(job["video"]),
            "--audio", str(job["audio"]),
            "--outfile", str(ctx["out"]),
        ]

    if model == "heygem":
        return [
            "--audio_path", str(job["audio"]),
            "--video_path", str(job["video"]),
        ]

    raise ValueError(model)


def resolve_raw_output(model, ctx, repo):
    """Return the file the model itself produced (before copying to outputs/)."""
    if model in ("latentsync", "wav2lip"):
        return Path(ctx["out"])
    if model == "heygem":
        return Path(repo) / "1004-r.mp4"
    if model == "musetalk":
        rd = Path(ctx["result_dir"])
        if not rd.exists():
            return None
        mp4s = [p for p in rd.rglob("*.mp4") if "_concat" not in p.name]
        if not mp4s:
            mp4s = list(rd.rglob("*.mp4"))
        if not mp4s:
            return None
        return max(mp4s, key=lambda p: p.stat().st_mtime)
    return None


def resolve_raw_output_after_failure(model, ctx, repo):
    """Resolve whatever exists, so a failed run still reports any partial output."""
    return resolve_raw_output(model, ctx, repo)


# ---------------------------------------------------------------------------
# Running one job
# ---------------------------------------------------------------------------
def run_job(job, out_path, args, run_dir, repo, spec):
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{job['name']}__{job['video'].stem}")
    metrics_path = run_dir / f"{tag}.worker.json"
    log_path = run_dir / f"{tag}.log"

    ctx = {"out": out_path, "run_dir": run_dir}
    inf_args = build_inf_args(args.model, job, ctx, args, tag)

    # remove stale artifacts so a crash cannot look like a success
    if out_path.exists():
        out_path.unlink()
    if args.model == "heygem":
        stale = Path(repo) / "1004-r.mp4"
        if stale.exists():
            stale.unlink()

    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--worker", str(metrics_path), args.model, "--"] + inf_args

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.update(spec.get("env", {}))
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"

    status = "failed"
    spawn_epoch = time.time()
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, bufsize=0)
    try:
        with open(log_path, "wb") as lf:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                lf.write(chunk)
        rc = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        rc, status = -2, "interrupted"
    wall = time.perf_counter() - t0

    wm = {}
    if metrics_path.exists():
        try:
            wm = json.loads(metrics_path.read_text())
        except Exception:
            pass

    raw = resolve_raw_output(args.model, ctx, repo)
    raw_exists = bool(raw and raw.exists() and raw.stat().st_size > 0)
    if raw_exists and Path(raw).resolve() != Path(out_path).resolve():
        shutil.copy2(raw, out_path)
    out_exists = out_path.exists() and out_path.stat().st_size > 0

    if status != "interrupted":
        status = "success" if (rc == 0 and out_exists) else "failed"

    vp, ap = job["vp"], job["ap"]
    op = probe_media(out_path) if out_exists else {}
    gpu = wm.get("gpu", {})
    run_s = wm.get("run_s")
    out_dur = op.get("duration_s")
    in_dur = vp.get("duration_s")
    total_mb = gpu.get("total_mb")
    dev_peak = gpu.get("device_used_peak_mb")
    res_peak = gpu.get("peak_reserved_mb")

    warnings = []
    if ap.get("duration_s") and in_dur and abs(ap["duration_s"] - in_dur) > 1.0:
        warnings.append(
            f"input audio ({ap['duration_s']:.1f}s) and video ({in_dur:.1f}s) differ by "
            f"{abs(ap['duration_s'] - in_dur):.1f}s; the model may truncate or loop")
    if args.model == "heygem":
        warnings.append("HeyGem run.py includes an internal 10 s sleep before inference")

    result = {
        "model": args.model,
        "name": job["name"], "status": status, "return_code": rc,
        "error": wm.get("error"),
        "warnings": warnings,
        "input_video": {"path": str(job["video"]), **vp},
        "input_audio": {"path": str(job["audio"]), **ap},
        "output_video": {"path": str(out_path), "exists": out_exists,
                         "raw_output": str(raw) if raw else None, **op},
        "timing": {
            "boot_s": round(wm["boot_epoch"] - spawn_epoch, 2) if wm.get("boot_epoch") else None,
            "inference_s": run_s,
            "total_wall_s": round(wall, 2),
        },
        "derived": {
            "rtf_compute_s_per_output_s": round(run_s / out_dur, 2) if run_s and out_dur else None,
            "output_frames_per_compute_s": round(op["frames"] / run_s, 2) if run_s and op.get("frames") else None,
            "audio_minus_output_len_s": round(ap["duration_s"] - out_dur, 3)
            if ap.get("duration_s") and out_dur else None,
            "video_minus_output_len_s": round(in_dur - out_dur, 3) if in_dur and out_dur else None,
        },
        "vram": {
            "gpu_name": gpu.get("name"), "gpu_total_mb": total_mb,
            "pytorch_peak_allocated_mb": gpu.get("peak_allocated_mb"),
            "pytorch_peak_reserved_mb": res_peak,
            "device_used_start_mb": gpu.get("device_used_start_mb"),
            "device_used_peak_mb": dev_peak,
            "non_pytorch_overhead_mb": round(dev_peak - res_peak, 1) if dev_peak and res_peak else None,
            "headroom_mb": round(total_mb - dev_peak, 1) if total_mb and dev_peak else None,
            "device_peak_pct_of_total": round(100 * dev_peak / total_mb, 1) if total_mb and dev_peak else None,
            "gpu_util_avg_pct": gpu.get("util_avg_pct"), "gpu_util_peak_pct": gpu.get("util_peak_pct"),
            "samples": gpu.get("samples"),
            "torch_version": gpu.get("torch_version"), "cuda_version": gpu.get("cuda_version"),
        },
        "cpu_ram_peak_mb": wm.get("cpu_ram_peak_mb"),
        "log_file": str(log_path),
    }
    timeline = gpu.get("timeline") or []
    stride = max(1, math.ceil(len(timeline) / 400))
    result["vram_timeline_columns"] = ["t_s", "device_used_mb", "pytorch_allocated_mb", "pytorch_reserved_mb"]
    result["vram_timeline"] = timeline[::stride]

    if timeline:
        png = run_dir / f"{tag}.vram.png"
        if make_vram_plot(timeline, total_mb, png, f"{spec['title']} / {job['video'].name}"):
            result["vram_plot"] = str(png)
    return result


def make_vram_plot(timeline, total_mb, out_png, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    t = [s[0] for s in timeline]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(t, [s[1] / 1024 for s in timeline], label="Device used (incl. CUDA ctx + non-PyTorch)")
    ax.plot(t, [s[3] / 1024 for s in timeline], label="PyTorch reserved")
    ax.plot(t, [s[2] / 1024 for s in timeline], label="PyTorch allocated")
    if total_mb:
        ax.axhline(total_mb / 1024, ls="--", lw=1, color="gray", label=f"GPU slice total ({total_mb / 1024:.1f} GiB)")
    ax.set_xlabel("seconds since worker start")
    ax.set_ylabel("GiB")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def system_info(repo):
    _, drv = sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    _, gl = sh(["nvidia-smi", "-L"])
    _, commit = sh(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"])
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE"),
        "gres": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version.split()[0], "python_exe": sys.executable,
        "platform": platform.platform(), "nvidia_driver": drv, "nvidia_smi_L": gl,
        "model_commit": commit,
    }


def settings_dict(args):
    if args.model == "latentsync":
        return {"steps": args.steps, "guidance": args.guidance, "deepcache": args.deepcache}
    if args.model == "musetalk":
        return {"version": "v15", "fp16": True, "batch_size": args.batch_size}
    if args.model == "wav2lip":
        return {"checkpoint": "checkpoints/wav2lip_gan.pth"}
    if args.model == "heygem":
        return {"work_id": "1004 (fixed by run.py)"}
    return {}


def write_reports(run_dir, meta, runs):
    ok = [r for r in runs if r["status"] == "success"]
    rtfs = [r["derived"]["rtf_compute_s_per_output_s"] for r in ok
            if r["derived"]["rtf_compute_s_per_output_s"]]
    summary = {
        "jobs_total": len(runs), "jobs_success": len(ok), "jobs_failed": len(runs) - len(ok),
        "sum_total_wall_s": round(sum(r["timing"]["total_wall_s"] for r in runs), 2),
        "avg_rtf": round(sum(rtfs) / len(rtfs), 2) if rtfs else None,
        "max_device_peak_mb": max((r["vram"]["device_used_peak_mb"] or 0) for r in runs) if runs else None,
    }
    report = {**meta, "summary": summary, "runs": runs}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    cols = ["model", "name", "status", "video_len_s", "audio_len_s", "output_len_s", "resolution",
            "inference_s", "total_wall_s", "rtf", "pytorch_peak_alloc_mb", "pytorch_peak_reserved_mb",
            "device_peak_mb", "gpu_total_mb", "cpu_ram_peak_mb", "output_path"]
    with open(run_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in runs:
            ov, v = r["output_video"], r["vram"]
            w.writerow([r["model"], r["name"], r["status"], r["input_video"].get("duration_s"),
                        r["input_audio"].get("duration_s"), ov.get("duration_s"),
                        f"{ov.get('width')}x{ov.get('height')}" if ov.get("width") else "",
                        r["timing"]["inference_s"], r["timing"]["total_wall_s"],
                        r["derived"]["rtf_compute_s_per_output_s"], v["pytorch_peak_allocated_mb"],
                        v["pytorch_peak_reserved_mb"], v["device_used_peak_mb"], v["gpu_total_mb"],
                        r["cpu_ram_peak_mb"], ov.get("path")])

    title = meta["model_title"]
    L = [f"# {title} benchmark report", "",
         f"- Generated: {meta['generated_at']}",
         f"- Model: **{title}** (`{meta['model']}`)",
         f"- Host: `{meta['system']['hostname']}`  |  Slurm job: {meta['system']['slurm_job_id'] or 'interactive/none'}",
         f"- GPU: {next((r['vram']['gpu_name'] for r in runs if r['vram']['gpu_name']), '-')}",
         f"- Settings: `{json.dumps(meta['settings'])}`",
         f"- Model commit: {meta['system']['model_commit']}",
         f"- Jobs: {summary['jobs_success']}/{summary['jobs_total']} succeeded", "",
         "## Summary", "",
         "| Input | Status | Video len | Audio len | Output len | Inference | Total wall | RTF | Peak VRAM (device) | PyTorch peak alloc |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        v = r["vram"]
        L.append(f"| {r['name']} | {r['status']} | {f2(r['input_video'].get('duration_s'), 1, 's')} | "
                 f"{f2(r['input_audio'].get('duration_s'), 1, 's')} | {f2(r['output_video'].get('duration_s'), 1, 's')} | "
                 f"{f2(r['timing']['inference_s'], 1, 's')} | {f2(r['timing']['total_wall_s'], 1, 's')} | "
                 f"{f2(r['derived']['rtf_compute_s_per_output_s'], 1, 'x')} | "
                 f"{f2((v['device_used_peak_mb'] or 0) / 1024 if v['device_used_peak_mb'] else None, 2, ' GiB')} | "
                 f"{f2((v['pytorch_peak_allocated_mb'] or 0) / 1024 if v['pytorch_peak_allocated_mb'] else None, 2, ' GiB')} |")
    L += ["", "RTF = compute seconds spent per second of output video (lower is faster).", ""]
    for r in runs:
        v, t, d = r["vram"], r["timing"], r["derived"]
        iv, ia, ov = r["input_video"], r["input_audio"], r["output_video"]
        L += [f"## {r['name']}  ({r['status']})", "",
              f"- Input video: `{iv['path']}`: {f2(iv.get('duration_s'), 2, 's')}, "
              f"{iv.get('width')}x{iv.get('height')} @ {iv.get('fps')} fps, {iv.get('frames')} frames, {iv.get('size_mb')} MB",
              f"- Input audio: `{ia['path']}`: {f2(ia.get('duration_s'), 2, 's')}, "
              f"{ia.get('sample_rate')} Hz, {ia.get('size_mb')} MB",
              f"- Output video: `{ov['path']}`: {f2(ov.get('duration_s'), 2, 's')}, "
              f"{ov.get('width')}x{ov.get('height')} @ {ov.get('fps')} fps, {ov.get('frames')} frames, {ov.get('size_mb')} MB",
              f"- Timing: boot {f2(t['boot_s'], 1, 's')}, inference (model load + diffusion) {f2(t['inference_s'], 1, 's')}, "
              f"total wall {f2(t['total_wall_s'], 1, 's')}",
              f"- Throughput: RTF {f2(d['rtf_compute_s_per_output_s'], 2, 'x')}, "
              f"{f2(d['output_frames_per_compute_s'], 2)} output frames per compute-second",
              f"- Length check: audio - output = {f2(d['audio_minus_output_len_s'], 2, 's')}, "
              f"video - output = {f2(d['video_minus_output_len_s'], 2, 's')}",
              f"- CPU RAM peak: {f2(r['cpu_ram_peak_mb'], 0, ' MB')}"]
        if r.get("warnings"):
            L += ["- Warnings: " + "; ".join(r["warnings"])]
        L += ["", "**GPU memory**", "",
              "| Metric | MiB | Meaning |", "|---|---:|---|",
              f"| GPU slice total | {f2(v['gpu_total_mb'], 0)} | memory of the MIG slice you were given |",
              f"| PyTorch peak allocated | {f2(v['pytorch_peak_allocated_mb'], 0)} | tensors actually in use (weights, latents, activations) |",
              f"| PyTorch peak reserved | {f2(v['pytorch_peak_reserved_mb'], 0)} | PyTorch caching allocator pool (allocated + cache) |",
              f"| Device used, start | {f2(v['device_used_start_mb'], 0)} | before inference started (CUDA context) |",
              f"| Device used, peak | {f2(v['device_used_peak_mb'], 0)} | everything on the slice, sampled every 0.5 s |",
              f"| Non-PyTorch overhead | {f2(v['non_pytorch_overhead_mb'], 0)} | device peak minus reserved: CUDA context, cuDNN workspaces, ONNX Runtime |",
              f"| Headroom | {f2(v['headroom_mb'], 0)} | free memory at the peak ({f2(v['device_peak_pct_of_total'], 1, '%')} of slice used) |",
              ""]
        if v.get("gpu_util_avg_pct") is not None:
            L += [f"GPU utilisation: avg {v['gpu_util_avg_pct']}%, peak {v['gpu_util_peak_pct']}%", ""]
        if r.get("vram_plot"):
            L += [f"![VRAM timeline]({Path(r['vram_plot']).name})", ""]
        if r["status"] != "success":
            L += [f"**Failure:** return code {r['return_code']}. See `{Path(r['log_file']).name}`.", ""]
            if r.get("error"):
                L += ["```", r["error"][-1500:], "```", ""]
    (run_dir / "report.md").write_text("\n".join(L))
    return summary


def print_summary(runs):
    print("\n" + "=" * 100)
    print(f"{'Input':<22}{'Status':<10}{'Vid':>7}{'Aud':>7}{'Out':>7}{'Infer':>9}{'Wall':>9}{'RTF':>7}{'VRAM peak':>12}{'Alloc peak':>12}")
    print("-" * 100)
    for r in runs:
        v = r["vram"]
        gib = lambda x: f2(x / 1024, 2, "G") if x else "-"
        print(f"{r['name'][:21]:<22}{r['status']:<10}"
              f"{f2(r['input_video'].get('duration_s'), 1, 's'):>7}{f2(r['input_audio'].get('duration_s'), 1, 's'):>7}"
              f"{f2(r['output_video'].get('duration_s'), 1, 's'):>7}{f2(r['timing']['inference_s'], 1, 's'):>9}"
              f"{f2(r['timing']['total_wall_s'], 1, 's'):>9}{f2(r['derived']['rtf_compute_s_per_output_s'], 1, 'x'):>7}"
              f"{gib(v['device_used_peak_mb']):>12}{gib(v['pytorch_peak_allocated_mb']):>12}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------
def do_compare(reports_root):
    reports_root = Path(reports_root)
    files = sorted(reports_root.glob("*/report.json"))
    if not files:
        sys.exit(f"No report.json files under {reports_root}")
    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        for r in d.get("runs", []):
            ov, v = r.get("output_video", {}), r.get("vram", {})
            rows.append({
                "model": d.get("model", "?"),
                "input": r.get("name"),
                "status": r.get("status"),
                "video_len_s": r.get("input_video", {}).get("duration_s"),
                "audio_len_s": r.get("input_audio", {}).get("duration_s"),
                "output_len_s": ov.get("duration_s"),
                "inference_s": r.get("timing", {}).get("inference_s"),
                "total_wall_s": r.get("timing", {}).get("total_wall_s"),
                "rtf": r.get("derived", {}).get("rtf_compute_s_per_output_s"),
                "device_peak_mb": v.get("device_used_peak_mb"),
                "pytorch_peak_alloc_mb": v.get("pytorch_peak_allocated_mb"),
                "report": str(f),
            })
    out_dir = reports_root / f"comparison_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2, default=str))
    L = ["# Model comparison", "",
         "| Model | Input | Status | Video | Audio | Output | Inference | Wall | RTF | Peak VRAM | PyTorch alloc |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model']} | {r['input']} | {r['status']} | "
                 f"{f2(r['video_len_s'], 1, 's')} | {f2(r['audio_len_s'], 1, 's')} | "
                 f"{f2(r['output_len_s'], 1, 's')} | {f2(r['inference_s'], 1, 's')} | "
                 f"{f2(r['total_wall_s'], 1, 's')} | {f2(r['rtf'], 2, 'x')} | "
                 f"{f2((r['device_peak_mb'] or 0) / 1024 if r['device_peak_mb'] else None, 2, ' GiB')} | "
                 f"{f2((r['pytorch_peak_alloc_mb'] or 0) / 1024 if r['pytorch_peak_alloc_mb'] else None, 2, ' GiB')} |")
    (out_dir / "comparison.md").write_text("\n".join(L) + "\n")
    print(f"Compared {len(rows)} runs from {len(files)} reports")
    print(f"Written: {out_dir / 'comparison.json'}")
    print(f"Written: {out_dir / 'comparison.md'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        metrics_path, model = sys.argv[2], sys.argv[3]
        inf = sys.argv[4:]
        if inf[:1] == ["--"]:
            inf = inf[1:]
        worker_main(metrics_path, model, inf)
        return

    ap = argparse.ArgumentParser(description="Batch-run a lip-sync model and write benchmark reports")
    ap.add_argument("--model", choices=sorted(MODELS), default=None)
    ap.add_argument("--inputs", default=str(DEFAULT_INPUTS), help="folder with one sub-folder per demo")
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    ap.add_argument("--repo", default=None, help="override the model repo path")
    ap.add_argument("--select", default=None, help='e.g. "1,3", "1-3" or "all" (skips the prompt)')
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--list", action="store_true", help="only list inputs and exit")
    ap.add_argument("--compare", action="store_true", help="merge reports under <outputs>/reports")
    ap.add_argument("--force", action="store_true", help="continue even if preflight fails")
    ap.add_argument("--offline", action="store_true", help="set HF_HUB_OFFLINE=1")
    ap.add_argument("--steps", type=int, default=20, help="LatentSync inference steps")
    ap.add_argument("--guidance", type=float, default=1.5, help="LatentSync guidance scale")
    ap.add_argument("--no-deepcache", dest="deepcache", action="store_false",
                    help="LatentSync: disable DeepCache")
    ap.add_argument("--batch-size", type=int, default=1, help="MuseTalk batch size")
    args = ap.parse_args()

    inputs = Path(args.inputs)
    outputs = Path(args.outputs)

    if args.compare:
        do_compare(outputs / "reports")
        return

    print(f"Host: {socket.gethostname()}   Python: {sys.executable}")
    if not inputs.is_dir():
        sys.exit(f"Inputs folder not found: {inputs}")

    if not args.list:
        if not args.model:
            sys.exit("--model is required (latentsync | musetalk | wav2lip | heygem)")
        spec = MODELS[args.model]
        repo = Path(args.repo) if args.repo else spec["repo"]
        if not preflight(args.model, repo, spec) and not args.force:
            sys.exit("Preflight failed. Fix the FAIL lines above (or use --force).")

    items = discover(inputs)
    if not items:
        sys.exit(f"No sub-folders with both a video and an audio file under {inputs}")
    for it in items:
        it["vp"], it["ap"] = probe_media(it["video"]), probe_media(it["audio"])
    print_listing(items)
    if args.list:
        return

    sel_text = args.select
    while True:
        if sel_text is None:
            sel_text = input("Select inputs (e.g. 1,3 | 1-3 | all) [Enter = all, q = quit]: ")
            if sel_text.strip().lower() in ("q", "quit", "exit"):
                return
        try:
            chosen = parse_selection(sel_text, len(items))
            break
        except ValueError as e:
            print(f"  ! {e}")
            sel_text = None
            if args.select is not None:
                sys.exit(1)

    jobs = [items[i] for i in chosen]
    print("\nWill process: " + ", ".join(j["name"] for j in jobs))
    print(f"Model: {args.model}   Settings: {json.dumps(settings_dict(args))}")
    if not args.yes and input("Start? [Y/n]: ").strip().lower() in ("n", "no"):
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs / "reports" / f"{args.model}_run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "model_title": spec["title"],
        "system": system_info(repo),
        "settings": settings_dict(args),
    }

    runs, used = [], set()
    for n, job in enumerate(jobs, 1):
        base = f"{job['video'].stem}-{args.model}.mp4"
        if base in used:
            base = f"{job['name']}_{base}"
        used.add(base)
        out_path = outputs / base
        print("\n" + "#" * 100)
        print(f"# [{n}/{len(jobs)}] {job['name']}   ->   {out_path}")
        print("#" * 100)
        res = run_job(job, out_path, args, run_dir, repo, spec)
        runs.append(res)
        write_reports(run_dir, meta, runs)  # keep a valid report on disk after every job
        v = res["vram"]
        print(f"\n--> {res['status'].upper()} | inference {f2(res['timing']['inference_s'], 1, 's')} | "
              f"RTF {f2(res['derived']['rtf_compute_s_per_output_s'], 1, 'x')} | "
              f"VRAM peak {f2((v['device_used_peak_mb'] or 0) / 1024 if v['device_used_peak_mb'] else None, 2, ' GiB')}")
        if res["status"] == "interrupted":
            break

    print_summary(runs)
    print(f"\nReports written to: {run_dir}")
    for name in ("report.json", "report.md", "results.csv"):
        print(f"  {run_dir / name}")


if __name__ == "__main__":
    main()
