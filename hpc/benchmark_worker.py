#!/usr/bin/env python3
"""benchmark_worker.py - non-interactive run engine for the talking-head benchmark.

This file is ONLY executed inside Slurm jobs (batch jobs have no TTY, so it never
prompts). The interactive front-end lives in benchmark.py.

    benchmark_worker.py --plan PLAN.json --run-index 3   # one run
    benchmark_worker.py --plan PLAN.json --all           # every run, in order
    benchmark_worker.py --plan PLAN.json --report        # rebuild rollup only

It is also imported by benchmark.py for the shared helpers (scan_inputs, ...).
Stdlib only, so any of the project venvs can launch it.
"""

import argparse
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

try:                       # POSIX only; the driver also runs on Windows
    import resource
    _HAS_RESOURCE = True
except ImportError:        # pragma: no cover
    resource = None
    _HAS_RESOURCE = False
from pathlib import Path

HOME = Path.home()
PROJECTS = HOME / "projects"
DEFAULT_INPUTS = PROJECTS / "inputs"

MODELS = ("heygem", "latentsync", "musetalk", "wav2lip")

MODEL_ENV_PY = {
    "heygem": HOME / "envs" / "hygem-rtx50" / "bin" / "python",
    "latentsync": HOME / "envs" / "latentsync" / "bin" / "python",
    "musetalk": HOME / "envs" / "musetalk" / "bin" / "python",
    "wav2lip": HOME / "envs" / "latentsync" / "bin" / "python",
}

MODEL_PROJECT = {
    "heygem": PROJECTS / "hygem-rtx50",
    "latentsync": PROJECTS / "LatentSync",
    "musetalk": PROJECTS / "MuseTalk",
    "wav2lip": PROJECTS / "Wav2Lip",
}

QUALITY_PY = HOME / "envs" / "hygem-rtx50" / "bin" / "python"
QUALITY_SCRIPT = PROJECTS / "quality_metrics.py"

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi")
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".flac", ".aac")
AUDIO_WORDS = ("voice", "audio", "speech", "tts")


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def log(msg):
    print(f"[bench] {msg}", flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_dur(seconds):
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {sec:.0f}s"


def child_cpu_s():
    """CPU seconds (user+sys) of all reaped children so far."""
    if not _HAS_RESOURCE:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def summarize(rec):
    """Plain-language benchmark line + the numbers people actually quote."""
    model = rec.get("model")
    tag = rec.get("tag")
    inp = rec.get("input") or {}
    peaks = rec.get("peaks") or {}
    out = rec.get("output") or {}
    audio_s = (inp.get("audio") or {}).get("duration_s")
    video_in_s = (inp.get("video") or {}).get("duration_s")
    wall = rec.get("wall_s")
    rtf = rec.get("rtf")
    cpu = rec.get("cpu_time_s")
    vram = peaks.get("vram_mb")
    total = peaks.get("vram_total_mb")
    pct = peaks.get("vram_pct")
    out_s = out.get("duration_s")

    def f(x, fmt="{:.1f}"):
        return fmt.format(x) if isinstance(x, (int, float)) else "-"

    headline = (f"{model} [{tag}]: {f(audio_s)}s audio -> "
                f"{f(out_s)}s {out.get('resolution') or '?'} "
                f"@{f(out.get('fps'))}fps in {human_dur(wall)} "
                f"(RTF {f(rtf, '{:.2f}')}x realtime)")
    if rec.get("status") != "COMPLETED":
        headline = f"{model} [{tag}]: {rec.get('status')} - {rec.get('error') or rec.get('note') or ''}"[:300]
    return {
        "headline": headline,
        "status": rec.get("status"),
        "audio_s": audio_s,
        "video_in_s": video_in_s,
        "video_out_s": out_s,
        "wall_s": wall,
        "wall_human": human_dur(wall),
        "rtf": rtf,
        "rtf_note": (f"{f(wall)}s processing / {f(audio_s)}s audio"
                     if wall and audio_s else None),
        "cpu_time_s": cpu,
        "cpu_time_human": human_dur(cpu),
        "ram_peak_mb": peaks.get("cpu_rss_mb"),
        "vram_peak_mb": vram,
        "vram_total_mb": total,
        "vram_peak_pct": pct,
        "gpu_util_peak_pct": peaks.get("gpu_util_pct"),
        "resolution": out.get("resolution"),
        "fps": out.get("fps"),
        "output_size_mb": out.get("size_mb"),
        "output_path": out.get("path"),
        "gpu": rec.get("gpu_name"),
        "torch": rec.get("torch_version"),
        "cuda": rec.get("cuda_version"),
        "host": rec.get("host"),
        "slurm_job_id": rec.get("slurm_job_id"),
        "av_mismatch_s": inp.get("av_mismatch_s"),
        "samples": peaks.get("samples"),
        "test_mode": rec.get("test_mode"),
        "audio_out_diff_s": rec.get("audio_out_diff_s"),
        "throughput_jobs_per_gpu_hour": rec.get("throughput_jobs_per_gpu_hour"),
        "first_alloc_s": peaks.get("vram_first_alloc_s"),
        "gpu_util_avg_pct": peaks.get("gpu_util_avg_pct"),
        # quality metrics (null when the post-run quality job has not run)
        "lse_d": (rec.get("quality") or {}).get("metrics", {}).get("lse_d"),
        "lse_c": (rec.get("quality") or {}).get("metrics", {}).get("lse_c"),
        "fid": (rec.get("quality") or {}).get("metrics", {}).get("fid"),
        "fvd": (rec.get("quality") or {}).get("metrics", {}).get("fvd"),
        "ssim": (rec.get("quality") or {}).get("metrics", {}).get("ssim"),
        "psnr": (rec.get("quality") or {}).get("metrics", {}).get("psnr"),
        "lpips": (rec.get("quality") or {}).get("metrics", {}).get("lpips"),
        "csim": (rec.get("quality") or {}).get("metrics", {}).get("csim"),
        "lmd_mouth": (rec.get("quality") or {}).get("metrics", {}).get("lmd_mouth"),
        "niqe": (rec.get("quality") or {}).get("metrics", {}).get("niqe"),
        "mos": (rec.get("quality") or {}).get("metrics", {}).get("mos"),
    }


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- #
# input discovery
# --------------------------------------------------------------------------- #
def scan_inputs(inputs_dir):
    """Return a list of {folder, video, audio, tag, pair_index(1-based)}.

    Rule (confirmed with the operator): inside each sub-folder the *video* is the
    media file whose name does NOT look like audio (voice/audio/speech/tts), and
    the *audio* is any audio extension. Multiple files are paired by sorted order.
    """
    inputs_dir = Path(inputs_dir)
    pairs = []
    folders = sorted([p for p in inputs_dir.iterdir() if p.is_dir()])
    for idx, folder in enumerate(folders, start=1):
        vids, auds = [], []
        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            stem = f.stem.lower()
            if ext in AUDIO_EXT:
                auds.append(f)
            elif ext in VIDEO_EXT:
                vids.append(f)
            elif ext in (".mp4",) and any(w in stem for w in AUDIO_WORDS):
                auds.append(f)
        # a "voice*.mp4" style file is really audio; move those across
        still_vids = []
        for v in vids:
            if any(w in v.stem.lower() for w in AUDIO_WORDS):
                auds.append(v)
            else:
                still_vids.append(v)
        vids = still_vids
        if not vids or not auds:
            log(f"skip {folder.name}: {len(vids)} video / {len(auds)} audio")
            continue
        for i in range(min(len(vids), len(auds))):
            tag = folder.name if min(len(vids), len(auds)) == 1 else f"{folder.name}_{i + 1}"
            pairs.append({
                "tag": tag,
                "folder": str(folder),
                "video": str(vids[i]),
                "audio": str(auds[i]),
                "pair_index": idx,
            })
        if len(vids) != len(auds):
            log(f"WARN {folder.name}: {len(vids)} video vs {len(auds)} audio - "
                f"using {min(len(vids), len(auds))} pair(s)")
    return pairs


# --------------------------------------------------------------------------- #
# media / hardware probes
# --------------------------------------------------------------------------- #
def find_bin(name):
    """Locate ffprobe/ffmpeg: $PATH first, then the model venvs / common spots."""
    env_override = os.environ.get(name.upper())
    if env_override and Path(env_override).exists():
        return env_override
    found = shutil.which(name)
    if found:
        return found
    candidates = []
    for py in set(str(p) for p in MODEL_ENV_PY.values()):
        candidates.append(Path(py).parent / name)
    for root in (HOME / "miniconda3", HOME / "anaconda3", HOME / "miniforge3"):
        if root.exists():
            candidates += list(root.glob(f"envs/*/bin/{name}"))
            candidates.append(root / "bin" / name)
    candidates += [Path("/usr/local/bin") / name, Path("/usr/bin") / name,
                   Path("/opt/conda/bin") / name]
    for cand in candidates:
        try:
            if cand.exists() and os.access(cand, os.X_OK):
                return str(cand)
        except OSError:
            continue
    return name


FFPROBE = None


def _ffprobe_bin():
    global FFPROBE
    if FFPROBE is None:
        FFPROBE = find_bin("ffprobe")
    return FFPROBE


def ffprobe_json(path):
    try:
        out = subprocess.run(
            [_ffprobe_bin(), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, timeout=180,
        )
        if not (out.stdout or "").strip():
            return {"_error": (out.stderr or "ffprobe produced no output").strip()[:300]}
        return json.loads(out.stdout)
    except Exception as exc:  # noqa: BLE001
        return {"_error": str(exc)}


def media_info(path):
    path = Path(path) if path else None
    info = {"path": str(path) if path else None, "exists": False, "size_mb": None,
            "duration_s": None, "width": None, "height": None, "fps": None,
            "vcodec": None}
    if not path or not path.exists():
        return info
    info["exists"] = True
    info["size_mb"] = round(path.stat().st_size / 1e6, 3)
    data = ffprobe_json(path)
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        try:
            info["duration_s"] = round(float(fmt["duration"]), 2)
        except ValueError:
            pass
    for st in data.get("streams") or []:
        if st.get("codec_type") == "video":
            info["width"] = st.get("width")
            info["height"] = st.get("height")
            info["vcodec"] = st.get("codec_name")
            rate = st.get("avg_frame_rate") or "0/0"
            try:
                num, den = rate.split("/")
                if float(den):
                    info["fps"] = round(float(num) / float(den), 2)
            except Exception:  # noqa: BLE001
                pass
            break
    return info


def _num(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def _smi(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def nvidia_query():
    """Return (used_mb, total_mb, util_pct|None).

    MIG matters here: on a MIG slice `utilization.gpu` is not always queryable,
    so ask for the full tuple first and fall back to memory-only - one failing
    field used to kill the whole sample.
    """
    dev = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    mem_util = ["--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"]
    mem_only = ["--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits"]
    attempts = []
    if dev and dev.startswith("MIG-"):
        attempts += [["nvidia-smi", "-i", dev] + mem_util,
                     ["nvidia-smi", "-i", dev] + mem_only]
    attempts += [["nvidia-smi"] + mem_util, ["nvidia-smi"] + mem_only]

    for cmd in attempts:
        text = _smi(cmd)
        for line in text.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            used, total = _num(parts[0]), _num(parts[1])
            if used is None or total is None:
                continue
            util = _num(parts[2]) if len(parts) > 2 else None
            return used, total, util

    # last resort: sum of per-process GPU memory (no total, no util)
    text = _smi(["nvidia-smi", "--query-compute-apps=used_memory",
                 "--format=csv,noheader,nounits"])
    vals = [_num(v) for v in text.split()] if text else []
    vals = [v for v in vals if v is not None]
    if vals:
        return sum(vals), 0.0, None
    return None


def gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        return (out.stdout or "").strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None


def tree_rss_mb(pgid):
    """Sum RSS (MB) of every process in the child's process group."""
    total_kb = 0
    page_kb = os.sysconf("SC_PAGE_SIZE") // 1024
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as fh:
                fields = fh.read().split(b") ", 1)[1].split()
            if len(fields) < 22 or int(fields[2]) != pgid:   # pgrp
                continue
            # fields are offset by 3: field N of proc(5) == fields[N - 3].
            # rss is field 24 (pages); fields[22] would be vsize (bytes).
            total_kb += int(fields[21]) * page_kb
        except Exception:  # noqa: BLE001
            continue
    return total_kb / 1024.0


class Sampler(threading.Thread):
    """Polls GPU memory/util + process-tree RSS while the model runs."""

    def __init__(self, pgid, interval=0.5, t0=None):
        super().__init__(daemon=True)
        self.pgid = pgid
        self.interval = interval
        self.t0 = t0 or time.time()
        self.first_alloc_s = None
        self.util_sum = 0.0
        self.util_samples = 0
        # NOTE: never name this _stop - Thread._stop() is an internal method in
        # Python 3.9 and shadowing it makes join() raise
        # "'Event' object is not callable".
        self._stop_evt = threading.Event()
        self.peak_vram = 0.0
        self.peak_util = 0.0
        self.util_known = False
        self.peak_rss = 0.0
        self.vram_total = 0.0
        self.samples = 0
        self.baseline_vram = 0.0

    def run(self):
        first = True
        while not self._stop_evt.is_set():
            gpu = nvidia_query()
            if gpu:
                used, total, util = gpu
                if first:
                    self.baseline_vram = used
                    first = False
                self.vram_total = max(self.vram_total, total)
                self.peak_vram = max(self.peak_vram, used)
                if (self.first_alloc_s is None
                        and used > self.baseline_vram + 500):
                    self.first_alloc_s = round(time.time() - self.t0, 2)
                if util is not None:
                    self.util_known = True
                    self.util_sum += util
                    self.util_samples += 1
                    self.peak_util = max(self.peak_util, util)
            try:
                self.peak_rss = max(self.peak_rss, tree_rss_mb(self.pgid))
            except Exception:  # noqa: BLE001
                pass
            self.samples += 1
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=10)


TORCH_PROBE = """import sys
print('PYVER:' + sys.version.split()[0])
try:
    import torch
except Exception:
    print('TORCHVER:'); print('CUDAVER:'); print('DEVNAME:'); print('DEVTOTAL:')
else:
    print('TORCHVER:' + str(torch.__version__))
    print('CUDAVER:' + str(torch.version.cuda or ''))
    try:
        props = torch.cuda.get_device_properties(0)
        print('DEVNAME:' + str(props.name))
        print('DEVTOTAL:' + str(props.total_memory))
    except Exception:
        print('DEVNAME:'); print('DEVTOTAL:')
"""


EMPTY_TORCH = {"python": None, "torch": None, "cuda": None,
               "device": None, "device_total_mb": None}


def env_torch_version(py):
    """Python/torch/CUDA + the *slice* memory total (nvidia-smi reports the
    parent GPU total on MIG, torch reports the instance size)."""
    try:
        out = subprocess.run([str(py), "-c", TORCH_PROBE],
                             capture_output=True, text=True, timeout=240)
        if out.returncode != 0:
            return dict(EMPTY_TORCH)
        info = dict(EMPTY_TORCH)
        for line in (out.stdout or "").splitlines():
            key, _, value = line.partition(":")
            value = value.strip()
            if key == "PYVER":
                info["python"] = value or None
            elif key == "TORCHVER":
                info["torch"] = value or None
            elif key == "CUDAVER":
                info["cuda"] = value or None
            elif key == "DEVNAME":
                info["device"] = value or None
            elif key == "DEVTOTAL":
                try:
                    info["device_total_mb"] = round(float(value) / 1e6, 1) if value else None
                except ValueError:
                    info["device_total_mb"] = None
        return info
    except Exception:  # noqa: BLE001
        return dict(EMPTY_TORCH)


def newest_artifact(root, since_ts, patterns=("*.mp4", "*.mkv", "*.mov")):
    root = Path(root)
    best = None
    if not root.exists():
        return None
    for pattern in patterns:
        for path in root.rglob(pattern):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= since_ts and (best is None or mtime > best[0]):
                best = (mtime, path)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# per-model command construction
# --------------------------------------------------------------------------- #
def build_command(model, run, outdir):
    """Return (cmd, cwd, extra_env, hint_path, search_roots, select_index)."""
    video, audio = run["video"], run["audio"]
    tag = run["tag"]
    py = MODEL_ENV_PY[model]
    cwd = MODEL_PROJECT[model]

    if model == "heygem":
        inner = (f"source ./env.sh && exec {py} run.py "
                 f"--audio_path {audio} --video_path {video}")
        return (["bash", "-lc", inner], cwd, {}, cwd / "1004-r.mp4", [], None)

    if model == "wav2lip":
        target = outdir / "wav2lip" / f"{tag}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(py), "inference.py",
               "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
               "--face", video, "--audio", audio,
               "--outfile", str(target)]
        return (cmd, cwd, {}, target, [], None)

    if model == "musetalk":
        run_dir = outdir / "musetalk" / f"_raw_{tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg = run_dir / "task.yaml"
        cfg.write_text(
            "task_0:\n"
            f'  video_path: "{video}"\n'
            f'  audio_path: "{audio}"\n'
            "  bbox_shift: 0\n"
        )
        cmd = [str(py), "-u", "-m", "scripts.inference",
               "--inference_config", str(cfg),
               "--result_dir", str(run_dir / "out"),
               "--unet_model_path", "models/musetalkV15/unet.pth",
               "--unet_config", "models/musetalkV15/musetalk.json",
               "--whisper_dir", "models/whisper",
               "--version", "v15",
               "--use_float16",
               "--batch_size", "1"]
        env = {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "PYTHONUNBUFFERED": "1"}
        return (cmd, cwd, env, None, [run_dir / "out"], None)

    if model == "latentsync":
        # LatentSync ships its own multi-input harness; drive it for one demo only
        sel = str(run["pair_index"])
        cmd = [str(py), "-u", "latentsync_benchmark.py",
               "--inputs", run["inputs_dir"],
               "--select", sel,
               "--outputs", str(outdir / "latentsync" / f"_raw_{tag}"),
               "--offline", "--yes"]
        return (cmd, cwd, {"HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
                None, [outdir / "latentsync" / f"_raw_{tag}"], None)

    raise SystemExit(f"unknown model: {model}")


# --------------------------------------------------------------------------- #
# quality metrics (post-run, separate process so it cannot skew the sampler)
# --------------------------------------------------------------------------- #
def infer_test_mode(run):
    text = " ".join(str(run.get(k) or "") for k in ("tag", "folder", "video")).lower()
    return "reconstruction" if "recon" in text else "cross_audio"


def run_quality(plan, run, unified):
    """Run quality_metrics.py after a successful run; never fail the run."""
    if plan.get("quality") is False:
        return {"skipped": "quality disabled in plan"}
    if not QUALITY_SCRIPT.exists() or not QUALITY_PY.exists():
        return {"skipped": "quality script or python missing"}

    tag, model = run["tag"], run["model"]
    qdir = Path(plan["outdir"]) / model / "_quality"
    qdir.mkdir(parents=True, exist_ok=True)
    qjson = qdir / f"{tag}.json"
    qlog = qdir / f"{tag}.log"
    mode = run.get("test_mode") or infer_test_mode(run)

    cmd = [str(QUALITY_PY), "-u", str(QUALITY_SCRIPT),
           "--input-video", str(run["video"]),
           "--output-video", str(unified),
           "--audio", str(run["audio"]),
           "--mode", mode,
           "--out-json", str(qjson),
           "--device", "cuda"]
    started = time.time()
    log("quality: " + " ".join(cmd))
    rc = None
    try:
        with open(qlog, "wb") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, timeout=3600)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        return {"error": "quality metrics timed out after 3600s",
                "log": str(qlog), "mode": mode}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "log": str(qlog),
                "mode": mode}

    if qjson.exists():
        try:
            data = json.loads(qjson.read_text())
            return {
                "path": str(qjson),
                "log": str(qlog),
                "mode": mode,
                "runtime_s": round(time.time() - started, 2),
                "metrics": data.get("metrics") or {},
                "errors": data.get("errors") or {},
                "details": data.get("details") or {},
                "device": data.get("device"),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": f"cannot read quality json: {exc}",
                    "log": str(qlog), "mode": mode}
    return {"error": f"quality rc={rc}, no json", "log": str(qlog), "mode": mode}


# --------------------------------------------------------------------------- #
# one run
# --------------------------------------------------------------------------- #
def run_one(plan, run, timeout=0):
    model, tag = run["model"], run["tag"]
    outdir = Path(plan["outdir"])
    model_dir = outdir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = model_dir / f"{tag}.log"
    json_path = model_dir / f"{tag}.json"

    log("=" * 70)
    log(f"run #{run['index']}: {model} / {tag}")

    if not Path(run["video"]).exists():
        raise SystemExit(f"missing video: {run['video']}")
    if not Path(run["audio"]).exists():
        raise SystemExit(f"missing audio: {run['audio']}")

    run_for_cmd = dict(run)
    run_for_cmd["inputs_dir"] = plan["inputs_dir"]

    cmd, cwd, extra_env, hint, roots, _ = build_command(model, run_for_cmd, outdir)
    if not cwd.exists():
        raise SystemExit(f"project dir missing: {cwd}")

    a_info = media_info(run["audio"])
    v_info = media_info(run["video"])
    av_mismatch = None
    if a_info["duration_s"] and v_info["duration_s"]:
        av_mismatch = round(abs(a_info["duration_s"] - v_info["duration_s"]), 2)

    env = dict(os.environ)
    env.update(extra_env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    log("cmd: " + " ".join(cmd))
    started = time.time()
    started_iso = utc_now()
    note, status, exit_code = "", "FAILED", None

    cpu_before = child_cpu_s()
    cpu_time = None

    with open(log_path, "wb") as log_fh:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        sampler = Sampler(proc.pid, t0=started)
        sampler.start()
        try:
            exit_code = proc.wait(timeout=timeout or None)
        except subprocess.TimeoutExpired:
            note = f"killed after {timeout}s timeout"
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                pass
            exit_code = proc.wait()
        finally:
            wall = time.time() - started
            sampler.stop()
            cpu_time = max(child_cpu_s() - cpu_before, 0.0)

    if exit_code == 0:
        status = "COMPLETED"
    elif note:
        status = "TIMEOUT"
    else:
        try:
            tail = log_path.read_text(errors="replace").strip().splitlines()[-4:]
            note = " | ".join(t.strip() for t in tail)[:400]
        except Exception:  # noqa: BLE001
            note = "no log tail available"

    produced = Path(hint) if hint and Path(hint).exists() else None
    if produced is None:
        for root in roots:
            produced = newest_artifact(root, started)
            if produced:
                break
    if produced is None:
        produced = newest_artifact(outdir, started)   # last resort

    # A wrapper can exit 0 while its inference failed (LatentSync reports the
    # failure in its own report).  Never call a run complete without a video.
    if status == "COMPLETED" and produced is None:
        status = "FAILED"
        note = (note + " | no output video produced").strip(" |")

    unified = model_dir / f"{tag}.mp4"
    if produced and produced.exists() and status == "COMPLETED":
        try:
            if produced.resolve() != unified.resolve():
                shutil.copy2(produced, unified)
            produced = unified
        except Exception as exc:  # noqa: BLE001
            note = (note + f" | copy failed: {exc}").strip(" |")
    o_info = media_info(produced) if produced and produced.exists() else media_info(None)

    audio_s = a_info["duration_s"] or 0
    torch_info = env_torch_version(MODEL_ENV_PY[model])
    # torch reports the MIG slice size; nvidia-smi's total is the parent GPU
    vram_total = torch_info.get("device_total_mb") or sampler.vram_total
    fps_eff = None
    if o_info["duration_s"] and o_info["fps"]:
        fps_eff = o_info["fps"]

    record = {
        "index": run["index"],
        "model": model,
        "tag": tag,
        "status": status,
        "exit_code": exit_code,
        "start_utc": started_iso,
        "end_utc": utc_now(),
        "wall_s": round(wall, 2),
        "rtf": round(wall / audio_s, 3) if audio_s else None,
        "cpu_time_s": round(cpu_time, 2) if cpu_time is not None else None,
        "env_python": str(MODEL_ENV_PY[model]),
        "python_version": torch_info["python"],
        "torch_version": torch_info["torch"],
        "cuda_version": torch_info["cuda"],
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch_info.get("device") or gpu_name(),
        "input": {
            "video": v_info,
            "audio": a_info,
            "av_mismatch_s": av_mismatch,
            "av_mismatch": bool(av_mismatch and av_mismatch > 0.5),
        },
        "peaks": {
            "vram_mb": round(sampler.peak_vram, 1),
            "vram_total_mb": round(vram_total, 1) if vram_total else None,
            "vram_pct": (round(100.0 * sampler.peak_vram / vram_total, 1)
                         if vram_total else None),
            "vram_baseline_mb": round(sampler.baseline_vram, 1),
            "vram_first_alloc_s": sampler.first_alloc_s,
            "gpu_util_pct": (round(sampler.peak_util, 1)
                             if sampler.util_known else None),
            "gpu_util_avg_pct": (round(sampler.util_sum / sampler.util_samples, 1)
                                 if sampler.util_samples else None),
            "cpu_rss_mb": round(sampler.peak_rss, 1),
            "samples": sampler.samples,
        },
        "output": {
            "path": o_info["path"],
            "size_mb": o_info["size_mb"],
            "duration_s": o_info["duration_s"],
            "resolution": (f"{o_info['width']}x{o_info['height']}"
                           if o_info["width"] else None),
            "fps": o_info["fps"],
            "vcodec": o_info["vcodec"],
        },
        "command": " ".join(cmd),
        "cwd": str(cwd),
        "log_file": str(log_path),
        "note": note,
        "test_mode": run.get("test_mode") or infer_test_mode(run),
        "audio_out_diff_s": (round(o_info["duration_s"] - a_info["duration_s"], 3)
                            if o_info.get("duration_s") and a_info.get("duration_s")
                            else None),
        "throughput_jobs_per_gpu_hour": (round(3600.0 / wall, 2)
                                         if wall else None),
    }

    unified_path = Path(unified)
    if status == "COMPLETED" and unified_path.exists():
        record["quality"] = run_quality(plan, run, unified_path)
    record["summary"] = summarize(record)

    atomic_write_json(json_path, record)
    log(f"-> {status} rc={exit_code} wall={record['wall_s']}s "
        f"vram_peak={record['peaks']['vram_mb']}MB out={o_info['size_mb']}MB")
    if note:
        log(f"-> note: {note}")
    return record


# --------------------------------------------------------------------------- #
# rollup report (atomic snapshot, safe if several jobs finish together)
# --------------------------------------------------------------------------- #
REPORT_COLS = [
    "model", "tag", "test_mode", "status", "exit_code", "wall_s", "rtf",
    "av_mismatch_s", "peak_vram_mb", "peak_vram_pct", "peak_gpu_util",
    "gpu_util_avg", "first_alloc_s", "peak_rss_mb", "out_size_mb",
    "out_duration_s", "audio_out_diff_s", "resolution", "out_fps",
    "throughput_jobs_per_gpu_hour", "torch_version", "cuda_version",
    "gpu_name", "slurm_job_id", "host",
    "lse_d", "lse_c", "fid", "fvd", "ssim", "psnr", "lpips", "csim",
    "lmd_mouth", "niqe", "mos",
    "log_file", "note", "cpu_time_s", "headline",
]


def flatten(rec):
    row = {
        "model": rec.get("model"),
        "tag": rec.get("tag"),
        "status": rec.get("status"),
        "exit_code": rec.get("exit_code"),
        "wall_s": rec.get("wall_s"),
        "rtf": rec.get("rtf"),
        "test_mode": rec.get("test_mode"),
        "av_mismatch_s": (rec.get("input") or {}).get("av_mismatch_s"),
        "peak_vram_mb": (rec.get("peaks") or {}).get("vram_mb"),
        "peak_vram_pct": (rec.get("peaks") or {}).get("vram_pct"),
        "peak_gpu_util": (rec.get("peaks") or {}).get("gpu_util_pct"),
        "gpu_util_avg": (rec.get("peaks") or {}).get("gpu_util_avg_pct"),
        "first_alloc_s": (rec.get("peaks") or {}).get("vram_first_alloc_s"),
        "peak_rss_mb": (rec.get("peaks") or {}).get("cpu_rss_mb"),
        "out_size_mb": (rec.get("output") or {}).get("size_mb"),
        "out_duration_s": (rec.get("output") or {}).get("duration_s"),
        "audio_out_diff_s": rec.get("audio_out_diff_s"),
        "throughput_jobs_per_gpu_hour": rec.get("throughput_jobs_per_gpu_hour"),
        "resolution": (rec.get("output") or {}).get("resolution"),
        "out_fps": (rec.get("output") or {}).get("fps"),
        "lse_d": ((rec.get("quality") or {}).get("metrics") or {}).get("lse_d"),
        "lse_c": ((rec.get("quality") or {}).get("metrics") or {}).get("lse_c"),
        "fid": ((rec.get("quality") or {}).get("metrics") or {}).get("fid"),
        "fvd": ((rec.get("quality") or {}).get("metrics") or {}).get("fvd"),
        "ssim": ((rec.get("quality") or {}).get("metrics") or {}).get("ssim"),
        "psnr": ((rec.get("quality") or {}).get("metrics") or {}).get("psnr"),
        "lpips": ((rec.get("quality") or {}).get("metrics") or {}).get("lpips"),
        "csim": ((rec.get("quality") or {}).get("metrics") or {}).get("csim"),
        "lmd_mouth": ((rec.get("quality") or {}).get("metrics") or {}).get("lmd_mouth"),
        "niqe": ((rec.get("quality") or {}).get("metrics") or {}).get("niqe"),
        "mos": ((rec.get("quality") or {}).get("metrics") or {}).get("mos"),
        "torch_version": rec.get("torch_version"),
        "cuda_version": rec.get("cuda_version"),
        "gpu_name": rec.get("gpu_name"),
        "slurm_job_id": rec.get("slurm_job_id"),
        "host": rec.get("host"),
        "log_file": rec.get("log_file"),
        "note": rec.get("note"),
        "cpu_time_s": rec.get("cpu_time_s"),
        "headline": (rec.get("summary") or {}).get("headline"),
    }
    return row


def collect_records(outdir):
    outdir = Path(outdir)
    records = []
    for js in sorted(outdir.glob("*/*.json")):
        if js.parent.name.startswith("_"):
            continue
        try:
            rec = json.loads(js.read_text())
        except Exception:  # noqa: BLE001
            continue
        # merge a post-run quality snapshot if one exists
        qpath = outdir / str(rec.get("model")) / "_quality" / f"{rec.get('tag')}.json"
        if qpath.exists():
            try:
                qdata = json.loads(qpath.read_text())
                rec["quality"] = {
                    "path": str(qpath),
                    "mode": qdata.get("mode"),
                    "runtime_s": qdata.get("runtime_s"),
                    "metrics": qdata.get("metrics") or {},
                    "errors": qdata.get("errors") or {},
                    "details": qdata.get("details") or {},
                    "device": qdata.get("device"),
                }
            except Exception:  # noqa: BLE001
                pass
        # backfill the plain-language summary (also upgrades older records)
        summary = summarize(rec)
        if rec.get("summary") != summary:
            rec["summary"] = summary
            try:
                atomic_write_json(js, rec)
            except Exception:  # noqa: BLE001
                pass
        records.append(rec)
    records.sort(key=lambda r: (r.get("model", ""), r.get("tag", "")))
    return records


def refresh_report(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    records = collect_records(outdir)
    rows = [flatten(r) for r in records]

    atomic_write_json(outdir / "report.json", {
        "generated": utc_now(),
        "outdir": str(outdir),
        "runs": rows,
        "records": records,
    })

    buf = []
    for path in [outdir / "results.csv"]:
        tmp = path.with_suffix(".csv.tmp")
        import io
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=REPORT_COLS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        atomic_write_text(path, text.getvalue())

    lines = ["# Talking-head benchmark report", "",
             f"Generated: {utc_now()}", "",
             f"Runs captured: {len(rows)}", "",
             "| " + " | ".join(REPORT_COLS) + " |",
             "|" + "|".join(["---"] * len(REPORT_COLS)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            "-" if row.get(c) in (None, "") else str(row.get(c))
            for c in REPORT_COLS) + " |")
    lines += ["", "RTF = wall_seconds / audio_seconds (lower is faster).", ""]
    atomic_write_text(outdir / "report.md", "\n".join(lines))
    log(f"report refreshed: {outdir} ({len(rows)} run(s))")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--run-index", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--timeout", type=int, default=0)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    outdir = Path(plan["outdir"])

    log(f"host={socket.gethostname()} "
        f"SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '-')} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '-')}")

    if args.report:
        refresh_report(outdir)
        return

    runs = plan["runs"]
    if args.all:
        targets = runs
    elif args.run_index is not None:
        targets = [r for r in runs if r["index"] == args.run_index]
        if not targets:
            raise SystemExit(f"no run with index {args.run_index}")
    else:
        raise SystemExit("give --all, --run-index N or --report")

    failures = 0
    for run in targets:
        try:
            rec = run_one(plan, run, timeout=args.timeout)
            if rec["status"] != "COMPLETED":
                failures += 1
        except SystemExit as exc:
            log(f"ABORT run #{run['index']}: {exc}")
            failures += 1
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR run #{run['index']}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            try:   # never lose the run: keep a stub record so the report shows it
                atomic_write_json(
                    outdir / run["model"] / f"{run['tag']}.json",
                    {"index": run["index"], "model": run["model"],
                     "tag": run["tag"], "status": "ERROR",
                     "error": f"{type(exc).__name__}: {exc}",
                     "wall_s": None, "rtf": None})
            except Exception:  # noqa: BLE001
                pass
            failures += 1
        finally:
            try:
                refresh_report(outdir)
            except Exception as exc:  # noqa: BLE001
                log(f"report refresh failed: {exc}")

    log(f"DONE: {len(targets) - failures}/{len(targets)} completed")


if __name__ == "__main__":
    main()
