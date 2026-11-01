#!/usr/bin/env python3
"""quality_metrics.py - compute lip-sync / fidelity quality metrics for one run.

Designed to be called from benchmark_worker.py (or a Slurm quality job) after a
model has produced its output video.  It never prompts and always writes a JSON
snapshot, even when individual metrics fail.

Metrics
-------
LSE-D / LSE-C          SyncNet v2 (ByteDance LatentSync auxiliary checkpoint)
FID                    InceptionV3 Frechet distance (torch-fidelity)
FVD                    I3D Frechet distance (LatentSync torchscript I3D)
SSIM / PSNR / LPIPS    reconstruction only (output frame vs input frame)
CSIM                   ArcFace cosine similarity (InsightFace buffalo_l)
LMD                    68-landmark mouth distance, reconstruction only
NIQE                   no-reference image quality (patched niqe package)
sharpness              Laplacian variance (full frame + mouth crop)
MOS                    manual; always null here

Usage
-----
  quality_metrics.py --input-video IN.mp4 --output-video OUT.mp4 \
      --audio AUDIO.mp3 --mode reconstruction|cross_audio \
      --out-json OUT.json [--device cuda] [--skip-fid] [--skip-fvd]
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")

HOME = Path.home()
PROJECTS = HOME / "projects"
LATENTSYNC = PROJECTS / "LatentSync"
ASSETS = PROJECTS / "quality_assets"
SYNCNET_CKPT = LATENTSYNC / "checkpoints" / "auxiliary" / "syncnet_v2.model"
I3D_CKPT = LATENTSYNC / "checkpoints" / "auxiliary" / "i3d_torchscript.pt"
NIQE_MAT = ASSETS / "modelparameters.mat"
FFMPEG = HOME / "envs" / "latentsync" / "bin" / "ffmpeg"
FFPROBE = HOME / "envs" / "latentsync" / "bin" / "ffprobe"

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm")

_SYNC_MODEL = None
_SYNC_DEVICE = None
_FACE_ENGINE = None
_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_I3D_MODEL = None
_I3D_DEVICE = None
_NIQE = None


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    print(f"[quality] {msg}", flush=True)


def rnd(value, digits=4):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def ffprobe_info(path):
    """Best-effort ffprobe metadata; falls back to OpenCV."""
    info = {"duration_s": None, "width": None, "height": None, "fps": None}
    if FFPROBE.exists():
        try:
            out = subprocess.run(
                [str(FFPROBE), "-v", "error", "-print_format", "json",
                 "-show_format", "-show_streams", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            data = json.loads(out.stdout or "{}")
            dur = (data.get("format") or {}).get("duration")
            if dur:
                info["duration_s"] = round(float(dur), 3)
            for st in data.get("streams") or []:
                if st.get("codec_type") == "video":
                    info["width"] = st.get("width")
                    info["height"] = st.get("height")
                    num, den = (st.get("avg_frame_rate") or "0/0").split("/")
                    if float(den):
                        info["fps"] = round(float(num) / float(den), 3)
                    break
        except Exception:
            pass
    if not info["duration_s"] or not info["width"]:
        cap = cv2.VideoCapture(str(path))
        info["fps"] = info["fps"] or round(cap.get(cv2.CAP_PROP_FPS) or 25.0, 3)
        info["width"] = info["width"] or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = info["height"] or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count and info["fps"]:
            info["duration_s"] = round(count / info["fps"], 3)
        cap.release()
    return info


def extract_audio(path, out_wav):
    if not FFMPEG.exists():
        raise RuntimeError(f"ffmpeg not found: {FFMPEG}")
    cmd = [str(FFMPEG), "-v", "error", "-y", "-i", str(path),
           "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_wav)]
    subprocess.run(cmd, check=True, timeout=300,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_wav


def read_wav(path):
    from scipy.io import wavfile
    sr, audio = wavfile.read(str(path))
    # Keep int16: the reference SyncNet implementation feeds raw int16 MFCC.
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype(np.int16)
    return sr, audio


# --------------------------------------------------------------------------- #
# face engine (InsightFace buffalo_l, CPU for reliability on MIG)
# --------------------------------------------------------------------------- #
class FaceEngine:
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.last_bbox = None
        self.detections = 0
        self.misses = 0

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, 640.0 / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        try:
            faces = self.app.get(small)
        except Exception:
            faces = []
        if not faces:
            self.misses += 1
            return None
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        bbox = np.asarray(face.bbox, dtype=np.float32) / scale
        if self.last_bbox is None:
            self.last_bbox = bbox
        else:
            self.last_bbox = 0.55 * bbox + 0.45 * self.last_bbox
        emb = getattr(face, "normed_embedding", None)
        lmk = getattr(face, "landmark_3d_68", None)
        if lmk is not None:
            lmk = np.asarray(lmk, dtype=np.float32)[:, :2]
        self.detections += 1
        return {"bbox": self.last_bbox.copy(),
                "embedding": None if emb is None else np.asarray(emb, dtype=np.float32),
                "landmark68": lmk}


def crop_face(frame, bbox, size=224, margin=0.35):
    h, w = frame.shape[:2]
    if bbox is None:
        side = min(h, w) * 0.7
        cx, cy = w / 2.0, h / 2.0
    else:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        side = max(x2 - x1, y2 - y1) * (1.0 + margin)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    x1 = int(max(0, min(w - 1, round(cx - half))))
    x2 = int(max(x1 + 2, min(w, round(cx + half))))
    y1 = int(max(0, min(h - 1, round(cy - half))))
    y2 = int(max(y1 + 2, min(h, round(cy + half))))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def mouth_crop(frame, bbox):
    h, w = frame.shape[:2]
    if bbox is None:
        x1, x2 = int(w * 0.3), int(w * 0.7)
        y1, y2 = int(h * 0.45), int(h * 0.8)
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 + int(bw * 0.12))
        x2 = min(w, x2 - int(bw * 0.12))
        y1 = max(0, y1 + int(bh * 0.52))
        y2 = min(h, y2)
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size else frame


# --------------------------------------------------------------------------- #
# one-pass video analysis
# --------------------------------------------------------------------------- #
def analyze_video(path, engine, sample_fps=3.0, sync_fps=25.0):
    """Decode once; collect face crops (25 fps), embeddings/landmarks (3 fps),
    small full frames, NIQE/sharpness samples and 224px full frames for FVD."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_step = max(1, int(round(fps / max(sample_fps, 0.1))))
    sync_period = 1.0 / max(sync_fps, 1.0)
    det0, miss0 = engine.detections, engine.misses

    face_crops, fvd_frames = [], []
    small_frames, emb_list, lmk_list, emb_times = [], [], [], []
    niqe_grays, sharp_full, sharp_mouth = [], [], []
    bboxes, times = [], []
    frame_idx = 0
    next_sync = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps

        if frame_idx % sample_step == 0:
            det = engine.detect(frame)
            if det is not None:
                emb_list.append(det["embedding"])
                lmk_list.append(det["landmark68"])
                emb_times.append(t)
                bboxes.append(det["bbox"])
            elif engine.last_bbox is not None:
                bboxes.append(engine.last_bbox.copy())

            small = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            small_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp_full.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            mouth = mouth_crop(frame, engine.last_bbox)
            if mouth.size:
                mg = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                sharp_mouth.append(float(cv2.Laplacian(mg, cv2.CV_64F).var()))
            # NIQE on a face-centred crop (2x bbox): stable, face-region aware.
            # A tight mouth crop is numerically unstable for this estimator.
            face_niqe = crop_face(frame, engine.last_bbox, 256, margin=1.0)
            niqe_grays.append(cv2.cvtColor(face_niqe, cv2.COLOR_BGR2GRAY))
            times.append(t)

        # resample to a true 25 fps timeline (duplicate a frame when 24 -> 25)
        while next_sync <= t + 0.5 / fps:
            bbox = engine.last_bbox
            face_crops.append(crop_face(frame, bbox, 224, margin=0.35))
            fvd_frames.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
            next_sync += sync_period

        frame_idx += 1

    cap.release()
    return {
        "path": str(path),
        "fps": round(float(fps), 3),
        "frames": frame_idx,
        "duration_s": round(frame_idx / fps, 3) if fps else None,
        "face_crops": face_crops,
        "fvd_frames": fvd_frames,
        "small_frames": small_frames,
        "embeddings": emb_list,
        "landmarks": lmk_list,
        "landmark_times": emb_times,
        "boxes": bboxes,
        "times": times,
        "niqe_grays": niqe_grays,
        "sharp_full": sharp_full,
        "sharp_mouth": sharp_mouth,
        "face_detection_rate": ((engine.detections - det0)
                               / max(1, (engine.detections - det0) + (engine.misses - miss0))),
    }


# --------------------------------------------------------------------------- #
# SyncNet (LSE-D / LSE-C)
# --------------------------------------------------------------------------- #
def load_syncnet(device):
    global _SYNC_MODEL, _SYNC_DEVICE
    if _SYNC_MODEL is not None and _SYNC_DEVICE == device:
        return _SYNC_MODEL
    import torch
    spec = importlib.util.spec_from_file_location(
        "ls_syncnet", str(LATENTSYNC / "eval" / "syncnet" / "syncnet.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load LatentSync syncnet.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.S()
    state = torch.load(str(SYNCNET_CKPT), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    _SYNC_MODEL, _SYNC_DEVICE = model, device
    return model


def calc_pdist(feat1, feat2, vshift=15):
    import torch
    win = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(torch.nn.functional.pairwise_distance(
            feat1[[i], :].repeat(win, 1), feat2p[i:i + win, :]))
    return dists


def compute_lse(face_crops, audio, sr, device, batch_size=20, vshift=15):
    import python_speech_features as psf
    import torch

    if len(face_crops) < 10:
        raise RuntimeError(f"too few face frames: {len(face_crops)}")
    if sr != 16000:
        raise RuntimeError(f"audio must be 16 kHz, got {sr}")
    if len(audio) < 16000 * 0.4:
        raise RuntimeError("audio too short")

    model = load_syncnet(device)
    im = np.stack(face_crops, axis=3)              # H,W,C,N
    im = np.expand_dims(im, axis=0)                # 1,H,W,C,N
    im = np.transpose(im, (0, 3, 4, 1, 2))         # 1,C,N,H,W
    imtv = torch.from_numpy(im.astype(np.float32))

    mfcc = psf.mfcc(audio, sr)
    mfcc = np.stack([np.array(i) for i in zip(*mfcc)])  # 13,T
    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.from_numpy(cc.astype(np.float32))

    min_length = min(len(face_crops), int(math.floor(len(audio) / 640)))
    last = min_length - 5
    if last <= 5:
        raise RuntimeError(f"not enough aligned frames: {last}")

    im_feat, cc_feat = [], []
    with torch.no_grad():
        for i in range(0, last, batch_size):
            im_batch = [imtv[:, :, v:v + 5, :, :] for v in range(i, min(last, i + batch_size))]
            im_out = model.forward_lip(torch.cat(im_batch, 0).to(device))
            im_feat.append(im_out.cpu())

            cc_batch = [cct[:, :, :, v * 4:v * 4 + 20] for v in range(i, min(last, i + batch_size))]
            cc_out = model.forward_aud(torch.cat(cc_batch, 0).to(device))
            cc_feat.append(cc_out.cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)
    dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
    mean_dists = torch.mean(torch.stack(dists, 1), 1)
    min_dist, minidx = torch.min(mean_dists, 0)
    av_offset = vshift - int(minidx.item())
    conf = torch.median(mean_dists) - min_dist
    return {"lse_d": float(min_dist.item()), "lse_c": float(conf.item()),
            "av_offset": av_offset, "frames_used": int(last)}


# --------------------------------------------------------------------------- #
# FID / FVD
# --------------------------------------------------------------------------- #
def _write_frames(frames, folder, limit=150):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not frames:
        return 0
    idxs = np.linspace(0, len(frames) - 1, min(limit, len(frames))).astype(int)
    for k, i in enumerate(idxs):
        cv2.imwrite(str(folder / f"{k:04d}.jpg"), frames[int(i)],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return len(idxs)


def compute_fid(frames_a, frames_b, device, limit=150):
    import torch_fidelity
    if len(frames_a) < 10 or len(frames_b) < 10:
        raise RuntimeError(f"too few frames for FID: {len(frames_a)} vs {len(frames_b)}")
    tmp = Path(tempfile.mkdtemp(prefix="fid_", dir="/tmp"))
    try:
        da, db = tmp / "a", tmp / "b"
        na = _write_frames(frames_a, da, limit)
        nb = _write_frames(frames_b, db, limit)
        out = torch_fidelity.calculate_metrics(
            input1=str(da), input2=str(db),
            cuda=(device == "cuda"), fid=True, kid=False, isc=False, verbose=False)
        return float(out["frechet_inception_distance"]), {"frames": [na, nb]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_i3d(device):
    global _I3D_MODEL, _I3D_DEVICE
    if _I3D_MODEL is not None and _I3D_DEVICE == device:
        return _I3D_MODEL
    import torch
    model = torch.jit.load(str(I3D_CKPT), map_location="cpu").eval().to(device)
    _I3D_MODEL, _I3D_DEVICE = model, device
    return model


def _fvd_clips(frames, clip_len=16, stride=8, max_clips=24):
    if len(frames) < clip_len:
        return []
    starts = list(range(0, len(frames) - clip_len + 1, stride))
    if len(starts) > max_clips:
        picks = np.linspace(0, len(starts) - 1, max_clips).astype(int)
        starts = [starts[int(i)] for i in picks]
    return [np.stack(frames[s:s + clip_len], axis=0) for s in starts]


def _frechet(mu1, c1, mu2, c2):
    import scipy.linalg
    d = mu1.shape[0]
    eps = max(1e-6, 1e-4 * float(np.trace(c1) + np.trace(c2)) / (2.0 * d))
    c1 = c1 + np.eye(d, dtype=np.float64) * eps
    c2 = c2 + np.eye(d, dtype=np.float64) * eps
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(c1.dot(c2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(c1 + c2 - 2.0 * covmean))


def compute_fvd(frames_a, frames_b, device):
    import torch
    clips_a = _fvd_clips(frames_a)
    clips_b = _fvd_clips(frames_b)
    n = min(len(clips_a), len(clips_b))
    if n < 2:
        raise RuntimeError(f"not enough 16-frame clips: {len(clips_a)} vs {len(clips_b)}")
    clips_a, clips_b = clips_a[:n], clips_b[:n]
    model = load_i3d(device)

    def feats(clips):
        out = []
        for i in range(0, len(clips), 4):
            batch = np.stack(clips[i:i + 4], axis=0)          # B,F,H,W,C (BGR)
            batch = batch[:, :, :, :, ::-1].astype(np.float32) / 255.0
            x = torch.from_numpy(batch).permute(0, 4, 1, 2, 3).to(device)
            with torch.no_grad():
                out.append(model(x, rescale=False, resize=False,
                                 return_features=True).cpu().numpy())
        return np.concatenate(out, axis=0)

    fa, fb = feats(clips_a), feats(clips_b)
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    c_a, c_b = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    return _frechet(mu_a, c_a, mu_b, c_b), {"clips": n}


# --------------------------------------------------------------------------- #
# SSIM / PSNR / LPIPS / CSIM / LMD
# --------------------------------------------------------------------------- #
def compute_recon_metrics(inp, out, max_pairs=100):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    n = min(len(inp["small_frames"]), len(out["small_frames"]), max_pairs)
    if n < 3:
        raise RuntimeError(f"too few paired frames: {n}")
    ssims, psnrs = [], []
    pairs_a, pairs_b = [], []
    for i in range(n):
        a = inp["small_frames"][i]
        b = out["small_frames"][i]
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        ssims.append(float(structural_similarity(ga, gb, data_range=255)))
        psnrs.append(float(peak_signal_noise_ratio(ga, gb, data_range=255)))
        pairs_a.append(a)
        pairs_b.append(b)
    return (float(np.mean(ssims)), float(np.mean(psnrs)),
            pairs_a, pairs_b, {"pairs": n})


def compute_lpips(pairs_a, pairs_b, device, batch=16):
    global _LPIPS_MODEL, _LPIPS_DEVICE
    import torch
    import lpips as lpips_pkg
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != device:
        _LPIPS_MODEL = lpips_pkg.LPIPS(net="alex", verbose=False).eval().to(device)
        _LPIPS_DEVICE = device
    values = []
    with torch.no_grad():
        for i in range(0, len(pairs_a), batch):
            a = np.stack([cv2.resize(x, (256, 256)) for x in pairs_a[i:i + batch]])
            b = np.stack([cv2.resize(x, (256, 256)) for x in pairs_b[i:i + batch]])
            ta = torch.from_numpy(a.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).to(device)
            tb = torch.from_numpy(b.astype(np.float32) / 127.5 - 1.0).permute(0, 3, 1, 2).to(device)
            values.append(_LPIPS_MODEL(ta, tb).flatten().cpu().numpy())
    return float(np.mean(np.concatenate(values)))


def compute_csim(inp, out):
    ref = [e for e in inp["embeddings"] if e is not None]
    cur = [e for e in out["embeddings"] if e is not None]
    if not ref or not cur:
        raise RuntimeError("no face embeddings")
    ref_mean = np.mean(np.stack(ref, 0), axis=0)
    ref_mean = ref_mean / max(1e-8, float(np.linalg.norm(ref_mean)))
    sims = [float(np.dot(ref_mean, e) / max(1e-8, float(np.linalg.norm(e)))) for e in cur]
    return float(np.mean(sims)), {"input_faces": len(ref), "output_faces": len(cur)}


def compute_lmd(inp, out):
    a = [x for x in inp["landmarks"] if x is not None]
    b = [x for x in out["landmarks"] if x is not None]
    n = min(len(a), len(b))
    if n < 3:
        raise RuntimeError(f"too few landmark pairs: {n}")
    mouth_vals, all_vals = [], []
    for i in range(n):
        la, lb = a[i], b[i]
        if la.shape[0] < 68 or lb.shape[0] < 68:
            continue
        iod = float(np.linalg.norm(la[36:42].mean(0) - la[42:48].mean(0)))
        if iod < 1e-6:
            iod = float(np.linalg.norm(la[45] - la[36]))
        if iod < 1e-6:
            continue
        mouth_vals.append(float(np.mean(np.linalg.norm(lb[48:68] - la[48:68], axis=1)) / iod))
        all_vals.append(float(np.mean(np.linalg.norm(lb - la, axis=1)) / iod))
    if not mouth_vals:
        raise RuntimeError("no valid landmark pairs")
    return (float(np.mean(mouth_vals)), float(np.mean(all_vals)),
            {"pairs": len(mouth_vals)})


# --------------------------------------------------------------------------- #
# NIQE / sharpness
# --------------------------------------------------------------------------- #
def load_niqe():
    global _NIQE
    if _NIQE is not None:
        return _NIQE
    sys.path.insert(0, str(ASSETS))
    import niqe_fixed as nq
    _NIQE = nq
    return nq


def compute_niqe(grays):
    nq = load_niqe()
    old = os.getcwd()
    os.chdir(str(ASSETS))     # niqe_fixed loads modelparameters.mat relatively
    try:
        vals = []
        for g in grays:
            try:
                # the niqe package's MSCN normaliser assumes img in [0, 1];
                # feeding 0-255 inflates the score by ~2 orders of magnitude
                vals.append(float(nq.niqe(g.astype(np.float64) / 255.0)))
            except Exception:
                continue
        if not vals:
            raise RuntimeError("NIQE failed on every sample")
        # median: robust to occasional degenerate blocks in this estimator
        return (float(np.median(vals)),
                {"valid_samples": len(vals), "samples": len(grays),
                 "stat": "median", "crop": "face 2x bbox"})
    finally:
        os.chdir(old)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-video", required=True)
    ap.add_argument("--output-video", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--mode", choices=("reconstruction", "cross_audio"), default="cross_audio")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-fid", action="store_true")
    ap.add_argument("--skip-fvd", action="store_true")
    ap.add_argument("--max-fid-frames", type=int, default=150)
    args = ap.parse_args()

    started = time.time()
    result = {
        "schema": 1,
        "generated": now_iso(),
        "mode": args.mode,
        "input_video": args.input_video,
        "output_video": args.output_video,
        "audio": args.audio,
        "metrics": {},
        "details": {},
        "errors": {},
        "runtime_s": None,
    }

    try:
        import torch
        device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    except Exception:
        device = "cpu"
    result["device"] = device

    try:
        for p in (args.input_video, args.output_video, args.audio):
            if not Path(p).exists():
                raise FileNotFoundError(p)

        v_in = ffprobe_info(args.input_video)
        v_out = ffprobe_info(args.output_video)
        a_info = ffprobe_info(args.audio)
        result["details"]["media"] = {"input": v_in, "output": v_out, "audio": a_info}
        if v_out.get("duration_s") and a_info.get("duration_s"):
            result["metrics"]["audio_out_diff_s"] = rnd(v_out["duration_s"] - a_info["duration_s"], 3)

        wav = Path(tempfile.mkdtemp(prefix="qaudio_", dir="/tmp")) / "audio.wav"
        extract_audio(args.audio, wav)
        sr, audio = read_wav(wav)
        result["details"]["audio_samples"] = int(len(audio))

        engine = FaceEngine()
        log(f"analysing input {args.input_video}")
        inp = analyze_video(args.input_video, engine)
        log(f"analysing output {args.output_video}")
        out = analyze_video(args.output_video, engine)
        result["details"]["input_frames"] = inp["frames"]
        result["details"]["output_frames"] = out["frames"]
        result["details"]["face_detection_rate"] = rnd(out["face_detection_rate"], 3)
        result["details"]["input_face_detection_rate"] = rnd(inp["face_detection_rate"], 3)

        def safe(name, fn):
            try:
                value = fn()
                if isinstance(value, tuple):
                    result["metrics"][name] = rnd(value[0], 4)
                    if len(value) > 1 and isinstance(value[1], dict):
                        result["details"][name] = value[1]
                else:
                    result["metrics"][name] = rnd(value, 4)
                log(f"{name}: {result['metrics'].get(name)}")
            except Exception as exc:
                result["errors"][name] = f"{type(exc).__name__}: {exc}"[:300]
                log(f"{name} FAILED: {result['errors'][name]}")

        try:
            lse = compute_lse(out["face_crops"], audio, sr, device)
            result["metrics"]["lse_d"] = rnd(lse["lse_d"], 4)
            result["metrics"]["lse_c"] = rnd(lse["lse_c"], 4)
            result["metrics"]["av_offset"] = lse["av_offset"]
            result["details"]["lse"] = {"frames_used": lse["frames_used"]}
            log(f"lse_d: {result['metrics']['lse_d']}  lse_c: {result['metrics']['lse_c']}")
        except Exception as exc:
            result["errors"]["lse"] = f"{type(exc).__name__}: {exc}"[:300]
            log(f"lse FAILED: {result['errors']['lse']}")

        if not args.skip_fid:
            safe("fid", lambda: compute_fid(inp["fvd_frames"], out["fvd_frames"], device,
                                           limit=args.max_fid_frames))
        if not args.skip_fvd:
            safe("fvd", lambda: compute_fvd(inp["fvd_frames"], out["fvd_frames"], device))

        safe("csim", lambda: compute_csim(inp, out))

        if args.mode == "reconstruction":
            try:
                ssim, psnr, pa, pb, detail = compute_recon_metrics(inp, out)
                result["metrics"]["ssim"] = rnd(ssim, 4)
                result["metrics"]["psnr"] = rnd(psnr, 4)
                result["details"]["recon"] = detail
                log(f"ssim: {result['metrics']['ssim']}  psnr: {result['metrics']['psnr']}")
            except Exception as exc:
                result["errors"]["ssim_psnr"] = f"{type(exc).__name__}: {exc}"[:300]
                log(f"ssim/psnr FAILED: {result['errors']['ssim_psnr']}")
                pa = pb = None
            if pa and pb:
                safe("lpips", lambda: compute_lpips(pa, pb, device))
            try:
                lmd_mouth, lmd_all, lmd_detail = compute_lmd(inp, out)
                result["metrics"]["lmd_mouth"] = rnd(lmd_mouth, 5)
                result["metrics"]["lmd_all"] = rnd(lmd_all, 5)
                result["details"]["lmd"] = lmd_detail
                log(f"lmd_mouth: {result['metrics']['lmd_mouth']}")
            except Exception as exc:
                result["errors"]["lmd"] = f"{type(exc).__name__}: {exc}"[:300]
                log(f"lmd FAILED: {result['errors']['lmd']}")

        safe("niqe", lambda: compute_niqe(out["niqe_grays"]))
        if out["sharp_full"]:
            result["metrics"]["sharpness"] = rnd(float(np.mean(out["sharp_full"])), 2)
            result["metrics"]["mouth_sharpness"] = rnd(float(np.mean(out["sharp_mouth"])), 2) \
                if out["sharp_mouth"] else None
        result["metrics"]["mos"] = None
        result["metrics"]["mos_note"] = "manual 1-5 rating; not automated"

    except Exception as exc:
        result["errors"]["fatal"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-4000:]
        log(f"FATAL: {exc}")

    result["runtime_s"] = round(time.time() - started, 2)
    atomic_write_json(args.out_json, result)
    log(f"wrote {args.out_json} in {result['runtime_s']}s")
    if result["errors"].get("fatal"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
