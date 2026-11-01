import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/aryannavale/projects")
import quality_metrics as qm

os.chdir("/home/aryannavale/projects/quality_assets")
import niqe_fixed

VIDEOS = [
    "/home/aryannavale/projects/outputs/bench-20261101-001509/wav2lip/recon-10s.mp4",
    "/home/aryannavale/projects/outputs/bench-20261101-001509/heygem/recon-10s.mp4",
    "/home/aryannavale/projects/outputs/bench-20261101-001509/latentsync/recon-10s.mp4",
    "/home/aryannavale/projects/outputs/bench-20261031-143801/heygem/marathi-demo-2.mp4",
]

for video in VIDEOS:
    engine = qm.FaceEngine()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 3.0)))
    face_vals, mouth_vals = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            engine.detect(frame)
            face = qm.crop_face(frame, engine.last_bbox, 256, margin=0.35)
            fg = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            try:
                face_vals.append(float(niqe_fixed.niqe(fg.astype(np.float64) / 255.0)))
            except Exception:
                pass
            mouth = qm.mouth_crop(frame, engine.last_bbox)
            if mouth.size:
                mg = cv2.resize(cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY), (256, 256))
                try:
                    mouth_vals.append(float(niqe_fixed.niqe(mg.astype(np.float64) / 255.0)))
                except Exception:
                    pass
        idx += 1
    cap.release()
    name = video.split("/")[-2] + "/" + video.split("/")[-1]
    print("===", name)
    for label, vals in (("face", face_vals), ("mouth", mouth_vals)):
        a = np.array(vals)
        if a.size:
            print(f"  {label:>5}: n={a.size:3d} min={np.nanmin(a):8.3f} "
                  f"p50={np.nanmedian(a):8.3f} mean={np.nanmean(a):8.3f} max={np.nanmax(a):10.3f}")
