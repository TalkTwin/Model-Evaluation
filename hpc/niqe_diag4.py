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
    "/home/aryannavale/projects/outputs/bench-20261101-001509/musetalk/recon-10s.mp4",
]

for video in VIDEOS:
    engine = qm.FaceEngine()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 3.0)))
    out = {1.0: [], 2.0: []}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            engine.detect(frame)
            for margin in (1.0, 2.0):
                crop = qm.crop_face(frame, engine.last_bbox, 256, margin=margin)
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                try:
                    out[margin].append(float(niqe_fixed.niqe(g.astype(np.float64) / 255.0)))
                except Exception:
                    pass
        idx += 1
    cap.release()
    name = video.split("/")[-2] + "/" + video.split("/")[-1]
    print("===", name)
    for margin, vals in out.items():
        a = np.array(vals)
        if a.size:
            print(f"  margin={margin}: n={a.size:3d} min={np.nanmin(a):8.3f} "
                  f"p50={np.nanmedian(a):8.3f} mean={np.nanmean(a):8.3f} max={np.nanmax(a):10.3f}")
