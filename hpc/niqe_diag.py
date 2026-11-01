import os
import sys

import cv2
import numpy as np

sys.path.insert(0, "/home/aryannavale/projects")
import quality_metrics as qm

os.chdir("/home/aryannavale/projects/quality_assets")
import niqe_fixed

VIDEO = "/home/aryannavale/projects/outputs/bench-20261101-001509/wav2lip/recon-10s.mp4"

engine = qm.FaceEngine()
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
step = max(1, int(round(fps / 3.0)))
grays = []
vals = []
idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % step == 0:
        engine.detect(frame)
        mouth = qm.mouth_crop(frame, engine.last_bbox)
        if mouth.size:
            mg = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(mg, (256, 256), interpolation=cv2.INTER_AREA)
            grays.append(g)
            try:
                vals.append(float(niqe_fixed.niqe(g.astype(np.float64) / 255.0)))
            except Exception as exc:
                vals.append(float("nan"))
    idx += 1
cap.release()

vals = np.array(vals)
print("frames", len(grays), "valid", int(np.isfinite(vals).sum()))
print("min", float(np.nanmin(vals)), "median", float(np.nanmedian(vals)),
      "max", float(np.nanmax(vals)), "mean", float(np.nanmean(vals)))
print("first 10", [round(float(v), 3) for v in vals[:10]])
print("compute_niqe:", qm.compute_niqe(grays))
