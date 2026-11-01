import os
import sys

import cv2
import numpy as np

ASSETS = "/home/aryannavale/projects/quality_assets"
sys.path.insert(0, ASSETS)
os.chdir(ASSETS)
import niqe_fixed

VIDEOS = [
    "/home/aryannavale/projects/outputs/bench-20261101-001509/wav2lip/recon-10s.mp4",
    "/home/aryannavale/projects/outputs/bench-20261101-001509/heygem/recon-10s.mp4",
    "/home/aryannavale/projects/outputs/bench-20261031-143801/heygem/marathi-demo-2.mp4",
]

for video in VIDEOS:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / 3.0)))
    scores = {256: [], 512: [], "native": []}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for key, size in ((256, (256, 256)), (512, (512, 512))):
                im = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
                try:
                    scores[key].append(float(niqe_fixed.niqe(im.astype(np.float64) / 255.0)))
                except Exception:
                    pass
            try:
                scores["native"].append(float(niqe_fixed.niqe(gray.astype(np.float64) / 255.0)))
            except Exception:
                pass
        idx += 1
    cap.release()
    print("===", video.split("/")[-2] + "/" + video.split("/")[-1])
    for key, vals in scores.items():
        a = np.array(vals)
        if a.size:
            print(f"  {str(key):>6}: n={a.size:3d} min={np.nanmin(a):8.3f} "
                  f"p50={np.nanmedian(a):8.3f} mean={np.nanmean(a):8.3f} max={np.nanmax(a):10.3f}")
