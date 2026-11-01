import os
import sys

import cv2
import numpy as np

ASSETS = "/home/aryannavale/projects/quality_assets"
sys.path.insert(0, ASSETS)
os.chdir(ASSETS)
import niqe_fixed

CAP = "/home/aryannavale/projects/outputs/bench-20261101-001509/wav2lip/recon-10s.mp4"
cap = cv2.VideoCapture(CAP)
ok, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
mouth = gray[int(h * 0.55):int(h * 0.95), int(w * 0.30):int(w * 0.70)]
for name, img in (("full", gray), ("mouth", mouth)):
    for scale in (1.0, 255.0):
        val = float(niqe_fixed.niqe(img.astype(np.float64) / scale))
        print(f"{name:6} /{scale:g}: {val:.4f}")
