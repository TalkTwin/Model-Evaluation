import os
import sys

import cv2
import numpy as np

ASSETS = "/home/aryannavale/projects/quality_assets"
sys.path.insert(0, ASSETS)
os.chdir(ASSETS)
import niqe_fixed

cap = cv2.VideoCapture("/home/aryannavale/projects/inputs/marathi-demo-2/marathi-demo-2.mp4")
ok, frame = cap.read()
cap.release()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
print("shape", gray.shape, "range", gray.min(), gray.max())
try:
    print("niqe real frame", float(niqe_fixed.niqe(gray)))
except Exception as exc:
    print("failed:", type(exc).__name__, exc)
h, w = gray.shape
crop = gray[int(h * 0.35):int(h * 0.75), int(w * 0.35):int(w * 0.75)]
try:
    print("niqe face crop", float(niqe_fixed.niqe(crop)))
except Exception as exc:
    print("crop failed:", type(exc).__name__, exc)
