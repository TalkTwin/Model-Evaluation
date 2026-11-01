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
small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
print("full frame 256x256 /255:", float(niqe_fixed.niqe(small.astype(np.float64) / 255.0)))
print("full frame 256x256 /1  :", float(niqe_fixed.niqe(small.astype(np.float64))))
