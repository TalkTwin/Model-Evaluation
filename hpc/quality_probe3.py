#!/usr/bin/env python3
"""Probe FID/NIQE/I3D usability and pre-download weights on the login node."""
import os
import tempfile
import traceback

import cv2
import numpy as np
import torch

print("=== FID (downloads Inception weights) ===")
try:
    import torch_fidelity
    d1 = tempfile.mkdtemp(prefix="fid_a_")
    d2 = tempfile.mkdtemp(prefix="fid_b_")
    rng = np.random.default_rng(0)
    for i in range(8):
        cv2.imwrite(os.path.join(d1, f"{i:03d}.jpg"), rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        cv2.imwrite(os.path.join(d2, f"{i:03d}.jpg"), rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
    out = torch_fidelity.calculate_metrics(
        input1=d1, input2=d2, cuda=False, fid=True, kid=False, isc=False, verbose=False)
    print("FID result:", out)
except Exception:
    traceback.print_exc()

print("\n=== NIQE ===")
try:
    import niqe
    pkg = os.path.dirname(niqe.__file__)
    gray = np.random.default_rng(1).integers(0, 255, (256, 256), dtype=np.uint8)
    old = os.getcwd()
    os.chdir(pkg)
    try:
        print("niqe:", float(niqe.niqe(gray)))
    finally:
        os.chdir(old)
except Exception:
    traceback.print_exc()

print("\n=== I3D forward ===")
try:
    p = "/home/aryannavale/projects/LatentSync/checkpoints/auxiliary/i3d_torchscript.pt"
    m = torch.jit.load(p, map_location="cpu").eval()
    x = torch.rand(2, 3, 16, 224, 224)
    with torch.no_grad():
        y = m(x, rescale=False, resize=False, return_features=True)
    print("i3d features:", tuple(y.shape))
except Exception:
    traceback.print_exc()
