#!/usr/bin/env python3
"""Second probe: SyncNet checkpoint, insightface landmarks, niqe signature."""
import importlib.util
import inspect
import os
import traceback

import cv2
import numpy as np
import torch

LS = "/home/aryannavale/projects/LatentSync"

print("=== syncnet direct import ===")
try:
    spec = importlib.util.spec_from_file_location(
        "ls_syncnet", os.path.join(LS, "eval/syncnet/syncnet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    S = mod.S
    obj = torch.load(os.path.join(LS, "checkpoints/auxiliary/syncnet_v2.model"),
                     map_location="cpu", weights_only=False)
    print("checkpoint type:", type(obj))
    if isinstance(obj, dict):
        print("keys:", len(obj), list(obj.keys())[:5])
        model = S()
        missing, unexpected = model.load_state_dict(obj, strict=False)
        print("missing:", len(missing), missing[:8])
        print("unexpected:", len(unexpected), unexpected[:8])
        model.eval()
        with torch.no_grad():
            out = model.forward_lip(torch.zeros(2, 3, 5, 224, 224))
            out2 = model.forward_aud(torch.zeros(2, 1, 13, 20))
        print("forward_lip:", tuple(out.shape), "forward_aud:", tuple(out2.shape))
except Exception:
    traceback.print_exc()

print("\n=== insightface on real frame ===")
try:
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    cap = cv2.VideoCapture("/home/aryannavale/projects/inputs/marathi-demo-2/marathi-demo-2.mp4")
    ok, frame = cap.read()
    cap.release()
    print("frame:", frame.shape if ok else None)
    faces = app.get(frame)
    print("faces:", len(faces))
    if faces:
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        print("bbox:", f.bbox)
        print("kps:", getattr(f, "kps", None))
        print("landmark_3d_68:", getattr(f, "landmark_3d_68", None).shape
              if getattr(f, "landmark_3d_68", None) is not None else None)
        print("landmark_2d_106:", getattr(f, "landmark_2d_106", None).shape
              if getattr(f, "landmark_2d_106", None) is not None else None)
        print("embedding:", f.normed_embedding.shape, float(np.linalg.norm(f.normed_embedding)))
except Exception:
    traceback.print_exc()

print("\n=== niqe signature ===")
try:
    import niqe
    print(inspect.signature(niqe.niqe))
    print(inspect.getsource(niqe.main)[:1200])
except Exception:
    traceback.print_exc()
