#!/usr/bin/env python3
"""Probe the quality-metric building blocks available in ~/envs/hygem-rtx50."""
import os
import sys
import traceback

LATENTSYNC = "/home/aryannavale/projects/LatentSync"
sys.path.insert(0, LATENTSYNC)

print("=== torch ===")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())

print("\n=== syncnet checkpoint ===")
try:
    from eval.syncnet.syncnet import S
    ckpt = os.path.join(LATENTSYNC, "checkpoints/auxiliary/syncnet_v2.model")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    print("type", type(obj))
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print("keys", len(keys), keys[:6])
        model = S()
        missing, unexpected = model.load_state_dict(obj, strict=False)
        print("missing", len(missing), missing[:5])
        print("unexpected", len(unexpected), unexpected[:5])
    elif isinstance(obj, torch.nn.Module):
        print("module class", type(obj).__name__)
except Exception:
    traceback.print_exc()

print("\n=== i3d torchscript ===")
try:
    p = os.path.join(LATENTSYNC, "checkpoints/auxiliary/i3d_torchscript.pt")
    m = torch.jit.load(p, map_location="cpu").eval()
    print("loaded i3d ok")
except Exception:
    traceback.print_exc()

print("\n=== niqe API ===")
try:
    import niqe
    print("module", niqe)
    print("attrs", [a for a in dir(niqe) if not a.startswith("_")][:20])
except Exception:
    traceback.print_exc()

print("\n=== insightface buffalo_l (downloads on first use) ===")
try:
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("models", list(app.models.keys()))
    for task, model in app.models.items():
        print(" ", task, type(model).__name__, getattr(model, "input_shape", None))
except Exception:
    traceback.print_exc()

print("\n=== lpips ===")
try:
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False)
    print("lpips alex ok")
except Exception:
    traceback.print_exc()

print("\n=== torch-fidelity ===")
try:
    import torch_fidelity
    print("torch_fidelity", torch_fidelity.__version__)
except Exception:
    traceback.print_exc()
