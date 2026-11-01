import os
import numpy as np
import niqe

os.chdir("/home/aryannavale/projects/quality_assets")
rng = np.random.default_rng(1)
g = rng.integers(0, 255, (256, 256), dtype=np.uint8)
score = float(niqe.niqe(g))
print("niqe noise", score)
g2 = np.full((256, 256), 128, dtype=np.uint8)
try:
    score2 = float(niqe.niqe(g2))
    print("niqe flat", score2)
except Exception as exc:
    print("niqe flat failed:", type(exc).__name__, exc)
