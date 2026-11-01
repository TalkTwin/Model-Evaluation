import hashlib
from pathlib import Path

base = Path("outputs/bench-20261031-152743/inputs")
for d in sorted(base.iterdir()):
    if not d.is_dir():
        continue
    for f in sorted(d.glob("*.mp4")):
        h = hashlib.md5(f.read_bytes()).hexdigest()[:12]
        print(f"{d.name:24} {f.name:24} {f.stat().st_size/1024/1024:8.2f} MB  {h}")
