import json
from pathlib import Path

FOLDERS = [
    Path("/home/aryannavale/projects/outputs/bench-20261031-152743"),
    Path("/home/aryannavale/projects/outputs/bench-20261031-143801"),
    Path("/home/aryannavale/projects/outputs/bench-20261101-001509"),
]

bad = 0
for folder in FOLDERS:
    print(f"=== {folder.name} ===")
    for p in sorted(folder.glob("*/_quality/*.json")):
        d = json.loads(p.read_text())
        m = d.get("metrics") or {}
        val = m.get("niqe")
        flag = ""
        try:
            if val is None or not (1.0 <= float(val) <= 6.0):
                flag = "  <-- CHECK"
                bad += 1
        except Exception:
            flag = "  <-- BAD"
            bad += 1
        print(f"  {p.parent.parent.name:12} {p.stem:20} niqe={val}{flag}")
print(f"suspicious: {bad}")
