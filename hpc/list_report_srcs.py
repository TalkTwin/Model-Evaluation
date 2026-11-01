import re
from pathlib import Path

base = Path("outputs/bench-20261031-152743")
for name in ("benchmark_report.html", "quality_report.html"):
    p = base / name
    s = p.read_text(encoding="utf-8")
    srcs = re.findall(r'<(?:video|source)[^>]+src="([^"]+)"', s)
    print(f"=== {name}: {len(srcs)} sources ===")
    for src in srcs:
        target = (p.parent / src).resolve()
        print(f"{'OK ' if target.exists() else 'MISS'} {src}")
