import re
from pathlib import Path

base = Path("reports")
for p in sorted(base.glob("*.html")):
    s = p.read_text(encoding="utf-8")
    srcs = re.findall(r'<(?:video|source)[^>]+src="([^"]+)"', s)
    missing = []
    for src in srcs:
        target = (p.parent / src).resolve()
        if not target.exists():
            missing.append(src)
    print(f"{p.name}: {len(srcs)} sources, {len(set(srcs))} unique, missing={missing}")
    if missing:
        raise SystemExit(1)
print("all video sources resolve")
