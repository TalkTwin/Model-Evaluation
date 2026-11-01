import re
from pathlib import Path

base = Path("reports")
for p in sorted(base.glob("*.html")):
    s = p.read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"#][^"]*)"', s)
    missing = []
    for ref in refs:
        if ref.startswith(("http:", "https:", "mailto:", "data:")):
            continue
        target = (p.parent / ref).resolve()
        if not target.exists():
            missing.append(ref)
    print(f"{p.name}: {len(refs)} local refs, missing={missing}")
