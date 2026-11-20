#!/usr/bin/env python3
"""Sanity-check wav2lip_report.html: structure, links, assets, leftovers."""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "wav2lip_report.html"

html = PAGE.read_text(encoding="utf-8")
errors = []
warns = []

print(f"bytes={len(html.encode('utf-8')):,}")
print(f"videos={html.count('<video')}")
print(f"gallery_cards={html.count('class=\"clip\"') + html.count('class=\"clip small\"')}")
print(f"tables={html.count('<table')} table_rows={html.count('<tr>')}")
print(f"section_open={html.count('<section')} section_close={html.count('</section>')}")
print(f"div_open={html.count('<div')} div_close={html.count('</div>')}")

# video sources
srcs = re.findall(r'<video[^>]+src="([^"]+)"', html)
missing = sorted({s for s in srcs if not (ROOT / s).exists()})
print(f"video_srcs={len(srcs)} unique={len(set(srcs))} missing={missing}")
if missing:
    errors.append(f"missing videos: {missing}")

# local links / assets
hrefs = re.findall(r'(?:href|src)="([^"#][^"]*)"', html)
local = sorted({h for h in hrefs if not re.match(r'^(https?:|data:|mailto:)', h)})
missing_links = [h for h in local if not (ROOT / h.split("?")[0]).exists()]
print(f"local_refs={len(local)} missing={missing_links}")
if missing_links:
    errors.append(f"missing local refs: {missing_links}")

# leftover template braces (but ignore CSS/JS braces)
tmpl = re.findall(r'\{[a-z_][a-z0-9_]*(?:\([^)]*\))?\}', html)
print(f"unresolved_template={tmpl[:10]}")
if tmpl:
    errors.append(f"unresolved template placeholders: {tmpl[:5]}")

for bad in ("None", "nan", "NaN", "— s"):
    hits = html.count(bad)
    if hits:
        warns.append(f"found {bad!r} x{hits}")

if html.count("<section") != html.count("</section>"):
    errors.append("unbalanced <section>")
if html.count("<div") != html.count("</div>"):
    errors.append("unbalanced <div>")

# expected counts
if html.count("<video") != 15:
    warns.append(f"expected 15 <video> tags, got {html.count('<video')}")

print("\nERRORS:" if errors else "\nERRORS: none")
for e in errors:
    print(f"  - {e}")
print("WARNINGS:" if warns else "WARNINGS: none")
for w in warns:
    print(f"  - {w}")

sys.exit(1 if errors else 0)
