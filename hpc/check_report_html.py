from pathlib import Path
import re

p = Path("outputs/bench-20261031-152743/quality_report.html")
s = p.read_text(encoding="utf-8")
print("size", len(s))
print("sections", len(re.findall(r"<section", s)))
print("videos", len(re.findall(r"<video", s)))
print("cards", len(re.findall(r'class="card"', s)))
print("nav links:")
for href, label in re.findall(r'<a href="#([^"]+)">([^<]+)</a>', s):
    print("  ", href, "->", label)
print("recon srcs", len(re.findall(r'src="recon/', s)))
print("long srcs", len(re.findall(r'src="\.\./bench-20261031-143801/', s)))
print("cross srcs", len(re.findall(r'src="\./', s)))
print("niqe values", len(re.findall(r">2\.\d+<", s)))
print("None cells", s.count(">None<"))
