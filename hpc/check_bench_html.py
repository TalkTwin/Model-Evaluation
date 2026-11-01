import re
import sys
from pathlib import Path

p = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/bench-20261031-152743/benchmark_report.html")
s = p.read_text(encoding="utf-8")
print("global table CSS:", bool(re.search(r"(?m)^\s*table\s*\{", s)))
print("stats tables:", s.count('class="stats-table"'))
print("group-row:", s.count("group-row"))
print("duration-row:", s.count("duration-row"))
print("model-cell rows:", s.count('class="model-cell"'))
print("best cells:", len(re.findall(r'class="[^"]*\bbest\b', s)))
print("videos:", len(re.findall(r"<source[^>]+src=", s)))
print("legends:", s.count('class="table-legend"'))
for m in re.finditer(r"<table[^>]*>", s):
    print(m.group(0))
