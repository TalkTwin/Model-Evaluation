from pathlib import Path

p = Path("outputs/bench-20261031-152743/benchmark_report.html")
s = p.read_text(encoding="utf-8")
needle = '<tr class="group-row"><th colspan="2">Performance</th></tr>'
count = s.count(needle)
print("performance groups found:", count)
if count != 16:
    raise SystemExit("unexpected stats table count")
s = s.replace(needle, '<table class="stats-table"><tbody>' + needle)
p.write_text(s, encoding="utf-8")
print("repaired", p)
