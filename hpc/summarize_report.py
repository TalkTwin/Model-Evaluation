import json
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "report.json"
data = json.loads(p.read_text())
print(f"{'model':12} {'tag':10} {'status':10} {'wall':>7} {'rtf':>7} {'dur':>6} {'quality':>8} {'lse_d':>7} {'csim':>6} {'errors'}")
for r in data.get("records", []):
    q = r.get("quality") or {}
    m = q.get("metrics") or {}
    errs = q.get("errors") or {}
    print(f"{r.get('model',''):12} {r.get('tag',''):10} {r.get('status',''):10} "
          f"{str(r.get('wall_s')):>7} {str(r.get('rtf')):>7} "
          f"{str((r.get('output') or {}).get('duration_s')):>6} "
          f"{'yes' if q else 'no':>8} {str(m.get('lse_d')):>7} {str(m.get('csim')):>6} "
          f"{','.join(errs) if errs else '-'}")
