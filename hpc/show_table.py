import sys
from pathlib import Path

path = Path(sys.argv[1])
needle = sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 1
s = path.read_text(encoding="utf-8")
start = 0
for n in range(count):
    i = s.find(needle, start)
    if i < 0:
        print(f"[not found after {n}]")
        break
    end = s.find("</table>", i)
    print(s[max(0, i - 60):end + 9])
    print("\n" + "=" * 80 + "\n")
    start = end + 9
