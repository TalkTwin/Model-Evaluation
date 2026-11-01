import json
import sys

plan = json.loads(open(sys.argv[1]).read())
for r in plan["runs"]:
    print(r["index"], r["model"], r["tag"], r.get("test_mode"))
