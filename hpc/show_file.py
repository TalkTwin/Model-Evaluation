import sys
p = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dryrun.log"
with open(p, "r", errors="replace") as f:
    text = f.read()
print(text[-5000:])
print("\n---END---")
