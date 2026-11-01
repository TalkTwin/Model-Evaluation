import sys
import time

path = sys.argv[1]
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
delay = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
out = sys.stdout.buffer
with open(path, "rb") as f:
    while True:
        data = f.read(chunk)
        if not data:
            break
        out.write(data)
        out.flush()
        time.sleep(delay)
