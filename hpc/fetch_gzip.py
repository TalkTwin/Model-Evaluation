import gzip
import shlex
import subprocess
import sys
from pathlib import Path

host = sys.argv[1]
remote = sys.argv[2]
local = Path(sys.argv[3])
timeout = int(sys.argv[4]) if len(sys.argv) > 4 else 600

cmd = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
       host, f"gzip -c {shlex.quote(remote)}"]
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                      timeout=timeout)
if proc.returncode != 0:
    print(proc.stderr.decode(errors="replace")[-500:])
    raise SystemExit(f"ssh rc={proc.returncode}")
data = gzip.decompress(proc.stdout)
local.parent.mkdir(parents=True, exist_ok=True)
local.write_bytes(data)
print(f"wrote {local} ({len(data)} bytes, gz {len(proc.stdout)})")
