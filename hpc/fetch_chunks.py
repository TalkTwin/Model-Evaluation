"""Fetch a remote file through a low-MTU link in tiny chunks.

The VPN path drops packets larger than ~1300 bytes; a normal transfer stalls
once TCP sends a full-size segment.  Each chunk is fetched over its own short
SSH connection so no connection ever sends a large burst.
"""
import gzip
import subprocess
import sys
from pathlib import Path

HOST = sys.argv[1]
REMOTE = sys.argv[2]           # remote .gz file
LOCAL = Path(sys.argv[3])
CHUNK = int(sys.argv[4]) if len(sys.argv) > 4 else 700
TIMEOUT = int(sys.argv[5]) if len(sys.argv) > 5 else 40
TRIES = int(sys.argv[6]) if len(sys.argv) > 6 else 4


def ssh_capture(cmd, timeout=TIMEOUT):
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
         "-o", "Compression=no", HOST, cmd],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="replace")[-300:])
    return proc.stdout


size = int(ssh_capture(f"stat -c %s {REMOTE}").decode().strip())
print(f"remote gz size: {size} bytes, chunk={CHUNK}", flush=True)

parts = []
offset = 0
index = 0
while offset < size:
    want = min(CHUNK, size - offset)
    index += 1
    for attempt in range(1, TRIES + 1):
        try:
            data = ssh_capture(
                f"dd if={REMOTE} bs={CHUNK} skip={index - 1} count=1 2>/dev/null")
            if len(data) == want:
                parts.append(data)
                offset += len(data)
                print(f"chunk {index}: {len(data)}/{want} ok "
                      f"({offset}/{size})", flush=True)
                break
            print(f"chunk {index}: got {len(data)}/{want}, retry {attempt}",
                  flush=True)
        except subprocess.TimeoutExpired:
            print(f"chunk {index}: timeout, retry {attempt}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"chunk {index}: {exc}, retry {attempt}", flush=True)
    else:
        raise SystemExit(f"chunk {index} failed after {TRIES} tries")

blob = b"".join(parts)
if len(blob) != size:
    raise SystemExit(f"size mismatch: {len(blob)} != {size}")
data = gzip.decompress(blob)
LOCAL.parent.mkdir(parents=True, exist_ok=True)
LOCAL.write_bytes(data)
print(f"wrote {LOCAL} ({len(data)} bytes)")
