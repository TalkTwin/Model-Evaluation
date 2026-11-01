from pathlib import Path
import shutil

p = Path("/home/aryannavale/projects/LatentSync/latentsync/utils/util.py")
src = p.read_text()
old = (
    '        temp_dir = "temp"\n'
    '        if os.path.exists(temp_dir):\n'
    '            shutil.rmtree(temp_dir)\n'
    '        os.makedirs(temp_dir, exist_ok=True)\n'
)
new = (
    '        # Benchmark patch: a shared relative "temp" dir races when several\n'
    '        # jobs run in parallel; its NFS cleanup also fails on .nfsXXXX.\n'
    '        import tempfile\n'
    '        temp_dir = tempfile.mkdtemp(prefix="latentsync_frames_")\n'
)
if new in src:
    print("already patched")
elif old in src:
    bak = p.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copy2(p, bak)
    p.write_text(src.replace(old, new, 1))
    print("patched", p, "backup:", bak)
else:
    raise SystemExit("pattern not found; file may have changed")
