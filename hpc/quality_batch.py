#!/usr/bin/env python3
"""quality_batch.py - compute quality metrics for all completed runs in an outdir.

Run one job per model (each gets its own MIG slice):

  python quality_batch.py --outdir ~/projects/outputs/bench-... --model wav2lip
  python quality_batch.py --outdir ~/projects/outputs/bench-... --model all
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_worker import (  # noqa: E402
    HOME, MODEL_PROJECT, QUALITY_PY, QUALITY_SCRIPT, infer_test_mode,
    log, refresh_report,
)

OUTPUTS = HOME / "projects" / "outputs"


def load_records(outdir, model=None):
    outdir = Path(outdir)
    recs = []
    for js in sorted(outdir.glob("*/*.json")):
        if js.parent.name.startswith("_"):
            continue
        if model and js.parent.name != model:
            continue
        try:
            rec = json.loads(js.read_text())
        except Exception:
            continue
        if rec.get("status") != "COMPLETED":
            continue
        out = (rec.get("output") or {}).get("path")
        if not out or not Path(out).exists():
            continue
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-fid", action="store_true")
    ap.add_argument("--skip-fvd", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    models = None if args.model == "all" else [args.model]
    recs = load_records(outdir, None)
    if models:
        recs = [r for r in recs if r.get("model") in models]
    if args.limit:
        recs = recs[:args.limit]

    log(f"{len(recs)} completed run(s) in {outdir}")
    done = failed = skipped = 0
    for rec in recs:
        model, tag = rec["model"], rec["tag"]
        qdir = outdir / model / "_quality"
        qdir.mkdir(parents=True, exist_ok=True)
        qjson = qdir / f"{tag}.json"
        qlog = qdir / f"{tag}.log"
        if qjson.exists() and not args.force:
            log(f"skip {model}/{tag}: {qjson.name} exists")
            skipped += 1
            continue

        mode = rec.get("test_mode") or infer_test_mode(rec)
        cmd = [str(QUALITY_PY), "-u", str(QUALITY_SCRIPT),
               "--input-video", str((rec.get("input") or {}).get("video", {}).get("path")),
               "--output-video", str((rec.get("output") or {}).get("path")),
               "--audio", str((rec.get("input") or {}).get("audio", {}).get("path")),
               "--mode", mode,
               "--out-json", str(qjson),
               "--device", "cuda"]
        if args.skip_fid:
            cmd.append("--skip-fid")
        if args.skip_fvd:
            cmd.append("--skip-fvd")
        log("run " + " ".join(cmd))
        started = time.time()
        with open(qlog, "wb") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL)
        if qjson.exists():
            done += 1
            log(f"ok {model}/{tag} rc={proc.returncode} in {time.time() - started:.1f}s")
        else:
            failed += 1
            log(f"FAIL {model}/{tag} rc={proc.returncode} (see {qlog})")

    try:
        refresh_report(outdir)
    except Exception as exc:  # noqa: BLE001
        log(f"refresh_report failed: {exc}")
    log(f"DONE done={done} skipped={skipped} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
