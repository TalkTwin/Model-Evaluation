#!/usr/bin/env python3
"""Run the multilingual Wav2Lip benchmark by reusing benchmark_worker.py.

The shared worker already measures wall time, RTF, GPU memory, CPU RSS and
writes one JSON per run.  Wav2Lip needs three extra flags for static images,
so this wrapper extends build_command() without touching the shared file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "projects"))
import benchmark_worker as bw  # noqa: E402

_orig_build_command = bw.build_command


def build_command(model, run, outdir):
    cmd, cwd, env, hint, roots, select_index = _orig_build_command(model, run, outdir)
    if model == "wav2lip":
        if run.get("static"):
            cmd += ["--static", "True"]
        if run.get("fps"):
            cmd += ["--fps", str(run["fps"])]
        if run.get("pads"):
            cmd += ["--pads"] + [str(v) for v in run["pads"]]
    return cmd, cwd, env, hint, roots, select_index


bw.build_command = build_command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--run-index", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    outdir = Path(plan["outdir"])
    if args.report:
        bw.refresh_report(outdir)
        return

    runs = plan["runs"]
    if args.all:
        targets = runs
    elif args.run_index is not None:
        targets = [r for r in runs if r["index"] == args.run_index]
        if not targets:
            raise SystemExit(f"no run with index {args.run_index}")
    else:
        raise SystemExit("give --all, --run-index N or --report")

    failures = 0
    for run in targets:
        try:
            rec = bw.run_one(plan, run)
            if rec["status"] != "COMPLETED":
                failures += 1
        except SystemExit as exc:
            print(f"ABORT run #{run['index']}: {exc}", flush=True)
            failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR run #{run['index']}: {type(exc).__name__}: {exc}",
                  flush=True)
            failures += 1
        finally:
            try:
                bw.refresh_report(outdir)
            except Exception as exc:  # noqa: BLE001
                print(f"report refresh failed: {exc}", flush=True)

    print(f"DONE: {len(targets) - failures}/{len(targets)} completed", flush=True)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
