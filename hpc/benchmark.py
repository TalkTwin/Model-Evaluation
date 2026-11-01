#!/usr/bin/env python3
"""benchmark.py - interactive front-end for the talking-head benchmark.

Run this on the LOGIN NODE (it needs no GPU; it only asks questions and submits
Slurm jobs). All heavy work happens in benchmark_worker.py inside the job.

Flow
----
  1. which INPUT folders to use  (number list, range, or "all")
  2. which MODELS to run         (same syntax, includes "all")
  3. sequential (one after another) or parallel (all jobs at once)
  4. confirmation -> creates outputs/bench-<timestamp>/ and submits

Everything lands in ONE place:

  ~/projects/outputs/bench-20261031-014500/
  |-- plan.json                  what you chose + job ids
  |-- report.json / results.csv / report.md       (rollup, one schema)
  |-- logs/                      slurm %x_%j.out/.err
  |-- heygem/       <tag>.mp4  <tag>.json  <tag>.log
  |-- latentsync/   ...
  |-- musetalk/     ...
  |-- wav2lip/      ...

Non-interactive (for scripting):
  benchmark.py --inputs 1,2 --models all --mode parallel --yes
"""

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_worker import (  # noqa: E402
    DEFAULT_INPUTS, HOME, MODELS, atomic_write_json, log, scan_inputs, utc_now,
)

PROJECTS = HOME / "projects"
OUTPUTS = PROJECTS / "outputs"

MODEL_LABEL = {
    "heygem": "HeyGem      (RTX-50 fork)      env hygem-rtx50  torch 2.7.0+cu128",
    "latentsync": "LatentSync  (ByteDance)       env latentsync   torch 2.7.1+cu128",
    "musetalk": "MuseTalk    (v1.5)             env musetalk     torch 2.7.1+cu128",
    "wav2lip": "Wav2Lip     (wav2lip_gan)       env latentsync   torch 2.7.1+cu128",
}

# resource profile per model (single-run job)
PROFILE = {
    "heygem":     {"cpus": 4, "mem": "16G", "time": "02:00:00"},
    "latentsync": {"cpus": 8, "mem": "32G", "time": "03:00:00"},
    "musetalk":   {"cpus": 4, "mem": "16G", "time": "02:00:00"},
    "wav2lip":    {"cpus": 4, "mem": "16G", "time": "02:00:00"},
}

SEQUENTIAL_TIME = "08:00:00"
NODE = "aicoeserver05"
GRES = "gpu:1g.24gb:1"


# --------------------------------------------------------------------------- #
# selection parsing
# --------------------------------------------------------------------------- #
def parse_selection(text, count):
    """Accept 'all', '1', '1,3', '1-3', '1 2 3', '1,2-4'. Return sorted 1-based list."""
    text = (text or "").strip().lower()
    if text in ("all", "a", "*"):
        return list(range(1, count + 1))
    if not text:
        return []
    picked = set()
    for chunk in text.replace(";", ",").replace(" ", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            if not (lo.strip().isdigit() and hi.strip().isdigit()):
                raise ValueError(f"bad range: {chunk}")
            lo, hi = int(lo), int(hi)
            if lo > hi:
                lo, hi = hi, lo
            picked.update(range(lo, hi + 1))
        else:
            if not chunk.isdigit():
                raise ValueError(f"bad number: {chunk}")
            picked.add(int(chunk))
    bad = [p for p in picked if p < 1 or p > count]
    if bad:
        raise ValueError(f"out of range: {bad} (1-{count})")
    return sorted(picked)


def ask_selection(prompt, count, default_all=False):
    while True:
        try:
            raw = input(prompt).strip()
        except EOFError:
            raise SystemExit("\nno TTY - pass --inputs/--models/--mode on the command line")
        if not raw and default_all:
            return list(range(1, count + 1))
        try:
            picked = parse_selection(raw, count)
            if picked:
                return picked
            print("  ! kuch select nahi kiya, dobara try karo (ya 'all')")
        except ValueError as exc:
            print(f"  ! {exc}")


# --------------------------------------------------------------------------- #
# plan building
# --------------------------------------------------------------------------- #
def build_plan(pairs, models, mode, outdir, inputs_dir=None):
    runs, index = [], 0
    for model in models:                    # model-major keeps heygem runs apart
        for pair in pairs:
            text = f"{pair['tag']} {pair['folder']}".lower()
            test_mode = "reconstruction" if "recon" in text else "cross_audio"
            runs.append({
                "index": index,
                "model": model,
                "tag": pair["tag"],
                "folder": pair["folder"],
                "video": pair["video"],
                "audio": pair["audio"],
                "pair_index": pair["pair_index"],
                "test_mode": test_mode,
                "status": "pending",
            })
            index += 1
    plan = {
        "created": utc_now(),
        "created_host": socket.gethostname(),
        "created_by": os.environ.get("USER") or "unknown",
        "outdir": str(outdir),
        "inputs_dir": str(inputs_dir or DEFAULT_INPUTS),
        "mode": mode,
        "quality": True,
        "models": models,
        "pairs": pairs,
        "runs": runs,
        "jobs": [],
    }
    return plan


def show_summary(pairs, models, mode, outdir):
    print("\n" + "=" * 74)
    print("RUN PLAN")
    print("=" * 74)
    print("inputs:")
    for pair in pairs:
        print(f"  - {pair['tag']:<16} video: {Path(pair['video']).name}")
        print(f"  {'':<18} audio: {Path(pair['audio']).name}")
    print("models:")
    for m in models:
        print(f"  - {MODEL_LABEL[m]}")
    print(f"mode:      {mode}")
    total = len(pairs) * len(models)
    print(f"total runs: {total}")
    print(f"output:    {outdir}")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# submission
# --------------------------------------------------------------------------- #
def submit(plan, sbatch_file, dry_run=False):
    outdir = Path(plan["outdir"])
    logs = outdir / "logs"
    if not dry_run:
        logs.mkdir(parents=True, exist_ok=True)
    jobs = []

    def run_sbatch(args, label):
        cmd = ["sbatch", "--parsable"] + args + [str(sbatch_file)]
        if dry_run:
            print("DRY-RUN: " + " ".join(cmd))
            return f"dry-{label}"
        res = subprocess.run(cmd, capture_output=True, text=True)
        job_id = (res.stdout or "").strip().split(";")[0].strip()
        if res.returncode != 0 or not job_id:
            raise SystemExit(f"sbatch failed for {label}: "
                             f"{(res.stderr or res.stdout).strip()}")
        print(f"  submitted {label:<34} -> job {job_id}")
        return job_id

    common = [
        "--partition=gpu", f"--nodelist={NODE}", f"--gres={GRES}",
        f"--output={logs}/%x_%j.out", f"--error={logs}/%x_%j.err",
    ]

    if plan["mode"] == "sequential":
        job_id = run_sbatch(
            common + [
                "--job-name=bench-seq",
                "--cpus-per-task=8", "--mem=32G", f"--time={SEQUENTIAL_TIME}",
                f"--export=ALL,PLAN={outdir}/plan.json,RUN_MODE=all,RUN_INDEX=",
            ],
            "sequential (all runs in one job)")
        jobs.append({"label": "sequential", "job_id": job_id,
                     "runs": [r["index"] for r in plan["runs"]]})
    else:
        prev_heygem = None
        for run in plan["runs"]:
            prof = PROFILE[run["model"]]
            extra = []
            # HeyGem always writes the same filename -> never let two collide
            if run["model"] == "heygem" and prev_heygem:
                extra.append(f"--dependency=afterany:{prev_heygem}")
            job_id = run_sbatch(
                common + [
                    f"--job-name=bench-{run['model']}-{run['tag']}",
                    f"--cpus-per-task={prof['cpus']}", f"--mem={prof['mem']}",
                    f"--time={prof['time']}",
                    f"--export=ALL,PLAN={outdir}/plan.json,RUN_MODE=single,"
                    f"RUN_INDEX={run['index']}",
                ] + extra,
                f"{run['model']}/{run['tag']}"
                + (" (after previous heygem)" if extra else ""))
            jobs.append({"label": f"{run['model']}/{run['tag']}",
                         "job_id": job_id, "runs": [run["index"]]})
            if run["model"] == "heygem":
                prev_heygem = job_id

    plan["jobs"] = jobs
    atomic_write_json(outdir / "plan.json", plan)
    return jobs


def print_next_steps(plan):
    outdir = plan["outdir"]
    print("\n" + "=" * 74)
    print("DONE - jobs submitted")
    print("=" * 74)
    print(f"output dir : {outdir}")
    print("monitor    : squeue -u $USER")
    for job in plan["jobs"]:
        print(f"             sacct -j {job['job_id']} --format=JobID,State,Elapsed,ExitCode")
    print(f"logs       : tail -f {outdir}/logs/bench-*.out")
    print(f"report     : cat {outdir}/report.md")
    print(f"rebuild    : ~/envs/hygem-rtx50/bin/python ~/projects/benchmark_worker.py "
          f"--plan {outdir}/plan.json --report")
    print("=" * 74)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global SEQUENTIAL_TIME

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", help='number list / range / "all" (folders in ~/projects/inputs)')
    ap.add_argument("--models", help='number list / range / "all"')
    ap.add_argument("--mode", choices=["sequential", "parallel"])
    ap.add_argument("--time", help="override --time for sequential mode (HH:MM:SS)")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    ap.add_argument("--dry-run", action="store_true", help="print sbatch commands only")
    ap.add_argument("--inputs-dir", default=str(DEFAULT_INPUTS))
    args = ap.parse_args()

    inputs_dir = Path(args.inputs_dir)
    if args.time:
        SEQUENTIAL_TIME = args.time

    print("\n" + "=" * 74)
    print("TALKING-HEAD BENCHMARK  -  HeyGem / LatentSync / MuseTalk / Wav2Lip")
    print("=" * 74)

    pairs_all = scan_inputs(inputs_dir)
    if not pairs_all:
        raise SystemExit(f"no input pairs found in {inputs_dir}")
    if not Path(HOME / "envs" / "hygem-rtx50" / "bin" / "python").exists():
        raise SystemExit("hygem-rtx50 env not found")

    # ---- 1. inputs -------------------------------------------------------- #
    print("\nSTEP 1/3  -  kaun se inputs?\n")
    for i, pair in enumerate(pairs_all, start=1):
        print(f"  [{i}] {pair['tag']}")
        print(f"      video: {Path(pair['video']).name}")
        print(f"      audio: {Path(pair['audio']).name}")
    if args.inputs is not None:
        idx = parse_selection(args.inputs, len(pairs_all))
    else:
        idx = ask_selection(
            f"\nselect inputs (1-{len(pairs_all)}, e.g. 1 / 1,2 / 1-2 / all): ",
            len(pairs_all), default_all=True)
    pairs = [pairs_all[i - 1] for i in idx]

    # ---- 2. models -------------------------------------------------------- #
    print("\nSTEP 2/3  -  kaun se models?\n")
    for i, model in enumerate(MODELS, start=1):
        print(f"  [{i}] {MODEL_LABEL[model]}")
    if args.models is not None:
        midx = parse_selection(args.models, len(MODELS))
    else:
        midx = ask_selection(
            f"\nselect models (1-{len(MODELS)}, e.g. 1 / 1,3 / all): ",
            len(MODELS), default_all=True)
    models = [MODELS[i - 1] for i in midx]

    # ---- 3. mode ---------------------------------------------------------- #
    print("\nSTEP 3/3  -  kaise chalao?\n")
    print("  [1] ek ke baad ek  (sequential - ek job, ek hi GPU slot, 8h)")
    print("  [2] sab ek saath   (parallel - alag job per run, 8 MIG slices)")
    if args.mode:
        mode = args.mode
    else:
        while True:
            try:
                raw = input("\nselect mode (1/2) [1]: ").strip() or "1"
            except EOFError:
                raw = "1"
            if raw in ("1", "sequential", "seq"):
                mode = "sequential"
                break
            if raw in ("2", "parallel", "par"):
                mode = "parallel"
                break
            print("  ! 1 ya 2")

    # ---- plan ------------------------------------------------------------- #
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = OUTPUTS / f"bench-{stamp}"
    plan = build_plan(pairs, models, mode, outdir, inputs_dir)
    plan["sequential_time"] = SEQUENTIAL_TIME

    show_summary(pairs, models, mode, outdir)

    sbatch_file = Path(__file__).resolve().parent / "benchmark.sbatch"
    if not sbatch_file.exists():
        raise SystemExit(f"missing {sbatch_file}")

    if args.dry_run:
        print("\nDRY RUN - nothing written, nothing submitted\n")
        submit(plan, sbatch_file, dry_run=True)
        return

    if not args.yes:
        try:
            ok = input("start? (y/N): ").strip().lower()
        except EOFError:
            ok = "n"
        if ok not in ("y", "yes"):
            print("cancelled - kuch submit nahi kiya.")
            return

    for model in models:
        (outdir / model).mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    atomic_write_json(outdir / "plan.json", plan)
    log(f"plan written: {outdir}/plan.json  ({len(plan['runs'])} runs)")

    print()
    submit(plan, sbatch_file, dry_run=False)
    print_next_steps(plan)


if __name__ == "__main__":
    main()
