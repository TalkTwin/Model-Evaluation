#!/bin/bash
#SBATCH --job-name=bench
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Submitted by benchmark.py with:
#   --export=ALL,PLAN=<outdir>/plan.json,RUN_MODE=all|single,RUN_INDEX=<n>
# Resource lines above are defaults; benchmark.py overrides them per model.

set -uo pipefail

PLAN="${PLAN:?PLAN env not set}"
RUN_MODE="${RUN_MODE:-single}"
RUN_INDEX="${RUN_INDEX:-}"
OUTDIR="$(dirname "$PLAN")"
PY="$HOME/envs/hygem-rtx50/bin/python"
WORKER="$HOME/projects/benchmark_worker.py"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "PLAN: $PLAN"
echo "RUN_MODE: $RUN_MODE  RUN_INDEX: ${RUN_INDEX:-none}"
date

mkdir -p "$OUTDIR/logs"

test -r "$PLAN" || { echo "PLAN not readable: $PLAN"; exit 2; }
test -r "$WORKER" || { echo "WORKER not readable: $WORKER"; exit 2; }
test -x "$PY" || { echo "python not executable: $PY"; exit 2; }

if [ "$RUN_MODE" = "all" ]; then
  exec "$PY" -u "$WORKER" --plan "$PLAN" --all
else
  [ -n "$RUN_INDEX" ] || { echo "RUN_INDEX required for single mode"; exit 2; }
  exec "$PY" -u "$WORKER" --plan "$PLAN" --run-index "$RUN_INDEX"
fi
