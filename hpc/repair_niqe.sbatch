#!/bin/bash
#SBATCH --job-name=repair-niqe
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -uo pipefail

PY="$HOME/envs/hygem-rtx50/bin/python"
OUT="$HOME/projects/outputs"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
date

"$PY" -u "$HOME/projects/repair_niqe.py" \
    --outdir "$OUT/bench-20261031-152743" \
    --outdir "$OUT/bench-20261031-143801" \
    --outdir "$OUT/bench-20261101-001509"
rc=$?

echo "repair finished rc=$rc"
date
exit $rc
