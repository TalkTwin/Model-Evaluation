#!/bin/bash
#SBATCH --job-name=bench-latentsync
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/aryannavale/projects/outputs/latentsync/bench_%j.out
#SBATCH --error=/home/aryannavale/projects/outputs/latentsync/bench_%j.err

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
PY=/home/aryannavale/envs/latentsync/bin/python
export PATH="$HOME/bin:$(dirname "$PY"):$PATH"

echo "===== BENCH LATENTSYNC ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
nvidia-smi -L

cd /home/aryannavale/projects/benchmarks

"$PY" -u benchmark_models.py \
  --model latentsync \
  --select all \
  --yes \
  --offline

echo "===== DONE $(date) ====="
