#!/bin/bash
#SBATCH --job-name=bench-heygem
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/aryannavale/projects/outputs/heygem/bench_%j.out
#SBATCH --error=/home/aryannavale/projects/outputs/heygem/bench_%j.err

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
PY=/home/aryannavale/envs/hygem-rtx50/bin/python
export PATH="$HOME/bin:$(dirname "$PY"):$PATH"

echo "===== BENCH HEYGEM ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
nvidia-smi -L

# HeyGem's onnxruntime needs the nvidia libs (cufft etc.) from its env.
# env.sh appends to $LD_LIBRARY_PATH, so make sure `set -u` cannot trip on it.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/aryannavale/projects/hygem-rtx50
source env.sh

cd /home/aryannavale/projects/benchmarks

"$PY" -u benchmark_models.py \
  --model heygem \
  --select all \
  --yes \
  --offline

echo "===== DONE $(date) ====="
