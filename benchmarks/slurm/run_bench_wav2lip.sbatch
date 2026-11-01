#!/bin/bash
#SBATCH --job-name=bench-wav2lip
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/aryannavale/projects/outputs/wav2lip/bench_%j.out
#SBATCH --error=/home/aryannavale/projects/outputs/wav2lip/bench_%j.err

set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/aryannavale/envs/latentsync/bin/python
export PATH="$HOME/bin:$(dirname "$PY"):$PATH"

echo "===== BENCH WAV2LIP ====="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NOT_SET}"
nvidia-smi -L

cd /home/aryannavale/projects/benchmarks

"$PY" -u benchmark_models.py \
  --model wav2lip \
  --select all \
  --yes \
  --offline

echo "===== DONE $(date) ====="
