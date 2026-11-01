#!/bin/bash
#SBATCH --job-name=quality
#SBATCH --partition=gpu
#SBATCH --nodelist=aicoeserver05
#SBATCH --gres=gpu:1g.24gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Submitted with:  --export=ALL,MODEL=<model>,OUTDIRS="<dir1> <dir2>"
# Runs quality_batch.py over one or more benchmark output folders.

set -uo pipefail

MODEL="${MODEL:?MODEL env not set}"
OUTDIRS="${OUTDIRS:?OUTDIRS env not set}"
PY="$HOME/envs/hygem-rtx50/bin/python"
BATCH="$HOME/projects/quality_batch.py"

echo "hostname: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL: $MODEL"
echo "OUTDIRS: $OUTDIRS"
date

test -r "$BATCH" || { echo "missing $BATCH"; exit 2; }
test -x "$PY" || { echo "missing $PY"; exit 2; }

rc=0
for d in $OUTDIRS; do
    echo
    echo "=================================================================="
    echo "quality: model=$MODEL outdir=$d"
    echo "=================================================================="
    "$PY" -u "$BATCH" --outdir "$d" --model "$MODEL" || rc=1
done

echo
echo "quality batch finished rc=$rc"
date
exit $rc
