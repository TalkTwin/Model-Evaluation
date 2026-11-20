#!/bin/bash
set -uo pipefail
cd "$HOME/projects" || exit 1
mkdir -p logs
LOG="$HOME/projects/logs/wav2lip_multilingual_batch.log"
: > "$LOG"
export PATH=/usr/bin:$PATH
export PYTHONUNBUFFERED=1
MIG=MIG-60a375d5-8693-57ab-8a24-63f3be2d2782
{
  echo "=== batch start $(date) ==="
  for i in 1 2 3 4 5 6 7 8 9 10 11; do
    echo "=== run $i $(date) ==="
    CUDA_VISIBLE_DEVICES="$MIG" "$HOME/envs/latentsync/bin/python" \
      run_wav2lip_multilingual.py \
      --plan plan_wav2lip_multilingual.json \
      --run-index "$i"
    rc=$?
    echo "=== run $i exit=$rc $(date) ==="
    if [ "$rc" -ne 0 ]; then
      echo "=== stopping at failed run $i ==="
      break
    fi
  done
  echo "=== batch done $(date) ==="
} >> "$LOG" 2>&1
