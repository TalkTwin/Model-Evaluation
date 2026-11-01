#!/bin/bash
set -euo pipefail
cd /home/aryannavale/projects/inputs
FF=/home/aryannavale/envs/latentsync/bin/ffmpeg
for d in 5 10 15; do
  dir="recon-${d}s"
  mkdir -p "$dir"
  ln -sfn ../marathi-demo-2/marathi-demo-2.mp4 "$dir/marathi-demo-2.mp4"
  "$FF" -v error -y -i marathi-demo-2/marathi-demo-2.mp4 -t "$d" -vn \
    -ac 1 -ar 16000 -acodec pcm_s16le "$dir/marathi-demo-voice-2-recon-${d}s.wav"
  echo "=== $dir ==="
  ls -lh "$dir"
done
