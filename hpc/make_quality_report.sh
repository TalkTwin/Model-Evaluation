#!/bin/bash
set -euo pipefail
cd /home/aryannavale/projects
PY=/home/aryannavale/envs/hygem-rtx50/bin/python

"$PY" make_quality_report.py \
  --recon outputs/bench-20261101-001509 recon \
  --cross outputs/bench-20261031-152743 . \
  --cross outputs/bench-20261031-143801 ../bench-20261031-143801 \
  --out outputs/bench-20261031-152743/quality_report.html \
  --input-video inputs/marathi-demo-2/marathi-demo-2.mp4

ls -l outputs/bench-20261031-152743/quality_report.html
grep -c "duration-block" outputs/bench-20261031-152743/quality_report.html
