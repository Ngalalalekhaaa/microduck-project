#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /home/duck1/deploy_hd1910/.venv/bin/python "$release_dir/scripts/supported_home.py" \
  --config /home/duck1/deploy_hd1910/robot.json --calibration-hold "$@"
