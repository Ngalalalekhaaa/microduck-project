#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /home/duck1/deploy_hd1910/.venv/bin/python "$release_dir/scripts/home_calibration.py" \
  --config /home/duck1/deploy_hd1910/robot.json \
  --model "$release_dir/models/pose/latest/home_calibration.onnx" \
  --contract "$release_dir/models/pose/latest/home_calibration_contract.json" \
  --current-check log-only --ram-p 8 "$@"
