#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$release_dir/run_home_calibration.sh" \
  --model "$release_dir/models/force_ratio/latest/home_calibration.onnx" \
  --contract "$release_dir/models/force_ratio/latest/home_calibration_contract.json" \
  --mode run --seconds 10 "$@"
