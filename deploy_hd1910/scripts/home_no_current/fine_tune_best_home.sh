#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Explicitly pinned to the evaluated model/contract pair; never follows training.
exec "$release_dir/run_home_calibration.sh" \
  --mode run --seconds 10 \
  --model "$release_dir/models/pose/selected/home_calibration.onnx" \
  --contract "$release_dir/models/pose/selected/home_calibration_contract.json" "$@"
