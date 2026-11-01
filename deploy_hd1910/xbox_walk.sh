#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mode=${1:-flat}
seconds=${2:-60}
max_vx=${3:-0.3}
bash walk_latest_hd1910.sh "$mode" 0 "$seconds" --select-only
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec .venv/bin/python scripts/home_then_policy.py --move --seconds "$seconds" \
  --vx 0 --ramp-seconds 3 --log-only-target-angles \
  --gamepad --gamepad-buttons --max-vx "$max_vx" --max-vy 0.3 --max-wz 1.0
