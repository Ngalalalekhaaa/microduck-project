#!/usr/bin/env bash
set -euo pipefail
walk_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$walk_dir/.venv/bin/python" "$walk_dir/scripts/home_then_policy.py" \
  --config "$walk_dir/robot.home_actions_20261101.json" \
  --log-only-target-angles "$@"
