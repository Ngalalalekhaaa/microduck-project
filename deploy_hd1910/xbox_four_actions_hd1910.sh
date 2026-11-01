#!/usr/bin/env bash
# Explicitly enable the four updated models on the unused Xbox D-pad.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# > 1 )); then
  echo '用法: bash xbox_four_actions_hd1910.sh [策略秒数；默认60，0为持续运行]' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec .venv/bin/python scripts/model_suite.py run --four-actions \
  --seconds "${1:-60}" --max-vx 0.2
