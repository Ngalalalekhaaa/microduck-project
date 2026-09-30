#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ $# -eq 0 ]]; then
  movement=(--move)
elif [[ $# -eq 1 && $1 == --preview ]]; then
  movement=()
else
  echo '用法：bash walk02.sh [--preview]' >&2
  exit 2
fi
exec .venv/bin/python scripts/home_then_policy.py \
  --seconds 10 --vx 0.2 --ramp-seconds 3 --max-target-jump-deg 45 "${movement[@]}"
