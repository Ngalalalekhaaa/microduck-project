#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# > 3 )); then
  echo '用法: bash xbox_friction_hd1910.sh [flat|backlash] [秒数] [最大前后速度m/s]' >&2
  exit 2
fi
variant="${1:-flat}"
seconds="${2:-60}"
max_vx="${3:-0.3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Validate the requested maximum before selecting a model or entering HOME.
.venv/bin/python - "$max_vx" <<'PY'
import math
import sys

maximum = float(sys.argv[1])
if not math.isfinite(maximum) or not 0 < maximum <= 0.4:
    raise SystemExit('手柄最大前后速度须为大于 0、不超过 0.4 m/s')
PY
bash walk_friction_hd1910.sh "$variant" 0 "$seconds" --select-only
exec .venv/bin/python scripts/home_then_policy.py \
  --move --seconds "$seconds" --vx 0 --ramp-seconds 3 \
  --log-only-target-angles --gamepad --gamepad-buttons \
  --max-vx "$max_vx" --max-vy 0.3 --max-wz 1.0
