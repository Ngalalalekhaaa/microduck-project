#!/usr/bin/env bash
# Run any compatible HD-1910 model with the existing Xbox A/Menu/B workflow.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if (( $# < 1 || $# > 3 )); then
  echo '用法: bash xbox_model_hd1910.sh 模型.onnx [秒数] [最大前后速度m/s]' >&2
  exit 2
fi
model="$1"
seconds="${2:-0}"
max_vx="${3:-0.3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python scripts/select_policy_model.py "$model" "$seconds" "$max_vx"
exec .venv/bin/python scripts/home_then_policy.py \
  --move --seconds "$seconds" --vx 0 --ramp-seconds 3 \
  --log-only-target-angles --gamepad --gamepad-buttons \
  --max-vx "$max_vx" --max-vy 0.3 --max-wz 1.0
