#!/usr/bin/env bash
# Select a friction-randomization candidate, preserving the robot calibration.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
variant="${1:-}"
if [[ "$variant" != flat && "$variant" != backlash ]] || (( $# > 4 )); then
  echo '用法: bash walk_friction_hd1910.sh flat|backlash [速度m/s] [秒数] [--select-only]' >&2
  exit 2
fi
vx="${2:-0.2}"
seconds="${3:-60}"
mode="${4:-}"
if [[ "$mode" != '' && "$mode" != --select-only ]]; then
  echo '第四个参数仅支持 --select-only' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python - "$variant" "$vx" "$seconds" <<'PY'
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from microduck_deploy.config import load_config, save_config
from microduck_deploy.policy import OnnxPolicy

variant, vx, seconds = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if not (math.isfinite(vx) and 0 <= vx <= 0.3
        and math.isfinite(seconds) and 3 <= seconds <= 60):
    raise SystemExit('速度须为 0–0.3 m/s，时长须为 3–60 秒（含 3 秒速度渐增）')

manifest = json.loads(Path('friction_candidate_manifest.json').read_text())
entry = manifest['models'][variant]
expected_file = f'models/hd1910_friction_{variant}_4000.onnx'
if entry['file'] != expected_file or entry['iteration'] != 4000:
    raise SystemExit('候选模型清单与所选 4000 轮文件不一致')
model = Path(entry['file'])
expected_hash = entry['sha256']
if not isinstance(expected_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_hash):
    raise SystemExit('候选模型清单中的 SHA256 无效')
if hashlib.sha256(model.read_bytes()).hexdigest() != expected_hash:
    raise SystemExit('候选模型校验失败，请重新传输模型和清单')

# Validate the exported HOME, joint order, raw observations and action scale.
# The checkpoint's original training configuration must be checked separately.
OnnxPolicy(model)
if not manifest.get('training_config_verified', False):
    print('候选模型尚未附带原训练配置；本次按现有 HD-1910 的 HOME 和观测约定运行。', flush=True)

cfg = load_config('robot.json')
if cfg.get('model') != str(model):
    cfg['model'] = str(model)
    save_config('robot.json', cfg)
print(f'已选择摩擦随机化候选 {variant} 4000 轮: {model}', flush=True)
PY
if [[ "$mode" == --select-only ]]; then
  exit 0
fi
exec .venv/bin/python scripts/home_then_policy.py \
  --move --seconds "$seconds" --vx "$vx" --ramp-seconds 3 \
  --log-only-target-angles
