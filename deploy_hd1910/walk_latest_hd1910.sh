#!/usr/bin/env bash
# Select an evaluated HD-1910 checkpoint, then use the existing robot runtime.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
variant="${1:-}"
if [[ "$variant" != flat && "$variant" != backlash ]]; then
  echo '用法: bash walk_latest_hd1910.sh flat|backlash [速度m/s] [秒数] [--select-only]' >&2
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
import sys
from pathlib import Path
from microduck_deploy.config import load_config, save_config
from microduck_deploy.policy import OnnxPolicy

variant, vx, seconds = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if not (math.isfinite(vx) and 0 <= vx <= 0.3
        and math.isfinite(seconds) and 3 <= seconds <= 60):
    raise SystemExit('速度须为 0–0.3 m/s，时长须为 3–60 秒（含 3 秒速度渐增）')
manifest = json.loads(Path('hd1910_latest_manifest.json').read_text())
entry = manifest['models'][variant]
model = Path(entry['file'])
if hashlib.sha256(model.read_bytes()).hexdigest() != entry['sha256']:
    raise SystemExit('模型校验失败，请重新传输更新包')
OnnxPolicy(model)
cfg = load_config('robot.json')
cfg['model'] = str(model)
save_config('robot.json', cfg)
print(f"已选择 HD-1910 {variant} {entry['iteration']} 轮: {model}", flush=True)
PY
if [[ "$mode" == --select-only ]]; then
  exit 0
fi
exec .venv/bin/python scripts/home_then_policy.py \
  --move --seconds "$seconds" --vx "$vx" --ramp-seconds 3 \
  --max-target-jump-deg 45 --log-only-target-angles
