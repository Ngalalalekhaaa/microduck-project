#!/usr/bin/env bash
# Train XgoDuck getup (StandUp), then walking (Velocity) on the same machine.
# The two tasks do not share checkpoints — velocity always starts from scratch.
#
# Usage (from repo root, or anywhere):
#   bash scripts/train_xgoduck_getup_then_velocity.sh
#   NUM_ENVS=2048 bash scripts/train_xgoduck_getup_then_velocity.sh --gpu-ids 0
#   bash scripts/train_xgoduck_getup_then_velocity.sh --skip-getup
#   bash scripts/train_xgoduck_getup_then_velocity.sh --rough
#
# Extra flags after the script's own options are forwarded to both `uv run train`
# calls (e.g. --agent.max-iterations 5 for a smoke test).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v uv >/dev/null 2>&1; then
  TRAIN=(uv run train)
elif [[ -x "$ROOT/.venv/bin/train" ]]; then
  TRAIN=("$ROOT/.venv/bin/train")
else
  echo "error: need \`uv\` on PATH, or $ROOT/.venv/bin/train" >&2
  exit 1
fi

SKIP_GETUP=0
ROUGH=0
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-getup)
      SKIP_GETUP=1
      shift
      ;;
    --rough)
      ROUGH=1
      shift
      ;;
    --)
      shift
      FORWARD+=("$@")
      break
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

NUM_ENVS="${NUM_ENVS:-4096}"
HAS_NUM_ENVS=0
for arg in "${FORWARD[@]+"${FORWARD[@]}"}"; do
  if [[ "$arg" == --env.scene.num-envs || "$arg" == --env.scene.num-envs=* ]]; then
    HAS_NUM_ENVS=1
    break
  fi
done

TRAIN_ARGS=()
if [[ "$HAS_NUM_ENVS" -eq 0 ]]; then
  TRAIN_ARGS+=(--env.scene.num-envs "$NUM_ENVS")
fi
TRAIN_ARGS+=("${FORWARD[@]+"${FORWARD[@]}"}")

if [[ "$ROUGH" -eq 1 ]]; then
  GETUP_TASK="Mjlab-StandUp-Rough-XgoDuck"
  VEL_TASK="Mjlab-Velocity-Rough-XgoDuck"
else
  GETUP_TASK="Mjlab-StandUp-Flat-XgoDuck"
  VEL_TASK="Mjlab-Velocity-Flat-XgoDuck"
fi

run_stage() {
  local label="$1"
  local task="$2"
  shift 2
  echo
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${label}: ${TRAIN[*]} ${task} $*"
  echo "============================================================"
  "${TRAIN[@]}" "$task" "$@"
}

if [[ "$SKIP_GETUP" -eq 0 ]]; then
  run_stage "1/2 getup" "$GETUP_TASK" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip getup (--skip-getup)"
fi

run_stage "2/2 velocity" "$VEL_TASK" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done. logs under logs/rsl_rl/xgoduck_standup and logs/rsl_rl/xgoduck_velocity"
