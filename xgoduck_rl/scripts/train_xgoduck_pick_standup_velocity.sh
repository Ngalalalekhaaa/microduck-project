#!/usr/bin/env bash
# Train XgoDuck GroundPick, then StandUp, then Velocity on the same machine.
# The three tasks do not share checkpoints — each stage starts from scratch.
#
# Usage (from repo root, or anywhere):
#   bash scripts/train_xgoduck_pick_standup_velocity.sh
#   NUM_ENVS=2048 bash scripts/train_xgoduck_pick_standup_velocity.sh --gpu-ids 0
#   bash scripts/train_xgoduck_pick_standup_velocity.sh --skip-pick
#   bash scripts/train_xgoduck_pick_standup_velocity.sh --skip-pick --skip-standup
#   bash scripts/train_xgoduck_pick_standup_velocity.sh --rough
#
# Extra flags after the script's own options are forwarded to every `train`
# call (e.g. --agent.max-iterations 5 for a smoke test).

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

SKIP_PICK=0
SKIP_STANDUP=0
ROUGH=0
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pick|--skip-groundpick|--skip-ground-pick)
      SKIP_PICK=1
      shift
      ;;
    --skip-standup|--skip-getup)
      SKIP_STANDUP=1
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
  PICK_TASK="Mjlab-GroundPick-Rough-XgoDuck"
  STANDUP_TASK="Mjlab-StandUp-Rough-XgoDuck"
  VEL_TASK="Mjlab-Velocity-Rough-XgoDuck"
else
  PICK_TASK="Mjlab-GroundPick-Flat-XgoDuck"
  STANDUP_TASK="Mjlab-StandUp-Flat-XgoDuck"
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

if [[ "$SKIP_PICK" -eq 0 ]]; then
  run_stage "1/3 ground-pick" "$PICK_TASK" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip ground-pick (--skip-pick)"
fi

if [[ "$SKIP_STANDUP" -eq 0 ]]; then
  run_stage "2/3 standup" "$STANDUP_TASK" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] skip standup (--skip-standup)"
fi

run_stage "3/3 velocity" "$VEL_TASK" "${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}"

echo
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done. logs under"
echo "  logs/rsl_rl/xgoduck_ground_pick"
echo "  logs/rsl_rl/xgoduck_standup"
echo "  logs/rsl_rl/xgoduck_velocity"
