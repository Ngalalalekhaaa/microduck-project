#!/usr/bin/env bash
set -euo pipefail
midpoint_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "$midpoint_dir/.venv/bin/python" "$midpoint_dir/scripts/home_midpoint.py" "$@"
