#!/usr/bin/env bash
set -euo pipefail
release_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$release_dir/run_home_calibration.sh" --mode run --seconds 10 "$@"
