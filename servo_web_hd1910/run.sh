#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo '请先运行 bash install.sh'
  exit 1
fi
exec .venv/bin/python server.py --port /dev/ttyS2 --ids 1-14 \
  --host 0.0.0.0 --http-port 8080 "$@"
