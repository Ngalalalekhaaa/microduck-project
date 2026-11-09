#!/usr/bin/env bash
# 兼容入口；实际代码集中在 sensors/IMX219，必须显式指定 USER@IP。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$root/sensors/IMX219/scripts/capture_frame.sh" "$@"
