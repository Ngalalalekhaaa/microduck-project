#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 - <<'PY'
import platform, shutil, sys
print('系统:', platform.platform(), '架构:', platform.machine(), 'Python:', sys.version.split()[0])
if sys.version_info < (3, 11):
    raise SystemExit('ONNX Runtime 1.24.4需要Python3.11或更新的64位系统；请先确认系统版本。')
if sys.maxsize <= 2**32:
    raise SystemExit('需要64位Python和系统，不能使用32位armhf镜像。')
free=shutil.disk_usage('.').free/1024**3
print(f'剩余空间: {free:.2f} GiB')
if free < 0.7:
    raise SystemExit('请先留出至少0.7GiB空间安装CPU依赖。')
PY
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  # A failed ensurepip step can leave a working interpreter without pip.
  # Re-running venv repairs that partial environment without clearing it.
  python3 -m venv .venv || {
    echo '缺少venv时，Debian/Ubuntu可运行: sudo apt-get install python3-venv' >&2
    exit 1
  }
fi
.venv/bin/python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt
chmod +x run.sh
.venv/bin/python -m microduck_deploy self-test
echo '安装完成。下一步: ./run.sh init，再 ./run.sh map-ids --profile servo_ids.user.json，然后 ./run.sh doctor'
