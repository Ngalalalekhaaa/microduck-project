#!/usr/bin/env python3
"""Read-only IMX219 / ISP report. No stream, I2C register access, or service change."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess


def command(argv):
    if not shutil.which(argv[0]):
        return {'command': argv, 'error': 'command not installed'}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'command': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': argv, 'error': str(exc)}


def text(path):
    try:
        return Path(path).read_bytes().replace(b'\0', b'').decode(errors='replace').strip()
    except OSError as exc:
        return 'unavailable: ' + str(exc)


def inspect_iq(path):
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {'path': str(path), 'error': str(exc)}
    enabled = re.search(rb'"ae_calib"\s*:\s*\{\s*"CommCtrl"\s*:\s*\{\s*"Enable"\s*:\s*([01])', data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
            'rkaiq_ae_enabled': bool(int(enabled[1])) if enabled else None}


def collect():
    report = {'time_utc': datetime.now(timezone.utc).isoformat(),
              'read_only': True, 'model': text('/proc/device-tree/model'),
              'architecture': platform.machine(), 'kernel': platform.release(),
              'os_release': text('/etc/os-release'), 'devices': {}, 'commands': []}
    for path in sorted(Path('/sys/class/video4linux').glob('*/name')):
        report['devices']['/dev/' + path.parent.name] = text(path)
    report['sensor_driver'] = str(Path('/sys/bus/i2c/devices/2-0010/driver').resolve())
    report['module_name'] = text('/proc/device-tree/i2c@fe5b0000/camera-imx219@10/rockchip,camera-module-name')
    # Include the overlay names only, not root UUIDs or unrelated boot parameters.
    ext = text('/boot/extlinux/extlinux.conf')
    report['boot_overlays'] = re.findall(r'^[ \t]*fdtoverlays[ \t]+(.+)$', ext, re.M | re.I)
    iq = Path('/etc/iqfiles/imx219_rpi-camera-v2_default.json')
    report['iq'] = inspect_iq(iq)
    commands = [['id'], ['systemctl', 'status', 'rkaiq_3A', '--no-pager'],
                ['systemctl', 'cat', 'rkaiq_3A.service'],
                ['systemctl', 'show', 'rkaiq_3A.service',
                 '--property=Type,ExecStart,ExecStartPre,ExecStartPost,Environment,DropInPaths'],
                ['systemctl', 'is-active', 'mediad'],
                ['journalctl', '-u', 'rkaiq_3A', '-n', '80', '--no-pager'],
                ['v4l2-ctl', '--list-devices']]
    video = [p for p, name in report['devices'].items() if name in ('rkisp_mainpath', 'rkisp_selfpath')]
    if video:
        commands.append(['fuser', *video])
    for device, name in report['devices'].items():
        if name == 'rkisp_mainpath':
            commands.extend([['v4l2-ctl', '-d', device, '--get-fmt-video'],
                             ['v4l2-ctl', '-d', device, '--list-ctrls']])
    for media in sorted(Path('/dev').glob('media[0-9]*')):
        commands.append(['media-ctl', '-d', str(media), '-p'])
    if Path('/usr/bin/rkaiq_3A_server').is_file():
        commands.append(['ldd', '/usr/bin/rkaiq_3A_server'])
    report['commands'] = [command(argv) for argv in commands]
    report['note'] = 'V4L2 controls are cache values, not sensor-register verification. No frame was captured.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='保存 JSON 到指定文件；省略时只输出终端')
    args = parser.parse_args()
    result = json.dumps(collect(), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print('诊断报告：' + str(args.output))
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
