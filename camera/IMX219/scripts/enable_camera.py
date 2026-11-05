#!/usr/bin/env python3
"""Preview the IMX219 overlay on Radxa ZERO 3 vendor 6.1; --apply backs up extlinux.

Only /boot/extlinux/extlinux.conf is changed. No service is stopped or started,
no reboot is scheduled, and the existing UART/J5/audio overlays are retained.
"""
import argparse
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

EXT = Path('/boot/extlinux/extlinux.conf')
CAMERA_NAMES = ('radxa-zero3-rpi-camera-v2.dtbo',
                'rk3568-radxa-zero3-rpi-camera-v2.dtbo',
                'microduck-media-imx219.dtbo')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def command(*args, optional=False):
    argv = [str(a) for a in args]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    if result.returncode:
        if optional:
            return None
        raise ValueError(repr(argv) + '\n' + result.stdout + result.stderr)
    return result.stdout.strip()


def parse_boot(text):
    """Reject ambiguous layouts; return only the single entry's FDT and overlays."""
    require('\0' not in text, 'extlinux 含 NUL，停止修改。')
    labels = list(re.finditer(r'^[ \t]*label[ \t]+\S+[ \t]*$', text, re.M | re.I))
    require(len(labels) == 1, '只支持单启动项 extlinux；其他布局请人工核对。')
    fdts = list(re.finditer(r'^[ \t]*fdt[ \t]+(\S+)[ \t]*$', text, re.M | re.I))
    require(len(fdts) == 1 and fdts[0].start() > labels[0].start(),
            '需要启动项内唯一 FDT 文件，FDTDIR 或隐式 FDT 不受支持。')
    require(fdts[0][1].startswith('/boot/'), 'FDT 必须是 /boot/ 下绝对路径。')
    overlays = list(re.finditer(r'^([ \t]*fdtoverlays(?:[ \t]+|$))([^\r\n]*)', text, re.M | re.I))
    require(len(overlays) <= 1, 'fdtoverlays 重复，停止修改。')
    require(not overlays or overlays[0].start() > labels[0].start(), 'fdtoverlays 必须位于启动项内。')
    old = overlays[0][2].split() if overlays else []
    require(all(p.startswith('/boot/') and '#' not in p for p in old),
            'overlay 路径必须是 /boot/ 下绝对路径；行尾注释请先移至独立行。')
    require(len(set(old)) == len(old), '原 fdtoverlays 有重复文件，停止修改。')
    camera = [p for p in old if 'camera' in Path(p).name.lower() or 'imx' in Path(p).name.lower()]
    require(all(Path(p).name in CAMERA_NAMES for p in camera) and len(camera) <= 1,
            '已有其他/重复摄像头 overlay，停止修改以避免引脚冲突。')
    return fdts[0][1], old, overlays[0] if overlays else None


def boot_plan(text, camera):
    base, old, match = parse_boot(text)
    require(camera.startswith('/boot/'), '摄像头 overlay 必须在 /boot/ 下。')
    existing = [p for p in old if Path(p).name in CAMERA_NAMES]
    new = old.copy()
    if not existing:
        new.append(camera)
    if new == old:
        return base, old, new, text
    if match:
        separator = '' if match[1][-1:].isspace() else ' '
        updated = text[:match.start(2)] + separator + ' '.join(new) + text[match.end(2):]
    else:
        updated = text + ('' if text.endswith('\n') else '\n') + '    fdtoverlays ' + ' '.join(new) + '\n'
    return base, old, new, updated


def get(tree, node, prop, kind='s', optional=False):
    return command('fdtget', '-t', kind, tree, node, prop, optional=optional)


def validate_trees(before, after):
    """Compare robot bus state before/after, including absent optional properties."""
    checks = {}
    for symbol in ('i2c3', 'uart2'):
        node = get(before, '/__symbols__', symbol)
        require(get(after, '/__symbols__', symbol) == node, symbol + ' 节点改变。')
        for prop, kind in (('status', 's'), ('pinctrl-0', 'u'), ('clock-frequency', 'u')):
            a = get(before, node, prop, kind, optional=True)
            b = get(after, node, prop, kind, optional=True)
            require(a == b, f'{symbol}.{prop} 改变：{a!r} -> {b!r}')
            checks[symbol + '.' + prop] = b
        if symbol == 'i2c3':
            a = get(before, node + '/fusb302@22', 'status', optional=True)
            b = get(after, node + '/fusb302@22', 'status', optional=True)
            require(a == b, 'J5/I2C3 FUSB302 状态改变。')
            checks['i2c3.fusb302.status'] = b
    require('sony,imx219' in command('dtc', '-I', 'dtb', '-O', 'dts', after), '合并后没有 IMX219。')
    for symbol in ('rkisp', 'rkisp_vir0', 'csi2_dphy0'):
        node = get(after, '/__symbols__', symbol)
        state = get(after, node, 'status')
        require(state == 'okay', symbol + ' 未启用。')
        checks[symbol] = state
    return checks


def write_boot(path, before, after, backup_root=Path('/var/backups/microduck-camera-boot')):
    """No overwrite if changed since planning; one timestamped original per apply."""
    require(path.read_text() == before, '预演期间启动配置发生变化，请重新执行。')
    if before == after:
        return None
    backup = backup_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    shutil.copy2(path, backup / 'extlinux.conf')
    (backup / 'change.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile=str(path), tofile=str(path) + '.new')))
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(after.encode())
            stream.flush()
            os.fsync(stream.fileno())
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='预演通过后备份并写入；不会自动重启')
    args = parser.parse_args()
    try:
        model = Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode()
        require('Radxa ZERO 3' in model and platform.machine() == 'aarch64', '仅支持 Radxa ZERO 3 / aarch64。')
        require(platform.release().startswith('6.1.') and 'vendor' in platform.release(),
                '只核对过 vendor 6.1；此脚本不会更换内核。')
        for tool in ('dtc', 'fdtoverlay', 'fdtget'):
            require(shutil.which(tool), '缺少工具：' + tool + '（安装 device-tree-compiler）')
        require(EXT.is_file(), '没有 /boot/extlinux/extlinux.conf，当前启动布局不受支持。')
        before = EXT.read_text()
        base, old, _ = parse_boot(before)
        choices = [Path(base).parent / 'overlay' / n for n in CAMERA_NAMES[:2]]
        choices.append(Path('/boot/dtbo/radxa-zero3-rpi-camera-v2.dtbo'))
        camera = next((p for p in old if Path(p).name in CAMERA_NAMES), None)
        camera = camera or next((str(p) for p in choices if p.is_file()), None)
        require(camera is not None, '系统缺少 IMX219 Camera V2 overlay；不要替换成 OV5647 Camera V1.3。')
        base, old, new, after = boot_plan(before, camera)
        for file in (base, *new):
            require(Path(file).is_file(), '启动依赖缺失：' + file)
        with tempfile.TemporaryDirectory(prefix='microduck-camera-dt-') as directory:
            baseline = Path(directory) / 'baseline.dtb'
            merged = Path(directory) / 'merged.dtb'
            if old:
                command('fdtoverlay', '-i', base, '-o', baseline, *old)
            else:
                shutil.copy2(base, baseline)
            command('fdtoverlay', '-i', base, '-o', merged, *new)
            checks = validate_trees(baseline, merged)
        print('VALIDATED', json.dumps(checks, ensure_ascii=False))
        print(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile=str(EXT), tofile=str(EXT) + ' (planned)')))
        if not args.apply:
            print('仅预览；启动配置与服务未改变。加 --apply 才会备份并写入。')
            return 0
        require(os.geteuid() == 0, '--apply 需要 sudo。')
        backup = write_boot(EXT, before, after)
        print('备份：' + str(backup) if backup else '已经启用，无需写入。')
        print('未重启、未操作舵机服务。若刚启用 overlay，请按说明自行重启后运行 doctor.py。')
        return 0
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print('错误：' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
