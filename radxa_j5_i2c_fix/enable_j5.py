#!/usr/bin/env python3
"""Enable official J5 I2C on the user's Armbian extlinux layout after DT validation."""
from __future__ import annotations

import argparse
from datetime import datetime
import difflib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
VERSION = '1.1.0'
NAME = 'microduck-j5-i2c'
OVERLAY = f'/boot/overlay-user/{NAME}.dtbo'
BASE = '/boot/dtb/rockchip/rk3566-radxa-zero3.dtb'
EXT = '/boot/extlinux/extlinux.conf'
ENV = '/boot/armbianEnv.txt'
OLD_NAMES = ['uart2-m0', 'dwc3-peripheral']
OLD_PATHS = [f'/boot/dtb/rockchip/overlay/rk3568-{x}.dtbo' for x in OLD_NAMES]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def plan_configs(env, ext):
    """Deliberately support only the reported layout; preserve unrelated settings."""
    entries = {}
    for line in env.splitlines():
        match = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)', line)
        if match:
            key, value = match.groups()
            require(key not in entries, f'armbianEnv 重复字段: {key}')
            entries[key] = value
    require(entries.get('overlay_prefix') == 'rk3568', 'overlay_prefix 与已核对布局不同')
    require(entries.get('fdtfile') == 'rockchip/rk3566-radxa-zero3.dtb', 'fdtfile 与已核对板型不同')
    require(entries.get('overlays', '').split() == OLD_NAMES, '现有 overlays 已变化，需重新核对')
    require(entries.get('user_overlays', '').split() in ([], [NAME]), '已有其他 user_overlays，需检查引脚冲突')
    env_new = env
    if not entries.get('user_overlays', '').split():
        if 'user_overlays' in entries:
            env_new = re.sub(r'^user_overlays=.*$', f'user_overlays={NAME}', env, flags=re.M)
        else:
            env_new = env.rstrip('\n') + f'\nuser_overlays={NAME}\n'
    label_matches = list(re.finditer(r'^[ \t]*label[ \t]+(\S+)[ \t]*$', ext, flags=re.M | re.I))
    labels = [match[1] for match in label_matches]
    defaults = re.findall(r'^\s*default\s+(\S+)\s*$', ext, flags=re.M | re.I)
    require(labels == ['armbian'] and defaults == ['armbian'], '只支持已核对的单一 armbian 启动项')
    fdts = re.findall(r'^\s*fdt\s+(\S+)\s*$', ext, flags=re.M | re.I)
    require(fdts == [BASE], 'extlinux fdt 路径与已核对布局不同')
    matches = list(re.finditer(r'^(?P<head>[ \t]*fdtoverlays[ \t]+)(?P<value>[^\r\n]+)$', ext, re.M | re.I))
    require(len(matches) == 1, '需要唯一的 fdtoverlays 行')
    match = matches[0]
    paths = match['value'].split()
    require(paths in (OLD_PATHS, OLD_PATHS + [OVERLAY]), 'extlinux overlays 已变化，需重新核对')
    require(label_matches[0].end() <= match.start(), 'fdtoverlays 必须位于 armbian 启动项中')
    ext_new = ext
    if OVERLAY not in paths:
        ext_new = ext[:match.end('value')] + ' ' + OVERLAY + ext[match.end('value'):]
    return {ENV: env_new, EXT: ext_new}


def command(*args):
    result = subprocess.run([str(x) for x in args], text=True, capture_output=True)
    require(result.returncode == 0, f'命令失败: {shlex.join([str(x) for x in args])}\n{result.stderr}')
    return result.stdout.strip()


def validate_m0_pins(pins):
    # Each record is bank, pin, mux function, pinconfig phandle.
    # Vendor DTs list SCL (A1) before SDA (A0); record order is immaterial.
    expected = {(1, 0, 1), (1, 1, 1)}
    require(len(pins) == 8 and {tuple(pins[i:i + 3]) for i in (0, 4)} == expected,
            f'M0引脚必须恰好是GPIO1_A0/A1且功能号为1；实际 rockchip,pins={pins}')


def compile_and_check(root, work):
    for tool in ('dtc', 'fdtoverlay', 'fdtget'):
        require(shutil.which(tool), '先安装工具: sudo apt-get install device-tree-compiler')
    base = root / BASE.lstrip('/')
    require(base.is_file(), f'缺少 {base}')
    compatible = command('fdtget', '-t', 's', base, '/', 'compatible').split()
    require('rockchip,rk3566' in compatible and any(x.startswith('radxa,') for x in compatible),
            '基准DTB不是已核对的 Radxa RK3566 板型')
    # Missing target-path nodes must fail, rather than creating a fictitious USB node.
    command('fdtget', '-p', base, '/i2c@fe5c0000/fusb302@22')
    overlay = work / f'{NAME}.dtbo'
    command('dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', overlay, HERE / 'i2c3-pihat.dts')
    existing = [root / p.lstrip('/') for p in OLD_PATHS]
    require(all(p.is_file() for p in existing), '现有串口/USB overlay 文件缺失')
    merged = work / 'preview.dtb'
    command('fdtoverlay', '-i', base, '-o', merged, *existing, overlay)
    get = lambda node, prop, kind='s': command('fdtget', '-t', kind, merged, node, prop)
    i2c = '/i2c@fe5c0000'
    require(get('/__symbols__', 'i2c3') == i2c, 'I2C3符号指向异常')
    m0 = get('/__symbols__', 'i2c3m0_xfer')
    require(get(i2c, 'status') == 'okay', 'I2C3未启用')
    require(get(i2c, 'clock-frequency', 'u') == '400000', 'I2C3时钟验证失败')
    require(get(i2c, 'pinctrl-0', 'u') == get(m0, 'phandle', 'u'), 'I2C3未选中M0引脚')
    pins = [int(x) for x in get(m0, 'rockchip,pins', 'u').split()]
    print(f'M0节点 {m0}，实际 rockchip,pins={pins}')
    validate_m0_pins(pins)
    require(get(i2c + '/fusb302@22', 'status') == 'disabled', 'USB-C PD控制器未停用')
    print('设备树预演通过：I2C3 → GPIO1_A0/A1（J5），400kHz，FUSB302 disabled。')
    return overlay.read_bytes()


def atomic_write(path, data):
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def apply_files(root, original, replacements, dtbo):
    target = root / OVERLAY.lstrip('/')
    require(not target.is_symlink(), '目标overlay不可为符号链接')
    if target.exists():
        require(target.read_bytes() == dtbo, '已存在不同内容的同名overlay，停止以免覆盖')
    for name, before in original.items():
        require((root / name.lstrip('/')).read_bytes() == before, '配置在检查期间发生变化，请重试')
    pending = {name: data.encode() for name, data in replacements.items() if data.encode() != original[name]}
    if not pending and target.exists():
        print('配置已一致，没有重复修改。若尚未重启，请重启后检查。')
        return None
    backup = root / 'boot' / ('microduck-j5-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    backup.mkdir(mode=0o700)
    for name in original:
        shutil.copy2(root / name.lstrip('/'), backup / Path(name).name)
    print(f'原始配置已备份到 {backup}', flush=True)
    existed = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write(target, dtbo)
        # The boot entries are changed only after the validated overlay is installed.
        for name, data in pending.items():
            atomic_write(root / name.lstrip('/'), data)
        os.sync()
    except BaseException as exc:
        failures = []
        for name, data in original.items():
            try:
                atomic_write(root / name.lstrip('/'), data)
            except BaseException as rollback_error:
                failures.append(f'{name}: {rollback_error}')
        # Keep the valid overlay if any boot entry might still refer to it.
        if not failures and not existed:
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                failures.append(f'清理新overlay: {rollback_error}')
        os.sync()
        detail = '; '.join(failures) if failures else '原配置已恢复'
        raise ValueError(f'写入失败: {exc}；恢复结果: {detail}；备份: {backup}') from exc
    print(f'已保存；原始配置备份在 {backup}')
    print('回退命令（恢复后再重启）：')
    for name in original:
        print(shlex.join(['sudo', 'cp', '-p', str(backup / Path(name).name), str(root / name.lstrip('/'))]))
    print('应用成功。尚未重启；接下来执行 sudo reboot。')
    return backup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=VERSION)
    parser.add_argument('--apply', action='store_true', help='验证通过后备份并写入启动配置；不自动重启')
    args = parser.parse_args()
    print(f'J5 修复工具 {VERSION}')
    root = Path('/')
    if args.apply:
        require(os.geteuid() == 0, '--apply 需要 sudo python3 enable_j5.py --apply')
    require(all(not (root / name.lstrip('/')).is_symlink() for name in (ENV, EXT)),
            '启动配置是符号链接，需要先核对实际路径')
    original = {name: (root / name.lstrip('/')).read_bytes() for name in (ENV, EXT)}
    replacements = plan_configs(*(original[name].decode() for name in (ENV, EXT)))
    with tempfile.TemporaryDirectory(prefix='microduck-j5-check-') as directory:
        dtbo = compile_and_check(root, Path(directory))
    for name, after in replacements.items():
        print(''.join(difflib.unified_diff(original[name].decode().splitlines(True),
                                         after.splitlines(True), fromfile=name, tofile=name + ' (计划)')), end='')
    print('此配置停用USB-C FUSB302的PD协商；适用于当前HAT供5V的接法。')
    if args.apply:
        apply_files(root, original, replacements, dtbo)
    else:
        print('只读检查完成，未修改 /boot。应用命令: sudo python3 enable_j5.py --apply')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f'停止：{exc}\n如预演或写入失败，请保留输出，不要重启。')
