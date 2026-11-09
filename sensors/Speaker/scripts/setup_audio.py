#!/usr/bin/env python3
"""Preview/apply only the HAT audio overlays on Radxa Armbian extlinux boot."""
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

HERE = Path(__file__).resolve().parents[1]
VENDOR = HERE / 'vendor'
EXT = Path('/boot/extlinux/extlinux.conf')
ENV = Path('/boot/armbianEnv.txt')
DEST = Path('/boot/overlay-user')
BUS = 'microduck-media-i2c3'
AUDIO = 'microduck-media-aic3104'
SYSTEM_TOOL_PATH = '/usr/sbin:/sbin:/usr/bin:/bin'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_tool(name):
    # Non-login SSH sessions commonly omit sbin even when kmod is installed.
    path = shutil.which(str(name)) or shutil.which(str(name), path=SYSTEM_TOOL_PATH)
    require(path, f'缺少 {name}，见接入说明的依赖安装命令。')
    return path


def command(*argv):
    resolved = [resolve_tool(argv[0]), *[str(x) for x in argv[1:]]]
    result = subprocess.run(resolved, capture_output=True, text=True)
    require(result.returncode == 0, shlex.join([str(x) for x in argv]) + '\n' + result.stdout + result.stderr)
    return result.stdout.strip()


def boot_plan(ext, env):
    """Preserve all existing overlays, serial/IMU options, root/console arguments."""
    require('\x00' not in ext, 'extlinux 含 NUL 字节，拒绝修改；请先核对启动配置。')
    labels = re.findall(r'^\s*label\s+(\S+)\s*$', ext, re.M | re.I)
    require(len(labels) == 1, '只支持单启动项 extlinux；多启动项需人工选择。')
    label_pos = re.search(r'^\s*label\s+\S+\s*$', ext, re.M | re.I).end()
    fdts = re.findall(r'^\s*fdt\s+(\S+)\s*$', ext, re.M | re.I)
    require(len(fdts) == 1 and fdts[0].startswith('/boot/'), '需要唯一 /boot/ 开头的 fdt 路径。')
    matches = list(re.finditer(r'^([ \t]*fdtoverlays)(?:[ \t]+([^\r\n]*))?$', ext, re.M | re.I))
    require(len(matches) <= 1, 'fdtoverlays 行重复。')
    require(not matches or matches[0].start() >= label_pos, 'fdtoverlays 必须在启动项内。')
    old = (matches[0][2] or '').split() if matches else []
    require(all(p.startswith('/boot/') for p in old), '现有 overlay 不是绝对 /boot/ 路径，需核对。')
    env_lines = {}
    for line in env.splitlines():
        m = re.fullmatch(r'([A-Za-z_][A-Za-z0-9_]*)=(.*)', line)
        if m:
            require(m[1] not in env_lines, 'armbianEnv 重复字段：' + m[1])
            env_lines[m[1]] = m[2]
    all_names = ' '.join(old + [env_lines.get('overlays', ''), env_lines.get('user_overlays', '')])
    require('i2c-gpio' not in all_names, '检测到软件 I2C overlay；不能和硬件 I2C3 同时启用。')
    configured = old + env_lines.get('overlays', '').split() + env_lines.get('user_overlays', '').split()
    require(all(AUDIO in p for p in configured if 'aic3104' in p),
            '已有其他音频 overlay；请保留原配置，先用 aplay -l 核对声卡。')
    names = []
    if not any('i2c3-pihat' in x or 'microduck-j5-i2c' in x or BUS in x for x in old):
        names.append(BUS)
    names.append(AUDIO)
    new = list(old)
    for name in names:
        path = str(DEST / (name + '.dtbo'))
        if path not in new:
            new.append(path)
    if matches:
        m = matches[0]
        ext_new = ext[:m.start()] + m[1] + ' ' + ' '.join(new) + ext[m.end():]
    else:
        ext_new = ext.rstrip('\n') + '\n    fdtoverlays ' + ' '.join(new) + '\n'
    # extlinux is authoritative on the reported board. Also keep armbianEnv in sync
    # so a future boot script regeneration retains the user overlays.
    words = env_lines.get('user_overlays', '').split()
    for path in new:
        if path.startswith(str(DEST) + '/'):
            word = Path(path).stem
            if word not in words:
                words.append(word)
    env_new = re.sub(r'^user_overlays=.*$', 'user_overlays=' + ' '.join(words), env, flags=re.M)
    if 'user_overlays' not in env_lines:
        env_new = env.rstrip('\n') + '\nuser_overlays=' + ' '.join(words) + '\n'
    return fdts[0], old, new, names, ext_new, env_new


def validate_tree(tree):
    get = lambda n, p, kind='s': command('fdtget', '-t', kind, tree, n, p)
    i2c = get('/__symbols__', 'i2c3')
    m0 = get('/__symbols__', 'i2c3m0_xfer')
    require(get(i2c, 'status') == 'okay', 'I2C3 未启用。')
    require(0 < int(get(i2c, 'clock-frequency', 'u')) <= 400000, 'I2C3 频率不适合 codec。')
    require(get(i2c, 'pinctrl-0', 'u') == get(m0, 'phandle', 'u'), 'I2C3 未选中 M0/J5 引脚。')
    pins = [int(x) for x in get(m0, 'rockchip,pins', 'u').split()]
    require(len(pins) == 8 and {tuple(pins[i:i+3]) for i in (0, 4)} == {(1, 0, 1), (1, 1, 1)},
            'M0 引脚不是 GPIO1_A0/A1。')
    require(get(i2c + '/fusb302@22', 'status') == 'disabled', 'FUSB302 仍占用 I2C3。')
    require(get(i2c + '/codec@18', 'status') == 'okay', 'codec 未启用。')
    require(get('/sound-aic3104', 'simple-audio-card,name') == 'aic3104', 'ALSA 卡名异常。')
    i2s = get('/__symbols__', 'i2s3_2ch')
    require(get(i2s, 'status') == 'okay', 'I2S3 未启用。')


def install_driver(kernel):
    available = subprocess.run([resolve_tool('modinfo'), '-k', kernel, 'snd-soc-tlv320aic3x-i2c'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if available.returncode == 0:
        print('内核已有 AIC3x I2C 模块，保留现有驱动。')
        return
    require(Path('/lib/modules', kernel, 'build').is_dir(),
            '缺少运行中内核对应 headers；先安装匹配版本，不能直接切换内核。')
    text = (VENDOR / 'aic3x-dkms/dkms.conf').read_text()
    version = re.search(r'^PACKAGE_VERSION="([^"]+)"', text, re.M)[1]
    source = Path('/usr/src') / ('aic3x-' + version)
    if source.exists():
        require(all((source / p.name).is_file() and (source / p.name).read_bytes() == p.read_bytes()
                    for p in (VENDOR / 'aic3x-dkms').iterdir()),
                f'{source} 已有不同来源驱动，不覆盖。')
    else:
        shutil.copytree(VENDOR / 'aic3x-dkms', source)
    print(command('dkms', 'install', '-m', 'aic3x', '-v', version, '-k', kernel))
    command('depmod', kernel)


def backup_write(path, data, tag):
    if path.exists():
        if path.read_bytes() == data:
            return
        backup = path.with_name(path.name + '.bak.' + tag)
        shutil.copy2(path, backup)
        print('备份：' + str(backup))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.chmod(path.stat().st_mode & 0o777 if path.exists() else 0o644)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='预演通过后安装驱动和 overlay；不重启')
    args = parser.parse_args(argv)
    try:
        model = Path('/proc/device-tree/model').read_text().lower()
        require('radxa' in model and ('zero 3' in model or 'zero3' in model), '此入口只支持 Radxa Zero 3。')
        kernel = command('uname', '-r')
        require(kernel.startswith('6.1.') and 'vendor' in kernel, '此音频驱动只核对了 vendor 6.1；不会自动更换内核。')
        for tool in ('dtc', 'fdtoverlay', 'fdtget', 'modinfo'):
            resolve_tool(tool)
        require(EXT.is_file() and ENV.is_file(), '未找到本机已核对的 Armbian extlinux 启动布局。')
        ext, env = EXT.read_text(), ENV.read_text()
        base, old, overlays, names, new_ext, new_env = boot_plan(ext, env)
        with tempfile.TemporaryDirectory(prefix='microduck-audio-') as directory:
            work = Path(directory)
            generated = {}
            for name, src in ((BUS, 'i2c3-pihat.dts'), (AUDIO, 'aic3104-i2c3.dts')):
                if name in names:
                    target = work / (name + '.dtbo')
                    command('dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', target, VENDOR / src)
                    generated[str(DEST / target.name)] = target
            for path in old:
                require(Path(path).is_file(), '原 overlay 文件缺失：' + path)
            merged = work / 'merged.dtb'
            command('fdtoverlay', '-i', base, '-o', merged, *[generated.get(p, p) for p in overlays])
            validate_tree(merged)
            print('设备树预演通过：保留原 overlay，添加 HAT I2C3 M0 + AIC3104 音频。')
            for path, before, after in ((EXT, ext, new_ext), (ENV, env, new_env)):
                print(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                  fromfile=str(path), tofile=str(path) + ' (planned)')))
            if not args.apply:
                print('仅预览；未更改系统。执行相同命令并加 --apply 才安装。')
                return 0
            require(os.geteuid() == 0, '--apply 需要 sudo。')
            install_driver(kernel)  # Build successfully before changing boot configuration.
            require(EXT.read_text() == ext and ENV.read_text() == env, '启动配置在预演期间变化，请重新运行。')
            tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            for path, source in generated.items():
                backup_write(Path(path), source.read_bytes(), tag)
            backup_write(ENV, new_env.encode(), tag)
            backup_write(EXT, new_ext.encode(), tag)
            print('已安装；未重启、未启动任何舵机服务。按接入说明在卸力断开舵机电源后重启。')
    except (OSError, ValueError) as exc:
        print('错误：' + str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
