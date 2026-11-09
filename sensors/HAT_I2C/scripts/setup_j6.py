#!/usr/bin/env python3
"""Preview J6 software I2C on Radxa ZERO 3 vendor 6.1; --apply writes backups.

Never reboots, probes I2C addresses, changes J5 or starts robot services.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import difflib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parents[1]
NAME = 'microduck-j6-i2c-gpio'
TARGET = Path('/boot/overlay-user') / (NAME + '.dtbo')
EXT = Path('/boot/extlinux/extlinux.conf')
ENV = Path('/boot/armbianEnv.txt')
TOOL_PATH = '/usr/sbin:/sbin:/usr/bin:/bin'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve_tool(name):
    path = shutil.which(name) or shutil.which(name, path=TOOL_PATH)
    require(path, '缺少工具：' + name)
    return path


def command(*args):
    argv = [resolve_tool(str(args[0])), *map(str, args[1:])]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    require(result.returncode == 0, ' '.join(map(str, args)) + '\n' + result.stdout + result.stderr)
    return result.stdout.strip()


def boot_plan(ext, env):
    """Append one overlay, keeping unrelated text and NUL padding unchanged."""
    require('\0' not in ext, 'extlinux 含 NUL，停止修改。')
    labels = list(re.finditer(r'^[ \t]*label[ \t]+\S+[ \t]*$', ext, re.M | re.I))
    require(len(labels) == 1, '仅支持单启动项 extlinux。')
    fdts = list(re.finditer(r'^[ \t]*fdt[ \t]+(\S+)[ \t]*$', ext, re.M | re.I))
    require(len(fdts) == 1 and fdts[0].start() > labels[0].start(), '需要启动项内唯一 FDT。')
    require(fdts[0][1].startswith('/boot/'), 'FDT 必须为 /boot/ 下绝对路径。')
    matches = list(re.finditer(r'^([ \t]*fdtoverlays)(?:[ \t]+([^\r\n]*))?$', ext, re.M | re.I))
    require(len(matches) <= 1 and (not matches or matches[0].start() > labels[0].start()),
            'fdtoverlays 重复或不在启动项内。')
    old = (matches[0][2] or '').split() if matches else []
    require(len(old) == len(set(old)), 'fdtoverlays 存在重复项。')
    require(all(p.startswith('/boot/') and '#' not in p for p in old), 'overlay 必须为绝对 /boot/ 路径，无行尾注释。')
    require(all(Path(p).name != TARGET.name or p == str(TARGET) for p in old), '发现其他路径的同名 J6 overlay。')
    new = old if str(TARGET) in old else old + [str(TARGET)]
    if new == old:
        ext_new = ext
    elif matches:
        m = matches[0]
        ext_new = ext[:m.start()] + m[1] + ' ' + ' '.join(new) + ext[m.end():]
    else:
        ext_new = ext + ('' if ext.endswith('\n') else '\n') + '    fdtoverlays ' + ' '.join(new) + '\n'
    env_matches = list(re.finditer(r'^user_overlays=([^\r\n\x00]*)', env, re.M))
    require(len(env_matches) <= 1, 'armbianEnv user_overlays 字段重复。')
    words = env_matches[0][1].split() if env_matches else []
    for p in new:
        if p.startswith('/boot/overlay-user/') and Path(p).stem not in words:
            words.append(Path(p).stem)
    if env_matches:
        m = env_matches[0]
        env_new = env[:m.start(1)] + ' '.join(words) + env[m.end(1):]
    else:
        # Put new text before trailing NUL padding instead of editing the padding.
        body = env.rstrip('\0')
        env_new = body + ('' if body.endswith('\n') else '\n') + 'user_overlays=' + ' '.join(words) + '\n' + env[len(body):]
    return fdts[0][1], old, new, ext_new, env_new


def read_dtb(data):
    """Read DTB properties as raw bytes, without lossy DTS text conversion."""
    require(len(data) >= 40, 'DTB 太短。')
    header = struct.unpack_from('>10I', data)
    magic, total, off, strings = header[:4]
    require(magic == 0xD00DFEED and total <= len(data), '无效 DTB。')
    result, stack = {}, []
    while off < total:
        token, = struct.unpack_from('>I', data, off)
        off += 4
        if token == 1:
            end = data.index(b'\0', off, total)
            stack.append(data[off:end].decode())
            off = (end + 4) & ~3
            result['/' + '/'.join(stack[1:])] = {}
        elif token == 2:
            stack.pop()
        elif token == 3:
            size, nameoff = struct.unpack_from('>II', data, off)
            off += 8
            start = strings + nameoff
            end = data.index(b'\0', start, total)
            key = data[start:end].decode()
            result['/' + '/'.join(stack[1:])][key] = data[off:off + size]
            off = (off + size + 3) & ~3
        elif token == 4:
            pass
        elif token == 9:
            return result
        else:
            raise ValueError('未知 DTB token：' + str(token))
    raise ValueError('DTB 缺少结束标记。')


def cells(value):
    require(len(value) % 4 == 0, 'DT cells 长度错误。')
    return struct.unpack('>' + 'I' * (len(value) // 4), value)


def string(value):
    return value.rstrip(b'\0').decode()


def validate_trees(before_data, after_data):
    """Preserve EVERY old property; only add the declared J6 nodes/metadata."""
    before, after = read_dtb(before_data), read_dtb(after_data)
    symbols = after['/__symbols__']
    gpio_path = string(symbols['gpio3'])
    pinctrl_path = string(symbols['pinctrl'])
    pull_path = string(symbols['pcfg_pull_none'])
    allowed_roots = ('/i2c-j6', pinctrl_path + '/j6-i2c-gpio')
    additions = {'/aliases': {'i2c10'}, '/__symbols__': {'i2c_j6', 'j6_i2c_pins'}}
    for path, props in before.items():
        require(path in after, '原节点被移除：' + path)
        for key, value in props.items():
            require(after[path].get(key) == value, '原属性发生改变：' + path + ':' + key)
    for path, props in after.items():
        if any(path == p or path.startswith(p + '/') for p in allowed_roots):
            continue
        require(path in before or path in additions, '增加了非 J6 节点：' + path)
        extra = props.keys() - before.get(path, {}).keys()
        require(extra <= additions.get(path, set()), '增加了非 J6 属性：' + path + ':' + repr(extra))
    bus = after['/i2c-j6']
    require(string(after['/aliases']['i2c10']) == '/i2c-j6', 'i2c10 alias 异常。')
    require(string(bus['compatible']) == 'i2c-gpio' and string(bus['status']) == 'okay', 'J6 总线未启用。')
    gpio_handle = cells(after[gpio_path]['phandle'])[0]
    require(cells(bus['sda-gpios']) == (gpio_handle, 20, 6), 'SDA GPIO3_C4/open-drain 不正确。')
    require(cells(bus['scl-gpios']) == (gpio_handle, 11, 6), 'SCL GPIO3_B3/open-drain 不正确。')
    require('i2c-gpio,scl-output-only' not in bus, 'SCL 必须可读以支持时钟延展。')
    for key in ('i2c-gpio,sda-open-drain', 'i2c-gpio,scl-open-drain'):
        require(key not in bus, '不要使用跳过开漏模拟的旧属性：' + key)
    require(cells(bus['i2c-gpio,delay-us']) == (10,) and cells(bus['i2c-gpio,timeout-ms']) == (100,), '时序参数异常。')
    pins = after[pinctrl_path + '/j6-i2c-gpio/pins']
    require(cells(bus['pinctrl-0']) == cells(pins['phandle']), 'J6 pinctrl 未绑定。')
    pull = cells(after[pull_path]['phandle'])[0]
    require(cells(pins['rockchip,pins']) == (3, 20, 0, pull, 3, 11, 0, pull), 'J6 引脚复用或上下拉异常。')
    return {'bus': '/dev/i2c-10', 'SDA': 'GPIO3_C4 / physical 7 / J6.3',
            'SCL': 'GPIO3_B3 / physical 29 / J6.4', 'preserved_nodes': len(before)}


def atomic_write(path, data, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
            tmp.chmod(mode)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def apply_files(changes, backup_root=Path('/var/backups/microduck-j6-i2c')):
    """Write (path, expected old bytes or None, new bytes); rollback on failure."""
    pending = [(p, old, new) for p, old, new in changes if old != new]
    for path, old, _ in changes:
        require((path.read_bytes() if path.exists() else None) == old, '文件在预演后发生变化：' + str(path))
    if not pending:
        return None
    backup = backup_root / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    backup.mkdir(parents=True)
    manifest, modes = [], {}
    for index, (path, old, _) in enumerate(pending):
        modes[path] = path.stat().st_mode & 0o777 if old is not None else 0o644
        saved = str(index) + '-' + path.name
        if old is not None:
            (backup / saved).write_bytes(old)
        manifest.append({'path': str(path), 'backup': saved if old is not None else None})
    (backup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    done = []
    try:
        for path, old, new in pending:
            atomic_write(path, new, modes[path])
            done.append((path, old))
    except BaseException:
        for path, old in reversed(done):
            if old is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write(path, old, modes[path])
        raise
    return backup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='预演通过后备份并写入；不会自动重启')
    args = parser.parse_args(argv)
    try:
        model = Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode().lower()
        require('radxa' in model and ('zero 3' in model or 'zero3' in model) and platform.machine() == 'aarch64',
                '只支持 Radxa ZERO 3 / aarch64。')
        require(platform.release().startswith('6.1.') and 'vendor' in platform.release(), '只核对了 vendor 6.1；不会更换内核。')
        command('modinfo', 'i2c-gpio')
        require(EXT.is_file() and ENV.is_file(), '需要已核对的 Armbian extlinux 启动布局。')
        ext_bytes, env_bytes = EXT.read_bytes(), ENV.read_bytes()
        ext, env = ext_bytes.decode(), env_bytes.decode()
        base, old, new, ext_new, env_new = boot_plan(ext, env)
        for path in (base, *old):
            require(Path(path).is_file(), '启动依赖缺失：' + path)
        with tempfile.TemporaryDirectory(prefix='microduck-j6-dt-') as directory:
            work = Path(directory)
            compiled, baseline, merged = (work / name for name in ('j6.dtbo', 'before.dtb', 'after.dtb'))
            command('dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', compiled, HERE / 'overlays' / (NAME + '.dts'))
            blob = compiled.read_bytes()
            target_old = TARGET.read_bytes() if TARGET.exists() else None
            require(target_old is None or target_old == blob, '已有不同内容的 J6 overlay，拒绝覆盖：' + str(TARGET))
            if old:
                command('fdtoverlay', '-i', base, '-o', baseline, *old)
            else:
                shutil.copy2(base, baseline)
            baseline_tree = read_dtb(baseline.read_bytes())
            if str(TARGET) not in old:
                require('/i2c-j6' not in baseline_tree and 'i2c10' not in baseline_tree.get('/aliases', {}),
                        '现有设备树已占用 /i2c-j6 或 i2c10。')
                require(not Path('/sys/class/i2c-adapter/i2c-10').exists(), '运行中的 i2c-10 已被占用。')
                command('fdtoverlay', '-i', baseline, '-o', merged, compiled)
            else:
                shutil.copy2(baseline, merged)
            checks = validate_trees(baseline.read_bytes(), merged.read_bytes())
        print('VALIDATED', json.dumps(checks, ensure_ascii=False))
        for path, before, after in ((EXT, ext, ext_new), (ENV, env, env_new)):
            print(''.join(difflib.unified_diff(before.replace('\0', '<NUL>').splitlines(True),
                after.replace('\0', '<NUL>').splitlines(True), fromfile=str(path), tofile=str(path) + ' (planned)')))
        if not args.apply:
            print('仅预览；未修改启动配置或服务。加 --apply 写入，随后需要重启。')
            return 0
        require(os.geteuid() == 0, '--apply 需要 sudo。')
        backup = apply_files([(TARGET, target_old, blob), (ENV, env_bytes, env_new.encode()),
                              (EXT, ext_bytes, ext_new.encode())])
        print('备份：' + str(backup) if backup else '配置已存在，无需写入。')
        print('未重启、未扫描总线、未启动舵机服务。重启后确认 /dev/i2c-10 对应 /i2c-j6。')
        return 0
    except (OSError, ValueError, KeyError, struct.error, subprocess.TimeoutExpired) as exc:
        print('错误：' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
