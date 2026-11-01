"""Preview or apply the authorized 4.0..8.4 V software range; no hardware I/O."""
import argparse
import copy
import fcntl
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microduck_deploy.config import load_config, save_config


def updated_config(cfg):
    old = cfg['control'].get('servo_voltage_range')
    if old not in ([5.0, 8.4], [4.0, 8.4]):
        raise ValueError(f'当前电压范围为{old}，与预期不符，未覆盖')
    result = copy.deepcopy(cfg)
    result['control']['servo_voltage_range'] = [4.0, 8.4]
    return result


def require_stopped():
    state = subprocess.run(['systemctl', 'is-active', '--quiet', 'microduck-xbox.service'],
                           capture_output=True, text=True)
    if state.returncode not in (3, 4):
        raise ValueError('先执行 sudo systemctl stop microduck-xbox.service，再应用配置')
    controllers = {b'home_then_policy.py', b'policy_trial.py', b'supported_home.py',
                   b'imu_home_assist.py', b'straight_then_home.py'}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            continue
        except PermissionError:
            # systemd state above checks the installed controller service;
            # processes inaccessible to this user cannot be inspected here.
            continue
        if any(arg.rsplit(b'/', 1)[-1] in controllers for arg in args):
            raise ValueError('仍有手动控制流程运行，请退出后再应用配置')


def apply_config(path):
    path = Path(path).resolve()
    (path.parent / 'logs').mkdir(exist_ok=True)
    with (path.parent / 'logs' / 'supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_stopped()
        cfg = load_config(path)
        candidate = updated_config(cfg)
        if candidate == cfg:
            print('已是4.0～8.4V，无需修改。')
            return
        # save_config creates a timestamped backup and atomically replaces JSON.
        save_config(path, candidate)
        if load_config(path) != candidate:
            raise ValueError('配置回读与写入内容不一致')
        print(f'已备份并更新 {path}：5.0～8.4V → 4.0～8.4V。')
        print(f'原配置备份：{path.name}.bak.时间戳')
        print('其余配置及2、11号标定保留。重启遥控服务后生效；本脚本不访问舵机。')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'robot.json')
    parser.add_argument('--apply', action='store_true', help='备份并保存；默认只预览')
    args = parser.parse_args(argv)
    if args.apply:
        apply_config(args.config)
    else:
        cfg = load_config(args.config)
        candidate = updated_config(cfg)
        print(f"预览：{cfg['control']['servo_voltage_range']} → {candidate['control']['servo_voltage_range']}，尚未写入。")


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
