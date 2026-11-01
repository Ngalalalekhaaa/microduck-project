"""Update an installed boot service to the current Xbox speed limits."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.geteuid() != 0:
        raise SystemExit('Use sudo python3 scripts/apply_xbox_lower_limits.py')
    # Updating an active control session could unload a standing robot.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(arg.rsplit(b'/',1)[-1] in (b'policy_trial.py',b'supported_home.py') for arg in args):
            raise SystemExit('HOME或策略正在运行；先扶稳并按B停止，再更新限速。')
    verify = '''from microduck_deploy.config import load_config
from microduck_deploy.servo import ServoBus
cfg=load_config('robot.json')
with ServoBus(**cfg['serial']) as bus:
    states=[bus.read_configuration(j['id'])['torque'] for j in cfg['joints']]
if any(states): raise SystemExit('舵机仍上力；先托稳并按B卸力，再应用新限速。')
'''
    subprocess.run([str(ROOT/'.venv/bin/python'),'-c',verify],cwd=ROOT,check=True)
    config = Path('/etc/default/microduck-xbox')
    old = config.read_text()
    config.with_name('microduck-xbox.before-lower-limits').write_text(old)
    lines = [line for line in old.splitlines() if not line.startswith('MICRODUCK_MAX_VX=')]
    lines.append('MICRODUCK_MAX_VX=0.3')
    config.write_text('\n'.join(lines)+'\n')
    subprocess.run(['systemctl','restart','microduck-xbox'],check=True)
    print('已应用：前后±0.3m/s，左右±0.3m/s，转向±1rad/s。等待新的A键。')


if __name__ == '__main__':
    main()
