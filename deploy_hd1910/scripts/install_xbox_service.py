"""Install the Xbox boot listener; run with sudo, optionally --start."""
import argparse
import os
from pathlib import Path
import pwd
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def render_unit(template, user, root):
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
        raise ValueError('invalid service username')
    root = str(root)
    if not root.startswith('/') or any(char in root for char in '\n\r"\\%') or ' ' in root:
        raise ValueError('deployment path must be absolute, without spaces or systemd escapes')
    for token, value in {'USER': user, 'ROOT': root,
                         'PYTHON': root+'/.venv/bin/python',
                         'SCRIPT': root+'/scripts/xbox_autostart.py'}.items():
        template = template.replace('@'+token+'@', value)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user', default=os.environ.get('SUDO_USER'))
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    if os.geteuid() != 0 or not args.user or args.user == 'root':
        parser.error('run with sudo and --user <normal login username>')
    pwd.getpwnam(args.user)
    for path in (ROOT/'.venv/bin/python', ROOT/'robot.json',
                 ROOT/'scripts/xbox_autostart.py', ROOT/'xbox_friction_5000_hd1910.sh'):
        if not path.exists():
            raise SystemExit('Missing deployment file: '+str(path))
    unit = render_unit((ROOT/'systemd/microduck-xbox.service.in').read_text(), args.user, ROOT)
    target = Path('/etc/systemd/system/microduck-xbox.service')
    if target.exists() and target.read_text() != unit:
        target.with_suffix('.service.previous').write_text(target.read_text())
    target.write_text(unit)
    config = Path('/etc/default/microduck-xbox')
    if not config.exists():
        config.write_text('# flat or backlash, friction randomized 5000-iteration models\n'
                          'MICRODUCK_VARIANT=backlash\nMICRODUCK_SECONDS=0\nMICRODUCK_MAX_VX=0.3\n')
    subprocess.run(['systemd-analyze', 'verify', str(target)], check=True)
    subprocess.run(['systemctl', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', 'enable', 'microduck-xbox.service'], check=True)
    if args.start:
        subprocess.run(['systemctl', 'start', 'microduck-xbox.service'], check=True)
    print('开机自启动已启用；等待手柄和新的A键，不自动回HOME。')


if __name__ == '__main__':
    main()
