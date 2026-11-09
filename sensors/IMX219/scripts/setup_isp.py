#!/usr/bin/env python3
"""Radxa Zero 3 vendor 6.1 ISP setup; preview by default, --apply installs.

The accompanying capture_frame.sh owns exposure; rkaiq owns image colour.
Only camera userspace packages/configuration are installed, with a dated backup.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import urllib.request

POOL = 'https://radxa-repo.github.io/bullseye/pool/main/'
PACKAGES = [
    ('c/camera-engine-rkaiq/camera_engine_rkaiq_rk3568_arm64-fixed.deb',
     '5c03ca9b857d53a3c6917402efc3e37e130d0e0cfad701b9b2b96a31eaf01081'),
    ('r/rockchip-iqfiles/rockchip-iqfiles-rk356x_0.1.16_all.deb',
     '9a52cff23f336ba0671588edcaf803f1ffb3f695e466137622316ffceeda8eda'),
]
IQ = Path('/etc/iqfiles/imx219_rpi-camera-v2_default.json')
DROPIN = Path('/etc/systemd/system/rkaiq_3A.service.d/90-microduck-capture.conf')
PIN = Path('/usr/local/bin/microduck-pin-camera-mode')
SHIM = Path('/usr/local/lib/microduck-rkaiq-modinfo-shim.so')
ROOT = Path(__file__).resolve().parent.parent
SYSTEMD_ROOTS = tuple(Path(p) for p in ('/etc/systemd/system', '/run/systemd/system',
                                      '/usr/local/lib/systemd/system',
                                      '/usr/lib/systemd/system', '/lib/systemd/system'))

def run(args, check=True, timeout=180):
    print('COMMAND', args, flush=True)
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    print(r.stdout, r.stderr, 'EXIT', r.returncode, flush=True)
    if check and r.returncode:
        raise RuntimeError('Command failed: ' + repr(args))
    return r

def atomic(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            tmp.chmod(mode)
            tmp.replace(path)  # Replace symlink itself, never the package-owned target.
        finally:
            tmp.unlink(missing_ok=True)


def patch_iq(raw):
    """Only parser compatibility and AE ownership; keep official colour calibration."""
    raw = raw.replace(b'AECV2_STRATEGY_MODE_LOWLIGHT_PRIOR', b'AECV2_STRATEGY_MODE_LOWLIGHT')
    raw = raw.replace(b'CALIB_AWB_HDR_FRAME_CHOOSE_MODE_AUTO', b'CALIB_AWB_HDR_FR_CH_AUTO')
    raw, count = re.subn(rb'("ae_calib"\s*:\s*\{\s*"CommCtrl"\s*:\s*\{\s*"Enable"\s*:\s*)[01]',
                         rb'\g<1>0', raw)
    if count != 1:
        raise RuntimeError('Unexpected IQ auto-exposure configuration')
    json.loads(raw)
    return raw


def service_properties(names, allow_missing=False):
    result = run(['systemctl', 'show', 'rkaiq_3A.service', '--property=' + ','.join(names)], check=False)
    properties = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode and not (allow_missing and properties.get('LoadState') == 'not-found'):
        raise RuntimeError('Cannot query rkaiq service state: ' + result.stderr)
    return properties


def check_dropins(properties=None, roots=SYSTEMD_ROOTS):
    """Stop before modifying a board whose camera service has other local policy.

    A robot.conf sorts after 90-microduck-capture.conf. Do not assume numeric
    prefixes win, or remove settings installed by the main Microduck stack.
    Inspect both the loaded configuration and files awaiting daemon-reload.
    """
    if properties is None:
        properties = service_properties(('LoadState', 'DropInPaths'), allow_missing=True)
    if properties.get('LoadState') not in ('loaded', 'not-found'):
        raise RuntimeError('Cannot safely replace rkaiq service configuration: ' + repr(properties))
    paths = set(shlex.split(properties.get('DropInPaths', '')))
    for root in roots:
        for directory in ('rkaiq_3A.service.d', 'service.d'):
            paths.update(str(p) for p in (root / directory).glob('*.conf'))
    foreign = sorted(p for p in paths if Path(p) != DROPIN)
    if foreign:
        raise RuntimeError('Existing rkaiq/system service drop-ins require review; nothing was disabled or overwritten:\n'
                           + '\n'.join(foreign)
                           + '\nRun systemctl cat rkaiq_3A.service to inspect them. '
                             'Do not mix this standalone capture setup with upstream robot.conf/mediad configuration.')


def verify_effective_service(properties):
    """Validate the actual systemd result before starting this capture stack."""
    expected = {'ExecStart': ['/usr/bin/rkaiq_3A_server'], 'ExecStartPre': [str(PIN)],
                'ExecStartPost': [], 'ExecStop': []}
    if properties.get('Type') != 'simple':
        raise RuntimeError('Effective rkaiq Type is not simple: ' + repr(properties))
    for key, paths in expected.items():
        actual = re.findall(r'\bpath=([^\s;]+)', properties.get(key, ''))
        if actual != paths or (not paths and properties.get(key, '').strip()):
            raise RuntimeError('Unexpected effective ' + key + ': ' + repr(properties.get(key)))
    environment = shlex.split(properties.get('Environment', ''))
    preload = [item for item in environment if item.startswith('LD_PRELOAD=')]
    if preload != ['LD_PRELOAD=' + str(SHIM)]:
        raise RuntimeError('Effective LD_PRELOAD differs from this package: ' + repr(preload))
    if properties.get('EnvironmentFiles', '').strip() or properties.get('UnsetEnvironment', '').strip():
        raise RuntimeError('Extra service environment configuration requires review: ' + repr(properties))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--package-dir', type=Path,
                        help='离线 .deb 目录；默认优先使用包内 packages/，不存在时从固定官方 URL 下载')
    args = parser.parse_args()
    if args.package_dir is None:
        bundled = ROOT / 'packages'
        if any((bundled / Path(path).name).exists() for path, _ in PACKAGES):
            args.package_dir = bundled
    print(json.dumps({'sensor': 'IMX219', 'mode': '1920x1080', 'isp': 'rkaiq with matched IQ',
                      'exposure_owner': 'capture_frame.sh', 'apply': args.apply,
                      'package_dir': str(args.package_dir) if args.package_dir else None,
                      'packages': [POOL+p for p,_ in PACKAGES]}, indent=2), flush=True)
    if not args.apply:
        return
    model = Path('/proc/device-tree/model').read_bytes().replace(b'\0', b'').decode()
    if os.geteuid() != 0 or 'Radxa ZERO 3' not in model or platform.machine() != 'aarch64':
        raise RuntimeError('Run as root on Radxa ZERO 3 aarch64')
    if not platform.release().startswith('6.1.') or 'vendor' not in platform.release():
        raise RuntimeError('This setup is validated for the vendor 6.1 kernel only')
    names = {str('/dev/' + p.parent.name): p.read_text().strip()
             for p in Path('/sys/class/video4linux').glob('video*/name')}
    mainpaths = [p for p, name in names.items() if name == 'rkisp_mainpath']
    if len(mainpaths) != 1:
        raise RuntimeError('Camera overlay / rkisp_mainpath is not ready')
    module = Path('/proc/device-tree/i2c@fe5b0000/camera-imx219@10/rockchip,camera-module-name').read_bytes().rstrip(b'\0')
    if module != b'rpi-camera-v2':
        raise RuntimeError('Camera module does not match supplied IQ: ' + repr(module))
    for tool in ('gcc', 'v4l2-ctl', 'media-ctl', 'fuser', 'dpkg', 'dpkg-query',
                 'systemctl', 'ldd', 'ldconfig'):
        if not shutil.which(tool):
            raise RuntimeError('Missing prerequisite: ' + tool)
    capture_nodes = [p for p, name in names.items() if name in ('rkisp_mainpath', 'rkisp_selfpath')]
    if run(['fuser', *capture_nodes], check=False).returncode != 1:
        raise RuntimeError('Close the camera capture process before setup')
    if run(['systemctl', 'is-active', '--quiet', 'mediad'], check=False).returncode == 0:
        raise RuntimeError('mediad is active; stop its camera capture before selecting this standalone capture stack')
    check_dropins()  # Before backups, package installation, or any service change.
    shim_source = Path(__file__).resolve().parent.parent/'vendor/rkaiq-modinfo-shim.c'
    if not shim_source.is_file():
        raise RuntimeError('Keep the whole IMX219 directory; vendor/rkaiq-modinfo-shim.c is required')
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    backup = Path('/var/backups/microduck-camera-isp')/stamp
    backup.mkdir(parents=True)
    previous = {}
    for path in (IQ, DROPIN, PIN, SHIM):
        previous[str(path)] = {'exists': path.exists() or path.is_symlink(),
                               'symlink': os.readlink(path) if path.is_symlink() else None}
        if path.exists() or path.is_symlink():
            destination = backup/str(path).lstrip('/')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination, follow_symlinks=False)
    (backup/'files-before.json').write_text(json.dumps(previous, indent=2)+'\n')
    (backup/'service-before.txt').write_text(run(['systemctl', 'status', 'rkaiq_3A', '--no-pager'], check=False).stdout)
    print('BACKUP', backup, flush=True)
    with tempfile.TemporaryDirectory(prefix='microduck-isp-debs-') as directory:
        debs = []
        for path, digest in PACKAGES:
            local = args.package_dir/Path(path).name if args.package_dir else Path(directory)/Path(path).name
            if not args.package_dir:
                with urllib.request.urlopen(POOL+path, timeout=30) as response:
                    local.write_bytes(response.read())
            if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Package SHA256 mismatch: ' + str(local))
            debs.append(str(local.resolve()))
        installed = run(['dpkg-query', '-W', '-f=${db:Status-Status}', 'libdrm2'], check=False)
        if installed.stdout.strip() != 'installed':
            run(['apt-get', 'install', '-y', '--no-install-recommends', 'libdrm2'])
        run(['systemctl', 'stop', 'rkaiq_3A'], check=False)
        run(['dpkg', '-i', *debs])
    run(['ldconfig'])
    if 'not found' in run(['ldd', '/usr/bin/rkaiq_3A_server']).stdout:
        raise RuntimeError('rkaiq runtime library dependency is missing')
    source = Path('/usr/share/rockchip-iqfiles-rk356x')/IQ.name
    raw = patch_iq(source.read_bytes())
    atomic(IQ, raw)
    with tempfile.TemporaryDirectory(prefix='microduck-isp-shim-') as directory:
        built = Path(directory)/'shim.so'
        run(['gcc', '-shared', '-fPIC', '-O2', '-o', str(built), str(shim_source), '-ldl'])
        atomic(SHIM, built.read_bytes(), 0o755)
    atomic(PIN, b'''#!/usr/bin/env python3
import glob, re, subprocess, time
for attempt in range(40):
    for node in glob.glob('/dev/media*'):
        result = subprocess.run(['media-ctl', '-d', node, '-p'], capture_output=True, text=True, timeout=5)
        match = re.search(r'^- entity [0-9]+: (m[0-9]+_[bf]_imx219 [0-9-]+)', result.stdout, re.M)
        if match:
            subprocess.run(['media-ctl', '-d', node, '--set-v4l2', '"'+match[1]+'":0[fmt:SRGGB10_1X10/1920x1080]'], check=True, timeout=5)
            print('Pinned', match[1], '1920x1080', flush=True)
            raise SystemExit(0)
    time.sleep(0.25)
raise SystemExit('IMX219 sensor entity not found')
''', 0o755)
    unit = f'''# Camera colour via rkaiq, exposure via capture_frame.sh.
[Unit]
StartLimitIntervalSec=30
StartLimitBurst=3
[Service]
Type=simple
Environment=LD_PRELOAD={SHIM}
ExecStart=
ExecStart=/usr/bin/rkaiq_3A_server
ExecStop=
ExecStartPre=
ExecStartPre={PIN}
ExecStartPost=
Restart=on-failure
RestartSec=2
StandardOutput=journal
StandardError=journal
'''
    atomic(DROPIN, unit.encode())
    run(['systemctl', 'daemon-reload'])
    check_dropins()
    effective = service_properties(('Type', 'ExecStart', 'ExecStartPre', 'ExecStartPost', 'ExecStop',
                                    'Environment', 'EnvironmentFiles', 'UnsetEnvironment', 'DropInPaths'))
    verify_effective_service(effective)
    run(['systemctl', 'enable', 'rkaiq_3A'])
    run(['systemctl', 'restart', 'rkaiq_3A'])
    time.sleep(2)
    run(['systemctl', 'is-active', 'rkaiq_3A'])
    run(['journalctl', '-u', 'rkaiq_3A', '-n', '45', '--no-pager'])
    report = dict(backup=str(backup), iq_sha256=hashlib.sha256(IQ.read_bytes()).hexdigest(),
                  sensor_mode='1920x1080', exposure_owner='capture_frame.sh', effective_service=effective)
    (backup/'installed.json').write_text(json.dumps(report, indent=2)+'\n')
    print('INSTALLED', json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
