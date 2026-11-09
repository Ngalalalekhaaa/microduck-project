"""Offline regressions for transfer, boot preservation, and IQ ownership.

Run with: python3 -m unittest discover -s tests -v
No robot connection or privileges are required.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


boot, isp, doctor = load('enable_camera'), load('setup_isp'), load('doctor')
CAMERA = '/boot/dtb/rockchip/overlay/radxa-zero3-rpi-camera-v2.dtbo'
BASE = '''timeout 10
default primary
label primary
    linux /boot/vmlinuz-6.1.115-vendor-rk35xx
    initrd /boot/initrd.img-6.1.115-vendor-rk35xx
    fdt /boot/dtb/rockchip/rk3566-radxa-zero3.dtb
    append root=UUID=keep console=ttyS2,1500000 cma=256M
'''
UART = '/boot/dtb/rockchip/overlay/rk3568-uart2-m0.dtbo'
J5 = '/boot/overlay-user/microduck-j5-i2c.dtbo'


class BootTests(unittest.TestCase):
    def test_existing_robot_overlays_and_arguments_preserved(self):
        original = BASE + '    fdtoverlays ' + UART + ' ' + J5 + '\n# keep this comment\n'
        _, old, new, planned = boot.boot_plan(original, CAMERA)
        self.assertEqual(old, [UART, J5])
        self.assertEqual(new, [UART, J5, CAMERA])
        self.assertEqual(planned, original.replace(J5, J5 + ' ' + CAMERA))
        self.assertEqual(boot.boot_plan(planned, CAMERA)[3], planned)

    def test_fresh_board_without_j5_or_existing_overlays(self):
        _, old, new, planned = boot.boot_plan(BASE, CAMERA)
        self.assertEqual(old, [])
        self.assertEqual(new, [CAMERA])
        self.assertTrue(planned.startswith(BASE))

    def test_empty_overlay_directive_gets_a_separator(self):
        text = BASE + '    fdtoverlays\n'
        planned = boot.boot_plan(text, CAMERA)[3]
        self.assertIn('fdtoverlays ' + CAMERA + '\n', planned)
        self.assertEqual(boot.parse_boot(planned)[1], [CAMERA])

    def test_already_installed_custom_overlay_is_reused(self):
        text = BASE + '    fdtoverlays /boot/overlay-user/microduck-media-imx219.dtbo\n'
        self.assertEqual(boot.boot_plan(text, CAMERA)[3], text)

    def test_ambiguous_or_conflicting_layouts_are_rejected(self):
        cases = [BASE + 'label other\n', BASE.replace('    fdt ', '    fdtdir '),
                 BASE + '    fdtoverlays relative.dtbo\n',
                 BASE + '    fdtoverlays /boot/overlay/camera-v1.3.dtbo\n',
                 BASE + '    fdtoverlays ' + UART + '\n    fdtoverlays ' + J5 + '\n',
                 'fdtoverlays ' + UART + '\n' + BASE,
                 BASE + '    fdtoverlays ' + UART + ' # comment\n',
                 BASE + '    fdtoverlays ' + UART + ' ' + UART + '\n', BASE + '\0']
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                boot.boot_plan(text, CAMERA)

    def test_write_backup_idempotence_and_concurrent_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'extlinux.conf'
            path.write_text(BASE)
            path.chmod(0o640)
            after = boot.boot_plan(BASE, CAMERA)[3]
            backup = boot.write_boot(path, BASE, after, Path(directory) / 'backups')
            self.assertEqual((backup / 'extlinux.conf').read_text(), BASE)
            self.assertEqual(path.read_text(), after)
            self.assertEqual(path.stat().st_mode & 0o777, 0o640)
            self.assertIsNone(boot.write_boot(path, after, after, Path(directory) / 'backups'))
            with self.assertRaisesRegex(ValueError, '变化'):
                boot.write_boot(path, BASE, after, Path(directory) / 'backups')

    def test_tree_guard_rejects_changed_j5_pin_selection(self):
        def fake_get(tree, node, prop, kind='s', optional=False):
            if node == '/__symbols__':
                return '/' + prop
            if tree == 'after' and node == '/i2c3' and prop == 'pinctrl-0':
                return '999'
            return 'okay' if prop == 'status' else '1'
        with patch.object(boot, 'get', side_effect=fake_get):
            with self.assertRaisesRegex(ValueError, 'pinctrl-0'):
                boot.validate_trees('before', 'after')

    @unittest.skipUnless(all(shutil.which(x) for x in ('dtc', 'fdtoverlay', 'fdtget')),
                         'device-tree-compiler required for real DT merge fixture')
    def test_real_device_tree_merge_preserves_bus_and_rejects_changed_pin(self):
        # Synthetic DT, deliberately unrelated to any real board pinout.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / 'base.dts'
            source.write_text('''/dts-v1/;
/ {
  pins_a: pins_a { };
  pins_b: pins_b { };
  i2c3: i2c { status = "okay"; pinctrl-0 = <&pins_a>; clock-frequency = <400000>;
    fusb302@22 { status = "disabled"; };
  };
  uart2: uart { status = "okay"; pinctrl-0 = <&pins_b>; };
  rkisp: isp { status = "disabled"; };
  rkisp_vir0: isp_virtual { status = "disabled"; };
  csi2_dphy0: csi_phy { status = "disabled"; };
};''')
            base = path / 'base.dtb'
            boot.command('dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', base, source)
            for conflict in (False, True):
                overlay = path / 'camera.dts'
                overlay.write_text('''/dts-v1/; /plugin/;
&rkisp { status = "okay"; };
&rkisp_vir0 { status = "okay"; };
&csi2_dphy0 { status = "okay"; camera { compatible = "sony,imx219"; }; };
''' + ('&i2c3 { pinctrl-0 = <&pins_b>; };' if conflict else ''))
                compiled, merged = path / 'camera.dtbo', path / 'merged.dtb'
                boot.command('dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', compiled, overlay)
                boot.command('fdtoverlay', '-i', base, '-o', merged, compiled)
                if conflict:
                    with self.assertRaisesRegex(ValueError, 'pinctrl-0'):
                        boot.validate_trees(base, merged)
                else:
                    checked = boot.validate_trees(base, merged)
                    self.assertEqual(checked['i2c3.clock-frequency'], '400000')
                    self.assertEqual(checked['i2c3.fusb302.status'], 'disabled')


class IspTests(unittest.TestCase):
    def test_foreign_dropin_on_disk_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conf = root / 'rkaiq_3A.service.d/robot.conf'
            conf.parent.mkdir()
            text = '[Service]\nEnvironment=LD_PRELOAD=/usr/local/lib/old-shim.so\n'
            conf.write_text(text)
            with self.assertRaisesRegex(RuntimeError, 'robot.conf'):
                isp.check_dropins({'LoadState': 'loaded', 'DropInPaths': ''}, roots=(root,))
            self.assertEqual(conf.read_text(), text)

    def test_loaded_and_global_dropins_are_also_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'custom.conf'):
            isp.check_dropins({'LoadState': 'loaded',
                               'DropInPaths': '/run/systemd/system/rkaiq_3A.service.d/custom.conf'}, roots=())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conf = root / 'service.d/global.conf'
            conf.parent.mkdir()
            conf.write_text('[Service]\nEnvironmentFile=/tmp/environment\n')
            with self.assertRaisesRegex(RuntimeError, 'global.conf'):
                isp.check_dropins({'LoadState': 'not-found', 'DropInPaths': ''}, roots=(root,))

    def test_fresh_board_missing_unit_and_existing_own_dropin_are_allowed(self):
        result = subprocess.CompletedProcess(['systemctl'], 1,
                                             'LoadState=not-found\nDropInPaths=\n', '')
        with patch.object(isp, 'run', return_value=result):
            isp.check_dropins(roots=())
        isp.check_dropins({'LoadState': 'loaded', 'DropInPaths': str(isp.DROPIN)}, roots=())
        with patch.object(isp, 'run', return_value=subprocess.CompletedProcess(['systemctl'], 1, '', 'no bus')):
            with self.assertRaisesRegex(RuntimeError, 'no bus'):
                isp.check_dropins(roots=())

    def test_effective_service_rejects_wrong_shim_or_added_start_hooks(self):
        good = {'Type': 'simple', 'ExecStart': '{ path=/usr/bin/rkaiq_3A_server ; argv[]=/usr/bin/rkaiq_3A_server ; }',
                'ExecStartPre': '{ path=' + str(isp.PIN) + ' ; argv[]=' + str(isp.PIN) + ' ; }',
                'ExecStartPost': '', 'ExecStop': '', 'Environment': 'LD_PRELOAD=' + str(isp.SHIM),
                'EnvironmentFiles': '', 'UnsetEnvironment': ''}
        isp.verify_effective_service(good)
        cases = [('Environment', 'LD_PRELOAD=/usr/local/lib/old.so'),
                 ('ExecStartPre', good['ExecStartPre'] + ' { path=/bin/old-hook ; argv[]=/bin/old-hook ; }'),
                 ('ExecStartPost', '{ path=/bin/old-hook ; argv[]=/bin/old-hook ; }'),
                 ('EnvironmentFiles', '/etc/old-camera-environment'), ('Type', 'forking')]
        for key, value in cases:
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                isp.verify_effective_service(dict(good, **{key: value}))

    def test_iq_permission_failure_is_reported_by_doctor(self):
        with patch.object(Path, 'read_bytes', side_effect=PermissionError('permission denied')):
            result = doctor.inspect_iq(Path('/etc/iqfiles/example.json'))
        self.assertIn('permission denied', result['error'])

    def test_atomic_iq_replaces_link_without_changing_package_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'package-iq.json'
            source.write_bytes(b'original')
            target = Path(directory) / 'etc-iq.json'
            target.symlink_to(source)
            isp.atomic(target, b'patched')
            self.assertEqual(source.read_bytes(), b'original')
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), b'patched')

    def test_iq_only_changes_ae_and_known_enum_spellings(self):
        original = {'ae_calib': {'CommCtrl': {'Enable': 1},
                                'strategy': 'AECV2_STRATEGY_MODE_LOWLIGHT_PRIOR'},
                    'awb_calib': {'strategy': 'CALIB_AWB_HDR_FRAME_CHOOSE_MODE_AUTO',
                                  'gain': [1.1, 1, 1.3]},
                    'ccm': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    'unrelated': {'CommCtrl': {'Enable': 1}}}
        patched = json.loads(isp.patch_iq(json.dumps(original).encode()))
        self.assertEqual(patched['ae_calib']['CommCtrl']['Enable'], 0)
        self.assertEqual(patched['awb_calib']['gain'], original['awb_calib']['gain'])
        self.assertEqual(patched['ccm'], original['ccm'])
        self.assertEqual(patched['unrelated'], original['unrelated'])
        with self.assertRaises(RuntimeError):
            isp.patch_iq(b'{"ae_calib": {"unexpected": 1}}')

    def test_preview_is_portable_and_does_not_require_root(self):
        result = subprocess.run(['python3', str(ROOT / 'scripts/setup_isp.py')],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['apply'])
        self.assertTrue((isp.ROOT / 'vendor/rkaiq-modinfo-shim.c').is_file())


class CaptureTests(unittest.TestCase):
    def test_explicit_target_required_and_bad_target_rejected(self):
        for args in ([], ['-oProxyCommand=bad'], ['$(echo bad)@host'], ['user@host', 'a', 'b']):
            with self.subTest(args=args):
                result = subprocess.run(['bash', str(ROOT / 'scripts/capture_frame.sh'), *args],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'local prerequisites must be present')
    def test_existing_output_is_never_overwritten_or_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / 'frame.png'
            prior.write_bytes(b'previous photo')
            result = subprocess.run(['bash', str(ROOT / 'scripts/capture_frame.sh'),
                                     'colleague@192.0.2.10', directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(prior.read_bytes(), b'previous photo')
            self.assertFalse((Path(directory) / 'capture.json').exists())

    def test_embedded_remote_python_and_bash_syntax(self):
        script = ROOT / 'scripts/capture_frame.sh'
        result = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = script.read_text().split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        compile(source, 'remote_capture', 'exec')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for transfer/decode fixture')
    def test_portable_transfer_hash_and_failure_log_retrieval(self):
        # The transport is a fixture; real ffmpeg verifies metadata-driven decoding.
        # The IMX219 capture/AE hardware loop is deliberately not claimed as tested here.
        for success in (True, False):
            with self.subTest(success=success), tempfile.TemporaryDirectory(prefix='camera space ') as directory:
                work = Path(directory)
                portable = work / 'copy of package'
                (portable / 'scripts').mkdir(parents=True)
                shutil.copy2(ROOT / 'scripts/capture_frame.sh', portable / 'scripts')
                fixture = work / 'remote'
                fixture.mkdir()
                raw = bytes([100] * 16 + [128] * 8)
                (fixture / 'frame.nv12').write_bytes(raw)
                (fixture / 'capture.log').write_text('fixture: complete diagnostic log\n')
                metadata = dict(success=success, remote_log='/tmp/fixture/capture.log',
                                raw_path='/tmp/fixture/frame.nv12', width=4, height=4,
                                bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                y_min=100, y_max=100, y_mean=100, ae_converged=True,
                                requested_controls={'exposure': 500, 'analogue_gain': 256},
                                warnings=[], color_range='pc')
                (fixture / 'capture.json').write_text(json.dumps(metadata))
                fakebin = work / 'bin'
                fakebin.mkdir()
                (fakebin / 'ssh').write_text('''#!/usr/bin/env python3
import json,os,pathlib,sys
sys.stdin.read()
p=pathlib.Path(os.environ['FIXTURE'])/'capture.json'
print(p.read_text())
sys.exit(0 if json.loads(p.read_text())['success'] else 1)
''')
                (fakebin / 'scp').write_text('''#!/usr/bin/env python3
import os,pathlib,shutil,sys
shutil.copyfile(pathlib.Path(os.environ['FIXTURE'])/sys.argv[-2].rsplit('/',1)[-1],sys.argv[-1])
''')
                for executable in fakebin.iterdir():
                    executable.chmod(0o755)
                environment = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ['PATH'], FIXTURE=str(fixture))
                result = subprocess.run(['bash', str(portable / 'scripts/capture_frame.sh'), 'colleague@192.0.2.10'],
                                        env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0 if success else 1, result.stdout + result.stderr)
                outputs = list((portable / 'captures').glob('*/*'))
                self.assertEqual(len(outputs), 1)
                self.assertEqual((outputs[0] / 'capture.log').read_text(), 'fixture: complete diagnostic log\n')
                if success:
                    self.assertEqual((outputs[0] / 'frame.nv12').read_bytes(), raw)
                    self.assertTrue((outputs[0] / 'frame.png').read_bytes().startswith(b'\x89PNG'))
                else:
                    self.assertFalse((outputs[0] / 'frame.png').exists())


if __name__ == '__main__':
    unittest.main()
