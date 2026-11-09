import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('setup_j6', ROOT / 'scripts/setup_j6.py')
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)
EXT = ('default armbian\nlabel armbian\n  linux /boot/Image\n  fdt /boot/base.dtb\n'
       '  append root=UUID=example quiet\n'
       '  fdtoverlays /boot/uart.dtbo /boot/overlay-user/microduck-j5-i2c.dtbo /boot/camera.dtbo /boot/overlay-user/microduck-media-aic3104.dtbo\n')
ENV = ('overlays=uart2m0\n' + '\0' * 304 + '\n'
       'user_overlays=microduck-j5-i2c microduck-media-aic3104\nusbstoragequirks=example\n')


class BootTests(unittest.TestCase):
    def test_preservation_and_idempotence(self):
        base, old, new, ext, env = setup.boot_plan(EXT, ENV)
        self.assertEqual(base, '/boot/base.dtb')
        self.assertEqual(new, old + [str(setup.TARGET)])
        self.assertEqual(ext.replace(' ' + str(setup.TARGET), ''), EXT)
        self.assertEqual(env.replace(' ' + setup.NAME, ''), ENV)
        self.assertEqual(setup.boot_plan(ext, env)[3:], (ext, env))

    def test_reject_ambiguous_boot(self):
        for text in (EXT + 'label rescue\n', EXT + '  fdtoverlays /boot/x.dtbo\n',
                     EXT.replace('/boot/base.dtb', 'base.dtb'), EXT + '\0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                setup.boot_plan(text, ENV)
        with self.assertRaises(ValueError):
            setup.boot_plan(EXT, ENV + 'user_overlays=other\n')

    def test_preserve_nuls_when_inserting_env(self):
        original = 'other=value\n' + '\0' * 42
        env = setup.boot_plan(EXT, original)[4]
        self.assertTrue(env.endswith('\0' * 42))
        self.assertIn('other=value\nuser_overlays=', env)

    def test_rollback_and_concurrent_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / 'first', root / 'second'
            first.write_bytes(b'old')
            real_write = setup.atomic_write
            def fail_second(path, data, mode):
                if path == second:
                    raise OSError('simulated write failure')
                return real_write(path, data, mode)
            with patch.object(setup, 'atomic_write', side_effect=fail_second), self.assertRaises(OSError):
                setup.apply_files([(first, b'old', b'new'), (second, None, b'new')], root / 'backups')
            self.assertEqual(first.read_bytes(), b'old')
            self.assertFalse(second.exists())
            with self.assertRaises(ValueError):
                setup.apply_files([(first, b'stale', b'new')], root / 'backups')
            self.assertEqual(first.read_bytes(), b'old')


@unittest.skipUnless(all(shutil.which(x) for x in ('dtc', 'fdtoverlay')), 'requires device-tree-compiler')
class DeviceTreeTests(unittest.TestCase):
    def test_compiled_overlay_preserves_old_tree_and_uses_j6_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src, base, overlay, merged = (root / x for x in ('base.dts', 'base.dtb', 'j6.dtbo', 'merged.dtb'))
            src.write_text('''/dts-v1/;
            / {
                compatible = "radxa,zero3", "rockchip,rk3566";
                aliases { i2c3 = "/i2c-existing"; };
                gpio3: gpio { gpio-controller; #gpio-cells = <2>; };
                pinctrl: pinctrl { pcfg_pull_none: pull-none { bias-disable; }; };
                i2c3: i2c-existing { status = "okay"; clock-frequency = <400000>; };
                sound { status = "okay"; };
            };''')
            for args in (['dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', str(base), str(src)],
                         ['dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', str(overlay), str(ROOT / 'overlays' / (setup.NAME + '.dts'))],
                         ['fdtoverlay', '-i', str(base), '-o', str(merged), str(overlay)]):
                subprocess.run(args, check=True, capture_output=True)
            checks = setup.validate_trees(base.read_bytes(), merged.read_bytes())
            self.assertEqual(checks['bus'], '/dev/i2c-10')
            # Re-running on the already-merged tree is valid and changes nothing.
            setup.validate_trees(merged.read_bytes(), merged.read_bytes())
            # Altering any old subsystem must fail even if J6 itself is correct.
            broken = merged.read_bytes().replace(b'okay\0', b'fail\0', 1)
            with self.assertRaises(ValueError):
                setup.validate_trees(base.read_bytes(), broken)


if __name__ == '__main__':
    unittest.main()
