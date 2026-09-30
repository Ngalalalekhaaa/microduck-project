"""Checks using temporary boot trees only; no access to the host's /boot.

Run: python3 -m unittest discover -s radxa_j5_i2c_fix -p 'test_*.py' -v
The device-tree tests need dtc, fdtoverlay and fdtget on PATH.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import enable_j5 as fix


ENV = """verbosity=1
bootlogo=false
console=display
extraargs=cma=256M
overlay_prefix=rk3568
fdtfile=rockchip/rk3566-radxa-zero3.dtb
rootdev=UUID=34d3c17e-5271-473f-acb6-7f5d55b3bb28
rootfstype=ext4
overlays=uart2-m0 dwc3-peripheral
usbstoragequirks=0x2537:0x1066:u,0x2537:0x1068:u
"""
EXT = """default armbian
timeout 3
menu title Armbian (Radxa U-Boot)

label armbian
    linux /boot/vmlinuz-6.1.115-vendor-rk35xx
    initrd /boot/initrd.img-6.1.115-vendor-rk35xx
    fdt /boot/dtb/rockchip/rk3566-radxa-zero3.dtb
    fdtoverlays /boot/dtb/rockchip/overlay/rk3568-uart2-m0.dtbo /boot/dtb/rockchip/overlay/rk3568-dwc3-peripheral.dtbo
    append root=UUID=34d3c17e-5271-473f-acb6-7f5d55b3bb28 rootwait rw console=tty1 consoleblank=0 cma=256M
"""


def snapshot(tree):
    return {str(p.relative_to(tree)): p.read_bytes()
            for p in tree.rglob('*') if p.is_file()}


class ConfigPlanTests(unittest.TestCase):
    def test_exact_reported_layout_preserves_other_options(self):
        result = fix.plan_configs(ENV, EXT)
        self.assertEqual(result[fix.ENV], ENV + f'user_overlays={fix.NAME}\n')
        old = ' '.join(fix.OLD_PATHS)
        self.assertEqual(result[fix.EXT], EXT.replace(old, old + ' ' + fix.OVERLAY))

    def test_repeated_plan_is_identical(self):
        first = fix.plan_configs(ENV, EXT)
        self.assertEqual(fix.plan_configs(first[fix.ENV], first[fix.EXT]), first)

    def test_empty_user_overlays_is_replaced_without_duplicate(self):
        result = fix.plan_configs(ENV + 'user_overlays=\n', EXT)
        self.assertEqual(result[fix.ENV], ENV + f'user_overlays={fix.NAME}\n')

    def test_incompatible_env_is_rejected(self):
        invalid = [
            ENV.replace('overlay_prefix=rk3568', 'overlay_prefix=rk3588'),
            ENV.replace('rk3566-radxa-zero3.dtb', 'rk3566-other.dtb'),
            ENV.replace('overlays=uart2-m0 dwc3-peripheral', 'overlays=uart2-m0'),
            ENV + 'user_overlays=i2c-gpio-pihat\n',
            ENV + 'overlay_prefix=rk3568\n',
        ]
        for env in invalid:
            with self.subTest(env=env), self.assertRaises(ValueError):
                fix.plan_configs(env, EXT)

    def test_incompatible_extlinux_is_rejected(self):
        invalid = [
            EXT.replace('default armbian', 'default rescue'),
            EXT + '\nlabel rescue\n',
            EXT.replace(fix.BASE, '/boot/dtb/other.dtb'),
            EXT.replace('rk3568-uart2-m0.dtbo', 'rk3568-uart2-m1.dtbo'),
            EXT.replace('    fdtoverlays', '    # fdtoverlays'),
            EXT + '    fdtoverlays ' + ' '.join(fix.OLD_PATHS) + '\n',
        ]
        for ext in invalid:
            with self.subTest(ext=ext), self.assertRaises(ValueError):
                fix.plan_configs(ENV, ext)


class TemporaryBootTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-microduck-j5-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'root'
        self.work = Path(self.temp.name) / 'compile'
        self.work.mkdir()
        self.original = {fix.ENV: ENV.encode(), fix.EXT: EXT.encode()}
        for name, data in self.original.items():
            target = self.root / name.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o640)
        self.replacements = fix.plan_configs(ENV, EXT)
        self.overlay = self.root / fix.OVERLAY.lstrip('/')

    def apply(self, dtbo=b'test-dtbo', original=None, replacements=None):
        with patch.object(fix.os, 'sync'), redirect_stdout(io.StringIO()):
            return fix.apply_files(self.root, original or self.original,
                                   replacements or self.replacements, dtbo)


class ApplyTests(TemporaryBootTests):
    def test_apply_saves_backups_and_preserves_config_modes(self):
        backup = self.apply()
        self.assertEqual(backup.parent, self.root / 'boot')
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        for name, before in self.original.items():
            self.assertEqual((backup / Path(name).name).read_bytes(), before)
            target = self.root / name.lstrip('/')
            self.assertEqual(target.read_text(), self.replacements[name])
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
        self.assertEqual(self.overlay.read_bytes(), b'test-dtbo')

    def test_second_apply_is_noop_without_new_backup(self):
        self.apply()
        before = snapshot(self.root)
        current = {name: (self.root / name.lstrip('/')).read_bytes() for name in self.original}
        result = self.apply(original=current)
        self.assertIsNone(result)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(len(list((self.root / 'boot').glob('microduck-j5-backup-*'))), 1)

    def test_existing_different_overlay_rejected_before_writes(self):
        self.overlay.parent.mkdir(parents=True)
        self.overlay.write_bytes(b'conflicting-overlay')
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(snapshot(self.root), before)

    def test_symlink_overlay_rejected_before_writes(self):
        self.overlay.parent.mkdir(parents=True)
        linked = self.work / 'other.dtbo'
        linked.write_bytes(b'test-dtbo')
        self.overlay.symlink_to(linked)
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(snapshot(self.root), before)

    def test_concurrent_config_change_rejected_before_writes(self):
        changed = self.root / fix.EXT.lstrip('/')
        changed.write_bytes(changed.read_bytes() + b'# concurrently updated\n')
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.apply()
        self.assertEqual(snapshot(self.root), before)

    def test_second_config_write_failure_rolls_back_first_config_and_new_overlay(self):
        real_write = fix.atomic_write
        failed = False
        first_was_updated = False

        def fail_once(path, data):
            nonlocal failed, first_was_updated
            if path == self.root / fix.EXT.lstrip('/') and not failed:
                failed = True
                first_was_updated = ((self.root / fix.ENV.lstrip('/')).read_text()
                                     == self.replacements[fix.ENV])
                raise OSError('injected failure on second configuration write')
            return real_write(path, data)

        with patch.object(fix, 'atomic_write', side_effect=fail_once):
            with self.assertRaisesRegex(ValueError, 'injected failure'):
                self.apply()
        self.assertTrue(failed)
        self.assertTrue(first_was_updated)
        self.assertFalse(self.overlay.exists())
        for name, before in self.original.items():
            self.assertEqual((self.root / name.lstrip('/')).read_bytes(), before)
        backups = list((self.root / 'boot').glob('microduck-j5-backup-*'))
        self.assertEqual(len(backups), 1)
        for name, before in self.original.items():
            self.assertEqual((backups[0] / Path(name).name).read_bytes(), before)

    def test_rollback_failure_keeps_overlay_and_still_restores_other_config(self):
        real_write = fix.atomic_write
        env_path = self.root / fix.ENV.lstrip('/')
        ext_path = self.root / fix.EXT.lstrip('/')
        calls = []

        def fail_apply_and_one_restore(path, data):
            calls.append((path, data))
            if path == ext_path and data == self.replacements[fix.EXT].encode():
                raise OSError('injected apply failure')
            if path == env_path and data == self.original[fix.ENV]:
                raise OSError('injected environment restore failure')
            return real_write(path, data)

        with patch.object(fix, 'atomic_write', side_effect=fail_apply_and_one_restore):
            with self.assertRaisesRegex(ValueError, 'injected environment restore failure') as caught:
                self.apply()
        self.assertIn('microduck-j5-backup-', str(caught.exception))
        self.assertEqual(self.overlay.read_bytes(), b'test-dtbo')
        self.assertIn((ext_path, self.original[fix.EXT]), calls)
        self.assertEqual(ext_path.read_bytes(), self.original[fix.EXT])
        self.assertEqual(env_path.read_text(), self.replacements[fix.ENV])


@unittest.skipUnless(all(shutil.which(x) for x in ('dtc', 'fdtoverlay', 'fdtget')),
                     'device-tree-compiler tools are required')
class DeviceTreeTests(TemporaryBootTests):
    def compile_dts(self, text, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = self.work / (destination.name + '.dts')
        source.write_text(text)
        subprocess.run(['dtc', '-@', '-I', 'dts', '-O', 'dtb', '-o', str(destination), str(source)],
                       check=True, capture_output=True, text=True)

    def fixtures(self, *, m0_symbol=True, target_node=True, compatible='radxa,zero-3w', pins='1 1 1 &cfg 1 0 1 &cfg'):
        source = '''/dts-v1/;
        / {
            compatible = "COMPAT", "rockchip,rk3566";
            #address-cells = <1>;
            #size-cells = <1>;
            pinctrl {
                cfg: config { bias-pull-up; };
                M0LABELm0 { rockchip,pins = <PINS>; };
                i2c3m1_xfer: m1 { rockchip,pins = <3 13 4 &cfg 3 14 4 &cfg>; };
            };
            i2c3: i2c@fe5c0000 {
                reg = <0xfe5c0000 0x1000>;
                #address-cells = <1>;
                #size-cells = <0>;
                status = "okay";
                clock-frequency = <100000>;
                pinctrl-names = "default";
                pinctrl-0 = <&i2c3m1_xfer>;
                TARGET
            };
        };'''
        source = source.replace('COMPAT', compatible).replace('PINS', pins)
        source = source.replace('M0LABEL', 'i2c3m0_xfer: ' if m0_symbol else '')
        source = source.replace('TARGET', 'fusb302@22 { reg = <0x22>; status = "okay"; };' if target_node else '')
        self.compile_dts(source, self.root / fix.BASE.lstrip('/'))
        for index, path in enumerate(fix.OLD_PATHS):
            self.compile_dts('''/dts-v1/; /plugin/; / {
                fragment@0 { target-path = "/"; __overlay__ {
                    existing-overlay-INDEX = "preserved";
                }; };
            };'''.replace('INDEX', str(index)), self.root / path.lstrip('/'))

    def checked_compile(self):
        with redirect_stdout(io.StringIO()):
            return fix.compile_and_check(self.root, self.work)

    def test_real_overlay_selects_m0_disables_fusb302_preserves_old_overlays(self):
        self.fixtures()
        before = snapshot(self.root)
        dtbo = self.checked_compile()
        self.assertGreater(len(dtbo), 100)
        merged = self.work / 'preview.dtb'
        get = lambda node, prop: fix.command('fdtget', merged, node, prop)
        self.assertEqual(get('/i2c@fe5c0000', 'pinctrl-0'), get('/pinctrl/m0', 'phandle'))
        self.assertEqual(get('/i2c@fe5c0000/fusb302@22', 'status'), 'disabled')
        self.assertEqual(get('/i2c@fe5c0000', 'clock-frequency'), '400000')
        for index in range(2):
            self.assertEqual(get('/', f'existing-overlay-{index}'), 'preserved')
        self.assertEqual(snapshot(self.root), before)

    def test_m0_sda_first_order_is_also_accepted_without_boot_write(self):
        self.fixtures(pins='1 0 1 &cfg 1 1 1 &cfg')
        before = snapshot(self.root)
        dtbo = self.checked_compile()
        self.assertGreater(len(dtbo), 100)
        self.assertEqual(snapshot(self.root), before)

    def test_missing_m0_symbol_rejected_without_boot_write(self):
        self.fixtures(m0_symbol=False)
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.checked_compile()
        self.assertEqual(snapshot(self.root), before)

    def test_missing_fusb302_target_rejected_without_boot_write(self):
        self.fixtures(target_node=False)
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.checked_compile()
        self.assertEqual(snapshot(self.root), before)

    def test_other_vendor_board_rejected_without_boot_write(self):
        self.fixtures(compatible='other,rk3566-board')
        before = snapshot(self.root)
        with self.assertRaises(ValueError):
            self.checked_compile()
        self.assertEqual(snapshot(self.root), before)

    def test_incorrect_m0_pin_numbers_rejected_without_boot_write(self):
        self.assert_invalid_m0_pins('1 2 1 &cfg 1 3 1 &cfg')

    def assert_invalid_m0_pins(self, pins):
        self.fixtures(pins=pins)
        before = snapshot(self.root)
        actual = [int(value) for value in fix.command(
            'fdtget', '-t', 'u', self.root / fix.BASE.lstrip('/'),
            '/pinctrl/m0', 'rockchip,pins').split()]
        with self.assertRaises(ValueError) as caught:
            self.checked_compile()
        self.assertIn(str(actual), str(caught.exception))
        self.assertEqual(snapshot(self.root), before)

    def test_duplicate_m0_pin_is_rejected_without_boot_write(self):
        for pin in (0, 1):
            with self.subTest(pin=pin):
                self.assert_invalid_m0_pins(f'1 {pin} 1 &cfg 1 {pin} 1 &cfg')

    def test_wrong_m0_mux_function_is_rejected_without_boot_write(self):
        for pins in ('1 1 4 &cfg 1 0 1 &cfg', '1 1 1 &cfg 1 0 4 &cfg'):
            with self.subTest(pins=pins):
                self.assert_invalid_m0_pins(pins)

    def test_extra_m0_record_is_rejected_without_boot_write(self):
        self.assert_invalid_m0_pins('1 1 1 &cfg 1 0 1 &cfg 1 2 1 &cfg')

    def test_missing_m0_record_is_rejected_without_boot_write(self):
        self.assert_invalid_m0_pins('1 1 1 &cfg')

    def test_validated_dtbo_can_be_applied_with_config_backups(self):
        self.fixtures()
        dtbo = self.checked_compile()
        backup = self.apply(dtbo=dtbo)
        self.assertTrue(backup.is_dir())
        self.assertEqual(self.overlay.read_bytes(), dtbo)
        for name, replacement in self.replacements.items():
            self.assertEqual((self.root / name.lstrip('/')).read_text(), replacement)


if __name__ == '__main__':
    unittest.main(verbosity=2)
