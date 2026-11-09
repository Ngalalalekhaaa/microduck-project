#!/usr/bin/env python3
"""No-hardware checks for audio format, dry-run and boot configuration preservation."""
from array import array
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


player, generator, installer = (module(name) for name in ('play', 'generate_speech', 'setup_audio'))


class SpeakerTests(unittest.TestCase):
    def test_installer_resolves_sbin_tools_in_non_login_ssh_path(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            bin_dir, sbin_dir = directory / 'bin', directory / 'sbin'
            bin_dir.mkdir()
            sbin_dir.mkdir()
            tool = sbin_dir / 'modinfo'
            tool.write_text('#!/bin/sh\nprintf "test-kmod\\n"\n')
            tool.chmod(0o755)
            with mock.patch.dict(os.environ, {'PATH': str(bin_dir)}), \
                    mock.patch.object(installer, 'SYSTEM_TOOL_PATH', str(sbin_dir)):
                self.assertIsNone(installer.shutil.which('modinfo'))
                self.assertEqual(installer.resolve_tool('modinfo'), str(tool))
                self.assertEqual(installer.command('modinfo', '--version'), 'test-kmod')
                # Its existing-module path uses the same resolver and requires no root writes.
                installer.install_driver('test-kernel')
                with self.assertRaisesRegex(ValueError, '缺少 absent-tool'):
                    installer.resolve_tool('absent-tool')

    def test_bundled_greeting_format_padding_and_peak(self):
        with wave.open(str(ROOT / 'audio/hello_microduck.wav'), 'rb') as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (2, 2, 48000))
            self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), 3.568, places=3)
            samples = array('h', wav.readframes(wav.getnframes()))
        if sys.byteorder != 'little':
            samples.byteswap()
        self.assertLessEqual(max(abs(x) for x in samples), 19661)
        self.assertGreater(max(abs(x) for x in samples), 19650)
        self.assertFalse(any(samples[:48000]))
        self.assertFalse(any(samples[-48000:]))

    def test_peak_normalization_preserves_channels_and_rejects_silence(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.wav', Path(folder) / 'out.wav'
            for values in ((-32768, 1000, 16000, -8000), (0, 0, 0, 0)):
                samples = array('h', values)
                if sys.byteorder != 'little':
                    samples.byteswap()
                with wave.open(str(source), 'wb') as wav:
                    wav.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
                    wav.writeframes(samples.tobytes())
                if any(values):
                    generator.normalize(source, output, padding=0)
                    with wave.open(str(output), 'rb') as wav:
                        actual = array('h', wav.readframes(wav.getnframes()))
                    if sys.byteorder != 'little':
                        actual.byteswap()
                    self.assertEqual(len(actual), 4)
                    self.assertLessEqual(max(abs(x) for x in actual), 19661)
                    self.assertLess(actual[0], 0)
                    self.assertGreater(actual[1], 0)
                else:
                    with self.assertRaises(ValueError):
                        generator.normalize(source, output)

    def test_dry_run_works_outside_module_without_alsa(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/play.py'), '--dry-run'],
                                cwd='/tmp', capture_output=True, text=True, env={'PATH': ''})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('116,116', result.stdout)
        self.assertIn('plughw:CARD=aic3104,DEV=0', result.stdout)

    def test_melody_has_stereo_frames_and_bounded_peak(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'melody.wav'
            player.make_melody(path)
            self.assertGreater(player.wav_info(path), 20)
            with wave.open(str(path), 'rb') as wav:
                samples = array('h', wav.readframes(wav.getnframes()))
            if sys.byteorder != 'little':
                samples.byteswap()
            self.assertLessEqual(max(abs(x) for x in samples), 13107)
            self.assertEqual(samples[::2], samples[1::2])

    def test_boot_plan_preserves_other_devices_and_is_idempotent(self):
        ext = ('default linux\nlabel linux\n linux /boot/vmlinuz\n fdt /boot/dtb/board.dtb\n'
               ' append root=UUID=example console=ttyS2\n fdtoverlays /boot/uart2m0.dtbo '
               '/boot/radxa-zero3-rpi-camera-v2.dtbo /boot/overlay-user/microduck-j5-i2c.dtbo\n')
        env = 'overlays=uart2m0\nuser_overlays=microduck-j5-i2c\nother=value\n\x00\x00'
        base, old, overlays, names, new_ext, new_env = installer.boot_plan(ext, env)
        self.assertEqual(names, ['microduck-media-aic3104'])
        self.assertEqual(overlays[:-1], old)
        self.assertIn(' append root=UUID=example console=ttyS2\n', new_ext)
        self.assertIn('\x00\x00', new_env)
        self.assertIn('other=value\n', new_env)
        again = installer.boot_plan(new_ext, new_env)
        self.assertEqual(again[-2:], (new_ext, new_env))
        no_camera = ext.replace('/boot/radxa-zero3-rpi-camera-v2.dtbo ', '')
        self.assertNotIn('camera', installer.boot_plan(no_camera, env)[-2])

    def test_boot_plan_refuses_conflicting_audio_or_software_bus(self):
        ext = 'label linux\n fdt /boot/board.dtb\n fdtoverlays /boot/other-aic3104.dtbo\n'
        with self.assertRaises(ValueError):
            installer.boot_plan(ext, '')
        with self.assertRaises(ValueError):
            installer.boot_plan(ext.replace('other-aic3104', 'i2c-gpio-pihat'), '')

    def test_boot_plan_handles_empty_directive_and_rejects_duplicate_or_nul(self):
        ext = 'label linux\n fdt /boot/board.dtb\n fdtoverlays\n'
        result = installer.boot_plan(ext, '')[-2]
        self.assertEqual(result.count('fdtoverlays'), 1)
        self.assertIn('microduck-media-i2c3.dtbo', result)
        self.assertIn('microduck-media-aic3104.dtbo', result)
        with self.assertRaises(ValueError):
            installer.boot_plan(ext + ' fdtoverlays /boot/another.dtbo\n', '')
        with self.assertRaises(ValueError):
            installer.boot_plan(ext + '\x00', '')


if __name__ == '__main__':
    unittest.main()
