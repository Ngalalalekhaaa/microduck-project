"""ID imports must preserve machine settings and invalidate stale calibration."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from microduck_deploy.__main__ import main
from microduck_deploy.config import (
    ROOT, apply_joint_ids, default_config, load_config, record_reference,
    require_calibration, save_config,
)
from microduck_deploy.policy import HOME, JOINT_NAMES


def calibrated_config():
    cfg = default_config()
    cfg['serial']['port'] = '/dev/ttyUSB7'
    cfg['imu']['i2c_bus'] = 3
    cfg['imu_mount_verified'] = True
    cfg['model'] = 'models/hd1910_backlash.onnx'
    for index, joint in enumerate(cfg['joints']):
        joint.update(direction=1 if index % 2 else -1, mapping_verified=True)
    record_reference(cfg, [2000] * 14, 'home', HOME)
    return cfg


class IdMappingTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / 'servo_ids.user.json').read_text())

    def run_import(self, path, profile_path):
        with patch('microduck_deploy.servo.ServoBus') as bus, \
             patch('microduck_deploy.runtime.ImuReader') as imu, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', str(path), 'map-ids',
                                   '--profile', str(profile_path)]), 0)
        bus.assert_not_called()
        imu.assert_not_called()

    def test_user_ids_preserve_order_settings_and_unmodified_right_leg(self):
        cfg = calibrated_config()
        before = copy.deepcopy(cfg)
        changed = apply_joint_ids(cfg, self.profile)
        self.assertEqual(changed, list(JOINT_NAMES[:9]))
        self.assertEqual([j['name'] for j in cfg['joints']], list(JOINT_NAMES))
        self.assertEqual([j['id'] for j in cfg['joints']], list(range(1, 15)))
        for key in before.keys() - {'joints', 'calibration'}:
            self.assertEqual(cfg[key], before[key])
        for joint in cfg['joints'][:9]:
            self.assertIsNone(joint['direction'])
            self.assertIsNone(joint['zero_tick'])
            self.assertFalse(joint['mapping_verified'])
        self.assertEqual(cfg['joints'][9:], before['joints'][9:])
        self.assertIsNone(cfg['calibration'])
        with self.assertRaisesRegex(ValueError, 'map-joint'):
            require_calibration(cfg)

    def test_import_into_new_config_does_not_infer_any_direction_or_zero(self):
        cfg = default_config()
        apply_joint_ids(cfg, self.profile)
        for joint in cfg['joints']:
            self.assertIsNone(joint['direction'])
            self.assertIsNone(joint['zero_tick'])
            self.assertFalse(joint['mapping_verified'])

    def test_cli_saves_with_backup_and_never_opens_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            save_config(path, calibrated_config())
            before = path.read_bytes()
            self.run_import(path, ROOT / 'servo_ids.user.json')
            self.assertEqual([j['id'] for j in load_config(path)['joints']], list(range(1, 15)))
            backups = list(path.parent.glob('robot.json.bak.*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), before)

    def test_identical_ids_keep_valid_calibration_and_file_bytes(self):
        cfg = calibrated_config()
        for joint in cfg['joints']:
            joint['id'] = self.profile[joint['name']]
        record_reference(cfg, [2000] * 14, 'home', HOME)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            save_config(path, cfg)
            before = path.read_bytes()
            self.run_import(path, ROOT / 'servo_ids.user.json')
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('robot.json.bak.*')), [])
            require_calibration(load_config(path))

    def test_id_swap_is_atomic_and_clears_both_changed_mappings(self):
        cfg = calibrated_config()
        profile = {j['name']: j['id'] for j in cfg['joints']}
        first, second = JOINT_NAMES[:2]
        profile[first], profile[second] = profile[second], profile[first]
        self.assertEqual(apply_joint_ids(cfg, profile), [first, second])
        self.assertEqual([j['id'] for j in cfg['joints'][:2]], [21, 20])
        self.assertTrue(all(not j['mapping_verified'] for j in cfg['joints'][:2]))
        self.assertIsNone(cfg['calibration'])

    def test_invalid_profiles_leave_memory_file_and_backups_unchanged(self):
        invalid = [[], {k: v for k, v in self.profile.items() if k != JOINT_NAMES[0]},
                   {**self.profile, 'mouth': 34}]
        invalid.extend({**self.profile, JOINT_NAMES[0]: value}
                       for value in (2, True, 1.0, '1', -1, 254, None))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            profile_path = Path(directory) / 'invalid.json'
            cfg = calibrated_config()
            save_config(path, cfg)
            before = copy.deepcopy(cfg)
            before_bytes = path.read_bytes()
            for profile in invalid:
                with self.subTest(profile=profile):
                    with self.assertRaises(ValueError):
                        apply_joint_ids(cfg, profile)
                    self.assertEqual(cfg, before)
                    profile_path.write_text(json.dumps(profile))
                    with patch('microduck_deploy.servo.ServoBus') as bus, \
                         self.assertRaises(ValueError):
                        main(['--config', str(path), 'map-ids', '--profile', str(profile_path)])
                    bus.assert_not_called()
                    self.assertEqual(path.read_bytes(), before_bytes)
                    self.assertEqual(list(path.parent.glob('robot.json.bak.*')), [])


if __name__ == '__main__':
    unittest.main()
