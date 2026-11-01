"""Preserve calibration and other controls while applying the 4 V lower bound."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import apply_voltage4 as apply
from microduck_deploy.config import default_config, load_config


class VoltageUpdateTests(unittest.TestCase):
    def test_update_backup_preserves_all_other_configuration(self):
        cfg = default_config()
        cfg['control']['servo_voltage_range'] = [5.0, 8.4]
        cfg['joints'][10]['zero_tick'] = 2028.0
        cfg['calibration'] = {'home_ticks': [1979, 1971]}
        original = copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            path.write_text(json.dumps(cfg))
            with patch.object(apply, 'require_stopped'):
                apply.apply_config(path)
                apply.apply_config(path)
            backups = list(path.parent.glob('robot.json.bak.*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), original)
            saved = load_config(path)
            self.assertEqual(saved['control']['servo_voltage_range'], [4.0, 8.4])
            saved['control']['servo_voltage_range'] = [5.0, 8.4]
            self.assertEqual(saved, original)

    def test_preview_does_not_write(self):
        cfg = default_config()
        cfg['control']['servo_voltage_range'] = [5.0, 8.4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            path.write_text(json.dumps(cfg))
            before = path.read_bytes()
            apply.main(['--config', str(path)])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob('robot.json.bak.*')), [])

    def test_running_control_refuses_write(self):
        cfg = default_config()
        cfg['control']['servo_voltage_range'] = [5.0, 8.4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'robot.json'
            path.write_text(json.dumps(cfg))
            before = path.read_bytes()
            with patch.object(apply, 'require_stopped', side_effect=ValueError('控制仍在运行')):
                with self.assertRaises(ValueError):
                    apply.apply_config(path)
            self.assertEqual(path.read_bytes(), before)

    def test_unexpected_upper_bound_not_overwritten(self):
        cfg = default_config()
        cfg['control']['servo_voltage_range'] = [5.0, 9.0]
        with self.assertRaises(ValueError):
            apply.updated_config(cfg)


if __name__ == '__main__':
    unittest.main()
