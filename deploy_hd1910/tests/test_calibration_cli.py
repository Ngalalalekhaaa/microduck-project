"""Calibration display is hardware-read-only and does not need an IMU."""
import io
import tempfile
import json
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from microduck_deploy.__main__ import main
from microduck_deploy.config import default_config, record_reference, save_config
from microduck_deploy.policy import STRAIGHT_REFERENCE, HOME, ticks_to_q, q_to_ticks
from microduck_deploy.servo import ServoFeedback


class CalibrationDisplayTests(unittest.TestCase):
    def test_unconfirmed_calibration_fails_before_opening_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'robot.json'
            save_config(path,default_config())
            with patch('microduck_deploy.servo.ServoBus') as bus:
                with self.assertRaises(ValueError):
                    main(['--config',str(path),'watch','--angles'])
                bus.assert_not_called()

    def test_angles_use_calibrated_sign_without_imu_or_motor_writes(self):
        cfg=default_config()
        for joint in cfg['joints']:
            joint.update(direction=-1,mapping_verified=True)
        record_reference(cfg,np.full(14,1900),'zero',np.zeros(14))
        self.assertFalse(cfg['imu_mount_verified'])
        class ReadOnlyBus:
            def __enter__(self): return self
            def __exit__(self,*_): pass
            # No goal/torque/gain interfaces: any hardware write would fail.
            def read_feedback_many(self,ids,*,sync):
                assert ids == [32] and not sync
                return {32:SimpleNamespace(position_ticks=1843)}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'robot.json'
            save_config(path,cfg)
            output=io.StringIO()
            with patch('microduck_deploy.servo.ServoBus',return_value=ReadOnlyBus()), \
                 patch('microduck_deploy.__main__.time.monotonic',side_effect=[0,.1,.6]), \
                 patch('microduck_deploy.__main__.time.sleep'), redirect_stdout(output):
                result=main(['--config',str(path),'watch','--angles','--id','32','--seconds','.5'])
            self.assertEqual(result,0)
            self.assertIn('head_yaw ID=32 raw=1843 q=+5.01deg HOME_delta=+5.01deg',output.getvalue())


class StraightCliTests(unittest.TestCase):
    def make_config(self):
        cfg = default_config()
        for i, joint in enumerate(cfg['joints']):
            joint.update(direction=1 if i % 2 else -1, mapping_verified=True)
        return cfg

    def readonly_bus(self, cfg, *, moving=False):
        ids = [j['id'] for j in cfg['joints']]
        class ReadOnlyBus:
            calls = 0
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read_configuration(self, sid):
                return {'torque':0, 'mode':4, 'angle_resolution':1,
                        'position_min':0, 'position_max':4095}
            def read_feedback_many(self, selected, *, sync):
                assert selected == ids and sync
                self.calls += 1
                offset = 20 if moving and self.calls > 1 else 0
                return {sid:ServoFeedback(sid, 1900 + i*7 + offset, 0, 0, 7.4, 30, 0, 0,
                                           1900 + i*7 + offset, 0, time.monotonic())
                        for i, sid in enumerate(ids)}
        return ReadOnlyBus()

    def test_straight_capture_saves_measured_ticks_mapping_and_home_without_writes(self):
        cfg = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.json'
            save_config(path, cfg)
            with patch('microduck_deploy.servo.ServoBus', return_value=self.readonly_bus(cfg)), \
                 patch('microduck_deploy.__main__.time.sleep'), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config',str(path),'calibrate','--pose','straight']),0)
            saved = json.loads(path.read_text())
            ticks = np.arange(14)*7 + 1900
            calibration = saved['calibration']
            self.assertEqual(calibration['pose'], 'straight')
            self.assertEqual(calibration['captured_ticks'], ticks.tolist())
            self.assertEqual(calibration['joint_ids'], [j['id'] for j in cfg['joints']])
            zero = [j['zero_tick'] for j in saved['joints']]
            sign = [j['direction'] for j in saved['joints']]
            np.testing.assert_allclose(ticks_to_q(ticks, zero, sign), STRAIGHT_REFERENCE, atol=1e-14)
            np.testing.assert_array_equal(calibration['home_ticks'], q_to_ticks(HOME, zero, sign))
            self.assertEqual(len(list(Path(directory).glob('robot.json.bak.*'))), 1)

    def test_unstable_straight_capture_does_not_overwrite_existing_file(self):
        cfg = self.make_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.json'
            save_config(path, cfg)
            before = path.read_bytes()
            with patch('microduck_deploy.servo.ServoBus', return_value=self.readonly_bus(cfg,moving=True)), \
                 patch('microduck_deploy.__main__.time.sleep'), self.assertRaisesRegex(ValueError, '8刻度'):
                main(['--config',str(path),'calibrate','--pose','straight'])
            self.assertEqual(path.read_bytes(), before)

    def test_home_plan_is_readonly_without_imu_or_onnx_and_keeps_config(self):
        cfg = self.make_config()
        record_reference(cfg,np.arange(14)*7+1900,'straight',STRAIGHT_REFERENCE)
        self.assertFalse(cfg['imu_mount_verified'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.json'
            report_path = Path(directory)/'plan.jsonl'
            save_config(path,cfg)
            before = path.read_bytes()
            result = io.StringIO()
            with patch('microduck_deploy.servo.ServoBus',return_value=self.readonly_bus(cfg)), \
                 patch('microduck_deploy.runtime.log_path',return_value=report_path), \
                 patch('microduck_deploy.runtime.ImuReader') as imu, \
                 patch('microduck_deploy.runtime.OnnxPolicy') as policy, redirect_stdout(result):
                self.assertEqual(main(['--config',str(path),'home-plan','--from-pose','straight']),0)
            imu.assert_not_called()
            policy.assert_not_called()
            self.assertEqual(path.read_bytes(),before)
            plan = json.loads(report_path.with_suffix('.json').read_text())
            self.assertEqual(plan['duration_s'],8)
            self.assertEqual(len(plan['joints']),14)
            self.assertIn('left_knee',result.getvalue())
            self.assertIn('只读预览通过',result.getvalue())

    def test_run_parser_does_not_accept_reference_start_exception(self):
        with patch('microduck_deploy.servo.ServoBus') as bus, \
             patch('sys.stderr',io.StringIO()), self.assertRaises(SystemExit) as exit:
            main(['run','--from-pose','straight','--enable-motion'])
        self.assertEqual(exit.exception.code,2)
        bus.assert_not_called()


if __name__ == '__main__':
    unittest.main()
