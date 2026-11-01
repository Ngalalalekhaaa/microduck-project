"""Supported hip-only gain transitions and cleanup, without real hardware."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import hip_position_hold as hold
from microduck_deploy.config import calibration_arrays, default_config, record_reference
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks


def calibrated():
    cfg = default_config()
    for sid, joint in zip(list(range(1, 10))+list(range(11, 16)), cfg['joints']):
        joint.update(id=sid, mapping_verified=True, direction=1 if sid == 15 else -1)
    record_reference(cfg, np.full(14, 2048), 'straight', STRAIGHT_REFERENCE)
    return cfg


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += max(seconds, .001)


class Bus:
    def __init__(self, cfg, clock, *, error=5, auto_enable=True):
        self.clock, self.auto_enable = clock, auto_enable
        self.ids = [j['id'] for j in cfg['joints']]
        self.home = dict(zip(self.ids, map(int, q_to_ticks(HOME, *calibration_arrays(cfg)))))
        self.goals, self.ticks = self.home.copy(), self.home.copy()
        self.gains = {sid: {'p': 5, 'd': 0, 'i': 0} for sid in self.ids}
        self.torque = dict.fromkeys(self.ids, 1)
        self.bounds = {sid: (0, 4095) for sid in self.ids}
        self.currents = dict.fromkeys(self.ids, 100.)
        self.temperatures = dict.fromkeys(self.ids, 35.)
        self.voltages = dict.fromkeys(self.ids, 7.2)
        for sid in hold.HIP_IDS:
            self.ticks[sid] += error
        self.events = []
        self.fail_gain_once = None
        self.fail_gain_always = None
        self.interrupt_off_once = None
        self.stale = False
        self.stuck = False
        self.overcurrent_after_gain = None
        self.bad_goal_once = None
        self.observer = None

    def read_configuration(self, sid):
        return {'mode': 4, 'angle_resolution': 1, 'position_min': self.bounds[sid][0],
                'position_max': self.bounds[sid][1], 'phase': 52,
                'eeprom_gains': {'p': 32, 'd': 40, 'i': 0},
                'ram_gains': self.gains[sid].copy(), 'torque': self.torque[sid],
                'motion_profile': {'goal_position_ticks': self.goals[sid], 'goal_current_raw': 500,
                                   'goal_speed_raw': 0, 'torque_limit_raw': 1000}}

    def read(self, sid, address, size):
        if (address, size) == (40, 1): return bytes((self.torque[sid],))
        if (address, size) == (42, 2): return self.goals[sid].to_bytes(2, 'little')
        if (address, size) == (50, 3): return bytes(self.gains[sid][key] for key in ('p', 'd', 'i'))
        raise AssertionError((sid, address, size))

    def read_feedback(self, sid):
        return SimpleNamespace(position_ticks=self.ticks[sid], velocity_raw=0,
                               current_ma=self.currents[sid], voltage_v=self.voltages[sid],
                               temperature_c=self.temperatures[sid], status=0,
                               received_at=self.clock()-(.3 if self.stale else 0),
                               goal_ticks=self.goals[sid])

    def read_feedback_many(self, ids, *, sync):
        if self.observer: self.observer(self)
        return {sid: self.read_feedback(sid) for sid in ids}

    def set_torque(self, ids, enabled, *, verify=True):
        for sid in ids:
            self.events.append(('torque', sid, int(enabled)))
            self.torque[sid] = int(enabled)
            if not enabled and self.interrupt_off_once == sid:
                self.interrupt_off_once = None
                raise KeyboardInterrupt

    def set_pid_ram(self, ids, p, d, i, *, verify=True):
        for sid in ids:
            if self.torque[sid] != 0:
                raise AssertionError('gain changed while powered')
            self.events.append(('gain', sid, p, d, i))
            self.gains[sid] = {'p': p, 'd': d, 'i': i}
            if self.overcurrent_after_gain == sid:
                self.currents[sid] = 900.
            if self.fail_gain_once == sid:
                self.fail_gain_once = None
                raise OSError('gain acknowledgement lost')
            if self.fail_gain_always == sid:
                raise OSError('gain interface unavailable')

    def sync_positions(self, targets):
        self.events.append(('positions', targets.copy()))
        for sid, target in targets.items():
            self.goals[sid] = target
            if self.auto_enable:
                self.torque[sid] = 1
            if not self.stuck:
                self.ticks[sid] = target
            if self.bad_goal_once == sid:
                self.bad_goal_once = None
                self.goals[sid] += 1


class HoldCoreTests(unittest.TestCase):
    def make(self, *, moving=True, error=5, auto_enable=True, persist=None):
        cfg, clock, report = calibrated(), Clock(), {}
        bus = Bus(cfg, clock, error=error, auto_enable=auto_enable)
        task = hold.HipPositionHold(cfg, bus, report, moving=moving, clock=clock,
                                    sleep=clock.sleep, stop=lambda: None, persist=persist)
        return task, bus, cfg, report

    def test_readonly_can_measure_without_any_servo_writes(self):
        task, bus, _, report = self.make(moving=False)
        task.preflight()
        self.assertTrue(task.observe(seconds=.5))
        self.assertEqual(bus.events, [])
        self.assertEqual(report['status'], 'read_only_reached')
        with self.assertRaises(ValueError): task.run()
        self.assertEqual(bus.events, [])

    def test_readonly_not_reached_reports_false_and_never_attempts_fix(self):
        task, bus, _, report = self.make(moving=False, error=8)
        task.preflight()
        self.assertFalse(task.observe(seconds=.5))
        self.assertEqual(bus.events, [])
        self.assertEqual(report['status'], 'read_only_not_reached')

    def test_readonly_cannot_pass_if_an_unselected_joint_is_off_or_has_another_goal(self):
        for fault in ('off', 'goal'):
            task, bus, _, report = self.make(moving=False, error=0)
            if fault == 'off': bus.torque[9] = 0
            else: bus.goals[9] += 1
            task.preflight()
            self.assertFalse(task.observe(seconds=.5))
            self.assertFalse(report['measurement']['all_body_goals_held'])
            self.assertEqual(bus.events, [])

    def test_sequence_only_four_hips_seed_before_enable_and_small_home_steps(self):
        for auto in (False, True):
            task, bus, cfg, report = self.make(auto_enable=auto)
            before = copy.deepcopy(cfg)
            task.preflight()
            original = {sid: bus.read_configuration(sid) for sid in bus.ids if sid not in hold.HIP_IDS}
            self.assertTrue(task.run(p=16))
            self.assertEqual(cfg, before)
            self.assertEqual(report['status'], 'holding_home')
            self.assertTrue(report['gains_verified'])
            self.assertEqual(task.touched, [2, 3, 12, 13])
            for sid in bus.ids:
                self.assertEqual(bus.goals[sid], bus.home[sid])
                self.assertEqual(bus.torque[sid], 1)
                if sid not in hold.HIP_IDS:
                    self.assertEqual(bus.read_configuration(sid), original[sid])
            for sid in hold.HIP_IDS:
                events = [row for row in bus.events if (row[0] != 'positions' and row[1] == sid)
                          or (row[0] == 'positions' and sid in row[1])]
                self.assertEqual(events[0], ('torque', sid, 0))
                self.assertEqual(events[1], ('gain', sid, 16, 0, 0))
                self.assertEqual(events[2], ('positions', {sid: bus.home[sid]+5}))
                positions = [event[1][sid] for event in events if event[0] == 'positions']
                self.assertTrue(all(abs(a-b) <= 1 for a, b in zip(positions, positions[1:])))
                if not auto: self.assertEqual(events[3], ('torque', sid, 1))
            json.dumps(report)

    def test_restore_walking_p5_does_not_edit_policy_or_calibration(self):
        task, bus, cfg, report = self.make()
        for sid in hold.HIP_IDS: bus.gains[sid]['p'] = 16
        original = copy.deepcopy(cfg)
        task.preflight()
        self.assertTrue(task.run(p=5))
        self.assertEqual(cfg, original)
        self.assertTrue(all(bus.gains[sid] == {'p': 5, 'd': 0, 'i': 0} for sid in bus.ids))

    def test_backup_failure_or_preflight_failure_never_writes(self):
        def fail(): raise OSError('disk full')
        task, bus, _, _ = self.make(persist=fail)
        with self.assertRaises(OSError): task.preflight()
        self.assertEqual(bus.events, [])
        for fault in ('torque', 'goal', 'gain', 'bounds', 'stale', 'large_error'):
            task, bus, _, _ = self.make()
            if fault == 'torque': bus.torque[9] = 0
            elif fault == 'goal': bus.goals[9] += 1
            elif fault == 'gain': bus.gains[3] = {'p': 32, 'd': 40, 'i': 0}
            elif fault == 'bounds': bus.bounds[3] = (0, bus.home[3]+2)
            elif fault == 'stale': bus.stale = True
            else: bus.ticks[3] += 100
            with self.subTest(fault=fault), self.assertRaises(ValueError): task.preflight()
            self.assertEqual(bus.events, [])

    def test_current_voltage_and_temperature_rejected_even_in_readonly(self):
        for moving in (False, True):
            for field, value in (('currents', 800), ('temperatures', 60), ('voltages', 3.9)):
                task, bus, _, _ = self.make(moving=moving)
                getattr(bus, field)[3] = value
                with self.subTest(moving=moving, field=field), self.assertRaises(ValueError): task.preflight()
                self.assertEqual(bus.events, [])

    def test_ambiguous_gain_write_rolls_back_only_transitioned_joints(self):
        task, bus, _, report = self.make()
        task.preflight()
        bus.fail_gain_once = 3
        with self.assertRaises(OSError): task.run(p=16)
        self.assertEqual(set(report['cleanup']), {'2', '3'})
        self.assertTrue(all(bus.gains[sid]['p'] == 5 for sid in bus.ids))
        self.assertTrue(all(bus.torque[sid] == 1 for sid in bus.ids))
        for sid in (2, 3):
            self.assertEqual(bus.goals[sid], bus.ticks[sid])
            self.assertEqual(bus.goals[sid], bus.home[sid]+5)
        self.assertEqual(bus.goals[12], bus.home[12])
        self.assertEqual(bus.goals[13], bus.home[13])

    def test_interrupt_between_torque_off_and_gain_write_is_also_recovered(self):
        task, bus, _, report = self.make()
        task.preflight()
        bus.interrupt_off_once = 2
        with self.assertRaises(KeyboardInterrupt): task.run(p=16)
        self.assertEqual(set(report['cleanup']), {'2'})
        self.assertEqual(bus.gains[2]['p'], 5)
        self.assertEqual(bus.torque[2], 1)
        self.assertEqual(bus.goals[2], bus.ticks[2])

    def test_stop_before_first_transition_keeps_everything_unchanged(self):
        task, bus, _, _ = self.make()
        task.preflight()
        def stop(): raise KeyboardInterrupt
        task.stop = stop
        with self.assertRaises(KeyboardInterrupt): task.run()
        self.assertEqual(bus.events, [])

    def test_unrecoverable_gain_fault_leaves_only_affected_hip_off(self):
        task, bus, _, report = self.make()
        task.preflight()
        bus.fail_gain_always = 3
        with self.assertRaises(OSError): task.run()
        self.assertEqual(report['cleanup']['3']['status'], 'left_torque_off')
        self.assertEqual(bus.torque[3], 0)
        self.assertTrue(all(bus.torque[sid] == 1 for sid in bus.ids if sid != 3))
        self.assertEqual(report['cleanup']['2']['torque'], 1)

    def test_overcurrent_during_transition_does_not_blindly_restore_torque(self):
        task, bus, _, report = self.make()
        task.preflight()
        bus.overcurrent_after_gain = 2
        with self.assertRaises(ValueError): task.run()
        self.assertEqual(report['cleanup']['2']['status'], 'left_torque_off')
        self.assertEqual(bus.torque[2], 0)
        self.assertTrue(all(bus.torque[sid] == 1 for sid in bus.ids if sid != 2))

    def test_target_readback_failure_triggers_current_position_recovery(self):
        task, bus, _, report = self.make()
        task.preflight()
        bus.bad_goal_once = 2
        with self.assertRaises(ValueError): task.run()
        self.assertEqual(bus.gains[2]['p'], 5)
        self.assertEqual(bus.goals[2], bus.ticks[2])
        self.assertEqual(bus.torque[2], 1)

    def test_p16_may_apply_but_unreachable_position_must_not_claim_success(self):
        task, bus, _, report = self.make(error=8)
        task.preflight()
        bus.stuck = True
        self.assertFalse(task.run())
        self.assertEqual(report['status'], 'gains_applied_but_target_not_reached')
        self.assertTrue(report['gains_verified'])
        self.assertEqual(report['cleanup'], {})
        self.assertTrue(all(bus.gains[sid]['p'] == 16 and bus.torque[sid] == 1 for sid in hold.HIP_IDS))

    def test_oscillation_after_settle_restores_prior_gains(self):
        task, bus, _, report = self.make(error=0)
        task.preflight()
        counter = [0]
        def oscillate(b):
            if all(b.gains[sid]['p'] == 16 for sid in hold.HIP_IDS) and task.clock() > 10.5:
                counter[0] += 1
                b.ticks[3] = b.home[3]+(3 if counter[0] % 2 else -3)
        bus.observer = oscillate
        with self.assertRaisesRegex(ValueError, '晃动'): task.run()
        self.assertFalse(report['measurement']['stable'])
        self.assertTrue(all(bus.gains[sid]['p'] == 5 for sid in hold.HIP_IDS))

    def test_config_change_or_deadline_before_motion_has_no_write(self):
        for fault in ('config', 'time'):
            task, bus, cfg, _ = self.make()
            task.preflight()
            if fault == 'config': cfg['control']['ram_p'] = 12
            else: task.clock.now += 121
            with self.subTest(fault=fault), self.assertRaises(ValueError): task.run()
            self.assertEqual(bus.events, [])

    def test_config_file_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.json'
            cfg = calibrated()
            text = json.dumps(cfg)
            path.write_text(text)
            clock, report = Clock(), {}
            bus = Bus(cfg, clock)
            task = hold.HipPositionHold(cfg, bus, report, moving=True, config_path=path,
                                        clock=clock, sleep=clock.sleep, stop=lambda: None)
            task.preflight()
            self.assertTrue(task.run())
            self.assertEqual(path.read_text(), text)


if __name__ == '__main__':
    unittest.main()
