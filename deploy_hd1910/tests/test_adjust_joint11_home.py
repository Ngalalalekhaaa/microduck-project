"""Verify one-joint encoder correction and targeted torque/position writes."""
import copy
from contextlib import contextmanager
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adjust_joint11_home as adjust
from microduck_deploy.config import default_config, record_reference, calibration_arrays, load_config
from microduck_deploy.homing import validate_reference_calibration
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks, ticks_to_q


def calibrated():
    cfg = default_config()
    for sid, joint in enumerate(cfg['joints'], 1):
        joint.update(id=sid, mapping_verified=True, direction=-1)
    record_reference(cfg, np.full(14, 2000), 'straight', STRAIGHT_REFERENCE)
    return cfg


def duck1_calibrated():
    cfg = default_config()
    ids = list(range(1, 10)) + list(range(11, 16))
    for sid, joint in zip(ids, cfg['joints']):
        joint.update(id=sid, mapping_verified=True, direction=1 if sid == 15 else -1)
    record_reference(cfg, 2000 + np.arange(14) * 7, 'straight', STRAIGHT_REFERENCE)
    return cfg


class FakeBus:
    def __init__(self, cfg, *, auto_enable=False):
        self.home = q_to_ticks(HOME, *calibration_arrays(cfg))
        ids = [joint['id'] for joint in cfg['joints']]
        self.torque = {sid: 1 for sid in ids}
        self.ticks = dict(zip(ids, map(int, self.home)))
        self.goals = self.ticks.copy()
        self.writes = []
        self.events = []
        self.auto_enable = auto_enable

    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read(self, sid, address, count):
        self.events.append(('read', sid, address, count))
        if (address, count) == (40, 1):
            return bytes([self.torque[sid]])
        assert (address, count) == (42, 2)
        return self.goals[sid].to_bytes(2, 'little')
    def read_feedback(self, sid):
        self.events.append(('feedback', sid))
        return SimpleNamespace(position_ticks=self.ticks[sid], goal_ticks=self.goals[sid])
    def set_torque(self, ids, enabled, *, verify):
        event = ('torque', list(ids), enabled)
        self.writes.append(event)
        self.events.append(event)
        for sid in ids: self.torque[sid] = int(enabled)
    def sync_positions(self, targets):
        event = ('positions', dict(targets))
        self.writes.append(event)
        self.events.append(event)
        self.goals.update(targets)
        if self.auto_enable:
            # HD1910/HLS-2 can enable torque when its target is written.
            for sid in targets:
                self.torque[sid] = 1


@contextmanager
def saved_duck1_adjustment(*, auto_enable=True):
    """Reproduce the board's saved calibration / released selected joint."""
    original = duck1_calibrated()
    old_home = q_to_ticks(HOME, *calibration_arrays(original))
    target = int(old_home[10]) + 80
    cfg = adjust.shifted_config(original, target, 12)
    bus = FakeBus(cfg, auto_enable=auto_enable)
    bus.torque[12] = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = root / 'robot.json'
        adjust.save_config(config, original)
        old_digest = adjust.digest(config)
        adjust.save_config(config, cfg)
        logs = root / 'logs'
        logs.mkdir()
        session_path = logs / 'joint11_home_adjustment.json'
        session = {
            # Existing failed saves carry the hash from before calibration,
            # with no "saved config" hash. Recovery must accept this format.
            'config_sha256': old_digest,
            'before': adjust.snapshot(FakeBus(original), bus.torque),
            'ids': [12], 'saved': True,
            'new_home_ticks': {'12': target},
            'phase': 'config_saved_selected_torque_off',
        }
        session_path.write_text(json.dumps(session) + '\n')
        with patch.object(adjust, 'ROOT', root), \
                patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                patch.object(adjust.time, 'sleep'):
            yield cfg, bus, config, session_path


class AdjustmentTests(unittest.TestCase):
    def test_both_hips_save_together_and_other_twelve_remain_unchanged(self):
        cfg = calibrated()
        bus = FakeBus(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--ids', '2', '11', '--config', str(config)])
                bus.ticks[2] -= 10
                bus.ticks[11] += 12
                adjust.main(['save', '--ids', '2', '11', '--config', str(config)])
            self.assertEqual(bus.writes, [('torque', [2, 11], False),
                ('positions', {2: bus.ticks[2], 11: bus.ticks[11]}),
                ('torque', [2], True), ('torque', [11], True)])
            after = load_config(config)
            validate_reference_calibration(after, 'straight')
            for old, new in zip(cfg['joints'], after['joints']):
                if old['id'] not in (2, 11):
                    self.assertEqual(old, new)
            self.assertEqual(after['calibration']['home_ticks'][1], bus.ticks[2])
            self.assertEqual(after['calibration']['home_ticks'][10], bus.ticks[11])
            self.assertEqual(len(list(root.glob('robot.json.bak.*'))), 1)

    def test_extend_existing_servo11_adjustment_without_losing_manual_position(self):
        cfg = calibrated()
        bus = FakeBus(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--config', str(config)])
                bus.ticks[11] += 7
                adjusted = bus.ticks[11]
                adjust.main(['release', '--ids', '2', '11', '--config', str(config)])
                self.assertEqual(bus.ticks[11], adjusted)
                self.assertEqual(bus.writes, [('torque', [11], False), ('torque', [2, 11], False)])
                with self.assertRaisesRegex(ValueError, '--ids'):
                    adjust.main(['save', '--config', str(config)])
                adjust.main(['save', '--ids', '2', '11', '--config', str(config)])
            self.assertEqual(load_config(config)['calibration']['home_ticks'][10], adjusted)

    def test_unstable_left_hip_prevents_both_calibrations_from_being_saved(self):
        cfg = calibrated()
        bus = FakeBus(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            before = config.read_bytes()
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--ids', '2', '11', '--config', str(config)])
                original_feedback = bus.read_feedback
                def drifting_feedback(sid):
                    if sid == 2:
                        bus.ticks[2] += 1
                    return original_feedback(sid)
                with patch.object(bus, 'read_feedback', side_effect=drifting_feedback):
                    with self.assertRaisesRegex(ValueError, '2号仍在移动'):
                        adjust.main(['save', '--ids', '2', '11', '--config', str(config)])
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(bus.writes, [('torque', [2, 11], False)])

    def test_shift_preserves_other_joints_and_straight_recovery(self):
        cfg = calibrated()
        before = copy.deepcopy(cfg)
        old = q_to_ticks(HOME, *calibration_arrays(cfg))
        candidate = adjust.shifted_config(cfg, int(old[10]) + 12)
        self.assertEqual(cfg, before)
        self.assertEqual(candidate['joints'][:10], cfg['joints'][:10])
        self.assertEqual(candidate['joints'][11:], cfg['joints'][11:])
        validate_reference_calibration(candidate, 'straight')
        new = q_to_ticks(HOME, *calibration_arrays(candidate))
        expected = old.copy()
        expected[10] += 12
        np.testing.assert_array_equal(new, expected)

    def test_release_save_only_write_servo11_and_backup(self):
        cfg = calibrated()
        bus = FakeBus(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--config', str(config)])
                self.assertEqual(bus.writes, [('torque', [11], False)])
                bus.ticks[11] += 9
                adjust.main(['save', '--config', str(config)])
            self.assertEqual(bus.writes, [('torque', [11], False),
                                        ('positions', {11: bus.ticks[11]}), ('torque', [11], True)])
            self.assertEqual(len(list(root.glob('robot.json.bak.*'))), 1)
            validate_reference_calibration(load_config(config), 'straight')
            self.assertEqual(load_config(config)['calibration']['home_ticks'][10], bus.ticks[11])

    def test_mixed_hold_or_changed_home_goal_refused(self):
        cfg = calibrated()
        bus = FakeBus(cfg)
        bus.torque[11] = 0
        bus.goals[3] += 1
        with self.assertRaises(ValueError):
            adjust.check_hold(cfg, adjust.snapshot(bus, range(1, 15)), released=True)

    def test_reference_encoder_boundary_refused(self):
        cfg = calibrated()
        with self.assertRaises(ValueError):
            adjust.shifted_config(cfg, 4095)


class Duck1AdjustmentTests(unittest.TestCase):
    def test_release_only_servo12_preserves_config_goals_and_other_torque(self):
        cfg = duck1_calibrated()
        bus = FakeBus(cfg)
        before_goals = bus.goals.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            before = config.read_bytes()
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                adjust.main(['release', '--ids', '12', '--config', str(config)])
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(bus.writes, [('torque', [12], False)])
            self.assertEqual(bus.goals, before_goals)
            self.assertEqual(bus.torque[12], 0)
            self.assertTrue(all(state == 1 for sid, state in bus.torque.items() if sid != 12))
            self.assertNotIn(10, bus.torque)
            self.assertIn(15, bus.torque)

    def test_save_only_servo12_preserves_other_calibration_and_straight_reference(self):
        cfg = duck1_calibrated()
        previous_adjustments = [{'id': 2, 'delta_ticks': 3, 'method': 'previous adjustment'}]
        cfg['calibration']['manual_home_adjustments'] = copy.deepcopy(previous_adjustments)
        cfg['control']['policy_overrun_threshold_s'] = .030
        bus = FakeBus(cfg)
        before_goals = bus.goals.copy()
        index = next(i for i, joint in enumerate(cfg['joints']) if joint['id'] == 12)
        self.assertEqual(cfg['joints'][index]['name'], 'right_hip_roll')
        before_reference = ticks_to_q(cfg['calibration']['captured_ticks'], *calibration_arrays(cfg))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--ids', '12', '--config', str(config)])
                bus.ticks[12] += 9
                before_save_event = len(bus.events)
                adjust.main(['save', '--ids', '12', '--samples', '1', '--config', str(config)])
                save_events = bus.events[before_save_event:]
            after = load_config(config)
            self.assertEqual(bus.writes, [('torque', [12], False),
                                         ('positions', {12: bus.ticks[12]}),
                                         ('torque', [12], True)])
            # Initial status + one sample + final calibration check + fresh
            # prewrite check + postwrite + postenable. Only one captures the
            # calibration sample; direct register 42 reads validate the goal.
            self.assertEqual(save_events.count(('feedback', 12)), 6)
            preload_index = next(i for i, event in enumerate(save_events)
                                 if event[0] == 'positions')
            enable_index = next(i for i, event in enumerate(save_events)
                                if event == ('torque', [12], True))
            self.assertIn(('feedback', 12), save_events[preload_index + 1:enable_index])
            self.assertTrue(all(state == 1 for state in bus.torque.values()))
            self.assertEqual({sid: tick for sid, tick in bus.goals.items() if sid != 12},
                             {sid: tick for sid, tick in before_goals.items() if sid != 12})
            self.assertEqual(len(list(root.glob('robot.json.bak.*'))), 1)
        validate_reference_calibration(after, 'straight')
        np.testing.assert_allclose(
            ticks_to_q(after['calibration']['captured_ticks'], *calibration_arrays(after)),
            before_reference, rtol=0, atol=1e-10)
        adjustments = after['calibration']['manual_home_adjustments']
        self.assertEqual(adjustments[:-1], previous_adjustments)
        self.assertEqual(adjustments[-1]['id'], 12)
        self.assertEqual(adjustments[-1]['delta_ticks'], 9)
        self.assertEqual(adjustments[-1]['old_home_tick'], before_goals[12])
        self.assertEqual(adjustments[-1]['new_home_tick'], bus.ticks[12])
        expected = copy.deepcopy(cfg)
        expected['joints'][index]['zero_tick'] += 9
        expected['calibration']['captured_ticks'][index] += 9
        expected['calibration']['home_ticks'][index] += 9
        expected['calibration']['manual_home_adjustments'] = adjustments
        # Compare the whole config: joint directions, other HOME/reference
        # entries, IMU mapping, model and control settings must all survive.
        self.assertEqual(after, expected)

    def test_duck1_servo11_is_yaw_and_rejected_before_bus_access(self):
        cfg = duck1_calibrated()
        self.assertEqual(next(j['name'] for j in cfg['joints'] if j['id'] == 11),
                         'right_hip_yaw')
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'robot.json'
            adjust.save_config(config, cfg)
            before = config.read_bytes()
            for command in ('status', 'release', 'save', 'hold'):
                with self.subTest(command=command), \
                        patch.object(adjust, 'PositionOnlyBus') as open_bus, \
                        patch.object(adjust.subprocess, 'run') as run:
                    with self.assertRaisesRegex(ValueError, 'ID11不是已标定的right_hip_roll'):
                        adjust.main([command, '--ids', '11', '--config', str(config)])
                    open_bus.assert_not_called()
                    run.assert_not_called()
                    self.assertEqual(config.read_bytes(), before)

    def test_single_sample_keeps_final_position_recheck(self):
        cfg = duck1_calibrated()
        bus = FakeBus(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            before = config.read_bytes()
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--ids', '12', '--config', str(config)])
                original = bus.read_feedback
                right_hip_reads = 0

                def moved_after_sample(sid):
                    nonlocal right_hip_reads
                    if sid == 12:
                        right_hip_reads += 1
                        if right_hip_reads == 3:
                            bus.ticks[sid] += 3
                    return original(sid)

                with patch.object(bus, 'read_feedback', side_effect=moved_after_sample):
                    with self.assertRaisesRegex(ValueError, '12号在采样后移动'):
                        adjust.main(['save', '--ids', '12', '--samples', '1', '--config', str(config)])
            self.assertEqual(config.read_bytes(), before)
            self.assertEqual(bus.writes, [('torque', [12], False)])
            self.assertEqual(bus.torque[12], 0)

    def test_goal_readback_failure_keeps_servo12_off_but_preserves_saved_adjustment(self):
        for failure in ('wrong_goal', 'read_error'):
            with self.subTest(failure=failure):
                cfg = duck1_calibrated()
                bus = FakeBus(cfg)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = root / 'robot.json'
                    adjust.save_config(config, cfg)
                    with patch.object(adjust, 'ROOT', root), \
                            patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                            patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                            patch.object(adjust.time, 'sleep'):
                        adjust.main(['release', '--ids', '12', '--config', str(config)])
                        bus.ticks[12] += 9
                        original = bus.read

                        def fail_goal_readback(sid, address, count):
                            result = original(sid, address, count)
                            if ((sid, address, count) == (12, 42, 2)
                                    and any(event[0] == 'positions' for event in bus.writes)):
                                if failure == 'read_error':
                                    raise OSError('goal readback unavailable')
                                return (int.from_bytes(result, 'little') + 1).to_bytes(2, 'little')
                            return result

                        with patch.object(bus, 'read', side_effect=fail_goal_readback):
                            with self.assertRaises((ValueError, OSError)):
                                adjust.main(['save', '--ids', '12', '--samples', '1', '--config', str(config)])
                    after = load_config(config)
                    self.assertEqual(after['calibration']['home_ticks'][10], bus.ticks[12])
                    validate_reference_calibration(after, 'straight')
                    self.assertEqual(bus.torque[12], 0)
                    self.assertFalse(any(event[0] == 'torque' and event[2] for event in bus.writes))
                    for event in bus.writes:
                        self.assertEqual(set(event[1]), {12})

    def test_invalid_sample_counts_rejected_before_bus_access(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'robot.json'
            adjust.save_config(config, duck1_calibrated())
            for count in ('0', '21'):
                with self.subTest(count=count), patch.object(adjust, 'PositionOnlyBus') as open_bus:
                    with self.assertRaises((ValueError, SystemExit)):
                        adjust.main(['save', '--ids', '12', '--samples', count, '--config', str(config)])
                    open_bus.assert_not_called()

    def test_post_save_failures_release_only_servo12_and_keep_saved_calibration(self):
        failures = ('enable_readback', 'keyboard_after_enable',
                    'first_session_write', 'holding_session_write')
        for failure in failures:
            with self.subTest(failure=failure):
                cfg = duck1_calibrated()
                bus = FakeBus(cfg)
                before_goals = bus.goals.copy()
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    config = root / 'robot.json'
                    session_path = root / 'logs' / 'joint11_home_adjustment.json'
                    adjust.save_config(config, cfg)
                    with patch.object(adjust, 'ROOT', root), \
                            patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                            patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                            patch.object(adjust.time, 'sleep'):
                        adjust.main(['release', '--ids', '12', '--config', str(config)])
                        bus.ticks[12] += 9
                        original_torque = bus.set_torque
                        original_write = Path.write_text

                        def interrupted_torque(ids, enabled, *, verify):
                            # Simulate actual torque-on followed by failed
                            # acknowledgement or an operator interruption.
                            original_torque(ids, enabled, verify=verify)
                            if enabled and failure == 'enable_readback':
                                raise OSError('torque enabled but readback lost')
                            if enabled and failure == 'keyboard_after_enable':
                                raise KeyboardInterrupt()

                        def failed_session_write(path, text, *args, **kwargs):
                            if path == session_path:
                                phase = json.loads(text)['phase']
                                if ((failure == 'first_session_write' and phase == 'config_saved')
                                        or (failure == 'holding_session_write' and phase == 'holding_new_home')):
                                    raise OSError('session storage unavailable')
                            return original_write(path, text, *args, **kwargs)

                        expected_error = KeyboardInterrupt if failure == 'keyboard_after_enable' else OSError
                        with patch.object(bus, 'set_torque', side_effect=interrupted_torque), \
                                patch.object(Path, 'write_text', failed_session_write):
                            with self.assertRaises(expected_error):
                                adjust.main(['save', '--ids', '12', '--samples', '1', '--config', str(config)])
                    self.assertEqual(load_config(config)['calibration']['home_ticks'][10], bus.ticks[12])
                    self.assertEqual(bus.torque[12], 0)
                    self.assertTrue(all(state == 1 for sid, state in bus.torque.items() if sid != 12))
                    self.assertEqual({sid: tick for sid, tick in bus.goals.items() if sid != 12},
                                     {sid: tick for sid, tick in before_goals.items() if sid != 12})
                    for event in bus.writes:
                        self.assertEqual(set(event[1]), {12})
                    if failure != 'first_session_write':
                        self.assertIn(('torque', [12], True), bus.writes)
                    else:
                        self.assertFalse(any(event[0] == 'positions' for event in bus.writes))
                    self.assertEqual(bus.writes[-1], ('torque', [12], False))
                    session = json.loads(session_path.read_text())
                    self.assertTrue(session['saved'])
                    self.assertEqual(session['phase'], 'config_saved_selected_torque_off')

    def test_generic_wrapper_requires_explicit_ids_and_accepts_duck1_id12(self):
        wrapper = Path(adjust.__file__).with_name('adjust_joint_home.py')
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'robot.json'
            adjust.save_config(config, duck1_calibrated())
            args = [str(wrapper), 'status', '--config', str(config), '--config-only']
            with patch.object(adjust, 'PositionOnlyBus') as open_bus, \
                    patch.object(sys, 'argv', args), patch.object(sys, 'stderr', io.StringIO()) as error:
                with self.assertRaises(SystemExit) as stopped:
                    runpy.run_path(str(wrapper), run_name='__main__')
                self.assertTrue(stopped.exception.code)
                self.assertIn('--ids', str(stopped.exception.code) + error.getvalue())
                open_bus.assert_not_called()
            with patch.object(adjust, 'PositionOnlyBus') as open_bus, \
                    patch.object(sys, 'argv', args + ['--ids', '12']):
                runpy.run_path(str(wrapper), run_name='__main__')
                open_bus.assert_not_called()

    def test_save_accepts_target_write_auto_enable_without_redundant_torque_write(self):
        cfg = duck1_calibrated()
        bus = FakeBus(cfg, auto_enable=True)
        initial_goals = bus.goals.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'robot.json'
            adjust.save_config(config, cfg)
            with patch.object(adjust, 'ROOT', root), \
                    patch.object(adjust, 'PositionOnlyBus', return_value=bus), \
                    patch.object(adjust.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(adjust.time, 'sleep'):
                adjust.main(['release', '--ids', '12', '--config', str(config)])
                bus.ticks[12] += 80
                adjust.main(['save', '--ids', '12', '--samples', '1', '--config', str(config)])
            after = load_config(config)
            self.assertEqual(after['calibration']['home_ticks'][10], bus.ticks[12])
            self.assertEqual(bus.writes, [('torque', [12], False),
                                         ('positions', {12: bus.ticks[12]})])
            self.assertTrue(all(state == 1 for state in bus.torque.values()))
            self.assertEqual({sid: goal for sid, goal in bus.goals.items() if sid != 12},
                             {sid: goal for sid, goal in initial_goals.items() if sid != 12})
            session = json.loads((root / 'logs' / 'joint11_home_adjustment.json').read_text())
            self.assertTrue(session['saved'])
            self.assertEqual(session['phase'], 'holding_new_home')
            validate_reference_calibration(after, 'straight')

    def test_hold_saved_home_only_writes_selected_servo_and_preserves_config_bytes(self):
        for auto_enable in (True, False):
            with self.subTest(auto_enable=auto_enable), \
                    saved_duck1_adjustment(auto_enable=auto_enable) as (_, bus, config, session_path):
                before = config.read_bytes()
                old_backups = set(config.parent.glob('robot.json.bak.*'))
                before_goals = bus.goals.copy()
                before_torque = bus.torque.copy()
                adjust.main(['hold', '--ids', '12', '--config', str(config)])
                self.assertEqual(config.read_bytes(), before)
                self.assertEqual(set(config.parent.glob('robot.json.bak.*')), old_backups)
                self.assertEqual(bus.goals, before_goals)
                expected_torque = before_torque.copy()
                expected_torque[12] = 1
                self.assertEqual(bus.torque, expected_torque)
                expected_writes = [('positions', {12: bus.ticks[12]})]
                if not auto_enable:
                    expected_writes.append(('torque', [12], True))
                self.assertEqual(bus.writes, expected_writes)
                self.assertEqual(json.loads(session_path.read_text())['phase'], 'holding_new_home')
                if not auto_enable:
                    preload = bus.events.index(expected_writes[0])
                    enable = bus.events.index(expected_writes[1])
                    self.assertIn(('feedback', 12), bus.events[preload + 1:enable])

    def test_hold_rejects_missing_or_incompatible_saved_session_without_writes(self):
        cases = ('missing', 'unsaved', 'wrong_ids', 'wrong_target', 'missing_target', 'saved_hash_changed',
                 'position_far', 'already_enabled', 'invalid_torque', 'other_goal_changed')
        for case in cases:
            with self.subTest(case=case), saved_duck1_adjustment() as (_, bus, config, session_path):
                before = config.read_bytes()
                session = json.loads(session_path.read_text())
                if case == 'missing':
                    session_path.unlink()
                else:
                    if case == 'unsaved': session['saved'] = False
                    elif case == 'wrong_ids': session['ids'] = [2]
                    elif case == 'wrong_target': session['new_home_ticks']['12'] += 1
                    elif case == 'missing_target': session.pop('new_home_ticks')
                    elif case == 'saved_hash_changed': session['saved_config_sha256'] = 'changed'
                    elif case == 'position_far': bus.ticks[12] += 3
                    elif case == 'already_enabled': bus.torque[12] = 1
                    elif case == 'invalid_torque': bus.torque[12] = 2
                    elif case == 'other_goal_changed': bus.goals[2] += 1
                    session_path.write_text(json.dumps(session) + '\n')
                with self.assertRaises((ValueError, OSError)):
                    adjust.main(['hold', '--ids', '12', '--config', str(config)])
                self.assertEqual(bus.writes, [])
                self.assertEqual(config.read_bytes(), before)

    def test_autoenabled_goal_failures_release_only_selected_joint_for_save_and_hold(self):
        failures = ('wrong_goal', 'read_error', 'invalid_torque', 'position_moved')
        for command in ('save', 'hold'):
            for failure in failures:
                with self.subTest(command=command, failure=failure), \
                        saved_duck1_adjustment() as (_, bus, config, session_path):
                    if command == 'save':
                        session = json.loads(session_path.read_text())
                        session.update(saved=False, config_sha256=adjust.digest(config),
                                       before=adjust.snapshot(bus, bus.torque))
                        session_path.write_text(json.dumps(session) + '\n')
                    before = config.read_bytes()
                    before_goals = bus.goals.copy()
                    original_feedback = bus.read_feedback
                    original_read = bus.read

                    def failed_goal_read(sid, address, count):
                        result = original_read(sid, address, count)
                        if ((sid, address, count) == (12, 42, 2)
                                and any(event[0] == 'positions' for event in bus.writes)):
                            self.assertEqual(bus.torque[12], 1)
                            if failure == 'read_error':
                                raise OSError('auto-enabled goal readback unavailable')
                            if failure == 'wrong_goal':
                                return (int.from_bytes(result, 'little') + 1).to_bytes(2, 'little')
                        return result

                    def failed_readback(sid):
                        feedback = original_feedback(sid)
                        if sid == 12 and any(event[0] == 'positions' for event in bus.writes):
                            self.assertEqual(bus.torque[12], 1)
                            if failure == 'invalid_torque': bus.torque[12] = 2
                            elif failure == 'position_moved': feedback.position_ticks += 3
                        return feedback

                    with patch.object(bus, 'read_feedback', side_effect=failed_readback), \
                            patch.object(bus, 'read', side_effect=failed_goal_read):
                        with self.assertRaises((ValueError, OSError)):
                            adjust.main([command, '--ids', '12', '--samples', '1',
                                         '--config', str(config)])
                    if command == 'hold':
                        self.assertEqual(config.read_bytes(), before)
                    else:
                        self.assertTrue(json.loads(session_path.read_text())['saved'])
                        self.assertEqual(load_config(config)['calibration']['home_ticks'][10],
                                         bus.ticks[12])
                    self.assertEqual(bus.writes[-1], ('torque', [12], False))
                    self.assertEqual(bus.torque[12], 0)
                    self.assertTrue(all(state == 1 for sid, state in bus.torque.items() if sid != 12))
                    self.assertEqual({sid: tick for sid, tick in bus.goals.items() if sid != 12},
                                     {sid: tick for sid, tick in before_goals.items() if sid != 12})
                    self.assertFalse(any(event[0] == 'torque' and event[2] for event in bus.writes))
                    self.assertTrue(all(set(event[1]) == {12} for event in bus.writes))


if __name__ == '__main__':
    unittest.main()
