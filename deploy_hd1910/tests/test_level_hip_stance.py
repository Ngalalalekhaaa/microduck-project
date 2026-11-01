"""Hardware-isolated contracts for the four-hip stance adjustment command."""
from contextlib import nullcontext
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import level_hip_stance as level
from microduck_deploy.config import calibration_arrays, default_config, load_config, record_reference
from microduck_deploy.homing import validate_reference_calibration
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks, ticks_to_q
from microduck_deploy.servo import build_packet, ServoBus


def calibrated():
    cfg = default_config()
    for sid, joint in zip(list(range(1, 10)) + list(range(11, 16)), cfg['joints']):
        joint.update(id=sid, mapping_verified=True, direction=1 if sid == 15 else -1)
    record_reference(cfg, np.full(14, 2048), 'straight', STRAIGHT_REFERENCE)
    cfg['imu_mount_verified'] = True
    # Preserve the user's already-completed ID12 HOME=2065 calibration.
    previous = q_to_ticks(HOME, *calibration_arrays(cfg))
    delta = 2065-int(previous[10])
    cfg['joints'][10]['zero_tick'] += delta
    cfg['calibration']['captured_ticks'][10] += delta
    cfg['calibration']['home_ticks'] = q_to_ticks(HOME, *calibration_arrays(cfg)).tolist()
    cfg['calibration']['manual_home_adjustments'] = [{'id': 12, 'new_home_tick': 2065}]
    return cfg


def state(timestamp=1., pitch=0., roll=0.):
    p, r = math.radians(pitch), math.radians(roll)
    gravity = np.array([math.sin(p), -math.sin(r)*math.cos(p), -math.cos(r)*math.cos(p)])
    return SimpleNamespace(timestamp=timestamp, accel_m_s2=-gravity*level.GRAVITY,
                           base_ang_vel=np.zeros(3))


def measurement(pitch=0., roll=0.):
    states = [state(1+i*.1, pitch, roll) for i in range(12)]
    return level.stationary_measurement(states)


class FakeImu:
    def __init__(self):
        self.value = state()
        self.closed = False
        self.failure = None
        self.timestamp_offset = 0.
    def latest(self, age):
        if self.failure:
            raise self.failure
        self.value.timestamp = level.time.monotonic() + self.timestamp_offset
        return self.value
    def close(self):
        self.closed = True


class FakeBus:
    def __init__(self, cfg):
        self.ids = [j['id'] for j in cfg['joints']]
        self.goals = dict(zip(self.ids, map(int, q_to_ticks(HOME, *calibration_arrays(cfg)))))
        self.ticks = self.goals.copy()
        self.torque = dict.fromkeys(self.ids, 1)
        self.reads = []
        self.writes = []
        self.bad_goal_after_write = False
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def read_configuration(self, sid):
        return dict(mode=4, angle_resolution=1, position_min=0, position_max=4095,
                    ram_gains={'p': 5, 'd': 0, 'i': 0})
    def read_feedback_many(self, ids, *, sync):
        return {sid: SimpleNamespace(position_ticks=self.ticks[sid], status=0,
                                      voltage_v=7.2, temperature_c=30)
                for sid in ids}
    def read(self, sid, address, size):
        self.reads.append((sid, address, size))
        if address == 40:
            assert size == 1
            return bytes([self.torque[sid]])
        assert (address, size) == (42, 2)
        goal = self.goals[sid]
        if self.bad_goal_after_write and self.writes and sid == 3:
            goal += 1
        return int(goal).to_bytes(2, 'little')
    def sync_positions(self, targets):
        self.writes.append(('positions', targets.copy()))
        self.goals.update(targets)
        self.ticks.update(targets)
    def set_torque(self, *args, **kwargs):
        raise AssertionError('No torque write is permitted')


def packet_goals(ids=(2, 3, 12, 13), *, address=42, size=2, tick=2048):
    params = bytearray((address, size))
    for sid in ids:
        params.extend([sid])
        params.extend(tick.to_bytes(size, 'little'))
    return build_packet(254, 131, params)


class PacketBoundaryTests(unittest.TestCase):
    def make_bus(self, motion):
        bus = object.__new__(level.HipOnlyBus)
        bus.allow_motion = motion
        return bus

    def test_default_only_reads_and_motion_exactly_four_position_goals(self):
        with patch.object(ServoBus, '_send', return_value=123.) as send:
            readonly = self.make_bus(False)
            self.assertEqual(readonly._send(build_packet(3, 2, bytes((42, 2)))), 123.)
            self.assertEqual(readonly._send(build_packet(254, 130, bytes((56, 15, 2, 3, 12, 13)))), 123.)
            with self.assertRaises(ValueError):
                readonly._send(packet_goals())
            moving = self.make_bus(True)
            moving._send(packet_goals())
            self.assertEqual(send.call_count, 3)

    def test_other_ids_or_any_torque_gain_eeprom_write_rejected(self):
        bus = self.make_bus(True)
        attempts = [packet_goals((2, 3, 12, 14)), packet_goals((3, 13)),
                    packet_goals((2, 3, 12, 12)), packet_goals(tick=4096),
                    packet_goals(address=40, size=1, tick=1),
                    packet_goals(address=50, size=3, tick=5),
                    packet_goals(address=31, size=2),
                    build_packet(3, 3, bytes((42, 0, 8))),
                    build_packet(3, 3, bytes((40, 1)))]
        with patch.object(ServoBus, '_send') as send:
            for packet in attempts:
                with self.subTest(packet=packet), self.assertRaises(ValueError):
                    bus._send(packet)
            send.assert_not_called()


class StationaryMeasurementTests(unittest.TestCase):
    def test_fresh_static_accel_recovers_pitch_roll_and_gravity(self):
        result = measurement(pitch=3., roll=-2.)
        self.assertAlmostEqual(result['pitch_deg'], 3., places=10)
        self.assertAlmostEqual(result['roll_deg'], -2., places=10)
        self.assertEqual(result['samples'], 12)
        self.assertGreater(result['window_s'], .6)
        self.assertLess(result['pitch_std_deg'], 1e-10)

    def test_duplicate_nonfinite_drift_rotation_non1g_and_short_window_rejected(self):
        cases = []
        duplicates = [state(1.) for _ in range(12)]
        cases.append(duplicates)
        nonfinite = [state(1+i*.1) for i in range(12)]
        nonfinite[2].accel_m_s2[0] = np.nan
        cases.append(nonfinite)
        drift = [state(1+i*.1, pitch=0 if i < 6 else .35) for i in range(12)]
        cases.append(drift)
        rotating = [state(1+i*.1) for i in range(12)]
        rotating[4].base_ang_vel[0] = .16
        cases.append(rotating)
        acceleration = [state(1+i*.1) for i in range(12)]
        acceleration[4].accel_m_s2 *= 1.2
        cases.append(acceleration)
        cases.append([state(1+i*.01) for i in range(12)])
        cases.append([state(1+i*.1) for i in range(7)])
        for i, states in enumerate(cases):
            with self.subTest(case=i), self.assertRaises(ValueError):
                level.stationary_measurement(states)


class CandidateTests(unittest.TestCase):
    def test_candidate_preserves_prior_id12_baseline_and_all_unselected_config(self):
        cfg = calibrated()
        before = copy.deepcopy(cfg)
        original_home = q_to_ticks(HOME, *calibration_arrays(cfg))
        changed = level.home_candidate(cfg, (5, -3), '/logs/example.json')
        self.assertEqual(cfg, before)
        self.assertEqual(original_home[10], 2065)
        self.assertEqual(changed['calibration']['home_ticks'][10], 2068)
        for index, joint in enumerate(cfg['joints']):
            if joint['id'] not in level.HIP_IDS:
                self.assertEqual(joint, changed['joints'][index])
                self.assertEqual(cfg['calibration']['home_ticks'][index],
                                 changed['calibration']['home_ticks'][index])
                self.assertEqual(cfg['calibration']['captured_ticks'][index],
                                 changed['calibration']['captured_ticks'][index])
            else:
                self.assertEqual({k:v for k,v in joint.items() if k != 'zero_tick'},
                                 {k:v for k,v in changed['joints'][index].items() if k != 'zero_tick'})
        self.assertEqual({k:v for k,v in cfg.items() if k not in ('joints', 'calibration')},
                         {k:v for k,v in changed.items() if k not in ('joints', 'calibration')})
        self.assertEqual({k:v for k,v in cfg['calibration'].items()
                          if k not in ('captured_ticks', 'home_ticks', 'manual_home_adjustments')},
                         {k:v for k,v in changed['calibration'].items()
                          if k not in ('captured_ticks', 'home_ticks', 'manual_home_adjustments')})
        self.assertEqual(changed['calibration']['manual_home_adjustments'][0],
                         cfg['calibration']['manual_home_adjustments'][0])
        self.assertEqual(len(changed['calibration']['manual_home_adjustments']), 5)
        np.testing.assert_allclose(
            ticks_to_q(cfg['calibration']['captured_ticks'], *calibration_arrays(cfg)),
            ticks_to_q(changed['calibration']['captured_ticks'], *calibration_arrays(changed)),
            atol=1e-10, rtol=0)
        validate_reference_calibration(changed, 'straight')


class PlantTests(unittest.TestCase):
    def setUp(self):
        self.cfg = calibrated()
        self.bus = FakeBus(self.cfg)
        self.imu = FakeImu()
        self.report = {}
        self.patch_stop = patch.object(level, 'check_stop')
        self.patch_stop.start()
        self.addCleanup(self.patch_stop.stop)
        self.patch_time = patch.object(level.time, 'monotonic', return_value=100.)
        self.patch_time.start()
        self.addCleanup(self.patch_time.stop)
        self.patch_sleep = patch.object(level.time, 'sleep')
        self.sleep = self.patch_sleep.start()
        self.addCleanup(self.patch_sleep.stop)

    def plant(self, moving=True, config_path=None):
        plant = level.StancePlant(self.cfg, self.bus, self.imu, self.report,
                                  moving=moving, config_path=config_path)
        plant.preflight()
        return plant

    def test_move_only_four_ids_one_tick_per_period_and_direct_goal_readback(self):
        plant = self.plant()
        originals = self.bus.goals.copy()
        plant.move((3, -2))
        self.assertEqual(len(self.bus.writes), 3)
        expected_offsets = [(1, -1), (2, -2), (3, -2)]
        for (_, targets), offset in zip(self.bus.writes, expected_offsets):
            self.assertEqual(set(targets), level.HIP_IDS)
            self.assertEqual(targets, level.hip_targets(self.cfg, offset))
        for sid in set(self.bus.ids)-level.HIP_IDS:
            self.assertEqual(self.bus.goals[sid], originals[sid])
            self.assertEqual(self.bus.torque[sid], 1)
        self.assertEqual(self.report['last_sent_offset_ticks'], [3, -2])
        self.assertIsNone(self.report['pending_targets'])
        self.assertTrue(all(call.args == (.1,) for call in self.sleep.call_args_list))
        for sid in level.HIP_IDS:
            self.assertIn((sid, 42, 2), self.bus.reads)

    def test_preflight_records_selected_hardware_gains_without_changing_them(self):
        self.plant()
        self.assertEqual(set(self.report['hardware_config']), {str(sid) for sid in level.HIP_IDS})
        for sid in level.HIP_IDS:
            self.assertEqual(self.report['hardware_config'][str(sid)],
                             self.bus.read_configuration(sid))
            self.assertEqual(self.report['hardware_config'][str(sid)]['ram_gains'],
                             {'p': 5, 'd': 0, 'i': 0})
        self.assertEqual(self.bus.writes, [])

    def test_readonly_move_rejected_without_writes(self):
        plant = self.plant(moving=False)
        with self.assertRaises(ValueError):
            plant.move((1, 0))
        self.assertEqual(self.bus.writes, [])

    def test_other_joint_off_or_wrong_goal_blocks_first_write(self):
        for changed in ('torque', 'goal'):
            with self.subTest(changed=changed):
                plant = self.plant()
                if changed == 'torque': self.bus.torque[7] = 0
                else: self.bus.goals[7] += 1
                with self.assertRaises(ValueError): plant.move((2, 0))
                self.assertEqual(self.bus.writes, [])
                self.bus.torque[7] = 1
                self.bus.goals[7] = plant.home[7]

    def test_goal_readback_mismatch_freezes_without_torque_write(self):
        plant = self.plant()
        self.bus.bad_goal_after_write = True
        with self.assertRaisesRegex(ValueError, '目标回读不一致'):
            plant.move((4, 0))
        self.assertEqual(len(self.bus.writes), 1)
        self.assertEqual(self.report['last_sent_offset_ticks'], [1, 0])
        self.assertTrue(all(self.bus.torque.values()))

    def test_imu_failure_blocks_write(self):
        plant = self.plant()
        self.imu.failure = ValueError('stale IMU')
        with self.assertRaisesRegex(ValueError, 'stale IMU'):
            plant.move((1, 0))
        self.assertEqual(self.bus.writes, [])

    def test_stale_future_or_nonfinite_imu_timestamps_block_goals(self):
        plant = self.plant()
        for offset in (-1., .01, float('nan')):
            with self.subTest(offset=offset):
                self.imu.timestamp_offset = offset
                with self.assertRaisesRegex(ValueError, '时间戳'):
                    plant.move((1, 0))
                self.assertEqual(self.bus.writes, [])

    def test_slow_servo_reads_block_goals_despite_fresh_imu(self):
        plant = self.plant()
        original_read = self.bus.read_feedback_many
        clock = [100.]

        def slow_read(*args, **kwargs):
            feedback = original_read(*args, **kwargs)
            clock[0] += .21
            return feedback

        with patch.object(level.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(self.bus, 'read_feedback_many', side_effect=slow_read):
            with self.assertRaisesRegex(ValueError, '读取超过200ms'):
                plant.move((1, 0))
        self.assertEqual(self.bus.writes, [])

    def test_config_change_blocks_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'robot.json'
            path.write_text(json.dumps(self.cfg))
            plant = self.plant(config_path=path)
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError, 'robot.json发生变化'):
                plant.move((1, 0))
            self.assertEqual(self.bus.writes, [])

    def test_sample_uses_actual_encoder_window_and_preserves_raw_joint_evidence(self):
        # Direction metadata must come from calibration, never be hard-coded.
        self.cfg['joints'][2]['direction'] = 1
        self.bus = FakeBus(self.cfg)
        clock = [100.]
        captured = []
        original_feedback = self.bus.read_feedback_many

        def advance(seconds):
            clock[0] += seconds

        def feedback(ids, *, sync):
            # Deliberately differ from commanded HOME, including the unselected
            # joints, so a goal-based estimate cannot pass this test.
            change = int(round((clock[0]-100.)*10)) % 7
            for sid in ids:
                self.bus.ticks[sid] = self.bus.goals[sid] + sid + change
            captured.append((clock[0], self.bus.ticks.copy()))
            return original_feedback(ids, sync=sync)

        def latest(age):
            return state(timestamp=clock[0], pitch=1.2, roll=-.8)

        with patch.object(level.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(level.time, 'sleep', side_effect=advance), \
             patch.object(self.bus, 'read_feedback_many', side_effect=feedback), \
             patch.object(self.imu, 'latest', side_effect=latest), \
             patch.object(level, 'stationary_measurement', wraps=level.stationary_measurement) as measure, \
             patch('sys.stdout', new_callable=io.StringIO) as printed:
            plant = self.plant()
            observed = plant.sample()

        accepted = measure.call_args.args[0]
        accepted_stamps = [s.timestamp for s in accepted]
        self.assertEqual(len(accepted_stamps), len(set(accepted_stamps)))
        self.assertTrue(all(stamp-100. >= 1.2 for stamp in accepted_stamps))
        rows = {stamp: ticks for stamp, ticks in captured}
        self.assertEqual(set(observed['joint_ticks']), {str(sid) for sid in level.HIP_IDS})
        self.assertEqual(observed['joint_directions'], {'2': -1, '3': 1, '12': -1, '13': -1})
        self.assertEqual(set(observed['joint_feedback']), {str(sid) for sid in self.bus.ids})
        for sid in self.bus.ids:
            ticks = [rows[stamp][sid] for stamp in accepted_stamps]
            detail = observed['joint_feedback'][str(sid)]
            self.assertAlmostEqual(detail['mean_tick'], np.mean(ticks))
            self.assertEqual(detail['min_tick'], min(ticks))
            self.assertEqual(detail['max_tick'], max(ticks))
            self.assertEqual(detail['goal'], self.bus.goals[sid])
            self.assertEqual(detail['torque'], 1)
            if sid in level.HIP_IDS:
                self.assertAlmostEqual(observed['joint_ticks'][str(sid)], np.mean(ticks))
                self.assertNotEqual(observed['joint_ticks'][str(sid)], self.bus.goals[sid])
                self.assertIn(str(sid), printed.getvalue())
        self.assertEqual(self.report['measurement_windows'][-1]['joint_feedback'],
                         observed['joint_feedback'])
        self.assertEqual(self.bus.writes, [])

    def test_adaptive_probe_timeout_retains_hold_without_writes(self):
        plant = self.plant()
        self.assertEqual(level.MAX_RUN_SECONDS, 420)
        with patch.object(level.time, 'monotonic', return_value=plant.started+181):
            plant.read()
        with patch.object(level.time, 'monotonic', return_value=plant.started+421):
            with self.assertRaisesRegex(ValueError, '420'):
                plant.read()
        self.assertEqual(self.bus.writes, [])
        self.assertTrue(all(self.bus.torque.values()))

    def test_duplicate_imu_sample_does_not_add_encoder_sample(self):
        clock = [100.]
        prior_stamp = [None]
        duplicate_at = [False]
        reads = []
        first_encoder_at_stamp = {}
        original_feedback = self.bus.read_feedback_many
        self.cfg['control']['imu_max_age_s'] = .15

        def advance(seconds):
            clock[0] += seconds

        def latest(age):
            if not duplicate_at[0] and clock[0] >= 101.6:
                duplicate_at[0] = True
                stamp = prior_stamp[0]
            else:
                stamp = clock[0]
                prior_stamp[0] = stamp
            first_encoder_at_stamp.setdefault(stamp, self.bus.ticks.copy())
            return state(timestamp=stamp)

        def feedback(ids, *, sync):
            for sid in ids:
                self.bus.ticks[sid] = self.bus.goals[sid] + len(reads)
            reads.append(self.bus.ticks.copy())
            return original_feedback(ids, sync=sync)

        with patch.object(level.time, 'monotonic', side_effect=lambda: clock[0]), \
             patch.object(level.time, 'sleep', side_effect=advance), \
             patch.object(self.bus, 'read_feedback_many', side_effect=feedback), \
             patch.object(self.imu, 'latest', side_effect=latest), \
             patch.object(level, 'stationary_measurement', wraps=level.stationary_measurement) as measure, \
             patch('sys.stdout', new_callable=io.StringIO):
            plant = self.plant()
            observed = plant.sample()
        accepted = measure.call_args.args[0]
        self.assertTrue(duplicate_at[0])
        self.assertEqual(observed['samples'], len(accepted))
        self.assertEqual(len({s.timestamp for s in accepted}), len(accepted))
        for sid in level.HIP_IDS:
            expected = np.mean([first_encoder_at_stamp[s.timestamp][sid] for s in accepted])
            self.assertAlmostEqual(observed['joint_ticks'][str(sid)], expected)
        self.assertEqual(self.bus.writes, [])


class SaveTests(unittest.TestCase):
    def test_save_only_converged_fresh_attitude_hash_and_tracking(self):
        for failure in (None, 'not_converged', 'pending', 'offset', 'attitude', 'hash', 'tracking'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                cfg = calibrated()
                path = Path(directory)/'robot.json'
                path.write_text(json.dumps(cfg))
                original = path.read_bytes()
                bus = FakeBus(cfg)
                report = {'status': 'converged', 'offset_ticks': [3, -2],
                          'pending_offset_ticks': None, 'calibration_saved': False}
                plant = level.StancePlant(cfg, bus, FakeImu(), {}, moving=True, config_path=path)
                plant.offset = np.array([3, -2])
                plant.targets.update(level.hip_targets(cfg, (3, -2)))
                rows = {str(sid): {'tick': plant.targets[sid]} for sid in plant.ids}
                fresh = measurement()
                if failure == 'not_converged': report['status'] = 'not_converged'
                if failure == 'pending': report['pending_offset_ticks'] = [4, -2]
                if failure == 'offset': report['offset_ticks'] = [4, -2]
                if failure == 'attitude': fresh = measurement(pitch=.6)
                if failure == 'hash': plant.config_hash = 'different'
                if failure == 'tracking': rows['3']['tick'] += 24
                with patch.object(plant, 'sample', return_value=fresh), \
                     patch.object(plant, 'read', return_value=(rows, state())):
                    if failure:
                        with self.assertRaises(ValueError):
                            level.save_success(plant, report, Path(directory)/'report.json', .5)
                        self.assertEqual(path.read_bytes(), original)
                        self.assertFalse(report['calibration_saved'])
                    else:
                        level.save_success(plant, report, Path(directory)/'report.json', .5)
                        self.assertTrue(report['calibration_saved'])
                        self.assertEqual(len(list(Path(directory).glob('robot.json.bak.*'))), 1)
                        saved = load_config(path)
                        self.assertEqual(saved['calibration']['home_ticks'][10], 2067)
                        self.assertEqual(report['saved_config_sha256'], level.digest(path))

    def test_interrupt_after_atomic_save_reports_actual_disk_state(self):
        for outcome in ('saved', 'foreign', 'unchanged'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                cfg = calibrated()
                path = Path(directory)/'robot.json'
                path.write_text(json.dumps(cfg))
                original = path.read_bytes()
                report = {'status': 'converged', 'offset_ticks': [3, -2],
                          'pending_offset_ticks': None, 'calibration_saved': False}
                plant = level.StancePlant(cfg, FakeBus(cfg), FakeImu(), {},
                                          moving=True, config_path=path)
                plant.offset = np.array([3, -2])
                plant.targets.update(level.hip_targets(cfg, (3, -2)))
                rows = {str(sid): {'tick': plant.targets[sid]} for sid in plant.ids}
                original_save = level.save_config

                def interrupted_save(save_path, candidate):
                    if outcome != 'unchanged':
                        original_save(save_path, candidate)
                    if outcome == 'foreign':
                        save_path.write_text('{"foreign": true}\n')
                    raise KeyboardInterrupt

                with patch.object(plant, 'sample', return_value=measurement()), \
                     patch.object(plant, 'read', return_value=(rows, state())), \
                     patch.object(level, 'save_config', side_effect=interrupted_save):
                    with self.assertRaises(KeyboardInterrupt):
                        level.save_success(plant, report, Path(directory)/'report.json', .5)
                if outcome == 'saved':
                    self.assertTrue(report['calibration_saved'])
                    self.assertEqual(report['saved_config_sha256'], level.digest(path))
                    saved = load_config(path)
                    self.assertEqual(report['saved_home_ticks'], saved['calibration']['home_ticks'])
                    self.assertEqual(saved['calibration']['home_ticks'][10], 2067)
                    self.assertEqual(len(list(Path(directory).glob('robot.json.bak.*'))), 1)
                elif outcome == 'foreign':
                    self.assertFalse(report['calibration_saved'])
                    self.assertTrue(report['calibration_save_state'].startswith('unknown'))
                    self.assertEqual(json.loads(path.read_text()), {'foreign': True})
                else:
                    self.assertFalse(report['calibration_saved'])
                    self.assertEqual(path.read_bytes(), original)
                    self.assertNotIn('calibration_save_state', report)


class MainTests(unittest.TestCase):
    def test_default_measurement_is_readonly_and_failure_never_saves_or_unloads(self):
        for moving in (False, True):
            with self.subTest(moving=moving), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root/'robot.json'
                cfg = calibrated()
                path.write_text(json.dumps(cfg))
                original = path.read_bytes()
                bus = FakeBus(cfg)
                imu = FakeImu()
                args = ['--config', str(path)]
                if moving:
                    args += ['--move', '--ground-supported', '--save-home']
                with patch.object(level, 'ROOT', root), \
                     patch.object(level, 'stop_signals', return_value=nullcontext()), \
                     patch.object(level, 'exclusive_serial', return_value=nullcontext()), \
                     patch.object(level, 'HipOnlyBus', return_value=bus), \
                     patch.object(level, 'ProcessImuReader', return_value=imu), \
                     patch.object(level.StancePlant, 'sample', return_value=measurement()), \
                     patch.object(level, 'level_hips', side_effect=ValueError('test measurement failure')), \
                     patch.object(level, 'check_stop'), \
                     patch('sys.stdout', new_callable=io.StringIO):
                    result = level.main(args)
                self.assertEqual(result, 1 if moving else 0)
                self.assertEqual(bus.writes, [])
                self.assertEqual(path.read_bytes(), original)
                self.assertTrue(imu.closed)
                report = json.loads(next((root/'logs').glob('hip_level_*.json')).read_text())
                self.assertEqual(report['status'], 'stopped' if moving else 'measured')
                self.assertFalse(report['calibration_saved'])
                self.assertEqual(report['algorithm_version'], 'adaptive-encoder-v2')

    def test_fault_after_small_move_retains_targets_and_never_saves_or_unloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'robot.json'
            cfg = calibrated()
            path.write_text(json.dumps(cfg))
            original = path.read_bytes()
            bus = FakeBus(cfg)
            imu = FakeImu()

            def move_then_fail(plant, report, **kwargs):
                plant.move((2, -1))
                raise ValueError('response changed after moving')

            with patch.object(level, 'ROOT', root), \
                 patch.object(level, 'stop_signals', return_value=nullcontext()), \
                 patch.object(level, 'exclusive_serial', return_value=nullcontext()), \
                 patch.object(level, 'HipOnlyBus', return_value=bus), \
                 patch.object(level, 'ProcessImuReader', return_value=imu), \
                 patch.object(level, 'level_hips', side_effect=move_then_fail), \
                 patch.object(level, 'check_stop'), \
                 patch.object(level.time, 'sleep'), \
                 patch('sys.stdout', new_callable=io.StringIO):
                result = level.main(['--config', str(path), '--move',
                                     '--ground-supported', '--save-home'])
            self.assertEqual(result, 1)
            self.assertEqual(len(bus.writes), 2)
            self.assertEqual(bus.writes[-1], ('positions', level.hip_targets(cfg, (2, -1))))
            self.assertTrue(all(bus.torque.values()))
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(imu.closed)
            report = json.loads(next((root/'logs').glob('hip_level_*.json')).read_text())
            self.assertEqual(report['last_sent_offset_ticks'], [2, -1])
            self.assertEqual(report['status'], 'stopped')
            self.assertFalse(report['calibration_saved'])


if __name__ == '__main__':
    unittest.main()
