"""HOME takeover tests with virtual time and in-memory hardware only."""

from copy import deepcopy
from dataclasses import replace
import csv
import json
import math
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import calibration_arrays, default_config
from microduck_deploy import home_calibration_hardware as hardware
from microduck_deploy.home_calibration_hardware import CurrentMapping, RawCurrentFeedback, current_mapping_template
from microduck_deploy.home_calibration_policy import HomeCalibrationContract
from microduck_deploy.home_calibration_runtime import CalibrationStop, run_session
from microduck_deploy.imu import ImuState
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks, ticks_to_q
from microduck_deploy.servo import ServoStatusError, ServoTimeout


class FakeClock:
    def __init__(self):
        self.value = 100.
        self.sleeps = []
        self.oversleep_once = 0.

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds + self.oversleep_once
        self.oversleep_once = 0.


class FakeBus:
    def __init__(self, cfg, clock):
        self.clock = clock
        self.ids = [joint['id'] for joint in cfg['joints']]
        self.goals = dict(zip(self.ids, q_to_ticks(HOME, *calibration_arrays(cfg)).tolist()))
        self.positions = self.goals.copy()
        self.configurations = {
            sid: {'register_profile': 'hls_2', 'torque': 1, 'mode': 4,
                  'angle_resolution': 1, 'ram_gains': {'p': 5, 'd': 0, 'i': 0},
                  'position_min': 0, 'position_max': 4095}
            for sid in self.ids
        }
        self.goal_writes = []
        self.torque_writes = []
        self.feedback_reads = 0
        self.torque_reads = []
        self.before_sample = None
        self.after_write = None
        self.fail_goal = False
        self.fail_release = False
        self.write_delay_s = 0.
        self.current_raw = 2
        self.driver_current_ma = 13.

    def read_configuration(self, sid):
        return deepcopy(self.configurations[sid])

    def read_feedback_many(self, ids, sync):
        assert sync is True and ids == self.ids
        self.feedback_reads += 1
        if self.before_sample:
            self.before_sample(self.feedback_reads)
        return {
            sid: RawCurrentFeedback(
                servo_id=sid, position_ticks=self.positions[sid], velocity_raw=0,
                load_raw=0, voltage_v=7.4, temperature_c=30, status=0, moving=0,
                goal_ticks=self.goals[sid], current_ma=self.driver_current_ma,
                received_at=self.clock(), current_register_raw=self.current_raw,
            ) for sid in ids
        }

    def sync_positions(self, targets):
        self.goal_writes.append((self.clock(), dict(targets)))
        if self.fail_goal:
            raise ServoTimeout('Ambiguous first goal write')
        self.clock.advance(self.write_delay_s)
        self.goals.update(targets)
        self.positions.update(targets)
        if self.after_write:
            self.after_write()

    def read(self, sid, address, count):
        assert (address, count) == (40, 1)
        self.torque_reads.append(sid)
        return bytes((self.configurations[sid]['torque'],))

    def set_torque(self, ids, enabled, verify):
        self.torque_writes.append((list(ids), enabled, verify))
        assert enabled is False and verify is True
        if self.fail_release:
            raise ServoTimeout('Torque OFF delivery unconfirmed')
        for sid in ids:
            self.configurations[sid]['torque'] = 0


class FakeImu:
    def __init__(self, clock):
        self.clock = clock
        self.gravity = np.array([0., 0., -1.])
        self.gyro = np.zeros(3)

    def latest(self, max_age):
        return ImuState(self.clock(), self.gyro.copy(), self.gravity.copy(), np.array([0., 0., 9.80665]))


class FakePolicy:
    def __init__(self):
        self.contract = HomeCalibrationContract(
            task_id='Mjlab-HomeCalibration-Flat-MicroDuck-HD1910',
            policy_family='hd1910_home_calibration_v1', run_path='fake/model_100.pt',
            joint_names=JOINT_NAMES, home_angles_rad=tuple(HOME.astype(np.float32)),
            control_dt=.05, actor_frame_dim=49, history_length=8, actor_dim=392,
            encoder_ticks=4096, current_normalization_epsilon_a=.02,
            delta_limit_rad=math.radians(.25), offset_limit_rad=math.radians(8),
            settle_seconds=.5,
        )
        self.observations = []
        self.on_infer = None

    def inspect(self):
        return {'model_sha256': 'fake', 'input_shape': [1, 392], 'output_shape': [1, 14]}

    def infer(self, observation):
        self.observations.append(np.asarray(observation).copy())
        if self.on_infer:
            result = self.on_infer(len(self.observations))
            if result is not None:
                return result
        return np.ones(14, dtype=np.float32)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    cfg = default_config()
    cfg['calibration'] = {'pose': 'home'}
    cfg['imu_mount_verified'] = True
    for index, joint in enumerate(cfg['joints']):
        joint.update(id=index + 1, direction=1, mapping_verified=True,
                     zero_tick=2048 - HOME[index] * 4096 / math.tau)
    clock = FakeClock()
    # Keep all actual measurement validation, replacing only its clock source.
    monkeypatch.setattr(hardware, 'time', SimpleNamespace(monotonic=clock))
    spec = current_mapping_template(cfg)
    for row in spec['joints']:
        row['verified'] = True
    state = SimpleNamespace(cfg=cfg, clock=clock, policy=FakePolicy(),
                            bus=FakeBus(cfg, clock), imu=FakeImu(clock),
                            mapping=CurrentMapping.from_dict(spec, cfg),
                            path=tmp_path/'session.json')

    def run(**overrides):
        kwargs = dict(mode='run', seconds=.2, report_path=state.path,
                      clock=clock, sleep=clock.sleep)
        kwargs.update(overrides)
        return run_session(state.cfg, state.policy, state.bus, state.imu, state.mapping, **kwargs)

    state.run = run
    state.report = lambda: json.loads(state.path.read_text())
    return state


def test_success_runs_at_20hz_settles_half_second_and_leaves_verified_hold(rig):
    report = rig.run(seconds=.65)
    writes = rig.bus.goal_writes
    assert len(writes) == 13
    np.testing.assert_allclose(np.diff([stamp for stamp, _ in writes]), .05, atol=1e-12)
    home_ticks = q_to_ticks(HOME, *calibration_arrays(rig.cfg))
    for _, goals in writes[:10]:
        np.testing.assert_array_equal(list(goals.values()), home_ticks)
    assert writes[10][1] != writes[9][1]
    np.testing.assert_allclose(report['command_correction_deg'], .75, atol=1e-6)
    assert rig.bus.torque_writes == []
    assert rig.bus.torque_reads == rig.bus.ids
    assert report['completed'] and report['goal_writes_sent'] == 13
    assert report['exit_state'] == 'last_goal_readback_verified_torque_kept_on'
    assert report['duration_elapsed_s'] == pytest.approx(.65)
    assert report['joint_mapping'] == [
        {key: joint[key] for key in ('name', 'id', 'zero_tick', 'direction')}
        for joint in rig.cfg['joints']
    ]
    # Reset history repeats the first sensor frame; then actual frames enter.
    first = rig.policy.observations[0].reshape(8, 49)
    np.testing.assert_array_equal(first, np.repeat(first[:1], 8, axis=0))


def test_end_positions_success_contains_all_joint_measurements_and_csv(rig, capsys):
    report = rig.run(seconds=.65)
    snapshot = report['end_joint_positions']
    assert snapshot['available'] and snapshot['fresh_readback']
    assert snapshot['source'] == 'normal_final_readback'
    rows = snapshot['joints']
    assert [row['id'] for row in rows] == rig.bus.ids
    assert [row['name'] for row in rows] == list(JOINT_NAMES)
    assert [row['actual_ticks'] for row in rows] == list(rig.bus.positions.values())
    assert [row['last_sent_goal_ticks'] for row in rows] == list(rig.bus.goals.values())
    np.testing.assert_allclose([row['tracking_error_deg'] for row in rows], 0.)
    np.testing.assert_allclose([row['actual_rad'] for row in rows], report['final_q_reported_rad'])
    with open(report['end_positions_csv']) as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 14
    assert all(row['source'] == 'normal_final_readback' for row in exported)
    event = json.loads(capsys.readouterr().out)
    assert event['event'] == 'end_joint_positions' and len(event['joints']) == 14


def test_tracking_fault_records_rejected_sample_and_separate_post_stop_positions(rig):
    def move_on_reads(count):
        if count == 3:
            rig.bus.positions[4] += 69  # The rejected sample exceeds 6 degrees.
        elif count == 4:
            rig.bus.positions[4] += 2   # Position changed again before final capture.
    rig.bus.before_sample = move_on_reads
    with pytest.raises(CalibrationStop, match='ID4: excessive tracking error'):
        rig.run()
    report = rig.report()
    assert len(rig.bus.goal_writes) == 1 and not rig.bus.torque_writes
    assert report['exit_state'] == 'commands_stopped_last_goal_retained_unverified'
    fault = report['last_observed_joint_positions']['joints'][3]
    end = report['end_joint_positions']
    assert end['source'] == 'post_stop_readback' and end['fresh_readback']
    assert fault['tracking_error_deg'] == pytest.approx(69 * 360 / 4096)
    assert end['joints'][3]['actual_ticks'] == fault['actual_ticks'] + 2
    assert len(end['joints']) == 14


def test_end_position_read_failure_preserves_original_fault_and_labels_old_sample(rig):
    def read_fault(count):
        if count == 3:
            rig.bus.positions[4] += 69
        elif count >= 4:
            raise ServoTimeout('Post-stop bus unavailable')
    rig.bus.before_sample = read_fault
    with pytest.raises(CalibrationStop, match='ID4: excessive tracking error'):
        rig.run()
    report = rig.report()
    end = report['end_joint_positions']
    assert 'excessive tracking error' in report['error']
    assert end['available'] and not end['fresh_readback']
    assert end['source'] == 'last_observed_sample'
    assert 'Post-stop bus unavailable' in end['readback_error']
    assert len(end['joints']) == 14
    with open(report['end_positions_csv']) as stream:
        rows = list(csv.DictReader(stream))
    assert all(row['source'] == 'last_observed_sample' and row['fresh_readback'] == 'False'
               for row in rows)


def test_end_positions_on_total_bus_failure_include_nulls_for_every_joint(rig):
    def no_feedback(_):
        raise ServoTimeout('No feedback')
    rig.bus.before_sample = no_feedback
    with pytest.raises(ValueError, match='No feedback'):
        rig.run()
    end = rig.report()['end_joint_positions']
    assert not end['available'] and not end['fresh_readback']
    assert [row['id'] for row in end['joints']] == rig.bus.ids
    assert all(row['actual_ticks'] is None and row['last_sent_goal_ticks'] is None
               for row in end['joints'])
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def test_final_position_read_happens_after_required_torque_release(rig):
    use_no_current_policy(rig)
    def overcurrent(count):
        if count == 3:
            rig.bus.driver_current_ma = 801.
        elif count == 4:
            assert all(c['torque'] == 0 for c in rig.bus.configurations.values())
    rig.bus.before_sample = overcurrent
    with pytest.raises(CalibrationStop, match='current exceeds'):
        rig.run()
    end = rig.report()['end_joint_positions']
    assert end['exit_state_at_capture'] == 'torque_off_verified'
    assert end['fresh_readback'] and end['source'] == 'post_stop_readback'


def test_failed_report_save_does_not_replace_motion_fault(rig, monkeypatch, capsys):
    from microduck_deploy import home_calibration_runtime as runtime
    def broken_report(*args):
        raise OSError('disk full')
    def tracking_fault(count):
        if count == 3:
            rig.bus.positions[4] += 69
    rig.bus.before_sample = tracking_fault
    monkeypatch.setattr(runtime, '_write_report', broken_report)
    with pytest.raises(CalibrationStop, match='ID4: excessive tracking error'):
        rig.run()
    event = json.loads(capsys.readouterr().out)
    assert 'disk full' in event['report_write_error']
    assert len(rig.bus.goal_writes) == 1 and not rig.bus.torque_writes


@pytest.mark.parametrize('expected_p,held_p,accepted', [(8, 8, True), (8, 5, False), (5, 8, False)])
def test_gain_override_requires_matching_hardware_without_changing_it(rig, expected_p, held_p, accepted):
    for cfg in rig.bus.configurations.values():
        cfg['ram_gains']['p'] = held_p
    if accepted:
        report = rig.run(expected_p=expected_p)
        assert report['completed']
        assert report['expected_ram_gains'] == {'p': 8, 'd': 0, 'i': 0}
        assert report['training_nominal_ram_p'] == 5
        assert report['gain_override_from_training'] is True
    else:
        with pytest.raises(ValueError):
            rig.run(expected_p=expected_p)
        assert rig.bus.goal_writes == rig.bus.torque_writes == []
    assert all(cfg['ram_gains']['p'] == held_p for cfg in rig.bus.configurations.values())


def test_late_wakeup_cannot_cause_catchup_goal_burst(rig):
    rig.clock.oversleep_once = .03
    sample_times = []
    rig.policy.on_infer = lambda _: sample_times.append(rig.clock())
    report = rig.run()
    intervals = np.diff([stamp for stamp, _ in rig.bus.goal_writes])
    assert np.all(intervals >= .05 - 1e-12)
    assert np.all(np.diff(sample_times) >= .05 - 1e-12)
    assert report['duration_elapsed_s'] == pytest.approx(.23)


def test_nonzero_serial_write_time_does_not_accumulate_over_ten_seconds(rig):
    use_no_current_policy(rig)
    rig.bus.write_delay_s = .002
    rig.bus.before_sample = lambda _: rig.clock.advance(.008)
    rig.policy.on_infer = lambda _: rig.clock.advance(.008)
    report = rig.run(seconds=10.)
    stamps = np.array([stamp for stamp, _ in rig.bus.goal_writes])
    rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    assert report['completed'] and report['samples'] == report['goal_writes_sent'] == 200
    np.testing.assert_allclose(np.diff(stamps), .05, atol=1e-12)
    np.testing.assert_allclose([row['cycle_s'] for row in rows], .018, atol=1e-12)
    np.testing.assert_allclose([row['timings']['sensor_read_s'] for row in rows], .008, atol=1e-12)
    np.testing.assert_allclose([row['timings']['inference_s'] for row in rows], .008, atol=1e-12)
    np.testing.assert_allclose([row['timings']['command_write_s'] for row in rows], .002, atol=1e-12)
    np.testing.assert_allclose([row['timings']['sample_to_dispatch_s'] for row in rows], .016, atol=1e-12)
    np.testing.assert_allclose(np.diff([row['timings']['dispatch_elapsed_s'] for row in rows]), .05, atol=1e-12)
    assert report['duration_elapsed_s'] == pytest.approx(10.024)
    assert report['exit_state'] == 'last_goal_readback_verified_torque_kept_on'
    assert rig.bus.torque_reads == rig.bus.ids and not rig.bus.torque_writes


def test_variable_processing_and_late_wake_preserve_dispatch_cadence_and_horizon(rig):
    use_no_current_policy(rig)
    rig.bus.write_delay_s = .002
    rig.bus.before_sample = lambda count: rig.clock.advance(.004 if count % 2 else .010)
    def inference(count):
        rig.clock.advance(.002 if count % 2 else .006)
        if count == 4:
            rig.clock.oversleep_once = .020
    rig.policy.on_infer = inference
    started = rig.clock() + .004  # First read is preflight, outside the episode.
    report = rig.run(seconds=10.)
    stamps = np.array([stamp for stamp, _ in rig.bus.goal_writes])
    rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    assert report['completed']
    assert np.all(np.diff(stamps) >= .05 - 1e-12)
    assert np.all(stamps < started + 10.)
    assert max(row['cycle_s'] for row in rows) < .05
    assert report['exit_state'] == 'last_goal_readback_verified_torque_kept_on'
    assert rig.bus.torque_reads == rig.bus.ids and not rig.bus.torque_writes


def test_repeated_dispatch_wait_jitter_moves_sample_phase_without_accumulating(rig):
    use_no_current_policy(rig)
    rig.bus.write_delay_s = .002
    rig.bus.before_sample = lambda _: rig.clock.advance(.008)
    pending_dispatch = [False]
    def inference(count):
        rig.clock.advance(.008 if count % 2 else .0078)
        pending_dispatch[0] = True
    rig.policy.on_infer = inference
    rig.bus.after_write = lambda: pending_dispatch.__setitem__(0, False)
    original_latest = rig.imu.latest
    def latest(max_age):
        if pending_dispatch[0]:
            rig.clock.advance(.0003)  # Post-wait IMU/lock overhead on every step.
        return original_latest(max_age)
    rig.imu.latest = latest
    def sleep(seconds):
        if pending_dispatch[0]:
            rig.clock.oversleep_once = .0003  # Repeated command-wait OS jitter.
        rig.clock.sleep(seconds)
    started = rig.clock() + .008
    report = rig.run(seconds=10., sleep=sleep)
    stamps = np.array([stamp for stamp, _ in rig.bus.goal_writes])
    rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    assert report['completed'] and 195 <= report['goal_writes_sent'] <= 200
    assert np.all(np.diff(stamps) >= .05 - 1e-12)
    sample_times = [row['timings']['dispatch_elapsed_s'] - row['timings']['sample_to_dispatch_s']
                    for row in rows]
    assert np.all(np.diff(sample_times) >= .05 - 1e-12)
    assert np.all(stamps < started + 10.)
    assert max(row['cycle_s'] for row in rows) < .019
    assert max(row['timings']['command_wait_s'] for row in rows) < .001
    assert report['exit_state'] == 'last_goal_readback_verified_torque_kept_on'
    assert rig.bus.torque_reads == rig.bus.ids and not rig.bus.torque_writes


def test_monitor_never_writes_or_accumulates_hypothetical_corrections(rig):
    actual = q_to_ticks(HOME + math.radians(3), *calibration_arrays(rig.cfg))
    rig.bus.goals = dict(zip(rig.bus.ids, actual.tolist()))
    offset = ticks_to_q(actual, *calibration_arrays(rig.cfg)) - np.asarray(rig.policy.contract.home_angles_rad)
    report = rig.run(mode='monitor')
    assert report['completed'] and report['goal_write_attempts'] == 0
    assert rig.bus.goal_writes == rig.bus.torque_writes == []
    for observation in rig.policy.observations:
        np.testing.assert_allclose(observation.reshape(8, 49)[-1, 35:], offset, atol=1e-7)
    samples = [json.loads(row) for row in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    for row in samples:
        np.testing.assert_allclose(row['proposed_target_rad'],
                                   np.asarray(rig.policy.contract.home_angles_rad) + offset + math.radians(.25), atol=1e-7)


def test_monitor_without_current_mapping_only_records_raw_sensors(rig):
    rig.mapping = None
    report = rig.run(mode='monitor')
    assert report['completed']
    assert rig.policy.observations == []
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def use_no_current_policy(rig):
    rig.policy.contract = replace(
        rig.policy.contract,
        task_id='Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910',
        policy_family='hd1910_home_calibration_no_current_v2',
        actor_frame_dim=34, actor_dim=272, current_normalization_epsilon_a=None,
    )
    rig.mapping = None


@pytest.mark.parametrize('mode', ['run', 'monitor'])
@pytest.mark.parametrize('current_check', ['stop', 'log-only'])
def test_no_current_policy_runs_with_no_mapping_and_does_not_skip_inference(rig, mode, current_check):
    use_no_current_policy(rig)
    report = rig.run(mode=mode, current_check=current_check)
    assert report['completed'] and report['requires_current_input'] is False
    assert report['current_safety_signal'] == 'driver_current_ma_abs'
    assert len(rig.policy.observations) == 4
    assert all(obs.shape == (272,) for obs in rig.policy.observations)
    assert len(rig.bus.goal_writes) == (4 if mode == 'run' else 0)
    assert rig.bus.torque_writes == []
    rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    assert all(row['inference_skipped'] is None and row['current_joint_a'] is None for row in rows)


@pytest.mark.parametrize('fault', ['positive_current', 'negative_current', 'nonfinite_current',
                                  'tilt', 'gyro', 'goal', 'tracking'])
def test_no_current_keeps_all_safety_checks_before_first_write(rig, fault):
    use_no_current_policy(rig)
    if fault in ('positive_current', 'negative_current', 'nonfinite_current'):
        rig.bus.driver_current_ma = {'positive_current': 801., 'negative_current': -801.,
                                     'nonfinite_current': float('nan')}[fault]
    elif fault == 'tilt':
        rig.imu.gravity = np.array([1., 0., 0.])
    elif fault == 'gyro':
        rig.imu.gyro[0] = .5
    elif fault == 'goal':
        rig.bus.goals[1] += 2
    else:
        rig.bus.positions[1] += 100
    with pytest.raises((ValueError, CalibrationStop)):
        rig.run()
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def test_no_current_overcurrent_after_takeover_requests_release(rig):
    use_no_current_policy(rig)
    def overcurrent_after_first_write(count):
        if count == 3:
            rig.bus.driver_current_ma = -801.
    rig.bus.before_sample = overcurrent_after_first_write
    with pytest.raises(CalibrationStop, match='current exceeds'):
        rig.run()
    assert len(rig.bus.goal_writes) == 1
    assert len(rig.bus.torque_writes) == 1


@pytest.mark.parametrize('phase,trigger,writes_before_fault', [
    ('preflight', 1, 0), ('control', 3, 1), ('final', 6, 4),
])
@pytest.mark.parametrize('current_check', ['stop', 'log-only'])
@pytest.mark.parametrize('current_ma', [-1235., 1235.])
def test_current_log_only_applies_to_each_phase_and_preserves_real_telemetry(
        rig, phase, trigger, writes_before_fault, current_check, current_ma):
    use_no_current_policy(rig)
    def high_current(count):
        if count >= trigger:
            rig.bus.driver_current_ma = current_ma
            rig.bus.current_raw = 190 | (0x8000 if current_ma < 0 else 0)
    rig.bus.before_sample = high_current
    if current_check == 'stop':
        with pytest.raises(CalibrationStop, match='current exceeds'):
            rig.run(current_check=current_check)
        report = rig.report()
        assert not report['completed'] and report['goal_writes_sent'] == writes_before_fault
        assert len(rig.bus.torque_writes) == int(writes_before_fault > 0)
        assert report['current_stop_ma'] == 800.
    else:
        report = rig.run(current_check=current_check)
        assert report['completed'] and report['goal_writes_sent'] == 4
        assert rig.bus.feedback_reads == 6 and not rig.bus.torque_writes
        assert report['current_stop_ma'] is None
        assert report['final_diagnostics']['driver_current_ma_by_id']['1'] == current_ma
        if phase != 'final':
            rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
            assert rows[-1]['diagnostics']['driver_current_ma_by_id']['1'] == current_ma
            assert rows[-1]['diagnostics']['current_register_raw_by_id']['1'] == rig.bus.current_raw
            assert rows[-1]['current_joint_a'] is None
    assert report['current_check'] == current_check


@pytest.mark.parametrize('fault,release', [
    ('temperature', True), ('current_status', True), ('temperature_status', True),
    ('tilt', True), ('gyro', True), ('voltage', False), ('stale', False),
    ('nonfinite_current', False), ('goal', False), ('tracking', False),
])
def test_current_log_only_keeps_other_feedback_protections_after_takeover(rig, fault, release):
    use_no_current_policy(rig)
    rig.bus.driver_current_ma = 1235.
    original_read = rig.bus.read_feedback_many
    def feedback(ids, sync):
        values = original_read(ids, sync)
        if rig.bus.feedback_reads != 3:
            return values
        row = values[1]
        if fault == 'temperature':
            row = replace(row, temperature_c=rig.cfg['control']['servo_max_temperature_c'])
        elif fault.endswith('_status'):
            row = replace(row, status=1 << (3 if fault == 'current_status' else 2))
        elif fault == 'voltage':
            row = replace(row, voltage_v=.1)
        elif fault == 'stale':
            row = replace(row, received_at=rig.clock() - .051)
        elif fault == 'nonfinite_current':
            row = replace(row, current_ma=float('nan'))
        elif fault == 'goal':
            row = replace(row, goal_ticks=row.goal_ticks + 2)
        elif fault == 'tracking':
            row = replace(row, position_ticks=row.position_ticks + 100)
        elif fault == 'tilt':
            rig.imu.gravity = np.array([1., 0., 0.])
        else:
            rig.imu.gyro[0] = 3.
        values[1] = row
        return values
    rig.bus.read_feedback_many = feedback
    with pytest.raises((ValueError, CalibrationStop)):
        rig.run(current_check='log-only')
    assert len(rig.bus.goal_writes) == 1
    assert len(rig.bus.torque_writes) == int(release)
    assert not rig.report()['completed']


def test_invalid_current_check_fails_before_sensor_reads_or_writes(rig):
    with pytest.raises(ValueError, match='current check'):
        rig.run(current_check='ignore')
    assert rig.bus.feedback_reads == 0
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def test_no_current_observation_is_independent_of_all_current_values_and_mappings(rig):
    use_no_current_policy(rig)
    # An unverified map is irrelevant to this actor and must not become a gate.
    rig.mapping = CurrentMapping.from_dict(current_mapping_template(rig.cfg), rig.cfg)
    def varying_current(count):
        rig.bus.current_raw = count * 123
        rig.bus.driver_current_ma = (-1 if count % 2 else 1) * count * 60.
    rig.bus.before_sample = varying_current
    report = rig.run(mode='monitor')
    assert report['completed']
    assert len(rig.policy.observations) == 4
    for observation in rig.policy.observations[1:]:
        np.testing.assert_array_equal(observation, rig.policy.observations[0])
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


@pytest.mark.parametrize('mode', ['run', 'monitor'])
@pytest.mark.parametrize('current_check', ['stop', 'log-only'])
def test_force_ratio_policy_runs_without_mapping_and_appends_only_fixed_reference(rig, mode, current_check):
    use_no_current_policy(rig)
    reference = np.arange(1, 15, dtype=float)
    reference /= reference.sum()
    rig.policy.contract = replace(
        rig.policy.contract,
        task_id='Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910',
        policy_family='hd1910_home_force_ratio_no_current_v1',
        actor_frame_dim=48, actor_dim=384, target_force_ratio=tuple(reference),
    )
    report = rig.run(mode=mode, current_check=current_check)
    assert report['completed'] and not report['requires_current_input']
    assert len(rig.policy.observations) == 4
    for observation in rig.policy.observations:
        assert observation.shape == (384,)
        np.testing.assert_array_equal(observation.reshape(8, 48)[:, 34:],
                                      np.repeat(reference.astype(np.float32)[None], 8, axis=0))
    assert len(rig.bus.goal_writes) == (4 if mode == 'run' else 0)
    assert rig.bus.torque_writes == []


@pytest.mark.parametrize('field,value', [('torque', 0), ('mode', 0),
                                        ('ram_gains', {'p': 16, 'd': 0, 'i': 0})])
def test_preflight_failure_never_changes_existing_hold_even_with_release_policy(rig, field, value):
    rig.bus.configurations[1][field] = value
    with pytest.raises(ValueError):
        rig.run(fault_action='release')
    assert rig.bus.goal_writes == rig.bus.torque_writes == []
    assert rig.report()['exit_state'] == 'no_motor_writes'


def test_wrong_initial_goal_is_rejected_without_returning_to_home(rig):
    rig.bus.goals[1] += 2
    with pytest.raises(CalibrationStop, match='goal readback'):
        rig.run()
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('current_check', ['stop', 'log-only'])
def test_missing_or_unverified_current_mapping_rejects_motion_before_hardware_reads(rig, missing, current_check):
    rig.mapping = None if missing else CurrentMapping.from_dict(current_mapping_template(rig.cfg), rig.cfg)
    with pytest.raises(ValueError, match='verified'):
        rig.run(current_check=current_check)
    assert rig.bus.feedback_reads == 0
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def test_slow_inference_cannot_send_first_goal(rig):
    rig.policy.on_infer = lambda _: rig.clock.advance(.051)
    with pytest.raises(CalibrationStop, match='target not sent'):
        rig.run()
    assert rig.bus.goal_writes == rig.bus.torque_writes == []
    assert not rig.report()['completed']
    assert rig.report()['duration_elapsed_s'] == pytest.approx(.051)


def test_slow_write_still_stops_further_goals_and_reports_elapsed(rig):
    rig.bus.write_delay_s = .051
    with pytest.raises(CalibrationStop, match='Control cycle exceeded'):
        rig.run()
    report = rig.report()
    assert not report['completed'] and report['goal_writes_sent'] == 1
    assert report['duration_elapsed_s'] == pytest.approx(.051)
    assert report['exit_state'] == 'commands_stopped_last_goal_retained_unverified'
    assert rig.bus.torque_writes == []
    row = json.loads(rig.path.with_suffix('.samples.jsonl').read_text())
    assert row['timings']['command_write_s'] == pytest.approx(.051)


@pytest.mark.parametrize('fault_action,release_count', [('hold', 0), ('release', 1)])
def test_failed_first_write_is_owned_but_never_reported_as_success(rig, fault_action, release_count):
    rig.bus.fail_goal = True
    with pytest.raises(ServoTimeout, match='Ambiguous'):
        rig.run(fault_action=fault_action)
    report = rig.report()
    assert not report['completed']
    assert report['goal_write_attempts'] == 1 and report['goal_writes_sent'] == 0
    assert len(rig.bus.torque_writes) == release_count
    np.testing.assert_array_equal(report['command_correction_last_sent_rad'], np.zeros(14))


def test_signal_during_inference_stops_before_first_write(rig):
    stop = threading.Event()
    rig.policy.on_infer = lambda _: stop.set()
    with pytest.raises(CalibrationStop, match='Stop requested'):
        rig.run(stop_event=stop)
    assert rig.bus.goal_writes == rig.bus.torque_writes == []


def test_signal_after_takeover_prevents_all_further_goals_and_keeps_hold_by_default(rig):
    stop = threading.Event()
    rig.bus.after_write = stop.set
    with pytest.raises(CalibrationStop, match='Stop requested'):
        rig.run(stop_event=stop)
    assert len(rig.bus.goal_writes) == 1
    assert rig.bus.torque_writes == []
    assert rig.report()['exit_state'] == 'commands_stopped_last_goal_retained_unverified'


@pytest.mark.parametrize('error,released', [(ServoTimeout('lost feedback'), False),
                                          (ServoStatusError(1, 1 << 2), True),
                                          (ServoStatusError(1, 1 << 3), True)])
def test_fault_release_is_limited_to_physical_danger_by_default(rig, error, released):
    def fail_after_first_goal(count):
        if count == 3:
            raise error
    rig.bus.before_sample = fail_after_first_goal
    with pytest.raises(ValueError):
        rig.run()
    assert len(rig.bus.goal_writes) == 1
    assert len(rig.bus.torque_writes) == int(released)
    assert not rig.report()['completed']


def test_release_failure_never_claims_torque_off_confirmed(rig):
    rig.bus.fail_goal = rig.bus.fail_release = True
    with pytest.raises(ServoTimeout):
        rig.run(fault_action='release')
    assert rig.report()['exit_state'] == 'torque_off_unconfirmed'


def test_current_limit_uses_verified_encoding_not_legacy_driver_decode(rig):
    spec = rig.mapping.to_dict()
    spec['encoding'] = 'twos_complement16'
    rig.mapping = CurrentMapping.from_dict(spec, rig.cfg)
    rig.bus.current_raw = 0xfffe  # verified mapping says -13 mA
    rig.bus.driver_current_ma = -(0xfffe & 0x7fff) * 6.5
    assert rig.run()['completed']
    assert rig.bus.torque_writes == []


def test_legacy_current_actor_log_only_keeps_actual_mapped_current_input(rig):
    rig.bus.current_raw = 190  # Verified mapping gives 1.235 A.
    report = rig.run(current_check='log-only')
    assert report['completed'] and report['requires_current_input']
    assert report['current_stop_ma'] is None and not rig.bus.torque_writes
    rows = [json.loads(line) for line in rig.path.with_suffix('.samples.jsonl').read_text().splitlines()]
    np.testing.assert_allclose(rows[-1]['current_joint_a'], 1.235)
    assert all(obs.shape == (392,) for obs in rig.policy.observations)


@pytest.mark.parametrize('fault', ['wrong_goal', 'torque_off', 'slow', 'stop'])
def test_final_readback_must_finish_on_time_without_stop_and_verify_goal_and_torque(rig, fault):
    stop = threading.Event()
    def final_fault(count):
        if count != 3:  # one preflight, one loop, one final read
            return
        if fault == 'wrong_goal':
            rig.bus.goals[1] += 2
        elif fault == 'torque_off':
            rig.bus.configurations[1]['torque'] = 0
        elif fault == 'slow':
            rig.clock.advance(.101)
        else:
            stop.set()
    rig.bus.before_sample = final_fault
    with pytest.raises(CalibrationStop):
        rig.run(seconds=.05, stop_event=stop)
    report = rig.report()
    assert not report['completed']
    assert len(rig.bus.goal_writes) == 1
    assert rig.bus.torque_writes == []
    assert report['exit_state'] == 'commands_stopped_last_goal_retained_unverified'


def test_invalid_inference_cannot_accumulate_into_last_sent_correction(rig):
    rig.policy.on_infer = lambda count: np.full(14, np.nan) if count == 12 else None
    with pytest.raises(ValueError, match='finite'):
        rig.run(seconds=.65)
    report = rig.report()
    assert report['goal_writes_sent'] == 11
    np.testing.assert_allclose(np.rad2deg(report['command_correction_last_sent_rad']), .25, atol=1e-6)
    assert rig.bus.torque_writes == []
    assert not report['completed']


def test_horizon_expiring_during_inference_does_not_apply_candidate(rig):
    rig.policy.on_infer = lambda count: rig.clock.advance(.03) if count == 11 else None
    report = rig.run(seconds=.52)
    assert report['completed'] and report['time_budget_reached']
    assert report['goal_writes_sent'] == 10
    np.testing.assert_array_equal(report['command_correction_rad'], np.zeros(14))
    assert max(stamp for stamp, _ in rig.bus.goal_writes) < 100.52
    assert rig.bus.torque_writes == []


def test_horizon_expiring_while_waiting_for_dispatch_does_not_send_again(rig):
    rig.bus.write_delay_s = .002
    rig.policy.on_infer = lambda count: rig.clock.advance(.035) if count == 1 else None
    report = rig.run(seconds=.08)
    assert report['completed'] and report['time_budget_reached']
    assert report['goal_writes_sent'] == 1 and len(rig.policy.observations) == 2
    assert rig.bus.goal_writes[0][0] == pytest.approx(100.035)
    assert report['exit_state'] == 'last_goal_readback_verified_torque_kept_on'
    np.testing.assert_array_equal(report['command_correction_rad'], np.zeros(14))
    assert rig.bus.torque_reads == rig.bus.ids and not rig.bus.torque_writes


def test_repeated_late_wakeups_end_at_wallclock_budget_without_catchup(rig):
    original_sleep = rig.clock.sleep
    def late_sleep(seconds):
        rig.clock.oversleep_once = .025
        original_sleep(seconds)
    report = rig.run(seconds=.2, sleep=late_sleep)
    assert report['completed'] and report['time_budget_reached']
    assert report['goal_writes_sent'] == 3
    assert max(stamp for stamp, _ in rig.bus.goal_writes) < 100.2
    assert np.all(np.diff([stamp for stamp, _ in rig.bus.goal_writes]) >= .05 - 1e-12)
