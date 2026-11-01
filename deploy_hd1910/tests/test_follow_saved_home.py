"""Saved HOME target following: virtual time, in-memory bus, no robot access."""

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import calibration_arrays, default_config
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks, ticks_to_q
from microduck_deploy.servo import ServoFeedback, ServoTimeout


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'follow_saved_home.py'
SPEC = importlib.util.spec_from_file_location('follow_saved_home_test', SCRIPT)
follow = importlib.util.module_from_spec(SPEC)
_path = sys.path[:]
try:
    SPEC.loader.exec_module(follow)
finally:
    sys.path[:] = _path


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


class FakeBus:
    def __init__(self, cfg, clock):
        self.clock = clock
        self.ids = [joint['id'] for joint in cfg['joints']]
        home = q_to_ticks(HOME, *calibration_arrays(cfg))
        self.positions = dict(zip(self.ids, (home + np.array([25, -25] * 7)).tolist()))
        self.goals = self.positions.copy()
        self.configurations = {
            sid: {'register_profile': 'hls_2', 'torque': 0, 'mode': 4,
                  'angle_resolution': 1, 'position_min': 0, 'position_max': 4095,
                  'phase': 52, 'ram_gains': {'p': 5, 'd': 0, 'i': 0},
                  'eeprom_gains': {'p': 32, 'd': 40, 'i': 0}}
            for sid in self.ids
        }
        self.events = []
        self.writes = []
        self.reads = 0
        self.before_read = None
        self.after_write = None
        self.overrides = {}
        self.bad_preload = False
        self.fail_gain_verify = False
        self.fail_first_goal = False
        self.soft_gain_write_allowed = False
        self.soft_gain_p = 8
        self.communication_retries = []

    def read_configuration(self, sid):
        self.events.append(('configuration', self.clock(), sid))
        value = deepcopy(self.configurations[sid])
        value['motion_profile'] = self.read_motion_profile(sid)
        return value

    def read_motion_profile(self, sid):
        self.events.append(('goal_read', self.clock(), sid))
        goal = self.goals[sid]
        if self.bad_preload and any(event[0] == 'positions' for event in self.writes):
            goal += 2
        return {'goal_position_ticks': goal}

    def read_feedback_many(self, ids, *, sync):
        assert list(ids) == self.ids and sync is True
        self.reads += 1
        self.clock.sleep(.001)
        if self.before_read:
            self.before_read(self.reads)
        result = {}
        for sid in ids:
            goal = self.read_motion_profile(sid)['goal_position_ticks']
            value = ServoFeedback(sid, self.positions[sid], 0, 0, 7.4, 30, 0, 0,
                                  goal, 13., self.clock())
            result[sid] = replace(value, **self.overrides)
        return result

    def set_pid_ram(self, ids, p, d, i, *, verify):
        assert verify and (p, d, i) == (8, 0, 0)
        assert all(self.configurations[sid]['torque'] == 0 for sid in ids)
        event = ('gains', self.clock(), list(ids), p, d, i)
        self.writes.append(event)
        self.events.append(event)
        for sid in ids:
            self.configurations[sid]['ram_gains'] = {'p': p, 'd': d, 'i': i}
        if self.fail_gain_verify:
            raise ServoTimeout('RAM gain readback failed')
        self.events.append(('gains_verified', self.clock(), list(ids)))

    def sync_positions(self, targets):
        event = ('positions', self.clock(), dict(targets))
        self.writes.append(event)
        self.events.append(event)
        if self.fail_first_goal:
            self.fail_first_goal = False
            raise ServoTimeout('Ambiguous first goal write')
        self.goals.update(targets)
        for sid, target in targets.items():
            if self.configurations[sid]['torque']:
                self.positions[sid] = target
        if self.after_write:
            self.after_write()

    def set_torque(self, ids, enabled, *, verify):
        assert verify
        event = ('torque', self.clock(), bool(enabled), list(ids))
        self.writes.append(event)
        self.events.append(event)
        for sid in ids:
            self.configurations[sid]['torque'] = int(enabled)
            if enabled:
                self.positions[sid] = self.goals[sid]
        self.events.append(('torque_verified', self.clock(), bool(enabled)))

    def read(self, sid, address, size):
        if (address, size) == (40, 1):
            return bytes([self.configurations[sid]['torque']])
        if (address, size) == (42, 2):
            return struct.pack('<H', self.read_motion_profile(sid)['goal_position_ticks'])
        if (address, size) == (50, 3):
            return bytes(self.configurations[sid]['ram_gains'][key] for key in ('p', 'd', 'i'))
        raise AssertionError(f'Unexpected register read {address}/{size}')


def saved_report(cfg, goals):
    """Minimal source schema taken from the completed 20261101 00:49:26 run.

    Actual positions intentionally differ from targets: this operation follows
    saved commands, so it must never substitute previously measured positions.
    """
    joints = [{key: joint[key] for key in ('name', 'id', 'zero_tick', 'direction')}
              for joint in cfg['joints']]
    return {
        'mode': 'run', 'completed': True,
        'exit_state': 'last_goal_readback_verified_torque_kept_on',
        'joint_mapping': joints,
        'expected_ram_gains': {'p': 8, 'd': 0, 'i': 0},
        'model': {'model_sha256': 'a' * 64, 'contract': {
            'joint_names': list(JOINT_NAMES), 'home_angles_rad': HOME.astype(np.float32).tolist(),
            'task_id': 'Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910',
            'policy_family': 'hd1910_home_calibration_no_current_v2',
            'delta_limit_rad': math.radians(.25), 'offset_limit_rad': math.radians(8),
        }},
        'final_goal_ticks': {str(sid): tick for sid, tick in goals.items()},
        'final_torque_readback': {str(sid): 1 for sid in goals},
        'end_joint_positions': {
            'available': True, 'fresh_readback': True, 'source': 'normal_final_readback',
            'age_at_end_s': .001,
            'exit_state_at_capture': 'last_goal_readback_verified_torque_kept_on',
            'joints': [{'id': joint['id'], 'name': joint['name'],
                        'goal_ticks': goals[joint['id']], 'last_sent_goal_ticks': goals[joint['id']],
                        'actual_ticks': goals[joint['id']] + 7, 'status': 0}
                       for joint in cfg['joints']],
        },
    }


@pytest.fixture
def rig(tmp_path):
    cfg = default_config()
    cfg['calibration'] = {'pose': 'home'}
    cfg['imu_mount_verified'] = True
    ids = list(range(1, 10)) + list(range(11, 16))
    for index, joint in enumerate(cfg['joints']):
        joint.update(id=ids[index], direction=-1 if index % 2 else 1,
                     zero_tick=2048.25 + index / 8, mapping_verified=True)
    target = q_to_ticks(HOME + np.linspace(-.05, .05, 14), *calibration_arrays(cfg))
    goals = dict(zip(ids, target.tolist()))
    raw = saved_report(cfg, goals)
    source_path = tmp_path / 'completed_source.json'
    source_path.write_text(json.dumps(raw))
    original = source_path.read_bytes()
    clock = Clock()
    bus = FakeBus(cfg, clock)
    state = SimpleNamespace(cfg=cfg, goals=goals, raw=raw, source_path=source_path,
                            clock=clock, bus=bus, report_path=tmp_path / 'follow.json')
    state.source = lambda: follow.load_saved_target(source_path, cfg)

    def run(**overrides):
        args = dict(move=True, supported_confirmed=True, seconds=10., expected_p=8,
                    report_path=state.report_path, clock=clock, sleep=clock.sleep)
        args.update(overrides)
        return follow.run_follow(cfg, state.source(), bus, **args)

    state.run = run
    yield state
    assert source_path.read_bytes() == original, 'Following must not overwrite the source session'


def held_at_saved_goal(rig):
    for config in rig.bus.configurations.values():
        config['torque'] = 1
        config['ram_gains'] = {'p': 8, 'd': 0, 'i': 0}
    rig.bus.goals = rig.goals.copy()
    rig.bus.positions = rig.goals.copy()


def test_loader_uses_verified_saved_goals_not_previous_actual_positions(rig):
    source = rig.source()
    assert source['goal_ticks'] == rig.goals
    assert len(source['source_sha256']) == 64
    assert all(source['goal_ticks'][row['id']] != row['actual_ticks']
               for row in rig.raw['end_joint_positions']['joints'])


@pytest.mark.parametrize('bad', [
    'not_completed', 'monitor', 'unverified_exit', 'stale_end', 'missing_end',
    'fallback_end', 'missing_id', 'extra_id', 'wrong_name', 'wrong_zero',
    'wrong_direction', 'wrong_home', 'wrong_joint_order', 'off_torque',
    'goal_disagrees', 'last_sent_disagrees', 'fractional_tick', 'outside_tick_range',
])
def test_source_validation_refuses_untrusted_or_incompatible_targets(rig, tmp_path, bad):
    raw = deepcopy(rig.raw)
    if bad == 'not_completed': raw['completed'] = False
    elif bad == 'monitor': raw['mode'] = 'monitor'
    elif bad == 'unverified_exit': raw['exit_state'] = 'commands_stopped_last_goal_retained_unverified'
    elif bad == 'stale_end': raw['end_joint_positions']['fresh_readback'] = False
    elif bad == 'missing_end': raw['end_joint_positions']['available'] = False
    elif bad == 'fallback_end': raw['end_joint_positions']['source'] = 'last_observed_sample'
    elif bad == 'missing_id': del raw['final_goal_ticks']['1']
    elif bad == 'extra_id': raw['final_goal_ticks']['10'] = 2000
    elif bad == 'wrong_name': raw['joint_mapping'][0]['name'] = 'right_hip_yaw'
    elif bad == 'wrong_zero': raw['joint_mapping'][0]['zero_tick'] += 1.
    elif bad == 'wrong_direction': raw['joint_mapping'][0]['direction'] *= -1
    elif bad == 'wrong_home': raw['model']['contract']['home_angles_rad'][0] += .1
    elif bad == 'wrong_joint_order': raw['model']['contract']['joint_names'].reverse()
    elif bad == 'off_torque': raw['final_torque_readback']['1'] = 0
    elif bad == 'goal_disagrees': raw['end_joint_positions']['joints'][0]['goal_ticks'] += 1
    elif bad == 'last_sent_disagrees': raw['end_joint_positions']['joints'][0]['last_sent_goal_ticks'] += 1
    elif bad == 'fractional_tick': raw['final_goal_ticks']['1'] += .5
    elif bad == 'outside_tick_range': raw['final_goal_ticks']['1'] = 4096
    path = tmp_path / 'bad_source.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        follow.load_saved_target(path, rig.cfg)
    assert rig.bus.writes == []


def test_preview_never_writes_gains_goals_or_torque(rig):
    rig.run(move=False, supported_confirmed=False)
    assert rig.bus.writes == []


def test_on_same_goal_observes_existing_inner_loop_without_any_write(rig):
    held_at_saved_goal(rig)
    report = rig.run()
    assert report['completed'] is True
    assert report['mode'] == 'observe_existing_goal'
    assert rig.bus.writes == []
    assert rig.bus.reads > 1
    assert all(c['torque'] == 1 for c in rig.bus.configurations.values())


@pytest.mark.parametrize('fault', ['different_goal', 'wrong_gain', 'mixed_torque', 'wrong_mode'])
def test_incompatible_takeover_fails_before_any_write(rig, fault):
    held_at_saved_goal(rig)
    if fault == 'different_goal': rig.bus.goals[1] += 10
    elif fault == 'wrong_gain': rig.bus.configurations[1]['ram_gains']['p'] = 5
    elif fault == 'mixed_torque': rig.bus.configurations[1]['torque'] = 0
    else: rig.bus.configurations[1]['mode'] = 0
    with pytest.raises(ValueError):
        rig.run()
    assert rig.bus.writes == []


def test_off_start_verifies_gains_and_all_preload_goals_before_enable_then_slews(rig):
    initial = rig.bus.positions.copy()
    report = rig.run()
    assert report['completed'] is True
    assert report['mode'] == 'move_to_saved_goal'
    events = rig.bus.events
    enable = next(i for i, event in enumerate(events) if event[0] == 'torque' and event[2])
    preload = next(i for i, event in enumerate(events) if event[0] == 'positions')
    assert events[preload][2] == initial
    assert any(event[0] == 'gains_verified' for event in events[:enable])
    assert {event[2] for event in events[preload + 1:enable] if event[0] == 'goal_read'} == set(rig.goals)
    path = [event for event in events[enable + 1:] if event[0] == 'positions']
    assert path and path[-1][2] == rig.goals
    assert path[-1][1] - events[enable][1] >= 6. - 1e-9
    assert np.all(np.diff([event[1] for event in path]) >= .05 - 1e-9)
    previous = initial
    for _, _, targets in path:
        delta = np.asarray([targets[sid] - previous[sid] for sid in rig.goals])
        assert np.max(np.abs(delta)) * 360 / 4096 <= .25 + 1e-12
        previous = targets
    assert rig.bus.goals == rig.bus.positions == rig.goals
    assert all(c['torque'] == 1 and c['ram_gains'] == {'p': 8, 'd': 0, 'i': 0}
               for c in rig.bus.configurations.values())


@pytest.mark.parametrize('fault', ['bad_preload', 'fail_gain_verify'])
def test_failed_preload_or_gain_readback_never_enables_torque(rig, fault):
    setattr(rig.bus, fault, True)
    with pytest.raises((ValueError, ServoTimeout)):
        rig.run()
    assert not any(event[0] == 'torque' and event[2] for event in rig.bus.writes)
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())


@pytest.mark.parametrize('field,value', [('received_at', 90.), ('status', 4),
                                        ('temperature_c', 65), ('voltage_v', 3.5)])
def test_sensor_precheck_failures_are_zero_write(rig, field, value):
    rig.bus.overrides = {field: value}
    with pytest.raises(ValueError):
        rig.run()
    assert rig.bus.writes == []


def test_high_current_is_logged_without_current_threshold_stop(rig):
    held_at_saved_goal(rig)
    rig.bus.overrides['current_ma'] = 1500.
    report = rig.run()
    assert report['completed'] is True
    assert rig.bus.writes == []
    assert report['current_check'] == 'log-only'
    assert all(sample['current_ma'][1] == 1500. for sample in report['samples'])


def test_move_requires_supported_confirmation_before_motor_writes(rig):
    with pytest.raises(ValueError):
        rig.run(supported_confirmed=False)
    assert rig.bus.writes == []


def test_fault_after_torque_enable_releases_all_servos(rig):
    def fault_after_enable(_):
        if any(event[0] == 'torque' and event[2] for event in rig.bus.writes):
            rig.bus.overrides['temperature_c'] = 65
    rig.bus.before_read = fault_after_enable
    with pytest.raises(ValueError):
        rig.run()
    assert any(event[0] == 'torque' and event[2] for event in rig.bus.writes)
    assert any(event[0] == 'torque' and not event[2] for event in rig.bus.writes)
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())


def test_stop_after_a_path_write_prevents_further_goals_and_releases(rig):
    stop = threading.Event()
    def stop_on_first_active_write():
        if all(c['torque'] == 1 for c in rig.bus.configurations.values()):
            stop.set()
    rig.bus.after_write = stop_on_first_active_write
    with pytest.raises(ValueError):
        rig.run(stop_event=stop)
    writes = [event for event in rig.bus.writes if event[0] == 'positions']
    assert len(writes) == 2  # OFF preload, then the first interpolated target.
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())


def test_changed_source_is_rejected_at_hardware_boundary_before_any_write(rig):
    source = rig.source()
    original = rig.source_path.read_bytes()
    try:
        # Remains valid JSON and semantically identical, but is no longer the
        # exact source whose hash was validated before hardware lock acquisition.
        rig.source_path.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='changed after validation'):
            follow.run_follow(rig.cfg, source, rig.bus, move=True,
                              supported_confirmed=True)
    finally:
        rig.source_path.write_bytes(original)
    assert rig.bus.writes == []


def test_ambiguous_first_goal_failure_attempts_verified_off_and_never_enables(rig):
    rig.bus.fail_first_goal = True
    with pytest.raises(ServoTimeout, match='Ambiguous first goal'):
        rig.run()
    assert [event[2] for event in rig.bus.writes if event[0] == 'torque'] == [False]
    report = json.loads(rig.report_path.read_text())
    assert report['completed'] is False
    assert report['exit_state'] == 'torque_off_verified'


def test_observation_completion_does_not_claim_unreached_target_is_reached(rig):
    held_at_saved_goal(rig)
    rig.bus.positions[1] += 25  # About 2.2 degrees of persistent tracking error.
    report = rig.run()
    assert report['completed'] is True
    assert report['target_reached'] is False
    assert report['end_joint_positions']['joints'][0]['actual_ticks'] == rig.goals[1] + 25
    assert rig.bus.writes == []


def test_final_goal_readback_mismatch_cannot_be_reported_as_success(rig):
    held_at_saved_goal(rig)
    initial_configuration = rig.bus.read_configuration
    calls = 0

    def changed_at_final_readback(sid):
        nonlocal calls
        calls += 1
        if calls == 15:
            rig.bus.goals[1] += 2
        return initial_configuration(sid)

    rig.bus.read_configuration = changed_at_final_readback
    with pytest.raises(ValueError, match='goal readback differs'):
        rig.run()
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())
    assert not json.loads(rig.report_path.read_text())['completed']


@pytest.mark.parametrize('blocked', [False, True])
def test_cli_uses_explicit_config_and_shared_lock_before_opening_bus(rig, monkeypatch, blocked):
    config_path = rig.source_path.parent / 'custom_robot.json'
    config_path.write_text(json.dumps(rig.cfg))
    lock_calls = []
    opened = []

    @contextmanager
    def hardware_lock(cfg, path):
        lock_calls.append((cfg, path))
        if blocked:
            raise RuntimeError('Hardware lock is already held')
        yield

    @contextmanager
    def signal_context(stop):
        yield

    @contextmanager
    def fake_bus(**serial_options):
        opened.append(serial_options)
        yield rig.bus

    monkeypatch.setattr(follow, 'exclusive_hardware', hardware_lock)
    monkeypatch.setattr(follow, 'stop_signals', signal_context)
    monkeypatch.setattr(follow, 'FollowSavedBus', fake_bus)
    original_run = follow.run_follow

    def virtual_run(*args, **kwargs):
        return original_run(*args, **kwargs, clock=rig.clock, sleep=rig.clock.sleep)

    monkeypatch.setattr(follow, 'run_follow', virtual_run)
    result = follow.main(['--source-report', str(rig.source_path),
                          '--config', str(config_path), '--report', str(rig.report_path)])
    assert result == int(blocked)
    assert len(lock_calls) == 1 and lock_calls[0][1] == config_path
    assert lock_calls[0][0]['joints'] == rig.cfg['joints']
    assert bool(opened) is not blocked
    assert rig.bus.writes == []


def test_failed_cleanup_readback_is_labeled_old_and_preserves_original_error(rig, monkeypatch):
    def fail_after_preload(reads):
        if reads == 3:
            raise ServoTimeout('Original observation communication failure')
        if reads > 3:
            raise ServoTimeout('Cleanup feedback unavailable')

    def fail_csv(*args):
        raise OSError('CSV storage unavailable')

    rig.bus.before_read = fail_after_preload
    monkeypatch.setattr(follow, '_write_end_positions', fail_csv)
    with pytest.raises(ServoTimeout, match='Original observation communication failure'):
        rig.run()
    report = json.loads(rig.report_path.read_text())
    snapshot = report['end_joint_positions']
    assert snapshot['source'] == 'last_observed_sample'
    assert snapshot['fresh_readback'] is False
    assert snapshot['age_at_end_s'] > .05
    assert len(snapshot['joints']) == 14
    assert 'Cleanup feedback unavailable' in report['end_position_readback_error']
    assert 'CSV storage unavailable' in report['end_positions_csv_error']
    assert 'Original observation communication failure' in report['error']
    assert report['exit_state'] == 'torque_off_verified'
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())


def test_missing_initial_feedback_records_14_unavailable_rows_without_writes(rig):
    def unavailable(_):
        raise ServoTimeout('No initial feedback')

    rig.bus.before_read = unavailable
    with pytest.raises(ServoTimeout, match='No initial feedback'):
        rig.run()
    report = json.loads(rig.report_path.read_text())
    snapshot = report['end_joint_positions']
    assert snapshot['available'] is False
    assert snapshot['source'] == 'unavailable'
    assert snapshot['fresh_readback'] is False
    assert len(snapshot['joints']) == 14
    assert all(row['actual_ticks'] is None and row['last_sent_goal_ticks'] is None
               for row in snapshot['joints'])
    assert rig.bus.writes == []


def test_late_wakeup_releases_instead_of_catching_up_commands(rig):
    def delayed_sleep(seconds):
        rig.clock.sleep(seconds + .2)

    with pytest.raises(ValueError, match='gap'):
        rig.run(sleep=delayed_sleep)
    goals = [event for event in rig.bus.writes if event[0] == 'positions']
    assert len(goals) == 1  # Only the OFF preload was sent.
    assert all(c['torque'] == 0 for c in rig.bus.configurations.values())


@pytest.mark.parametrize('protected', ['source', 'csv'])
def test_output_collision_is_rejected_before_hardware_access(rig, protected):
    output = rig.source_path if protected == 'source' else rig.report_path
    if protected == 'csv':
        output.with_suffix('.positions.csv').write_text('Existing position report\n')
    with pytest.raises(ValueError, match='Output'):
        rig.run(report_path=output)
    assert rig.bus.events == []
