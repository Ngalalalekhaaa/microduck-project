"""Supported HOME handoff tests with memory-only servos and virtual time."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import signal
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import calibration_arrays, default_config, record_reference
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks
from microduck_deploy.servo import ServoBus, ServoFeedback


SCRIPT = Path(__file__).parents[1] / 'scripts' / 'supported_home.py'
SPEC = importlib.util.spec_from_file_location('supported_calibration_hold_test', SCRIPT)
supported = importlib.util.module_from_spec(SPEC)
_PATH = sys.path[:]
try:
    SPEC.loader.exec_module(supported)
finally:
    sys.path[:] = _PATH


class Clock:
    def __init__(self):
        self.now = 100.

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += max(0., seconds)


class TimedEvent:
    def __init__(self, clock):
        self.clock, self.flag = clock, False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True

    def wait(self, timeout=None):
        if not self.flag:
            self.clock.advance(timeout or 0.)
        return self.flag


class MemoryBus:
    def __init__(self, cfg, clock):
        self.clock, self.cfg = clock, cfg
        self.ids = [joint['id'] for joint in cfg['joints']]
        self.home = dict(zip(self.ids, q_to_ticks(HOME, *calibration_arrays(cfg)).tolist()))
        nearby = HOME.copy()
        nearby[0] += .1
        self.positions = dict(zip(self.ids, q_to_ticks(nearby, *calibration_arrays(cfg)).tolist()))
        self.goals = self.positions.copy()
        self.configurations = {
            sid: {'register_profile': 'hls_2', 'torque': 0, 'mode': 4,
                  'angle_resolution': 1, 'position_min': 0, 'position_max': 4095,
                  'phase': 0, 'eeprom_gains': {'p': 16, 'd': 0, 'i': 0},
                  'ram_gains': {'p': 16, 'd': 0, 'i': 0}}
            for sid in self.ids
        }
        self.communication_retries = []
        self.events = []
        self.writes = []
        self.open_options = []
        self.soft_gain_write_allowed = False
        self.bad_preload = False
        self.initial_feedback_change = {}
        self.fail_tracking_after_preload = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read_configuration(self, sid):
        self.events.append(('configuration', sid))
        result = deepcopy(self.configurations[sid])
        result['motion_profile'] = self.read_motion_profile(sid)
        return result

    def read_motion_profile(self, sid):
        self.events.append(('goal_read', sid))
        tick = self.goals[sid]
        if self.bad_preload and any(event[0] == 'positions' for event in self.writes):
            tick += 2
        return {'goal_position_ticks': tick}

    def read_feedback_many(self, ids, *, sync):
        assert list(ids) == self.ids and sync is True
        self.clock.advance(.001)
        result = {}
        goals_written = any(event[0] == 'positions' for event in self.writes)
        for sid in ids:
            position = self.positions[sid]
            if self.fail_tracking_after_preload and goals_written:
                position += 58
            value = ServoFeedback(sid, position, 0, 0, 7.4, 30, 0, 0,
                                  self.goals[sid], 13., self.clock.now)
            if self.initial_feedback_change:
                value = replace(value, **self.initial_feedback_change)
            result[sid] = value
        return result

    def set_pid_ram(self, ids, p, d, i, *, verify):
        assert self.soft_gain_write_allowed and verify
        assert all(self.configurations[sid]['torque'] == 0 for sid in ids)
        self.writes.append(('gains', list(ids), p, d, i))
        self.events.append(self.writes[-1])
        for sid in ids:
            self.configurations[sid]['ram_gains'] = {'p': p, 'd': d, 'i': i}

    def sync_positions(self, targets):
        self.writes.append(('positions', dict(targets)))
        self.events.append(self.writes[-1])
        self.goals.update(targets)
        for sid, target in targets.items():
            if self.configurations[sid]['torque']:
                self.positions[sid] = target

    def set_torque(self, ids, enabled, *, verify):
        assert verify
        self.writes.append(('torque', list(ids), enabled))
        self.events.append(self.writes[-1])
        for sid in ids:
            self.configurations[sid]['torque'] = int(enabled)
            if enabled:
                self.positions[sid] = self.goals[sid]

    def read(self, sid, address, count):
        if (address, count) == (50, 3):
            return bytes(self.configurations[sid]['ram_gains'][key] for key in ('p', 'd', 'i'))
        if (address, count) == (40, 1):
            return bytes((self.configurations[sid]['torque'],))
        if (address, count) == (42, 2):
            return struct.pack('<H', self.read_motion_profile(sid)['goal_position_ticks'])
        raise AssertionError(f'Unexpected register read {address}/{count}')

    def restore_gains(self, address, size, values):
        assert (address, size) == (50, 3), 'No EEPROM or unexpected write is permitted'
        self.writes.append(('restore_gains', dict(values)))
        for sid, raw in values.items():
            assert self.configurations[sid]['torque'] == 0
            self.configurations[sid]['ram_gains'] = dict(zip(('p', 'd', 'i'), raw))


@pytest.fixture
def rig(tmp_path, monkeypatch):
    cfg = default_config()
    for index, joint in enumerate(cfg['joints']):
        joint.update(id=index + 1, direction=-1 if index % 2 else 1, mapping_verified=True)
    record_reference(cfg, np.full(14, 2048), 'straight', STRAIGHT_REFERENCE)
    directory = tmp_path/'existing-deployment'
    directory.mkdir()
    path = directory/'robot-A.json'
    path.write_text(json.dumps(cfg, indent=2) + '\n')
    original = path.read_bytes()
    release = tmp_path/'staged-release'
    release.mkdir()
    clock = Clock()
    bus = MemoryBus(cfg, clock)
    locks, opened = [], []
    active = [False]

    @contextmanager
    def exclusive(configuration, config_path):
        locks.append((deepcopy(configuration), Path(config_path).resolve()))
        assert not active[0]
        active[0] = True
        try:
            yield
        finally:
            active[0] = False

    def open_bus(**options):
        assert active[0], 'Special HOME preparation must acquire the shared lock first'
        opened.append(options)
        return bus

    monkeypatch.setattr(supported, 'DEPLOY', release)
    monkeypatch.setattr(supported, 'exclusive_hardware', exclusive)
    monkeypatch.setattr(supported, 'PositionOnlyBus', open_bus)
    monkeypatch.setattr(supported, 'ServoBus', SimpleNamespace(
        _sync_ram=lambda instance, address, size, values: instance.restore_gains(address, size, values)))
    monkeypatch.setattr(supported, 'time', SimpleNamespace(
        monotonic=clock.monotonic, sleep=clock.advance, strftime=lambda _: 'FAKE_TIME'))
    monkeypatch.setattr(supported, 'threading', SimpleNamespace(Event=lambda: TimedEvent(clock)))
    monkeypatch.setattr(supported, 'signal', SimpleNamespace(
        SIGINT=signal.SIGINT, SIGTERM=signal.SIGTERM, SIGHUP=signal.SIGHUP,
        signal=lambda *args: None))
    state = SimpleNamespace(cfg=cfg, config=path, original=original, bus=bus,
                            clock=clock, locks=locks, opened=opened)
    state.run = lambda *extra: supported.main(
        ['--config', str(path), '--calibration-hold', '--move', *extra])
    state.report = lambda: json.loads(next(tmp_path.rglob('supported_home_*.json')).read_text())
    yield state
    assert path.read_bytes() == original, 'Preparation must not modify the saved calibration'


def set_held(bus, p=5):
    for value in bus.configurations.values():
        value['torque'] = 1
        value['ram_gains'] = {'p': p, 'd': 0, 'i': 0}


@pytest.mark.parametrize('p', [5, 8])
def test_off_start_preloads_and_reads_every_goal_before_enabling_then_keeps_home(rig, p):
    rig.run(*(() if p == 5 else ('--ram-p', str(p))))
    assert rig.bus.writes[0][0] == 'gains'
    assert rig.bus.writes[0][2:] == (p, 0, 0)
    enable_index = next(i for i, event in enumerate(rig.bus.events)
                        if event[0] == 'torque' and event[2])
    preload_index = next(i for i, event in enumerate(rig.bus.events) if event[0] == 'positions')
    read_ids = {event[1] for event in rig.bus.events[preload_index + 1:enable_index]
                if event[0] == 'goal_read'}
    assert read_ids == set(rig.bus.ids)
    assert all(value['torque'] == 1 for value in rig.bus.configurations.values())
    assert all(value['ram_gains'] == {'p': p, 'd': 0, 'i': 0}
               for value in rig.bus.configurations.values())
    assert not any(event[0] == 'torque' and not event[2] for event in rig.bus.writes)
    assert rig.bus.goals == rig.bus.positions == rig.bus.home
    report = rig.report()
    assert report['target_reached'] is True
    assert report['log_only_feedback'] is False
    assert report['gains_changed'] is True
    assert report['requested_ram_gains'] == {'p': p, 'd': 0, 'i': 0}


@pytest.mark.parametrize('p', [5, 8])
def test_already_held_correct_gains_do_not_cycle_torque_or_rewrite_gains(rig, p):
    set_held(rig.bus, p=p)
    rig.run(*(() if p == 5 else ('--ram-p', str(p))))
    assert rig.bus.goals == rig.bus.home
    assert rig.bus.writes and all(event[0] == 'positions' for event in rig.bus.writes)
    assert rig.report()['gains_changed'] is False


@pytest.mark.parametrize('held,requested', [(5, 8), (8, 5)])
def test_gain_change_while_held_refuses_without_even_torque_off(rig, held, requested):
    set_held(rig.bus, p=held)
    with pytest.raises(ValueError, match='relax'):
        rig.run('--ram-p', str(requested))
    assert rig.bus.writes == []


@pytest.mark.parametrize('fault', ['mixed_torque', 'wrong_on_gain', 'wrong_mode', 'wrong_resolution'])
def test_invalid_initial_state_is_rejected_before_any_write_including_torque_off(rig, fault):
    set_held(rig.bus)
    if fault == 'mixed_torque':
        rig.bus.configurations[1]['torque'] = 0
    elif fault == 'wrong_on_gain':
        rig.bus.configurations[1]['ram_gains']['p'] = 16
    elif fault == 'wrong_mode':
        rig.bus.configurations[1]['mode'] = 0
    else:
        rig.bus.configurations[1]['angle_resolution'] = 2
    with pytest.raises((ValueError, AssertionError)):
        rig.run()
    assert rig.bus.writes == []


@pytest.mark.parametrize('field,value', [('current_ma', 900.), ('temperature_c', 55),
                                        ('voltage_v', 4.5), ('status', 4), ('received_at', 99.)])
def test_dedicated_daily_recovery_keeps_strict_initial_feedback_checks(rig, field, value):
    rig.bus.initial_feedback_change = {field: value}
    with pytest.raises(ValueError):
        rig.run()
    assert rig.bus.writes == []


def test_tracking_failure_after_preload_stops_the_path_and_cleans_up(rig):
    set_held(rig.bus)
    rig.bus.fail_tracking_after_preload = True
    with pytest.raises(ValueError, match='tracking|5 degrees'):
        rig.run()
    assert len([event for event in rig.bus.writes if event[0] == 'positions']) == 1
    assert any(event[0] == 'torque' and not event[2] for event in rig.bus.writes)
    assert rig.report()['result'] == 'aborted'


@pytest.mark.parametrize('p', [5, 8])
def test_bad_preload_readback_never_enables_torque(rig, p):
    rig.bus.bad_preload = True
    with pytest.raises((ValueError, AssertionError)):
        rig.run('--ram-p', str(p))
    assert not any(event[0] == 'torque' and event[2] for event in rig.bus.writes)
    assert all(value['torque'] == 0 for value in rig.bus.configurations.values())
    assert rig.report().get('target_reached') is not True
    assert all(value['ram_gains'] == {'p': 16, 'd': 0, 'i': 0}
               for value in rig.bus.configurations.values())


def test_explicit_config_drives_joint_mapping_and_shared_lock_location(rig):
    rig.run()
    assert len(rig.locks) == 1
    assert rig.locks[0][1] == rig.config.resolve()
    assert rig.locks[0][0]['joints'] == rig.cfg['joints']
    assert len(rig.opened) == 1
    assert rig.opened[0]['port'] == rig.cfg['serial']['port']
    assert rig.bus.home == dict(zip(
        rig.bus.ids, q_to_ticks(HOME, *calibration_arrays(rig.cfg)).tolist()))


@pytest.mark.parametrize('extra', [('--target', 'straight'),
                                   ('--jaw-command-fd', '123')])
def test_incompatible_modes_are_rejected_before_hardware_open(rig, extra):
    with pytest.raises((SystemExit, ValueError)):
        rig.run(*extra)
    assert rig.opened == []
    assert rig.bus.writes == []


@pytest.mark.parametrize('p', [5, 8])
def test_calibration_hold_preview_has_no_gain_torque_or_goal_writes(rig, p):
    supported.main(['--config', str(rig.config), '--calibration-hold', '--ram-p', str(p)])
    assert rig.opened
    assert rig.bus.writes == []


@pytest.mark.parametrize('p', [5, 8])
def test_explicit_gain_option_requires_dedicated_calibration_mode(rig, p):
    with pytest.raises(SystemExit):
        supported.main(['--config', str(rig.config), '--ram-p', str(p), '--soft-gains'])
    assert rig.opened == [] and rig.bus.writes == []


@pytest.mark.parametrize('p', [5, 8])
def test_home_bus_gain_gate_only_admits_requested_volatile_gain(monkeypatch, p):
    writes = []
    monkeypatch.setattr(ServoBus, '_sync_ram',
                        lambda self, address, size, values: writes.append((address, size, values)))
    bus = object.__new__(supported.PositionOnlyBus)
    bus.soft_gain_p = p
    with pytest.raises(ValueError):
        bus._sync_ram(50, 3, {1: bytes((p, 0, 0))})
    bus.soft_gain_write_allowed = True
    bus._sync_ram(50, 3, {1: bytes((p, 0, 0))})
    for gains in ((8 if p == 5 else 5, 0, 0), (16, 0, 0), (p, 1, 0), (p, 0, 1)):
        with pytest.raises(ValueError):
            bus._sync_ram(50, 3, {1: bytes(gains)})
    with pytest.raises(ValueError):
        bus._sync_ram(21, 3, {1: bytes((p, 0, 0))})
    assert writes == [(50, 3, {1: bytes((p, 0, 0))})]


def test_final_goal_mismatch_does_not_report_successful_handoff(rig, monkeypatch):
    read_configuration = rig.bus.read_configuration

    def changed_final_goal(sid):
        value = read_configuration(sid)
        if rig.bus.goals == rig.bus.home:
            value['motion_profile']['goal_position_ticks'] += 1
        return value

    monkeypatch.setattr(rig.bus, 'read_configuration', changed_final_goal)
    with pytest.raises(ValueError, match='final goal readback'):
        rig.run()
    assert rig.report()['result'] == 'aborted'
    assert rig.report().get('target_reached') is not True
    assert all(value['torque'] == 0 for value in rig.bus.configurations.values())


def test_shared_lock_refusal_does_not_open_or_write_serial(rig, monkeypatch):
    @contextmanager
    def already_owned(*args):
        raise RuntimeError('Another process owns the shared hardware lock')
        yield

    monkeypatch.setattr(supported, 'exclusive_hardware', already_owned)
    with pytest.raises(RuntimeError, match='shared hardware lock'):
        rig.run()
    assert rig.opened == []
    assert rig.bus.writes == []


@pytest.mark.parametrize('field,value', [('current_ma', -1235.), ('temperature_c', 56),
                                        ('voltage_v', 4.5)])
def test_straight_log_only_handoff_records_legacy_threshold_values_without_aborting(rig, field, value):
    # The value persists through initial checks, gain setup, preload readback,
    # every trajectory frame and settling, catching any accidentally strict call.
    rig.bus.initial_feedback_change = {field: value}
    rig.run('--log-only-feedback')
    report = rig.report()
    assert report['log_only_feedback_requested'] is True
    assert report['log_only_feedback'] is True
    assert report['software_current_stop_ma'] is None
    assert report['target_reached'] is True
    assert rig.bus.goals == rig.bus.home
    assert all(item['torque'] == 1 for item in rig.bus.configurations.values())
    assert all(sample[field][0] == value for sample in report['samples'])
    assert len(rig.locks) == 1


def test_log_only_reports_real_final_error_even_when_tracking_and_path_limits_are_exceeded(rig, monkeypatch):
    read_feedback = rig.bus.read_feedback_many

    def lagged_feedback(ids, *, sync):
        values = read_feedback(ids, sync=sync)
        if all(item['torque'] for item in rig.bus.configurations.values()):
            values = {sid: replace(value, position_ticks=value.position_ticks + 70)
                      for sid, value in values.items()}
        return values

    monkeypatch.setattr(rig.bus, 'read_feedback_many', lagged_feedback)
    rig.run('--log-only-feedback')
    report = rig.report()
    assert report['target_reached'] is False
    assert report['final_errors_ticks'] == [70] * 14
    assert report['result'] == 'home interpolation complete; torque remains ON'
    assert rig.bus.goals == rig.bus.home
    assert all(item['torque'] == 1 for item in rig.bus.configurations.values())


@pytest.mark.parametrize('field,value', [('status', 4), ('received_at', 99.)])
def test_log_only_still_rejects_initial_device_faults_and_stale_feedback_without_writes(rig, field, value):
    rig.bus.initial_feedback_change = {field: value}
    with pytest.raises(ValueError, match='stale/error feedback'):
        rig.run('--log-only-feedback')
    assert rig.bus.writes == []


@pytest.mark.parametrize('fault', ['status', 'stale', 'slow_cycle'])
def test_log_only_keeps_runtime_status_freshness_timing_and_torque_cleanup(rig, monkeypatch, fault):
    read_feedback = rig.bus.read_feedback_many

    def faulty_feedback(ids, *, sync):
        if all(item['torque'] for item in rig.bus.configurations.values()):
            if fault == 'status':
                rig.bus.initial_feedback_change = {'status': 4}
            elif fault == 'stale':
                rig.bus.initial_feedback_change = {'received_at': rig.clock.now - .2}
            else:
                rig.clock.advance(.21)
        return read_feedback(ids, sync=sync)

    monkeypatch.setattr(rig.bus, 'read_feedback_many', faulty_feedback)
    with pytest.raises(ValueError, match='stale/error feedback|200ms'):
        rig.run('--log-only-feedback')
    report = rig.report()
    assert report['result'] == 'aborted'
    assert report['failure_cleanup'] == 'torque OFF verified'
    assert report['failure_gain_cleanup'] == 'original RAM gains restored'
    assert all(item['torque'] == 0 for item in rig.bus.configurations.values())
    assert len([event for event in rig.bus.writes if event[0] == 'positions']) == 1


@pytest.mark.parametrize('high_current', [False, True])
def test_assembly_calibration_keeps_strict_effective_mode_even_when_log_only_requested(rig, monkeypatch, high_current):
    from microduck_deploy.assembly import record_assembly
    cfg = deepcopy(rig.cfg)
    profile = {
        'schema': 1, 'type': 'microduck_assembly_geometry',
        'geometry_source': 'Synthetic test fixture, not robot geometry',
        'hole_pitch_deg': 45,
        'joints': [dict(name=joint['name'], id=joint['id'], direction=joint['direction'],
                       center_tick=2048, base_q_deg=float(np.rad2deg(HOME[index])),
                       hole_steps=0, datum='Synthetic test datum', verified=True)
                   for index, joint in enumerate(cfg['joints'])],
    }
    record_assembly(cfg, profile)
    rig.bus.__dict__.update(MemoryBus(cfg, rig.clock).__dict__)
    monkeypatch.setattr(supported, 'load_config', lambda path: deepcopy(cfg))
    if high_current:
        rig.bus.initial_feedback_change = {'current_ma': 900.}
        with pytest.raises(ValueError, match='voltage/temperature/current'):
            rig.run('--log-only-feedback')
        assert rig.bus.writes == []
    else:
        rig.run('--log-only-feedback')
        report = rig.report()
        assert report['log_only_feedback_requested'] is True
        assert report['log_only_feedback'] is False
        assert report['software_current_stop_ma'] == 800
        assert report['target_reached'] is True
