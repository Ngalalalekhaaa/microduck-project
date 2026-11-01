"""Reuse policy validation without weakening the actual servo write boundary."""
import json
import os

import numpy as np
import pytest

from microduck_deploy import config, runtime
from microduck_deploy.policy import HOME, q_to_ticks
from microduck_deploy.terminal_command import TerminalCommand
from test_runtime import rig


def capture_policy_writes(r):
    writes = []
    original = r.bus.sync_positions

    def send(positions):
        if r.ctl.session.get('status') == 'running':
            writes.append(dict(positions))
        return original(positions)

    r.bus.sync_positions = send
    return writes


def count_running_calls(monkeypatch, r, name):
    original = getattr(runtime, name)
    calls = []

    def call(*args, **kwargs):
        if r.ctl.session.get('status') == 'running':
            calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, name, call)
    return calls


def test_each_policy_step_validates_once_and_sends_identical_raw_target(monkeypatch):
    with rig() as r:
        r.cfg['control']['log_only_target_angles'] = True
        # Outside the trial target bound but still within the feedback bound,
        # so several subsequent steps can run with the same unmodified target.
        r.policy.output[1] = -.35
        target = HOME + r.policy.output
        zeros, directions = config.calibration_arrays(r.cfg)
        expected = dict(zip(r.bus.ids, q_to_ticks(target, zeros, directions).tolist()))
        writes = capture_policy_writes(r)
        validation = count_running_calls(monkeypatch, r, 'validate_targets')
        warnings = count_running_calls(monkeypatch, r, 'target_angle_warnings')
        encodings = count_running_calls(monkeypatch, r, 'q_to_ticks')
        result = r.ctl.execute(seconds=.12, motion=True)
        rows = [row for line in r.path.read_text().splitlines()
                if (row := json.loads(line))['kind'] == 'sample']
        assert result['samples'] == len(writes) == len(rows) > 3
        assert len(validation) == len(warnings) == len(encodings) == len(writes)
        assert all(write == expected for write in writes)
        for row in rows:
            np.testing.assert_allclose(row['target'], target)
            assert 'left_hip_roll' in row['target_check']


@pytest.mark.parametrize('invalid', ['nonfinite', 'encoder', 'strict_angle'])
def test_invalid_policy_target_never_reaches_write_boundary(monkeypatch, invalid):
    with rig() as r:
        r.cfg['control']['log_only_target_angles'] = invalid != 'strict_angle'
        target = HOME.copy()
        target[0] = {'nonfinite': float('nan'), 'encoder': 20., 'strict_angle': 1.}[invalid]
        monkeypatch.setattr(runtime, 'action_to_targets', lambda action: target.copy())
        writes = capture_policy_writes(r)
        with pytest.raises((runtime.StopRequested, ValueError)):
            r.ctl.execute(seconds=.12, motion=True)
        assert len(r.policy.observations) == 1
        assert writes == []
        assert r.ctl.enabled is False
        assert any(event[0] == 'torque' and not event[2] for event in r.bus.events)


def test_q_arriving_after_policy_validation_prevents_servo_write(monkeypatch):
    master, slave = os.openpty()
    try:
        with os.fdopen(slave, 'r', encoding='utf-8') as stream, rig() as r:
            terminal = TerminalCommand(stream, clock=r.clock.monotonic)
            original = runtime.validate_targets
            writes = capture_policy_writes(r)

            def validated_then_q(*args, **kwargs):
                result = original(*args, **kwargs)
                if r.ctl.session.get('status') == 'running':
                    os.write(master, b'q\n')
                return result

            monkeypatch.setattr(runtime, 'validate_targets', validated_then_q)
            try:
                with pytest.raises(runtime.StopRequested, match='终端请求退出'):
                    r.ctl.execute(seconds=.12, motion=True, command_provider=terminal)
            finally:
                terminal.close()
            assert len(r.policy.observations) == 1
            assert writes == []
            assert r.ctl.enabled is False
    finally:
        os.close(master)


def test_validation_exceeding_policy_deadline_prevents_servo_write(monkeypatch):
    with rig() as r:
        original = runtime.validate_targets
        writes = capture_policy_writes(r)

        def slow_validation(*args, **kwargs):
            result = original(*args, **kwargs)
            if r.ctl.session.get('status') == 'running':
                r.clock.advance(.041)
            return result

        monkeypatch.setattr(runtime, 'validate_targets', slow_validation)
        with pytest.raises(runtime.StopRequested, match='目标准备后控制周期已超时'):
            r.ctl.execute(seconds=.12, motion=True)
        assert len(r.policy.observations) == 1
        assert writes == []
        fault = r.ctl.session['policy_cycle_timing_fault']
        assert fault['stage'] == 'before_target_send'
        assert fault['limit_ms'] == 40
        assert fault['cycle_ms'] > 40
        assert fault['policy_target_sent_this_step'] is False
        assert r.ctl.enabled is False


def test_home_trajectory_still_validates_every_send(monkeypatch):
    with rig() as r:
        original = runtime.validate_targets
        home_validations = []

        def validate(*args, **kwargs):
            if r.ctl.session.get('status') == 'initializing':
                home_validations.append((args[0].copy(), kwargs.copy()))
            return original(*args, **kwargs)

        monkeypatch.setattr(runtime, 'validate_targets', validate)
        r.ctl.execute(seconds=.12, motion=True, home_only=True)
        assert len(home_validations) == len(r.ctl.session['home_samples']) > 1
        assert all(not kwargs.get('return_warnings') for _, kwargs in home_validations)
        assert r.policy.observations == []
        zeros, directions = config.calibration_arrays(r.cfg)
        expected = dict(zip(r.bus.ids, q_to_ticks(HOME, zeros, directions).tolist()))
        assert [event[2] for event in r.bus.events if event[0] == 'positions'][-1] == expected


def test_direct_send_still_rejects_invalid_target():
    with rig() as r:
        target = HOME.copy()
        target[0] = 1.
        with pytest.raises(runtime.StopRequested, match='首次试机角度范围'):
            r.ctl._send(target, HOME)
        assert r.bus.events == []


@pytest.mark.parametrize('interruption', ['stop', 'deadline'])
def test_direct_send_rechecks_stop_and_deadline_after_validation(monkeypatch, interruption):
    with rig() as r:
        original = runtime.validate_targets

        def validate(*args, **kwargs):
            result = original(*args, **kwargs)
            if interruption == 'stop':
                r.ctl.stop.set()
            else:
                r.clock.advance(.041)
            return result

        monkeypatch.setattr(runtime, 'validate_targets', validate)
        with pytest.raises(runtime.StopRequested):
            r.ctl._send(HOME, HOME, cycle_started=r.clock.now)
        assert r.bus.events == []
