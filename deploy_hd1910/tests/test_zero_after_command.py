"""A fixed command becomes zero without interrupting the active policy loop."""
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy import runtime
from test_runtime import calibrated_config, rig


def stop_after_policy_time(r, seconds):
    """Request the ordinary controller stop after observing a policy sample."""
    original = r.policy.infer

    def infer(obs):
        # There must be no torque-OFF between the forward and zero-command stages.
        assert not any(event[0] == 'torque' and event[2] is False for event in r.bus.events)
        result = original(obs)
        if r.clock.now - r.ctl.session['policy_start_monotonic_s'] >= seconds:
            r.ctl.reason = 'test operator stop'
            r.ctl.stop.set()
        return result

    r.policy.infer = infer


def test_zero_after_uses_policy_time_preserves_history_and_runs_until_stopped():
    cfg = calibrated_config()
    cfg['control']['home_duration_s'] = 2.0
    with rig(cfg=cfg) as r:
        initial_time = r.clock.now
        r.policy.output[:] = .01
        stop_after_policy_time(r, 5.2)
        console = io.StringIO()
        with patch.object(r.ctl, '_home', wraps=r.ctl._home) as home, redirect_stdout(console):
            with pytest.raises(runtime.StopRequested, match='test operator stop'):
                r.ctl.execute(seconds=0, twist=(.3, .1, -.2), motion=True,
                              zero_command_after_s=5)
        home.assert_called_once()
        started = r.ctl.session['policy_start_monotonic_s']
        assert started - initial_time >= 2.0
        times = np.asarray(r.policy.times) - started
        observations = np.asarray(r.policy.observations)
        before = times < 5
        after = ~before
        assert before.any() and after.sum() >= 5
        np.testing.assert_allclose(observations[before, 48:51],
                                   np.tile([.3, .1, -.2], (before.sum(), 1)))
        np.testing.assert_array_equal(observations[after, 48:51], 0)
        # Timing the command must neither rerun HOME nor reset policy history.
        np.testing.assert_array_equal(observations[0, 34:48], 0)
        np.testing.assert_allclose(observations[1:, 34:48], .01)
        event = r.ctl.session['zero_command_transition']
        assert event['t'] == pytest.approx(times[after][0])
        assert 5 <= event['t'] < 5.04
        np.testing.assert_allclose(event['from'], [.3, .1, -.2])
        np.testing.assert_array_equal(event['to'], 0)
        assert console.getvalue().count('速度指令已归零，模型继续运行') == 1
        assert r.ctl.session['zero_command_after_s'] == 5
        assert r.ctl.session['unlimited_duration'] is True
        assert r.ctl.session['status'] == 'stopped'
        assert r.ctl.enabled is False
        off_events = [event for event in r.bus.events if event[0] == 'torque' and not event[2]]
        assert len(off_events) == 1
        assert off_events[0][1] - started >= 5.2
        saved = json.loads(r.path.with_suffix('.session.json').read_text())
        assert saved['zero_command_transition']['t'] == pytest.approx(event['t'])
        samples = [json.loads(line) for line in r.path.read_text().splitlines()
                   if json.loads(line)['kind'] == 'sample']
        zero_samples = [sample for sample in samples if sample['t'] >= 5]
        assert zero_samples
        assert all(sample['requested_command'] == [0, 0, 0] for sample in zero_samples)
        assert all(sample['command'] == [0, 0, 0] for sample in zero_samples)


def test_initial_ramp_finishes_before_continuous_zero_command():
    with rig() as r:
        stop_after_policy_time(r, .5)
        with pytest.raises(runtime.StopRequested, match='test operator stop'):
            r.ctl.execute(seconds=0, twist=(.3, 0, 0), motion=True,
                          command_ramp_s=.2, zero_command_after_s=.3)
        times = np.asarray(r.policy.times) - r.ctl.session['policy_start_monotonic_s']
        commands = np.asarray(r.policy.observations)[:, 48]
        assert 0 < commands[0] < .1
        assert np.all(np.diff(commands[times < .3]) >= 0)
        np.testing.assert_allclose(commands[(times >= .2) & (times < .3)], .3)
        np.testing.assert_array_equal(commands[times >= .3], 0)


def test_imu_failure_during_zero_command_still_unloads():
    with rig() as r:
        r.imu.fail_when = lambda: (
            r.ctl.session.get('zero_command_transition') is not None
            and r.clock.now - r.ctl.session['policy_start_monotonic_s'] >= .2)
        with pytest.raises(runtime.StopRequested, match='mock IMU stale'):
            r.ctl.execute(seconds=0, twist=(.3, 0, 0), motion=True,
                          zero_command_after_s=.1)
        assert r.ctl.session['zero_command_transition'] is not None
        assert r.ctl.enabled is False
        assert any(event[0] == 'torque' and event[2] is False for event in r.bus.events)
        assert r.ctl.session['stop_reason'] == 'mock IMU stale'


def test_default_finite_duration_still_finishes_and_unloads():
    with rig() as r:
        result = r.ctl.execute(seconds=.1, twist=(.3, 0, 0), motion=True)
        assert result['samples'] > 0
        np.testing.assert_allclose(np.asarray(r.policy.observations)[:, 48], .3)
        assert r.ctl.session['zero_command_after_s'] is None
        assert r.ctl.session['zero_command_transition'] is None
        assert r.ctl.session['status'] == 'completed'
        assert r.ctl.session['unlimited_duration'] is False
        assert r.ctl.enabled is False
        assert any(event[0] == 'torque' and event[2] is False for event in r.bus.events)


@pytest.mark.parametrize('override', [
    {'zero_command_after_s': 0},
    {'zero_command_after_s': -1},
    {'zero_command_after_s': float('nan')},
    {'zero_command_after_s': float('inf')},
    {'zero_command_after_s': -float('inf')},
    {'seconds': 5},
    {'home_only': True},
    {'command_provider': lambda: (.3, 0, 0)},
    {'command_ramp_s': .31},
    {'command_ramp_s': float('nan')},
])
def test_invalid_zero_after_options_rejected_before_any_bus_access(override):
    with rig() as r:
        options = dict(seconds=0, motion=True, zero_command_after_s=.3)
        options.update(override)
        with pytest.raises(ValueError):
            r.ctl.execute(**options)
        assert r.bus.events == []
        assert r.policy.observations == []
        assert not r.path.with_suffix('.session.json').exists()
