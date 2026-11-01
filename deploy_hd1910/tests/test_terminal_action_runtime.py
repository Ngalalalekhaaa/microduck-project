"""Real terminal lines select action sessions through a hardware-free Controller."""

from contextlib import contextmanager
import dataclasses
import json
import math
import os
import pty
import select
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy import runtime
from microduck_deploy.terminal_actions import TerminalActionCommand
from test_runtime import Policy, calibrated_config, rig


@contextmanager
def terminal(clock):
    master, slave = pty.openpty()
    source = TerminalActionCommand(
        SimpleNamespace(isatty=lambda: True, fileno=lambda: slave), clock=clock)

    def send(line):
        os.write(master, line.encode())
        assert select.select([slave], [], [], .5)[0]

    try:
        yield source, send
    finally:
        source.close()
        os.close(slave)
        os.close(master)


def attach_policies(r):
    sit, ground, roll = [Policy(r.clock) for _ in range(3)]
    r.ctl.sitstand_policy = sit
    r.ctl.ground_pick_policy = ground
    r.ctl.roulade_policy = roll
    for policy, output in zip((r.policy, sit, ground, roll), (.01, .02, .03, .04)):
        policy.output[:] = output
    return sit, ground, roll


def on_infer(policy, callback):
    original = policy.infer

    def infer(observation):
        value = original(observation)
        callback(observation, len(policy.observations))
        return value

    policy.infer = infer


def samples(r):
    return [row for line in r.path.read_text().splitlines()
            if (row := json.loads(line))['kind'] == 'sample']


def assert_unloaded(r):
    assert not r.ctl.enabled
    assert any(event[0] == 'torque' and event[2] is False for event in r.bus.events)


def test_terminal_sit_then_stand_routes_flags_and_preserves_action_history():
    with rig() as r, terminal(r.clock.monotonic) as (source, send):
        sit, ground, roll = attach_policies(r)
        send('7\n')  # Old HOME-stage input must not pre-arm sitting.
        on_infer(r.policy, lambda obs, count: send('7\n') if count == 1 else None)
        on_infer(sit, lambda obs, count: send('8\n') if count == 1 else (
            send('q\n') if count == 3 else None))
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=8, motion=True, command_provider=source)
        np.testing.assert_array_equal(np.asarray(sit.observations)[:, 48:51], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(sit.observations[0][34:48], .01)
        np.testing.assert_allclose(sit.observations[1][34:48], .02)
        assert not ground.observations and not roll.observations
        rows = samples(r)
        assert rows[0]['policy_mode'] == 'walking'
        assert [row['policy_mode'] for row in rows[-2:]] == ['sit', 'stand']
        assert len([row for row in rows if row['policy_mode'] == 'stopping']) >= 25
        assert_unloaded(r)


def test_terminal_ground_action_returns_to_walking_without_using_stand_model():
    with rig() as r, terminal(r.clock.monotonic) as (source, send):
        sit, ground, roll = attach_policies(r)
        recovered = []
        def walking(obs, count):
            if count == 1:
                send('9\n')
            elif ground.observations:
                recovered.append(obs.copy())
                if len(recovered) == 2:
                    send('q\n')

        on_infer(r.policy, walking)
        on_infer(ground, lambda obs, count: send('1\nr\n7\n9\n') if count == 1 else None)
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=8, motion=True, command_provider=source)
        commands = np.asarray(ground.observations)[:, 48:51]
        assert len(commands) >= 190
        np.testing.assert_allclose(commands[0], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(commands[:, :2], axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(commands[:, 2], 0)
        phase = np.unwrap(np.arctan2(commands[:, 1], commands[:, 0]))
        assert np.all(np.diff(phase) > 0)
        assert phase[-1] > 2 * np.pi - .05
        np.testing.assert_array_equal(r.policy.observations[-1][48:51], [0, 0, 0])
        np.testing.assert_allclose(recovered[0][34:48], .03)
        assert not sit.observations and not roll.observations
        assert samples(r)[-1]['policy_mode'] == 'walking'
        assert source.policy_mode == 'walking'
        assert_unloaded(r)


def test_terminal_roll_routes_zero_command_and_returns_to_stand_after_measured_rotation():
    cfg = calibrated_config()
    cfg['control']['roulade_enabled'] = True
    with rig(cfg=cfg) as r, terminal(r.clock.monotonic) as (source, send):
        sit, ground, roll = attach_policies(r)
        began = []
        on_infer(r.policy, lambda obs, count: send('r\n') if count == 1 else None)

        def rolling(observation, count):
            if count == 1:
                began.append(r.clock.now)
                send('1\n9\nr\n')  # These may not queue another behavior.

        on_infer(roll, rolling)
        on_infer(sit, lambda obs, count: send('q\n') if count == 2 else None)
        original_latest = r.imu.latest

        def measured_imu(max_age):
            original = original_latest(max_age)
            age = 0 if not began else r.clock.now - began[0]
            angle = age * math.pi if 0 < age < 2 else 0
            gyro = [0, math.pi, 0] if 0 < age < 2 else [0, 0, 0]
            return dataclasses.replace(original,
                projected_gravity=np.array([math.sin(angle), 0, -math.cos(angle)]),
                base_ang_vel=np.array(gyro))

        r.imu.latest = measured_imu
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=8, motion=True, command_provider=source)
        assert len(roll.observations) > 100
        np.testing.assert_array_equal(np.asarray(roll.observations)[:, 48:], 0)
        np.testing.assert_allclose(roll.observations[0][34:48], .01)
        assert len(sit.observations) == 2 and not ground.observations
        np.testing.assert_array_equal(sit.observations[0][48:51], [0, 0, 0])
        np.testing.assert_allclose(sit.observations[0][34:48], .04)
        last = samples(r)[-1]
        assert last['policy_mode'] == 'stand'
        assert last['roulade_progress']['saw_inversion']
        assert last['roulade_progress']['rotation_deg'] >= 330
        assert source.policy_mode == 'stand'
        assert_unloaded(r)


@pytest.mark.parametrize('key, action_index', [('7', 0), ('9', 1), ('r', 2)])
def test_q_interrupts_each_keyboard_action_and_unloads(key, action_index):
    cfg = calibrated_config()
    cfg['control']['roulade_enabled'] = True
    with rig(cfg=cfg) as r, terminal(r.clock.monotonic) as (source, send):
        policies = attach_policies(r)
        on_infer(r.policy, lambda obs, count: send(key + '\n') if count == 1 else None)
        on_infer(policies[action_index], lambda obs, count: send('q\n'))
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=8, motion=True, command_provider=source)
        assert len(policies[action_index].observations) == 1
        assert all(not policy.observations for index, policy in enumerate(policies) if index != action_index)
        assert_unloaded(r)
