import dataclasses
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy.gamepad import (GamepadReader, GamepadCommand, GamepadStop,
    PadSample, normalize_axis, axes_to_twist)
from microduck_deploy import runtime
from test_runtime import rig


def test_unsigned_signed_deadzone_and_robot_directions():
    assert normalize_axis(32768, 0, 65535) == 0
    assert normalize_axis(1000, -32768, 32767) == 0
    assert normalize_axis(0, 0, 65535) == -1
    assert normalize_axis(65535, 0, 65535) == 1
    assert axes_to_twist({'ABS_Y': -1, 'ABS_X': -1, 'ABS_RX': -1}) == (.3, .3, 1.)
    assert axes_to_twist({'ABS_Y': 1, 'ABS_X': 1, 'ABS_RX': 1}) == (-.3, -.3, -1.)


@pytest.mark.parametrize('forward_limit', [.1, .3, .4])
def test_backward_limit_keeps_forward_side_and_yaw_settings(forward_limit):
    axes = {'ABS_Y': -1., 'ABS_X': .5, 'ABS_RX': -.5}
    forward = axes_to_twist(axes, max_vx=forward_limit, max_vy=.2, max_wz=.6)
    assert forward == (forward_limit, -.1, .3)
    backward = axes_to_twist(dict(axes, ABS_Y=1.), max_vx=forward_limit, max_vy=.2, max_wz=.6)
    assert backward == (-min(forward_limit, .3), -.1, .3)
    halfway = axes_to_twist(dict(axes, ABS_Y=.5), max_vx=forward_limit, max_vy=.2, max_wz=.6)
    assert halfway == (-.5 * min(forward_limit, .3), -.1, .3)


def test_slewed_backward_command_reaches_point_three_then_forward_is_unchanged():
    now = [10.]
    sample = [PadSample(now[0], True, {'ABS_X': 0., 'ABS_Y': 1., 'ABS_RX': 0.}, {}, ())]
    command = GamepadCommand(SimpleNamespace(latest=lambda: sample[0]), clock=lambda: now[0])
    values = []
    for _ in range(60):
        now[0] += .02
        sample[0] = dataclasses.replace(sample[0], timestamp=now[0])
        values.append(command())
    np.testing.assert_allclose(values[-1], (-.3, 0., 0.))
    assert all(-.3 <= row[0] <= 0 for row in values)
    # The existing .3 m/s per second command slew remains in effect.
    assert all(abs(b[0] - a[0]) <= .006 + 1e-12 for a, b in zip(values, values[1:]))
    for _ in range(100):
        now[0] += .02
        sample[0] = dataclasses.replace(sample[0], timestamp=now[0],
                                      axes={'ABS_X': 0., 'ABS_Y': -1., 'ABS_RX': 0.})
        result = command()
    np.testing.assert_allclose(result, (.3, 0., 0.))


class FakeDevice:
    path = '/dev/input/event-test'
    name = 'Xbox'
    uniq = 'test'
    disconnected = False
    events = []
    def capabilities(self, absinfo=False):
        return {3: [0, 1, 2, 5]}
    def absinfo(self, code):
        if self.disconnected:
            raise OSError('disconnected')
        return SimpleNamespace(value=32768, min=0, max=65535)
    def read(self):
        if not self.events:
            raise BlockingIOError
        events, self.events = self.events, []
        return iter(events)
    def active_keys(self):
        return []
    def close(self):
        self.closed = True


def test_stationary_stick_heartbeat_brief_b_and_disconnect():
    device = FakeDevice()
    reader = GamepadReader(factory=lambda _: device)
    try:
        first = reader.latest().timestamp
        time.sleep(.06)
        assert reader.latest().timestamp > first
        assert reader.latest().axes == {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}
        reader.halt.set()
        reader.thread.join(timeout=.5)
        device.events = [SimpleNamespace(type=1, code=305, value=1),
                         SimpleNamespace(type=1, code=305, value=0)]
        assert reader._poll().stop_requested
        device.disconnected = True
        reader.halt = threading.Event()
        reader.thread = threading.Thread(target=reader._run)
        reader.thread.start()
        time.sleep(.06)
        assert not reader.latest().connected
    finally:
        reader.close()
    assert device.closed


def test_brief_home_and_menu_presses_are_retained_after_release():
    device = FakeDevice()
    reader = GamepadReader(factory=lambda _: device)
    try:
        reader.halt.set()
        reader.thread.join(timeout=.5)
        device.events = [SimpleNamespace(type=1, code=code, value=value)
                         for code in (304, 315) for value in (1, 0)]
        sample = reader._poll()
        assert sample.buttons == ()
        assert sample.button_press_counts == {304: 1, 315: 1}
        assert not sample.stop_requested
        assert reader._poll().button_press_counts == sample.button_press_counts
    finally:
        reader.close()


def test_command_slew_stale_disconnect_and_b():
    now = [10.]
    sample = [PadSample(now[0], True, {'ABS_X': -1, 'ABS_Y': -1, 'ABS_RX': -1}, {}, ())]
    reader = SimpleNamespace(latest=lambda: sample[0])
    command = GamepadCommand(reader, clock=lambda: now[0])
    for _ in range(50):
        now[0] += .02
        sample[0] = dataclasses.replace(sample[0], timestamp=now[0])
        values = command()
    np.testing.assert_allclose(values, (.3, .3, 1.))
    now[0] += .4
    assert command() == (0., 0., 0.)
    sample[0] = dataclasses.replace(sample[0], connected=False, timestamp=now[0])
    assert command() == (0., 0., 0.)
    sample[0] = dataclasses.replace(sample[0], stop_requested=True)
    with pytest.raises(GamepadStop):
        command()


def test_live_commands_reach_policy_observation_and_log():
    import json
    with rig() as r:
        requested = []
        def source():
            value = (.01 * len(requested), -.03, .2)
            requested.append(value)
            return value
        r.policy.output[:] = .01
        r.ctl.execute(seconds=.15, motion=True, command_provider=source)
        np.testing.assert_allclose(np.array(r.policy.observations)[:, 48:51], requested)
        np.testing.assert_allclose(r.policy.observations[1][34:48], .01)
        records = [json.loads(line) for line in r.path.read_text().splitlines()][1:]
        np.testing.assert_allclose([row['command'] for row in records], requested)
        assert r.ctl.session['command_source'] == 'live'


@pytest.mark.parametrize('source', [lambda: [float('nan'), 0, 0],
    lambda: [.401, 0, 0], lambda: [0, -.301, 0], lambda: [0, 0, 1.001], lambda: [0, 0]])
def test_bad_live_input_sends_no_policy_target_and_unloads(source):
    with rig() as r:
        with pytest.raises(runtime.StopRequested):
            r.ctl.execute(seconds=.1, motion=True, command_provider=source)
        assert r.policy.observations == []
        assert not r.ctl.enabled
        assert any(event[0] == 'torque' and event[2] is False for event in r.bus.events)


@pytest.mark.parametrize('command', [(.4, .3, 1.), (-.4, -.3, -1.)])
def test_new_command_extremes_reach_policy_without_range_stop(command):
    with rig() as r:
        r.ctl.execute(seconds=.1, motion=True, command_provider=lambda: command)
        assert r.policy.observations
        np.testing.assert_allclose(
            np.array(r.policy.observations)[:, 48:51],
            np.tile(command, (len(r.policy.observations), 1)))


def test_b_stops_before_policy_target_and_unloads():
    def stop():
        raise GamepadStop('B')
    with rig() as r:
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.1, motion=True, command_provider=stop)
        assert not r.ctl.enabled
        assert r.policy.observations == []


def test_stale_kernel_node_cannot_pass_disconnected_bluetooth_check():
    from microduck_deploy.gamepad import GamepadError
    device = FakeDevice()
    device.uniq = '40:8e:2c:70:2e:46'
    with patch('microduck_deploy.gamepad.bluez_connected', return_value=False):
        with pytest.raises(GamepadError, match='蓝牙连接已断开'):
            GamepadReader(factory=lambda _: device)
    assert device.closed


def test_native_rx_mapping_is_preferred_when_available():
    device = FakeDevice()
    device.capabilities = lambda absinfo=False: {3: [0, 1, 3, 4]}
    reader = GamepadReader(factory=lambda _: device)
    try:
        assert reader.metadata['axis_codes']['ABS_RX'] == 3
    finally:
        reader.close()
