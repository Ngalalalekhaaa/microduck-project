"""Immediate terminal keys and termios lifetime through the real Controller."""

from contextlib import contextmanager
import os
import pty
import select
import signal
import termios
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy import runtime
from microduck_deploy.terminal_actions import TerminalActionCommand
from test_runtime import calibrated_config, rig
from test_terminal_action_runtime import attach_policies, assert_unloaded, on_infer


@contextmanager
def immediate_terminal(clock):
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    source = TerminalActionCommand(
        SimpleNamespace(isatty=lambda: True, fileno=lambda: slave),
        clock=clock, single_key=True)

    def send(keys):
        os.write(master, keys.encode())
        assert select.select([slave], [], [], .5)[0]

    try:
        yield SimpleNamespace(source=source, send=send, master=master,
                              slave=slave, original=original)
    finally:
        source.close()
        os.close(slave)
        os.close(master)


def assert_immediate(t):
    flags = termios.tcgetattr(t.slave)[3]
    assert not flags & termios.ICANON
    assert flags & termios.ISIG  # Ctrl+C must retain its signal semantics.


def assert_restored(t):
    assert termios.tcgetattr(t.slave) == t.original


def test_single_key_direction_stop_and_quit_need_no_enter_and_restore_terminal():
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        assert_restored(t)  # Preparation remains canonical.

        def drive(observation, count):
            assert_immediate(t)
            if count == 1:
                t.send('1')
            elif count == 4:
                t.send('0')
            elif count == 7:
                t.send('q')

        on_infer(r.policy, drive)
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=1, motion=True, command_provider=t.source)
        observations = np.asarray(r.policy.observations)
        np.testing.assert_array_equal(observations[0, 48:51], 0)
        assert np.all(observations[1:4, 48] > 0)
        np.testing.assert_array_equal(observations[4:, 48:51], 0)
        assert len(observations) == 7
        assert_restored(t)
        assert_unloaded(r)


def test_forward_presets_reach_policy_observations_without_enter():
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        def drive(observation, count):
            keys = {1: 'j', 35: 'k', 55: 'l', 75: '1', 100: 'q'}
            if count in keys:
                t.send(keys[count])

        on_infer(r.policy, drive)
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=3, motion=True, command_provider=t.source)
        obs = np.array(r.policy.observations)
        for end, expected in ((35, .15), (55, .20), (75, .25), (100, .30)):
            np.testing.assert_allclose(obs[end - 5:end, 48], expected)
        np.testing.assert_array_equal(obs[:, 49:51], 0)
        assert_restored(t)
        assert_unloaded(r)


@pytest.mark.parametrize('key, index, expected_mode, expected_command', [
    ('7', 0, 'sit', [1, 0, 0]),
    ('8', 0, 'stand', [0, 0, 0]),
    ('9', 1, 'ground_pick', [1, 0, 0]),
    ('r', 2, 'roulade', [0, 0, 0]),
])
def test_single_action_keys_select_matching_model_and_commands_without_enter(
        key, index, expected_mode, expected_command):
    cfg = calibrated_config()
    cfg['control']['roulade_enabled'] = True
    with rig(cfg=cfg) as r, immediate_terminal(r.clock.monotonic) as t:
        policies = attach_policies(r)
        on_infer(r.policy, lambda obs, count: t.send(key) if count == 1 else None)

        def action(observation, count):
            assert_immediate(t)
            assert t.source.policy_mode == expected_mode
            np.testing.assert_allclose(observation[48:51], expected_command, atol=1e-7)
            t.send('q')

        on_infer(policies[index], action)
        with pytest.raises(runtime.StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=2, motion=True, command_provider=t.source)
        assert len(policies[index].observations) == 1
        assert all(not policy.observations for i, policy in enumerate(policies) if i != index)
        assert_restored(t)
        assert_unloaded(r)


def test_successful_duration_exit_restores_terminal_before_caller_cleanup():
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        on_infer(r.policy, lambda obs, count: assert_immediate(t))
        result = r.ctl.execute(seconds=.12, motion=True, command_provider=t.source)
        assert result['samples'] > 0
        assert_restored(t)
        assert_unloaded(r)


@pytest.mark.parametrize('failure', [RuntimeError('model failed'), KeyboardInterrupt()])
def test_policy_failure_or_keyboard_interrupt_restores_terminal(failure):
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        def fail(observation, count):
            assert_immediate(t)
            raise failure

        on_infer(r.policy, fail)
        with pytest.raises(type(failure)):
            r.ctl.execute(seconds=1, motion=True, command_provider=t.source)
        assert_restored(t)
        assert_unloaded(r)


def test_real_sigint_restores_terminal_and_previous_handler():
    original_handler = signal.getsignal(signal.SIGINT)
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        # The general mock rig disables handlers; enable the production installer
        # here so this process-local signal follows the same path as Ctrl+C.
        r.ctl._install_signals = runtime.Controller._install_signals.__get__(r.ctl)

        def interrupt(observation, count):
            assert_immediate(t)
            os.kill(os.getpid(), signal.SIGINT)

        on_infer(r.policy, interrupt)
        with pytest.raises(runtime.StopRequested, match='收到停止信号'):
            r.ctl.execute(seconds=1, motion=True, command_provider=t.source)
        assert signal.getsignal(signal.SIGINT) == original_handler
        assert_restored(t)
        assert_unloaded(r)


def test_eof_restores_terminal_and_unloads():
    with rig() as r, immediate_terminal(r.clock.monotonic) as t:
        original_read = os.read
        eof = []

        def disconnect(observation, count):
            eof.append(True)
            t.send('1')  # Wake the PTY; model the subsequent read returning EOF.

        def read(fd, size):
            return b'' if eof and fd == t.slave else original_read(fd, size)

        on_infer(r.policy, disconnect)
        # Keeping the PTY open lets us inspect its restored termios after EOF.
        with patch('microduck_deploy.terminal_command.os.read', side_effect=read):
            with pytest.raises(runtime.StopRequested, match='EOF'):
                r.ctl.execute(seconds=1, motion=True, command_provider=t.source)
        assert_restored(t)
        assert_unloaded(r)


def test_terminal_is_restored_before_imu_and_bus_cleanup_failures():
    with rig(owned=True) as r, immediate_terminal(r.clock.monotonic) as t:
        cleanup_calls = []

        def failing_close(name):
            def close():
                assert_restored(t)
                cleanup_calls.append(name)
                raise OSError(name + ' close failed')
            return close

        r.imu.close = failing_close('imu')
        r.bus.close = failing_close('bus')
        on_infer(r.policy, lambda obs, count: assert_immediate(t))
        r.ctl.execute(seconds=.12, motion=True, command_provider=t.source)
        assert cleanup_calls == ['imu', 'bus']
        assert_restored(t)
        assert_unloaded(r)
