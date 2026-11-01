"""Single-key controls use real PTYs, without touching physical robot devices."""
from contextlib import contextmanager
import os
import pty
import select
import termios
from types import SimpleNamespace

import pytest

from microduck_deploy.gamepad import GamepadStop
from microduck_deploy.terminal_actions import TerminalActionCommand
from microduck_deploy.terminal_command import TerminalCommand
from microduck_deploy import terminal_command
from test_terminal_command import Clock


ZERO = (0., 0., 0.)


@contextmanager
def terminal(cls=TerminalCommand, **options):
    master, slave = pty.openpty()
    stream = os.fdopen(slave, 'rb', buffering=0)
    clock = Clock()
    original = termios.tcgetattr(slave)
    source = cls(stream=stream, clock=clock, single_key=True, **options)

    def send(value):
        os.write(master, value)
        assert select.select([slave], [], [], .5)[0]

    try:
        yield SimpleNamespace(source=source, clock=clock, master=master,
                              slave=slave, stream=stream, original=original, send=send)
    finally:
        source.close()
        stream.close()
        try:
            os.close(master)
        except OSError:
            pass


def ready(t):
    assert t.source.start_policy() is None
    active = termios.tcgetattr(t.slave)
    assert not active[3] & (termios.ICANON | termios.ECHO | termios.ECHONL)
    assert active[3] & termios.ISIG
    assert active[6][termios.VINTR] == t.original[6][termios.VINTR]
    assert active[6][termios.VMIN] in (0, b'\0')
    assert active[6][termios.VTIME] in (0, b'\0')
    assert t.source() == ZERO


def tick(t, value=None, seconds=.02):
    if value is not None:
        t.send(value)
    t.clock.advance(seconds)
    return t.source()


@pytest.mark.parametrize('cls', [TerminalCommand, TerminalActionCommand])
@pytest.mark.parametrize('key,speed', [(b'1', .30), (b'j', .15), (b'k', .20), (b'l', .25)])
def test_forward_speed_presets_without_enter_preserve_slew_and_zero(cls, key, speed):
    with terminal(cls=cls) as t:
        ready(t)
        values = [tick(t, key.upper())]
        values.extend(tick(t) for _ in range(60))
        assert values[-1] == pytest.approx((speed, 0., 0.))
        assert all(0 <= value[0] <= speed + 1e-12 for value in values)
        assert max(b[0] - a[0] for a, b in zip([(0., 0., 0.), *values], values)) <= .006 + 1e-12
        assert tick(t, b'0') == ZERO


@pytest.mark.parametrize('key', [b'1', b'j', b'k', b'l'])
def test_forward_speed_presets_respect_explicit_lower_limit(key):
    with terminal(cls=TerminalActionCommand, max_vx=.1) as t:
        ready(t)
        tick(t, key)
        for _ in range(30):
            tick(t)
        assert t.source() == pytest.approx((.1, 0., 0.))


@pytest.mark.parametrize('key, expected', [
    (b'1', (.006, 0., 0.)), (b'2', (-.006, 0., 0.)),
    (b'3', (0., .006, 0.)), (b'4', (0., -.006, 0.)),
    (b'5', (0., 0., .02)), (b'6', (0., 0., -.02)),
])
def test_direction_and_zero_are_immediate_without_enter(key, expected):
    with terminal() as t:
        assert termios.tcgetattr(t.slave) == t.original
        ready(t)
        assert tick(t, key) == pytest.approx(expected)
        assert tick(t, b'0') == ZERO
        assert t.source.check_stop() is None


@pytest.mark.parametrize('key', [b'q', b'Q', b'\x04'])
def test_quit_and_ctrl_d_without_enter_restore_complete_terminal_settings(key):
    with terminal() as t:
        ready(t)
        tick(t, b'1')
        t.send(key)
        assert t.source.check_stop()
        assert t.source() == ZERO
        assert termios.tcgetattr(t.slave) == t.original
        t.source.close()
        t.source.close()
        assert termios.tcgetattr(t.slave) == t.original


def test_first_sample_is_zero_even_for_fresh_immediate_direction():
    with terminal() as t:
        assert t.source.start_policy() is None
        assert tick(t, b'1') == ZERO
        assert tick(t) == pytest.approx((.006, 0., 0.))


def test_old_complete_and_partial_input_is_flushed_before_noncanonical_activation():
    with terminal(TerminalActionCommand) as t:
        os.write(t.master, b'1\n7\n9r')
        ready(t)
        assert tick(t, b'\n', seconds=10.) == ZERO
        assert t.source.policy_mode == 'walking'
        assert tick(t, b'7') == ZERO
        tick(t, seconds=.61)
        assert t.source.policy_mode == 'sit'


def test_existing_canonical_q_before_activation_stops_without_changing_terminal():
    with terminal() as t:
        os.write(t.master, b'q\n')
        assert '退出' in t.source.start_policy()
        assert termios.tcgetattr(t.slave) == t.original


def test_enter_and_whitespace_never_cancel_or_create_motion():
    with terminal() as t:
        ready(t)
        tick(t, b'1')
        assert tick(t, b' \t\r\n')[0] == pytest.approx(.012)
        tick(t, b'0')
        assert tick(t, b'\r\n ') == ZERO


def test_sit_stand_and_ground_action_accept_single_keys_and_keep_busy_input_discarded():
    with terminal(TerminalActionCommand) as t:
        ready(t)
        tick(t, b'7')
        assert t.source.policy_mode == 'stopping'
        tick(t, seconds=.61)
        assert t.source.policy_mode == 'sit'
        tick(t, b'7')
        assert t.source.policy_mode == 'stand'
        tick(t, b'19r')
        tick(t, seconds=3.1)
        assert t.source.policy_mode == 'stand'
        tick(t, b'9')
        tick(t, seconds=.61)
        assert t.source.policy_mode == 'ground_pick'
        tick(t, b'1789r')
        tick(t, seconds=4.1)
        assert t.source.policy_mode == 'walking'
        assert tick(t, seconds=3.1) == ZERO
        assert t.source.policy_mode == 'walking'


@pytest.mark.parametrize('key', [b'r', b'R'])
def test_roll_key_and_runtime_recovery_callback_work_without_enter(key):
    with terminal(TerminalActionCommand) as t:
        ready(t)
        tick(t, key)
        tick(t, seconds=.61)
        assert t.source.policy_mode == 'roulade'
        t.source.finish_roulade()
        assert t.source.policy_mode == 'stand'
        t.send(b'q')
        assert t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original


def test_roll_timeout_restores_terminal_before_propagating_exception():
    with terminal(TerminalActionCommand, roulade_timeout_s=1.) as t:
        ready(t)
        tick(t, b'r')
        tick(t, seconds=.61)
        with pytest.raises(GamepadStop, match='超时'):
            tick(t, seconds=1.1)
        assert termios.tcgetattr(t.slave) == t.original


@pytest.mark.parametrize('sequence', [
    b'\x1b[A', b'\x1b[1;5C', b'\x1b[15~', b'\x1b[31r',
    b'\x1bOP', b'\x1bO1;2r', b'\x1bOQ', b'\x1b(0',
    b'\x1br', b'\x1b9', b'\x9b1;2r', b'\x8fQ',
    b'\x1b]0;19rQ\x07', b'\x1bP19rQ\x1b\\',
])
def test_escape_sequences_never_dispatch_embedded_digits_roll_or_quit(sequence):
    with terminal(TerminalActionCommand) as t:
        ready(t)
        for part in sequence:
            assert tick(t, bytes((part,))) == ZERO
            assert t.source.check_stop() is None
            assert t.source.policy_mode == 'walking'
        assert tick(t, b'1') == pytest.approx((.006, 0., 0.))


@pytest.mark.parametrize('begin, end', [(b'\x1b[200~', b'\x1b[201~'), (b'\x9b200~', b'\x9b201~')])
def test_bracketed_paste_payload_is_ignored_even_when_fragmented(begin, end):
    with terminal(TerminalActionCommand) as t:
        ready(t)
        for part in (begin[:2], begin[2:], b'1\n79rQ', b'\x1b[A', end[:3], end[3:]):
            assert tick(t, part) == ZERO
            assert t.source.check_stop() is None
            assert t.source.policy_mode == 'walking'
        assert tick(t, b'1') == pytest.approx((.006, 0., 0.))


def test_overlong_escape_sequence_stops_and_restores_instead_of_leaking_digits():
    with terminal() as t:
        ready(t)
        t.send(b'\x1b[' + b'1' * 65)
        assert '过长' in t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original


def test_single_key_flood_is_bounded_and_restores_terminal():
    with terminal() as t:
        ready(t)
        t.send(b'1' * 40)
        assert '过多' in t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original


def test_close_before_and_after_activation_is_idempotent_and_keeps_stdin_open():
    with terminal() as t:
        ready(t)
        t.source.close()
        t.source.close()
        assert termios.tcgetattr(t.slave) == t.original
        assert not t.stream.closed
    with terminal() as t:
        t.source.close()
        assert termios.tcgetattr(t.slave) == t.original


def test_read_failure_restores_terminal_immediately(monkeypatch):
    with terminal() as t:
        ready(t)
        t.send(b'1')
        def failed_read(*args):
            raise OSError('simulated read failure')
        monkeypatch.setattr(terminal_command.os, 'read', failed_read)
        assert '读取失败' in t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original


@pytest.mark.parametrize('error', [RuntimeError('unexpected'), KeyboardInterrupt()])
def test_unexpected_read_exception_or_ctrl_c_restores_terminal(monkeypatch, error):
    with terminal() as t:
        ready(t)
        def fail(*args):
            raise error
        monkeypatch.setattr(terminal_command.select, 'select', fail)
        with pytest.raises(type(error)):
            t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original


def test_activation_failure_rolls_back_partial_termios_change(monkeypatch):
    with terminal() as t:
        original_set = termios.tcsetattr
        calls = []
        def apply_then_fail(fd, when, attrs):
            calls.append(attrs)
            original_set(fd, when, attrs)
            if len(calls) == 1:
                raise termios.error('simulated activation failure')
        monkeypatch.setattr(terminal_command.termios, 'tcsetattr', apply_then_fail)
        assert '初始化失败' in t.source.start_policy()
        assert len(calls) == 2
        assert termios.tcgetattr(t.slave) == t.original


def test_failed_restore_can_retry_on_close_without_masking_stop_reason(monkeypatch):
    with terminal() as t:
        ready(t)
        original_set = termios.tcsetattr
        failed = []
        def fail_once(fd, when, attrs):
            if not failed:
                failed.append(True)
                raise termios.error('temporary restore failure')
            return original_set(fd, when, attrs)
        monkeypatch.setattr(terminal_command.termios, 'tcsetattr', fail_once)
        t.send(b'q')
        assert '退出' in t.source.check_stop()
        t.source.close()
        assert termios.tcgetattr(t.slave) == t.original


def test_closed_terminal_requests_stop_and_cleanup_does_not_raise():
    with terminal() as t:
        ready(t)
        os.close(t.master)
        assert t.source.check_stop()
        t.source.close()
        assert t.source() == ZERO


def test_exit_flushes_keys_waiting_behind_quit_before_returning_to_shell(monkeypatch):
    with terminal() as t:
        ready(t)
        original_read = os.read
        monkeypatch.setattr(terminal_command.os, 'read',
                            lambda fd, count: original_read(fd, 1))
        t.send(b'q1\n')
        assert '退出' in t.source.check_stop()
        assert termios.tcgetattr(t.slave) == t.original
        assert not select.select([t.slave], [], [], 0)[0]
        os.write(t.master, b'\n')
        assert select.select([t.slave], [], [], .5)[0]
        assert original_read(t.slave, 64) == b'\n'


def test_action_initialization_exception_after_termios_change_restores_it(monkeypatch):
    with terminal(TerminalActionCommand) as t:
        def fail():
            raise RuntimeError('reset failed')
        monkeypatch.setattr(t.source, '_reset_actions', fail)
        with pytest.raises(RuntimeError, match='reset failed'):
            t.source.start_policy()
        assert termios.tcgetattr(t.slave) == t.original


def test_mode_specific_hints_and_boolean_validation():
    with terminal(TerminalActionCommand) as t:
        assert '无需回车' in t.source.ready_message
    with pytest.raises(ValueError, match='single_key'):
        TerminalCommand(single_key='yes')
