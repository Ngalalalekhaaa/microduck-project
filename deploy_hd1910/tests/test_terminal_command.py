"""Terminal-only tests use local pseudoterminals, never robot hardware."""
from contextlib import contextmanager
import io
import os
import pty
import termios
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.terminal_command import TerminalCommand
from microduck_deploy import terminal_command


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds=.02):
        self.now += seconds


@contextmanager
def terminal(**options):
    master, slave = pty.openpty()
    stream = os.fdopen(slave, 'rb', buffering=0)
    clock = Clock()
    command = TerminalCommand(stream=stream, clock=clock, **options)
    try:
        yield SimpleNamespace(command=command, clock=clock, master=master,
                              stream=stream, write=lambda data: os.write(master, data))
    finally:
        command.close()
        stream.close()
        try:
            os.close(master)
        except OSError:
            pass


def ready(t):
    assert t.command.start_policy() is None
    assert t.command() == (0., 0., 0.)


def advance(t, seconds=1.2):
    for _ in range(round(seconds / .02)):
        t.clock.advance()
        value = t.command()
    return value


def test_idle_and_unterminated_line_do_not_block_or_change_velocity():
    with terminal() as t:
        ready(t)
        for _ in range(100):
            assert t.command.check_stop() is None
        t.write(b'1')
        assert advance(t) == (0., 0., 0.)
        t.write(b'\n')
        assert advance(t) == pytest.approx((.3, 0., 0.))


def test_start_discards_old_complete_and_partial_commands_and_preserves_terminal_mode():
    with terminal() as t:
        mode = termios.tcgetattr(t.stream.fileno())
        t.write(b'1\n')
        assert t.command.check_stop() is None
        assert advance(t) == (0., 0., 0.)
        t.write(b'2\n3')
        assert t.command.start_policy() is None
        # Enter cannot complete the old pending '3' after the activation flush.
        t.write(b'\n')
        assert advance(t) == (0., 0., 0.)
        assert termios.tcgetattr(t.stream.fileno()) == mode
        t.command.close()
        assert termios.tcgetattr(t.stream.fileno()) == mode
        assert not t.stream.closed


def test_first_policy_sample_is_zero_even_if_a_new_command_is_already_ready():
    with terminal() as t:
        assert t.command.start_policy() is None
        t.write(b'1\n')
        t.clock.advance(3.)
        assert t.command() == (0., 0., 0.)
        t.clock.advance(.02)
        assert t.command() == pytest.approx((.006, 0., 0.))


def test_start_clears_partial_bytes_already_read_into_the_internal_buffer():
    with terminal() as t:
        # Canonical Ctrl+D following text delivers that text without a newline.
        t.write(b'1\x04')
        assert t.command.check_stop() is None
        assert t.command.start_policy() is None
        t.write(b'\n')
        assert advance(t) == (0., 0., 0.)


@pytest.mark.parametrize(('key', 'expected'), [
    (b'1\n', (.3, 0., 0.)), (b'2\n', (-.3, 0., 0.)),
    (b'3\n', (0., .3, 0.)), (b'4\n', (0., -.3, 0.)),
    (b'5\n', (0., 0., 1.)), (b'6\n', (0., 0., -1.)),
])
def test_direction_signs_latch_and_slew(key, expected):
    with terminal() as t:
        ready(t)
        t.write(key)
        values = []
        for _ in range(60):
            t.clock.advance()
            values.append(t.command())
        assert values[-1] == pytest.approx(expected)
        changes = np.diff(np.array([(0., 0., 0.), *values]), axis=0)
        assert np.all(np.abs(changes) <= np.array([.006, .006, .02]) + 1e-10)
        # No new key is required to keep the last direction selected.
        assert advance(t) == pytest.approx(expected)


def test_number_one_is_point_three_even_when_configured_limit_is_point_four():
    with terminal(max_vx=.4) as t:
        ready(t)
        t.write(b'1\n')
        assert advance(t, 1.6) == pytest.approx((.3, 0., 0.))
        t.write(b'2\n')
        assert advance(t, 2.6) == pytest.approx((-.3, 0., 0.))


def test_zero_is_immediate_and_does_not_request_exit():
    with terminal() as t:
        ready(t)
        t.write(b'1\n')
        assert advance(t)[0] == pytest.approx(.3)
        t.write(b'0\n')
        assert t.command.check_stop() is None
        assert t.command() == (0., 0., 0.)
        assert advance(t) == (0., 0., 0.)
        t.write(b'3\n')
        assert advance(t) == pytest.approx((0., .3, 0.))


def test_multiple_lines_are_drained_without_textio_read_ahead():
    with terminal() as t:
        ready(t)
        t.write(b'1\n2\n3\n4\n5\n6\n0\n')
        assert t.command.check_stop() is None
        assert advance(t) == (0., 0., 0.)
        t.write(b'1\n3\n')
        assert advance(t) == pytest.approx((0., .3, 0.))


def test_invalid_input_zeros_velocity_but_empty_lines_leave_it_latched(capsys):
    with terminal() as t:
        ready(t)
        t.write(b'1\n')
        assert advance(t)[0] == pytest.approx(.3)
        t.write(b'\n \n')
        assert t.command() == pytest.approx((.3, 0., 0.))
        t.write(b'123\n')
        assert t.command.check_stop() is None
        assert t.command() == (0., 0., 0.)
        assert '无效输入已按0处理' in capsys.readouterr().out


@pytest.mark.parametrize('key', [b'q\n', b' STOP \n', b'exit\n'])
@pytest.mark.parametrize('active', [False, True])
def test_quit_is_sticky_before_or_after_policy_start(key, active):
    with terminal() as t:
        if active:
            ready(t)
            t.write(b'1\n')
            advance(t)
        t.write(key + b'1\n')
        reason = t.command.check_stop()
        assert reason and '退出' in reason
        assert t.command.start_policy() == reason
        assert t.command.check_stop() == reason
        assert t.command() == (0., 0., 0.)


def test_start_detects_pending_quit_before_flushing():
    with terminal() as t:
        t.write(b'q\n')
        assert '退出' in t.command.start_policy()
        assert t.command() == (0., 0., 0.)


@pytest.mark.parametrize('active', [False, True])
def test_ctrl_d_eof_requests_stop(active):
    with terminal() as t:
        if active:
            ready(t)
        t.write(b'\x04')
        reason = t.command.check_stop()
        assert reason and 'EOF' in reason
        assert t.command() == (0., 0., 0.)


def test_closed_terminal_requests_stop_instead_of_latching_motion():
    with terminal() as t:
        ready(t)
        t.write(b'1\n')
        advance(t)
        os.close(t.master)
        assert t.command.check_stop()
        assert t.command() == (0., 0., 0.)


def test_bad_descriptor_read_error_returns_a_stop_reason():
    with terminal() as t:
        ready(t)
        t.stream.close()
        assert '读取失败' in t.command.check_stop()
        assert t.command() == (0., 0., 0.)


@pytest.mark.parametrize('data', [b'1' * 65 + b'\n', b'1\n' * 40])
def test_input_limits_fail_closed_without_unbounded_drain(data):
    with terminal() as t:
        ready(t)
        t.write(data)
        assert t.command.check_stop()
        assert t.command() == (0., 0., 0.)


def test_many_lines_in_one_read_are_bounded(monkeypatch):
    with terminal() as t:
        ready(t)
        monkeypatch.setattr(terminal_command.select, 'select', lambda *args: ([t.stream.fileno()], [], []))
        calls = []
        def read(fd, count):
            calls.append((fd, count))
            return b'1\n' * 40
        monkeypatch.setattr(terminal_command.os, 'read', read)
        assert '输入过多' in t.command.check_stop()
        assert len(calls) == 1
        assert t.command() == (0., 0., 0.)


def test_stalled_clock_cannot_cause_instant_full_velocity():
    with terminal() as t:
        ready(t)
        t.write(b'1\n')
        t.clock.advance(20.)
        assert t.command() == pytest.approx((.03, 0., 0.))


@pytest.mark.parametrize('options', [
    {'max_vx': 0}, {'max_vx': .41}, {'max_vy': .31}, {'max_wz': 1.01},
    {'max_vx': float('nan')}, {'max_vy': float('inf')}, {'max_wz': -1},
])
def test_bad_speed_limits_rejected(options):
    with pytest.raises(ValueError):
        TerminalCommand(stream=io.StringIO(), **options)


def test_noninteractive_input_is_rejected_before_control():
    with pytest.raises(ValueError, match='交互终端'):
        TerminalCommand(stream=io.StringIO())


def test_missing_descriptor_is_rejected():
    with pytest.raises(ValueError, match='文件描述符'):
        TerminalCommand(stream=SimpleNamespace(isatty=lambda: True))
