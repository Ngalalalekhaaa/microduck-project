"""PTY action selections exercise the shared Xbox state machine without hardware."""
from contextlib import contextmanager
import os
import pty
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.gamepad import GamepadStop, RouladeGamepadCommand
from microduck_deploy.terminal_actions import TerminalActionCommand
from test_terminal_command import Clock


ZERO = (0., 0., 0.)


@contextmanager
def terminal(*, clock=None, **options):
    master, slave = pty.openpty()
    stream = os.fdopen(slave, 'rb', buffering=0)
    clock = Clock() if clock is None else clock
    command = TerminalActionCommand(stream=stream, clock=clock, **options)
    try:
        yield SimpleNamespace(command=command, clock=clock, master=master,
                              stream=stream, write=lambda value: os.write(master, value))
    finally:
        command.close()
        stream.close()
        try:
            os.close(master)
        except OSError:
            pass


def ready(t):
    assert t.command.start_policy() is None
    assert t.command() == ZERO


def tick(t, seconds=.02, line=None):
    if line is not None:
        t.write((line + '\n').encode())
    t.clock.advance(seconds)
    return t.command()


def advance(t, seconds):
    value = ZERO
    for _ in range(round(seconds / .02)):
        value = tick(t)
    return value


def ground(t):
    assert tick(t, line='9') == ZERO
    assert t.command.policy_mode == 'stopping'
    tick(t, .61)
    assert t.command.policy_mode == 'ground_pick'


def roll(t):
    assert tick(t, line='r') == ZERO
    assert t.command.policy_mode == 'stopping'
    tick(t, .61)
    assert t.command.policy_mode == 'roulade'


def test_shared_state_machine_and_seven_toggles_sit_then_stand():
    with terminal() as t:
        assert isinstance(t.command._actions, RouladeGamepadCommand)
        ready(t)
        tick(t, line='1')
        assert advance(t, 1.2)[0] == pytest.approx(.3)
        assert tick(t, line='7') == ZERO
        tick(t, .61)
        assert t.command.policy_mode == 'sit'
        tick(t, 1., line='1')
        assert t.command.policy_mode == 'sit'
        assert tick(t, line='7') == ZERO
        assert t.command.policy_mode == 'stand'
        tick(t, 1., line='8')
        tick(t, 2.1)
        assert t.command.policy_mode == 'stand'
        # Walking requested while sitting must not reappear after standing.
        assert tick(t, 1.) == ZERO
        assert t.command.policy_mode == 'stand'
        tick(t, line='1')
        assert t.command.policy_mode == 'walking'
        assert advance(t, 1.2)[0] == pytest.approx(.3)


def test_seven_switches_back_to_sit_after_standing_guard_and_eight_remains_compatible():
    with terminal() as t:
        ready(t)
        tick(t, line='7')
        tick(t, .61)
        tick(t, line='7')
        assert t.command.policy_mode == 'stand'
        tick(t, 3.1)
        tick(t, line='7')
        assert t.command.policy_mode == 'sit'
        tick(t, line='8')
        assert t.command.policy_mode == 'stand'


def test_accepted_busy_and_sitting_feedback_explains_what_happened(capsys):
    with terminal() as t:
        ready(t)
        tick(t, line='7')
        assert '已接收7：先停步0.6秒' in capsys.readouterr().out
        tick(t, line='7')
        assert '坐下前正在停步' in capsys.readouterr().out
        tick(t, .61)
        tick(t, line='9')
        assert '当前为坐姿，请先轻按7站起' in capsys.readouterr().out
        tick(t, line='7')
        assert '已接收7：切换站立模型' in capsys.readouterr().out
        tick(t, line='7')
        assert '正在站起，还需' in capsys.readouterr().out
        tick(t, 3.1)
        tick(t, line='7')
        assert '已接收7：切换坐下模型' in capsys.readouterr().out


def test_repeated_busy_notices_are_coalesced_without_adding_key_delay(capsys):
    with terminal() as t:
        ready(t)
        tick(t, line='7')
        capsys.readouterr()
        for _ in range(5):
            tick(t, line='7')
        assert capsys.readouterr().out.count('按键7未执行') == 1
        tick(t, .51)
        assert t.command.policy_mode == 'sit'
        # A new tap is accepted immediately after stopping has completed.
        tick(t, line='7')
        assert t.command.policy_mode == 'stand'
        assert '已接收7' in capsys.readouterr().out


class AdvancingClock(Clock):
    """Real clocks advance even between reads within one policy cycle."""

    def __call__(self):
        self.now += .000001
        return self.now


@pytest.mark.parametrize('key, expected', [('7', 'sit'), ('9', 'ground_pick'), ('r', 'roulade')])
def test_actions_use_snapshot_before_state_machine_clock_reads(key, expected, capsys):
    with terminal(clock=AdvancingClock()) as t:
        ready(t)
        assert t.command.latest() is t.command.latest()
        tick(t, line=key)
        assert t.command.policy_mode == 'stopping'
        tick(t, .61)
        assert t.command.policy_mode == expected
        assert not t.command._actions.disconnected_reported
        assert '输入断开或过期' not in capsys.readouterr().out
        if expected == 'sit':
            tick(t, line='7')
            assert t.command.policy_mode == 'stand'


def test_immediate_keys_with_advancing_clock_toggle_without_enter(capsys):
    with terminal(clock=AdvancingClock(), single_key=True) as t:
        ready(t)
        t.write(b'7')
        tick(t)
        assert t.command.policy_mode == 'stopping'
        tick(t, .61)
        assert t.command.policy_mode == 'sit'
        t.write(b'7')
        tick(t)
        assert t.command.policy_mode == 'stand'
        assert not t.command._actions.disconnected_reported
        assert '输入断开或过期' not in capsys.readouterr().out


def test_stand_from_walking_never_runs_sit_policy_and_does_not_toggle():
    with terminal() as t:
        ready(t)
        tick(t, line='8')
        assert t.command.policy_mode == 'stand'
        for _ in range(5):
            tick(t, .7, line='8')
            assert t.command.policy_mode == 'stand'
        tick(t, line='3')
        assert t.command.policy_mode == 'walking'
        assert advance(t, 1.2) == pytest.approx((0., .3, 0.))


def test_ground_full_phase_then_walking_hold_without_previous_or_busy_direction_queue():
    with terminal() as t:
        ready(t)
        tick(t, line='1')
        advance(t, 1.2)
        ground(t)
        np.testing.assert_allclose(t.command.ground_pick_command, (1., 0., 0.), atol=1e-12)
        for expected in ((0., 1., 0.), (-1., 0., 0.), (0., -1., 0.)):
            tick(t, 1., line='1\n7\n8\n9\nr')
            assert t.command.policy_mode == 'ground_pick'
            np.testing.assert_allclose(t.command.ground_pick_command, expected, atol=1e-10)
        tick(t, 1.01)
        assert t.command.policy_mode == 'walking'
        assert t.command._ground_walk_recovery_until is not None
        tick(t, 2.9, line='2\n9\nr')
        tick(t, .2)
        assert t.command.policy_mode == 'walking'
        assert tick(t, 2.) == ZERO
        tick(t, line='2')
        assert t.command.policy_mode == 'walking'
        assert advance(t, 1.2)[0] == pytest.approx(-.3)


def test_action_pressed_with_direction_same_read_discards_direction_and_repeat():
    with terminal() as t:
        ready(t)
        tick(t, line='9\n1\n9\nr\n7')
        tick(t, .61)
        assert t.command.policy_mode == 'ground_pick'
        tick(t, 4.01)
        tick(t, 4.)
        assert t.command.policy_mode == 'walking'
        assert t.command() == ZERO


def test_zero_does_not_abort_action_but_quit_does_request_stop():
    with terminal() as t:
        ready(t)
        ground(t)
        assert tick(t, line='0') == ZERO
        assert t.command.policy_mode == 'ground_pick'
        t.write(b'q\n9\n')
        reason = t.command.check_stop()
        assert reason and '退出' in reason
        assert t.command() == ZERO
        assert t.command.start_policy() == reason


def test_roll_once_requires_runtime_completion_then_stand_and_fresh_direction():
    with terminal() as t:
        ready(t)
        roll(t)
        tick(t, 1., line='r\n7\n8\n9\n1')
        assert t.command.policy_mode == 'roulade'
        t.command.finish_roulade()
        assert t.command.policy_mode == 'stand'
        tick(t, 2., line='r\n9\n1')
        assert t.command.policy_mode == 'stand'
        tick(t, 1.1)
        assert t.command.policy_mode == 'stand'
        tick(t, line='r')
        assert t.command.policy_mode == 'stopping'


def test_uncompleted_roll_times_out_instead_of_assuming_success():
    with terminal(roulade_timeout_s=2.) as t:
        ready(t)
        roll(t)
        with pytest.raises(GamepadStop, match='超时'):
            tick(t, 2.01)


@pytest.mark.parametrize('line', ['7', '8', '9', 'r'])
def test_old_complete_and_partial_action_input_is_discarded_at_start(line):
    with terminal() as t:
        t.write((line + '\n').encode())
        assert t.command.check_stop() is None
        t.write(line.encode())
        ready(t)
        t.write(b'\n')
        tick(t, 10.)
        assert t.command.policy_mode == 'walking'
        assert t.command() == ZERO


@pytest.mark.parametrize('line', ['7', '8', '9', 'r'])
def test_first_sample_remains_zero_walking_then_accepts_fresh_action(line):
    with terminal() as t:
        assert t.command.start_policy() is None
        t.write((line + '\n').encode())
        assert t.command() == ZERO
        assert t.command.policy_mode == 'walking'
        tick(t)
        assert t.command.policy_mode == ('stand' if line == '8' else 'stopping')


@pytest.mark.parametrize('action', ['7', '9', 'r'])
@pytest.mark.parametrize('end', ['q', 'eof', 'disconnect'])
def test_quit_or_terminal_loss_stops_in_each_action(action, end):
    with terminal() as t:
        ready(t)
        tick(t, line=action)
        tick(t, .61)
        if end == 'disconnect':
            os.close(t.master)
        else:
            t.write(b'q\n' if end == 'q' else b'\x04')
        assert t.command.check_stop()
        assert t.command() == ZERO


@pytest.mark.parametrize('line', ['7', '8', '9', 'r'])
def test_disabled_model_key_never_dispatches_and_zeros_velocity(line, capsys):
    with terminal(sitstand_enabled=False, ground_pick_enabled=False,
                  roulade_enabled=False) as t:
        ready(t)
        tick(t, line='1')
        advance(t, 1.2)
        assert tick(t, line=line) == ZERO
        tick(t, 10.)
        assert t.command.policy_mode == 'walking'
        assert '模型未启用' in capsys.readouterr().out


def test_terminal_velocity_signs_and_original_slew_are_preserved():
    with terminal(max_vx=.4) as t:
        ready(t)
        assert tick(t, line='1') == pytest.approx((.006, 0., 0.))
        assert advance(t, 2.) == pytest.approx((.3, 0., 0.))
        assert tick(t, line='0') == ZERO
        assert tick(t, line='2') == pytest.approx((-.006, 0., 0.))
        assert advance(t, 2.) == pytest.approx((-.3, 0., 0.))


@pytest.mark.parametrize('options', [
    {'sitstand_enabled': 'true'}, {'ground_pick_enabled': 1},
    {'roulade_enabled': None}, {'sitstand_enabled': False},
])
def test_invalid_action_configuration_rejected(options):
    with pytest.raises(ValueError):
        TerminalActionCommand(**options)
