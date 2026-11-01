"""Action-key timing with clocks that advance between individual reads.

A frozen test clock cannot expose a virtual-pad sample timestamped after the
consumer has already read its current time. These tests exercise that ordering
without weakening the real gamepad's freshness checks.
"""

import os
import time
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy import runtime
from test_runtime import rig
from test_terminal_action_runtime import attach_policies, assert_unloaded, on_infer
from test_terminal_single_key_runtime import immediate_terminal, assert_restored


class ReadingClock:
    """Deterministic elapsed work: every read is strictly later than the last."""

    def __init__(self, read_seconds=1e-5):
        self.now = 100.
        self.read_seconds = read_seconds

    def __call__(self):
        self.now += self.read_seconds
        return self.now

    def advance(self, seconds):
        self.now += seconds


def poll(t, clock, *, key=None, seconds=.02):
    if key is not None:
        t.send(key)
    clock.advance(seconds)
    return t.source()


@pytest.mark.parametrize('read_seconds', [1e-6, 1e-4])
def test_advancing_clock_seven_sits_then_seven_stands_and_q_restores(read_seconds):
    clock = ReadingClock(read_seconds)
    with immediate_terminal(clock) as t:
        assert t.source.start_policy() is None
        assert t.source() == (0., 0., 0.)
        assert poll(t, clock, key='7') == (0., 0., 0.)
        assert t.source.policy_mode == 'stopping'
        poll(t, clock, seconds=.61)
        assert t.source.policy_mode == 'sit'
        poll(t, clock, key='7')
        assert t.source.policy_mode == 'stand'
        poll(t, clock, key='q')
        assert '终端请求退出' in t.source.check_stop()
        assert_restored(t)


@pytest.mark.parametrize('key, mode', [('9', 'ground_pick'), ('r', 'roulade')])
def test_advancing_clock_ground_and_roll_keys_reach_the_selected_action(key, mode):
    clock = ReadingClock()
    with immediate_terminal(clock) as t:
        assert t.source.start_policy() is None
        t.source()
        poll(t, clock, key=key)
        assert t.source.policy_mode == 'stopping'
        poll(t, clock, seconds=.61)
        assert t.source.policy_mode == mode
        if mode == 'ground_pick':
            command = t.source.ground_pick_command
            assert np.linalg.norm(command[:2]) == pytest.approx(1)
            assert command[2] == 0
        poll(t, clock, key='q')
        assert_restored(t)


def test_advancing_clock_startup_discards_old_action_keys_without_losing_a_fresh_key():
    clock = ReadingClock()
    with immediate_terminal(clock) as t:
        os.write(t.master, b'7\n9\nr')  # Complete and partial canonical input.
        assert t.source.start_policy() is None
        t.source()
        poll(t, clock, seconds=1.)
        assert t.source.policy_mode == 'walking'
        poll(t, clock, key='7')
        poll(t, clock, seconds=.61)
        assert t.source.policy_mode == 'sit'
        poll(t, clock, key='q')
        assert_restored(t)


@pytest.mark.parametrize('key', ['7', '9', 'r'])
def test_real_monotonic_clock_accepts_a_fresh_single_action_key(key):
    # No wall-clock sleeps: accepting the request must immediately start the
    # normal stopping stage, well before its 0.6-second completion deadline.
    with immediate_terminal(time.monotonic) as t:
        assert t.source.start_policy() is None
        t.source()
        t.send(key)
        t.source()
        assert t.source.policy_mode == 'stopping'
        t.send('q')
        assert t.source.check_stop()
        assert_restored(t)


def test_controller_advancing_clock_seven_toggle_selects_sit_and_stand_models():
    with rig() as r:
        def monotonic():
            r.clock.advance(1e-5)
            return r.clock.now

        with patch.object(runtime.time, 'monotonic', monotonic), immediate_terminal(monotonic) as t:
            sit, ground, roll = attach_policies(r)
            on_infer(r.policy, lambda obs, count: t.send('7') if count == 1 else None)
            on_infer(sit, lambda obs, count: t.send('7') if count == 1 else (
                t.send('q') if count == 3 else None))
            with pytest.raises(runtime.StopRequested, match='终端请求退出'):
                r.ctl.execute(seconds=3, motion=True, command_provider=t.source)
            np.testing.assert_array_equal(np.asarray(sit.observations)[:, 48:51],
                                          [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            assert not ground.observations and not roll.observations
            assert_restored(t)
            assert_unloaded(r)
