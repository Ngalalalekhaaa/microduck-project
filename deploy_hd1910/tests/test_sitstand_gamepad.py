import dataclasses
from types import SimpleNamespace

import pytest

from microduck_deploy.gamepad import (
    GamepadError, GamepadStop, PadSample, SITSTAND_BUTTON, SitStandGamepadCommand,
)


class Pad:
    def __init__(self, *, buttons=(), counts=None, **options):
        self.now = 10.
        self.sample = PadSample(
            self.now, True, {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {},
            buttons, button_press_counts=counts or {},
        )
        self.command = SitStandGamepadCommand(
            SimpleNamespace(latest=lambda: self.sample), clock=lambda: self.now,
            **options,
        )

    def tick(self, dt=.02, **changes):
        self.now += dt
        changes.setdefault('timestamp', self.now)
        self.sample = dataclasses.replace(self.sample, **changes)
        return self.command()

    def y(self, count, *, down=False, **changes):
        return self.tick(buttons=(SITSTAND_BUTTON,) if down else (),
                         button_press_counts={SITSTAND_BUTTON: count}, **changes)


MOVING = {'ABS_X': -.5, 'ABS_Y': -1., 'ABS_RX': -.5}
ZERO = (0., 0., 0.)


def test_startup_and_first_active_call_discard_old_or_held_y():
    pad = Pad(buttons=(SITSTAND_BUTTON,), counts={SITSTAND_BUTTON: 7})
    assert pad.command.policy_mode == 'walking'
    # Another press can accumulate while startup/HOME runs before the loop.
    assert pad.y(8, down=True) == ZERO
    assert pad.command.policy_mode == 'walking'
    pad.y(8, down=True)
    assert pad.command.policy_mode == 'walking'
    pad.y(8)
    pad.y(9, down=True)
    assert pad.command.policy_mode == 'stopping'


def test_short_press_is_retained_and_gait_stops_before_sitting():
    pad = Pad()
    pad.tick()
    assert pad.tick(.1, axes=MOVING)[0] > 0
    # Y was pressed and released between policy calls: active keys are empty.
    assert pad.y(1) == ZERO
    assert pad.command.policy_mode == 'stopping'
    assert pad.tick(.59) == ZERO
    assert pad.command.policy_mode == 'stopping'
    assert pad.tick(.02) == ZERO
    assert pad.command.policy_mode == 'sit'
    for _ in range(5):
        assert pad.tick(.1, buttons=(SITSTAND_BUTTON,), axes=MOVING) == ZERO
        assert pad.command.policy_mode == 'sit'


def test_second_y_during_gait_stop_cancels_sitting_and_guards_rise():
    pad = Pad()
    pad.tick()
    pad.y(1)
    assert pad.y(2, axes=MOVING) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(2.99) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(.02) == ZERO
    assert pad.command.policy_mode == 'walking'
    assert pad.tick(.1)[0] == pytest.approx(.03)


def test_sticks_cannot_interrupt_sitting_or_first_three_seconds_of_rise():
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.y(1, axes=MOVING)
    assert pad.command.policy_mode == 'sit'
    assert pad.tick(5) == ZERO
    assert pad.command.policy_mode == 'sit'
    assert pad.y(2) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(2.99) == ZERO
    assert pad.command.policy_mode == 'stand'
    pad.tick(.02)
    assert pad.command.policy_mode == 'walking'
    value = pad.tick(.1)
    assert value == pytest.approx((.03, .03, .1))


def test_centered_sticks_keep_stand_and_y_from_stand_sits_immediately():
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.y(1)
    pad.y(2)
    assert pad.tick(4) == ZERO
    assert pad.command.policy_mode == 'stand'
    pad.y(3)
    assert pad.command.policy_mode == 'sit'


def test_no_count_fallback_only_uses_fresh_y_edges_and_not_x():
    pad = Pad()
    pad.tick()
    pad.tick(buttons=(307,))  # Existing Xbox X/jaw binding must not sit.
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=(SITSTAND_BUTTON,))
    assert pad.command.policy_mode == 'stopping'
    for _ in range(3):
        pad.tick(.1)
        assert pad.command.policy_mode == 'stopping'
    pad.tick(buttons=())
    pad.tick(buttons=(SITSTAND_BUTTON,))
    assert pad.command.policy_mode == 'stand'


def test_each_retained_press_is_counted_once_even_when_both_released():
    pad = Pad()
    pad.tick()
    pad.y(2)  # First starts stopping; second cancels into stand.
    assert pad.command.policy_mode == 'stand'
    for _ in range(4):
        pad.y(2)
        assert pad.command.policy_mode == 'stand'


@pytest.mark.parametrize('connected,age', [(False, 0), (True, .31), (True, -.1)])
def test_bad_input_freezes_mode_and_reconnect_discards_queued_y(connected, age):
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.y(1)
    assert pad.command.policy_mode == 'sit'
    pad.now += .02
    assert pad.tick(0, connected=connected, timestamp=pad.now-age,
                    button_press_counts={SITSTAND_BUTTON: 2}, axes=MOVING) == ZERO
    assert pad.command.policy_mode == 'sit'
    assert pad.y(3, connected=True, down=True) == ZERO
    assert pad.command.policy_mode == 'sit'
    assert pad.y(3, down=True) == ZERO
    assert pad.command.policy_mode == 'sit'
    pad.y(3)
    pad.y(4)
    assert pad.command.policy_mode == 'stand'


def test_disconnected_gait_stop_does_not_advance_posture():
    pad = Pad()
    pad.tick()
    pad.y(1)
    assert pad.tick(5, connected=False) == ZERO
    assert pad.command.policy_mode == 'stopping'


def test_b_stop_precedes_y_even_with_stale_disconnected_input():
    pad = Pad()
    pad.tick()
    with pytest.raises(GamepadStop):
        pad.y(1, stop_requested=True, connected=False)
    assert pad.command.policy_mode == 'walking'


@pytest.mark.parametrize('options', [{'stop_seconds': -1}, {'rise_seconds': float('nan')}])
def test_invalid_transition_wait_is_rejected(options):
    with pytest.raises(GamepadError):
        Pad(**options)
