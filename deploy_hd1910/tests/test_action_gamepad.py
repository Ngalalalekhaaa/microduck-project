import dataclasses
import math
from types import SimpleNamespace

import pytest

from microduck_deploy.gamepad import (
    ActionGamepadCommand, GamepadStop, GROUND_PICK_BUTTON,
    GROUND_PICK_SECONDS, PadSample, SITSTAND_BUTTON, SitStandGamepadCommand,
)


ZERO = (0., 0., 0.)
MOVING = {'ABS_X': -.5, 'ABS_Y': -1., 'ABS_RX': -.5}


class Pad:
    def __init__(self, *, buttons=(), counts=None, **options):
        self.now = 10.
        self.sample = PadSample(
            self.now, True, {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {},
            buttons, button_press_counts=counts or {},
        )
        self.command = ActionGamepadCommand(
            SimpleNamespace(latest=lambda: self.sample), clock=lambda: self.now,
            **options,
        )

    def tick(self, dt=.02, **changes):
        self.now += dt
        changes.setdefault('timestamp', self.now)
        self.sample = dataclasses.replace(self.sample, **changes)
        return self.command()

    def press(self, button, *, down=False, **changes):
        counts = dict(self.sample.button_press_counts)
        counts[button] = counts.get(button, 0) + 1
        return self.tick(buttons=(button,) if down else (),
                         button_press_counts=counts, **changes)

    def ground(self):
        self.tick()
        self.press(GROUND_PICK_BUTTON, axes=MOVING)
        assert self.command.policy_mode == 'stopping'
        assert self.tick(.61) == ZERO
        assert self.command.policy_mode == 'ground_pick'


def test_complete_four_second_circle_and_rise_guard_before_walking():
    pad = Pad()
    assert isinstance(pad.command, SitStandGamepadCommand)
    assert pad.command.ground_pick_command == (1., 0., 0.)
    pad.ground()
    started = pad.now
    for fraction in (0., .25, .5, .7, .75, .999):
        pad.now = started + GROUND_PICK_SECONDS * fraction
        assert pad.tick(0) == ZERO
        assert pad.command.policy_mode == 'ground_pick'
        expected = (math.cos(2 * math.pi * fraction),
                    math.sin(2 * math.pi * fraction), 0.)
        assert pad.command.ground_pick_command == pytest.approx(expected)
    pad.now = started + 4.001
    assert pad.tick(0) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(2.99) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(.02) == ZERO
    assert pad.command.policy_mode == 'walking'
    assert pad.tick(.1) == pytest.approx((.03, .03, .1))


def test_rb_waits_for_zero_velocity_stop_from_walking():
    pad = Pad()
    pad.tick()
    assert pad.tick(.1, axes=MOVING)[0] > 0
    assert pad.press(GROUND_PICK_BUTTON) == ZERO
    assert pad.command.policy_mode == 'stopping'
    assert pad.tick(.59) == ZERO
    assert pad.command.policy_mode == 'stopping'
    assert pad.tick(.02) == ZERO
    assert pad.command.policy_mode == 'ground_pick'


def test_rb_can_start_from_standing_but_not_sitting():
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.press(SITSTAND_BUTTON)
    assert pad.command.policy_mode == 'sit'
    assert pad.press(GROUND_PICK_BUTTON) == ZERO
    assert pad.command.policy_mode == 'sit'
    pad.press(SITSTAND_BUTTON)
    assert pad.command.policy_mode == 'stand'
    pad.tick(3.01)
    assert pad.press(GROUND_PICK_BUTTON) == ZERO
    assert pad.command.policy_mode == 'ground_pick'


@pytest.mark.parametrize('previous_action', ['sit', 'ground_pick'])
def test_rb_during_rise_guard_is_discarded_and_requires_a_new_press(previous_action):
    pad = Pad()
    if previous_action == 'sit':
        pad.tick()
        pad.press(SITSTAND_BUTTON)
        pad.tick(.61)
        pad.press(SITSTAND_BUTTON)
    else:
        pad.ground()
        pad.tick(4.01, axes={'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.})
    assert pad.command.policy_mode == 'stand'
    assert pad.press(GROUND_PICK_BUTTON, down=True) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(2.99) == ZERO
    assert pad.command.policy_mode == 'stand'
    # Crossing the guard deadline while holding RB must not queue an action.
    assert pad.tick(.1) == ZERO
    assert pad.command.policy_mode == 'stand'
    pad.tick(buttons=())
    assert pad.press(GROUND_PICK_BUTTON) == ZERO
    assert pad.command.policy_mode == 'stopping'


def test_ground_presses_are_consumed_without_queue_or_hold_retrigger():
    pad = Pad()
    pad.ground()
    pad.press(GROUND_PICK_BUTTON, down=True)
    pad.press(SITSTAND_BUTTON)
    pad.press(GROUND_PICK_BUTTON, down=True)
    assert pad.command.policy_mode == 'ground_pick'
    assert pad.tick(4.01, axes={'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}) == ZERO
    assert pad.command.policy_mode == 'stand'
    for _ in range(5):
        assert pad.tick(1) == ZERO
        assert pad.command.policy_mode == 'stand'
    pad.tick(buttons=())
    pad.press(GROUND_PICK_BUTTON)
    assert pad.command.policy_mode == 'stopping'


def test_rb_and_y_are_ignored_during_initial_stop_instead_of_cancelling():
    pad = Pad()
    pad.tick()
    pad.press(GROUND_PICK_BUTTON)
    pad.press(SITSTAND_BUTTON)
    pad.press(GROUND_PICK_BUTTON)
    assert pad.command.policy_mode == 'stopping'
    pad.tick(.61)
    assert pad.command.policy_mode == 'ground_pick'


def test_y_wins_simultaneous_y_rb_and_rb_is_not_queued():
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.tick(button_press_counts={SITSTAND_BUTTON: 1, GROUND_PICK_BUTTON: 1})
    assert pad.command.policy_mode == 'sit'
    pad.tick(5)
    assert pad.command.policy_mode == 'sit'
    pad.press(SITSTAND_BUTTON)
    assert pad.command.policy_mode == 'stand'
    pad.tick(5)
    assert pad.command.policy_mode == 'stand'


def test_startup_press_counts_and_held_rb_are_ignored():
    pad = Pad(buttons=(GROUND_PICK_BUTTON,), counts={GROUND_PICK_BUTTON: 4})
    pad.tick(button_press_counts={GROUND_PICK_BUTTON: 5})
    assert pad.command.policy_mode == 'walking'
    pad.tick()
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=())
    pad.press(GROUND_PICK_BUTTON)
    assert pad.command.policy_mode == 'stopping'


def test_rb_fallback_uses_fresh_edges_and_does_not_change_x_binding():
    pad = Pad(stop_seconds=0)
    pad.tick()
    pad.tick(buttons=(307,))
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=(GROUND_PICK_BUTTON,))
    assert pad.command.policy_mode == 'ground_pick'
    pad.tick(4.01)
    assert pad.command.policy_mode == 'stand'
    pad.tick(4)
    assert pad.command.policy_mode == 'stand'
    pad.tick(buttons=())
    pad.tick(buttons=(GROUND_PICK_BUTTON,))
    assert pad.command.policy_mode == 'ground_pick'


@pytest.mark.parametrize('connected,age', [(False, 0), (True, .31), (True, -.1)])
def test_bad_input_finishes_ground_and_discards_reconnect_presses(connected, age):
    pad = Pad()
    pad.ground()
    started = pad.now
    pad.now = started + 2.8
    assert pad.tick(0, connected=connected, timestamp=pad.now-age,
                    button_press_counts={GROUND_PICK_BUTTON: 9, SITSTAND_BUTTON: 7}) == ZERO
    assert pad.command.policy_mode == 'ground_pick'
    assert pad.command.ground_pick_command[1] == pytest.approx(math.sin(1.4 * math.pi))
    pad.now = started + 4.01
    assert pad.tick(0, connected=False) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(4, connected=False) == ZERO
    assert pad.tick(connected=True, axes={'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.},
                    buttons=(GROUND_PICK_BUTTON,),
                    button_press_counts={GROUND_PICK_BUTTON: 10, SITSTAND_BUTTON: 8}) == ZERO
    assert pad.command.policy_mode == 'stand'
    pad.tick()
    assert pad.command.policy_mode == 'stand'
    pad.tick(buttons=())
    pad.press(GROUND_PICK_BUTTON)
    assert pad.command.policy_mode == 'stopping'


def test_disconnect_during_accepted_stop_still_completes_action():
    pad = Pad()
    pad.tick()
    pad.press(GROUND_PICK_BUTTON)
    pad.tick(.61, connected=False)
    assert pad.command.policy_mode == 'ground_pick'
    pad.tick(3.99, connected=False)
    assert pad.command.policy_mode == 'ground_pick'
    pad.tick(.02, connected=False)
    assert pad.command.policy_mode == 'stand'


@pytest.mark.parametrize('phase', ['walking', 'stopping', 'ground_pick', 'sit', 'stand'])
@pytest.mark.parametrize('latched', [True, False])
def test_b_precedes_every_action_even_when_disconnected(phase, latched):
    pad = Pad()
    pad.tick()
    if phase in {'stopping', 'ground_pick'}:
        pad.press(GROUND_PICK_BUTTON)
        if phase == 'ground_pick':
            pad.tick(.61)
    if phase in {'sit', 'stand'}:
        pad.press(SITSTAND_BUTTON)
        pad.tick(.61)
        if phase == 'stand':
            pad.press(SITSTAND_BUTTON)
    with pytest.raises(GamepadStop):
        pad.tick(10, connected=False, stop_requested=latched,
                 buttons=() if latched else (305,),
                 button_press_counts={GROUND_PICK_BUTTON: 99, SITSTAND_BUTTON: 99})
    assert pad.command.policy_mode == phase


def test_y_retains_stop_cancellation_sit_and_rise_behaviour():
    pad = Pad()
    pad.tick()
    assert pad.press(SITSTAND_BUTTON, axes=MOVING) == ZERO
    assert pad.command.policy_mode == 'stopping'
    assert pad.press(SITSTAND_BUTTON) == ZERO
    assert pad.command.policy_mode == 'stand'
    assert pad.tick(2.99) == ZERO
    assert pad.tick(.02) == ZERO
    assert pad.command.policy_mode == 'walking'
    pad.press(SITSTAND_BUTTON)
    pad.tick(.61)
    assert pad.command.policy_mode == 'sit'
    assert pad.tick(5, axes=MOVING) == ZERO
    pad.press(SITSTAND_BUTTON)
    pad.tick(3.01, axes={'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.})
    assert pad.command.policy_mode == 'stand'


def test_regular_y_gait_stop_keeps_existing_disconnect_behaviour():
    pad = Pad()
    pad.tick()
    pad.press(SITSTAND_BUTTON)
    assert pad.tick(5, connected=False) == ZERO
    assert pad.command.policy_mode == 'stopping'
