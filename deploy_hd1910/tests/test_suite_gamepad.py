"""D-pad driver normalization and opt-in suite action transitions, no hardware."""
import dataclasses
from types import SimpleNamespace

import pytest

from microduck_deploy.gamepad import (
    DPAD_UP_BUTTON as UP, DPAD_DOWN_BUTTON as DOWN,
    DPAD_LEFT_BUTTON as LEFT, DPAD_RIGHT_BUTTON as RIGHT,
    GROUND_PICK_BUTTON as RB, ROULADE_BUTTON as LB, SITSTAND_BUTTON as Y,
    GamepadReader, GamepadStop, PadSample, SuiteGamepadCommand,
)
from test_gamepad import FakeDevice


ZERO = (0., 0., 0.)


class Pad:
    def __init__(self, **options):
        self.now = 10.
        self.sample = PadSample(self.now, True, {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {}, ())
        options.setdefault('roulade_enabled', True)
        options.setdefault('ground_pick_enabled', True)
        self.command = SuiteGamepadCommand(SimpleNamespace(latest=lambda: self.sample),
                                          clock=lambda: self.now, **options)

    def tick(self, dt=.02, **changes):
        self.now += dt
        changes.setdefault('timestamp', self.now)
        self.sample = dataclasses.replace(self.sample, **changes)
        return self.command()

    def press(self, *buttons, **changes):
        counts = dict(self.sample.button_press_counts)
        for button in buttons:
            counts[button] = counts.get(button, 0) + 1
        return self.tick(buttons=buttons, button_press_counts=counts, **changes)

    def roll(self):
        self.tick()
        self.press(DOWN)
        assert self.command.policy_mode == 'stopping'
        self.tick(.61)
        assert self.command.policy_mode == 'roulade'


def test_variants_start_with_existing_models_and_startup_holds_are_discarded():
    for button in (UP, DOWN, LEFT, RIGHT):
        pad = Pad()
        assert pad.command.walking_variant == 'original'
        assert pad.command.sitstand_variant == 'default'
        pad.press(button)
        pad.tick(1)
        assert pad.command.policy_mode == 'walking'
        assert pad.command.walking_variant == 'original'
        assert pad.command.sitstand_variant == 'default'


def test_up_stops_old_walk_then_switches_stands_and_requires_rise_guard():
    pad = Pad()
    pad.tick()
    pad.press(UP, axes={'ABS_X': 0., 'ABS_Y': -1., 'ABS_RX': 0.})
    assert pad.command.policy_mode == 'stopping'
    assert pad.command.walking_variant == 'original'
    assert pad.tick(.59) == ZERO
    assert pad.command.walking_variant == 'original'
    assert pad.tick(.02) == ZERO
    assert pad.command.walking_variant == 'updated'
    assert pad.command.policy_mode == 'stand'
    pad.tick(2.9)
    assert pad.command.policy_mode == 'stand'
    pad.tick(.11)
    assert pad.command.policy_mode == 'walking'
    pad.tick(.1)
    assert pad.command.walking_variant == 'updated'  # Held Up did not toggle again.
    pad.tick(buttons=())
    pad.press(UP)
    pad.tick(.61)
    assert pad.command.walking_variant == 'original'


@pytest.mark.parametrize('during', [UP, DOWN, LEFT, RIGHT, Y, RB, LB])
def test_walk_switch_consumes_buttons_including_its_completion_tick(during):
    pad = Pad()
    pad.tick()
    pad.press(UP)
    pad.press(during)
    pad.tick(.61)
    assert pad.command.walking_variant == 'updated'
    assert pad.command.policy_mode == 'stand'
    pad.tick(3.1)
    assert pad.command.policy_mode == 'stand'
    assert pad.command.sitstand_variant == 'default'


def test_sides_select_sit_policy_but_stand_with_current_variant_before_switch():
    pad = Pad()
    pad.tick()
    pad.press(LEFT)
    assert pad.command.sitstand_variant == 'friction'
    assert pad.command.policy_mode == 'stopping'
    pad.tick(.61)
    assert pad.command.policy_mode == 'sit'
    pad.press(RIGHT)
    assert pad.command.policy_mode == 'stand'
    assert pad.command.sitstand_variant == 'friction'
    pad.press(RIGHT)
    assert pad.command.sitstand_variant == 'friction'
    pad.tick(3.1)
    assert pad.command.policy_mode == 'stand'
    pad.tick(buttons=())
    pad.press(RIGHT)
    assert pad.command.sitstand_variant == 'com'
    assert pad.command.policy_mode == 'stopping'
    pad.tick(.61)
    assert pad.command.policy_mode == 'sit'
    pad.press(Y)
    assert pad.command.policy_mode == 'stand'
    assert pad.command.sitstand_variant == 'com'


def test_down_roll_uses_same_state_and_finish_callback_without_forging_lb():
    pad = Pad()
    pad.roll()
    assert LB not in pad.sample.button_press_counts
    assert pad.command._last_lb_count == 0
    for button in (UP, DOWN, LEFT, RIGHT, Y, RB, LB):
        pad.press(button)
        assert pad.command.policy_mode == 'roulade'
    pad.command.finish_roulade()
    assert pad.command.policy_mode == 'stand'
    for button in (UP, DOWN, LEFT, RIGHT):
        pad.press(button)
        assert pad.command.policy_mode == 'stand'
    pad.tick(3.1)
    assert pad.command.policy_mode == 'stand'
    assert pad.command.walking_variant == 'original'
    assert pad.command.sitstand_variant == 'default'


def test_down_is_disabled_without_explicit_roulade_enable_and_ignored_while_sitting():
    pad = Pad(roulade_enabled=False)
    pad.tick()
    pad.press(DOWN)
    pad.tick(.61)
    assert pad.command.policy_mode == 'walking'
    pad.press(LEFT)
    pad.tick(.61)
    pad.press(DOWN)
    assert pad.command.policy_mode == 'sit'


@pytest.mark.parametrize('button', [UP, DOWN, LEFT, RIGHT])
@pytest.mark.parametrize('busy', ['sit_pending', 'ground_pending', 'ground_active', 'roll_pending', 'roll_active', 'rise_guard'])
def test_dpad_edges_during_busy_states_are_consumed_not_queued(button, busy):
    pad = Pad()
    pad.tick()
    if busy.startswith('sit'):
        pad.press(Y)
    elif busy.startswith('ground'):
        pad.press(RB)
        if busy.endswith('active'):
            pad.tick(.61)
    elif busy.startswith('roll'):
        pad.press(LB)
        if busy.endswith('active'):
            pad.tick(.61)
    else:
        pad.press(Y)
        pad.tick(.61)
        pad.press(Y)
    old = (pad.command.walking_variant, pad.command.sitstand_variant)
    mode = pad.command.policy_mode
    pad.press(button)
    assert (pad.command.walking_variant, pad.command.sitstand_variant) == old
    assert pad.command.policy_mode == mode
    pad.tick()
    assert not pad.command._walking_switch_pending


@pytest.mark.parametrize('directions', [(UP, DOWN), (LEFT, RIGHT), (UP, RIGHT), (LEFT, DOWN)])
def test_ambiguous_directions_are_ignored(directions):
    pad = Pad()
    pad.tick()
    pad.press(*directions)
    assert pad.command.policy_mode == 'walking'
    assert pad.command.walking_variant == 'original'
    assert pad.command.sitstand_variant == 'default'


@pytest.mark.parametrize('legacy,expected', [(Y, 'sit'), (RB, 'ground_pick'), (LB, 'roulade')])
@pytest.mark.parametrize('button', [UP, DOWN, LEFT, RIGHT])
def test_legacy_action_edges_win_over_simultaneous_dpad(legacy, expected, button):
    pad = Pad()
    pad.tick()
    pad.press(button, legacy)
    pad.tick(.61)
    assert pad.command.policy_mode == expected
    assert pad.command.walking_variant == 'original'
    assert pad.command.sitstand_variant == 'default'


@pytest.mark.parametrize('started', [UP, DOWN])
@pytest.mark.parametrize('fault', ['B', 'disconnect', 'stale', 'future'])
def test_b_and_link_faults_abort_pending_walk_or_roll_before_any_switch(started, fault):
    pad = Pad()
    pad.tick()
    pad.press(started)
    changes = {'buttons': (305,)} if fault == 'B' else (
        {'connected': False} if fault == 'disconnect' else
        {'timestamp': pad.now - 1 if fault == 'stale' else pad.now + 1})
    with pytest.raises(GamepadStop):
        pad.tick(**changes)
    assert pad.command.walking_variant == 'original'


def test_down_keeps_roll_timeout_and_b_priority_after_start():
    pad = Pad()
    pad.roll()
    with pytest.raises(GamepadStop, match='超时'):
        pad.tick(5.01)
    pad = Pad()
    pad.roll()
    with pytest.raises(GamepadStop, match='B键'):
        pad.tick(5.01, buttons=(305, UP, DOWN))


def test_reconnect_discards_buttons_received_while_disconnected():
    pad = Pad()
    pad.tick()
    pad.tick(connected=False)
    pad.press(UP, connected=True)
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=())
    pad.press(UP)
    assert pad.command.policy_mode == 'stopping'


class DpadDevice(FakeDevice):
    def __init__(self, hats):
        self.hats = hats
        self.hat_values = {16: 0, 17: 0}
        self.keys = set()
        self.events = []

    def capabilities(self, absinfo=False):
        return {3: [0, 1, 2, 5] + ([16, 17] if self.hats else []), 1: [UP, DOWN, LEFT, RIGHT]}

    def absinfo(self, code):
        if code in (16, 17):
            assert self.hats, 'Reader must not query unsupported hat axes'
            return SimpleNamespace(value=self.hat_values[code], min=-1, max=1)
        return super().absinfo(code)

    def active_keys(self):
        return sorted(self.keys)


@pytest.fixture
def reader_factory():
    readers = []
    def make(device):
        reader = GamepadReader(factory=lambda _: device)
        reader.halt.set()
        reader.thread.join(timeout=.5)
        readers.append(reader)
        return reader
    yield make
    for reader in readers:
        reader.close()


def event(kind, code, value):
    return SimpleNamespace(type=kind, code=code, value=value)


@pytest.mark.parametrize('button,axis,value', [(UP, 17, -1), (DOWN, 17, 1), (LEFT, 16, -1), (RIGHT, 16, 1)])
def test_hat_fast_taps_hold_release_and_new_press_normalize_to_button_counts(reader_factory, button, axis, value):
    device = DpadDevice(True)
    reader = reader_factory(device)
    device.events = [event(3, axis, value), event(3, axis, 0)]
    sample = reader._poll()
    assert sample.buttons == () and sample.button_press_counts[button] == 1
    device.hat_values[axis] = value
    device.events = [event(3, axis, value)]
    assert reader._poll().button_press_counts[button] == 2
    assert reader._poll().button_press_counts[button] == 2
    assert button in reader._poll().buttons
    device.hat_values[axis] = 0
    reader._poll()
    device.hat_values[axis] = value
    assert reader._poll().button_press_counts[button] == 3


@pytest.mark.parametrize('button', [UP, DOWN, LEFT, RIGHT])
def test_key_dpad_fast_taps_and_repeat_do_not_require_hats(reader_factory, button):
    device = DpadDevice(False)
    reader = reader_factory(device)
    device.events = [event(1, button, 1), event(1, button, 0)]
    assert reader._poll().button_press_counts[button] == 1
    device.events = [event(1, button, 1)]
    device.keys.add(button)
    assert reader._poll().button_press_counts[button] == 2
    device.events = [event(1, button, 2), event(1, button, 1)]
    assert reader._poll().button_press_counts[button] == 2
    assert reader._poll().button_press_counts[button] == 2


def test_drivers_reporting_both_hat_and_key_do_not_double_count(reader_factory):
    device = DpadDevice(True)
    reader = reader_factory(device)
    device.events = [event(3, 17, -1), event(1, UP, 1), event(3, 17, 0), event(1, UP, 0)]
    assert reader._poll().button_press_counts[UP] == 1
    assert reader._poll().button_press_counts[UP] == 1


def test_initial_held_hat_is_visible_but_suite_does_not_execute_it(reader_factory):
    device = DpadDevice(True)
    device.hat_values[17] = -1
    reader = reader_factory(device)
    sample = reader.latest()
    assert UP in sample.buttons and sample.button_press_counts[UP] == 1
    command = SuiteGamepadCommand(reader, clock=lambda: sample.timestamp)
    assert command() == ZERO
    assert command.policy_mode == 'walking'
    assert command.walking_variant == 'original'
