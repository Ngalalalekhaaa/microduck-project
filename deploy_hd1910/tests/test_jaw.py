"""Mouth endpoints, press edges, inherited commands, and bus motion without hardware."""
import copy
import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from microduck_deploy.config import default_config
from microduck_deploy.gamepad import PadSample
from microduck_deploy.jaw import (
    JawButtonToggle, JawCommandReader, JawController, JawError,
    SOFT_GAINS, jaw_config, jaw_ids,
)
from microduck_deploy.servo import ServoFeedback


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now

    def advance(self, seconds=.02):
        self.now += seconds


def cfg():
    result = default_config()
    for sid, joint in enumerate(result['joints'], 1):
        joint['id'] = sid
    result['jaw'] = {'enabled': True, 'id': 15, 'mapping_verified': True,
                     'closed_tick': 344, 'open_tick': 51, 'transition_seconds': .8}
    return result


def sample(clock, count=0, **kwargs):
    return PadSample(clock(), True, {}, {}, (), button_press_counts={307: count}, **kwargs)


class Bus:
    def __init__(self, clock):
        self.clock = clock
        self.position, self.goal = 344, 2000
        self.conf = {'mode': 4, 'angle_resolution': 1, 'position_min': 0,
                     'position_max': 4095, 'torque': 0,
                     'ram_gains': {'p': 6, 'd': 20, 'i': 0}}
        self.events = []
        self.feedback_count = self.configuration_count = 0
        self.bad_goal_readback = False
        self.fail_preload = False
        self.voltage, self.temperature, self.status = 7.4, 30, 0
        self.soft_gain_write_allowed = False

    def read_configuration(self, sid):
        assert sid == 15
        self.configuration_count += 1
        return copy.deepcopy(self.conf)

    def feedback(self):
        return ServoFeedback(15, self.position, 0, 0, self.voltage,
                             self.temperature, self.status, 0, self.goal, 0, self.clock())

    def read_feedback(self, sid):
        assert sid == 15
        self.feedback_count += 1
        return self.feedback()

    def read(self, sid, address, size):
        assert sid == 15
        self.events.append(('read', address))
        if address == 40 and size == 1:
            return bytes((self.conf['torque'],))
        assert (address, size) == (42, 2)
        return int(self.goal + int(self.bad_goal_readback)).to_bytes(2, 'little')

    def sync_positions(self, targets):
        assert set(targets) == {15}
        self.events.append(('position', dict(targets)))
        self.goal = targets[15]
        if self.fail_preload:
            raise OSError('partial goal write')

    def set_pid_ram(self, ids, p, d, i, *, verify):
        assert ids == [15] and verify and self.soft_gain_write_allowed
        assert self.conf['torque'] == 0
        self.events.append(('gains', p, d, i))
        self.conf['ram_gains'] = {'p': p, 'd': d, 'i': i}

    def set_torque(self, ids, enabled, *, verify):
        assert ids == [15]
        self.events.append(('torque', enabled, verify))
        self.conf['torque'] = int(enabled)


def controller():
    clock = Clock()
    bus = Bus(clock)
    return JawController(cfg(), bus, clock=clock), bus, clock


def test_optional_or_disabled_mouth_performs_no_hardware_access():
    for configuration in (default_config(), dict(default_config(), jaw={'enabled': False})):
        bus = Mock()
        ctl = JawController(configuration, bus)
        assert jaw_ids(configuration) == []
        assert ctl.preflight() is None
        ctl.warmup()
        assert ctl.targets(command=True) == {}
        ctl.disable()
        assert bus.mock_calls == []


def test_configuration_preserves_all_fourteen_policy_joints():
    configuration = cfg()
    before = copy.deepcopy(configuration)
    settings = jaw_config(configuration)
    assert settings['button_code'] == 307
    assert jaw_ids(configuration) == [15]
    assert configuration == before
    assert len(configuration['joints']) == 14


@pytest.mark.parametrize('field,value', [
    ('enabled', 1), ('id', 1), ('id', True), ('id', 254),
    ('mapping_verified', False), ('closed_tick', -1), ('open_tick', 4096),
    ('open_tick', 344), ('open_tick', 344.), ('transition_seconds', 0),
    ('transition_seconds', float('nan')), ('transition_seconds', True),
    ('button_code', 305),
])
def test_invalid_or_unconfirmed_endpoints_never_reach_hardware(field, value):
    configuration = cfg()
    configuration['jaw'][field] = value
    bus = Mock()
    with pytest.raises(ValueError):
        JawController(configuration, bus)
    assert bus.mock_calls == []


def test_endpoints_do_not_wrap_through_zero():
    configuration = cfg()
    configuration['jaw'].update(closed_tick=4000, open_tick=50)
    with pytest.raises(ValueError, match='单圈'):
        jaw_config(configuration)


def test_x_is_new_press_toggle_with_held_initial_press_ignored():
    clock = Clock()
    reader = SimpleNamespace(latest=Mock(return_value=sample(clock, 1)))
    button = JawButtonToggle(reader, clock=clock)
    assert button.poll() is None
    reader.latest.return_value = sample(clock, 2)
    assert button.poll() is True
    assert button.poll() is None  # A held X never repeats.
    reader.latest.return_value = sample(clock, 3)
    assert button.poll() is False
    reader.latest.return_value = sample(clock, 5)
    assert button.poll() is False  # Two brief presses keep the final state.


@pytest.mark.parametrize('change', [
    {'stop_requested': True}, {'connected': False}, {'timestamp': 99.6},
])
def test_b_disconnect_and_stale_sample_discard_x(change):
    clock = Clock()
    reader = SimpleNamespace(latest=Mock(return_value=sample(clock)))
    button = JawButtonToggle(reader, clock=clock)
    assert button.poll() is None
    reader.latest.return_value = dataclasses.replace(sample(clock, 1), **change)
    assert button.poll() is None
    reader.latest.return_value = sample(clock, 1)
    assert button.poll() is None
    reader.latest.return_value = sample(clock, 2)
    assert button.poll() is True


def test_skipped_stage_keeps_state_and_baselines_old_x():
    clock = Clock()
    reader = SimpleNamespace(latest=Mock(return_value=sample(clock)))
    button = JawButtonToggle(reader, clock=clock)
    button.poll()
    reader.latest.return_value = sample(clock, 1)
    assert button.poll() is True
    reader.latest.return_value = sample(clock, 2)
    button.baseline()
    assert button.poll() is None
    reader.latest.return_value = sample(clock, 3)
    assert button.poll() is False
    button.reset()
    assert button.poll() is None
    reader.latest.return_value = sample(clock, 4)
    assert button.poll() is True


def test_explicit_hid_x_code_can_be_308():
    clock = Clock()
    reader = SimpleNamespace(latest=Mock(return_value=sample(clock)))
    button = JawButtonToggle(reader, button_code=308, clock=clock)
    button.poll()
    reader.latest.return_value = dataclasses.replace(sample(clock, 10), button_press_counts={307: 10, 308: 1})
    assert button.poll() is True


def test_command_pipe_preserves_partial_lines_and_takes_latest_state():
    read_fd, write_fd = os.pipe()
    command = JawCommandReader(read_fd)
    try:
        assert command.poll() is None
        os.write(write_fd, b'1')
        assert command.poll() is None
        os.write(write_fd, b'\n0\n1\n')
        assert command.poll() is True
        assert command.poll() is None
        os.write(write_fd, b'0\n')
        assert command.poll() is False
    finally:
        command.close()
        os.close(write_fd)
    command.close()


def test_command_pipe_eof_stops_motion_even_without_last_command():
    read_fd, write_fd = os.pipe()
    command = JawCommandReader(read_fd)
    os.close(write_fd)
    try:
        with pytest.raises(JawError, match='父进程'):
            command.poll()
    finally:
        command.close()


def test_last_pipe_command_is_discarded_when_parent_disconnects():
    read_fd, write_fd = os.pipe()
    command = JawCommandReader(read_fd)
    os.write(write_fd, b'1\n')
    os.close(write_fd)
    try:
        with pytest.raises(JawError, match='父进程'):
            command.poll()
    finally:
        command.close()


def test_invalid_pipe_command_cannot_arm_servo():
    read_fd, write_fd = os.pipe()
    command = JawCommandReader(read_fd)
    os.write(write_fd, b'OPEN\n')
    try:
        with pytest.raises(JawError, match='无效'):
            command.poll()
    finally:
        command.close()
        os.close(write_fd)


def test_preflight_reads_only_and_initial_none_is_no_action():
    ctl, bus, _ = controller()
    before = copy.deepcopy(bus.conf)
    result = ctl.preflight()
    assert result['position_ticks'] == 344
    assert ctl.targets() == {}
    assert bus.conf == before and bus.events == []
    assert not ctl.prepared and not ctl.active


@pytest.mark.parametrize('field,value', [
    ('mode', 0), ('angle_resolution', 2), ('position_min', 100),
    ('position_max', 343), ('torque', 2),
])
def test_incompatible_servo_configuration_rejects_without_writes(field, value):
    ctl, bus, _ = controller()
    bus.conf[field] = value
    with pytest.raises(JawError):
        ctl.preflight()
    assert bus.events == []


@pytest.mark.parametrize('field,value', [
    ('voltage', 3.9), ('voltage', 8.5), ('temperature', 60),
    ('status', 1), ('position', -1), ('position', 4096),
])
def test_bad_feedback_rejected_before_any_goal(field, value):
    ctl, bus, _ = controller()
    setattr(bus, field, value)
    with pytest.raises(JawError):
        ctl.preflight()
    assert bus.events == []


def test_first_command_preloads_current_position_before_torque_then_opens_gradually():
    ctl, bus, clock = controller()
    assert ctl.targets(command=True) == {15: 344}
    goal_index = next(i for i, event in enumerate(bus.events) if event[0] == 'position')
    enable_index = next(i for i, event in enumerate(bus.events) if event[:2] == ('torque', True))
    assert bus.events[goal_index] == ('position', {15: 344})
    assert bus.events[goal_index + 1] == ('read', 42)
    assert enable_index > goal_index + 1
    assert ctl.prepared and ctl.active
    targets = [344]
    for _ in range(40):
        clock.advance()
        target = ctl.targets()[15]
        targets.append(target)
        bus.sync_positions({15: target})
        bus.position = target
    assert targets[-1] == 51
    assert all(51 <= tick <= 344 for tick in targets)
    assert all(0 <= a - b <= 12 for a, b in zip(targets, targets[1:]))


def test_warmup_changes_only_ram_gains_and_speeds_first_x():
    ctl, bus, _ = controller()
    ctl.preflight()
    ctl.warmup()
    assert not ctl.prepared and not ctl.active
    assert bus.goal == 2000 and bus.conf['torque'] == 0
    assert bus.events == [('gains', 5, 0, 0)]
    assert not bus.soft_gain_write_allowed
    configurations = bus.configuration_count
    feedback = bus.feedback()
    reads = bus.feedback_count
    assert ctl.targets({15: feedback}, command=True) == {15: 344}
    assert bus.configuration_count == configurations
    assert bus.feedback_count == reads
    assert sum(event[0] == 'gains' for event in bus.events) == 1
    ctl.disable()
    assert bus.conf['ram_gains'] == {'p': 6, 'd': 20, 'i': 0}
    assert not bus.soft_gain_write_allowed


def test_same_batch_feedback_requires_no_independent_bus_read_after_prepare():
    ctl, bus, clock = controller()
    ctl.targets(command=True)
    reads = bus.feedback_count
    clock.advance()
    assert ctl.targets({15: bus.feedback()})[15] < 344
    assert bus.feedback_count == reads


def test_in_flight_reversal_is_continuous_and_returns_to_closed_endpoint():
    ctl, bus, clock = controller()
    ctl.targets(command=True)
    for _ in range(15):
        clock.advance()
        last = ctl.targets()[15]
    # A repeated state must not reset the running trajectory.
    assert ctl.targets(command=True)[15] == last
    assert ctl.targets(command=False)[15] == last
    reverse = []
    for _ in range(40):
        clock.advance()
        reverse.append(ctl.targets()[15])
    assert reverse[-1] == 344
    assert all(a <= b for a, b in zip(reverse, reverse[1:]))
    assert all(last <= tick <= 344 for tick in reverse)


def test_initial_hand_position_outside_endpoints_converges_without_clipping_jump():
    ctl, bus, clock = controller()
    bus.position = 350
    assert ctl.targets(command=True) == {15: 350}
    clock.advance()
    assert 344 < ctl.targets()[15] <= 350
    for _ in range(50):
        clock.advance()
        last = ctl.targets()[15]
    assert last == 51


@pytest.mark.parametrize('tick', [0, 377, 2000, 4000])
def test_first_command_rejects_position_far_from_calibrated_endpoints(tick):
    ctl, bus, _ = controller()
    bus.position = tick
    with pytest.raises(JawError, match='远离标定'):
        ctl.targets(command=True)
    assert not any(event[0] in ('position', 'torque') for event in bus.events)


def test_paused_loop_does_not_jump_to_endpoint():
    ctl, _, clock = controller()
    ctl.targets(command=True)
    clock.advance(5.)
    assert 340 <= ctl.targets()[15] <= 344


def test_live_torque_hold_is_taken_over_without_torque_off_or_gain_change():
    ctl, bus, _ = controller()
    bus.conf.update(torque=1, ram_gains=dict(SOFT_GAINS))
    assert ctl.targets(command=True) == {15: 344}
    assert not any(event[0] in ('torque', 'gains') for event in bus.events)
    ctl.disable()
    assert bus.events[-1] == ('torque', False, True)


def test_incompatible_live_hold_cannot_be_taken_over():
    ctl, bus, _ = controller()
    bus.conf['torque'] = 1
    with pytest.raises(JawError, match='P5'):
        ctl.preflight()
    assert bus.events == []


@pytest.mark.parametrize('failure', ['bad_goal_readback', 'fail_preload'])
def test_failed_or_unacknowledged_preload_is_not_armed_and_cleanup_unloads(failure):
    ctl, bus, _ = controller()
    setattr(bus, failure, True)
    with pytest.raises((JawError, OSError)):
        ctl.targets(command=True)
    assert ctl.active and not ctl.prepared
    assert not any(event[:2] == ('torque', True) for event in bus.events)
    ctl.disable()
    assert not ctl.active
    assert bus.conf['torque'] == 0
    assert bus.conf['ram_gains'] == {'p': 6, 'd': 20, 'i': 0}


def test_disable_can_defer_gain_restore_and_later_restore_only_fifteenth_servo():
    ctl, bus, _ = controller()
    ctl.warmup()
    ctl.disable(restore_gains=False)
    assert bus.conf['ram_gains'] == SOFT_GAINS
    ctl.disable(restore_gains=True)
    assert bus.conf['ram_gains'] == {'p': 6, 'd': 20, 'i': 0}


def test_stale_missing_and_wrong_id_synchronized_feedback_cannot_be_sent():
    ctl, bus, clock = controller()
    ctl.targets(command=True)
    feedback = bus.feedback()
    clock.advance(.041)
    with pytest.raises(JawError, match='过期'):
        ctl.targets({15: feedback})
    with pytest.raises(JawError, match='缺少'):
        ctl.targets({})
    with pytest.raises(JawError, match='ID'):
        ctl.targets(dataclasses.replace(bus.feedback(), servo_id=14))


def test_command_source_is_consumed_without_gamepad_reader_in_child():
    ctl, _, _ = controller()
    ctl.command_source = SimpleNamespace(poll=Mock(side_effect=[None, True, None]))
    assert ctl.targets() == {}
    assert ctl.targets() == {15: 344}
    assert ctl.targets() == {15: 344}
