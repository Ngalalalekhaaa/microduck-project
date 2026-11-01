"""Exercise the real 50 Hz controller with a fifteenth servo on its existing bus."""
import copy
import dataclasses
import io
import os
import threading
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from microduck_deploy import jaw, runtime
from microduck_deploy.gamepad import GamepadStop
from microduck_deploy.jaw import JawCommandReader, JawError, SOFT_GAINS
from microduck_deploy.servo import ServoFeedback
from test_runtime import Bus, Clock, Imu, Policy, TimedEvent, calibrated_config


class FifteenServoBus(Bus):
    """Extend the existing runtime fake, retaining a separate torque state per ID."""
    def __init__(self, cfg, clock):
        super().__init__(cfg, clock)
        self.motion_ids = self.ids + [15]
        self.positions[15] = 344
        self.torques = dict.fromkeys(self.motion_ids, 0)
        self.gains = {sid: {'p': 6, 'd': 20, 'i': 0} for sid in self.motion_ids}
        self.body_gains = {'p': 6, 'd': 20, 'i': 0}
        self.bad_jaw_goal = False
        self.feedback_fault = None
        self.policy = None

    def assert_ids(self, ids):
        assert list(ids) in (self.ids, self.motion_ids, [15])

    def read_configuration(self, sid):
        self.events.append(('configuration', self.clock.now, sid))
        return {'torque': self.torques[sid], 'mode': self.mode,
                'angle_resolution': self.angle_resolution,
                'position_min': 0, 'position_max': 4095,
                'ram_gains': copy.deepcopy(self.gains[sid])}

    def read_feedback(self, sid):
        assert sid == 15
        self.clock.advance(.001)
        self.events.append(('jaw_read', self.clock.now))
        return ServoFeedback(15, self.positions[15], 0, 0, 7.4, 30, 0, 0,
                             self.positions[15], 0, self.clock.now)

    def read_feedback_many(self, ids, sync):
        assert list(ids) == self.motion_ids and sync
        values = super().read_feedback_many(ids, sync)
        self.events[-1] += (list(ids),)
        if self.feedback_fault is not None and self.policy.observations:
            values[15] = dataclasses.replace(values[15], **self.feedback_fault)
        return values

    def read(self, sid, address, length):
        if (address, length) == (40, 1):
            self.events.append(('torque_readback', self.clock.now, sid))
            return bytes((self.torques[sid],))
        assert (address, length) == (42, 2)
        self.events.append(('goal_readback', self.clock.now, sid))
        return int(self.positions[sid] + int(self.bad_jaw_goal and sid == 15)).to_bytes(2, 'little')

    def set_torque(self, ids, enable, verify):
        self.assert_ids(ids)
        self.events.append(('torque', self.clock.now, enable, verify, list(ids)))
        if enable and self.fail_enable:
            raise OSError('mock partial torque-enable failure')
        if not enable and self.disable_failures:
            self.disable_failures -= 1
            raise OSError('mock torque-disable timeout')
        for sid in ids:
            self.torques[sid] = int(enable)

    def set_pid_ram(self, ids, p, d, i, *, verify=True):
        assert all(self.torques[sid] == 0 for sid in ids)
        self.events.append(('gains', self.clock.now, list(ids), (p, d, i)))
        for sid in ids:
            self.gains[sid] = {'p': p, 'd': d, 'i': i}


class StageCommands:
    """Issue X once in HOME and once in policy, preserving the parent's state."""
    def __init__(self):
        self.controller = None
        self.sent = set()
        self.polls = []

    def poll(self):
        stage = self.controller.session['status']
        self.polls.append(stage)
        if stage in self.sent:
            return None
        self.sent.add(stage)
        return stage != 'running'


@contextmanager
def jaw_rig(tmp_path, *, source=None, body_torque=0, mouth_torque=0):
    cfg = calibrated_config()
    cfg['jaw'] = {'enabled': True, 'id': 15, 'mapping_verified': True,
                  'closed_tick': 344, 'open_tick': 51, 'transition_seconds': .8}
    clock = Clock()
    bus, imu, policy = FifteenServoBus(cfg, clock), Imu(clock), Policy(clock)
    bus.policy = policy
    for sid in bus.ids:
        bus.torques[sid] = body_torque
        if body_torque:
            bus.gains[sid] = dict(SOFT_GAINS)
    bus.torques[15] = mouth_torque
    if mouth_torque:
        bus.gains[15] = dict(SOFT_GAINS)
    source = StageCommands() if source is None else source
    ctl = runtime.Controller(cfg, bus=bus, imu_reader=imu, policy=policy, jaw_source=source)
    if isinstance(source, StageCommands):
        source.controller = ctl
    ctl.stop = TimedEvent(clock)
    actual_jaw = jaw.JawController
    # The controller's production clock defaults to real monotonic time. Inject
    # the existing simulated runtime clock through the actual mouth class.
    def make_jaw(*args, **kwargs):
        return actual_jaw(*args, **kwargs, clock=clock.monotonic)
    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime.time, 'monotonic', clock.monotonic))
        stack.enter_context(patch.object(runtime, 'log_path', return_value=tmp_path / 'run.jsonl'))
        stack.enter_context(patch.object(ctl, '_install_signals'))
        stack.enter_context(patch.object(runtime.sys, 'stdin', SimpleNamespace(isatty=lambda: False)))
        stack.enter_context(patch.object(runtime, 'threading', SimpleNamespace(
            Thread=Mock(), current_thread=threading.current_thread,
            main_thread=threading.main_thread)))
        factory = stack.enter_context(patch.object(runtime, 'ServoBus', return_value=bus))
        stack.enter_context(patch.object(jaw, 'JawController', side_effect=make_jaw))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        yield SimpleNamespace(cfg=cfg, bus=bus, clock=clock, imu=imu, policy=policy,
                              ctl=ctl, source=source, factory=factory)


def releases(bus):
    return [event for event in bus.events if event[0] == 'torque' and event[2] is False]


def test_home_and_policy_share_fifteenth_servo_without_changing_model_dimensions(tmp_path):
    with jaw_rig(tmp_path) as r:
        before = copy.deepcopy(r.cfg)
        result = r.ctl.execute(seconds=.12, motion=True)
        assert result['samples'] >= 2
        assert r.cfg == before
        assert r.ctl.ids == r.bus.ids and len(r.ctl.ids) == 14
        assert r.ctl.motion_ids == r.bus.motion_ids and len(r.ctl.motion_ids) == 15
        assert r.policy.observations and all(obs.shape == (61,) for obs in r.policy.observations)
        assert all(action.shape == (14,) for action in r.policy.actions)
        assert all(len(sample['q']) == len(sample['target']) == 14 for sample in r.ctl.session['home_samples'])
        assert set(r.ctl.session['last_measurement']['feedback']) == set(r.bus.motion_ids)
        assert {'initializing', 'running'} <= set(r.source.polls)
        combined = [event for event in r.bus.events
                    if event[0] == 'positions' and set(event[2]) == set(r.bus.motion_ids)]
        assert combined and combined[0][1] < r.policy.times[0]
        assert any(event[1] >= r.policy.times[0] for event in combined)
        # Only the actual existing bus was used, even for first-command prepare.
        r.factory.assert_not_called()
        batch_reads = [event for event in r.bus.events if event[0] == 'read']
        assert all(event[2] == r.bus.motion_ids for event in batch_reads)
        # The only independent mouth reads are its two startup checks.
        assert len([event for event in r.bus.events if event[0] == 'jaw_read']) == 2
        assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))


def test_first_x_in_home_preloads_measured_mouth_goal_before_confirmed_enable(tmp_path):
    with jaw_rig(tmp_path) as r:
        r.bus.positions[15] = 342
        r.ctl.execute(seconds=.06, motion=True)
        mouth_preload = next(i for i, event in enumerate(r.bus.events)
                             if event[0] == 'positions' and set(event[2]) == {15})
        assert r.bus.events[mouth_preload][2] == {15: 342}
        assert r.bus.events[mouth_preload + 1][0] == 'goal_readback'
        assert r.bus.events[mouth_preload + 1][2] == 15
        enable = next(i for i, event in enumerate(r.bus.events)
                      if event[0] == 'torque' and event[2] is True and event[4] == [15])
        assert enable > mouth_preload + 1
        assert r.bus.events[enable][3] is True
        assert r.bus.events[enable][1] < r.policy.times[0]


def test_first_x_during_policy_does_not_move_mouth_during_home(tmp_path):
    with jaw_rig(tmp_path) as r:
        opened = False
        def command():
            nonlocal opened
            if r.ctl.session['status'] == 'running' and not opened:
                opened = True
                return True
            return None
        r.source.poll = command
        r.ctl.execute(seconds=.12, motion=True)
        jaw_writes = [event for event in r.bus.events
                      if event[0] == 'positions' and 15 in event[2]]
        assert jaw_writes and jaw_writes[0][1] >= r.policy.times[0]
        assert jaw_writes[0][2] == {15: 344}
        assert any(len(event[2]) == 15 for event in jaw_writes[1:])
        assert all(obs.shape == (61,) for obs in r.policy.observations)


def test_parent_pipe_x_command_is_consumed_in_real_home_loop(tmp_path):
    read_fd, write_fd = os.pipe()
    source = JawCommandReader(read_fd)
    os.write(write_fd, b'1\n')
    try:
        with jaw_rig(tmp_path, source=source) as r:
            r.ctl.execute(seconds=.06, motion=True)
            assert any(event[0] == 'positions' and 15 in event[2]
                       and len(event[2]) == 15 for event in r.bus.events)
            assert all(obs.shape == (61,) for obs in r.policy.observations)
    finally:
        source.close()
        os.close(write_fd)


@pytest.mark.parametrize('body_torque,mouth_torque', [(0, 1), (1, 0), (1, 1)])
def test_mouth_torque_is_independent_of_body_preflight_takeover(tmp_path, body_torque, mouth_torque):
    with jaw_rig(tmp_path, body_torque=body_torque, mouth_torque=mouth_torque) as r:
        result = r.ctl.execute(seconds=.06, motion=True, take_over_hold=bool(body_torque))
        assert result['samples'] > 0
        before = r.ctl.session['servo_configuration_before_motion']
        assert len(before) == 14 and 15 not in before
        assert {row['torque'] for row in before.values()} == {body_torque}
        assert set(r.bus.torques.values()) == {0}
        if mouth_torque:
            assert not any(event[0] == 'torque' and event[2] is True
                           and event[4] == [15] for event in r.bus.events)


def test_b_stop_releases_all_body_joints_and_mouth(tmp_path):
    with jaw_rig(tmp_path) as r:
        calls = []
        def command():
            calls.append(1)
            if len(calls) == 2:
                raise GamepadStop('Xbox B')
            return (0., 0., 0.)
        with pytest.raises(GamepadStop, match='Xbox B'):
            r.ctl.execute(seconds=.2, motion=True, command_provider=command)
        assert any(event[4] == r.bus.motion_ids and event[3] for event in releases(r.bus))
        assert set(r.bus.torques.values()) == {0}


@pytest.mark.parametrize('fault', [
    {'voltage_v': 3.9}, {'voltage_v': 8.5}, {'temperature_c': 60}, {'status': 1},
])
def test_fifteenth_servo_fault_is_checked_in_same_batch_and_unloads_every_servo(tmp_path, fault):
    with jaw_rig(tmp_path) as r:
        r.bus.feedback_fault = fault
        with pytest.raises(runtime.StopRequested):
            r.ctl.execute(seconds=.2, motion=True)
        assert 15 in r.ctl.session['fault_feedback']
        for field, value in fault.items():
            assert getattr(r.ctl.session['fault_feedback'][15], field) == value
        assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))
        assert set(r.bus.torques.values()) == {0}


def test_unacknowledged_mouth_preload_aborts_home_before_policy_and_unloads(tmp_path):
    with jaw_rig(tmp_path) as r:
        r.bus.bad_jaw_goal = True
        with pytest.raises(JawError, match='预装'):
            r.ctl.execute(seconds=.1, motion=True)
        assert r.policy.observations == []
        assert not any(event[0] == 'torque' and event[2] is True
                       and event[4] == [15] for event in r.bus.events)
        assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))
        assert set(r.bus.torques.values()) == {0}


def test_parent_pipe_disconnection_stops_child_and_releases_fifteenth_servo(tmp_path):
    read_fd, write_fd = os.pipe()
    source = JawCommandReader(read_fd)
    os.close(write_fd)
    try:
        with jaw_rig(tmp_path, source=source) as r:
            with pytest.raises(JawError, match='父进程'):
                r.ctl.execute(seconds=.1, motion=True)
            assert r.policy.observations == []
            assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))
            assert set(r.bus.torques.values()) == {0}
    finally:
        source.close()


def test_parent_pipe_eof_discards_queued_x_before_any_mouth_enable(tmp_path):
    read_fd, write_fd = os.pipe()
    source = JawCommandReader(read_fd)
    os.write(write_fd, b'1\n')
    os.close(write_fd)
    try:
        with jaw_rig(tmp_path, source=source) as r:
            with pytest.raises(JawError, match='父进程'):
                r.ctl.execute(seconds=.1, motion=True)
            assert not any(event[0] == 'positions' and 15 in event[2]
                           for event in r.bus.events)
            assert not any(event[0] == 'torque' and event[2] is True
                           and 15 in event[4] for event in r.bus.events)
            assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))
            assert set(r.bus.torques.values()) == {0}
    finally:
        source.close()


def test_slow_first_mouth_prepare_rejects_stale_body_packet_and_unloads(tmp_path):
    with jaw_rig(tmp_path) as r:
        original_read = r.bus.read
        def slow_mouth_readback(sid, address, length):
            result = original_read(sid, address, length)
            if sid == 15 and (address, length) == (42, 2):
                r.clock.advance(.06)
            return result
        r.bus.read = slow_mouth_readback
        with pytest.raises(runtime.StopRequested, match='嘴部准备后.*超时'):
            r.ctl.execute(seconds=.1, motion=True)
        mouth_preload = next(i for i, event in enumerate(r.bus.events)
                             if event[0] == 'positions' and set(event[2]) == {15})
        assert r.bus.events[mouth_preload][2] == {15: 344}
        # The first current-angle mouth preload may be delivered. No subsequent
        # body trajectory packet may be sent after its slow preparation.
        assert not any(event[0] == 'positions' and set(event[2]) & set(r.bus.ids)
                       for event in r.bus.events[mouth_preload + 1:])
        assert r.policy.observations == []
        assert any(event[4] == r.bus.motion_ids for event in releases(r.bus))
        assert set(r.bus.torques.values()) == {0}


def test_watchdog_emergency_packet_contains_body_and_mouth_on_existing_bus(tmp_path):
    with jaw_rig(tmp_path) as r:
        r.ctl.enabled = True
        r.ctl.heartbeat = r.clock.now - .2
        r.ctl._watchdog()
        assert r.ctl.stop.is_set()
        assert releases(r.bus)[-1][3:] == (False, r.bus.motion_ids)
        r.factory.assert_not_called()
