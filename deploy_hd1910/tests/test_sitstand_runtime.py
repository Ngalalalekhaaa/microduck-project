"""Exercise posture-policy routing through the actual hardware-free controller."""
import json

import numpy as np
import pytest

from microduck_deploy.gamepad import GamepadStop
from microduck_deploy.runtime import StopRequested, constrain_posture_target
from microduck_deploy.config import calibration_arrays
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks
from test_runtime import Policy, rig


class ModeCommands:
    def __init__(self, modes):
        self.modes = list(modes)
        self.index = 0
        self.policy_mode = 'walking'

    def __call__(self):
        mode = self.modes[min(self.index, len(self.modes)-1)]
        self.index += 1
        if mode == 'B':
            raise GamepadStop('B')
        self.policy_mode = mode
        # Deliberately nonzero in every mode: runtime itself must isolate
        # posture flags from sticks, velocity limits and startup ramping.
        return (.3, .2, .5)


def test_y_sequence_routes_binary_flags_and_preserves_shared_raw_history():
    with rig() as r:
        r.policy.output[:] = .01
        sit = Policy(r.clock)
        sit.output[:] = .02
        r.ctl.sitstand_policy = sit
        source = ModeCommands(['walking', 'stopping', 'sit', 'sit', 'stand', 'walking'])
        r.ctl.execute(seconds=.22, motion=True, command_provider=source, command_ramp_s=.20)
        assert len(sit.observations) == 3
        np.testing.assert_array_equal(np.array(sit.observations)[:, 48:51],
                                      [[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(np.array(sit.observations)[:, 51:], 0)
        np.testing.assert_allclose(sit.observations[0][34:48], .01)
        np.testing.assert_allclose(r.policy.observations[2][34:48], .02)
        np.testing.assert_array_equal(r.policy.observations[1][48:51], 0)
        rows = [json.loads(line) for line in r.path.read_text().splitlines()]
        samples = [row for row in rows if row['kind'] == 'sample']
        assert [row['policy_mode'] for row in samples[:6]] == source.modes
        assert [t['to'] for t in r.ctl.session['policy_transitions']] == [
            'stopping', 'sit', 'stand', 'walking']


def test_unavailable_sitstand_policy_stops_before_inference():
    with rig() as r:
        with pytest.raises(StopRequested, match='未加载站坐权重'):
            r.ctl.execute(seconds=.1, motion=True, command_provider=ModeCommands(['sit']))
        assert not r.policy.observations
        assert not r.ctl.enabled
        assert any(e[0] == 'torque' and e[2] is False for e in r.bus.events)


def test_b_during_sitting_keeps_existing_unload_path():
    with rig() as r:
        sit = Policy(r.clock)
        r.ctl.sitstand_policy = sit
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.2, motion=True,
                          command_provider=ModeCommands(['sit', 'B']))
        assert len(sit.observations) == 1
        assert not r.policy.observations
        assert not r.ctl.enabled
        assert any(e[0] == 'torque' and e[2] is False for e in r.bus.events)


def test_unknown_posture_mode_fails_without_dispatch():
    with rig() as r:
        with pytest.raises(StopRequested, match='姿态模式无效'):
            r.ctl.execute(seconds=.1, motion=True,
                          command_provider=ModeCommands(['unknown']))
        assert not r.policy.observations


def test_small_posture_overshoot_saturates_encoder_without_wrapping_or_recalibration():
    with rig() as r:
        zero, direction = calibration_arrays(r.cfg)
        target = HOME.copy()
        target[12] = direction[12] * (4097-zero[12]) * 2*np.pi/4096
        limited, names = constrain_posture_target(target, r.cfg)
        assert names == ['right_knee']
        assert q_to_ticks(limited, zero, direction)[12] == 4095
        assert q_to_ticks(HOME, zero, direction).shape == (14,)
        np.testing.assert_array_equal(calibration_arrays(r.cfg)[0], zero)
        np.testing.assert_array_equal(calibration_arrays(r.cfg)[1], direction)
        np.testing.assert_array_equal(target[:12], limited[:12])
        target[12] += direction[12]*np.deg2rad(6)
        with pytest.raises(StopRequested, match='超过5°'):
            constrain_posture_target(target, r.cfg)


@pytest.mark.parametrize('index,raw_tick', [(3, -80.), (12, 4175.)])
def test_large_posture_overshoot_identifies_joint_id_raw_ticks_and_degrees(index, raw_tick):
    with rig() as r:
        zero, direction = calibration_arrays(r.cfg)
        target = HOME.copy()
        target[index] = direction[index] * (raw_tick-zero[index]) * 2*np.pi/4096
        with pytest.raises(StopRequested) as caught:
            constrain_posture_target(target, r.cfg)
        message = str(caught.value)
        assert JOINT_NAMES[index] in message
        assert f'ID{r.cfg["joints"][index]["id"]}' in message
        assert f'原始目标{raw_tick:.2f}ticks' in message
        assert '越界7.03°' in message
        assert '超过5°' in message and '目标未发送' in message


def test_posture_encoder_threshold_remains_five_degrees():
    with rig() as r:
        zero, direction = calibration_arrays(r.cfg)
        target = HOME.copy()
        at_edge = direction[12] * (4095-zero[12]) * 2*np.pi/4096
        target[12] = at_edge + direction[12] * np.deg2rad(4.9)
        limited, names = constrain_posture_target(target, r.cfg)
        assert names == ['right_knee']
        assert q_to_ticks(limited, zero, direction)[12] == 4095
        target[12] = at_edge + direction[12] * np.deg2rad(5.1)
        with pytest.raises(StopRequested, match='越界5.10°'):
            constrain_posture_target(target, r.cfg)


@pytest.mark.parametrize('mode', ['sit', 'stand'])
@pytest.mark.parametrize('motion', [True, False])
def test_posture_constraint_rejection_persists_full_step_without_sending(mode, motion):
    with rig() as r:
        r.policy.output[:] = .01
        sit = Policy(r.clock)
        target = HOME.copy()
        target[12] = r.bus.direction[12] * (4175-r.bus.zero[12]) * 2*np.pi/4096
        sit.output[:] = target-HOME
        r.ctl.sitstand_policy = sit
        with pytest.raises(StopRequested, match='right_knee.*原始目标4175.00ticks.*越界7.03°'):
            r.ctl.execute(seconds=.2, motion=motion,
                          command_provider=ModeCommands(['walking', mode]))

        assert len(r.policy.observations) == len(sit.observations) == 1
        fault = r.ctl.session['rejected_policy_step']
        assert fault['policy_mode'] == mode
        assert fault['sent'] is False
        assert fault['t'] > 0
        assert np.shape(fault['q']) == np.shape(fault['qd']) == (14,)
        np.testing.assert_array_equal(fault['obs'], sit.observations[0])
        np.testing.assert_array_equal(fault['observation'], sit.observations[0])
        np.testing.assert_array_equal(fault['action'], sit.actions[0])
        np.testing.assert_array_equal(fault['raw_target'], HOME+sit.actions[0])
        np.testing.assert_array_equal(fault['target'], fault['raw_target'])
        np.testing.assert_array_equal(fault['previous_action'], r.policy.actions[0])
        np.testing.assert_array_equal(fault['previous_target'], HOME+r.policy.actions[0])
        np.testing.assert_array_equal(fault['command'], [int(mode == 'sit'), 0, 0])

        # The session JSON must retain the rejected step even though no sample
        # line is emitted for this failed inference/constraint cycle.
        saved = json.loads(r.path.with_suffix('.session.json').read_text())
        assert saved['status'] == 'stopped'
        persisted = saved['rejected_policy_step']
        assert persisted['policy_mode'] == mode and persisted['sent'] is False
        np.testing.assert_array_equal(persisted['raw_target'], fault['raw_target'])
        for key in ('q', 'qd', 'obs', 'action', 'previous_action', 'previous_target', 'command'):
            assert key in persisted

        sent = [event[2] for event in r.bus.events if event[0] == 'positions']
        if motion:
            home_ticks = q_to_ticks(HOME, r.bus.zero, r.bus.direction)
            first_ticks = q_to_ticks(HOME+r.policy.actions[0], r.bus.zero, r.bus.direction)
            allowed = [dict(zip(r.bus.ids, ticks)) for ticks in (home_ticks, first_ticks)]
            assert sent and all(positions in allowed for positions in sent)
        else:
            assert sent == []
        assert not r.ctl.enabled
