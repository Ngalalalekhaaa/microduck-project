"""Exercise real ONNX contracts and behavior dispatch without robot writes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.gamepad import GamepadStop
from microduck_deploy.policy import HOME, DEFAULT_MODEL_PATH, OnnxPolicy, PolicyContractError, build_obs61
from microduck_deploy.runtime import StopRequested, check_posture
from test_runtime import Policy, calibrated_config, rig


def observation(command):
    return build_obs61(np.zeros(3), [0, 0, -1], HOME, np.zeros(14), np.zeros(14), twist=command)


def test_ground_model_requires_explicit_contract_and_accepts_complete_phase():
    model = DEFAULT_MODEL_PATH.parent / 'hd1910_ground_pick_2249.onnx'
    with pytest.raises(PolicyContractError, match='command_names'):
        OnnxPolicy(model)
    with pytest.raises(PolicyContractError, match='command_names'):
        OnnxPolicy(model, policy_kind='sitstand')
    policy = OnnxPolicy(model, policy_kind='ground_pick')
    assert policy.session.get_providers() == ['CPUExecutionProvider']
    for phase in (0, .375, .425, .7, .8, .995, 1):
        value = policy.infer(observation([np.cos(2*np.pi*phase), np.sin(2*np.pi*phase), 0]))
        assert value.shape == (14,) and value.dtype == np.float32
        assert np.isfinite(value).all()


def test_new_sitstand_model_accepts_both_flags_and_retains_walking_contract():
    model = DEFAULT_MODEL_PATH.parent / 'hd1910_sitstand_1699.onnx'
    policy = OnnxPolicy(model, policy_kind='sitstand')
    stand, sit = [policy.infer(observation([flag, 0, 0])) for flag in (0, 1)]
    assert not np.allclose(stand, sit)
    with pytest.raises(PolicyContractError, match='command_names'):
        OnnxPolicy(model, policy_kind='ground_pick')


class Commands:
    def __init__(self, modes, command=(0., 1., 0.)):
        self.modes = modes
        self.index = 0
        self.policy_mode = 'walking'
        self.ground_pick_command = command

    def __call__(self):
        mode = self.modes[min(self.index, len(self.modes)-1)]
        self.index += 1
        if mode == 'B':
            raise GamepadStop('B')
        self.policy_mode = mode
        return (.3, .2, .5)


def test_ground_phase_bypasses_velocity_ramp_and_preserves_raw_action_history():
    with rig() as r:
        r.policy.output[:] = .01
        ground, sit = Policy(r.clock), Policy(r.clock)
        ground.output[:] = .02
        sit.output[:] = .03
        r.ctl.ground_pick_policy, r.ctl.sitstand_policy = ground, sit
        source = Commands(['walking', 'stopping', 'ground_pick', 'ground_pick', 'stand', 'walking'])
        r.ctl.execute(seconds=.22, motion=True, command_provider=source, command_ramp_s=.20)
        assert len(ground.observations) == 2
        np.testing.assert_array_equal(np.array(ground.observations)[:, 48:51], [[0, 1, 0], [0, 1, 0]])
        np.testing.assert_array_equal(np.array(ground.observations)[:, 51:], 0)
        np.testing.assert_allclose(ground.observations[0][34:48], .01)
        np.testing.assert_allclose(sit.observations[0][34:48], .02)
        np.testing.assert_allclose(r.policy.observations[2][34:48], .03)
        rows = [json.loads(line) for line in r.path.read_text().splitlines()]
        assert [x['policy_mode'] for x in rows if x['kind']=='sample'][:6] == source.modes


def test_missing_ground_policy_stops_before_behavior_dispatch():
    with rig() as r:
        with pytest.raises(StopRequested, match='未加载俯身权重'):
            r.ctl.execute(seconds=.1, motion=True, command_provider=Commands(['ground_pick']))
        assert not r.policy.observations
        assert not r.ctl.enabled


@pytest.mark.parametrize('phase', [(0, 0, 0), (1, 0, .1), (np.nan, 0, 0), (1, 0), (2, 0, 0)])
def test_malformed_phase_stops_before_ground_inference(phase):
    with rig() as r:
        ground = Policy(r.clock)
        r.ctl.ground_pick_policy = ground
        with pytest.raises(StopRequested, match='俯身相位指令无效'):
            r.ctl.execute(seconds=.1, motion=True, command_provider=Commands(['ground_pick'], phase))
        assert not ground.observations


def test_b_during_ground_uses_existing_unload_path():
    with rig() as r:
        ground = Policy(r.clock)
        r.ctl.ground_pick_policy = ground
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.1, motion=True, command_provider=Commands(['ground_pick', 'B']))
        assert len(ground.observations)==1
        assert not r.ctl.enabled
        assert any(x[0]=='torque' and x[2] is False for x in r.bus.events)


def test_intentional_ground_lean_has_only_a_mode_local_tilt_bound():
    cfg = calibrated_config()
    cfg['control']['ground_pick_tilt_limit_deg'] = 50
    theta = np.deg2rad(40)
    state = SimpleNamespace(projected_gravity=[np.sin(theta), 0, -np.cos(theta)], base_ang_vel=np.zeros(3))
    assert check_posture(HOME, state, cfg, policy_mode='ground_pick') == pytest.approx(40)
    for mode in ('walking', 'stopping', 'sit', 'stand'):
        with pytest.raises(StopRequested, match='机身倾斜'):
            check_posture(HOME, state, cfg, policy_mode=mode)
    theta = np.deg2rad(51)
    state.projected_gravity = [np.sin(theta), 0, -np.cos(theta)]
    with pytest.raises(StopRequested, match='机身倾斜'):
        check_posture(HOME, state, cfg, policy_mode='ground_pick')
