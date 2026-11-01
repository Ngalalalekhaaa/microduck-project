"""Calibration must learn physical correction, not exploit simulator-only zeros.

These tests exercise the real action and observation managers on CPU. The small
robot double supplies state, but never replaces the coordinate transforms,
incremental action, current observation, or history implementation under test.
"""

from copy import deepcopy
import math
import re
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.home_calibration_actions import HomeCalibrationActionCfg


TASK = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HOME = torch.tensor([
    0., -.0873, -.4579, -.0049, .4530, .3491, .3491, 0., 0.,
    0., .0873, .4579, .0049, -.4530,
])


class _Robot:
    def __init__(self, num_envs=3):
        self.joint_names = list(JOINT_NAMES)
        shape = (num_envs, len(self.joint_names))
        self.data = SimpleNamespace(
            joint_pos=HOME.repeat(num_envs, 1),
            joint_vel=torch.zeros(shape),
            default_joint_pos=HOME.repeat(num_envs, 1),
            default_joint_vel=torch.zeros(shape),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(num_envs, 1),
            root_link_ang_vel_b=torch.zeros(num_envs, 3),
            root_link_lin_vel_b=torch.zeros(num_envs, 3),
        )
        self.actuators = [SimpleNamespace(
            target_names=list(JOINT_NAMES),
            _prev_motor_torque=torch.zeros(shape),
            _bam_model=SimpleNamespace(kt=SimpleNamespace(value=.6237611235393989)),
        )]
        self.last_target = None

    def find_joints(self, patterns, preserve_order=False):
        del preserve_order
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [i for i, name in enumerate(self.joint_names)
               if any(re.fullmatch(pattern, name) for pattern in patterns)]
        return ids, [self.joint_names[i] for i in ids]

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        del joint_ids, env_ids
        self.last_target = target.clone()

    def write_joint_state_to_sim(self, position, velocity, env_ids=None, joint_ids=None):
        env_ids = slice(None) if env_ids is None else env_ids
        joint_ids = slice(None) if joint_ids is None else joint_ids
        if isinstance(env_ids, torch.Tensor) and not isinstance(joint_ids, slice):
            self.data.joint_pos[env_ids[:, None], joint_ids] = position
            self.data.joint_vel[env_ids[:, None], joint_ids] = velocity
        else:
            self.data.joint_pos[env_ids, joint_ids] = position
            self.data.joint_vel[env_ids, joint_ids] = velocity


def _environment(*, settle_seconds=0.):
    robot = _Robot()
    env = SimpleNamespace(
        scene={"robot": robot}, device="cpu", num_envs=3, step_dt=.05,
        common_step_counter=0, episode_length_buf=torch.zeros(3, dtype=torch.long),
        extras={},
    )
    action = HomeCalibrationActionCfg(
        entity_name="robot", settle_seconds=settle_seconds).build(env)
    env.action_manager = SimpleNamespace(
        get_term=lambda name: action if name == "home_calibration" else None,
        action=torch.zeros(3, 14),
    )
    return env, robot, action


def _advance(env):
    env.common_step_counter += 1
    env.episode_length_buf += 1


def _quiet_reset(env, env_ids=None, **overrides):
    params = dict(
        initial_position_noise_rad=0., current_gain_range=(1., 1.),
        current_bias_range_a=(0., 0.), current_noise_std_range_a=(0., 0.),
        current_quantization_range_a=(0., 0.), imu_mounting_error_deg=0.,
        gravity_noise_std=0., gyro_noise_std=0.,
    )
    params.update(overrides)
    mdp.reset_home_calibration(env, env_ids, **params)


def test_command_transform_uses_inverse_of_reported_encoder_bias():
    env, robot, action = _environment()
    bias = torch.linspace(-math.radians(3), math.radians(3), 14)
    action.bias.copy_(bias.repeat(3, 1))
    action.process_actions(torch.zeros(3, 14))
    action.apply_actions()
    # An apparent HOME command initially places the physical joints at HOME-b.
    torch.testing.assert_close(robot.last_target, HOME.repeat(3, 1) - action.bias)
    # Correcting the estimated offset by +b recovers physical HOME, not HOME-2b.
    action.command_offset.copy_(action.bias)
    action.process_actions(torch.zeros(3, 14))
    action.apply_actions()
    torch.testing.assert_close(robot.last_target, HOME.repeat(3, 1))


def test_increment_is_bounded_once_per_control_step_and_has_no_windup():
    _, robot, action = _environment()
    step = math.radians(.25)
    limit = math.radians(8)
    action.bias.zero_()
    action.process_actions(torch.full((3, 14), 100.))
    torch.testing.assert_close(action.command_offset, torch.full((3, 14), step))
    first = action.command_offset.clone()
    # Ten physics substeps must not integrate ten new controller actions.
    for _ in range(10):
        action.apply_actions()
    torch.testing.assert_close(action.command_offset, first)
    torch.testing.assert_close(robot.last_target, HOME.repeat(3, 1) + first)
    for _ in range(100):
        action.process_actions(torch.ones(3, 14))
    torch.testing.assert_close(action.command_offset, torch.full((3, 14), limit))
    action.process_actions(-torch.ones(3, 14))
    torch.testing.assert_close(action.command_offset, torch.full((3, 14), limit - step))
    for _ in range(100):
        action.process_actions(-torch.ones(3, 14))
    torch.testing.assert_close(action.command_offset, torch.full((3, 14), -limit))


def test_settling_window_ignores_action_before_enabling_bounded_correction():
    _, _, action = _environment(settle_seconds=.5)
    for _ in range(10):
        action.process_actions(torch.ones(3, 14))
        assert torch.all(action.command_offset == 0)
    action.process_actions(torch.ones(3, 14))
    torch.testing.assert_close(action.command_offset,
                               torch.full((3, 14), math.radians(.25)))
    action.reset(torch.tensor([1]))
    action.process_actions(torch.ones(3, 14))
    assert torch.all(action.command_offset[1] == 0)
    torch.testing.assert_close(action.command_offset[[0, 2]],
                               torch.full((2, 14), math.radians(.5)))


@pytest.mark.parametrize("selection", [torch.tensor([1]), slice(1, 2), None])
def test_action_reset_does_not_erase_other_worlds(selection):
    _, _, action = _environment()
    action.process_actions(torch.ones(3, 14))
    before = action.command_offset.clone()
    action.reset(selection)
    selected = slice(None) if selection is None else selection
    before[selected] = 0.
    torch.testing.assert_close(action.command_offset, before)


def test_reset_starts_at_apparent_home_instead_of_exposing_hidden_bias():
    env, robot, action = _environment()
    _quiet_reset(env)
    assert action.bias.abs().max() <= math.radians(3)
    assert torch.count_nonzero(action.bias) > 0
    torch.testing.assert_close(robot.data.joint_pos + action.bias,
                               HOME.repeat(3, 1))
    expected = torch.round(HOME * 4096 / math.tau) * math.tau / 4096 - HOME
    observed = mdp.home_calibration_observation(env)
    torch.testing.assert_close(observed[:, :14], expected.repeat(3, 1), atol=1e-7, rtol=0)
    # Two independent calls in the same control step must see the same sample.
    torch.testing.assert_close(mdp.home_calibration_observation(env), observed)


def test_actor_cannot_read_true_pose_or_bias_but_pose_cost_can():
    env, robot, action = _environment()
    _quiet_reset(env)
    action.bias.zero_()
    action.bias[1] = math.radians(2)
    action.bias[2] = -math.radians(2)
    robot.data.joint_pos.copy_(HOME - action.bias)
    action.command_offset.zero_()
    _advance(env)
    actor = mdp.home_calibration_observation(env)
    assert actor.shape == (3, 49)
    # The physical angles and hidden offsets differ, while every available
    # encoder/current/IMU/command signal has been held equal.
    torch.testing.assert_close(actor[0], actor[1])
    torch.testing.assert_close(actor[0], actor[2])
    error = mdp.home_calibration_true_joint_error(env)
    torch.testing.assert_close(error, -action.bias)
    cost = mdp.home_calibration_pose_error(env, scale_rad=math.radians(3))
    assert cost[0] == 0
    assert cost[1] > cost[0] and cost[2] > cost[0]
    torch.testing.assert_close(cost[1], cost[2])
    # Changing only the sensor bias cannot improve a physical-pose reward.
    action.bias.zero_()
    torch.testing.assert_close(mdp.home_calibration_pose_error(
        env, scale_rad=math.radians(3)), cost)


def test_current_observation_keeps_sign_and_absolute_scale():
    env, robot, _ = _environment()
    _quiet_reset(env)
    kt = robot.actuators[0]._bam_model.kt.value
    current = torch.zeros(3, 14)
    current[:, 0] = .1
    current[:, 1] = -.2
    robot.actuators[0]._prev_motor_torque.copy_(current * kt)
    _advance(env)
    frame = mdp.home_calibration_observation(env)
    assert torch.all(frame[:, 14] > 0) and torch.all(frame[:, 15] < 0)
    torch.testing.assert_close(frame[:, 28], torch.full((3,), .3))
    assert torch.all(frame[:, 16:28] == 0)
    robot.actuators[0]._prev_motor_torque *= 2
    _advance(env)
    doubled = mdp.home_calibration_observation(env)
    torch.testing.assert_close(doubled[:, 28], torch.full((3,), .6))
    # Zero current is well defined, with no ratio spikes or NaNs.
    robot.actuators[0]._prev_motor_torque.zero_()
    _advance(env)
    zero = mdp.home_calibration_observation(env)
    assert torch.isfinite(zero).all()
    assert torch.all(zero[:, 14:29] == 0)


def test_noisy_current_is_sampled_once_per_step_and_return_is_not_cache_alias():
    env, robot, _ = _environment()
    _quiet_reset(env, current_noise_std_range_a=(.01, .01))
    torch.manual_seed(13)
    first = mdp.home_calibration_observation(env)
    second = mdp.home_calibration_observation(env)
    torch.testing.assert_close(first, second)
    second.fill_(1000.)
    torch.testing.assert_close(mdp.home_calibration_observation(env), first)
    _advance(env)
    third = mdp.home_calibration_observation(env)
    assert not torch.equal(third[:, 14:29], first[:, 14:29])
    # A reset with no global-step advance refreshes only the selected world.
    _quiet_reset(env, torch.tensor([1]), current_noise_std_range_a=(.01, .01))
    robot.data.joint_pos[1, 0] += .02
    after_reset = mdp.home_calibration_observation(env)
    torch.testing.assert_close(after_reset[[0, 2]], third[[0, 2]])
    assert not torch.equal(after_reset[1], third[1])


def test_history_is_frame_major_and_partial_reset_reseeds_only_selected_world():
    env, robot, action = _environment()
    _quiet_reset(env)
    actor_cfg = deepcopy(load_env_cfg(TASK).observations["actor"])
    manager = ObservationManager({"actor": actor_cfg}, env)
    first = manager.compute(update_history=True)["actor"].reshape(3, 8, 49)
    assert first.shape == (3, 8, 49)
    torch.testing.assert_close(first, first[:, :1].repeat(1, 8, 1))
    for amount in (.01, .02, .03):
        _advance(env)
        robot.data.joint_pos.copy_(HOME - action.bias + amount)
        before = manager.compute(update_history=True)["actor"].reshape(3, 8, 49).clone()
    torch.testing.assert_close(before[:, -1], mdp.home_calibration_observation(env))
    assert not torch.equal(before[:, 0, :14], before[:, -1, :14])
    selected = torch.tensor([1])
    action.reset(selected)
    _quiet_reset(env, selected)
    manager.reset(selected)
    after = manager.compute(update_history=True)["actor"].reshape(3, 8, 49)
    fresh = mdp.home_calibration_observation(env)
    torch.testing.assert_close(after[1], fresh[1].repeat(8, 1))
    torch.testing.assert_close(after[[0, 2], :-1], before[[0, 2], 1:])
    torch.testing.assert_close(after[[0, 2], -1], before[[0, 2], -1])


def test_registered_calibration_family_keeps_walking_contract_separate():
    cfg = load_env_cfg(TASK)
    model = cfg.scene.entities["robot"].spec_fn().compile()
    assert model.nu == 14
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(.05)
    assert list(cfg.actions) == ["home_calibration"]
    assert isinstance(cfg.actions["home_calibration"], HomeCalibrationActionCfg)
    actor = cfg.observations["actor"]
    assert len(actor.terms) == 1
    assert actor.history_length == 8
    assert next(iter(actor.terms.values())).func is mdp.home_calibration_observation
    assert not cfg.commands
    assert "encoder_bias" not in cfg.events
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    assert any(term.func is mdp.reset_home_calibration for term in cfg.events.values())
    pose = [term for term in cfg.rewards.values()
            if term.func is mdp.home_calibration_pose_error]
    assert len(pose) == 1 and pose[0].weight < 0
    assert load_rl_cfg(TASK).actor.obs_normalization
    assert load_rl_cfg(TASK).experiment_name != load_rl_cfg(
        "Mjlab-Velocity-Flat-MicroDuck-HD1910").experiment_name
    walking = load_env_cfg("Mjlab-Velocity-Flat-MicroDuck-HD1910")
    assert walking.decimation * walking.sim.mujoco.timestep == pytest.approx(.02)
    assert list(walking.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
        "command", "head_command", "body_command",
    ]
