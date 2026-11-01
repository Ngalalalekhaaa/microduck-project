"""The encoder/IMU HOME family never obtains motor load or hidden offsets."""

from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.home_calibration_actions import HomeCalibrationActionCfg
from mjlab_microduck.tasks.home_calibration_runner import home_calibration_metadata
from mjlab_microduck.tasks.microduck_home_calibration_env_cfg import (
    NO_CURRENT_TASK_ID, TASK_ID, make_microduck_home_calibration_env_cfg,
)
from test_home_calibration import HOME, _Robot, _quiet_reset


def environment(num_envs=3, zero_probability=.1):
    cfg = load_env_cfg(NO_CURRENT_TASK_ID)
    robot = _Robot(num_envs)
    env = SimpleNamespace(
        scene={"robot": robot}, device="cpu", num_envs=num_envs,
        step_dt=.05, common_step_counter=0, extras={},
    )
    cfg.actions["home_calibration"].zero_bias_probability = zero_probability
    action = cfg.actions["home_calibration"].build(env)
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    _quiet_reset(env)
    return env, robot, action, cfg


def no_proxy(*args, **kwargs):
    raise AssertionError("No-current sensor frames must never read motor load")


def test_registry_keeps_v1_and_v2_contracts_separate():
    old = load_env_cfg(TASK_ID)
    new = load_env_cfg(NO_CURRENT_TASK_ID)
    assert old.actions["home_calibration"].observation_frame_dim == 49
    assert old.actions["home_calibration"].zero_bias_probability == 0
    action = new.actions["home_calibration"]
    assert action.observation_frame_dim == 34
    assert action.zero_bias_probability == .1
    assert action.delta_limit_rad == math.radians(.25)
    assert action.offset_limit_rad == math.radians(8)
    assert action.settle_seconds == .5
    assert action.bias_range_rad == (-math.radians(3), math.radians(3))
    assert new.decimation * new.sim.mujoco.timestep == .05
    for group in ("actor", "critic"):
        assert new.observations[group].terms["sensor_frame"].func is mdp.home_calibration_no_current_observation
        assert old.observations[group].terms["sensor_frame"].func is mdp.home_calibration_observation
    assert set(new.observations["critic"].terms) == set(old.observations["critic"].terms)
    assert load_rl_cfg(NO_CURRENT_TASK_ID).experiment_name == "microduck_hd1910_home_calibration_no_current"
    assert load_rl_cfg(NO_CURRENT_TASK_ID).experiment_name != load_rl_cfg(TASK_ID).experiment_name
    for profile in ("train", "nominal", "stress"):
        cfg = make_microduck_home_calibration_env_cfg(play=True, evaluation_profile=profile,
                                                     include_current=False)
        assert cfg.observations["actor"].history_length == 8
        assert cfg.actions["home_calibration"].observation_frame_dim == 34


def test_no_current_frame_has_only_reported_encoder_imu_and_known_offset(monkeypatch):
    monkeypatch.setattr(mdp, "home_calibration_motor_current_proxy", no_proxy)
    env, robot, action, cfg = environment()
    action.bias.copy_(torch.tensor([0., .02, -.03])[:, None].repeat(1, 14))
    robot.data.joint_pos.copy_(HOME - action.bias)
    robot.data.root_link_ang_vel_b.copy_(torch.tensor([.1, -.2, .3]).repeat(3, 1))
    frame = mdp.home_calibration_no_current_observation(env)
    assert frame.shape == (3, 34)
    expected_q = torch.round(HOME / (math.tau / 4096)) * (math.tau / 4096) - HOME
    torch.testing.assert_close(frame[:, :14], expected_q.repeat(3, 1), atol=1e-7, rtol=0)
    torch.testing.assert_close(frame[:, 14:17], robot.data.projected_gravity_b)
    torch.testing.assert_close(frame[:, 17:20], robot.data.root_link_ang_vel_b)
    torch.testing.assert_close(frame[:, 20:34], action.command_offset)
    # Identical available sensors can conceal different physical angles.
    torch.testing.assert_close(frame[0], frame[1])
    torch.testing.assert_close(frame[0], frame[2])
    error = mdp.home_calibration_pose_error(env)
    assert error[0] == 0 and error[1] > 0 and error[2] > 0
    critic_sensor = cfg.observations["critic"].terms["sensor_frame"].func(env)
    torch.testing.assert_close(critic_sensor, frame)
    # Cache outputs are independent snapshots, and no actuator is even needed.
    robot.actuators = None
    frame.fill_(42.)
    assert not torch.any(mdp.home_calibration_no_current_observation(env) == 42.)


def test_no_current_history_caching_and_selected_reset(monkeypatch):
    monkeypatch.setattr(mdp, "home_calibration_motor_current_proxy", no_proxy)
    env, robot, action, cfg = environment()
    _quiet_reset(env, gyro_noise_std=.01)
    manager = ObservationManager({"actor": deepcopy(cfg.observations["actor"])}, env)
    first = manager.compute(update_history=True)["actor"].reshape(3, 8, 34)
    torch.testing.assert_close(first, first[:, :1].repeat(1, 8, 1))
    cached = mdp.home_calibration_no_current_observation(env)
    torch.testing.assert_close(cached, mdp.home_calibration_no_current_observation(env))
    env.common_step_counter += 1
    robot.data.joint_pos += .02
    before = manager.compute(update_history=True)["actor"].reshape(3, 8, 34)
    torch.testing.assert_close(before[:, :-1], first[:, 1:])
    assert not torch.equal(before[:, -1], before[:, -2])
    _quiet_reset(env, torch.tensor([1]))
    manager.reset(torch.tensor([1]))
    after = manager.compute(update_history=True)["actor"].reshape(3, 8, 34)
    torch.testing.assert_close(after[1], after[1, -1:].repeat(8, 1))
    torch.testing.assert_close(after[[0, 2], :-1], before[[0, 2], 1:])
    torch.testing.assert_close(after[[0, 2], -1], before[[0, 2], -1])
    assert action.observation_cache.shape == (3, 34)


def test_no_current_metadata_matches_live_272_dim_history():
    env, _, action, cfg = environment()
    env.observation_manager = ObservationManager({"actor": deepcopy(cfg.observations["actor"])}, env)
    metadata = home_calibration_metadata(env, "model_0.pt")
    assert metadata["task_id"] == NO_CURRENT_TASK_ID
    assert metadata["policy_family"] == "hd1910_home_calibration_no_current_v2"
    assert metadata["actor_frame_dim"] == 34
    assert metadata["actor_dim"] == 272
    assert metadata["history_length"] == 8
    assert metadata["load_signal"] == metadata["current_input"] == "none"
    assert metadata["role"] == "simulation_candidate; encoder_and_imu_only_home_correction"
    assert not any(k in metadata for k in ("current_normalization_epsilon_a", "current_sign", "current_normalization"))
    assert [(x["name"], x["start"], x["size"]) for x in json.loads(metadata["observation_schema_json"])] == [
        ("reported_joint_position_minus_home", 0, 14), ("projected_gravity", 14, 3),
        ("body_gyro", 17, 3), ("command_offset_from_home", 20, 14),
    ]
    assert metadata["joint_names"] == list(action.joint_names)
    assert json.loads(json.dumps(metadata)) == metadata


def test_zero_bias_bucket_is_per_robot_and_preserves_unselected_worlds():
    torch.manual_seed(481)
    env, robot, action, _ = environment(num_envs=128, zero_probability=.1)
    all_zero = torch.all(action.bias == 0, dim=1)
    assert 3 <= int(all_zero.sum()) <= 30
    torch.testing.assert_close(robot.data.joint_pos + action.bias, HOME.repeat(128, 1))
    before = action.bias.clone()
    action.cfg.zero_bias_probability = 1.
    _quiet_reset(env, torch.tensor([0, 2]))
    assert torch.all(action.bias[[0, 2]] == 0)
    torch.testing.assert_close(action.bias[3:], before[3:])
    torch.testing.assert_close(action.bias[1], before[1])
    torch.testing.assert_close(robot.data.joint_pos[[0, 2]], HOME.repeat(2, 1))
    torch.testing.assert_close(action.processed_actions[[0, 2]], HOME.repeat(2, 1))


@pytest.mark.parametrize("probability", [-.1, 1.1, float("nan")])
def test_zero_bias_probability_validation(probability):
    with pytest.raises(ValueError, match="zero_bias_probability"):
        HomeCalibrationActionCfg(zero_bias_probability=probability)
