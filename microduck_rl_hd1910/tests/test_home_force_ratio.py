"""A privileged torque-ratio objective must not leak measured torque to policy."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.observation_manager import ObservationManager
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.home_calibration_runner import home_calibration_metadata
from mjlab_microduck.tasks.microduck_home_force_ratio_env_cfg import (
    REFERENCE_JOINT_NAMES, TASK_ID, load_home_force_reference,
    make_microduck_home_force_ratio_env_cfg,
)
from test_home_calibration import HOME
from test_home_calibration_no_current import environment, no_proxy


TARGET = tuple([1 / 14] * 14)


@pytest.fixture
def reference(tmp_path):
    data = {
        "schema_version": 1,
        "reference_kind": "absolute_joint_actuator_torque_fraction",
        "gate_passed": True,
        "joint_names": list(REFERENCE_JOINT_NAMES),
        "home_angles_rad": HOME.tolist(),
        "target_joint_torque_fraction": list(TARGET),
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(data))
    return path, data


class Scene(dict):
    pass


def force_environment():
    env, robot, action, _ = environment()
    env.scene = Scene(env.scene)
    env.scene.sensors = {"feet_ground_contact": SimpleNamespace(
        data=SimpleNamespace(found=torch.ones(3, 2, 1, dtype=torch.bool)))}
    robot.data.qfrc_actuator = torch.full((3, 14), .1)
    return env, robot, action


def test_reference_loader_and_factory_keep_one_fixed_target(reference):
    path, data = reference
    assert load_home_force_reference(path) == data
    cfg = make_microduck_home_force_ratio_env_cfg(reference_path=path)
    assert cfg.actions["home_calibration"].observation_frame_dim == 34
    assert cfg.observations["actor"].history_length == 8
    for group in ("actor", "critic"):
        term = cfg.observations[group].terms["sensor_frame"]
        assert term.func is mdp.home_force_ratio_observation
        assert term.params["target_ratio"] == TARGET
    assert "true_home_tracking" not in cfg.rewards
    assert "true_home_error" not in cfg.rewards
    assert cfg.rewards["force_ratio_tracking"].weight == 4.
    assert cfg.rewards["force_ratio_error"].weight == -.25
    assert cfg.rewards["force_ratio_error"].params["scale"] == .03
    assert all(k in cfg.rewards for k in ("trunk_tilt", "joint_motion", "body_motion", "double_support", "failure"))


def test_registered_force_task_uses_measured_reference_artifact():
    reference = load_home_force_reference()
    assert reference["gate_passed"] is True
    assert reference["reference_kind"] == "absolute_joint_actuator_torque_fraction"
    target = tuple(reference["target_joint_torque_fraction"])
    cfg = load_env_cfg(TASK_ID)
    for group in ("actor", "critic"):
        assert cfg.observations[group].terms["sensor_frame"].params["target_ratio"] == target
    assert cfg.rewards["force_ratio_error"].params["target_ratio"] == target
    assert cfg.rewards["force_ratio_tracking"].params["target_ratio"] == target
    assert load_rl_cfg(TASK_ID).experiment_name == "microduck_hd1910_home_force_ratio_no_current"
    env, _, _ = force_environment()
    env.observation_manager = ObservationManager({"actor": deepcopy(cfg.observations["actor"])}, env)
    actual = env.observation_manager.compute(update_history=True)["actor"].reshape(3, 8, 48)
    torch.testing.assert_close(actual[:, :, 34:], torch.tensor(target).repeat(3, 8, 1))
    exported = home_calibration_metadata(env, "measured_reference_smoke.pt")
    assert json.loads(exported["target_force_ratio_json"]) == list(target)


@pytest.mark.parametrize("key,value", [
    ("gate_passed", False), ("schema_version", True), ("schema_version", 2),
    ("reference_kind", "motor_current"), ("joint_names", list(reversed(REFERENCE_JOINT_NAMES))),
    ("home_angles_rad", [0.] * 14), ("target_joint_torque_fraction", [0.] * 14),
    ("target_joint_torque_fraction", [float("nan")] + [1 / 14] * 13),
    ("target_joint_torque_fraction", [-.1] + [1.1 / 13] * 13),
])
def test_bad_reference_cannot_become_a_training_target(reference, key, value):
    path, data = reference
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_home_force_reference(path)


def test_actual_ratio_uses_absolute_generalized_actuator_torque_not_motor_current(monkeypatch):
    monkeypatch.setattr(mdp, "home_calibration_motor_current_proxy", no_proxy)
    env, robot, action = force_environment()
    torque = torch.arange(1., 15.) * torch.tensor([1., -1.] * 7)
    robot.data.qfrc_actuator.copy_(torque.repeat(3, 1))
    expected = torque.abs() / torque.abs().sum()
    torch.testing.assert_close(mdp.home_force_ratio_actual(env), expected.repeat(3, 1))
    torch.testing.assert_close(mdp.home_force_ratio_total_torque(env), torque.abs().sum().repeat(3))
    before = mdp.home_force_ratio_actual(env)
    robot.data.qfrc_actuator *= -3
    torch.testing.assert_close(mdp.home_force_ratio_actual(env), before)
    # Selectors, not assumed contiguous joint positions, define the order.
    action.joint_ids = torch.arange(13, -1, -1)
    torch.testing.assert_close(mdp.home_force_ratio_actual(env), before.flip(-1))


def test_perfect_ratio_only_rewards_loaded_double_support_and_never_rewards_nan():
    env, robot, _ = force_environment()
    reward = mdp.home_force_ratio_tracking(env, TARGET)
    cost = mdp.home_force_ratio_error(env, TARGET)
    torch.testing.assert_close(reward, torch.ones(3))
    torch.testing.assert_close(cost, torch.zeros(3), atol=1e-10, rtol=0)
    robot.data.qfrc_actuator[0] = 0.
    env.scene.sensors["feet_ground_contact"].data.found[1, 0] = False
    robot.data.qfrc_actuator[2, 0] = float("nan")
    assert torch.all(mdp.home_force_ratio_tracking(env, TARGET) == 0)
    assert torch.all(mdp.home_force_ratio_error(env, TARGET) >= 1)
    assert torch.isfinite(mdp.home_force_ratio_error(env, TARGET)).all()
    assert "HomeForce/ratio_rmse" in env.extras["log"]
    robot.data.qfrc_actuator.fill_(torch.finfo(torch.float32).max)
    assert torch.all(mdp.home_force_ratio_tracking(env, TARGET) == 0)
    assert torch.isfinite(mdp.home_force_ratio_error(env, TARGET)).all()


def test_reward_is_exp_of_normalized_mse_and_cost_penalizes_wrong_ratio():
    env, robot, _ = force_environment()
    robot.data.qfrc_actuator[:, 0] = .3
    expected = ((mdp.home_force_ratio_actual(env) - torch.tensor(TARGET)) / .03).square().mean(-1)
    torch.testing.assert_close(mdp.home_force_ratio_tracking(env, TARGET), torch.exp(-expected))
    torch.testing.assert_close(mdp.home_force_ratio_error(env, TARGET), expected)
    assert torch.all(expected > 0)


def test_actor_frame_and_history_include_only_target_ratio_not_measured_ratio(reference, monkeypatch):
    monkeypatch.setattr(mdp, "home_calibration_motor_current_proxy", no_proxy)
    env, robot, action = force_environment()
    cfg = make_microduck_home_force_ratio_env_cfg(reference_path=reference[0])
    before = mdp.home_force_ratio_observation(env, TARGET)
    robot.data.qfrc_actuator *= torch.arange(1., 15.)
    robot.data.qfrc_actuator[1] = 0
    after = mdp.home_force_ratio_observation(env, TARGET)
    assert before.shape == (3, 48)
    assert action.observation_cache.shape == (3, 34)
    torch.testing.assert_close(before, after)
    torch.testing.assert_close(after[:, 34:], torch.tensor(TARGET).repeat(3, 1))
    env.observation_manager = ObservationManager({"actor": deepcopy(cfg.observations["actor"])}, env)
    history = env.observation_manager.compute(update_history=True)["actor"]
    assert history.shape == (3, 384)
    torch.testing.assert_close(history.reshape(3, 8, 48), after[:, None].repeat(1, 8, 1))
    metadata = home_calibration_metadata(env, "model_force_0.pt")
    assert metadata["task_id"] == TASK_ID
    assert metadata["policy_family"] == "hd1910_home_force_ratio_no_current_v1"
    assert metadata["actor_dim"] == 384 and metadata["actor_frame_dim"] == 48
    assert metadata["load_signal"] == metadata["current_input"] == "none"
    assert metadata["objective"] == "joint_torque_magnitude_ratio"
    assert metadata["force_ratio_input"] == "fixed_reference_target_only; no_measured_force_or_current"
    assert metadata["force_ratio_definition"] == "abs(qfrc_actuator_servo)/sum(abs(qfrc_actuator_servo))"
    assert json.loads(metadata["target_force_ratio_json"]) == list(TARGET)
    assert json.loads(metadata["observation_schema_json"])[-1] == {
        "name": "target_joint_torque_fraction", "start": 34, "size": 14, "unit": "dimensionless"}
    assert "current_normalization_epsilon_a" not in metadata
