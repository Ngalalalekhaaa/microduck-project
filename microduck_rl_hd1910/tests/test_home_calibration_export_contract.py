"""Exported HOME contracts describe the live sensor and action interface."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks.home_calibration_runner import home_calibration_metadata


def _env(*, history=8, params=None, dt=.05):
    names = [f"joint_{index}" for index in range(14)]
    action = SimpleNamespace(
        joint_names=names,
        home=torch.linspace(-.5, .5, 14).repeat(2, 1),
        cfg=SimpleNamespace(
            delta_limit_rad=math.radians(.6),
            offset_limit_rad=math.radians(9),
            settle_seconds=1.1,
        ),
    )
    observation = SimpleNamespace(history_length=history, params=params or {})

    def get_action(name):
        assert name == "home_calibration"
        return action

    def get_observation(group, name):
        assert (group, name) == ("actor", "sensor_frame")
        return observation

    env = SimpleNamespace(
        step_dt=dt,
        action_manager=SimpleNamespace(get_term=get_action),
        observation_manager=SimpleNamespace(
            get_term_cfg=get_observation,
            group_obs_dim={"actor": (49 * (history or 1),)},
        ),
    )
    return env, action


def test_metadata_uses_live_timing_history_sensor_parameters_and_action_limits():
    env, action = _env(
        history=5, dt=.025,
        params={"encoder_ticks": 8192, "current_normalization_epsilon_a": .07},
    )
    metadata = home_calibration_metadata(env, "model_123.pt")
    assert metadata["control_dt"] == .025
    assert metadata["history_length"] == 5
    assert metadata["actor_frame_dim"] == 49
    assert metadata["actor_dim"] == 245
    assert metadata["encoder_ticks"] == 8192
    assert metadata["current_normalization_epsilon_a"] == .07
    assert metadata["current_normalization"] == "I/max(sum(abs(I)),current_normalization_epsilon_a)"
    assert metadata["current_sign"] == "positive_motor_torque_along_positive_model_joint_axis"
    assert metadata["delta_limit_rad"] == action.cfg.delta_limit_rad
    assert metadata["offset_limit_rad"] == action.cfg.offset_limit_rad
    assert metadata["settle_seconds"] == 1.1
    assert metadata["joint_names"] == action.joint_names
    assert metadata["default_joint_pos"] == action.home[0].tolist()
    assert json.loads(metadata["home_angles_rad_json"]) == action.home[0].tolist()
    assert metadata["run_path"] == "model_123.pt"
    # The sidecar must remain JSON serializable, without device tensors.
    assert json.loads(json.dumps(metadata)) == metadata


@pytest.mark.parametrize("history,dimension", [(8, 392), (0, 49)])
def test_metadata_records_default_sensor_processing_and_history_disabled(history, dimension):
    env, _ = _env(history=history)
    metadata = home_calibration_metadata(env, "checkpoint.pt")
    assert metadata["history_length"] == (history or 1)
    assert metadata["actor_dim"] == dimension
    assert metadata["encoder_ticks"] == 4096
    assert metadata["current_normalization_epsilon_a"] == .02
    assert metadata["history_order"] == "oldest_to_newest_frame_major_first_sample_repeated_on_reset"


def test_export_is_calibration_interface_without_privileged_or_walking_channels():
    env, _ = _env()
    metadata = home_calibration_metadata(env, "checkpoint.pt")
    assert metadata["policy_family"] == "hd1910_home_calibration_v1"
    assert metadata["task_id"] == "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
    schema = json.loads(metadata["observation_schema_json"])
    assert [(entry["start"], entry["size"]) for entry in schema] == [
        (0, 14), (14, 14), (28, 1), (29, 3), (32, 3), (35, 14),
    ]
    assert [entry["name"] for entry in schema] == [
        "reported_joint_position_minus_home", "signed_current_fraction",
        "sum_absolute_current", "projected_gravity", "body_gyro",
        "command_offset_from_home",
    ]
    assert metadata["observation_normalization"] == "baked_into_onnx"
    assert "command_offset+clip(output,-1,1)*delta_limit_rad" in metadata["action_semantics"]
    assert metadata["deployment_target"] == (
        "existing_calibrated_HOME+command_offset; do_not_subtract_unknown_training_bias")
    assert "hardware_mapping_not_validated" in metadata["load_signal"]
