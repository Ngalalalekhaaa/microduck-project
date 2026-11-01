"""HOME runner and explicit export contract for its separate policy interface."""

import json
from pathlib import Path

from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner


def home_calibration_metadata(env, run_path: str) -> dict:
    """Describe the sensor history and stateful action wrapper required at inference."""
    from .microduck_home_calibration_env_cfg import (
        FRAME_DIM, TASK_ID, NO_CURRENT_FRAME_DIM, NO_CURRENT_TASK_ID,
    )
    from .mdp import home_calibration_no_current_observation, home_force_ratio_observation

    action = env.action_manager.get_term("home_calibration")
    observation = env.observation_manager.get_term_cfg("actor", "sensor_frame")
    history_length = observation.history_length or 1
    layout = [
        {"name": "reported_joint_position_minus_home", "start": 0, "size": 14, "unit": "rad"},
        {"name": "signed_current_fraction", "start": 14, "size": 14, "unit": "dimensionless"},
        {"name": "sum_absolute_current", "start": 28, "size": 1, "unit": "A"},
        {"name": "projected_gravity", "start": 29, "size": 3, "unit": "unit_vector"},
        {"name": "body_gyro", "start": 32, "size": 3, "unit": "rad/s"},
        {"name": "command_offset_from_home", "start": 35, "size": 14, "unit": "rad"},
    ]
    metadata = {
        "task_id": TASK_ID,
        "policy_family": "hd1910_home_calibration_v1",
        "run_path": run_path,
        "joint_names": list(action.joint_names),
        "default_joint_pos": action.home[0].detach().cpu().tolist(),
        "home_angles_rad_json": json.dumps(action.home[0].detach().cpu().tolist()),
        "control_dt": env.step_dt,
        "actor_frame_dim": FRAME_DIM,
        "history_length": history_length,
        "actor_dim": env.observation_manager.group_obs_dim["actor"][0],
        "encoder_ticks": observation.params.get("encoder_ticks", 4096),
        "current_normalization_epsilon_a": observation.params.get("current_normalization_epsilon_a", .02),
        "current_normalization": "I/max(sum(abs(I)),current_normalization_epsilon_a)",
        "current_sign": "positive_motor_torque_along_positive_model_joint_axis",
        "history_order": "oldest_to_newest_frame_major_first_sample_repeated_on_reset",
        "observation_schema_json": json.dumps(layout, separators=(",", ":")),
        "observation_normalization": "baked_into_onnx",
        "action_semantics": "command_offset=clip(command_offset+clip(output,-1,1)*delta_limit_rad,-offset_limit_rad,offset_limit_rad)",
        "delta_limit_rad": action.cfg.delta_limit_rad,
        "offset_limit_rad": action.cfg.offset_limit_rad,
        "settle_seconds": action.cfg.settle_seconds,
        "deployment_target": "existing_calibrated_HOME+command_offset; do_not_subtract_unknown_training_bias",
        "load_signal": "synthetic_signed_phase_current_A=BAM_motor_torque/kt; hardware_mapping_not_validated",
        "role": "simulation_candidate; current walking runtime has a different observation/action contract",
    }
    sensor_function = getattr(observation, "func", None)
    if sensor_function in (home_calibration_no_current_observation, home_force_ratio_observation):
        layout = [
            {"name": "reported_joint_position_minus_home", "start": 0, "size": 14, "unit": "rad"},
            {"name": "projected_gravity", "start": 14, "size": 3, "unit": "unit_vector"},
            {"name": "body_gyro", "start": 17, "size": 3, "unit": "rad/s"},
            {"name": "command_offset_from_home", "start": 20, "size": 14, "unit": "rad"},
        ]
        metadata.update({
            "task_id": NO_CURRENT_TASK_ID,
            "policy_family": "hd1910_home_calibration_no_current_v2",
            "actor_frame_dim": NO_CURRENT_FRAME_DIM,
            "observation_schema_json": json.dumps(layout, separators=(",", ":")),
            "load_signal": "none",
            "current_input": "none",
            "role": "simulation_candidate; encoder_and_imu_only_home_correction",
        })
        for key in ("current_normalization_epsilon_a", "current_normalization", "current_sign"):
            metadata.pop(key)
        if sensor_function is home_force_ratio_observation:
            from .microduck_home_force_ratio_env_cfg import TASK_ID as FORCE_TASK_ID, FRAME_DIM as FORCE_FRAME_DIM
            layout.append({"name": "target_joint_torque_fraction", "start": 34,
                           "size": 14, "unit": "dimensionless"})
            metadata.update({
                "task_id": FORCE_TASK_ID,
                "policy_family": "hd1910_home_force_ratio_no_current_v1",
                "actor_frame_dim": FORCE_FRAME_DIM,
                "observation_schema_json": json.dumps(layout, separators=(",", ":")),
                "role": "simulation_candidate; encoder_and_imu_only_home_force_ratio_correction",
                "objective": "joint_torque_magnitude_ratio",
                "force_ratio_definition": "abs(qfrc_actuator_servo)/sum(abs(qfrc_actuator_servo))",
                "force_ratio_input": "fixed_reference_target_only; no_measured_force_or_current",
                "target_force_ratio_json": json.dumps(list(observation.params["target_ratio"])),
            })
    return metadata


class HomeCalibrationOnPolicyRunner(MjlabOnPolicyRunner):
    """Use the standard normalized exporter with HOME-specific metadata."""

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Calibration episodes start with apparent HOME and a settling interval.
        # Artificially advancing the timeout clock would shorten that procedure.
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)

    def save(self, path: str, infos=None) -> None:
        super().save(path, infos)
        directory = Path(path).parent
        filename = "home_calibration.onnx"
        self.export_policy_to_onnx(str(directory), filename)
        metadata = home_calibration_metadata(self.env.unwrapped, path)
        attach_metadata_to_onnx(str(directory / filename), metadata)
        (directory / "home_calibration_contract.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
