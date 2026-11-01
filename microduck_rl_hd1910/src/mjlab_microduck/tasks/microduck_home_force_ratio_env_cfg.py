"""A fixed torque-magnitude ratio objective with encoder/IMU-only feedback.

Measured simulation torques are privileged reward information. The actor sees
only encoder/IMU history, its own cumulative command, and one fixed reference
target. This recipe does not establish arbitrary target-ratio generalization.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from mjlab.managers import RewardTermCfg

from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from . import mdp
from .microduck_home_calibration_env_cfg import (
    HISTORY_LENGTH, MicroduckHomeCalibrationNoCurrentRlCfg,
    make_microduck_home_calibration_env_cfg,
)


TASK_ID = "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"
FRAME_DIM = 48
ACTOR_DIM = FRAME_DIM * HISTORY_LENGTH
REFERENCE_PATH = Path(__file__).with_name("home_force_reference.json")
REFERENCE_JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)


def load_home_force_reference(path: str | Path | None = None) -> dict:
    """Load a measured, gated simulation reference with exact model conventions."""
    source = Path(path) if path is not None else REFERENCE_PATH
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("HOME force reference requires schema_version=1")
    if (data.get("reference_kind") != "absolute_joint_actuator_torque_fraction"
            or data.get("gate_passed") is not True):
        raise ValueError("HOME force reference must pass its physical measurement gates")
    if data.get("joint_names") != list(REFERENCE_JOINT_NAMES):
        raise ValueError("HOME force reference joint order differs from the 14-servo model")
    home = data.get("home_angles_rad")
    if not isinstance(home, list) or len(home) != 14:
        raise ValueError("HOME force reference requires 14 HOME angles in radians")
    for name, value in zip(REFERENCE_JOINT_NAMES, home):
        expected = [angle for pattern, angle in HOME_FRAME.joint_pos.items()
                    if re.fullmatch(pattern, name)]
        if (len(expected) != 1 or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not math.isclose(value, expected[0], rel_tol=0., abs_tol=5e-8)):
            raise ValueError("HOME force reference angles differ from the model HOME")
    ratio = data.get("target_joint_torque_fraction")
    if (not isinstance(ratio, list) or len(ratio) != 14
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in ratio)
            or not math.isclose(sum(ratio), 1., rel_tol=0., abs_tol=1e-6)):
        raise ValueError("HOME force target requires 14 finite nonnegative fractions summing to one")
    return data


def make_microduck_home_force_ratio_env_cfg(
    play: bool = False,
    evaluation_profile: str | None = None,
    reference_path: str | Path | None = None,
):
    """Build the fixed-reference 384D family; refuse a missing/invalid reference."""
    reference = load_home_force_reference(reference_path)
    target = tuple(reference["target_joint_torque_fraction"])
    cfg = make_microduck_home_calibration_env_cfg(
        play=play, evaluation_profile=evaluation_profile, include_current=False,
    )
    # The shared sensor cache remains 34D; append the fixed target after it.
    for group in ("actor", "critic"):
        sensor = cfg.observations[group].terms["sensor_frame"]
        sensor.func = mdp.home_force_ratio_observation
        sensor.params = {"target_ratio": target}
    del cfg.rewards["true_home_tracking"]
    del cfg.rewards["true_home_error"]
    params = {"target_ratio": target, "scale": .03, "minimum_total_torque_nm": .02}
    cfg.rewards["force_ratio_tracking"] = RewardTermCfg(
        func=mdp.home_force_ratio_tracking, weight=4., params=dict(params),
    )
    cfg.rewards["force_ratio_error"] = RewardTermCfg(
        func=mdp.home_force_ratio_error, weight=-.25, params=dict(params),
    )
    return cfg


MicroduckHomeForceRatioRlCfg = deepcopy(MicroduckHomeCalibrationNoCurrentRlCfg)
MicroduckHomeForceRatioRlCfg.experiment_name = "microduck_hd1910_home_force_ratio_no_current"
MicroduckHomeForceRatioRlCfg.run_name = "home_force_ratio_no_current_bias3deg_history8"
