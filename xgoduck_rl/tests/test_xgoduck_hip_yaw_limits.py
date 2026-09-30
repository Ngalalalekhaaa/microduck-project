"""XgoDuck hip-yaw mechanical ranges (left / right are mirrored)."""

import mujoco

from mjlab_microduck.robot.xgoduck.convert_from_urdf import JOINT_RANGES
from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_ALLCOLLISIONS_XML,
    XGODUCK_WALK_XML,
)

# User-specified XgoDuck hip-yaw limits (rad). Inward travel is the smaller side.
LEFT_HIP_YAW = (-0.22, 0.45)
RIGHT_HIP_YAW = (-0.45, 0.22)


def _jnt_range(model: mujoco.MjModel, name: str) -> tuple[float, float]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lo, hi = model.jnt_range[jid]
    return float(lo), float(hi)


def test_convert_from_urdf_joint_ranges_match_user_limits():
    assert JOINT_RANGES["left_hip_yaw"] == LEFT_HIP_YAW
    assert JOINT_RANGES["right_hip_yaw"] == RIGHT_HIP_YAW


def test_walk_and_allcollisions_xml_hip_yaw_ranges():
    for path in (XGODUCK_WALK_XML, XGODUCK_ALLCOLLISIONS_XML):
        model = mujoco.MjModel.from_xml_path(str(path))
        assert _jnt_range(model, "left_hip_yaw") == LEFT_HIP_YAW
        assert _jnt_range(model, "right_hip_yaw") == RIGHT_HIP_YAW
