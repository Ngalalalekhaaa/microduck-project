"""XGODUCK v2 inertial: CAD mass ratios, uniformly scaled to 0.8 kg."""

import mujoco
import numpy as np

from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_ALLCOLLISIONS_XML,
    XGODUCK_WALK_XML,
)

# Pre-scale SolidWorks CAD masses (kg). head_pitch stays on the URDF value.
CAD_BODY_MASS = {
    "trunk_base": 0.223,
    "yaw2roll": 0.029,
    "hip_l": 0.010,
    "upper_leg_left": 0.053,
    "leg": 0.025,
    "ankle_left": 0.029,
    "neck": 0.041,
    "neck_pitch": 0.0040048,
    "yaw_roll_motion": 0.033,
    "jaw_soft": 0.157,
    "bearing_roll": 0.029,
    "hip_l_2": 0.010,
    "upper_leg_right": 0.053,
    "leg_2": 0.025,
    "ankle_right": 0.029,
}

TARGET_MASS = 0.8
# Walk-model trunk CoM (body frame). IMU stays at the SW mass-properties origin.
CAD_TRUNK_COM = np.array([-0.021, -0.000013, -0.023873])
CAD_IMU_POS = np.array([-0.013982, -0.000013, -0.023873])


def test_xgoduck_total_mass_scaled_to_0_8kg():
    model = mujoco.MjModel.from_xml_path(str(XGODUCK_WALK_XML))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    total = float(model.body_subtreemass[bid])
    assert abs(total - TARGET_MASS) < 1e-6, total

    cad_sum = sum(CAD_BODY_MASS.values())
    scale = TARGET_MASS / cad_sum
    for body, cad in CAD_BODY_MASS.items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        got = float(model.body_mass[i])
        expect = cad * scale
        assert abs(got - expect) < 1e-6, f"{body}: {got} != {expect}"

    np.testing.assert_allclose(model.body_ipos[bid], CAD_TRUNK_COM, atol=1e-6)

    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    np.testing.assert_allclose(model.site_pos[sid], CAD_IMU_POS, atol=1e-6)


def test_xgoduck_allcollisions_matches_walk_inertials():
    walk = mujoco.MjModel.from_xml_path(str(XGODUCK_WALK_XML))
    allc = mujoco.MjModel.from_xml_path(str(XGODUCK_ALLCOLLISIONS_XML))
    assert walk.nbody == allc.nbody
    for i in range(walk.nbody):
        name = walk.body(i).name
        j = mujoco.mj_name2id(allc, mujoco.mjtObj.mjOBJ_BODY, name)
        assert j >= 0, name
        np.testing.assert_allclose(
            allc.body_mass[j], walk.body_mass[i], atol=1e-9, err_msg=f"{name} mass"
        )
        np.testing.assert_allclose(
            allc.body_ipos[j], walk.body_ipos[i], atol=1e-9, err_msg=f"{name} com"
        )
        np.testing.assert_allclose(
            allc.body_inertia[j], walk.body_inertia[i], atol=1e-12, err_msg=f"{name} I"
        )
