#!/usr/bin/env python3
"""Developer-only CPU FK check of the deployment straight calibration reference.

Run with the training venv (MuJoCo and numpy required), without installing either
the training package or its XML/assets on the robot. No physics steps, renderer,
GPU, hardware access, or files are used beyond the XML/assets and policy module.
An explicit --output path writes the JSON report after every check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
DEPLOY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DEPLOY_ROOT.parent
DEFAULT_XML = (
    WORKSPACE_ROOT / "microduck_rl_hd1910/src/mjlab_microduck/robot/microduck/robot_walk.xml"
)
EXPECTED_XML_SHA256 = "80fc4424ed71e430ec4919110dfdb5c714dc1c4e1138e1832a16865cf31eb8c8"
sys.path.insert(0, str(DEPLOY_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from microduck_deploy import policy  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, description: str, atol: float = 1e-10) -> None:
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=description)


def verify(xml_path: Path) -> dict:
    """Return the full report; raise on a changed model or invalid reference."""
    xml_path = xml_path.resolve()
    xml_sha256 = hashlib.sha256(xml_path.read_bytes()).hexdigest()
    require(xml_sha256 == EXPECTED_XML_SHA256,
            "XML differs from the HD training model used for this reference")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    names = tuple(policy.JOINT_NAMES)
    hinge_names = tuple(
        model.joint(i).name for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    )
    require(len(names) == 14 and names == hinge_names,
            "Canonical 14-joint order differs from the XML hinge order")
    ids = np.array([model.joint(name).id for name in names])
    addresses = model.jnt_qposadr[ids]
    reference = np.asarray(policy.STRAIGHT_REFERENCE, dtype=np.float64)
    home = np.asarray(policy.HOME, dtype=np.float64)
    require(reference.shape == home.shape == (14,), "Expected two 14D poses")
    require(bool(np.isfinite(reference).all() and np.isfinite(home).all()),
            "Reference and HOME must be finite")
    close(policy.policy_calibration_reference("straight"), reference,
          "Calibration API differs from STRAIGHT_REFERENCE", atol=0)

    trunk_id = model.body("trunk_base").id

    def pose(q):
        mujoco.mj_resetData(model, data)
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)

    def trunk_rotation():
        return data.xmat[trunk_id].reshape(3, 3)

    def relative_position(world_position):
        return trunk_rotation().T @ (world_position - data.xpos[trunk_id])

    def anchor(name):
        return relative_position(data.xanchor[model.joint(name).id])

    pose(np.zeros(14))
    zero_anchors = {name: anchor(name).copy() for name in names}
    zero_offsets_mm = {}
    derived_angles = []
    for side in ("left", "right"):
        offset = (zero_anchors[f"{side}_knee"] -
                  zero_anchors[f"{side}_hip_pitch"]) * 1000
        zero_offsets_mm[side] = offset.tolist()
        close(offset[[0, 2]], [-35.7771, -22.0], "ZERO thigh sagittal dimensions")
        derived_angles.append(float(np.arctan2(-offset[0], -offset[2])))
    angle = derived_angles[0]
    close(derived_angles, [angle, angle], "Left/right derived angles", atol=1e-12)
    close(policy.STRAIGHT_ANGLE_RAD, angle, "Canonical straight angle", atol=1e-12)
    expected = np.zeros(14)
    for side, sign in (("left", -1), ("right", 1)):
        for joint in ("hip_pitch", "knee"):
            expected[names.index(f"{side}_{joint}")] = sign * angle
    close(reference, expected, "Canonical straight reference", atol=1e-12)

    limits = model.jnt_range[ids].copy()
    require(bool(model.jnt_limited[ids].all()), "An active joint is not limited")
    close(policy.JOINT_LIMITS, limits, "Deployment versus XML joint limits", atol=1e-12)
    # 1001 points including both endpoints. Each scalar range is an interval, so
    # the endpoints also prove the whole linear segment satisfies angular limits.
    weights = np.linspace(0.0, 1.0, 1001)
    samples = reference[None, :] + weights[:, None] * (home - reference)[None, :]
    margins = np.minimum(samples - limits[:, 0], limits[:, 1] - samples)
    require(bool((margins >= 0).all()), "Straight-to-HOME path exceeds XML limits")

    pose(reference)
    anchors = {name: anchor(name).copy() for name in names}
    axes = {
        name: (trunk_rotation().T @ data.xaxis[model.joint(name).id]).tolist()
        for name in names
    }
    legs = {}
    for side in ("left", "right"):
        hka = np.stack([anchors[f"{side}_{name}"]
                        for name in ("hip_pitch", "knee", "ankle")])
        close(hka[:, 0], [0.004] * 3, f"{side} H/K/A vertical side projection")
        require(bool((np.diff(hka[:, 2]) < 0).all()), "H/K/A vertical order")
        legs[side] = {
            "hip_knee_ankle_trunk_mm": (hka * 1000).tolist(),
            "sagittal_x_spread_m": float(np.ptp(hka[:, 0])),
            "projected_thigh_and_shin_length_mm": (-np.diff(hka[:, 2]) * 1000).tolist(),
            "anchors_collinear_in_3d": False,
        }
    left = np.array(legs["left"]["hip_knee_ankle_trunk_mm"])
    right = np.array(legs["right"]["hip_knee_ankle_trunk_mm"])
    close(left * [1, -1, 1], right, "Left/right H/K/A mirror symmetry")

    site_frames = {}
    for name in ("left_foot", "right_foot", "head_camera"):
        site_id = model.site(name).id
        rotation = trunk_rotation().T @ data.site_xmat[site_id].reshape(3, 3)
        close(rotation, np.eye(3), f"{name} aligned with trunk reference frame")
        site_frames[name] = {
            "position_trunk_mm": (relative_position(data.site_xpos[site_id]) * 1000).tolist(),
            "rotation_site_to_trunk": rotation.tolist(),
            "max_abs_error_from_identity": float(np.max(np.abs(rotation - np.eye(3)))),
        }
    neck_delta = anchors["head_pitch"] - anchors["neck_pitch"]
    close(neck_delta, [0, 0, 0.050], "Neck pitch to head pitch axis vertical")

    foot_meshes = {}
    for side in ("left", "right"):
        geom_id = model.geom(f"{side}_foot_collision").id
        require(model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH,
                "Foot collision geometry is not a mesh")
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:start + count]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
        trunk_vertices = (world_vertices - data.xpos[trunk_id]) @ trunk_rotation()
        foot_meshes[side] = {
            "minimum_vertex_trunk_z_m": float(trunk_vertices[:, 2].min()),
            "minimum_vertex_world_z_at_xml_default_root_m": float(world_vertices[:, 2].min()),
        }

    delta = home - reference
    max_delta = float(np.max(np.abs(delta)))
    max_names = [names[i] for i in np.flatnonzero(np.isclose(np.abs(delta), max_delta))]
    return {
        "schema_version": 1,
        "result": "passed",
        "scope": "仅 CPU 正向运动学与关节角限位检验；未运行动力学或硬件。",
        "sources": {
            "xml_path": str(xml_path),
            "xml_sha256": xml_sha256,
            "policy_path": str(Path(policy.__file__).resolve()),
            "canonical_constants": ["STRAIGHT_ANGLE_RAD", "STRAIGHT_REFERENCE", "HOME", "JOINT_NAMES"],
            "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__,
        },
        "coordinate_frame": "trunk_base: X 向机器人前，Y 向机器人左，Z 向上；位置单位见字段名。",
        "joint_names": list(names),
        "angle_derivation": {
            "formula": "atan2(35.7771, 22.0)，两参数单位均为 mm，来自 XML 的 ZERO 髋到膝侧投影偏移",
            "zero_hip_to_knee_trunk_mm": zero_offsets_mm,
            "derived_angle_rad": angle,
            "derived_angle_deg": float(np.rad2deg(angle)),
            "canonical_angle_rad": float(policy.STRAIGHT_ANGLE_RAD),
        },
        "straight_reference_rad": reference.tolist(),
        "straight_reference_deg": np.rad2deg(reference).tolist(),
        "home_rad": home.tolist(),
        "home_minus_straight_rad": delta.tolist(),
        "home_minus_straight_deg": np.rad2deg(delta).tolist(),
        "home_transition": {
            "maximum_absolute_delta_rad": max_delta,
            "maximum_absolute_delta_deg": float(np.rad2deg(max_delta)),
            "maximum_delta_joints": max_names,
            "exceeds_previous_start_distance_0_6_rad": bool(max_delta > 0.6),
        },
        "straight_joint_anchors_trunk_mm": {name: (v * 1000).tolist() for name, v in anchors.items()},
        "straight_positive_joint_axes_trunk": axes,
        "legs": legs,
        "neck_axis_displacement_trunk_mm": (neck_delta * 1000).tolist(),
        "site_frames": site_frames,
        "xml_joint_limits_rad": limits.tolist(),
        "interpolation_limits": {
            "formula": "q(t) = STRAIGHT_REFERENCE + t * (HOME - STRAIGHT_REFERENCE)",
            "samples": len(weights),
            "includes_endpoints": True,
            "all_samples_within_limits": True,
            "minimum_margin_per_joint_rad": margins.min(axis=0).tolist(),
            "continuous_segment_within_limits": True,
            "continuous_proof": "各关节上下限构成凸区间；两个端点在限位内，则线性插值全程在限位内。",
        },
        "spawn_geometry": {
            "xml_default_trunk_world_z_m": float(data.xpos[trunk_id, 2]),
            "foot_collision_mesh_vertex_extrema": foot_meshes,
            "note": "直腿姿态不可直接沿用 XML 的 0.12 m 根高度做地面站立试验：鞋底顶点低于 z=0；需按实际碰撞几何设置初始高度。",
        },
        "not_proven": [
            "H/K/A 仅侧投影竖直，轴锚点 Y 坐标有偏置，不能称三维锚点共线。",
            "foot 参考坐标系与躯干平行；没有断言曲面鞋底的所有点共面。",
            "没有验证过渡全程无自碰撞、无地面碰撞、静态平衡或动力学稳定。",
            "没有验证国产改件的实际尺寸、装配方向、编码器零点或硬件机械限位。",
            "模型关节限位通过不等同于真机可安全运动；本脚本不发送舵机命令。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML,
                        help="Exact local HD training robot_walk.xml; mesh assets must be alongside it")
    parser.add_argument("--output", type=Path,
                        help="Write JSON here after all checks pass; otherwise do not write files")
    args = parser.parse_args()
    report = verify(args.xml)
    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    angle = report["angle_derivation"]["derived_angle_deg"]
    delta = report["home_transition"]["maximum_absolute_delta_deg"]
    print(f"PASS: straight angle {angle:.12f} deg; max HOME delta {delta:.12f} deg")
    print("PASS: H/K/A side projection, foot/head frames, vertical neck, 1001 in-limit samples")
    print("Scope: CPU FK and angular limits only; collisions, stability and hardware unverified")
    if args.output is not None:
        print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
