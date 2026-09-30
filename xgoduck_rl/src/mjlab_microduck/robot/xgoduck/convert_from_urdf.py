#!/usr/bin/env python3
"""Convert an XgoDuck URDF into MJCF for the training tasks.

Regenerates robot_walk.xml, robot_allcollisions.xml, scene XML, and copies
meshes into assets/. Joint names drop the URDF `_joint` suffix. Body names
used by the tasks are remapped (yaw2roll, neck, jaw_soft, bearing_roll, ...).

SW2URDF masses are STL volume × 1000 kg/m³. CAD_MASSES replace them and
inertia is scaled with mass. Trunk CoM uses the SolidWorks mass-properties origin.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE / "urdf" / "XGODUCK.urdf"

# Joint ranges (radians).
JOINT_RANGES: dict[str, tuple[float, float]] = {
    "left_hip_yaw": (-0.22, 0.45),
    "left_hip_roll": (-0.3839724354387525, 0.3839724354387525),
    "left_hip_pitch": (-1.570796326795005, 1.5707963267947882),
    "left_knee": (-1.570796326795012, 1.570796326794781),
    "left_ankle": (-1.5707963267949019, 1.5707963267948912),
    "neck_pitch": (math.radians(-135), 1.0471975511965976),
    "head_pitch": (math.radians(-105), math.radians(105)),
    "head_yaw": (-2.967059728390373, 2.967059728390348),
    "head_roll": (-0.43633231299859127, 0.4363323129985735),
    "right_hip_yaw": (-0.45, 0.22),
    "right_hip_roll": (-0.3839724354387525, 0.3839724354387525),
    "right_hip_pitch": (-1.5707963267949268, 1.5707963267948664),
    "right_knee": (-1.570796326794932, 1.570796326794861),
    "right_ankle": (-1.5707963267949054, 1.5707963267948877),
}

# URDF link → body names used by the tasks.
BODY_RENAME: dict[str, str] = {
    "trunk_base": "trunk_base",
    "left_hip_yaw": "yaw2roll",
    "left_hip_roll": "hip_l",
    "left_hip_pitch": "upper_leg_left",
    "left_knee": "leg",
    "left_ankle": "ankle_left",
    "neck_pitch": "neck",
    "head_pitch": "neck_pitch",
    "head_yaw": "yaw_roll_motion",
    "head_roll": "jaw_soft",
    "right_hip_yaw": "bearing_roll",
    "right_hip_roll": "hip_l_2",
    "right_hip_pitch": "upper_leg_right",
    "right_knee": "leg_2",
    "right_ankle": "ankle_right",
}

# Allcollisions standup: every link gets a named *_collision mesh so mjlab's
# CollisionCfg (geom_names_expr=.*_collision) can enable world contact.
# trunk/head also share bit 1 so they self-collide; other links stay floor-only.
ALLCOLLISION_NAMES = {
    "trunk_base": "trunk_collision",
    "yaw2roll": "left_hip_yaw_collision",
    "hip_l": "left_hip_roll_collision",
    "upper_leg_left": "left_thigh_collision",
    "leg": "left_shin_collision",
    "ankle_left": "left_foot_collision",
    "neck": "neck_collision",
    "neck_pitch": "head_pitch_collision",
    "yaw_roll_motion": "head_yaw_collision",
    "jaw_soft": "head_collision",
    "bearing_roll": "right_hip_yaw_collision",
    "hip_l_2": "right_hip_roll_collision",
    "upper_leg_right": "right_thigh_collision",
    "leg_2": "right_shin_collision",
    "ankle_right": "right_foot_collision",
}

# Walk model: trunk and shins also get a non-colliding self mesh.
SELF_COLLISION_BODIES = {"trunk_base", "leg", "leg_2"}

# mouth_tip in jaw_soft. Position is hand-placed (CAD auto-site sat on the
# cheek). Quaternion maps CAD-zero jaw_soft axes (+X right, +Y up, +Z back)
# onto MicroDuck's mouth convention (+X forward, +Y left, +Z up): site +X =
# body −Z, +Y = body −X, +Z = body +Y.
MOUTH_TIP_POS = np.array([0.0, -0.01234994, -0.07399628])
MOUTH_TIP_QUAT = "0.5 -0.5 0.5 0.5"

ACTUATED_JOINTS = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


MICRODUCK_DEFAULTS = """\
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <default>
    <default class="microduck">
      <joint frictionloss="0.1" armature="0.005"/>
      <position kp="50" dampratio="1"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
        <geom group="3" conaffinity="0"/>
      </default>
    </default>
  </default>
  <default>
    <default class="chosen_actuator">
      <!-- HLS1910 companion PD for bare MuJoCo viewing. Training uses BAM. -->
      <geom contype="0" conaffinity="0"/>
      <joint damping="0.109" frictionloss="0.012" armature="0.0018"/>
      <position kp="0.84" kv="0.0" forcerange="-1.04 1.04" ctrlrange="-10.0 10.0"/>
    </default>
  </default>
  <sensor>
    <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>
    <gyro name="angular-velocity" site="imu" noise="0.005"/>
    <gyro name="imu_ang_vel" site="imu"/>
    <velocimeter name="imu_lin_vel" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
    <subtreeangmom name="root_angmom" body="trunk_base"/>
  </sensor>
  <default>
    <default class="self_collision_only">
      <geom group="3" contype="0" conaffinity="0"/>
    </default>
    <equality solref="0.002 1" solimp="0.99 0.999 0.0005 0.5 2"/>
  </default>
"""


def _fmt(vals) -> str:
    return " ".join(f"{float(v):.8g}" for v in vals)


# SolidWorks CAD masses (kg), 2026-09-14 xgoduck_v2 mass properties.
# SW2URDF wrote STL volume in litres (ρ=1000); replace and scale I ∝ m.
# head_pitch was not re-measured and keeps the URDF value.
CAD_MASSES: dict[str, float] = {
    "trunk_base": 0.223,
    "left_hip_yaw": 0.029,
    "right_hip_yaw": 0.029,
    "left_hip_roll": 0.010,
    "right_hip_roll": 0.010,
    "left_hip_pitch": 0.053,
    "right_hip_pitch": 0.053,
    "left_knee": 0.025,
    "right_knee": 0.025,
    "left_ankle": 0.029,
    "right_ankle": 0.029,
    "neck_pitch": 0.041,
    "head_yaw": 0.033,
    "head_roll": 0.157,
}

# Trunk CoM from SolidWorks mass properties (m, link frame). URDF CoM is the
# uniform-density mesh centroid and is ~1.5 mm off once materials are applied.
CAD_TRUNK_COM = [-0.007, -0.000013, -0.023873]
# IMU stays at the SolidWorks trunk mass-properties origin; only inertial CoM moved.
CAD_IMU_POS = [-0.013982, -0.000013, -0.023873]


def apply_cad_masses(inertials: dict) -> None:
    for link, cad in CAD_MASSES.items():
        info = inertials[link]
        old = info["mass"]
        if old <= 0.0:
            continue
        scale = cad / old
        info["mass"] = cad
        if abs(scale - 1.0) > 1e-12:
            info["fullinertia"] = [v * scale for v in info["fullinertia"]]
        print(f"  cad mass {link:16s} {old:.8g} -> {cad:.8g}  scale={scale:.5f}")


def apply_cad_trunk_com(inertials: dict) -> None:
    old = list(inertials["trunk_base"]["pos"])
    inertials["trunk_base"]["pos"] = list(CAD_TRUNK_COM)
    print(f"  trunk CoM {old} -> {CAD_TRUNK_COM}")


TARGET_TOTAL_MASS = 0.8


def scale_total_mass(inertials: dict, target: float = TARGET_TOTAL_MASS) -> float:
    """Uniform density scale of every link so sum(mass) == target. I ∝ m, CoM unchanged."""
    current = sum(info["mass"] for info in inertials.values())
    scale = target / current
    for name, info in inertials.items():
        old = info["mass"]
        info["mass"] = old * scale
        info["fullinertia"] = [v * scale for v in info["fullinertia"]]
        print(f"  scale {name:16s} {old:.8g} -> {info['mass']:.8g}")
    print(f"  total {current:.8g} -> {target:.8g}  scale={scale:.6f}")
    return scale


def parse_urdf_inertials(urdf_path: Path) -> dict[str, dict]:
    tree = ET.parse(urdf_path)
    out: dict[str, dict] = {}
    for link in tree.getroot().findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None or name is None:
            continue
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
        out[name] = {
            "pos": xyz,
            "mass": float(mass.get("value")) if mass is not None else 0.0,
            "fullinertia": [
                float(inertia.get("ixx")),
                float(inertia.get("iyy")),
                float(inertia.get("izz")),
                float(inertia.get("ixy")),
                float(inertia.get("ixz")),
                float(inertia.get("iyz")),
            ],
        }
    return out


def import_urdf(urdf_path: Path, mesh_dir: Path) -> tuple[ET.Element, mujoco.MjModel, mujoco.MjData]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_urdf = tmp_path / "XGODUCK.urdf"
        text = urdf_path.read_text()
        text = text.replace("package://xgoduck_v2/meshes/", "")
        text = text.replace("package://XGODUCK/meshes/", "")
        local_urdf.write_text(text)
        for stl in mesh_dir.glob("*.STL"):
            shutil.copy2(stl, tmp_path / stl.name)
        spec = mujoco.MjSpec.from_file(str(local_urdf))
        xml = spec.to_xml()
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return ET.fromstring(xml), model, data


def mesh_world_verts(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> np.ndarray:
    mesh_id = int(model.geom_dataid[geom_id])
    adr = int(model.mesh_vertadr[mesh_id])
    n = int(model.mesh_vertnum[mesh_id])
    verts = np.array(model.mesh_vert[adr : adr + n])
    R = data.geom_xmat[geom_id].reshape(3, 3)
    p = data.geom_xpos[geom_id]
    return verts @ R.T + p


def body_frame_from_world(model: mujoco.MjModel, data: mujoco.MjData, body: str, world_pt: np.ndarray) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    R = data.xmat[bid].reshape(3, 3)
    p = data.xpos[bid]
    return (world_pt - p) @ R


def compute_sites(model: mujoco.MjModel, data: mujoco.MjData, inertials: dict) -> dict[str, np.ndarray]:
    sites: dict[str, np.ndarray] = {}
    # IMU at trunk CoM (URDF inertial origin); trunk is welded to world in the raw import
    sites["imu"] = np.array(CAD_IMU_POS)

    for urdf_body, site_name in (("left_ankle", "left_foot"), ("right_ankle", "right_foot")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, urdf_body)
        gid = int(np.where(model.geom_bodyid == bid)[0][0])
        w = mesh_world_verts(model, data, gid)
        zmin = float(w[:, 2].min())
        contact = w[w[:, 2] < zmin + 0.003]
        sites[site_name] = body_frame_from_world(model, data, urdf_body, contact.mean(axis=0))

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_roll")
    gid = int(np.where(model.geom_bodyid == bid)[0][0])
    w = mesh_world_verts(model, data, gid)
    # Front of the CAD head is +X in the assembled world frame.
    front = w[w[:, 0] > w[:, 0].max() - 0.008]
    camera_world = front.mean(axis=0)
    tof_world = camera_world + np.array([0.0, 0.02, 0.0])
    sites["mouth_tip"] = MOUTH_TIP_POS.copy()
    sites["head_camera"] = body_frame_from_world(model, data, "head_roll", camera_world)
    sites["tof"] = body_frame_from_world(model, data, "head_roll", tof_world)
    sites["head_imu"] = np.array(inertials["head_roll"]["pos"])
    return sites


def standing_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    zs = []
    for name in ("left_ankle", "right_ankle"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gid = int(np.where(model.geom_bodyid == bid)[0][0])
        zs.append(float(mesh_world_verts(model, data, gid)[:, 2].min()))
    # 1 mm clearance so the sole is not intersecting the floor at INIT
    return -min(zs) + 0.001


# IMU site stays at the original SolidWorks trunk origin (CAD_IMU_POS).

# STAND pose: hip_pitch and ankle are ±24°.
_DEG24 = math.radians(24)
XGODUCK_STAND_QPOS = (
    0.0,
    -0.08726646259971647,
    -_DEG24,
    -0.004940,
    _DEG24,
    0.3490658503988659,
    0.3490658503988659,
    0.0,
    0.0,
    0.0,
    0.08726646259971647,
    _DEG24,
    0.004940,
    -_DEG24,
)
MICRODUCK_SIT_QPOS = (
    0.0, 0.0, -0.5236, 1.0472, 0.0,
    0.5, 1.6, 0.0, 0.0,
    0.0, 0.0, 0.5236, -1.0472, 0.0,
)
MICRODUCK_FOLD_QPOS = (
    0.0, 0.0, 1.57, 1.57, 0.0,
    1.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -1.57, -1.57, 0.0,
)
# Sitstand SIT: knee ±1.35, hip_pitch ∓0.4079, ankle/hip_roll 0; neck stays HOME.
XGODUCK_SITSTAND_SIT_QPOS = (
    0.0,
    0.0,
    -0.4079,
    1.35,
    0.0,
    0.3490658503988659,
    0.3490658503988659,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4079,
    -1.35,
    0.0,
)


def trunk_z_for_pose(xml_path: Path, joint_qpos14: tuple[float, ...], clearance: float = 0.001) -> float:
    """Trunk height so ankle meshes sit `clearance` above the floor at this pose."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[7 : 7 + 14] = joint_qpos14
    mujoco.mj_forward(model, data)
    zs = []
    for name in ("ankle_left", "ankle_right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        gid = int(np.where(model.geom_bodyid == bid)[0][0])
        zs.append(float(mesh_world_verts(model, data, gid)[:, 2].min()))
    return float(data.qpos[2] - min(zs) + clearance)


def _fmt_q14(q: tuple[float, ...]) -> str:
    return " ".join(f"{float(v):.8g}" for v in q)


def inertial_xml(link: str, inertials: dict) -> str:
    info = inertials[link]
    return (
        f'<inertial pos="{_fmt(info["pos"])}" mass="{info["mass"]:.8g}" '
        f'fullinertia="{_fmt(info["fullinertia"])}"/>'
    )


def site_xml(name: str, pos: np.ndarray, extra: str = "") -> str:
    return f'<site group="3" name="{name}" pos="{_fmt(pos)}"{extra}/>'


def geom_visual(mesh: str) -> str:
    return f'<geom type="mesh" class="visual" mesh="{mesh}"/>'


def geom_collision(mesh: str, name: str | None = None) -> str:
    n = f' name="{name}"' if name else ""
    return f'<geom type="mesh"{n} class="collision" mesh="{mesh}"/>'


def geom_self(mesh: str) -> str:
    return f'<geom type="mesh" class="self_collision_only" mesh="{mesh}"/>'


def emit_body(
    raw_body: ET.Element,
    inertials: dict,
    sites: dict[str, np.ndarray],
    *,
    allcollisions: bool,
    indent: int,
) -> str:
    urdf_name = raw_body.get("name")
    body_name = BODY_RENAME[urdf_name]
    pos = raw_body.get("pos")
    quat = raw_body.get("quat")
    pad = "  " * indent
    joint_name = urdf_name  # after stripping _joint, URDF link == joint stem
    lo, hi = JOINT_RANGES[joint_name]
    mesh = urdf_name  # mesh assets keep URDF link names

    lines = [
        f'{pad}<body name="{body_name}" pos="{pos}" quat="{quat}">',
        f'{pad}  <joint axis="0 0 1" name="{joint_name}" type="hinge" '
        f'range="{lo} {hi}" class="chosen_actuator"/>',
        f"{pad}  {inertial_xml(urdf_name, inertials)}",
        f"{pad}  {geom_visual(mesh)}",
    ]

    if allcollisions and body_name in ALLCOLLISION_NAMES:
        lines.append(f"{pad}  {geom_collision(mesh, ALLCOLLISION_NAMES[body_name])}")
    elif (not allcollisions) and body_name in ("ankle_left", "ankle_right"):
        foot = "left_foot_collision" if body_name == "ankle_left" else "right_foot_collision"
        lines.append(f"{pad}  {geom_collision(mesh, foot)}")
    elif (not allcollisions) and body_name in SELF_COLLISION_BODIES:
        lines.append(f"{pad}  {geom_self(mesh)}")

    if body_name == "ankle_left":
        lines.append(f"{pad}  {site_xml('left_foot', sites['left_foot'])}")
    elif body_name == "ankle_right":
        lines.append(f"{pad}  {site_xml('right_foot', sites['right_foot'])}")
    elif body_name == "jaw_soft":
        cam = sites["head_camera"]
        lines.append(f"{pad}  {site_xml('head_camera', cam)}")
        lines.append(
            f'{pad}  <camera name="head_camera" pos="{_fmt(cam)}" quat="0 0 -1 0"/>'
        )
        lines.append(f"{pad}  {site_xml('tof', sites['tof'])}")
        lines.append(f"{pad}  {site_xml('head_imu', sites['head_imu'])}")
        lines.append(
            f"{pad}  {site_xml('mouth_tip', sites['mouth_tip'], extra=f' quat=\"{MOUTH_TIP_QUAT}\"')}"
        )

    for child in raw_body.findall("body"):
        lines.append(
            emit_body(child, inertials, sites, allcollisions=allcollisions, indent=indent + 1)
        )
    lines.append(f"{pad}</body>")
    return "\n".join(lines)


def build_robot_xml(
    raw_root: ET.Element,
    inertials: dict,
    sites: dict[str, np.ndarray],
    z0: float,
    *,
    allcollisions: bool,
) -> str:
    worldbody = raw_root.find("worldbody")
    assert worldbody is not None
    children = "\n".join(
        emit_body(b, inertials, sites, allcollisions=allcollisions, indent=3)
        for b in worldbody.findall("body")
    )
    meshes = "\n".join(
        f'    <mesh file="{name}.stl"/>' for name in BODY_RENAME
    )
    actuators = "\n".join(
        f'    <position class="chosen_actuator" name="{j}" joint="{j}"/>'
        for j in ACTUATED_JOINTS
    )
    trunk_visual = geom_visual("trunk_base")
    if allcollisions:
        trunk_extra = geom_collision("trunk_base", ALLCOLLISION_NAMES["trunk_base"])
    else:
        trunk_extra = geom_self("trunk_base")

    return f"""<?xml version="1.0" ?>
<!-- XgoDuck MJCF. Mass and inertia from CAD, scaled to 0.8 kg. -->
<mujoco model="xgoduck">
{MICRODUCK_DEFAULTS}
  <worldbody>
    <body name="trunk_base" pos="0 0 {z0:.6g}" quat="1 0 0 0" childclass="microduck">
      <freejoint name="trunk_base_freejoint"/>
      {inertial_xml("trunk_base", inertials)}
      {trunk_visual}
      {trunk_extra}
      {site_xml("imu", sites["imu"])}
{children}
    </body>
  </worldbody>
  <asset>
{meshes}
  </asset>
  <actuator>
{actuators}
  </actuator>
  <equality/>
</mujoco>
"""


def build_scene_xml(
    robot_file: str,
    z0: float,
    stand_z: float,
    sit_z: float,
    fold_z: float,
) -> str:
    zeros14 = "0 " * 14
    stand = _fmt_q14(XGODUCK_STAND_QPOS)
    sit = _fmt_q14(MICRODUCK_SIT_QPOS)
    fold = _fmt_q14(MICRODUCK_FOLD_QPOS)
    return f"""<mujoco model="scene">
    <include file="{robot_file}" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
            height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
            rgb2="0.1 0.2 0.3"
            markrgb="0.8 0.8 0.8" width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
            reflectance="0.2" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />
    </worldbody>
    <keyframe>
        <!-- STAND: hip_pitch / ankle ±24°. Trunk z puts the soles 1 mm above the floor. -->
        <key name="STAND" qpos="
        0 0 {stand_z:.6g} 1 0 0 0
        {stand}"
         ctrl="{stand}"
        />
        <key name="SIT" qpos="
        0 0 {sit_z:.6g} 1 0 0 0
        {sit}"
         ctrl="{sit}"
        />
        <key name="FOLD" qpos="
        0 0 {fold_z:.6g} 1 0 0 0
        {fold}"
         ctrl="{fold}"
        />
        <key name="INIT" qpos="0 0 {z0:.6g} 1 0 0 0 {zeros14.strip()}" ctrl="{zeros14.strip()}"/>
    </keyframe>
</mujoco>
"""


def report_whole_com(
    xml_path: Path,
    joint_qpos14: tuple[float, ...],
    label: str,
    trunk_z: float,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[2] = trunk_z
    data.qpos[7 : 7 + 14] = joint_qpos14
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    world = np.array(data.subtree_com[bid], dtype=float)
    R = data.xmat[bid].reshape(3, 3)
    p = np.array(data.xpos[bid], dtype=float)
    trunk = (world - p) @ R
    imu_world = np.array(data.site_xpos[sid], dtype=float)
    imu = (world - imu_world) @ R
    print(
        f"  {label}: total={float(model.body_subtreemass[bid]):.6f} kg\n"
        f"    world (m)              {world}\n"
        f"    vs trunk_base origin   {trunk}\n"
        f"    vs IMU (trunk axes)    {imu}"
    )


def rewrite_urdf_inertials(urdf_path: Path, inertials: dict) -> None:
    """Overwrite <mass> / <inertia> in a SW2URDF file; keep CoM origin xyz."""
    import re

    text = urdf_path.read_text()
    for name, info in inertials.items():
        ixx, iyy, izz, ixy, ixz, iyz = info["fullinertia"]

        def _mass_repl(m: re.Match, mass: float = info["mass"]) -> str:
            return f'{m.group(1)}{mass:.8g}{m.group(3)}'

        text, n = re.subn(
            rf'(<link\s+name="{re.escape(name)}">\s*<inertial>.*?<mass\s+value=")([^"]+)(")',
            _mass_repl,
            text,
            count=1,
            flags=re.S,
        )
        if n != 1:
            raise RuntimeError(f"mass replace failed for {name} in {urdf_path}")

        inertia_attrs = (
            f'ixx="{ixx:.8g}"\n        ixy="{ixy:.8g}"\n        ixz="{ixz:.8g}"\n        '
            f'iyy="{iyy:.8g}"\n        iyz="{iyz:.8g}"\n        izz="{izz:.8g}"'
        )

        def _inertia_repl(m: re.Match, attrs: str = inertia_attrs) -> str:
            return f"{m.group(1)}{attrs}{m.group(3)}"

        text, n = re.subn(
            rf'(<link\s+name="{re.escape(name)}">\s*<inertial>.*?<inertia\s+)(.*?)(\s*/>)',
            _inertia_repl,
            text,
            count=1,
            flags=re.S,
        )
        if n != 1:
            raise RuntimeError(f"inertia replace failed for {name} in {urdf_path}")
    urdf_path.write_text(text)


def rewrite_urdf_origin(urdf_path: Path, link: str, xyz: list[float]) -> None:
    import re

    text = urdf_path.read_text()
    origin = _fmt(xyz)

    def _repl(m: re.Match, pos: str = origin) -> str:
        return f"{m.group(1)}{pos}{m.group(3)}"

    text, n = re.subn(
        rf'(<link\s+name="{re.escape(link)}">\s*<inertial>\s*<origin\s+xyz=")([^"]+)(")',
        _repl,
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise RuntimeError(f"origin replace failed for {link} in {urdf_path}")
    urdf_path.write_text(text)


def copy_meshes(mesh_dir: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for stl in sorted(mesh_dir.glob("*.STL")):
        shutil.copy2(stl, dest / (stl.stem + ".stl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    inertials = parse_urdf_inertials(args.urdf)
    print("CAD mass overrides:")
    apply_cad_masses(inertials)
    apply_cad_trunk_com(inertials)
    print("Uniform scale to 0.8 kg:")
    scale_total_mass(inertials)
    raw_root, model, data = import_urdf(args.urdf, args.meshes)
    sites = compute_sites(model, data, inertials)
    print(
        f"trunk CoM {inertials['trunk_base']['pos']}  "
        f"mass={inertials['trunk_base']['mass']:.8g}"
    )
    z0 = standing_height(model, data)
    print(f"CAD-zero height z0={z0:.6f}")
    print("sites:")
    for k, v in sites.items():
        print(f"  {k}: {v}")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    copy_meshes(args.meshes, out / "assets")
    urdf_out = out / "urdf"
    urdf_out.mkdir(parents=True, exist_ok=True)
    urdf_text = args.urdf.read_text().replace("package://xgoduck_v2/", "package://XGODUCK/")
    dest_urdf = urdf_out / "XGODUCK.urdf"
    dest_urdf.write_text(urdf_text)
    rewrite_urdf_inertials(dest_urdf, inertials)
    rewrite_urdf_origin(dest_urdf, "trunk_base", CAD_TRUNK_COM)
    csv_src = args.urdf.with_suffix(".csv")
    if csv_src.exists():
        shutil.copy2(csv_src, urdf_out / "XGODUCK.csv")

    walk = build_robot_xml(raw_root, inertials, sites, z0, allcollisions=False)
    allc = build_robot_xml(raw_root, inertials, sites, z0, allcollisions=True)
    (out / "robot_walk.xml").write_text(walk)
    (out / "robot_allcollisions.xml").write_text(allc)

    stand_z = trunk_z_for_pose(out / "robot_walk.xml", XGODUCK_STAND_QPOS)
    sit_z = trunk_z_for_pose(out / "robot_walk.xml", MICRODUCK_SIT_QPOS)
    fold_z = trunk_z_for_pose(out / "robot_walk.xml", MICRODUCK_FOLD_QPOS)
    sitstand_z = trunk_z_for_pose(out / "robot_walk.xml", XGODUCK_SITSTAND_SIT_QPOS)
    print(
        f"CAD-zero z0={z0:.6f}  STAND z={stand_z:.6f}  SIT z={sit_z:.6f}  "
        f"FOLD z={fold_z:.6f}  SITSTAND z={sitstand_z:.6f}"
    )

    (out / "scene_walk.xml").write_text(
        build_scene_xml("robot_walk.xml", z0, stand_z, sit_z, fold_z)
    )
    (out / "scene.xml").write_text(
        build_scene_xml("robot_allcollisions.xml", z0, stand_z, sit_z, fold_z)
    )

    # Sanity compile
    for name in ("robot_walk.xml", "robot_allcollisions.xml", "scene_walk.xml", "scene.xml"):
        m = mujoco.MjModel.from_xml_path(str(out / name))
        print(f"compiled {name}: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom}")

    print("whole-robot CoM:")
    zeros14 = (0.0,) * 14
    report_whole_com(out / "robot_walk.xml", zeros14, "INIT (all joints 0)", z0)
    report_whole_com(out / "robot_walk.xml", XGODUCK_STAND_QPOS, "STAND / HOME", stand_z)


if __name__ == "__main__":
    main()
