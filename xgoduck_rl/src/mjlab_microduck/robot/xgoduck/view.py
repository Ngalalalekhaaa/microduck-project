"""MuJoCo viewer for XGODUCK (empty floor scene).

Draws site frames as RGB axes:
  red   = site +X
  green = site +Y
  blue  = site +Z
Sites: trunk imu (policy), head_imu, mouth_tip (ground-pick +X).
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_XML = Path(__file__).resolve().parent / "scene_walk.xml"

# Visible on a ~14 cm robot — axes stick out past the trunk/head.
_AXIS_LEN = 0.18
_AXIS_W = 0.006
_ORIGIN_R = 0.01
_RGB = (
    (1.0, 0.15, 0.1, 1.0),
    (0.15, 0.85, 0.2, 1.0),
    (0.15, 0.4, 1.0, 1.0),
)

# (site name, origin rgba, site_size scale)
_FRAME_SITES = (
    ("imu", (1.0, 0.85, 0.1, 1.0), 1.0),
    ("head_imu", (0.9, 0.2, 0.85, 1.0), 0.8),
    ("mouth_tip", (0.15, 0.95, 0.95, 1.0), 0.8),
)

pick_conf = np.array([0.0, -5.0, -80.0, 0.0, 50.0, 
        -120.0, -90.0, 0.0, 0.0, 
        0.0, 5.0, 80.0, 0.0, -50.0])
pick_conf = pick_conf / 180.0 * np.pi

def _add_sphere(scn: mujoco.MjvScene, pos: np.ndarray, radius: float, rgba) -> None:
    scn.ngeom += 1
    geom = scn.geoms[scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos,
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )


def _add_arrow(scn: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, rgba) -> None:
    scn.ngeom += 1
    geom = scn.geoms[scn.ngeom - 1]
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        _AXIS_W,
        start,
        end,
    )


def _draw_site_frame(scn: mujoco.MjvScene, data: mujoco.MjData, sid: int, origin_rgba) -> None:
    pos = data.site_xpos[sid].copy()
    mat = data.site_xmat[sid].reshape(3, 3)
    _add_sphere(scn, pos, _ORIGIN_R, origin_rgba)
    for i, rgba in enumerate(_RGB):
        _add_arrow(scn, pos, pos + _AXIS_LEN * mat[:, i], rgba)


def _axis_name(vec: np.ndarray) -> str:
    names = ("+Xfwd", "-Xback", "+Yleft", "-Yright", "+Zup", "-Zdown")
    world = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=float,
    )
    return names[int(np.argmax(world @ vec))]


def _print_site(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        print(f"  {name}: (site 不存在)")
        return
    bid = int(model.site_bodyid[sid])
    body = model.body(bid).name
    pos_body = model.site_pos[sid]
    quat = model.site_quat[sid]
    mat = data.site_xmat[sid].reshape(3, 3)
    print(f"  {name}  挂在 body={body}")
    print(f"    site 相对 body: pos={pos_body}  quat(wxyz)={quat}")
    print(f"    世界坐标:       {data.site_xpos[sid]}")
    for i, ax in enumerate("XYZ"):
        v = mat[:, i]
        print(f"    +{ax} (RGB[{i}]) 世界方向 {v}  ≈ {_axis_name(v)}")
    if name == "mouth_tip":
        x = mat[:, 0]
        align = float(-x[2])  # site +X · world down; ground-pick reward
        print(
            f"    ground-pick alignment (−X·worldZ) = {align:+.3f}  "
            f"(+1=嘴+X朝下, 0=水平, -1=朝上)"
        )


def main() -> None:
    xml_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML
    print(f"加载: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    print(
        f"body={model.nbody} joint={model.njnt} geom={model.ngeom} actuator={model.nu}"
    )
    keys = [model.key(i).name for i in range(model.nkey)]
    if keys:
        print(f"keyframes: {', '.join(keys)}")

    stand = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if stand < 0:
        print("未找到 STAND keyframe，使用模型默认 qpos")
        stand_ctrl = None
    else:
        stand_ctrl = np.array(model.key_ctrl[stand], dtype=np.float64, copy=True)
        data.qpos[:] = model.key_qpos[stand]
        data.ctrl[:] = stand_ctrl
        # data.ctrl[:] = pick_conf
        mujoco.mj_forward(model, data)
        print("ctrl 跟踪 STAND")

    print("坐标轴: 红=+X  绿=+Y  蓝=+Z")
    print("  黄球=躯干 imu（策略）  品红=head_imu  青球=mouth_tip（ground-pick 嘴+X）")
    print("右键面板拖动滑块 | 按 2/3 切换视觉/碰撞组")
    print("STAND 下 site 朝向:")
    site_ids: list[tuple[int, tuple[float, float, float, float]]] = []
    for name, rgba, scale in _FRAME_SITES:
        _print_site(model, data, name)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            continue
        model.site_size[sid] = [_ORIGIN_R * scale, 0.0, 0.0]
        model.site_rgba[sid] = list(rgba)
        site_ids.append((sid, rgba))

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        # Sites are group 3. This MuJoCo build has no mjVIS_SITE flag;
        # sitegroup toggles them (Rendering tab → Site Group 3).
        viewer.opt.sitegroup[3] = 1
        while viewer.is_running():
            step_start = time.time()
            if stand_ctrl is not None:
                data.ctrl[:] = stand_ctrl
                # data.ctrl[:] = pick_conf
            mujoco.mj_step(model, data)
            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                for sid, rgba in site_ids:
                    _draw_site_frame(scn, data, sid, rgba)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
