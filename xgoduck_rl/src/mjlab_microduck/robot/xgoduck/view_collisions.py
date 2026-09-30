"""XgoDuck viewer for the full-collision scene, plus actor IMU obs printout.

Loads scene.xml and prints base_ang_vel and projected_gravity at 0.5 Hz
(body-frame ω and R^T g, g_world = [0, 0, -1]).
"""
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from view import (  # noqa: E402
    _draw_site_frame,
    _print_site,
    _ORIGIN_R,
)

DEFAULT_XML = _HERE / "scene.xml"
_PRINT_HZ = 0.5
_G_WORLD = np.array([0.0, 0.0, -1.0])  # mjlab entity gravity_vec_w


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice | None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return None
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return slice(adr, adr + dim)


def _rl_imu_obs(model: mujoco.MjModel, data: mujoco.MjData, trunk_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Match mjlab actor terms (no mounting-misalignment DR).

    projected_gravity_b = quat_apply_inverse(root_quat, [0,0,-1])
    root_link_ang_vel_b = quat_apply_inverse(root_quat, cvel[root, 0:3])
    """
    R = data.xmat[trunk_id].reshape(3, 3)
    gravity_b = R.T @ _G_WORLD
    ang_vel_w = data.cvel[trunk_id, 0:3]
    ang_vel_b = R.T @ ang_vel_w
    return ang_vel_b, gravity_b


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
        mujoco.mj_forward(model, data)
        print("ctrl 跟踪 STAND（关节角与 view.py 相同）")

    print("全碰撞网格: geom group 3 打开，视觉半透明")
    print("IMU 坐标轴: 红=+X  绿=+Y  蓝=+Z  | 黄球=躯干 imu  品红=head_imu")
    print(f"每 {1.0 / _PRINT_HZ:.1f}s 打印 RL 观测 base_ang_vel / projected_gravity")
    print("STAND 下 IMU 朝向:")
    _print_site(model, data, "imu")
    _print_site(model, data, "head_imu")

    imu_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_imu")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro_sl = _sensor_slice(model, "imu_ang_vel")

    if imu_id >= 0:
        model.site_size[imu_id] = [_ORIGIN_R, 0.0, 0.0]
        model.site_rgba[imu_id] = [1.0, 0.85, 0.1, 1.0]
    if head_id >= 0:
        model.site_size[head_id] = [_ORIGIN_R * 0.8, 0.0, 0.0]
        model.site_rgba[head_id] = [0.9, 0.2, 0.85, 1.0]

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        viewer.opt.sitegroup[3] = 1
        for g in range(6):
            viewer.opt.geomgroup[g] = 1
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
        while viewer.is_running():
            step_start = time.time()
            if stand_ctrl is not None:
                data.ctrl[:] = stand_ctrl
            mujoco.mj_step(model, data)
            with viewer.lock():
                scn = viewer.user_scn
                scn.ngeom = 0
                if imu_id >= 0:
                    _draw_site_frame(scn, data, imu_id, (1.0, 0.85, 0.1, 1.0))
                if head_id >= 0:
                    _draw_site_frame(scn, data, head_id, (0.9, 0.2, 0.85, 1.0))
            viewer.sync()

            now = time.monotonic()
            if now - last_print >= 1.0 / _PRINT_HZ:
                last_print = now
                w_b, g_b = _rl_imu_obs(model, data, trunk_id)
                extra = ""
                if gyro_sl is not None:
                    w_imu = data.sensordata[gyro_sl]
                    extra = (
                        f"  gyro_site=[{w_imu[0]:+.3f} {w_imu[1]:+.3f} {w_imu[2]:+.3f}]"
                    )
                print(
                    f"t={data.time:7.2f}  "
                    f"ang_vel=[{w_b[0]:+.3f} {w_b[1]:+.3f} {w_b[2]:+.3f}] rad/s  "
                    f"gravity=[{g_b[0]:+.3f} {g_b[1]:+.3f} {g_b[2]:+.3f}]"
                    f"{extra}",
                    flush=True,
                )

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
