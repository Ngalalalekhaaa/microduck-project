#!/usr/bin/env python3
"""Record new sit/stand and ground-pick ONNX policies on two servo models.

Run with microduck_rl/.venv/bin/python. No robot hardware is imported.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

from render_hd1910_controller import load_scene

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "microduck_rl_hd1910"
ROBOT_DIR = REPO / "src/mjlab_microduck/robot/microduck"
INFERENCE = REPO / "scripts/infer_policy.py"
INPUT_DIR = Path("/home/luckysir/文档/xwechat_files/wxid_s317mgfyvs8k22_37de/msg/file/2026-10")
DEFAULT_SIT = INPUT_DIR / "model_1699_sitstand.onnx"
DEFAULT_GROUND = INPUT_DIR / "model_2249_ground_pick.onnx"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
ORIGINAL_SESSION = ort.InferenceSession


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_session(path, *args, **kwargs):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ORIGINAL_SESSION(str(path), options, providers=["CPUExecutionProvider"])


def load_inference():
    spec = importlib.util.spec_from_file_location("comparison_inference", INFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_descendant(model, body, ancestor):
    while body > 0:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def command_at(kind, t):
    if kind == "sitstand":
        sit = 3 <= t < 8 or 13 <= t < 18
        return np.array([float(sit), 0, 0], dtype=np.float32), "坐下 / 保持坐姿" if sit else "站起 / 保持站姿"
    if kind != "ground_pick":
        raise ValueError(kind)
    phase = (t % 4.0) / 4.0
    status = ("俯身" if phase < .375 else "低位保持" if phase < .425
              else "起身" if phase < .80 else "站立保持")
    return np.array([np.cos(2*np.pi*phase), np.sin(2*np.pi*phase), 0], dtype=np.float32), f"{status}  相位 {phase:.2f} / 4秒完整周期"


class Video:
    def __init__(self, path, fps):
        self.proc = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "1280x720",
            "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE)
        self.count = 0

    def write(self, frame):
        self.proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def draw_frame(renderer, model, data, variant, kind, t, state, status, cameras):
    xyz, tilt, velocity, first_fall = state
    canvas = Image.new("RGB", (1280, 720), (17, 25, 39))
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(FONT, 27)
    normal = ImageFont.truetype(FONT, 21)
    small = ImageFont.truetype(FONT, 18)
    name = "HD-1910 动力学" if variant == "hd1910" else "原版 XL330 动力学"
    task = "站立 ↔ 坐下 | 1699" if kind == "sitstand" else "俯身贴地 ↔ 起身 | 2249"
    draw.text((24, 7), f"Microduck  |  {name}  |  {task}", font=title, fill=(241, 246, 255))
    draw.text((24, 48), f"仿真 {t:05.2f} s    指令：{status}", font=normal, fill=(110, 213, 242))
    vis = mujoco.MjvOption()
    vis.geomgroup[3:] = 0
    for i, camera in enumerate(cameras):
        camera.lookat[:] = [xyz[0], xyz[1], .13]
        renderer.update_scene(data, camera=camera, scene_option=vis)
        # Render visual mesh group; collision group is deliberately hidden.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        canvas.paste(Image.fromarray(renderer.render()), (i * 640, 88))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 96), "斜前视角", font=small, fill="white", stroke_width=1, stroke_fill="black")
    draw.text((660, 96), "侧面视角", font=small, fill="white", stroke_width=1, stroke_fill="black")
    draw.line((640, 88, 640, 648), fill=(48, 65, 85), width=2)
    draw.text((24, 654), f"躯干高度 {xyz[2]*100:.1f} cm    倾角 {tilt:.1f}°    机身前向速度 {velocity[0]:+.2f} m/s", font=normal, fill=(235, 240, 250))
    if kind == "ground_pick":
        height = data.site("mouth_tip").xpos[2]
        draw.text((870, 654), f"嘴尖高度 {height*100:.1f} cm", font=small, fill=(170, 230, 200))
    elif first_fall is not None:
        draw.text((870, 654), f"倾角曾 >70°：{first_fall:.2f} s", font=small, fill=(255, 160, 120))
    draw.text((24, 687), "MuJoCo + BAM 名义工况 · 原速 · 无重置 · 两版相同几何和质量惯量，替换舵机动力学", font=small, fill=(157, 175, 197))
    return canvas


def record(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.index}_{args.variant}_{args.kind}"
    policy_path = (args.sitstand if args.kind == "sitstand" else args.ground_pick).resolve()
    scene = ROBOT_DIR / "scene.xml"
    init_log = io.StringIO()
    with redirect_stdout(init_log):
        module = load_inference()
        model, data, controller, physics = load_scene(str(scene), args.variant, timestep=.005, lag_steps=4)
        model.opt.iterations = 30
        model.opt.ls_iterations = 50
        model.vis.global_.offwidth = 640
        model.vis.global_.offheight = 560
        with patch.object(module.ort, "InferenceSession", cpu_session):
            policy = module.PolicyInference(
                model, data, action_scale=1.0, bam_ctrl=controller,
                new_cmd_obs=True, use_projected_gravity=True,
                **{("sitstand_onnx_path" if args.kind == "sitstand" else "walking_onnx_path"): str(policy_path)},
            )
    (output / f"{prefix}.init.log").write_text(init_log.getvalue())
    session = policy.ort_session
    metadata = session.get_modelmeta().custom_metadata_map
    assert session.get_inputs()[0].shape == [1, 61]
    assert session.get_outputs()[0].shape == [1, 14]
    names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    assert names == metadata["joint_names"].split(",")
    np.testing.assert_allclose(policy.default_pose, np.fromstring(metadata["default_joint_pos"], sep=","), atol=.00051)
    assert float(metadata["action_scale"]) == 1.0
    root = model.joint("trunk_base_freejoint")
    qa, va = int(root.qposadr[0]), int(root.dofadr[0])
    mujoco.mj_resetData(model, data)
    data.qpos[qa:qa+7] = [0, 0, .12, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    initial_xyz = data.qpos[qa:qa+3].copy()
    joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=int)
    joint_ranges = model.jnt_range[joint_ids].copy()
    head_bodies = {i for i in range(model.nbody)
                   if _is_descendant(model, i, int(model.body("neck").id))}
    ground_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0}
    cameras = []
    for azimuth in (145, 90):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = .57
        camera.azimuth = azimuth
        camera.elevation = -13
        cameras.append(camera)

    trace = {k: [] for k in ("time", "command", "obs", "action", "xyz", "body_velocity", "tilt_deg", "qpos", "ncon",
                            "joint_pos", "joint_vel", "joint_target", "joint_torque", "mouth_height", "mouth_down_alignment", "head_ground_contact", "head_ground_force_n")}
    first_fall = None
    finite = True
    fps = 25
    writer = Video(output / f"{prefix}.mp4", fps) if not args.no_video else None
    next_frame = 0
    snapshots = set()
    started = time.perf_counter()
    with mujoco.Renderer(model, height=560, width=640) as renderer:
        for step in range(round(args.seconds / .02)):
            t = round(step * .02, 8)
            command, status = command_at(args.kind, t)
            if args.kind == "sitstand":
                policy.sit_mode = bool(command[0])
                policy._update_command()
            else:
                # The upstream helper exits at phase .7. Drive all of [0,1)
                # directly so the trained rise phase (.425-.80) completes.
                policy.command[:] = 0
                policy.command[:3] = command
            obs = policy.get_observations().copy()
            np.testing.assert_allclose(obs[48:51], command, atol=1e-7)
            assert np.all(obs[51:] == 0)
            if not np.isfinite(obs).all():
                finite = False
                break
            action = policy.infer()
            if not np.isfinite(action).all():
                finite = False
                break
            policy.apply_action(action)
            head_touch = False
            head_force = 0.0
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model, data)
                for ci in range(data.ncon):
                    contact = data.contact[ci]
                    g1, g2 = int(contact.geom1), int(contact.geom2)
                    body1, body2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                    if ((g1 in ground_geoms and body2 in head_bodies) or
                            (g2 in ground_geoms and body1 in head_bodies)):
                        head_touch = True
                        force = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, force)
                        head_force = max(head_force, float(np.linalg.norm(force[:3])))
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                finite = False
                break
            xyz = data.qpos[qa:qa+3].copy()
            quat = data.qpos[qa+3:qa+7].copy()
            gravity = policy.quat_rotate_inverse(quat, np.array([0., 0., -1.]))
            tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
            velocity = policy.quat_rotate_inverse(quat, data.qvel[va:va+3]).copy()
            if tilt > 70 and first_fall is None:
                first_fall = float(data.time)
            mouth = data.site("mouth_tip")
            for key, value in zip(trace, (float(data.time), command, obs, action, xyz, velocity, tilt, data.qpos.copy(), int(data.ncon),
                                         data.qpos[policy.joint_qpos_indices].copy(), data.qvel[policy.joint_qvel_indices].copy(),
                                         controller.q_target.copy(), data.actuator_force.copy(), float(mouth.xpos[2]),
                                         float(-mouth.xmat.reshape(3,3)[2,0]), head_touch, head_force)):
                trace[key].append(value)
            if writer and step % 2 == 1:
                frame = draw_frame(renderer, model, data, args.variant, args.kind, float(data.time), (xyz, tilt, velocity, first_fall), status, cameras)
                writer.write(frame)
                marker = int(round(data.time * 100))
                if marker in (100, 400, 700, 1000, 1500, 2000, 2300) and marker not in snapshots:
                    frame.save(output / f"{prefix}_{marker/100:05.2f}s.jpg", quality=90)
                    snapshots.add(marker)
                next_frame += 1
            if step % 250 == 249:
                print(json.dumps({"clip": prefix, "sim_s": round(float(data.time), 2), "tilt": round(tilt, 2), "z": round(float(xyz[2]), 3), "wall_s": round(time.perf_counter()-started, 1)}), flush=True)
    if writer:
        writer.close()
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    np.savez_compressed(output / f"{prefix}.npz", **arrays)
    summaries = []
    windows = [(0, 3), (3, 8), (8, 13), (13, 18), (18, 24)] if args.kind == "sitstand" else [(t, t+4) for t in range(0, int(args.seconds), 4)]
    for lo, hi in windows:
        valid = (arrays["time"] > lo + 1) & (arrays["time"] <= hi)
        if not valid.any():
            continue
        summaries.append({"window_s": [lo, hi], "measurement_s": [lo+1, hi],
                          "mean_z_m": float(arrays["xyz"][valid, 2].mean()),
                          "mean_tilt_deg": float(arrays["tilt_deg"][valid].mean()),
                          "mean_body_velocity_m_s": arrays["body_velocity"][valid].mean(axis=0).tolist(),
                          "mean_mouth_height_m": float(arrays["mouth_height"][valid].mean()),
                          "min_mouth_height_m": float(arrays["mouth_height"][valid].min()),
                          "head_ground_contact_frames": int(arrays["head_ground_contact"][valid].sum())})
    cycles = []
    if args.kind == "ground_pick":
        for start in range(0, int(args.seconds), 4):
            cycle_mask = (arrays["time"] > start) & (arrays["time"] <= start+4)
            return_mask = (arrays["time"] > start+3.5) & (arrays["time"] <= start+4)
            if not return_mask.any():
                continue
            cycles.append({"window_s": [start,start+4],
                           "min_mouth_height_m": float(arrays["mouth_height"][cycle_mask].min()),
                           "max_mouth_down_alignment": float(arrays["mouth_down_alignment"][cycle_mask].max()),
                           "return_z_m": float(arrays["xyz"][return_mask,2].mean()),
                           "return_tilt_deg": float(arrays["tilt_deg"][return_mask].mean()),
                           "return_joint_home_rms_deg": float(np.degrees(np.sqrt(np.mean((arrays["joint_pos"][return_mask]-policy.default_pose)**2)))),
                           "head_ground_contact_frames": int(arrays["head_ground_contact"][cycle_mask].sum())})
    report = {
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "variant": args.variant, "kind": args.kind, "policy": str(policy_path), "policy_sha256": sha256(policy_path),
        "metadata": metadata, "scene": str(scene), "scene_sha256": sha256(scene),
        "robot_xml_sha256": sha256(ROBOT_DIR / "robot_groundcontact.xml"),
        "inference_source": str(INFERENCE), "inference_sha256": sha256(INFERENCE),
        "renderer_source_sha256": sha256(__file__), "controller_source_sha256": sha256(Path(__file__).with_name("render_hd1910_controller.py")),
        "physics": physics, "solver_iterations": int(model.opt.iterations), "solver_ls_iterations": int(model.opt.ls_iterations),
        "policy_home_rad": policy.default_pose.tolist(), "control_hz": 50, "decimation": 4,
        "initial_root_xyz": initial_xyz.tolist(), "requested_sim_seconds": args.seconds,
        "completed_sim_seconds": round(float(data.time), 8), "finite": finite,
        "first_tilt_above_70deg_s": first_fall, "maximum_tilt_deg": float(arrays["tilt_deg"].max()),
        "displacement_xyz_m": (arrays["xyz"][-1] - initial_xyz).tolist(),
        "window_metrics": summaries, "video_frames": writer.count if writer else 0, "video_fps": fps,
        "ground_pick_cycles": cycles,
        "joint_names": names, "joint_ranges_rad": joint_ranges.tolist(),
        "joint_measured_min_rad": arrays["joint_pos"].min(axis=0).tolist(),
        "joint_measured_max_rad": arrays["joint_pos"].max(axis=0).tolist(),
        "max_joint_speed_rad_s": float(np.abs(arrays["joint_vel"]).max()),
        "max_target_step_deg": float(np.degrees(np.abs(np.diff(arrays["joint_target"],axis=0)).max())),
        "max_abs_joint_torque_nm": float(np.abs(arrays["joint_torque"]).max()),
        "max_head_ground_force_n": float(arrays["head_ground_force_n"].max()),
        "robot_total_mass_kg": float(model.body_mass.sum()),
        "command_schedule": ("stand 0-3s, sit 3-8s, stand 8-13s, sit 13-18s, stand 18-24s" if args.kind == "sitstand"
                             else "continuous complete 4s cycles; twist=[cos(2pi*phase),sin(2pi*phase),0]; no early .7 cutoff"),
        "wall_seconds": time.perf_counter()-started,
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": ort.__version__},
        "limitations": [
            "HD1910 uses public LuwuDynamics 1910 M6 servo parameters; robot geometry/mass/inertia remain the original Microduck values, not a measured physical replica.",
            "Nominal deterministic CPU MuJoCo/BAM rollout; not exact MuJoCo Warp training replay or hardware validation.",
            "Both models use plain non-backlash geometry; the +/-1 degree backlash assumption is not enabled.",
            "No pushes, randomized voltage/friction, observation noise, IMU misalignment, or joint-velocity observation lag. Actuator delay is fixed at four physics steps.",
            "XML groundcontact collisions retained; named collision overrides and feet contact parameters applied. Contact/solver behavior may differ from assembled training tasks.",
            "The two ONNX run_path values mention HD1910, but their complete original training configurations were not supplied; names alone cannot verify actuator parameters.",
            "Ground-pick simulated without added mouth payload. Intentional body lean is not a fall criterion; inspect completed return phases and contact video.",
            "Both camera views show the same continuous rollout; no reset or hidden cut after a fall.",
        ],
    }
    (output / f"{prefix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": prefix, "finite": finite, "first_fall_s": first_fall, "output": str(output)}, ensure_ascii=False), flush=True)
    if not finite or abs(data.time - args.seconds) > .005:
        raise RuntimeError("rollout incomplete or nonfinite")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["hd1910", "original"], required=True)
    p.add_argument("--kind", choices=["sitstand", "ground_pick"], required=True)
    p.add_argument("--index", default="1")
    p.add_argument("--sitstand", type=Path, default=DEFAULT_SIT)
    p.add_argument("--ground-pick", type=Path, default=DEFAULT_GROUND)
    p.add_argument("--seconds", type=float, default=24)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "videos/20261023_action_comparison")
    record(p.parse_args())


if __name__ == "__main__":
    main()
