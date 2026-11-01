#!/usr/bin/env python3
"""Record nominal CPU MuJoCo rollouts of the same ONNX on two servo models.

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
DEFAULT_SIT = Path("/home/luckysir/文档/xwechat_files/wxid_s317mgfyvs8k22_37de/msg/file/2026-10/2026-10-15_07-49-15_b1f68ec66a874b42a079217f1787d081.onnx")
DEFAULT_WALK = ROOT / "deploy_hd1910/models/hd1910_backlash_30000.onnx"
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


def command_at(kind, t):
    if kind == "sitstand":
        sit = 3 <= t < 8 or 13 <= t < 18
        return np.array([float(sit), 0, 0], dtype=np.float32), "坐下 / 保持坐姿" if sit else "站起 / 保持站姿"
    if t < 3:
        vx = 0.0
    elif t < 5:
        vx = .15 * (t - 3) / 2
    elif t < 10:
        vx = .15
    elif t < 12:
        vx = .15 + .15 * (t - 10) / 2
    elif t < 19:
        vx = .30
    elif t < 21:
        vx = .30 * (21 - t) / 2
    else:
        vx = 0.0
    return np.array([vx, 0, 0], dtype=np.float32), f"前进目标 {vx:.2f} m/s" if vx else "零速度 / 站立"


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
    task = "站立 ↔ 坐下" if kind == "sitstand" else "行走：当前部署 30000 权重"
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
    if first_fall is not None:
        draw.text((870, 654), f"曾倾覆：{first_fall:.2f} s (>70°)", font=small, fill=(255, 160, 120))
    draw.text((24, 687), "MuJoCo + BAM 名义工况 · 原速 · 连续仿真无重置 · 两版沿用相同几何和质量惯量", font=small, fill=(157, 175, 197))
    return canvas


def record(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.index}_{args.variant}_{args.kind}"
    policy_path = (args.sitstand if args.kind == "sitstand" else args.walking).resolve()
    scene = ROBOT_DIR / ("scene.xml" if args.kind == "sitstand" else "scene_walk.xml")
    init_log = io.StringIO()
    with redirect_stdout(init_log):
        module = load_inference()
        model, data, controller, physics = load_scene(str(scene), args.variant, timestep=.005, lag_steps=4)
        model.opt.iterations = 30 if args.kind == "sitstand" else 10
        model.opt.ls_iterations = 50 if args.kind == "sitstand" else 20
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
    cameras = []
    for azimuth in (145, 90):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance = .57
        camera.azimuth = azimuth
        camera.elevation = -13
        cameras.append(camera)

    trace = {k: [] for k in ("time", "command", "obs", "action", "xyz", "body_velocity", "tilt_deg", "qpos", "ncon")}
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
            else:
                policy.vel_cmd[:] = command
            policy._update_command()
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
            for _ in range(4):
                controller.update()
                mujoco.mj_step(model, data)
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
            for key, value in zip(trace, (float(data.time), command, obs, action, xyz, velocity, tilt, data.qpos.copy(), int(data.ncon))):
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
    windows = [(0, 3), (3, 8), (8, 13), (13, 18), (18, 24)] if args.kind == "sitstand" else [(0, 3), (5, 10), (12, 19), (21, 24)]
    for lo, hi in windows:
        valid = (arrays["time"] > lo + 1) & (arrays["time"] <= hi)
        if not valid.any():
            continue
        summaries.append({"window_s": [lo, hi], "measurement_s": [lo+1, hi],
                          "mean_z_m": float(arrays["xyz"][valid, 2].mean()),
                          "mean_tilt_deg": float(arrays["tilt_deg"][valid].mean()),
                          "mean_body_velocity_m_s": arrays["body_velocity"][valid].mean(axis=0).tolist()})
    report = {
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "variant": args.variant, "kind": args.kind, "policy": str(policy_path), "policy_sha256": sha256(policy_path),
        "metadata": metadata, "scene": str(scene), "scene_sha256": sha256(scene),
        "robot_xml_sha256": sha256(ROBOT_DIR / ("robot_groundcontact.xml" if args.kind == "sitstand" else "robot_walk.xml")),
        "inference_source": str(INFERENCE), "inference_sha256": sha256(INFERENCE),
        "renderer_source_sha256": sha256(__file__), "controller_source_sha256": sha256(Path(__file__).with_name("render_hd1910_controller.py")),
        "physics": physics, "solver_iterations": int(model.opt.iterations), "solver_ls_iterations": int(model.opt.ls_iterations),
        "policy_home_rad": policy.default_pose.tolist(), "control_hz": 50, "decimation": 4,
        "initial_root_xyz": initial_xyz.tolist(), "requested_sim_seconds": args.seconds,
        "completed_sim_seconds": round(float(data.time), 8), "finite": finite,
        "first_tilt_above_70deg_s": first_fall, "maximum_tilt_deg": float(arrays["tilt_deg"].max()),
        "displacement_xyz_m": (arrays["xyz"][-1] - initial_xyz).tolist(),
        "window_metrics": summaries, "video_frames": writer.count if writer else 0, "video_fps": fps,
        "wall_seconds": time.perf_counter()-started,
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": ort.__version__},
        "limitations": [
            "HD1910 uses public LuwuDynamics 1910 M6 servo parameters; robot geometry/mass/inertia remain the original Microduck values, not a measured physical replica.",
            "Nominal deterministic CPU MuJoCo/BAM rollout; not exact MuJoCo Warp training replay or hardware validation.",
            "Both models use plain non-backlash geometry; the +/-1 degree backlash assumption is not enabled.",
            "No pushes, randomized voltage/friction, observation noise, IMU misalignment, or joint-velocity observation lag. Actuator delay is fixed at four physics steps.",
            "XML groundcontact collisions retained; named collision overrides and feet contact parameters applied. Contact/solver behavior may differ from assembled training tasks.",
            "The walking checkpoint's complete original training configuration was not supplied; its export manifest records this.",
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
    p.add_argument("--kind", choices=["sitstand", "walking"], required=True)
    p.add_argument("--index", default="1")
    p.add_argument("--sitstand", type=Path, default=DEFAULT_SIT)
    p.add_argument("--walking", type=Path, default=DEFAULT_WALK)
    p.add_argument("--seconds", type=float, default=24)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "videos/20261017_policy_comparison")
    record(p.parse_args())


if __name__ == "__main__":
    main()
