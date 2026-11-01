#!/usr/bin/env python3
"""Reproducible CPU BAM HD1910 walk evaluation and saved 25 fps dual-view MP4s.

microduck_rl/.venv/bin/python scripts/evaluate_walk_20261028.py
microduck_rl/.venv/bin/python scripts/evaluate_walk_20261028.py --render

Every case starts independently; each saved video is one uninterrupted rollout.
No hardware access, action clipping, or action EMA is used.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime
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
from PIL import Image, ImageDraw, ImageFont

from render_action_comparison import cpu_session, load_inference, sha256
from render_hd1910_controller import load_scene

ROOT = Path(__file__).resolve().parents[1]
ROBOT_DIR = ROOT / "microduck_rl_hd1910/src/mjlab_microduck/robot/microduck"
SCENE = ROBOT_DIR / "scene_walk.xml"
MODEL = ROOT / "reports/20261028_model_suite/input/model_1849_walk.onnx"
REPORT = ROOT / "reports/20261028_model_suite/walk"
VIDEOS = ROOT / "videos/20261028_model_suite/walk"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CASES = {
    "stand": (0., 0., 0.),
    "forward_010": (.1, 0., 0.),
    "forward_020": (.2, 0., 0.),
    "forward_030": (.3, 0., 0.),
    "backward_010": (-.1, 0., 0.),
    "left_010": (0., .1, 0.),
    "right_010": (0., -.1, 0.),
    "turn_left_050": (0., 0., .5),
    "turn_right_050": (0., 0., -.5),
}
LABELS = dict(zip(CASES, ("零速站立", "前进 0.10", "前进 0.20", "前进 0.30", "后退 0.10", "左移 0.10", "右移 0.10", "左转 0.50", "右转 0.50")))


def build(seed):
    """Seed 0 is nominal; 1/2 are explicitly sampled sensitivity cases."""
    rng = np.random.default_rng(seed)
    lag = 4 if seed == 0 else (3 if seed == 1 else 6)
    model, data, controller, physics = load_scene(SCENE, "hd1910", lag_steps=lag)
    model.opt.iterations, model.opt.ls_iterations = 10, 20
    model.vis.global_.offwidth, model.vis.global_.offheight = 640, 560
    conditions = {"seed": seed, "nominal": seed == 0, "lag_steps": lag,
                  "initial_rpy_rad": [0., 0., 0.], "initial_joint_noise_rad": [0.]*14,
                  "friction_scale": [1.]*14, "com_x_offsets_m": {}}
    if seed:
        # Scale BAM's computed velocity-independent friction budget, matching
        # FrictionDRBamActuator; scaling dof_frictionloss before update is a no-op.
        fs = rng.uniform(.8, 1.2, model.nu)
        original = controller.model.compute_frictions
        def scaled_friction(*args, **kwargs):
            frictionloss, damping = original(*args, **kwargs)
            return np.asarray(frictionloss) * fs, damping
        controller.model.compute_frictions = scaled_friction
        conditions["friction_scale"] = fs.tolist()
        selected = [i for i in range(1, model.nbody)
                    if model.body(i).name in {"trunk_base", "neck", "neck_pitch", "yaw_roll_motion", "bottom_head_shell", "jaw_soft"}]
        # Local-X offsets independently on the named trunk/head bodies. These
        # are sensitivity samples, not a claim to reproduce the sender's DR.
        for bid in selected:
            dx = float(rng.uniform(-.015, .015))
            model.body_ipos[bid, 0] += dx
            conditions["com_x_offsets_m"][model.body(bid).name] = dx
        mujoco.mj_setConst(model, data)
        conditions["initial_rpy_rad"] = [*rng.uniform(-np.deg2rad(2), np.deg2rad(2), 2), 0.]
        conditions["initial_joint_noise_rad"] = rng.uniform(-.01, .01, 14).tolist()
    return model, data, controller, physics, conditions


def evaluate(case, seed, seconds):
    stem = f"walk1849_{case}_seed{seed}"
    started = time.perf_counter()
    init = io.StringIO()
    with redirect_stdout(init):
        module = load_inference()
        model, data, controller, physics, conditions = build(seed)
        with patch.object(module.ort, "InferenceSession", cpu_session):
            policy = module.PolicyInference(model, data, walking_onnx_path=str(MODEL),
                                           action_scale=1., bam_ctrl=controller,
                                           new_cmd_obs=True, use_projected_gravity=True)
    (REPORT / f"{stem}.init.log").write_text(init.getvalue())
    metadata = policy.ort_session.get_modelmeta().custom_metadata_map
    names = [model.joint(int(i)).name for i in model.actuator_trnid[:, 0]]
    assert names == metadata["joint_names"].split(",")
    assert policy.ort_session.get_inputs()[0].shape == [1, 61]
    assert policy.ort_session.get_outputs()[0].shape == [1, 14]
    np.testing.assert_allclose(policy.default_pose, np.fromstring(metadata["default_joint_pos"], sep=","), atol=.00051)
    assert float(metadata["action_scale"]) == 1.
    qa = int(model.joint("trunk_base_freejoint").qposadr[0])
    va = int(model.joint("trunk_base_freejoint").dofadr[0])
    mujoco.mj_resetData(model, data)
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.asarray(conditions["initial_rpy_rad"]), "xyz")
    data.qpos[qa:qa+7] = [0., 0., .12, *quat]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose + np.asarray(conditions["initial_joint_noise_rad"])
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    keys = ("time", "command", "obs", "action", "xyz", "body_velocity", "body_ang_velocity", "tilt_deg",
            "qpos", "qvel", "joint_pos", "joint_vel", "joint_target", "joint_torque", "ncon")
    trace = {key: [] for key in keys}
    first_fall = None
    first_low = None
    finite = True
    for step in range(round(seconds / .02)):
        # 0-2 s stand, 2-3 s command ramp, 3 s onward constant command.
        # Metrics begin at 4 s, excluding initialization and the ramp.
        t = step * .02
        command = np.asarray(CASES[case], dtype=np.float32) * min(1., max(0., t-2.))
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
        xyz, quat = data.qpos[qa:qa+3].copy(), data.qpos[qa+3:qa+7].copy()
        gravity = policy.quat_rotate_inverse(quat, np.array([0., 0., -1.]))
        tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
        velocity = policy.quat_rotate_inverse(quat, data.qvel[va:va+3]).copy()
        gyro = policy.get_base_ang_vel().copy()
        if tilt > 70. and first_fall is None:
            first_fall = round(float(data.time), 6)
        if xyz[2] < .065 and first_low is None:
            first_low = round(float(data.time), 6)
        values = (float(data.time), command, obs, action, xyz, velocity, gyro, tilt,
                  data.qpos.copy(), data.qvel.copy(), data.qpos[policy.joint_qpos_indices].copy(),
                  data.qvel[policy.joint_qvel_indices].copy(), controller.q_target.copy(), data.actuator_force.copy(), int(data.ncon))
        for key, value in zip(keys, values):
            trace[key].append(value)
    a = {key: np.asarray(value) for key, value in trace.items()}
    np.savez_compressed(REPORT / f"{stem}.npz", **a)
    window = a["time"] >= 4.
    if not window.any():
        raise RuntimeError(f"{stem}: incomplete rollout before measurement window")
    target = np.asarray(CASES[case])
    measured = np.column_stack((a["body_velocity"][:, :2], a["body_ang_velocity"][:, 2]))
    ranges = model.jnt_range[model.actuator_trnid[:, 0]].copy()
    over_target = (a["joint_target"] < ranges[:, 0]) | (a["joint_target"] > ranges[:, 1])
    over_pos = (a["joint_pos"] < ranges[:, 0]-.01) | (a["joint_pos"] > ranges[:, 1]+.01)
    jumps = np.abs(np.diff(np.vstack((policy.default_pose, a["joint_target"])), axis=0))
    result = {
        "case": case, "stem": stem, "seed": seed, "conditions": conditions,
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "policy": str(MODEL), "policy_sha256": sha256(MODEL), "metadata": metadata,
        "physics": physics, "scene_sha256": sha256(SCENE), "robot_xml_sha256": sha256(ROBOT_DIR / "robot_walk.xml"),
        "source_sha256": sha256(__file__), "controller_sha256": sha256(Path(__file__).with_name("render_hd1910_controller.py")),
        "inference_sha256": sha256(ROOT / "microduck_rl_hd1910/scripts/infer_policy.py"),
        "control_hz": 50, "seconds_requested": seconds, "seconds_completed": float(data.time),
        "command_vx_vy_wz": target.tolist(), "measurement_window_s": [4., seconds],
        "mean_measured_vx_vy_wz": measured[window].mean(axis=0).tolist(),
        "rms_error_vx_vy_wz": np.sqrt(np.mean((measured[window]-target)**2, axis=0)).tolist(),
        "first_tilt_above_70deg_s": first_fall, "first_trunk_below_065m_s": first_low,
        "max_tilt_deg": float(a["tilt_deg"].max()), "steady_mean_tilt_deg": float(a["tilt_deg"][window].mean()),
        "steady_mean_trunk_z_m": float(a["xyz"][window, 2].mean()),
        "end_xyz_m": a["xyz"][-1].tolist(), "finite": finite,
        "joint_names": names, "joint_ranges_rad": ranges.tolist(),
        "joint_target_min_rad": a["joint_target"].min(axis=0).tolist(),
        "joint_target_max_rad": a["joint_target"].max(axis=0).tolist(),
        "joint_pos_min_rad": a["joint_pos"].min(axis=0).tolist(),
        "joint_pos_max_rad": a["joint_pos"].max(axis=0).tolist(),
        "target_exceeds_xml_range_frames_per_joint": over_target.sum(axis=0).tolist(),
        "measured_exceeds_xml_range_by_001rad_frames_per_joint": over_pos.sum(axis=0).tolist(),
        "max_target_jump_rad": float(jumps.max()), "max_target_jump_per_joint_rad": jumps.max(axis=0).tolist(),
        "max_joint_speed_rad_s": float(np.abs(a["joint_vel"]).max()),
        "max_joint_torque_nm": float(np.abs(a["joint_torque"]).max()),
        "wall_seconds": time.perf_counter()-started,
        "limitations": [
            "Local scene_walk.xml and public BAM HD1910 M6; sender's original training configuration was not supplied.",
            "CPU MuJoCo comparison, not exact Warp replay. Original Microduck geometry/mass/inertia, no calibrated physical robot model.",
            "No physical backlash, IMU misalignment, observation noise, or one-control-step joint-velocity observation delay.",
            "Seed 0 is nominal; seeds 1/2 combine initial ±2deg tilt/±.01rad joints, BAM friction .8-1.2, local-X trunk/head CoM ±.015m, actuator lag 3 or 6 steps.",
            "XML target overshoot is reported, not clipped: low-gain BAM allows targets beyond mechanical joint range by design.",
            "Encoder limits require the actual robot's latest calibration; target arrays are saved for separate offline conversion.",
            "Fall flag is tilt >70deg or trunk height <.065m, not a deployment-ready verdict. Check actual tracking and videos.",
        ],
    }
    (REPORT / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ("stem", "mean_measured_vx_vy_wz", "max_tilt_deg", "first_tilt_above_70deg_s", "finite", "wall_seconds")}), flush=True)
    return result


def render(stem):
    report = json.loads((REPORT / f"{stem}.json").read_text())
    a = np.load(REPORT / f"{stem}.npz")
    with redirect_stdout(io.StringIO()):
        model, data, _, _, _ = build(report["seed"])
    video = VIDEOS / f"{stem}.mp4"
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                             "-pixel_format", "rgb24", "-video_size", "1280x720", "-framerate", "25", "-i", "-",
                             "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", str(video)], stdin=subprocess.PIPE)
    fonts = [ImageFont.truetype(FONT, size) for size in (25, 20, 17)]
    cameras = []
    for azimuth in (145, 90):
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.azimuth, c.elevation = .62, azimuth, -13
        cameras.append(c)
    vis = mujoco.MjvOption()
    vis.geomgroup[3:] = 0
    with mujoco.Renderer(model, height=560, width=640) as renderer:
        for i in range(1, len(a["time"]), 2):
            data.qpos[:] = a["qpos"][i]
            data.qvel[:] = a["qvel"][i]
            data.time = float(a["time"][i])
            mujoco.mj_forward(model, data)
            frame = Image.new("RGB", (1280, 720), (17, 25, 39))
            for j, camera in enumerate(cameras):
                camera.lookat[:] = [*a["xyz"][i, :2], .13]
                renderer.update_scene(data, camera=camera, scene_option=vis)
                frame.paste(Image.fromarray(renderer.render()), (j*640, 88))
            d = ImageDraw.Draw(frame)
            condition = "名义工况" if report["seed"] == 0 else f"扰动 seed {report['seed']}"
            d.text((20, 6), f"Microduck | model_1849_walk.onnx | {LABELS[report['case']]} | {condition}", fill="white", font=fonts[0])
            vx, vy, wz = a["command"][i]
            d.text((20, 45), f"{data.time:05.2f} s    指令 vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  wz={wz:+.2f} rad/s", fill=(110,213,242), font=fonts[1])
            d.text((20, 96), "斜前视角", fill="white", font=fonts[2], stroke_width=1, stroke_fill="black")
            d.text((660, 96), "侧面视角", fill="white", font=fonts[2], stroke_width=1, stroke_fill="black")
            actual = a["body_velocity"][i]
            d.text((20, 650), f"实测 vx={actual[0]:+.2f}  vy={actual[1]:+.2f} m/s  wz={a['body_ang_velocity'][i,2]:+.2f} rad/s    倾角 {a['tilt_deg'][i]:.1f}°    躯干 {a['xyz'][i,2]*100:.1f} cm", fill=(235,240,250), font=fonts[1])
            d.text((20, 685), "CPU MuJoCo+BAM HD1910 · 原速 · 本片连续无重置 · 0–2秒站立 / 2–3秒指令渐变 · 仅本地匹配模型仿真", fill=(157,175,197), font=fonts[2])
            proc.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
            if i == len(a["time"])-1 or abs(data.time-8.) < .01:
                frame.save(VIDEOS / f"{stem}_{data.time:05.2f}s.jpg", quality=90)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed: {video}")
    report["video"] = str(video)
    report["video_sha256"] = sha256(video)
    report["video_frames"] = len(range(1, len(a["time"]), 2))
    report["video_fps"] = 25
    (REPORT / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"saved_video": str(video), "frames": report["video_frames"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render previously saved nominal traces without rerunning inference")
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=12.)
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else ([0] if args.render else [0,1,2])
    if args.seconds <= 4. or round(args.seconds/.04)*.04 != args.seconds:
        parser.error("seconds must be >4 and a multiple of .04")
    if any(seed not in (0,1,2) for seed in seeds):
        parser.error("supported seeds: 0 (nominal), 1 and 2 (sensitivity samples)")
    REPORT.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for case in args.cases:
            if args.render:
                render(f"walk1849_{case}_seed{seed}")
            else:
                evaluate(case, seed, args.seconds)


if __name__ == "__main__":
    main()
