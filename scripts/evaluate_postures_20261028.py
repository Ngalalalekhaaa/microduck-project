#!/usr/bin/env python3
"""Reproducible CPU/BAM posture evaluation; no robot hardware access.

Run with microduck_rl/.venv/bin/python. Baseline recordings are 1280x720/25 fps.
Perturbations are a deterministic sensitivity test, not recovered training DR.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
import io
import json
import os
from pathlib import Path
import shlex
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import ImageDraw, ImageFont

from render_action_comparison import (
    FONT, INFERENCE, ROBOT_DIR, ROOT, Video, _is_descendant, command_at,
    cpu_session, draw_frame, load_inference, sha256,
)
from render_hd1910_controller import load_scene

MODELS = {
    "sitstand_1699": ("model_1699_sitstand.onnx", "sitstand"),
    "sitstand_friction_2598": ("model_2598_sitstand_friction.onnx", "sitstand"),
    "sitstand_com_2598": ("model_2598_sitstand_com.onnx", "sitstand"),
    "ground_pick_2249": ("model_2249_ground_pick.onnx", "ground_pick"),
}
CONDITIONS = {
    "baseline": (1.0, 0.0),
    "friction_0p8": (0.8, 0.0),
    "friction_1p2": (1.2, 0.0),
    "com_x_minus_15mm": (1.0, -0.015),
    "com_x_plus_15mm": (1.0, 0.015),
}
COM_BODIES = ("trunk_base", "neck", "neck_pitch", "yaw_roll_motion", "jaw_soft")
DT, DECIMATION, FPS = 0.005, 4, 25


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def evaluate(args, model_key, condition):
    filename, kind = MODELS[model_key]
    friction_scale, com_x = CONDITIONS[condition]
    seconds = 13.0 if kind == "sitstand" else 12.0
    policy_path = (args.input / filename).resolve()
    prefix = f"{model_key}__{condition}"
    output = args.output.resolve()
    video_dir = args.videos.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    init_log = io.StringIO()
    with redirect_stdout(init_log):
        module = load_inference()
        model, data, controller, physics = load_scene(
            ROBOT_DIR / "scene.xml", "hd1910", timestep=DT, lag_steps=4)
        model.opt.iterations = 30
        model.opt.ls_iterations = 50
        model.vis.global_.offwidth = 640
        model.vis.global_.offheight = 560
        modified_com = []
        for name in COM_BODIES:
            body = model.body(name)
            before = body.ipos.copy()
            body.ipos[0] += com_x
            modified_com.append({"body": name, "before_body_ipos": before.tolist(),
                                 "after_body_ipos": body.ipos.copy().tolist()})
        mujoco.mj_setConst(model, data)
        with patch.object(module.ort, "InferenceSession", cpu_session):
            policy = module.PolicyInference(
                model, data, action_scale=1.0, bam_ctrl=controller,
                new_cmd_obs=True, use_projected_gravity=True,
                **{("sitstand_onnx_path" if kind == "sitstand" else "walking_onnx_path"): str(policy_path)},
            )
    (output / f"{prefix}.init.log").write_text(init_log.getvalue())
    metadata = policy.ort_session.get_modelmeta().custom_metadata_map
    names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
    assert policy.ort_session.get_inputs()[0].shape == [1, 61]
    assert policy.ort_session.get_outputs()[0].shape == [1, 14]
    assert names == metadata["joint_names"].split(",")
    np.testing.assert_allclose(policy.default_pose, np.fromstring(metadata["default_joint_pos"], sep=","), atol=.00051)
    assert float(metadata["action_scale"]) == 1.0
    assert not policy.use_delay, "Only physics-step BAM delay must be active"
    root = model.joint("trunk_base_freejoint")
    qa, va = int(root.qposadr[0]), int(root.dofadr[0])
    mujoco.mj_resetData(model, data)
    data.qpos[qa:qa+7] = [0, 0, .12, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    controller.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    head_bodies = {i for i in range(model.nbody) if _is_descendant(model, i, int(model.body("neck").id))}
    ground_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == 0}
    trace = {key: [] for key in (
        "time", "command", "obs", "action", "xyz", "body_velocity", "tilt_deg",
        "qpos", "joint_pos", "joint_vel", "joint_target", "applied_joint_target",
        "joint_torque", "mouth_height", "mouth_down_alignment", "head_ground_contact",
        "head_ground_force_n", "bam_frictionloss", "bam_damping")}
    video = condition == "baseline" and not args.no_video or args.all_videos
    writer = Video(video_dir / f"{prefix}.mp4", FPS) if video else None
    cameras = []
    for azimuth in (135, 90):
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.azimuth, camera.elevation = .57, azimuth, -13
        cameras.append(camera)
    started = time.perf_counter()
    first_tilt_70 = None
    finite = True
    renderer_context = mujoco.Renderer(model, height=560, width=640) if writer else nullcontext()
    try:
        with renderer_context as renderer:
            for step in range(round(seconds / (DT * DECIMATION))):
                t = round(step * DT * DECIMATION, 8)
                command, status = command_at(kind, t)
                if kind == "sitstand":
                    policy.sit_mode = bool(command[0])
                    policy._update_command()
                else:
                    # All four seconds are required, including trained rise/hold.
                    # The older interactive helper cuts off prematurely at phase .7.
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
                policy.apply_action(action)  # raw ONNX: no EMA, target clipping or slew limits.
                np.testing.assert_allclose(controller.q_target, policy.default_pose + action, atol=1e-6)
                head_touch, head_force = False, 0.0
                for _ in range(DECIMATION):
                    controller.update()
                    # BAM reassigns its budget every physics step; multiply only
                    # AFTER update, matching FrictionDRBamActuator's budget hook.
                    # Viscous damping is intentionally unchanged.
                    model.dof_frictionloss[controller.dof_indexes] *= friction_scale
                    mujoco.mj_step(model, data)
                    for ci in range(data.ncon):
                        contact = data.contact[ci]
                        g1, g2 = int(contact.geom1), int(contact.geom2)
                        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
                        if (g1 in ground_geoms and b2 in head_bodies) or (g2 in ground_geoms and b1 in head_bodies):
                            head_touch = True
                            force = np.zeros(6)
                            mujoco.mj_contactForce(model, data, ci, force)
                            head_force = max(head_force, float(np.linalg.norm(force[:3])))
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    finite = False
                    break
                xyz, quat = data.qpos[qa:qa+3].copy(), data.qpos[qa+3:qa+7].copy()
                gravity = policy.quat_rotate_inverse(quat, np.array([0., 0., -1.]))
                tilt = float(np.degrees(np.arccos(np.clip(-gravity[2], -1, 1))))
                velocity = policy.quat_rotate_inverse(quat, data.qvel[va:va+3]).copy()
                if tilt > 70 and first_tilt_70 is None:
                    first_tilt_70 = float(data.time)
                mouth = data.site("mouth_tip")
                values = (float(data.time), command, obs, action, xyz, velocity, tilt,
                          data.qpos.copy(), data.qpos[policy.joint_qpos_indices].copy(),
                          data.qvel[policy.joint_qvel_indices].copy(), controller.q_target.copy(),
                          controller.applied_q_target.copy(), data.actuator_force.copy(),
                          float(mouth.xpos[2]), float(-mouth.xmat.reshape(3, 3)[2, 0]),
                          head_touch, head_force, model.dof_frictionloss[controller.dof_indexes].copy(),
                          model.dof_damping[controller.dof_indexes].copy())
                for key, value in zip(trace, values, strict=True):
                    trace[key].append(value)
                if writer and step % 2 == 1:
                    frame = draw_frame(renderer, model, data, "hd1910", kind, float(data.time),
                                       (xyz, tilt, velocity, first_tilt_70), status, cameras)
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, 1279, 43), fill=(17, 25, 39))
                    draw.text((24, 7), f"Microduck HD1910 | {model_key} | {condition}",
                              font=ImageFont.truetype(FONT, 25), fill=(241, 246, 255))
                    draw.rectangle((0, 683, 1279, 719), fill=(17, 25, 39))
                    draw.text((24, 687), "MuJoCo CPU + BAM · 50 Hz 策略 · 原速连续录像 · 无重置 · 无动作滤波/裁剪",
                              font=ImageFont.truetype(FONT, 18), fill=(157, 175, 197))
                    writer.write(frame)
                    if round(data.time, 2) in (1., 2., 4., 7., 10., 12.):
                        frame.save(video_dir / f"{prefix}_{data.time:05.2f}s.jpg", quality=90)
                if step % 250 == 249:
                    print(json.dumps({"run": prefix, "sim_s": round(float(data.time), 2),
                                      "tilt_deg": round(tilt, 2), "trunk_z_m": round(float(xyz[2]), 4)}), flush=True)
    finally:
        if writer:
            writer.close()
    arrays = {key: np.asarray(values) for key, values in trace.items()}
    np.savez_compressed(output / f"{prefix}.npz", **arrays)
    if not len(arrays["time"]):
        raise RuntimeError(f"No finite samples for {prefix}")
    windows = []
    for lo, hi, label in ((2, 3, "initial_stand"), (7, 8, "sit"), (12, 13, "return_stand")) if kind == "sitstand" else ():
        mask = (arrays["time"] > lo) & (arrays["time"] <= hi + 1e-8)
        target_z = .06 if label == "sit" else .115
        windows.append({"label": label, "measurement_s": [lo, hi],
                        "mean_z_m": float(arrays["xyz"][mask, 2].mean()),
                        "mean_tilt_deg": float(arrays["tilt_deg"][mask].mean()),
                        "max_tilt_deg": float(arrays["tilt_deg"][mask].max()),
                        "mean_joint_speed_rad_s": float(np.abs(arrays["joint_vel"][mask]).mean()),
                        "screen_pass": bool((np.abs(arrays["xyz"][mask, 2].mean() - target_z) < .025)
                                            and (arrays["tilt_deg"][mask].mean() < 30))})
    cycles = []
    for start in range(0, int(seconds), 4) if kind == "ground_pick" else ():
        cycle = (arrays["time"] > start) & (arrays["time"] <= start + 4 + 1e-8)
        returning = (arrays["time"] > start + 3.5) & (arrays["time"] <= start + 4 + 1e-8)
        low, rz, rt = float(arrays["mouth_height"][cycle].min()), float(arrays["xyz"][returning, 2].mean()), float(arrays["tilt_deg"][returning].mean())
        cycles.append({"window_s": [start, start + 4], "min_mouth_height_m": low,
                       "return_z_m": rz, "return_tilt_deg": rt,
                       "max_mouth_down_alignment": float(arrays["mouth_down_alignment"][cycle].max()),
                       "return_joint_home_rms_deg": float(np.degrees(np.sqrt(np.mean((arrays["joint_pos"][returning] - policy.default_pose) ** 2)))),
                       "head_ground_contact_samples": int(arrays["head_ground_contact"][cycle].sum()),
                       "screen_pass": bool(low < .025 and rz > .09 and rt < 30)})
    passed = finite and all(w["screen_pass"] for w in windows or cycles)
    scene = ROBOT_DIR / "scene.xml"
    report = {
        "created_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "model_key": model_key, "kind": kind, "condition": condition,
        "policy": str(policy_path), "policy_sha256": sha256(policy_path), "metadata": metadata,
        "physics": physics, "friction_budget_scale": friction_scale,
        "body_local_x_com_shift_m": com_x, "com_bodies": modified_com,
        "control_hz": 50, "action_ema": False, "action_clipping": False,
        "joint_velocity_observation_delay_steps": 0,
        "initial_root_xyz": [0, 0, .12], "requested_sim_seconds": seconds,
        "completed_sim_seconds": round(float(data.time), 8), "finite": finite,
        "heuristic_screen_pass": passed,
        "screen_definition": "Final 1s of each sitstand command: |mean z-target|<0.025m (stand .115m/sit .060m) and mean tilt<30deg. Groundpick: each cycle min mouth z<.025m and last .5s mean trunk z>.09m/tilt<30deg. Screening is not hardware validation.",
        "first_tilt_above_70deg_s": first_tilt_70, "maximum_tilt_deg": float(arrays["tilt_deg"].max()),
        "displacement_xyz_m": (arrays["xyz"][-1] - [0, 0, .12]).tolist(),
        "window_metrics": windows, "ground_pick_cycles": cycles,
        "video_path": str(video_dir / f"{prefix}.mp4") if writer else None,
        "video_frames": writer.count if writer else 0, "video_fps": FPS,
        "trajectory_path": str(output / f"{prefix}.npz"),
        "trajectory_alignment": "obs/action/command/target at control tick t; state/time sampled after four .005s physics steps at t+.02; applied_joint_target is delayed BAM target.",
        "joint_names": names, "policy_home_rad": policy.default_pose.tolist(),
        "joint_ranges_rad": model.jnt_range[np.asarray(model.actuator_trnid[:, 0], dtype=int)].tolist(),
        "joint_measured_min_rad": arrays["joint_pos"].min(axis=0).tolist(),
        "joint_measured_max_rad": arrays["joint_pos"].max(axis=0).tolist(),
        "joint_target_min_rad": arrays["joint_target"].min(axis=0).tolist(),
        "joint_target_max_rad": arrays["joint_target"].max(axis=0).tolist(),
        "max_joint_speed_rad_s": float(np.abs(arrays["joint_vel"]).max()),
        "max_target_step_deg": float(np.degrees(np.abs(np.diff(arrays["joint_target"], axis=0)).max())),
        "max_abs_joint_torque_nm": float(np.abs(arrays["joint_torque"]).max()),
        "max_head_ground_force_n": float(arrays["head_ground_force_n"].max()),
        "robot_total_mass_kg": float(model.body_mass.sum()), "wall_seconds": time.perf_counter() - started,
        "command_schedule": "stand0-3/sit3-8/stand8-13 seconds" if kind == "sitstand" else "3 continuous complete4s cycles: [cos(2pi*t/4),sin(2pi*t/4),0]; no early .7 cutoff",
        "source_sha256": {str(p): sha256(p) for p in (Path(__file__), INFERENCE, Path(__file__).with_name("render_action_comparison.py"), Path(__file__).with_name("render_hd1910_controller.py"), scene, ROBOT_DIR / "robot_groundcontact.xml")},
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": ort.__version__},
        "limitations": [
            "Deterministic CPU MuJoCo/BAM sensitivity evaluation, not exact replay of the unavailable original training configuration or a hardware validation.",
            "Geometry/mass/inertia are the repository Microduck model, not measurements of the physical replica; HD1910 uses public 1910 M6 parameters.",
            "Friction factors scale only BAM non-viscous budget each physics step; viscous damping, floor friction, battery/drop and latency remain nominal.",
            "CoM test adds the SAME signed local-X offset to trunk_base and the four neck/head bodies; other axes, mass, inertia stay fixed. This is an additional test assumption, not recovered randomization.",
            "The repository HEAD_BODY_NAMES also contains bearing_roll, which XML attaches to the right hip. It is deliberately excluded from this anatomically named head sensitivity test.",
            "No noise, pushes, payload, randomized initial states, IMU mounting bias, backlash or joint-velocity observation delay. One deterministic rollout per condition is not a success probability.",
            "Tilt >70deg is a diagnostic only; ground-pick intentionally leans. All saved videos run continuously with no reset/cut after failure.",
        ],
    }
    json_write(output / f"{prefix}.json", report)
    print(json.dumps({"completed": prefix, "finite": finite, "heuristic_screen_pass": passed,
                      "video": report["video_path"]}, ensure_ascii=False), flush=True)
    if not finite or abs(data.time - seconds) > .005:
        raise RuntimeError(f"Incomplete/nonfinite rollout {prefix}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/20261028_model_suite/input")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/20261028_model_suite/postures")
    parser.add_argument("--videos", type=Path, default=ROOT / "videos/20261028_model_suite/postures")
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--all-videos", action="store_true")
    args = parser.parse_args()
    reports = []
    for key in args.models:
        for condition in args.conditions:
            if MODELS[key][1] == "ground_pick" and condition != "baseline":
                continue
            reports.append(evaluate(args, key, condition))
    json_write(args.output / "summary.json", {
        "models": args.models, "conditions": args.conditions,
        "runs": [{k: r[k] for k in ("model_key", "condition", "finite", "heuristic_screen_pass", "maximum_tilt_deg", "window_metrics", "ground_pick_cycles", "video_path", "trajectory_path")} for r in reports],
    })
    command = " ".join(shlex.quote(str(x)) for x in (
        ROOT / "microduck_rl/.venv/bin/python", Path(__file__).resolve(), "--input", args.input.resolve(),
        "--output", args.output.resolve(), "--videos", args.videos.resolve(), "--models", *args.models,
        "--conditions", *args.conditions))
    command += " --no-video" if args.no_video else ""
    command += " --all-videos" if args.all_videos else ""
    (args.output / "replay.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")


if __name__ == "__main__":
    main()
