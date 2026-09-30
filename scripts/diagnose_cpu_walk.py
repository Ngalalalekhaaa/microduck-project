#!/usr/bin/env python3
"""Compare walking ONNX policies with the upstream CPU MuJoCo/BAM controller.

This imports the unmodified infer_policy.py. Only ONNX Runtime session setup is
wrapped to select its CPU provider and one thread; policy/control methods are
unchanged. This is a fixed nominal CPU diagnostic, not training/Warp replay.
"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "microduck_rl"
UPSTREAM = REPO / "scripts/infer_policy.py"
SCENE = REPO / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward_015": (0.15, 0.0, 0.0),
    "forward_020": (0.20, 0.0, 0.0),
    "forward_025": (0.25, 0.0, 0.0),
    "forward_030": (0.30, 0.0, 0.0),
    "forward_040": (0.40, 0.0, 0.0),
    "turn_050": (0.0, 0.0, 0.5),
}
OBS_NAMES = ["base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
             "actions", "command", "head_command", "body_command"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_session(path, *args, **kwargs):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ORIGINAL_SESSION(path, sess_options=options, providers=["CPUExecutionProvider"])


ORIGINAL_SESSION = ort.InferenceSession


def load_upstream():
    spec = importlib.util.spec_from_file_location("upstream_infer_policy", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_state(sample, n_joints):
    # Sample 0 exactly matches upstream main(); others add small reproducible
    # pose perturbations, shared across every policy and command.
    rng = np.random.default_rng(20260928 + sample)
    joint_delta = np.zeros(n_joints) if sample == 0 else rng.uniform(-0.01, 0.01, n_joints)
    roll, pitch = (0.0, 0.0) if sample == 0 else rng.uniform(-0.01, 0.01, 2)
    quat = np.array([
        np.cos(roll / 2) * np.cos(pitch / 2),
        np.sin(roll / 2) * np.cos(pitch / 2),
        np.cos(roll / 2) * np.sin(pitch / 2),
        -np.sin(roll / 2) * np.sin(pitch / 2),
    ])
    return joint_delta, quat


def trial(module, policy_path, label, command_name, sample, seconds, output):
    command = np.asarray(COMMANDS[command_name], dtype=np.float32)
    init_log = io.StringIO()
    with redirect_stdout(init_log):
        bam_model = module.load_bam_model(module.BAM_KP_FW, 7.4, None)
        model, data, bam, _ = module.load_mujoco_with_bam(
            str(SCENE), bam_model, 0.005, 0.1, module.BAM_VIN_MIN)
        with patch.object(module.ort, "InferenceSession", cpu_session):
            policy = module.PolicyInference(
                model, data, walking_onnx_path=str(policy_path), bam_ctrl=bam,
                action_scale=1.0, new_cmd_obs=True, use_projected_gravity=True,
                delay_min_lag=0, delay_max_lag=0,
            )
        policy.set_vel_cmd(*command)
    assert policy.ort_session.get_providers() == ["CPUExecutionProvider"]
    assert policy.ort_session.get_inputs()[0].shape == [1, 61]
    assert policy.ort_session.get_outputs()[0].shape == [1, 14]
    metadata = policy.ort_session.get_modelmeta().custom_metadata_map
    assert metadata["observation_names"].split(",") == OBS_NAMES
    assert float(metadata["action_scale"]) == 1.0
    joint_names = [model.joint(int(model.actuator_trnid[i, 0])).name
                   for i in range(model.nu)]
    assert joint_names == metadata["joint_names"].split(",")
    # Export metadata is rounded to 3 decimal places; controller uses exact HOME.
    np.testing.assert_allclose(np.fromstring(metadata["default_joint_pos"], sep=","),
                               policy.default_pose, atol=0.00051)

    mujoco.mj_resetData(model, data)
    joint_delta, quat = initial_state(sample, model.nu)
    free_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa, va = int(model.jnt_qposadr[free_id]), int(model.jnt_dofadr[free_id])
    data.qpos[qa:qa + 3] = [0.0, 0.0, 0.125]
    data.qpos[qa + 3:qa + 7] = quat
    data.qpos[policy.joint_qpos_indices] = policy.default_pose + joint_delta
    bam.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)
    initial_obs = policy.get_observations().copy()
    np.testing.assert_allclose(initial_obs[3:6], policy.get_projected_gravity(), atol=1e-7)
    np.testing.assert_allclose(initial_obs[48:51], command, atol=1e-7)
    assert initial_obs.shape == (61,) and np.all(initial_obs[51:61] == 0)
    initial_xyz = data.qpos[qa:qa + 3].copy()

    times, observations, actions, positions, velocities, tilts, yaw, joint_speed = (
        [] for _ in range(8))
    failure = None
    for step in range(round(seconds / 0.02)):
        obs = policy.get_observations().copy()
        if not np.isfinite(obs).all():
            failure = "nonfinite_observation"
            break
        assert obs.shape == (61,)
        np.testing.assert_allclose(obs[48:51], command, atol=1e-7)
        assert np.all(obs[51:61] == 0)
        action = policy.infer()
        if not np.isfinite(action).all():
            failure = "nonfinite_action"
            break
        policy.apply_action(action)
        for _ in range(4):
            bam.update()
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            failure = "nonfinite_state"
            break
        quat = data.qpos[qa + 3:qa + 7].copy()
        body_velocity = policy.quat_rotate_inverse(quat, data.qvel[va:va + 3])
        gravity = policy.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1.0, 1.0))))
        qw, qx, qy, qz = quat
        times.append(float(data.time))
        observations.append(obs)
        actions.append(action)
        positions.append(data.qpos[qa:qa + 3].copy())
        velocities.append([body_velocity[0], body_velocity[1], data.qvel[va + 5]])
        tilts.append(tilt)
        yaw.append(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        joint_speed.append(np.max(np.abs(data.qvel[policy.joint_qvel_indices])))
        if tilt > 70.0:
            failure = "tilt_above_70deg"
            break

    times, observations, actions, positions, velocities, tilts, yaw = (
        np.asarray(values) for values in
        (times, observations, actions, positions, velocities, tilts, yaw))
    valid = (times >= 1.0) & (tilts <= 70.0)
    trace = output / f"{label}_{command_name}_sample{sample}.npz"
    np.savez_compressed(trace, time=times, obs_pre=observations, action=actions,
                        position_world=positions, velocity_body=velocities,
                        tilt_deg=tilts, yaw_rad=yaw)
    result = {
        "policy": label, "command": command_name, "sample": sample,
        "command_vx_vy_wz": command.tolist(), "seconds_completed": float(data.time),
        "failure": failure, "initial_joint_delta_rad": joint_delta.tolist(),
        "initial_root_quat_wxyz": initial_state(sample, model.nu)[1].tolist(),
        "initial_obs_61": initial_obs.tolist(), "command_slots_48_61": initial_obs[48:61].tolist(),
        "mean_actual_vx_vy_wz_after_1s": velocities[valid].mean(axis=0).tolist() if valid.any() else None,
        "mean_abs_error_after_1s": np.abs(velocities[valid] - command).mean(axis=0).tolist() if valid.any() else None,
        "displacement_xyz_m": (positions[-1] - initial_xyz).tolist() if len(positions) else None,
        "yaw_delta_rad": float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]) if len(yaw) else None,
        "max_tilt_deg": float(tilts.max()) if len(tilts) else None,
        "p95_tilt_deg_after_1s": float(np.percentile(tilts[valid], 95)) if valid.any() else None,
        "max_joint_speed_rad_s": float(max(joint_speed)) if joint_speed else None,
        "mean_abs_action": float(np.abs(actions).mean()) if len(actions) else None,
        "trace": str(trace),
    }
    return result, metadata, init_log.getvalue(), {
        "ctrlrange_enabled": model.actuator_ctrllimited.tolist(),
        "forcerange_nm": model.actuator_forcerange.tolist(),
        "floor_and_feet_contact": {
            name: {"friction": model.geom(name).friction.tolist(),
                   "condim": int(model.geom(name).condim[0]),
                   "contype": int(model.geom(name).contype[0]),
                   "conaffinity": int(model.geom(name).conaffinity[0])}
            for name in ("floor", "left_foot_collision", "right_foot_collision")
        },
        "bam_actuator_class": type(bam_model.actuator).__name__,
        "bam_max_current": bam_model.actuator.max_current,
    }


def summarize(trials):
    output = {}
    for label in dict.fromkeys(item["policy"] for item in trials):
        output[label] = {}
        for command in dict.fromkeys(item["command"] for item in trials if item["policy"] == label):
            rows = [item for item in trials if item["policy"] == label and item["command"] == command]
            v = [row["mean_actual_vx_vy_wz_after_1s"] for row in rows
                 if row["mean_actual_vx_vy_wz_after_1s"] is not None]
            output[label][command] = {
                "samples": len(rows), "failures": sum(row["failure"] is not None for row in rows),
                "mean_actual_vx_vy_wz_after_1s": np.mean(v, axis=0).tolist() if v else None,
                "individual_actual_vx_vy_wz_after_1s": v,
                "mean_displacement_xyz_m": np.mean([row["displacement_xyz_m"] for row in rows], axis=0).tolist(),
                "max_tilt_deg": max(row["max_tilt_deg"] for row in rows),
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--commands", nargs="+", choices=list(COMMANDS), default=list(COMMANDS))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cpu_walk_diagnosis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    policies = {label: Path(path).resolve() for label, path in
                (value.split("=", 1) for value in args.policy)}
    assert all(path.is_file() for path in policies.values())
    source_hash = sha256(UPSTREAM)
    module = load_upstream()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "upstream_infer_path": str(UPSTREAM), "upstream_infer_sha256": source_hash,
        "scene": str(SCENE), "scene_sha256": sha256(SCENE),
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": ort.__version__},
        "policies": {label: {"path": str(path), "sha256": sha256(path)} for label, path in policies.items()},
        "settings": {"seconds_per_trial": args.seconds, "samples": args.samples,
                     "physics_dt": 0.005, "control_hz": 50, "decimation": 4,
                     "vin": 7.4, "vin_drop_gain": 0.1, "vin_min": 6.0,
                     "kp_fw": 200.0, "action_scale": 1.0, "current_limit": None,
                     "new_cmd_obs": True, "projected_gravity": True,
                     "actuator_delay": 0, "fall_tilt_deg": 70.0,
                     "initial_root_z_m": 0.125, "tracking_warmup_excluded_s": 1.0,
                     "providers": ["CPUExecutionProvider"]},
        "limitations": [
            "CPU MuJoCo solver and BAM implementation differ from MuJoCo Warp training.",
            "Fixed nominal voltage/friction/gains; no training startup/reset domain randomization, pushes or curricula.",
            "Upstream CPU default has no actuator delay or IMU/joint-velocity observation lag; training uses them.",
            "Small initial pose perturbations are diagnostic samples, not the training reset distribution.",
            "First tilt above 70 degrees stops a trial; motion after a fall is excluded.",
            "Official alpha model's exact training commit/config is unavailable; comparison is behavioral.",
            "Upstream scene_walk XML/contact defaults retained; training EntityCfg collision overrides are not applied.",
        ],
        "trials": [], "summary": {},
    }
    started = time.perf_counter()
    for label, path in policies.items():
        for command in args.commands:
            for sample in range(args.samples):
                result, metadata, init_log, physics = trial(module, path, label, command, sample,
                                                          args.seconds, args.output)
                report["trials"].append(result)
                report["policies"][label]["onnx_metadata"] = metadata
                report["physics"] = physics
                report["summary"] = summarize(report["trials"])
                report["elapsed_wall_seconds"] = time.perf_counter() - started
                (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
                (args.output / f"{label}_{command}_sample{sample}.init.log").write_text(init_log)
                print(json.dumps({key: result[key] for key in
                                  ("policy", "command", "sample", "failure", "mean_actual_vx_vy_wz_after_1s",
                                   "displacement_xyz_m", "max_tilt_deg")}), flush=True)
    assert sha256(UPSTREAM) == source_hash, "upstream inference source changed during diagnosis"
    print(f"Report: {args.output / 'diagnosis.json'}", flush=True)


if __name__ == "__main__":
    main()
