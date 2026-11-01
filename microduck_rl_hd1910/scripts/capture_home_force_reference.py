"""Capture a gated nominal HOME actuator-torque reference in CPU simulation.

The privileged pose feedback is used only to collect this reference. It is not
an actor observation or a deployable controller. No hardware is accessed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "src/mjlab_microduck/tasks/home_force_reference.json"


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(args):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks.microduck_home_calibration_env_cfg import make_microduck_home_calibration_env_cfg
    from mjlab_microduck.tasks.microduck_hd1910_env_cfg import HD1910_M6_JSON

    configure_torch_backends()
    torch.set_num_threads(4)
    cfg = make_microduck_home_calibration_env_cfg(evaluation_profile="nominal")
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.rewards = {}
    cfg.terminations = {}
    cfg.episode_length_s = args.warmup_seconds + args.sample_seconds + 1.
    cfg.actions["home_calibration"].bias_range_rad = (0., 0.)
    cfg.events["home_calibration_reset"].params.update(
        initial_position_noise_rad=0., imu_mounting_error_deg=0.,
        gravity_noise_std=0., gyro_noise_std=0.,
    )
    cfg.events["reset_base"].params["pose_range"].update(
        roll=(0., 0.), pitch=(0., 0.), yaw=(0., 0.),
    )
    source_files = [Path(__file__), HD1910_M6_JSON,
                    ROOT / "src/mjlab_microduck/tasks/microduck_home_calibration_env_cfg.py",
                    ROOT / "src/mjlab_microduck/tasks/home_calibration_actions.py",
                    ROOT / "src/mjlab_microduck/tasks/mdp.py"]
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "versions": {name: version(name) for name in ("mjlab", "mujoco", "torch", "warp-lang")},
        "source_sha256": {str(path.relative_to(ROOT)): source_hash(path) for path in source_files},
        "device": "cpu", "seed": args.seed,
        "reference_conditions": "nominal parameters, zero calibration bias, no initial or IMU perturbation, freely standing double support",
        "actuator_assumption": "HD1910 1910_m6.json model prior, firmware P5/D0/I0; not a measured hardware load reference",
        "torque_source": "robot.data.qfrc_actuator at the end of each 20 Hz control step, selected by the 14 servo joint IDs",
        "torque_meaning": "Applied actuator generalized torque in joint space; excludes separately modelled friction and is not the full external gearbox load",
        "reference_aggregation": "mean(abs(joint_actuator_torque), time) / sum(mean(abs(joint_actuator_torque), time))",
        "privileged_feedback": "Warmup only: delta=clip(-integral_gain_per_s*(q_true-HOME)*control_dt, +/-delta_limit), accumulated by the normal HOME action term; sample window holds the resulting offset with zero increments",
    }
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset(seed=args.seed)
        action = env.action_manager.get_term("home_calibration")
        robot = env.scene["robot"]
        if not math.isclose(env.step_dt, .05) or action.action_dim != 14:
            raise RuntimeError("Reference collection requires 14 joints at 20 Hz")
        warmup_steps = math.ceil(args.warmup_seconds / env.step_dt)
        sample_steps = math.ceil(args.sample_seconds / env.step_dt)
        samples = []
        maximum_warmup_tilt = 0.
        for step in range(warmup_steps + sample_steps):
            with torch.no_grad():
                error = robot.data.joint_pos[:, action.joint_ids] - action.home
                request = -args.integral_gain * env.step_dt * error / action.cfg.delta_limit_rad
                if step >= warmup_steps:
                    request.zero_()
                env.step(request.clamp(-1., 1.))
            q_error = (robot.data.joint_pos[:, action.joint_ids] - action.home)[0].cpu().numpy().copy()
            torque = robot.data.qfrc_actuator[:, action.joint_ids][0].cpu().numpy().copy()
            gravity = robot.data.projected_gravity_b[0].cpu().numpy()
            tilt = float(np.rad2deg(np.arccos(np.clip(-gravity[2], -1., 1.))))
            height = float((robot.data.root_link_pos_w[0, 2] - env.scene.env_origins[0, 2]).item())
            contact = env.scene.sensors["feet_ground_contact"].data.found[:, :2].reshape(1, 2, -1).any(-1)
            support = contact[0].cpu().numpy().astype(bool)
            if not np.isfinite(np.r_[q_error, torque, tilt, height]).all():
                raise RuntimeError(f"Nonfinite reference state at {(step + 1) * env.step_dt:.2f}s")
            if tilt >= 25. or height <= .085:
                raise RuntimeError(f"Reference controller lost standing at {(step + 1) * env.step_dt:.2f}s: tilt={tilt:.3f}, height={height:.4f}")
            if step < warmup_steps:
                maximum_warmup_tilt = max(maximum_warmup_tilt, tilt)
            else:
                samples.append({"time_s": (step + 1) * env.step_dt,
                                "true_joint_error_deg": np.rad2deg(q_error).tolist(),
                                "joint_actuator_torque_nm": torque.tolist(),
                                "tilt_deg": tilt, "trunk_height_m": height,
                                "foot_contact": support.tolist()})
            if (step + 1) % 20 == 0:
                print(f"t={(step + 1) * env.step_dt:.1f}s max_pose_error={np.rad2deg(np.abs(q_error)).max():.4f}deg tilt={tilt:.3f}deg", flush=True)
        errors = np.asarray([row["true_joint_error_deg"] for row in samples])
        torque = np.asarray([row["joint_actuator_torque_nm"] for row in samples])
        absolute = np.abs(torque)
        average = absolute.mean(axis=0)
        total = absolute.sum(axis=1)
        support = np.asarray([row["foot_contact"] for row in samples])
        tilt = np.asarray([row["tilt_deg"] for row in samples])
        height = np.asarray([row["trunk_height_m"] for row in samples])
        stats = {
            "true_joint_mae_deg": np.abs(errors).mean(axis=0).tolist(),
            "true_joint_rmse_deg": np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
            "maximum_mean_absolute_joint_error_deg": float(np.abs(errors).mean(axis=0).max()),
            "maximum_absolute_joint_error_deg": float(np.abs(errors).max()),
            "maximum_tilt_deg": float(tilt.max()), "maximum_warmup_tilt_deg": maximum_warmup_tilt,
            "minimum_trunk_height_m": float(height.min()), "mean_trunk_height_m": float(height.mean()),
            "double_support_fraction": float(support.all(axis=1).mean()),
            "foot_support_fractions": support.mean(axis=0).tolist(),
            "mean_sum_absolute_torque_nm": float(total.mean()),
            "minimum_sum_absolute_torque_nm": float(total.min()),
            "std_sum_absolute_torque_nm": float(total.std()),
        }
        gates = {
            "maximum_mean_joint_error_below_half_degree": stats["maximum_mean_absolute_joint_error_deg"] <= .5,
            "maximum_tilt_below_ten_degrees": stats["maximum_tilt_deg"] < 10.,
            "double_support_at_least_95_percent": stats["double_support_fraction"] >= .95,
            "minimum_total_torque_above_0_02_nm": stats["minimum_sum_absolute_torque_nm"] > .02,
        }
        passed = all(gates.values())
        ratio = average / average.sum() if average.sum() > .02 else None
        return {
            "schema_version": 1, "reference_kind": "absolute_joint_actuator_torque_fraction",
            "gate_passed": passed, "gates": gates, "statistics": stats,
            "joint_names": list(action.joint_names), "home_angles_rad": action.home[0].cpu().tolist(),
            "target_joint_torque_fraction": None if ratio is None else ratio.tolist(),
            "mean_absolute_joint_actuator_torque_nm": average.tolist(),
            "std_absolute_joint_actuator_torque_nm": absolute.std(axis=0).tolist(),
            "final_command_offset_rad": action.command_offset[0].cpu().tolist(),
            "controller": {"integral_gain_per_s": args.integral_gain,
                           "delta_limit_rad": action.cfg.delta_limit_rad,
                           "offset_limit_rad": action.cfg.offset_limit_rad,
                           "initial_settle_seconds": action.cfg.settle_seconds},
            "control_dt_s": env.step_dt, "warmup_seconds": warmup_steps * env.step_dt,
            "sample_seconds": sample_steps * env.step_dt, "sample_count": len(samples),
            "provenance": provenance, "samples": samples,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--warmup-seconds", type=float, default=8.)
    parser.add_argument("--sample-seconds", type=float, default=4.)
    parser.add_argument("--integral-gain", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=1031)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 4 for value in (args.warmup_seconds, args.sample_seconds)):
        parser.error("Warmup and sampling must each last at least four seconds")
    if not math.isfinite(args.integral_gain) or args.integral_gain <= 0:
        parser.error("Integral gain must be positive and finite")
    if args.output.exists():
        parser.error(f"Refusing to overwrite an existing reference: {args.output}")
    report_path = args.report or args.output.with_suffix(".capture.json")
    if report_path.resolve() == args.output.resolve():
        parser.error("Diagnostic report must be separate from the gated reference artifact")
    try:
        report = capture(args)
    except Exception as exc:
        report = {"gate_passed": False, "error": f"{type(exc).__name__}: {exc}"}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Capture report: {report_path.resolve()}")
    if not report["gate_passed"]:
        print(json.dumps({key: report[key] for key in ("error", "gates", "statistics") if key in report}, indent=2))
        raise SystemExit("Reference gates failed; no reference artifact written")
    artifact = {key: value for key, value in report.items() if key != "samples"}
    artifact["capture_report_sha256"] = source_hash(report_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Gated reference: {args.output.resolve()}")
    print(json.dumps(artifact["statistics"], indent=2))


if __name__ == "__main__":
    main()
