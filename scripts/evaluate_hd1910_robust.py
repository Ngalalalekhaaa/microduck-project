"""Evaluate HD1910 tracking and first-episode failure in fixed simulation profiles.

Loads actor weights only, so checkpoint curricula counters cannot change the
evaluation. All terminal states are captured before manual resets. No hardware
modules, filtering or command-target clipping are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG = ROOT / "reports/hd1910_target_logging_20261001/before/robot.json"
COMMANDS = {
    "stand": (0.0, 0.0, 0.0),
    "forward10": (0.10, 0.0, 0.0),
    "forward20": (0.20, 0.0, 0.0),
    "forward25": (0.25, 0.0, 0.0),
    "backward10": (-0.10, 0.0, 0.0),
    "backward20": (-0.20, 0.0, 0.0),
    "backward25": (-0.25, 0.0, 0.0),
    "lateral20": (0.0, 0.20, 0.0),
    "lateral_minus20": (0.0, -0.20, 0.0),
    "turn50": (0.0, 0.0, 0.50),
    "turn_minus50": (0.0, 0.0, -0.50),
    "turn100": (0.0, 0.0, 1.0),
    "turn_minus100": (0.0, 0.0, -1.0),
    "forwardturn": (0.20, 0.0, 0.50),
}


def json_value(value: Any) -> Any:
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def load_calibration(path: Path) -> dict:
    config = json.loads(path.read_text())
    joints = config["joints"]
    names = [joint["name"] for joint in joints]
    zero = np.asarray([joint["zero_tick"] for joint in joints], dtype=float)
    direction = np.asarray([joint["direction"] for joint in joints], dtype=float)
    bounds = np.asarray(config["target_limits_rad"], dtype=float)
    if len(names) != 14 or len(set(names)) != 14 or bounds.shape != (14, 2):
        raise ValueError("Calibration must contain 14 distinct ordered joints and target limits")
    if not np.isfinite(zero).all() or not np.isin(direction, [-1, 1]).all():
        raise ValueError("Calibration zero ticks must be finite and directions must be +/-1")
    if not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Calibration limits must be finite and strictly ordered")
    return {"names": names, "zero": zero, "direction": direction, "bounds": bounds,
            "max_jump": float(config["control"]["max_target_jump_rad"]),
            "source": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "limits_scope": "Configured deployment target bounds; physical clearance is not measured"}


def encoder_outside(target: np.ndarray, zero: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Match deployment's rounding and calibrated single-turn bounds, without clipping."""
    ticks = np.rint(zero + direction * target * 4096.0 / (2.0 * np.pi))
    return ~np.isfinite(ticks) | (ticks < 0) | (ticks > 4095)


def failure_direction(gravity: np.ndarray) -> str:
    if not np.isfinite(gravity).all():
        return "unknown"
    x, y = gravity[:2]
    if abs(x) >= abs(y):
        return "forward" if x >= 0 else "backward"
    return "left" if y >= 0 else "right"


class RolloutStats:
    """Keep first-episode membership after failed worlds are reset for the simulator."""

    def __init__(self, count: int, command: tuple, calibration: dict,
                 joint_bounds: np.ndarray, horizon: float, warmup: float, fall_tilt: float):
        self.count, self.command, self.calibration = count, np.asarray(command), calibration
        self.joint_bounds, self.horizon = joint_bounds, horizon
        self.warmup, self.fall_tilt = warmup, fall_tilt
        self.alive = np.ones(count, dtype=bool)
        self.failure_time = np.full(count, np.nan)
        self.failure_kind = [None] * count
        self.failures = []
        self.velocities, self.tilts, self.positions = [], [], []
        self.targets, self.jumps = [], []
        self.encoder_first_time = np.full(count, np.nan)
        self.encoder_envs = np.zeros(count, dtype=bool)
        self.encoder_joint_counts = np.zeros(14, dtype=np.int64)
        self.target_limit_joint_counts = np.zeros(14, dtype=np.int64)
        self.measured_limit_joint_counts = np.zeros(14, dtype=np.int64)
        self.physical_limit_joint_counts = np.zeros(14, dtype=np.int64)
        self.target_frames = self.encoder_frames = self.jump_frames = 0

    def mark_failure(self, mask: np.ndarray, time: float, kind: str) -> None:
        for i in np.flatnonzero(self.alive & mask):
            self.failure_time[i] = time
            self.failure_kind[i] = kind
            self.failures.append({"env_id": int(i), "time_s": time, "kind": kind})
        self.alive &= ~mask

    def capture(self, time: float, velocity: np.ndarray, gravity: np.ndarray,
                measured_q: np.ndarray, motor_q: np.ndarray, target: np.ndarray,
                jump: np.ndarray, terminated: np.ndarray, timed_out: np.ndarray,
                finite: np.ndarray, term_masks: dict[str, np.ndarray]) -> dict:
        before = self.alive.copy()
        tilt = np.rad2deg(np.arccos(np.clip(-gravity[:, 2], -1.0, 1.0)))
        valid_state = before & finite & np.isfinite(tilt)
        valid_target = before & np.isfinite(target).all(1) & np.isfinite(jump).all(1)
        self.tilts.append(tilt[valid_state].copy())
        self.positions.append(measured_q[valid_state].copy())
        self.targets.append(target[valid_target].copy())
        self.jumps.append(jump[valid_target].copy())
        bounds = self.calibration["bounds"]
        measured_outside = (measured_q < bounds[:, 0]) | (measured_q > bounds[:, 1])
        motor_outside = ((motor_q < self.joint_bounds[:, 0] - 1e-4)
                         | (motor_q > self.joint_bounds[:, 1] + 1e-4))
        self.measured_limit_joint_counts += measured_outside[valid_state].sum(0)
        self.physical_limit_joint_counts += motor_outside[valid_state].sum(0)
        target_outside = (target < bounds[:, 0]) | (target > bounds[:, 1])
        outside_encoder = encoder_outside(target, self.calibration["zero"],
                                          self.calibration["direction"])
        self.target_limit_joint_counts += target_outside[valid_target].sum(0)
        self.encoder_joint_counts += outside_encoder[valid_target].sum(0)
        encoder_bad = valid_target & outside_encoder.any(1)
        first_encoder = encoder_bad & ~self.encoder_envs
        self.encoder_first_time[first_encoder] = time
        self.encoder_envs |= encoder_bad
        self.target_frames += int(valid_target.sum())
        self.encoder_frames += int(encoder_bad.sum())
        self.jump_frames += int((valid_target & (np.abs(jump).max(1)
                                                 > self.calibration["max_jump"])).sum())
        failed = before & (~finite | ~np.isfinite(tilt) | terminated | timed_out
                           | (tilt > self.fall_tilt))
        for i in np.flatnonzero(failed):
            reasons = [name for name, mask in term_masks.items() if mask[i]]
            if not finite[i] or not np.isfinite(tilt[i]) or "nan_state" in reasons:
                kind = "nonfinite_state"
            elif tilt[i] > self.fall_tilt or "fell_over" in reasons:
                kind = "fall_" + failure_direction(gravity[i])
            elif timed_out[i]:
                kind = "time_out"
            else:
                kind = "task_termination"
            self.failure_time[i] = time
            self.failure_kind[i] = kind
            self.failures.append({"env_id": int(i), "time_s": time, "kind": kind,
                                  "termination_terms": reasons, "tilt_deg": tilt[i],
                                  "gravity_b": gravity[i].copy(),
                                  "joint_measured_rad": measured_q[i].copy(),
                                  "target_rad": target[i].copy()})
        self.alive &= ~failed
        valid_tracking = self.alive & finite & np.isfinite(velocity).all(1)
        if time >= self.warmup:
            self.velocities.append(velocity[valid_tracking].copy())
        return {"time_s": time, "survival_fraction": float(self.alive.mean()),
                "mean_vx": float(velocity[valid_tracking, 0].mean()) if valid_tracking.any() else None,
                "mean_vy": float(velocity[valid_tracking, 1].mean()) if valid_tracking.any() else None,
                "mean_wz": float(velocity[valid_tracking, 2].mean()) if valid_tracking.any() else None,
                "max_tilt_deg_first_episode": float(tilt[valid_state].max()) if valid_state.any() else None,
                "encoder_proxy_violations": int(encoder_bad.sum())}

    @staticmethod
    def samples(chunks: list[np.ndarray], width: int | None = None) -> np.ndarray:
        return np.concatenate(chunks) if chunks else np.empty((0, width) if width else (0,))

    def result(self) -> dict:
        velocity = self.samples(self.velocities, 3)
        target, jump = self.samples(self.targets, 14), self.samples(self.jumps, 14)
        measured, tilt = self.samples(self.positions, 14), self.samples(self.tilts)
        error = velocity - self.command
        mae = np.abs(error).mean(0) if len(error) else None
        rmse = np.sqrt(np.square(error).mean(0)) if len(error) else None
        return {"command_vx_vy_wz": self.command.tolist(),
                "num_envs": self.count, "survived": int(self.alive.sum()),
                "survival_fraction": float(self.alive.mean()),
                "mean_first_episode_seconds": float(np.nan_to_num(self.failure_time, nan=self.horizon).mean()),
                "first_failure_time_s": self.failure_time,
                "first_failure_kind": self.failure_kind, "failure_counts": dict(Counter(k for k in self.failure_kind if k)),
                "failures": self.failures,
                "valid_tracking_samples": len(velocity),
                "mean_actual_vx_vy_wz": velocity.mean(0) if len(velocity) else None,
                "mean_abs_error_vx_vy_wz": mae, "rmse_vx_vy_wz": rmse,
                "tracking_mae_normalized": float(np.mean(mae / [.25, .20, 1.0])) if mae is not None else None,
                "tilt_p95_deg": float(np.percentile(tilt, 95)) if len(tilt) else None,
                "max_tilt_deg": float(tilt.max()) if len(tilt) else None,
                "joint_measured_min_rad": measured.min(0) if len(measured) else None,
                "joint_measured_max_rad": measured.max(0) if len(measured) else None,
                "joint_measured_limit_violation_counts": self.measured_limit_joint_counts,
                "joint_motor_xml_limit_violation_counts": self.physical_limit_joint_counts,
                "target_min_rad": target.min(0) if len(target) else None,
                "target_max_rad": target.max(0) if len(target) else None,
                "max_abs_target_rad": float(np.abs(target).max()) if len(target) else None,
                "target_jump_p95_p99_max_rad": np.percentile(np.abs(jump).max(1), [95, 99, 100]) if len(jump) else None,
                "target_limit_violation_counts": self.target_limit_joint_counts,
                "target_samples": self.target_frames,
                "target_encoder_proxy_violation_frames": self.encoder_frames,
                "target_encoder_proxy_violation_fraction": self.encoder_frames / self.target_frames if self.target_frames else None,
                "target_encoder_proxy_violation_counts": self.encoder_joint_counts,
                "encoder_proxy_first_violation_s": self.encoder_first_time,
                "encoder_proxy_env_violation_fraction": float(self.encoder_envs.mean()),
                "deployment_proxy_survival_fraction": float((self.alive & ~self.encoder_envs).mean()),
                "target_jump_exceed_frames": self.jump_frames,
                "target_jump_exceed_fraction": self.jump_frames / self.target_frames if self.target_frames else None}


def aggregate_results(results: list[dict]) -> dict:
    """Use equal command/seed weights; retain missing tracking when all worlds fail."""
    if not results:
        return {}
    aggregate = {}
    for field in ("survival_fraction", "deployment_proxy_survival_fraction",
                  "target_encoder_proxy_violation_fraction", "target_jump_exceed_fraction",
                  "tracking_mae_normalized", "mean_abs_error_vx_vy_wz", "rmse_vx_vy_wz",
                  "tilt_p95_deg"):
        values = [r[field] for r in results if r[field] is not None]
        aggregate[field] = np.mean(values, axis=0).tolist() if values else None
    maxima = [r["max_abs_target_rad"] for r in results if r["max_abs_target_rad"] is not None]
    aggregate["max_abs_target_rad"] = max(maxima) if maxima else None
    aggregate["runs"] = len(results)
    aggregate["tracking_runs"] = sum(r["valid_tracking_samples"] > 0 for r in results)
    aggregate["failure_counts"] = dict(sum((Counter(r["failure_counts"]) for r in results), Counter()))
    return aggregate


def make_eval_cfg(task: str, profile: str, seed: int, count: int, seconds: float):
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnlyCfg
    if task.endswith("-HD1910-Robust"):
        from mjlab_microduck.tasks.microduck_hd1910_robust_env_cfg import make_microduck_hd1910_robust_velocity_env_cfg
        cfg = make_microduck_hd1910_robust_velocity_env_cfg(
            backlash="-Backlash-" in task, evaluation_profile=profile)
    else:
        cfg = deepcopy(load_env_cfg(task, play=False))
        # Legacy HD1910 has two DR range curricula. Freeze their FINAL values
        # before managers copy the cfg; reward schedules cannot affect inference.
        for name in ("com_range", "head_com_range"):
            term = cfg.curriculum.get(name)
            if term is not None:
                params = term.params
                radius = params["range_stages"][-1]["range"]
                cfg.events[params["event_name"]].params["ranges"] = (-radius, radius)
        if profile == "nominal":
            cfg.events = {k: v for k, v in cfg.events.items()
                          if k in ("expand_bam_friction_fields", "reset_action_history", "reset_base")}
            base = cfg.events.get("reset_base")
            if base is not None:
                base.params["pose_range"] = {"z": (.125, .125)}
                base.params["velocity_range"] = {}
            actuator = cfg.scene.entities["robot"].articulation.actuators[0]
            actuator.vin_range = (7.6, 7.6)
            actuator.vin_drop_gain_range = (.1, .1)
            actuator.delay_min_lag = actuator.delay_max_lag = 4
            for group in cfg.observations.values():
                group.enable_corruption = False
                for term in group.terms.values():
                    term.noise = None
                    term.delay_min_lag = term.delay_max_lag = 0
                    if "max_angle_deg" in term.params:
                        term.params["max_angle_deg"] = 0.
                    if "biased" in term.params:
                        term.params["biased"] = False
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = count, seed, False
    cfg.episode_length_s, cfg.curriculum = seconds + 2.0, {}
    # Training's bucket sampler assumes nonzero range maxima, so fixed zero
    # axes (standing, straight walking or pure yaw) are invalid for that class.
    # Use the ordinary sampler, which accepts equal lower/upper bounds, for
    # every evaluation task before managers copy or build the configuration.
    twist = VelocityCommandCommandOnlyCfg(**vars(cfg.commands["twist"]))
    cfg.commands["twist"] = twist
    twist.heading_command = False
    twist.ranges.heading = None
    for name in ("rel_standing_envs", "rel_heading_envs", "rel_turn_in_place_envs",
                 "rel_forward_envs", "rel_world_envs", "init_velocity_prob"):
        if hasattr(twist, name):
            setattr(twist, name, 0.)
    twist.resampling_time_range = (seconds + 10., seconds + 10.)
    for name in ("head_pose", "body_pose"):
        cfg.commands[name].ranges = tuple((0., 0.) for _ in cfg.commands[name].ranges)
        cfg.commands[name].resampling_time_range = (seconds + 10., seconds + 10.)
    return cfg


def write_csv_reports(output: Path, report: dict, calibration: dict) -> None:
    fields = ["profile", "seed", "command_name", "survival_fraction",
              "deployment_proxy_survival_fraction", "mean_first_episode_seconds",
              "tracking_mae_normalized", "target_encoder_proxy_violation_fraction",
              "target_jump_exceed_fraction", "max_abs_target_rad", "tilt_p95_deg",
              "valid_tracking_samples", "mae_vx", "mae_vy", "mae_wz",
              "rmse_vx", "rmse_vy", "rmse_wz"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in report["results"]:
            row = {field: result.get(field) for field in fields}
            for label, source in (("mae", "mean_abs_error_vx_vy_wz"), ("rmse", "rmse_vx_vy_wz")):
                for axis, value in zip(("vx", "vy", "wz"), result[source] or (None,) * 3):
                    row[f"{label}_{axis}"] = value
            writer.writerow(row)
    joint_fields = ["profile", "seed", "command_name", "joint_name", "measured_min_rad",
                    "measured_max_rad", "target_min_rad", "target_max_rad",
                    "measured_limit_violation_frames", "motor_xml_limit_violation_frames",
                    "target_limit_violation_frames", "target_encoder_proxy_violation_frames"]
    with (output / "joint_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=joint_fields)
        writer.writeheader()
        for result in report["results"]:
            for index, name in enumerate(calibration["names"]):
                row = {key: result[key] for key in ("profile", "seed", "command_name")}
                row["joint_name"] = name
                for key, source in (("measured_min_rad", "joint_measured_min_rad"),
                    ("measured_max_rad", "joint_measured_max_rad"), ("target_min_rad", "target_min_rad"),
                    ("target_max_rad", "target_max_rad"),
                    ("measured_limit_violation_frames", "joint_measured_limit_violation_counts"),
                    ("motor_xml_limit_violation_frames", "joint_motor_xml_limit_violation_counts"),
                    ("target_limit_violation_frames", "target_limit_violation_counts"),
                    ("target_encoder_proxy_violation_frames", "target_encoder_proxy_violation_counts")):
                    row[key] = result[source][index] if result[source] is not None else None
                writer.writerow(row)


def run(args: argparse.Namespace) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    import torch
    import mjlab.tasks  # noqa: F401 -- populate registry
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.models import MLPModel

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    calibration = load_calibration(args.robot_config)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if "actor_state_dict" not in checkpoint:
        raise ValueError("Evaluation requires a current RSL-RL actor_state_dict checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"task": args.task, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "checkpoint_iteration": checkpoint.get("iter"),
              "profile": args.profile, "seeds": args.seeds,
              "num_envs_per_seed_command": args.num_envs, "horizon_seconds": args.seconds,
              "warmup_seconds": args.warmup_seconds, "fall_tilt_deg": args.fall_tilt_deg,
              "calibration": calibration, "simulation_only": True,
              "metric_scope": "Tracking excludes warmup, failure step and all samples after first failure; state/target diagnostics include the terminal step before resets",
              "encoder_proxy_scope": "Raw targets mapped through saved calibration; no hardware stop, filter or clipping applied",
              "aggregate_scope": "Each command/seed contributes equal weight; tracking_runs exposes missing pre-failure tracking",
              "actor_config_source": "Requested task configuration; the checkpoint does not encode activation or full physics configuration",
              "checkpoint_environment_counter_restored": False,
              "results": [], "aggregate": {}}
    save_json(args.output / "evaluation.json", report)
    with torch.inference_mode():
        for seed in args.seeds:
            cfg = make_eval_cfg(args.task, args.profile, seed, args.num_envs, args.seconds)
            report.setdefault("profile_config", {"events": json_value(cfg.events),
                "actor_observations": json_value(cfg.observations["actor"]),
                "actuators": json_value(cfg.scene.entities["robot"].articulation.actuators),
                "terminations": json_value(cfg.terminations), "curriculum": {}})
            env = ManagerBasedRlEnv(cfg, device=args.device)
            agent = load_rl_cfg(args.task)
            wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
            try:
                actor_cfg = deepcopy(asdict(agent)["actor"])
                for key in ("class_name", "rnn_type", "rnn_hidden_dim", "rnn_num_layers", "cnn_cfg"):
                    actor_cfg.pop(key, None)
                actor = MLPModel(wrapped.get_observations(), asdict(agent)["obs_groups"],
                                 "actor", 14, **actor_cfg).to(args.device)
                actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
                actor.eval()
                robot = env.scene["robot"]
                action_term = env.action_manager.get_term("joint_pos")
                if list(action_term.target_names) != calibration["names"]:
                    raise ValueError("Policy action order does not match calibrated robot joint order")
                ids = action_term.target_ids
                home = robot.data.default_joint_pos[:, ids].clone()
                xml_bounds = robot.data.joint_pos_limits[0, ids].cpu().numpy().copy()
                report["joint_motor_xml_limits_rad"] = xml_bounds
                report["home_rad"] = home[0].cpu().numpy()
                for name in args.commands:
                    command = COMMANDS[name]
                    twist = env.command_manager.get_term("twist")
                    for field, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command):
                        setattr(twist.cfg.ranges, field, (value, value))
                    env.reset(seed=seed)
                    stats = RolloutStats(args.num_envs, command, calibration, xml_bounds,
                                         args.seconds, args.warmup_seconds, args.fall_tilt_deg)
                    previous = home.clone()
                    trace_path = args.output / f"{args.profile}_seed{seed}_{name}_trace.csv"
                    with trace_path.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=("time_s", "survival_fraction",
                            "mean_vx", "mean_vy", "mean_wz", "max_tilt_deg_first_episode",
                            "encoder_proxy_violations"))
                        writer.writeheader()
                        for step in range(round(args.seconds / env.step_dt)):
                            t = (step + 1) * env.step_dt
                            twist.command[:] = torch.tensor(command, device=args.device)
                            # Exact zero probabilities can still sample a zero
                            # uniform variate. Ensure no standing/world/heading
                            # mode rewrites the fixed body-frame command.
                            for flag in ("is_standing_env", "is_heading_env", "is_forward_env", "is_world_env"):
                                getattr(twist, flag).zero_()
                            twist.vel_command_w[:] = twist.command
                            for term in ("head_pose", "body_pose"):
                                env.command_manager.get_term(term).command.zero_()
                            obs = wrapped.get_observations()
                            # get_observations returns the previous step's cached
                            # obs: refresh the command slots before actor inference.
                            obs["actor"][:, 48:51] = torch.tensor(command, device=args.device)
                            obs["actor"][:, 51:61] = 0.
                            finite_obs = torch.isfinite(obs["actor"]).all(1).cpu().numpy()
                            stats.mark_failure(~finite_obs, t - env.step_dt, "nonfinite_observation")
                            obs["actor"].nan_to_num_()
                            actions = actor(obs)
                            finite_action = torch.isfinite(actions).all(1).cpu().numpy()
                            stats.mark_failure(~finite_action, t - env.step_dt, "nonfinite_action")
                            actions[torch.tensor(~stats.alive, device=args.device)] = 0.
                            _, _, _, _ = wrapped.step(actions)
                            target = action_term._processed_actions.clone()
                            data = robot.data
                            # BAM get_command uses output-side position through
                            # passive backlash; motor q stays separate for XML limits.
                            measured = robot.actuators[0].get_command(data).pos
                            motor_q = data.joint_pos[:, ids]
                            velocity = torch.cat((data.root_link_lin_vel_b[:, :2],
                                                  data.root_link_ang_vel_b[:, 2:3]), dim=1)
                            finite = (torch.isfinite(data.root_link_pos_w).all(1)
                                      & torch.isfinite(data.joint_pos).all(1)
                                      & torch.isfinite(data.joint_vel).all(1)
                                      & torch.isfinite(velocity).all(1))
                            terms = {key: env.termination_manager.get_term(key).cpu().numpy().copy()
                                     for key in env.termination_manager.active_terms}
                            row = stats.capture(t, velocity.cpu().numpy(),
                                data.projected_gravity_b.cpu().numpy(), measured.cpu().numpy(),
                                motor_q.cpu().numpy(), target.cpu().numpy(),
                                (target - previous).cpu().numpy(),
                                env.reset_terminated.cpu().numpy().copy(),
                                env.reset_time_outs.cpu().numpy().copy(), finite.cpu().numpy(), terms)
                            writer.writerow(row)
                            previous = target
                            if not stats.alive.any():
                                break
                            # Keep failed worlds excluded permanently, even when
                            # reset is required to let surviving worlds keep stepping.
                            pending = env.reset_terminated | env.reset_time_outs
                            failed_ids = pending.nonzero().flatten()
                            if failed_ids.numel():
                                env.reset(env_ids=failed_ids)
                                previous[failed_ids] = home[failed_ids]
                    result = json_value(stats.result())
                    result.update(command_name=name, seed=seed, profile=args.profile,
                                  trace_csv=str(trace_path.resolve()))
                    report["results"].append(result)
                    by_command = {key: aggregate_results([r for r in report["results"]
                                  if r["command_name"] == key]) for key in args.commands
                                  if any(r["command_name"] == key for r in report["results"])}
                    report["aggregate"] = {**aggregate_results(report["results"]), "results": by_command}
                    save_json(args.output / "evaluation.json", report)
                    write_csv_reports(args.output, report, calibration)
                    print(json.dumps({"command": name, "seed": seed,
                        "survival_fraction": result["survival_fraction"],
                        "mean_abs_error_vx_vy_wz": result["mean_abs_error_vx_vy_wz"],
                        "encoder_proxy_env_violation_fraction": result["encoder_proxy_env_violation_fraction"]}), flush=True)
            finally:
                wrapped.close()
    write_csv_reports(args.output, report, calibration)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("nominal", "stress"), default="stress")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345, 23456, 34567])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--warmup-seconds", type=float, default=1.)
    parser.add_argument("--fall-tilt-deg", type=float, default=45.)
    parser.add_argument("--commands", nargs="+", choices=list(COMMANDS), default=list(COMMANDS))
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--num-envs and --seconds must be positive")
    if not 0 <= args.warmup_seconds < args.seconds or not 0 < args.fall_tilt_deg < 180:
        parser.error("Warmup must be within the horizon and tilt threshold within (0, 180)")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.commands)) != len(args.commands):
        parser.error("Seeds and commands must be distinct")
    run(args)


if __name__ == "__main__":
    main()
