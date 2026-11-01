"""Offline HOME physics preflight and optional checkpoint evaluation.

Run with the existing training interpreter; this script never trains or connects
to hardware. A preflight passes only the nominal HOME stability screen, not
observability, learnability, calibration accuracy or real-world transfer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np


TASK = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
NO_CURRENT_TASK = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
FORCE_RATIO_TASK = "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"


class FirstEpisodeMetrics:
    """Freeze each world's statistics at its first failure or time limit.

    Inputs are CPU arrays. In particular, later recovered/fallen/reset states
    cannot improve a failed world's endpoint or trajectory error.
    """

    def __init__(self, initial_error_rad: np.ndarray):
        initial = np.rad2deg(np.asarray(initial_error_rad, dtype=float))
        if initial.ndim != 2 or not np.isfinite(initial).all():
            raise ValueError("initial joint error must be a finite [env, joint] array")
        self.initial = initial.copy()
        self.endpoint = initial.copy()
        self.active = np.ones(initial.shape[0], dtype=bool)
        self.count = np.zeros(initial.shape[0], dtype=int)
        self.abs_sum = np.zeros_like(initial)
        self.square_sum = np.zeros_like(initial)
        self.max_tilt = np.zeros(initial.shape[0])
        self.min_height = np.full(initial.shape[0], np.inf)
        self.last_time = np.zeros(initial.shape[0])
        self.stop_time: list[float | None] = [None] * initial.shape[0]
        self.stop_reasons: list[list[str]] = [[] for _ in self.active]

    def update(self, error_rad, tilt_deg, height_m, stop_reasons, time_s):
        error = np.rad2deg(np.asarray(error_rad, dtype=float))
        tilt = np.asarray(tilt_deg, dtype=float)
        height = np.asarray(height_m, dtype=float)
        if error.shape != self.initial.shape or tilt.shape != self.active.shape:
            raise ValueError("metric input shape changed during an episode")
        if height.shape != self.active.shape or len(stop_reasons) != len(self.active):
            raise ValueError("height/reason count must match the number of worlds")
        active = self.active.copy()
        finite = np.isfinite(error).all(axis=1) & np.isfinite(tilt) & np.isfinite(height)
        valid = active & finite
        self.endpoint[active] = error[active]
        self.abs_sum[valid] += np.abs(error[valid])
        self.square_sum[valid] += error[valid] ** 2
        self.count[valid] += 1
        self.max_tilt[valid] = np.maximum(self.max_tilt[valid], tilt[valid])
        self.min_height[valid] = np.minimum(self.min_height[valid], height[valid])
        self.last_time[active] = time_s
        for index in np.flatnonzero(active):
            reasons = list(stop_reasons[index])
            if not finite[index]:
                reasons.append("nonfinite_state")
            if reasons:
                self.stop_time[index] = time_s
                self.stop_reasons[index] = reasons
                self.active[index] = False

    def report(self, *, completed_horizon: bool):
        count = int(self.count.sum())
        survivors = self.endpoint[self.active] if completed_horizon else np.empty((0, 0))
        return {
            "initial_mae_deg_per_joint": np.abs(self.initial).mean(axis=0).tolist(),
            "endpoint_mae_deg_per_joint": np.abs(self.endpoint).mean(axis=0).tolist(),
            "endpoint_rmse_deg_per_joint": np.sqrt((self.endpoint ** 2).mean(axis=0)).tolist(),
            "endpoint_error_deg_per_env_joint": self.endpoint.tolist(),
            "endpoint_definition": "Last state of the first episode: failure/time-limit state, or requested horizon.",
            "trajectory_mae_deg_per_joint": (self.abs_sum.sum(axis=0) / count).tolist() if count else None,
            "trajectory_rmse_deg_per_joint": np.sqrt(self.square_sum.sum(axis=0) / count).tolist() if count else None,
            "survivor_endpoint_mae_deg_per_joint": np.abs(survivors).mean(axis=0).tolist() if survivors.size else None,
            "completed_requested_horizon": completed_horizon,
            "first_episode_horizon_survival_fraction": float(self.active.mean()) if completed_horizon else None,
            "first_episode_stop_time_s": self.stop_time,
            "first_episode_stop_reasons": self.stop_reasons,
            "observed_first_episode_duration_s": self.last_time.tolist(),
            "valid_samples_per_env": self.count.tolist(),
            "max_tilt_deg_per_env": self.max_tilt.tolist(),
            "minimum_height_m_per_env": self.min_height.tolist(),
        }


def _numpy(tensor):
    return tensor.detach().cpu().numpy()


def _state(env, joint_ids):
    import torch

    robot = env.scene["robot"]
    error = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))
    height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return _numpy(error), _numpy(tilt), _numpy(height)


def _geometry(env, joint_ids):
    robot = env.scene["robot"]
    foot_ids, foot_names = robot.find_geoms(r"^(left|right)_foot_collision$")
    if len(foot_ids) != 2:
        raise RuntimeError(f"Expected two named foot collision geoms, found {foot_names}")
    feet = robot.data.geom_pos_w[:, foot_ids]
    root = robot.data.root_link_pos_w
    return {
        "true_joint_angles_deg": np.rad2deg(_numpy(robot.data.joint_pos[:, joint_ids])).tolist(),
        "trunk_position_m_relative_to_env_origin": _numpy(root - env.scene.env_origins).tolist(),
        "foot_collision_geom_names": list(foot_names),
        "foot_collision_geom_frame_positions_m_relative_to_trunk": _numpy(feet - root[:, None, :]).tolist(),
        "foot_geometry_note": "Mesh geom frame origins are recorded; their z coordinates are not sole clearances.",
    }


def _noisy_nominal_start(env, joint_ids):
    """Same +/-0.01-rad screen as test_hd1910_home_hold_for_three_seconds."""
    import torch
    from mjlab.utils.lab_api.math import quat_from_euler_xyz

    robot = env.scene["robot"]
    row = torch.arange(env.num_envs, device=env.device)
    offsets = torch.tensor([0.0, 0.01, -0.01, 0.005], device=env.device)[row % 4]
    roll = torch.tensor([0.0, 0.01, -0.01, 0.01], device=env.device)[row % 4]
    pitch = torch.tensor([0.0, -0.01, 0.01, 0.01], device=env.device)[row % 4]
    root = robot.data.default_root_state.clone()
    root[:, :3] = env.scene.env_origins
    root[:, 2] += 0.125
    root[:, 3:7] = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
    root[:, 7:] = 0
    joints = robot.data.default_joint_pos.clone()
    signs = torch.tensor([1.0, -1.0] * 7, device=env.device)
    joints[:, joint_ids] += offsets[:, None] * signs
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
    for actuator in robot.actuators:
        actuator.reset()
    env.sim.forward()
    env.scene.update(dt=0.0)


def _rollout(env, checker, *, seconds, seed, policy=None, noisy_start=False,
             actor_dim=392, target_ratio=None):
    import torch
    from tensordict import TensorDict

    obs, _ = env.reset(seed=seed)
    checker.reset()
    action_term = env.action_manager.get_term("home_calibration")
    joint_ids = action_term.joint_ids
    if noisy_start:
        _noisy_nominal_start(env, joint_ids)
        # This case uses zero actions, so no stale initial actor frame is read.
    expected = (env.num_envs, actor_dim)
    if tuple(obs["actor"].shape) != expected:
        raise RuntimeError(f"HOME actor shape {tuple(obs['actor'].shape)} != {expected}")
    if env.action_manager.total_action_dim != 14:
        raise RuntimeError("HOME calibration requires 14 incremental actions")
    initial = _geometry(env, joint_ids)
    initial_bias = _numpy(action_term.bias).copy()
    errors, _, _ = _state(env, joint_ids)
    metrics = FirstEpisodeMetrics(errors)
    ratio_endpoint = np.zeros((env.num_envs, 14))
    ratio_total_endpoint = np.zeros(env.num_envs)
    ratio_l1_sum = np.zeros(env.num_envs)
    ratio_count = np.zeros(env.num_envs, dtype=int)
    endpoint_geometry = copy.deepcopy(initial)
    steps = math.ceil(seconds / env.step_dt)
    completed = True
    for step in range(steps):
        # DelayBuffer lazily allocates persistent tensors while stepping. Use
        # no_grad so a later env.reset outside this block can update them.
        with torch.no_grad():
            if policy is None:
                action = torch.zeros((env.num_envs, 14), device=env.device)
            else:
                action = policy(TensorDict(obs, batch_size=[env.num_envs]))
            if not torch.isfinite(action).all():
                raise RuntimeError("Checkpoint produced a nonfinite action")
            obs, _, _, _, _ = env.step(action)
        # Separate manager: no resets or extra observation-history updates.
        checker.compute()
        reasons = [[] for _ in range(env.num_envs)]
        for name in checker.active_terms:
            for index in np.flatnonzero(_numpy(checker.get_term(name))):
                reasons[index].append(name)
        error, tilt, height = _state(env, joint_ids)
        actor_finite = _numpy(torch.isfinite(obs["actor"]).all(dim=1))
        for index in np.flatnonzero(~actor_finite):
            reasons[index].append("nonfinite_actor_observation")
        active = metrics.active.copy()
        if target_ratio is not None:
            tau = np.abs(_numpy(env.scene["robot"].data.qfrc_actuator[:, joint_ids]))
            total = tau.sum(axis=1)
            ratio = tau / np.maximum(total[:, None], 1e-12)
            finite_force = np.isfinite(ratio).all(axis=1) & np.isfinite(total)
            for index in np.flatnonzero(~finite_force):
                reasons[index].append("nonfinite_force_ratio")
            valid = active & finite_force
            ratio_endpoint[active] = ratio[active]
            ratio_total_endpoint[active] = total[active]
            ratio_l1_sum[valid] += np.abs(ratio[valid] - target_ratio).sum(axis=1)
            ratio_count[valid] += 1
        metrics.update(error, tilt, height, reasons, (step + 1) * env.step_dt)
        # Capture geometric endpoints before continuing any failed worlds.
        geometry = _geometry(env, joint_ids)
        for key in ("true_joint_angles_deg", "trunk_position_m_relative_to_env_origin",
                    "foot_collision_geom_frame_positions_m_relative_to_trunk"):
            for index in np.flatnonzero(active):
                endpoint_geometry[key][index] = geometry[key][index]
        if not (np.isfinite(error).all() and np.isfinite(tilt).all()
                and np.isfinite(height).all() and actor_finite.all()):
            completed = step == steps - 1
            break
    result = metrics.report(completed_horizon=completed)
    if target_ratio is not None:
        result["force_ratio"] = {
            "target": list(target_ratio),
            "definition": "abs(qfrc_actuator_servo)/sum(abs(qfrc_actuator_servo))",
            "endpoint_ratio_per_env_joint": ratio_endpoint.tolist(),
            "endpoint_l1_error_per_env": np.abs(ratio_endpoint - target_ratio).sum(axis=1).tolist(),
            "trajectory_l1_error_per_env": (ratio_l1_sum / np.maximum(ratio_count, 1)).tolist(),
            "endpoint_sum_absolute_torque_nm": ratio_total_endpoint.tolist(),
        }
    result.update({
        "requested_duration_s": seconds,
        "control_dt_s": env.step_dt,
        "actor_shape": list(expected),
        "encoder_bias_deg_per_env_joint": np.rad2deg(initial_bias).tolist(),
        "initial_geometry": initial,
        "first_episode_endpoint_geometry": endpoint_geometry,
    })
    return result


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def run(args):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.termination_manager import TerminationManager
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.runners import OnPolicyRunner

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab_microduck.tasks.microduck_home_calibration_env_cfg import make_microduck_home_calibration_env_cfg

    configure_torch_backends()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this interpreter")
    force_ratio = getattr(args, "objective", "pose") == "force-ratio"
    no_current = getattr(args, "no_current", False) or force_ratio
    task = FORCE_RATIO_TASK if force_ratio else (NO_CURRENT_TASK if no_current else TASK)
    if force_ratio:
        from mjlab_microduck.tasks.microduck_home_force_ratio_env_cfg import make_microduck_home_force_ratio_env_cfg
        cfg = make_microduck_home_force_ratio_env_cfg(evaluation_profile=args.profile)
    else:
        cfg = make_microduck_home_calibration_env_cfg(
            evaluation_profile=args.profile, include_current=not no_current)
    target_ratio = cfg.observations["actor"].terms["sensor_frame"].params["target_ratio"] if force_ratio else None
    cfg.scene.num_envs = args.num_envs
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.rewards = {}
    cfg.episode_length_s = max(cfg.episode_length_s, args.seconds + 1, 4.0)
    termination_cfg = cfg.terminations
    cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        checker = TerminationManager(termination_cfg, env)
        term = env.action_manager.get_term("home_calibration")
        if no_current:
            # Evaluate exact-zero and random-bias populations separately.
            term.cfg.zero_bias_probability = 0.0
        def rollout(**kwargs):
            return _rollout(env, checker, actor_dim=384 if force_ratio else (272 if no_current else 392),
                            target_ratio=target_ratio, **kwargs)
        original_bias_range = term.cfg.bias_range_rad
        policy = None
        if args.checkpoint is not None:
            agent_cfg = load_rl_cfg(task)
            wrapper = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner_cls = load_runner_cls(task) or OnPolicyRunner
            runner = runner_cls(wrapper, asdict(agent_cfg), device=args.device)
            runner.load(str(args.checkpoint.resolve()), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
        cases = {}
        # Nuisance-profile runs use a separate nominal invocation for this gate.
        if args.profile == "nominal":
            term.cfg.bias_range_rad = (0.0, 0.0)
            hold = rollout(seconds=3.0, seed=args.seed, noisy_start=True)
            heights = np.asarray(hold["first_episode_endpoint_geometry"]["trunk_position_m_relative_to_env_origin"])[:, 2]
            hold["stability_screen_passed"] = bool(
                hold["completed_requested_horizon"]
                and hold["first_episode_horizon_survival_fraction"] == 1.0
                and max(hold["max_tilt_deg_per_env"]) < 15.0
                and np.all(heights > 0.09)
            )
            hold["stability_screen_thresholds"] = {"maximum_tilt_deg": 15.0, "minimum_final_trunk_height_m": 0.09}
            cases["true_home_hold_3s"] = hold
        for label, bias_range in (
            ("zero_action_zero_bias", (0.0, 0.0)),
            ("zero_action_random_bias", original_bias_range),
            ("zero_action_all_joints_bias_plus_1deg", (math.radians(1), math.radians(1))),
            ("zero_action_all_joints_bias_minus_1deg", (-math.radians(1), -math.radians(1))),
        ):
            term.cfg.bias_range_rad = bias_range
            cases[label] = rollout(seconds=args.seconds, seed=args.seed)
        if policy is not None:
            term.cfg.bias_range_rad = (0.0, 0.0)
            cases["checkpoint_zero_bias"] = rollout(
                seconds=args.seconds, seed=args.seed, policy=policy)
            term.cfg.bias_range_rad = original_bias_range
            cases["checkpoint_random_bias"] = rollout(
                seconds=args.seconds, seed=args.seed, policy=policy
            )
        return {
            "task": task,
            "current_input": not no_current,
            "objective": "force-ratio" if force_ratio else "pose",
            "mode": "checkpoint_evaluation" if policy is not None else "physics_preflight_only",
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "evaluation_profile": args.profile,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "device": args.device,
            "joint_names": list(term.joint_names),
            "ideal_home_angles_deg": np.rad2deg(_numpy(term.home[0])).tolist(),
            "source_assumptions": {
                "home": "robot/microduck_constants.py HOME_FRAME, this repository's STAND2 geometry",
                "actuator": "robot/hd1910/1910_m6.json public LuwuDynamics M6 prior; not measured on this robot",
                "bias_convention": "q_reported = q_true + bias; q_target_true = HOME + command_offset - bias",
                "current": ("No current in actor or critic sensor frames; no current mapping needed at deployment"
                            if no_current else "Simulated motor-current proxy and configured noise; no measured hardware current-to-torque calibration"),
                "nominal_hold": "Nominal profile only: fixed 7.4 V, zero calibration bias, +/-0.01 rad initial joint/root perturbations",
                "geometry": "True simulation joint angles and collision-geom frames; not encoder-biased observations",
                "failure_measurement": "Separate task TerminationManager evaluated after forward kinematics; metrics freeze at first stop; no automatic/manual resets during a rollout",
                "timeout": "Episode time limit extended beyond requested horizon for survival measurement",
                "limitations": "Preflight and zero-action baselines do not establish observability, learnability, trained success or real-world transfer. Uniform +/-1 deg probes do not test all independent offset directions.",
            },
            "cases": cases,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Optional local rsl_rl model_*.pt; normalizer is retained")
    parser.add_argument("--output", type=Path, default=Path("home_calibration_preflight.json"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("nominal", "train", "stress"), default="nominal")
    parser.add_argument("--no-current", action="store_true", help="Evaluate the separate 272D encoder/IMU task")
    parser.add_argument("--objective", choices=("pose", "force-ratio"), default="pose")
    args = parser.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--num-envs and --seconds must be positive")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    report = _json_safe(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output.resolve()}")
    hold = report["cases"].get("true_home_hold_3s")
    if hold is not None and not hold["stability_screen_passed"]:
        raise SystemExit("Nominal HOME stability screen failed; inspect the saved diagnostics before training.")


if __name__ == "__main__":
    main()
