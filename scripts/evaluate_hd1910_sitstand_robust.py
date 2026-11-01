"""Deterministic sit/stand cycles, capturing first failures before manual reset."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.models import MLPModel
from mjlab_microduck.tasks.microduck_hd1910_sitstand_robust_env_cfg import (
    TASK_ID, make_microduck_hd1910_sitstand_robust_env_cfg,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy_hd1910"))
from microduck_deploy.policy import HOME, JOINT_LIMITS  # no hardware imports


@torch.inference_mode()
def evaluate(checkpoint, profile, count, seed, calibration, ankle_p):
    cfg = make_microduck_hd1910_sitstand_robust_env_cfg(evaluation_profile=profile)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = count, seed, False
    cfg.episode_length_s = 15.
    cfg.commands["twist"].resampling_time_range = (1000., 1000.)
    cfg.commands["head_pose"].ranges = ((0., 0.),) * 4
    cfg.commands["head_pose"].zero_command_prob = 1.
    cfg.events["set_ground_state"].params.update({"sitting_prob": 0., "standing_prob": 1.})
    if profile == "nominal":
        cfg.events["sitstand_response"].params["ankle_kp_range"] = (ankle_p / 5.,) * 2
        cfg.events["reset_base"].params["pose_range"].update({
            "roll": (-.015, .015), "pitch": (-.015, .015),
        })
        cfg.events["reset_robot_joints"].params["position_range"] = (-.015, .015)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    result = {"checkpoint": str(checkpoint), "profile": profile, "num_envs": count,
              "seed": seed, "ankle_p": ankle_p if profile == "nominal" else "training DR",
              "schedule": "stand 0..3s; sit 3..8s; stand 8..13s", "failures": []}
    try:
        obs, _ = env.reset(seed=seed)
        agent = asdict(load_rl_cfg(TASK_ID))
        kwargs = {k: deepcopy(v) for k, v in agent["actor"].items()
                  if k in inspect.signature(MLPModel.__init__).parameters}
        actor = MLPModel(TensorDict(obs, batch_size=[count]), agent["obs_groups"],
                         "actor", 14, **kwargs).to(env.device).eval()
        actor.load_state_dict(torch.load(checkpoint, map_location=env.device, weights_only=True)["actor_state_dict"])
        command = env.command_manager.get_term("twist")
        alive = torch.ones(count, dtype=torch.bool, device=env.device)
        peak = torch.zeros(count, device=env.device)
        violations = torch.zeros(count, device=env.device)
        active_steps = torch.zeros_like(violations)
        windows = {k: {"height": torch.zeros(count, device=env.device),
                        "samples": torch.zeros(count, device=env.device)}
                   for k in ("initial_stand", "sit", "return_stand")}
        zero = torch.tensor([j["zero_tick"] for j in calibration["joints"]], device=env.device)
        direction = torch.tensor([j["direction"] for j in calibration["joints"]], device=env.device)
        low = torch.tensor(JOINT_LIMITS[:, 0] - .12, device=env.device)
        high = torch.tensor(JOINT_LIMITS[:, 1] + .12, device=env.device)
        home = torch.tensor(HOME, dtype=torch.float32, device=env.device)
        previous_target = None
        jump_max = 0.
        all_finite = True
        for step in range(650):
            t = step * .02
            command.vel_command_b.zero_()
            command.vel_command_b[:, 0] = float(3. <= t < 8.)
            obs = env.observation_manager.compute(update_history=False)
            # compute(False) intentionally returns the post-step cache. Replace
            # only the external command slots; do not push delay buffers twice.
            actor_obs = obs["actor"].clone()
            actor_obs[:, 48:51] = command.vel_command_b
            assert torch.all(actor_obs[:, 48] == float(3. <= t < 8.))
            action = actor(TensorDict({"actor": actor_obs}, batch_size=[count]))
            target = home + action
            if previous_target is not None and alive.any():
                jump_max = max(jump_max, float(torch.rad2deg((target-previous_target).abs()[alive]).max()))
            previous_target = target.clone()
            raw_ticks = zero + direction * target * 4096. / (2 * np.pi)
            overshoot_deg = torch.relu(-raw_ticks) + torch.relu(raw_ticks-4095.)
            encoder_bad = (overshoot_deg.max(dim=1).values * 360 / 4096 > 5.)
            _, _, terminated, timed_out, _ = env.step(action)
            robot = env.scene["robot"]
            tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1., 1.)))
            position = robot.data.joint_pos_biased
            joint_bad = ((position < low) | (position > high)).any(dim=1)
            finite = torch.isfinite(position).all(dim=1) & torch.isfinite(tilt) & torch.isfinite(action).all(dim=1)
            all_finite &= bool(finite.all())
            peak = torch.where(alive, torch.maximum(peak, tilt), peak)
            violations += (alive & (tilt > 15.)).float()
            active_steps += alive.float()
            bad = terminated | timed_out | encoder_bad | joint_bad | ~finite
            first = bad & alive
            for i in first.nonzero(as_tuple=False).flatten().tolist():
                reason = ("nonfinite" if not finite[i] else "encoder_target" if encoder_bad[i]
                          else "joint_feedback" if joint_bad[i] else "tilt" if tilt[i] > 35.
                          else "termination")
                result["failures"].append({"env": i, "time_s": round(t+.02, 3),
                                           "reason": reason, "tilt_deg": float(tilt[i])})
            alive &= ~bad
            z = robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
            for name, lo, hi in (("initial_stand", 2., 3.), ("sit", 7., 8.), ("return_stand", 12., 13.)):
                if lo <= t < hi:
                    windows[name]["height"] += torch.where(alive, z, 0.)
                    windows[name]["samples"] += alive.float()
            reset = (terminated | timed_out).nonzero(as_tuple=False).flatten()
            if len(reset):
                env.reset(env_ids=reset)
        success = alive.clone()
        summary = {}
        for name, ideal in (("initial_stand", .115), ("sit", .060), ("return_stand", .115)):
            values = windows[name]
            height = values["height"] / values["samples"].clamp_min(1)
            reached = (values["samples"] == 50) & ((height - ideal).abs() < .020)
            success &= reached
            summary[name] = {"height_success": int(reached.sum()),
                             "mean_height_m": float(height[reached].mean()) if reached.any() else None,
                             "all_survivor_mean_height_m": float(height[alive].mean()) if alive.any() else None,
                             "all_survivor_min_height_m": float(height[alive].min()) if alive.any() else None,
                             "all_survivor_max_height_m": float(height[alive].max()) if alive.any() else None}
        result.update({"passed": bool(success.all()), "survived": int(alive.sum()),
                       "successful_complete_cycles": int(success.sum()), "windows": summary,
                       "max_tilt_deg": float(peak.max()), "mean_peak_tilt_deg": float(peak.mean()),
                       "tilt_violation_frame_fraction": float(violations.sum()/active_steps.sum()),
                       "max_target_step_deg": jump_max, "finite": all_finite,
                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
    finally:
        env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", default=["nominal", "stress"])
    parser.add_argument("--ankle-p", type=float, default=5.)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for profile in args.profiles:
        result = evaluate(args.checkpoint, profile, args.num_envs, args.seed,
                          json.loads(args.calibration.read_text()), args.ankle_p)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
