"""Check realized DR and noisy HOME/SIT equilibrium on the actual Warp model."""

from copy import deepcopy
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks.microduck_hd1910_sitstand_robust_env_cfg import (
    ROBUST_STAGES, make_microduck_hd1910_sitstand_robust_env_cfg,
)
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SITTING_TARGET_OVERRIDES


@torch.inference_mode()
def main():
    cfg = make_microduck_hd1910_sitstand_robust_env_cfg(evaluation_profile="nominal")
    cfg.scene.num_envs = 16
    cfg.seed = 42
    defaults = {k: deepcopy(v.params) for k, v in cfg.events.items()}
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    report = {"realized_stages": [], "equilibrium": {}}
    try:
        for stage in ROBUST_STAGES:
            for name, params in stage["events"].items():
                env.event_manager.get_term_cfg(name).params.update(params)
            obs, _ = env.reset()
            assert obs["actor"].shape == (16, 61)
            asset = env.scene["robot"]
            actuator = asset.actuators[0]
            names = actuator._target_names
            ankle = [names.index(n) for n in ("left_ankle", "right_ankle")]
            expected = stage["events"]["sitstand_response"]["ankle_kp_range"]
            actual = actuator.kp_scale[:, ankle]
            assert actual.min() >= expected[0] - 1.e-6 and actual.max() <= expected[1] + 1.e-6
            bias = asset.data.encoder_bias.abs().max().item()
            assert bias <= stage["events"]["encoder_bias"]["bias_range"][1] + 1.e-6
            # Verify both actual mass scale and the independent head multiplier.
            ids = asset.indexing.body_ids
            mass = env.sim.model.body_mass[:, ids]
            nominal = env.sim.get_default_field("body_mass")[ids]
            ratio = mass / nominal
            trunk = asset.body_names.index("trunk_base")
            head = asset.body_names.index("jaw_soft")
            limits = stage["events"]["sitstand_inertia"]
            assert ratio[:, trunk].min() >= limits["mass_range"][0] - 1.e-6
            assert ratio[:, trunk].max() <= limits["mass_range"][1] + 1.e-6
            relative_head = ratio[:, head] / ratio[:, trunk]
            assert relative_head.min() >= limits["head_mass_range"][0] - 1.e-6
            assert relative_head.max() <= limits["head_mass_range"][1] + 1.e-6
            report["realized_stages"].append({
                "iteration": stage["step"] // 24,
                "ankle_p_min": float(actual.min()) * 5.,
                "ankle_p_max": float(actual.max()) * 5.,
                "max_bias_deg": float(torch.rad2deg(torch.tensor(bias))),
                "global_mass_scale_min": float(ratio[:, trunk].min()),
                "global_mass_scale_max": float(ratio[:, trunk].max()),
            })
        for name, params in defaults.items():
            env.event_manager.get_term_cfg(name).params = deepcopy(params)
        for mode in ("home", "sit"):
            ground = env.event_manager.get_term_cfg("set_ground_state").params
            ground.update({"sitting_prob": float(mode == "sit"),
                           "standing_prob": float(mode == "home"),
                           "sitting_joint_noise_std": .03,
                           "sitting_tilt_max": .035})
            env.reset()
            target = asset.data.default_joint_pos.clone()
            if mode == "sit":
                for index, value in SITTING_TARGET_OVERRIDES.items():
                    target[:, index] = value
            raw = target - asset.data.default_joint_pos
            maximum = torch.zeros(16, device=env.device)
            failures = torch.zeros(16, dtype=torch.bool, device=env.device)
            with torch.inference_mode():
                for _ in range(150):
                    _, _, terminated, _, _ = env.step(raw)
                    failures |= terminated
                    tilt = torch.rad2deg(torch.acos((-asset.data.projected_gravity_b[:, 2]).clamp(-1., 1.)))
                    maximum = torch.maximum(maximum, tilt)
            report["equilibrium"][mode] = {
                "seconds": 3., "worlds": 16, "failed_worlds": int(failures.sum()),
                "max_tilt_deg": float(maximum.max()), "mean_max_tilt_deg": float(maximum.mean()),
            }
            print(mode, json.dumps(report["equilibrium"][mode]), flush=True)
        report["passed"] = all(v["failed_worlds"] == 0 for v in report["equilibrium"].values())
    finally:
        env.close()
    path = Path(__file__).resolve().parents[1] / "runs/20261101_sitstand_robust/physical_check.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("Target equilibrium check failed")


if __name__ == "__main__":
    main()
