"""Opt-in GPU integration: MICRODUCK_TEST_GPU=1 python -m pytest this_file.

Run only after the existing training queue releases the GPU. These short
rollouts validate integration, not learned locomotion or real-world transfer.
"""

import os
import json
import math
from dataclasses import replace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401 -- register tasks


TASKS = [
    "Mjlab-Velocity-Flat-MicroDuck-HD1910",
    "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910",
]


@pytest.mark.skipif(os.environ.get("MICRODUCK_TEST_GPU") != "1", reason="explicit GPU opt-in required")
@pytest.mark.parametrize("task", TASKS)
def test_hd1910_rollout_and_partial_reset(task):
    assert torch.cuda.is_available(), "GPU validation requested, but CUDA is unavailable"
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=7)
        assert obs["actor"].shape == (4, 61)
        assert env.action_manager.total_action_dim == 14
        robot = env.scene["robot"]
        actuator = robot.actuators[0]
        assert type(actuator._bam_model.actuator).__name__ == "STS3215Actuator"
        assert actuator._bam_model.kt.value == pytest.approx(0.6237611235393989)
        if "Backlash" in task:
            assert actuator._backlash_mask.sum().item() == 14
        command = actuator.get_command(robot.data)
        # Consume the initial reset in every world before testing an isolated
        # reset. Otherwise all worlds would still discard the injected history.
        actuator.compute(replace(command, position_target=command.pos.clone()))
        assert not actuator._goal_reset_pending.any()
        actuator._bam_model.actuator.q_target_smooth = command.pos.clone() + 2.0
        actuator.reset(torch.tensor([0], device=env.device))
        actuator.compute(replace(command, position_target=command.pos.clone()))
        history = actuator._bam_model.actuator.q_target_smooth
        torch.testing.assert_close(history[0], command.pos[0])
        assert torch.all(history[1:] > command.pos[1:])
        env.reset(seed=7)
        for step in range(30):
            action = torch.zeros((4, 14), device=env.device)
            action[1, 0] = 0.1
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
            assert torch.isfinite(robot.data.joint_pos).all()
            if step == 9:
                env.reset(env_ids=torch.tensor([0], device=env.device))
    finally:
        env.close()


@pytest.mark.skipif(os.environ.get("MICRODUCK_TEST_GPU") != "1", reason="explicit GPU opt-in required")
@pytest.mark.parametrize("task", TASKS)
def test_hd1910_home_hold_for_three_seconds(task, record_property):
    """Gate long runs on an upright HOME at fixed 7.4 V, with small perturbations.

Four worlds: nominal HOME and three deterministic +/-0.01 rad perturbations
of the servo angles and root roll/pitch. No pushes, DR or automatic resets.
Pass requires maximum trunk tilt <15 degrees and final height >0.09 m in each
world. These are screening thresholds, not guarantees of hardware stability.
"""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz

    assert torch.cuda.is_available(), "GPU validation requested, but CUDA is unavailable"
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 4
    cfg.events = {"expand_bam_friction_fields": cfg.events["expand_bam_friction_fields"]}
    cfg.curriculum = {}
    cfg.terminations = {}
    cfg.rewards = {}
    cfg.auto_reset = False
    actuator_cfg = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator_cfg.vin_range = (7.4, 7.4)
    actuator_cfg.vin_drop_gain_range = (0.0, 0.0)
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        env.reset(seed=7)
        robot = env.scene["robot"]
        servo_ids = [i for i, name in enumerate(robot.joint_names) if not name.startswith("passive_")]
        assert len(servo_ids) == 14
        roll = torch.tensor([0.0, 0.01, -0.01, 0.01], device=env.device)
        pitch = torch.tensor([0.0, -0.01, 0.01, 0.01], device=env.device)
        root = robot.data.default_root_state.clone()
        root[:, :3] = env.scene.env_origins
        root[:, 2] += 0.125  # Same initial-height range as the walking recipe.
        root[:, 3:7] = quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))
        root[:, 7:] = 0.0
        joints = robot.data.default_joint_pos.clone()
        joint_signs = torch.tensor([1.0, -1.0] * 7, device=env.device)
        offsets = torch.tensor([0.0, 0.01, -0.01, 0.005], device=env.device)
        joints[:, servo_ids] += offsets[:, None] * joint_signs
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        robot.actuators[0].reset()
        env.sim.forward()
        env.scene.update(dt=0.0)

        max_tilt = torch.zeros(4, device=env.device)
        action = torch.zeros((4, 14), device=env.device)  # Default joint offsets = HOME.
        for _ in range(math.ceil(3.0 / env.step_dt)):
            obs, _, terminated, truncated, _ = env.step(action)
            assert not terminated.any() and not truncated.any(), "HOME trial must not auto-reset"
            assert obs["actor"].shape == (4, 61)
            assert torch.isfinite(obs["actor"]).all()
            gravity = robot.data.projected_gravity_b
            tilt = torch.rad2deg(torch.acos(torch.clamp(-gravity[:, 2], -1.0, 1.0)))
            max_tilt = torch.maximum(max_tilt, tilt)
            assert torch.isfinite(robot.data.joint_pos).all()
        height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
        report = {
            "task": task, "duration_s": 3.0, "supply_v": 7.4,
            "max_tilt_deg": max_tilt.tolist(), "final_height_m": height.tolist(),
            "tilt_limit_deg": 15.0, "minimum_height_m": 0.09,
        }
        print("HD1910_HOME_HOLD " + json.dumps(report))
        record_property("home_hold", json.dumps(report))
        assert torch.isfinite(max_tilt).all() and torch.all(max_tilt < 15.0), report
        assert torch.isfinite(height).all() and torch.all(height > 0.09), report
    finally:
        env.close()
