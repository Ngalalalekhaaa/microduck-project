"""Robust HD1910 contract, reward gradients and non-accumulating DR gates."""

from copy import deepcopy
import math
import os
from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import hd1910_observations as encoder_obs
from mjlab_microduck.tasks import hd1910_robust_mdp as mdp
from mjlab_microduck.tasks.microduck_hd1910_robust_env_cfg import (
    CONTINUATION_START_STEP,
    HEAD_COMMAND_RANGES,
    ROBUST_STAGES,
    make_microduck_hd1910_robust_velocity_env_cfg,
)


TASKS = (
    "Mjlab-Velocity-Flat-MicroDuck-HD1910-Robust",
    "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910-Robust",
)


@pytest.mark.parametrize("task", TASKS)
def test_registered_contract_and_resume_stages(task):
    cfg = load_env_cfg(task)
    model = cfg.scene.entities["robot"].spec_fn().compile()
    assert model.nu == 14
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "command", "head_command", "body_command",
    ]
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(.02)
    assert cfg.observations["actor"].terms["joint_pos"].func is encoder_obs.joint_pos_rel_quantized
    assert cfg.observations["actor"].terms["joint_vel"].func is encoder_obs.joint_vel_finite_difference
    assert cfg.observations["actor"].terms["joint_vel"].delay_max_lag == 0
    assert cfg.events["reset_encoder_history"].func is encoder_obs.reset_hd1910_encoder_history
    assert cfg.scene.entities["robot"].articulation.actuators[0].per_joint_randomization
    assert cfg.scene.entities["robot"].articulation.actuators[0].kp_fw == 5
    assert [s["step"] for s in cfg.curriculum["robustness"].params["stages"]] == [
        CONTINUATION_START_STEP + n * 24 for n in (0, 250, 750, 1500)
    ]
    assert cfg.commands["head_pose"].ranges == HEAD_COMMAND_RANGES
    assert cfg.commands["head_pose"].zero_command_prob == .8
    assert all(bounds == (0., 0.) for bounds in cfg.commands["body_pose"].ranges)
    assert cfg.commands["body_pose"].zero_command_prob == 1.0
    assert len(cfg.commands["head_pose"].ranges) == 4
    assert len(cfg.commands["body_pose"].ranges) == 6
    assert load_rl_cfg(task).experiment_name in (
        "microduck_hd1910_robust_flat", "microduck_hd1910_robust_backlash",
    )
    assert "head_home_l2" not in cfg.rewards
    assert cfg.rewards["head_command_l2"].func is mdp.hd1910_head_command_l2


def test_head_command_cost_follows_target_through_backlash():
    target = torch.tensor([[0., 0., .3, .06], [0., 0., .3, .06]])
    data = SimpleNamespace(
        # World 0 sits at HOME; world 1 reaches the command through gear play.
        joint_pos=torch.tensor([[0., 0., 0., 0., 0., 0.],
                                [0., 0., .29, .07, .01, -.01]]),
        default_joint_pos=torch.zeros(2, 6),
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: target),
        _head_pose_neck_ids=[0, 1, 2, 3],
        _head_pose_bl_ids=[0, 0, 4, 5],
        _head_pose_bl_mask=torch.tensor([0., 0., 1., 1.]),
    )
    cost = mdp.hd1910_head_command_l2(env)
    assert cost[0] > 0.
    assert cost[1].item() == pytest.approx(0., abs=1e-12)
    target.zero_()
    assert mdp.hd1910_head_command_l2(env)[1] > 0.


@pytest.mark.parametrize("backlash", [False, True])
def test_frozen_eval_profiles_and_factory_isolation(backlash):
    training = make_microduck_hd1910_robust_velocity_env_cfg(backlash=backlash)
    stress = make_microduck_hd1910_robust_velocity_env_cfg(backlash=backlash, evaluation_profile="stress")
    nominal = make_microduck_hd1910_robust_velocity_env_cfg(backlash=backlash, evaluation_profile="nominal")
    assert not stress.curriculum and not nominal.curriculum
    assert training.events["foot_friction"].params["ranges"] == (.65, 1.2)
    assert stress.events["foot_friction"].params["ranges"] == (.45, 1.25)
    response = stress.events["robust_actuator_response"].params
    assert response["voltage_range"] == (6.8, 8.) and response["voltage_floor"] == 6.4
    assert response["kp_range"] == (.88, 1.12)
    assert nominal.scene.entities["robot"].articulation.actuators[0].vin == 7.6
    assert "robust_actuator_response" not in nominal.events
    assert "push_robot" not in nominal.events and "encoder_bias" not in nominal.events
    assert not nominal.observations["actor"].enable_corruption
    assert nominal.scene.entities["robot"].spec_fn is stress.scene.entities["robot"].spec_fn
    stress.events["foot_friction"].params["ranges"] = (10., 20.)
    stress.scene.entities["robot"].init_state.joint_pos[r".*head_yaw.*"] = 1.
    again = make_microduck_hd1910_robust_velocity_env_cfg(backlash=backlash, play=True)
    assert again.events["foot_friction"].params["ranges"] == (.45, 1.25)
    assert again.scene.entities["robot"].init_state.joint_pos[r".*head_yaw.*"] == 0.
    baseline = load_env_cfg(TASKS[int(backlash)].removesuffix("-Robust"))
    assert baseline.rewards["track_linear_velocity"].weight == 2.
    assert not baseline.scene.entities["robot"].articulation.actuators[0].per_joint_randomization


def test_curriculum_updates_live_managers_at_saved_counter():
    cfg = make_microduck_hd1910_robust_velocity_env_cfg()
    events, rewards = deepcopy(cfg.events), deepcopy(cfg.rewards)
    env = SimpleNamespace(
        device="cpu", common_step_counter=CONTINUATION_START_STEP,
        event_manager=SimpleNamespace(get_term_cfg=events.__getitem__),
        reward_manager=SimpleNamespace(get_term_cfg=rewards.__getitem__),
    )
    assert mdp.hd1910_robust_curriculum(env, None, ROBUST_STAGES).item() == 0.
    env.common_step_counter += 750 * 24
    assert mdp.hd1910_robust_curriculum(env, None, ROBUST_STAGES).item() == 2.
    assert events["foot_friction"].params["ranges"] == (.5, 1.25)
    assert rewards["track_linear_velocity"].params["std"] == .2
    assert cfg.events["foot_friction"].params["ranges"] == (.65, 1.2)


def test_reward_prefers_low_speed_tracking_and_prices_yaw_error():
    # Same gait bounce in both worlds: only world 1 follows vx=.05.
    command = torch.tensor([[.05, 0., .3], [.05, 0., .3]])
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.tensor([[0., 0., .2], [.05, 0., .2]]),
        root_link_ang_vel_b=torch.tensor([[.4, .3, 0.], [.4, .3, .3]]),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda _: command))
    lin = mdp.track_hd1910_planar_velocity(env, std=.18)
    yaw = mdp.track_hd1910_yaw_velocity(env, std=.5)
    cost = mdp.hd1910_velocity_error_l1(env)
    assert lin[1] == 1. and lin[0] < lin[1]
    assert yaw[1] == 1. and yaw[0] < yaw[1]
    assert cost[1] == 0. and cost[0] > 0.
    assert torch.all(mdp.hd1910_vertical_velocity_l2(env) > 0.)


@pytest.mark.skipif(os.environ.get("MICRODUCK_TEST_GPU") != "1", reason="explicit GPU opt-in required")
@pytest.mark.parametrize("task", TASKS)
def test_warp_randomization_fields_partial_resets_and_rollout(task):
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 4
    cfg.auto_reset = False
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        obs, _ = env.reset(seed=17)
        assert obs["actor"].shape == (4, 61)
        robot, sim = env.scene["robot"], env.sim
        actuator = robot.actuators[0]
        assert actuator.kp_scale.shape == actuator.friction_scale.shape == (4, 14)
        assert torch.all(actuator.kp_scale >= .88) and torch.all(actuator.kp_scale <= 1.12)
        assert torch.std(actuator.kp_scale[0]) > .02
        assert torch.all(actuator.voltage_floor == 6.4)
        assert torch.all(actuator.vin_tensor >= 6.8) and torch.all(actuator.vin_tensor <= 8.)
        assert torch.all(actuator._delay_buffer.current_lags >= 2)
        assert torch.all(actuator._delay_buffer.current_lags <= 8)
        head = env.event_manager.get_term_cfg("robust_head_inertia").params["asset_cfg"]
        head_names = [robot.body_names[i] for i in head.body_ids]
        assert "bearing_roll" not in head_names and len(head_names) == 4

        # Fix samples to deterministic non-nominal values, then perform repeated
        # selected-world resets. This catches accumulation, shared model writes
        # and DR that accidentally touches worlds that were not reset.
        trunk_term = env.event_manager.get_term_cfg("robust_trunk_inertia")
        trunk_term.params.update(alpha_range=(math.log(1.05) / 2.,) * 2,
                                 t1_range=(.004, .004), t2_range=(0., 0.), t3_range=(0., 0.))
        response = env.event_manager.get_term_cfg("robust_actuator_response")
        response.params.update(kp_range=(1.1, 1.1), friction_range=(1.2, 1.2),
                               effort_range=(.9, .9), voltage_range=(7., 7.),
                               voltage_drop_range=(.2, .2), voltage_floor=6.7, lag_range=(8, 8))
        mu = env.event_manager.get_term_cfg("foot_friction")
        mu.params["ranges"] = (.5, .5)
        if "Backlash" in task:
            env.event_manager.get_term_cfg("robust_backlash").params["ranges"] = (1.25, 1.25)
        selected = torch.tensor([0], device=env.device)
        trunk_id = robot.indexing.body_ids[trunk_term.params["asset_cfg"].body_ids][0]
        before_mass = sim.model.body_mass[1:].clone()
        before_gains = actuator.kp_scale[1:].clone()
        default_mass = sim.get_default_field("body_mass")[trunk_id]
        default_pos = sim.get_default_field("body_ipos")[trunk_id]
        default_force = sim.get_default_field("actuator_forcerange")[actuator.global_ctrl_ids]
        for _ in range(3):
            obs, _ = env.reset(env_ids=selected)
            torch.testing.assert_close(sim.model.body_mass[0, trunk_id], default_mass * 1.05)
            torch.testing.assert_close(sim.model.body_ipos[0, trunk_id], default_pos + torch.tensor([.004, 0., 0.], device=env.device))
            torch.testing.assert_close(sim.model.body_mass[1:], before_mass)
            torch.testing.assert_close(actuator.kp_scale[0], torch.full((14,), 1.1, device=env.device))
            torch.testing.assert_close(actuator.kp_scale[1:], before_gains)
            torch.testing.assert_close(sim.model.actuator_forcerange[0, actuator.global_ctrl_ids], default_force * .9)
            assert torch.all(actuator.friction_scale[0] == 1.2)
            assert actuator._delay_buffer.current_lags[0] == 8
            assert actuator.voltage_floor[0].item() == pytest.approx(6.7)
            assert torch.all(actuator.voltage_floor[1:] == 6.4)
            raw_velocity = encoder_obs.joint_vel_finite_difference(
                env, **env.observation_manager.get_term_cfg("actor", "joint_vel").params,
            )
            assert torch.all(raw_velocity[0] == 0.)
        foot_ids = robot.indexing.geom_ids[mu.params["asset_cfg"].geom_ids]
        assert torch.all(sim.model.geom_friction[0, foot_ids, 0] == .5)
        if "Backlash" in task:
            term = env.event_manager.get_term_cfg("robust_backlash")
            ids = robot.indexing.joint_ids[term.params["asset_cfg"].joint_ids]
            torch.testing.assert_close(sim.model.jnt_range[0, ids], sim.get_default_field("jnt_range")[ids] * 1.25)
            assert torch.allclose(sim.model.jnt_range[0, ids, 0], -sim.model.jnt_range[0, ids, 1])
        for step in range(30):
            action = torch.zeros((4, 14), device=env.device)
            action[1, 0] = .05
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all() and torch.isfinite(robot.data.joint_pos).all()
        assert actuator._delay_buffer.current_lags[0] == 8  # manual lag remains held
        print(f"HD1910_ROBUST_DR_GATE {task}: 61/14, selected resets, finite rollout, effective DR")
    finally:
        env.close()
