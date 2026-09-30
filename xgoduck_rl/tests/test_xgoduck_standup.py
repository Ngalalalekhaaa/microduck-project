"""XGODUCK standup = MicroDuck rewards/collisions + drop-from-air + 2s freeze."""

import math

import pytest
from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401 — register task ids
from mjlab_microduck.robot.xgoduck_constants import XGODUCK_SIT_Z, XGODUCK_STAND_Z
from mjlab_microduck.tasks.microduck_standup_env_cfg import (
    EPISODE_LENGTH_S,
    SIT_Z,
    STAND_Z,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_microduck_standup_is_unchanged():
    cfg = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck")
    assert cfg.episode_length_s == EPISODE_LENGTH_S == 6.0
    assert cfg.events["set_ground_state"].func is microduck_mdp.set_random_ground_state
    assert "ground_state_mix" in cfg.curriculum
    assert "head_pose_tracking" in cfg.rewards
    assert "head_home_when_upright" not in cfg.rewards
    assert cfg.rewards["pose_stand_legs"].weight == pytest.approx(2.0)
    assert cfg.rewards["upright_linear"].weight == pytest.approx(1.5)


@pytest.mark.parametrize("task_id,play", [
    ("Mjlab-StandUp-Flat-XgoDuck", False),
    ("Mjlab-StandUp-Flat-XgoDuck", True),
    ("Mjlab-StandUp-Rough-XgoDuck", False),
    ("Mjlab-StandUp-Rough-XgoDuck", True),
])
def test_xgoduck_standup_drops_and_freezes(task_id, play):
    cfg = load_env_cfg(task_id, play=play)
    event = cfg.events["set_ground_state"]
    assert event.func is microduck_mdp.set_random_drop_state
    drop_z = XGODUCK_STAND_Z + 0.10
    assert event.params["drop_z_min"] == pytest.approx(drop_z - 0.005)
    assert event.params["drop_z_max"] == pytest.approx(drop_z + 0.005)
    assert event.params["euler_abs_max"] == pytest.approx(math.pi)
    assert "ground_state_mix" not in cfg.curriculum

    action = cfg.actions["joint_pos"]
    assert isinstance(action, microduck_mdp.FreezeThenJointPositionActionCfg)
    assert action.freeze_s == pytest.approx(2.0)
    assert action.scale == 1.0
    if "push_robot" in cfg.events:
        assert cfg.events["push_robot"].func is microduck_mdp.push_by_setting_velocity_after_delay
        assert cfg.events["push_robot"].params["delay_s"] == pytest.approx(2.0)


def test_xgoduck_standup_rewards_match_microduck():
    md = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck")
    xg = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    assert set(xg.rewards.keys()) == set(md.rewards.keys())
    for name, term in md.rewards.items():
        assert xg.rewards[name].func is term.func
        assert xg.rewards[name].weight == pytest.approx(term.weight)
    assert xg.rewards["pose_stand_legs"].weight == pytest.approx(2.0)
    assert xg.rewards["pose_stand_l1"].weight == pytest.approx(1.25)
    assert xg.rewards["upright_linear"].weight == pytest.approx(1.5)
    assert xg.rewards["upright_sharp"].weight == pytest.approx(1.5)
    assert "head_home_when_upright" not in xg.rewards
    assert xg.commands["head_pose"].ranges == md.commands["head_pose"].ranges
    assert xg.events["push_robot"].params["velocity_range"]["x"] == pytest.approx((-0.3, 0.3))
    assert xg.curriculum["push_magnitude"].params["push_stages"] == (
        md.curriculum["push_magnitude"].params["push_stages"]
    )
    assert xg.curriculum["upright_sharp_weight"].params["weight_stages"] == (
        md.curriculum["upright_sharp_weight"].params["weight_stages"]
    )


def test_xgoduck_standup_collision_names_match_microduck():
    md = load_env_cfg("Mjlab-StandUp-Flat-MicroDuck")
    xg = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    md_feet = next(s for s in md.scene.sensors if s.name == "feet_ground_contact")
    xg_feet = next(s for s in xg.scene.sensors if s.name == "feet_ground_contact")
    assert xg_feet.primary.pattern == md_feet.primary.pattern
    assert xg_feet.primary.pattern == r"^(left_foot_collision|right_foot_collision)$"
    assert (
        xg.events["foot_friction"].params["asset_cfg"].geom_names
        == md.events["foot_friction"].params["asset_cfg"].geom_names
        == ("left_foot_collision", "right_foot_collision")
    )
    assert xg.rewards["self_collisions"].params == md.rewards["self_collisions"].params


def test_xgoduck_standup_uses_xgoduck_heights():
    assert XGODUCK_SIT_Z != SIT_Z
    assert XGODUCK_STAND_Z != STAND_Z
    dz = XGODUCK_STAND_Z - STAND_Z
    cfg = load_env_cfg("Mjlab-StandUp-Flat-XgoDuck")
    r = cfg.rewards
    for name in (
        "height_stand",
        "height_stand_sharp",
        "height_stand_l1",
        "standing_composite",
    ):
        assert r[name].params["target_height"] == pytest.approx(XGODUCK_STAND_Z)
    assert r["com_upward_velocity"].params["max_height"] == pytest.approx(XGODUCK_STAND_Z + 0.010)
    assert r["upright_sharp"].params["height_low"] == pytest.approx(XGODUCK_SIT_Z)
    assert r["upright_sharp"].params["height_high"] == pytest.approx(XGODUCK_STAND_Z)
    assert r["arrival_damping"].params["height_low"] == pytest.approx(0.09 + dz)
    assert r["arrival_damping"].params["height_high"] == pytest.approx(0.11 + dz)
    assert r["head_pose_bias"].params["gate_height_low"] == pytest.approx(0.09 + dz)
    assert r["head_pose_bias"].params["gate_height_high"] == pytest.approx(0.11 + dz)
    assert r["body_pose_tracking"].params["nominal_height"] == pytest.approx(XGODUCK_STAND_Z)


def test_set_random_drop_state_writes_height_and_unit_quats():
    import torch

    class _Data:
        qpos = torch.zeros(8, 21)
        qvel = torch.ones(8, 20)

    class _Sim:
        data = _Data()

    class _Env:
        device = torch.device("cpu")
        sim = _Sim()

    env = _Env()
    env_ids = torch.arange(8)
    microduck_mdp.set_random_drop_state(
        env,
        env_ids,
        drop_z_min=0.23,
        drop_z_max=0.25,
        euler_abs_max=math.pi,
    )
    z = env.sim.data.qpos[:, 2]
    assert torch.all(z >= 0.23) and torch.all(z <= 0.25)
    quat = env.sim.data.qpos[:, 3:7]
    norms = torch.linalg.norm(quat, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5)
    assert torch.allclose(env.sim.data.qvel[:, :6], torch.zeros(8, 6))
    assert quat.std(dim=0).sum() > 0.1


def test_freeze_then_joint_position_zeros_frozen_envs():
    import torch

    class _Mgr:
        _action = torch.ones(4, 14)

    class _Env:
        step_dt = 0.02
        episode_length_buf = torch.tensor([0, 50, 99, 100])
        action_manager = _Mgr()

    term = microduck_mdp.FreezeThenJointPositionAction.__new__(
        microduck_mdp.FreezeThenJointPositionAction
    )
    term._env = _Env()
    term.cfg = type("C", (), {"freeze_s": 2.0})()
    term._raw_actions = torch.ones(4, 14)
    term._processed_actions = torch.ones(4, 14)
    term._scale = 1.0
    term._offset = torch.full((4, 14), 0.5)
    term.cfg.clip = None

    actions = torch.ones(4, 14)
    microduck_mdp.FreezeThenJointPositionAction.process_actions(term, actions)

    freeze = torch.tensor([True, True, True, False])
    assert torch.equal(term._raw_actions[freeze], torch.zeros(3, 14))
    assert torch.allclose(term._processed_actions[freeze], torch.full((3, 14), 0.5))
    assert torch.equal(term._env.action_manager._action[freeze], torch.zeros(3, 14))
    assert torch.equal(term._raw_actions[3], torch.ones(14))
    assert torch.allclose(term._processed_actions[3], torch.full((14,), 1.5))


def test_xgoduck_standup_trunk_com_matches_walk():
    import mujoco
    from mjlab_microduck.robot.xgoduck_constants import (
        XGODUCK_ALLCOLLISIONS_XML,
        XGODUCK_WALK_XML,
    )

    walk = mujoco.MjModel.from_xml_path(str(XGODUCK_WALK_XML))
    allc = mujoco.MjModel.from_xml_path(str(XGODUCK_ALLCOLLISIONS_XML))
    w = mujoco.mj_name2id(walk, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    a = mujoco.mj_name2id(allc, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    assert allc.body_ipos[a, 0] == pytest.approx(walk.body_ipos[w, 0], abs=1e-9)
    assert allc.body_mass[a] == pytest.approx(walk.body_mass[w], abs=1e-9)
