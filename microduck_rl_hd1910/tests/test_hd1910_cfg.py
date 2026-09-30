"""CPU checks for the HD1910 model and the shared walking-policy contract."""

from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

from bam.model import load_model
import mujoco
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
import numpy as np
import pytest
import torch

from mjlab_microduck.actuator.feetech_bam import (
    FeetechBamActuatorCfg,
    FeetechBacklashBamActuatorCfg,
)
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_ROBOT_CFG,
)
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import (
    HD1910_M6_JSON,
    make_microduck_hd1910_velocity_env_cfg,
)


BASE = "Mjlab-Velocity-Flat-MicroDuck"
BASE_BACKLASH = "Mjlab-Velocity-Flat-Backlash-MicroDuck"
FLAT = f"{BASE}-HD1910"
BACKLASH = f"{BASE_BACKLASH}-HD1910"
TASKS = (FLAT, BACKLASH)
MODEL_SHA256 = "ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb"
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
ACTOR_TERMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "joint_pos": 14,
    "joint_vel": 14,
    "actions": 14,
    "command": 3,
    "head_command": 4,
    "body_command": 6,
}


@lru_cache(maxsize=None)
def _model(task):
    # Compile native MuJoCo models only; never create a Warp/GPU environment.
    return load_env_cfg(task).scene.entities["robot"].spec_fn().compile()


def _joint_names(model):
    return [model.joint(i).name for i in range(model.njnt)
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


def _selected(names, patterns):
    if isinstance(patterns, str):
        patterns = (patterns,)
    return [i for i, name in enumerate(names)
            if any(re.fullmatch(pattern, name) for pattern in patterns)]


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("play", [False, True])
def test_hd1910_registered_with_dedicated_feetech_actuator(task, play):
    assert task in list_tasks()
    cfg = load_env_cfg(task, play=play)
    actuators = cfg.scene.entities["robot"].articulation.actuators
    assert len(actuators) == 1
    actuator = actuators[0]
    expected_type = FeetechBacklashBamActuatorCfg if task == BACKLASH else FeetechBamActuatorCfg
    assert type(actuator) is expected_type
    assert Path(actuator.json_path).resolve() == HD1910_M6_JSON.resolve()
    assert actuator.kp_fw == 5.0
    assert actuator.vin_range == (7.4, 8.0)
    assert actuator.vin_drop_gain_range == (0.0, 0.2)
    assert actuator.vin_min == 7.0
    assert (actuator.delay_min_lag, actuator.delay_max_lag) == (3, 6)
    assert cfg.decimation * cfg.sim.mujoco.timestep == pytest.approx(0.02)
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"


def test_hd1910_model_matches_pinned_upstream_copy():
    assert hashlib.sha256(HD1910_M6_JSON.read_bytes()).hexdigest() == MODEL_SHA256
    model = load_model(str(HD1910_M6_JSON))
    assert type(model.actuator).__name__ == "STS3215Actuator"
    assert model.kt.value == pytest.approx(0.6237611235393989)
    assert model.R.value == pytest.approx(4.910191564179625)
    assert model.armature.value == pytest.approx(0.0017819049974475338)
    assert model.load_friction_motor_quad.value > 0
    assert model.load_friction_external_stribeck.value > 0


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("play", [False, True])
def test_hd1910_keeps_matching_walk_geometry_and_home_pose(task, play):
    cfg = load_env_cfg(task, play=play)
    base_task = task.removesuffix("-HD1910")
    base = load_env_cfg(base_task, play=play)
    robot = cfg.scene.entities["robot"]
    original = base.scene.entities["robot"]
    template = MICRODUCK_WALK_BACKLASH_ROBOT_CFG if task == BACKLASH else MICRODUCK_WALK_ROBOT_CFG
    assert robot.spec_fn is template.spec_fn
    assert robot.init_state == original.init_state == template.init_state
    assert robot.collisions == original.collisions
    assert cfg.actions == base.actions

    model, base_model = _model(task), _model(base_task)
    assert [name for name in _joint_names(model) if not name.startswith("passive_")] == JOINTS
    assert model.nu == 14
    for field in ("body_mass", "body_inertia", "geom_type", "geom_size",
                  "geom_contype", "geom_conaffinity", "geom_bodyid"):
        np.testing.assert_array_equal(getattr(model, field), getattr(base_model, field))
    floor_collisions = [model.geom(i).name for i in range(model.ngeom)
                        if model.geom_contype[i] & 1 or model.geom_conaffinity[i] & 1]
    assert set(floor_collisions) == {"left_foot_collision", "right_foot_collision"}


@pytest.mark.parametrize("task", TASKS)
def test_joint_selectors_resolve_only_active_servos(task):
    cfg = load_env_cfg(task)
    names = _joint_names(_model(task))
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert [names[i] for i in _selected(names, actuator.target_names_expr)] == JOINTS
    for group in ("actor", "critic"):
        for name in ("joint_pos", "joint_vel"):
            asset_cfg = cfg.observations[group].terms[name].params["asset_cfg"]
            assert [names[i] for i in _selected(names, asset_cfg.joint_names)] == JOINTS
    pose = cfg.rewards["pose"].params["asset_cfg"]
    pose_names = [names[i] for i in _selected(names, pose.joint_names)]
    assert pose_names == JOINTS[:5] + JOINTS[9:]
    if task == BACKLASH:
        limits = cfg.rewards["dof_pos_limits"].params["asset_cfg"]
        assert [names[i] for i in _selected(names, limits.joint_names)] == JOINTS


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("play", [False, True])
def test_actor_observations_execute_as_61d_on_cpu(task, play):
    cfg = load_env_cfg(task, play=play)
    terms = cfg.observations["actor"].terms
    assert list(terms) == list(ACTOR_TERMS)
    names = _joint_names(_model(task))
    shape = (2, len(names))
    robot = SimpleNamespace(joint_names=names, data=SimpleNamespace(
        joint_pos=torch.zeros(shape), joint_pos_biased=torch.zeros(shape),
        joint_vel=torch.zeros(shape), default_joint_pos=torch.zeros(shape),
        default_joint_vel=torch.zeros(shape), root_link_ang_vel_b=torch.zeros(2, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1),
    ))
    commands = {
        "twist": torch.zeros(2, 3),
        "head_pose": torch.zeros(2, len(cfg.commands["head_pose"].ranges)),
        "body_pose": torch.zeros(2, len(cfg.commands["body_pose"].ranges)),
    }
    env = SimpleNamespace(
        scene={"robot": robot}, num_envs=2, device="cpu",
        action_manager=SimpleNamespace(action=torch.zeros(2, _model(task).nu)),
        command_manager=SimpleNamespace(get_command=commands.__getitem__),
    )
    observations = []
    for name, term in terms.items():
        params = deepcopy(term.params)
        if "asset_cfg" in params:
            asset_cfg = params["asset_cfg"]
            if asset_cfg.joint_names is not None:
                asset_cfg.joint_ids = _selected(names, asset_cfg.joint_names)
        value = term.func(env, **params)
        assert value.shape == (2, ACTOR_TERMS[name]), name
        assert torch.isfinite(value).all(), name
        observations.append(value)
    assert torch.cat(observations, dim=-1).shape == (2, 61)


def test_backlash_has_fourteen_zeroed_one_degree_hinges_and_encoder_observations():
    cfg = load_env_cfg(BACKLASH)
    model = _model(BACKLASH)
    passive_ids = [i for i in range(model.njnt)
                   if model.joint(i).name.endswith("_backlash")]
    assert [model.joint(i).name for i in passive_ids] == [f"passive_{name}_backlash" for name in JOINTS]
    assert len(passive_ids) == 14
    np.testing.assert_allclose(model.jnt_range[passive_ids],
                               np.tile(np.deg2rad([-1.0, 1.0]), (14, 1)), atol=1e-6)
    home = cfg.scene.entities["robot"].init_state.joint_pos
    for index in passive_ids:
        # InitialStateCfg uses the first matching pattern.
        initial = next(value for pattern, value in home.items()
                       if re.fullmatch(pattern, model.joint(index).name))
        assert initial == 0.0
    for group in ("actor", "critic"):
        terms = cfg.observations[group].terms
        assert terms["joint_pos"].func is microduck_mdp.joint_pos_rel_backlash
        assert terms["joint_vel"].func is microduck_mdp.joint_vel_rel_backlash
    assert cfg.observations["actor"].terms["joint_pos"].params["biased"] is True
    assert cfg.observations["critic"].terms["joint_pos"].params["biased"] is False


@pytest.mark.parametrize("backlash", [False, True])
def test_hd1910_factory_isolates_robot_and_observation_mutations(backlash):
    first = make_microduck_hd1910_velocity_env_cfg(backlash=backlash)
    robot = first.scene.entities["robot"]
    robot.articulation.actuators[0].kp_fw = 99.0
    robot.init_state.joint_pos[r".*left_hip_pitch.*"] = 1.0
    first.observations["actor"].terms["joint_pos"].params["biased"] = False
    second = make_microduck_hd1910_velocity_env_cfg(backlash=backlash)
    second_robot = second.scene.entities["robot"]
    template = MICRODUCK_WALK_BACKLASH_ROBOT_CFG if backlash else MICRODUCK_WALK_ROBOT_CFG
    assert second_robot is not robot and second_robot is not template
    assert second_robot.init_state == template.init_state
    assert second_robot.articulation.actuators[0].kp_fw == 5.0
    assert second.observations["actor"].terms["joint_pos"].params["biased"] is True
    for base_task in (BASE, BASE_BACKLASH):
        base_robot = load_env_cfg(base_task).scene.entities["robot"]
        base_actuator = base_robot.articulation.actuators[0]
        assert base_actuator.motor_name == "xl330"
        assert base_actuator.kp_fw == 200.0
        assert base_robot.init_state.joint_pos[r".*left_hip_pitch.*"] == -0.4579


def test_hd1910_training_runs_have_distinct_names_and_normalized_observations():
    runners = [load_rl_cfg(task) for task in (BASE, BASE_BACKLASH, FLAT, BACKLASH)]
    hd_names = {runner.experiment_name for runner in runners[2:]}
    assert len(hd_names) == 2
    assert hd_names.isdisjoint(runner.experiment_name for runner in runners[:2])
    for runner in runners[2:]:
        assert runner.actor.obs_normalization
