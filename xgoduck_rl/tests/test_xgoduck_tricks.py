"""Robot-specific geometry and task contracts for XgoDuck's dynamic tricks."""
import math
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.robot.xgoduck_constants import XGODUCK_STAND_Z
from mjlab_microduck.tasks import mdp

TASKS = ["Mjlab-BallKick-Flat-XgoDuck", "Mjlab-BallKick-Left-Flat-XgoDuck",
         "Mjlab-Roulade-Flat-XgoDuck"]


def home_model(cfg):
    robot = cfg.scene.entities["robot"]
    spec = robot.spec_fn()
    for col in robot.collisions:
        col.edit_spec(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = robot.init_state.pos
    data.qpos[3:7] = [1, 0, 0, 0]
    for j in range(1, model.njnt):
        name = model.joint(j).name
        for pattern, value in robot.init_state.joint_pos.items():
            if re.fullmatch(pattern, name):
                data.qpos[model.jnt_qposadr[j]] = value
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("play", [False, True])
def test_robot_contract(task, play):
    cfg = load_env_cfg(task, play=play)
    m, d = home_model(cfg)
    assert m.nu == 14
    assert [m.joint(j).name for j in range(1, m.njnt)] == [
        "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
        "neck_pitch", "head_pitch", "head_yaw", "head_roll",
        "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    ]
    assert d.qpos[2] == pytest.approx(XGODUCK_STAND_Z)
    assert "1910_m6.json" in cfg.scene.entities["robot"].articulation.actuators[0].json_path
    assert cfg.terminations["nan_state"].params["sensor_names"] == ("feet_ground_contact",)
    rl = load_rl_cfg(task)
    assert rl.logger == "tensorboard" and rl.actor.obs_normalization
    assert rl.algorithm.symmetry_cfg is None


@pytest.mark.parametrize("task,side", [(TASKS[0], "right"), (TASKS[1], "left")])
def test_ball_spawn_geometry_and_support(task, side):
    cfg = load_env_cfg(task)
    assert list(cfg.scene.entities) == ["robot", "ball"]
    assert list(cfg.events).index("reset_ball") > list(cfg.events).index("set_ground_state")
    m, d = home_model(cfg)
    gid = m.geom(f"{side}_foot_collision").id
    mesh = m.geom_dataid[gid]
    start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
    verts = m.mesh_vert[start:start + count] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    spawn = cfg.events["reset_ball"].params
    rear = spawn["offset"][0] - spawn["noise_xy"] - spawn["ball_radius"]
    assert rear - verts[:, 0].max() > 0.01
    assert abs(spawn["offset"][1] - d.geom_xpos[gid, 1]) < 0.002
    support = next(s for s in cfg.scene.sensors if s.name == "support_foot_ground_contact")
    assert support.primary.pattern == ("^left_foot_collision$" if side == "right" else "^right_foot_collision$")
    assert cfg.rewards["height_stand"].params["target_height"] == XGODUCK_STAND_Z
    assert "ball_position" not in cfg.observations["actor"].terms
    assert "ball_position" in cfg.observations["critic"].terms
    assert cfg.rewards["ball_speed_overshoot"].weight < 0


def test_roulade_head_axis_and_joint_limits():
    cfg = load_env_cfg(TASKS[2])
    m, d = home_model(cfg)
    spawn = cfg.events["set_roulade_state"].params
    axis = np.array(spawn["head_top_axis"])
    bid = m.body("jaw_soft").id
    assert (d.xmat[bid].reshape(3, 3) @ axis)[2] > 0.99
    for index, value in spawn["tuck_overrides"].items():
        assert m.jnt_range[index + 1, 0] < value < m.jnt_range[index + 1, 1]
        d.qpos[m.jnt_qposadr[index + 1]] = value
    pitch = math.radians(110)
    d.qpos[3:7] = [math.cos(pitch / 2), 0, math.sin(pitch / 2), 0]
    mujoco.mj_forward(m, d)
    env = SimpleNamespace(_roulade_head_body_id=bid, _roulade_head_top_axis=tuple(axis))
    asset = SimpleNamespace(data=SimpleNamespace(body_link_quat_w=torch.tensor(d.xquat[None])))
    assert mdp._head_top_down(env, asset).item()
    assert "fell_over" not in cfg.terminations
    assert cfg.rewards["gentle_landing"].weight > 0  # self-negating function
    for key in ("roulade_landing_composite", "roulade_height_after_roll", "roulade_landing_sharp", "roulade_stand_tax"):
        assert cfg.rewards[key].params["target_height"] == XGODUCK_STAND_Z
    play = load_env_cfg(TASKS[2], play=True)
    assert play.events["set_roulade_state"].params["midroll_prob"] == 0
    assert "roulade_spawn_mix" not in play.curriculum


def test_microduck_defaults_stay_separate():
    cfg = load_env_cfg("Mjlab-BallKick-Flat-MicroDuck")
    assert cfg.events["reset_ball"].params["offset"] == (0.09, -0.042)
    assert cfg.rewards["height_stand"].params["target_height"] == 0.115
    cfg = load_env_cfg("Mjlab-Roulade-Flat-MicroDuck")
    assert "head_top_axis" not in cfg.events["set_roulade_state"].params
    assert mdp._HEAD_TOP_AXIS == (0.882, 0.0, 0.471)
