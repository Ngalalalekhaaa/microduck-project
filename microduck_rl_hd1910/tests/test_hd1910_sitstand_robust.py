"""Deployment parity, physical DR and whole-motion tilt regression checks."""

from copy import deepcopy
from dataclasses import asdict
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_hd1910_sitstand_robust_env_cfg import (
    TASK_ID, ROBUST_STAGES, make_microduck_hd1910_sitstand_robust_env_cfg,
)
from mjlab_microduck.tasks.sitstand_warm_start import restore_onnx_actor


def test_shared_contract_and_source_factory_untouched():
    cfg = load_env_cfg(TASK_ID)
    assert list(cfg.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
        "actions", "command", "head_command", "body_command",
    ]
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(.02)
    assert cfg.actions["joint_pos"].scale == 1.
    assert cfg.actions["joint_pos"].clip is None
    assert cfg.events["encoder_bias"].mode == "reset"
    assert cfg.observations["actor"].terms["joint_pos"].params["biased"]
    assert not cfg.observations["critic"].terms["joint_pos"].params["biased"]
    assert cfg.rewards["whole_motion_tilt"].weight < 0
    assert cfg.terminations["tilt_failure"].params["limit_angle"] == pytest.approx(math.radians(35))
    assert "base_com" not in cfg.events and "randomize_head_com" not in cfg.events
    assert cfg.scene.entities["robot"].articulation.actuators[0].kp_fw == 5.
    source = load_env_cfg("Mjlab-SitStand-Flat-MicroDuck")
    assert "whole_motion_tilt" not in source.rewards
    assert source.scene.entities["robot"].articulation.actuators[0].json_path is None


@pytest.mark.parametrize("degrees", [0., 14.9, 15., 15.1, 20., 25., 90., 180.])
def test_tilt_penalty_sign_and_threshold_in_both_commands(degrees):
    a = math.radians(degrees)
    env = SimpleNamespace(extras={}, scene={"robot": SimpleNamespace(data=SimpleNamespace(
        projected_gravity_b=torch.tensor([[math.sin(a), 0., -math.cos(a)]]),
    ))})
    cost = mdp.hd1910_sitstand_tilt_cost(env).item()
    assert cost >= 0
    assert (cost > 1.e-6) == (degrees > 15.)
    assert -8. * cost <= 0
    # No command/contact/height data exists: penalty cannot accidentally gate on it.


def test_inertia_is_coupled_nonaccumulating_and_does_not_mistake_hip_for_head():
    defaults = {"body_mass": torch.tensor([2., 3., 4.]),
                "body_inertia": torch.ones(3, 3), "body_ipos": torch.zeros(3, 3)}
    model = SimpleNamespace(**{k: v.repeat((2,) + (1,) * v.ndim) for k, v in defaults.items()})
    asset = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.arange(3)),
                            body_names=["trunk_base", "jaw_soft", "bearing_roll"])
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": asset},
                          sim=SimpleNamespace(model=model, get_default_field=defaults.__getitem__))
    params = {"mass_range": (1.2, 1.2), "head_mass_range": (1.3, 1.3),
              "trunk_com_range": (.025, .025, .012), "head_com_range": (.015, .015, .01)}
    for _ in range(3):
        mdp.randomize_hd1910_sitstand_inertia(env, torch.tensor([0]), **params)
        assert torch.allclose(model.body_mass[0], torch.tensor([2.4, 4.68, 4.8]))
        assert torch.allclose(model.body_inertia[0, :, 0], torch.tensor([1.2, 1.56, 1.2]))
        assert torch.equal(model.body_mass[1], defaults["body_mass"])
        assert (model.body_ipos[0, 0].abs() <= torch.tensor([.025, .025, .012])).all()
    params.update({"mass_range": (1., 1.), "head_mass_range": (1., 1.),
                   "trunk_com_range": (0., 0., 0.), "head_com_range": (0., 0., 0.)})
    mdp.randomize_hd1910_sitstand_inertia(env, torch.tensor([0]), **params)
    for k, default in defaults.items():
        assert torch.equal(getattr(model, k)[0], default)


def test_calibration_bias_changes_the_physical_goal_with_opposite_sign():
    sent = []
    asset = SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.tensor([[.1, -.2]])),
                            set_joint_position_target=lambda target, **kw: sent.append(target.clone()))
    action = SimpleNamespace(_entity=asset, _target_ids=torch.tensor([0, 1]),
                             _processed_actions=torch.tensor([[.4, -.5]]))
    JointPositionAction.apply_actions(action)
    assert torch.allclose(sent[0], torch.tensor([[.3, -.3]]))


def test_stage_envelopes_and_frozen_profiles():
    assert [s["step"] for s in ROBUST_STAGES] == [i * 24 for i in (0, 500, 1500, 2500)]
    nominal = make_microduck_hd1910_sitstand_robust_env_cfg(evaluation_profile="nominal")
    stress = make_microduck_hd1910_sitstand_robust_env_cfg(evaluation_profile="stress")
    assert not nominal.curriculum and not stress.curriculum
    assert stress.events["sitstand_response"].params["ankle_kp_range"] == (.8, .95)
    assert stress.events["encoder_bias"].params["bias_range"][1] == pytest.approx(math.radians(5))
    assert nominal.events["encoder_bias"].params["bias_range"] == (0., 0.)
    for stage in ROBUST_STAGES:
        assert stage["reward_weights"]["whole_motion_tilt"] < 0
        assert stage["events"]["sitstand_response"]["ankle_kp_range"][1] < 1.


def test_original_onnx_actor_and_normalizer_are_preserved():
    cfg = asdict(load_rl_cfg(TASK_ID).actor)
    valid = inspect.signature(MLPModel.__init__).parameters
    cfg = {k: deepcopy(v) for k, v in cfg.items() if k in valid}
    actor = MLPModel(TensorDict({"actor": torch.zeros(1, 61)}, batch_size=[1]),
                     {"actor": ["actor"]}, "actor", 14, **cfg)
    source = Path(__file__).resolve().parents[2] / "deploy_hd1910/models/hd1910_sitstand_1699.onnx"
    report = restore_onnx_actor(actor, source)
    assert report["passed"] and report["max_abs_error"] < 1.e-4
    assert not report["critic_restored"] and not report["optimizer_restored"]
