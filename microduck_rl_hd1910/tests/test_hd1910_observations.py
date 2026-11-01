"""Check the encoder/time conventions that matter for transfer to Radxa."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.hd1910_observations import (
    joint_pos_rel_quantized,
    joint_vel_finite_difference,
    reset_hd1910_encoder_history,
)


TICK = math.tau / 4096


def _env(backlash=False):
    names = ["left_ankle", "right_ankle"]
    ids = [0, 1]
    if backlash:
        names = ["left_ankle", "passive_left_ankle_backlash",
                 "right_ankle", "passive_right_ankle_backlash"]
        ids = [0, 2]
    position = torch.zeros(3, len(names), dtype=torch.float64)
    default = torch.zeros_like(position)
    data = SimpleNamespace(
        joint_pos=position,
        joint_pos_biased=position.clone(),
        default_joint_pos=default,
    )
    env = SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data, joint_names=names)},
        common_step_counter=0,
        step_dt=0.02,
        episode_length_buf=torch.zeros(3, dtype=torch.long),
    )
    return env, SceneEntityCfg("robot", joint_ids=ids), data


def _advance(env, steps=1):
    env.common_step_counter += steps
    env.episode_length_buf += steps


@pytest.mark.parametrize("backlash", [False, True])
def test_quantizes_encoder_then_subtracts_exact_home(backlash):
    env, cfg, data = _env(backlash)
    ids = cfg.joint_ids
    data.default_joint_pos[:, ids] = 0.35
    data.joint_pos[:, ids] = torch.tensor([1.3 * TICK, -2.2 * TICK], dtype=torch.float64)
    data.joint_pos_biased.copy_(data.joint_pos)
    # Bias is applied once to the main joint, including on backlash models.
    data.joint_pos_biased[:, ids] += 2 * TICK
    encoder = data.joint_pos_biased[:, ids].clone()
    if backlash:
        data.joint_pos[:, [1, 3]] = torch.tensor([0.8 * TICK, -0.6 * TICK], dtype=torch.float64)
        encoder += data.joint_pos[:, [1, 3]]
    expected = torch.round(encoder / TICK) * TICK - 0.35
    torch.testing.assert_close(joint_pos_rel_quantized(env, True, cfg), expected)
    assert joint_pos_rel_quantized(env, True, cfg).shape == (3, 2)


@pytest.mark.parametrize("backlash", [False, True])
def test_velocity_matches_runtime_window_and_backlash_encoder(backlash):
    env, cfg, data = _env(backlash)
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg),
                               torch.zeros(3, 2, dtype=torch.float64))
    _advance(env)
    data.joint_pos[:, cfg.joint_ids] += 2 * TICK
    if backlash:
        data.joint_pos[:, [1, 3]] += TICK
    expected = (3 if backlash else 2) * TICK / 0.02
    result = joint_vel_finite_difference(env, asset_cfg=cfg)
    torch.testing.assert_close(result, torch.full((3, 2), expected, dtype=torch.float64))
    # Scaling/noise on a returned tensor must not alter the history cache.
    result.fill_(1000)
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg),
                               torch.full((3, 2), expected, dtype=torch.float64))
    _advance(env)
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg),
                               torch.zeros(3, 2, dtype=torch.float64))


@pytest.mark.parametrize("selection", [torch.tensor([1]), slice(1, 2), None])
def test_reset_reseeds_only_selected_worlds_without_velocity_spike(selection):
    env, cfg, data = _env()
    joint_vel_finite_difference(env, asset_cfg=cfg)
    _advance(env)
    data.joint_pos += TICK
    joint_vel_finite_difference(env, asset_cfg=cfg)
    reset_hd1910_encoder_history(env, selection)
    selected = slice(None) if selection is None else selection
    # A manual reset can happen without advancing the global control counter.
    data.joint_pos[selected] += 1.0
    env.episode_length_buf[selected] = 0
    expected = torch.full((3, 2), TICK / 0.02, dtype=torch.float64)
    expected[selected] = 0
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg), expected)
    _advance(env)
    data.joint_pos += TICK
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg),
                               torch.full((3, 2), TICK / 0.02, dtype=torch.float64))


def test_automatic_partial_reset_and_skipped_steps_use_correct_time():
    env, cfg, data = _env()
    joint_vel_finite_difference(env, asset_cfg=cfg)
    _advance(env)
    data.joint_pos += TICK
    joint_vel_finite_difference(env, asset_cfg=cfg)
    _advance(env, 2)
    data.joint_pos += 4 * TICK
    data.joint_pos[1] += 1
    env.episode_length_buf[1] = 0
    expected = torch.full((3, 2), 4 * TICK / 0.04, dtype=torch.float64)
    expected[1] = 0
    torch.testing.assert_close(joint_vel_finite_difference(env, asset_cfg=cfg), expected)


def test_changed_bias_on_reset_is_not_interpreted_as_joint_motion():
    env, cfg, data = _env()
    joint_vel_finite_difference(env, biased=True, asset_cfg=cfg)
    _advance(env)
    reset_hd1910_encoder_history(env, torch.tensor([1]))
    data.joint_pos_biased[1] += 0.05
    env.episode_length_buf[1] = 0
    torch.testing.assert_close(joint_vel_finite_difference(env, True, cfg),
                               torch.zeros(3, 2, dtype=torch.float64))


def test_rejects_invalid_encoder_or_control_resolution():
    env, cfg, _ = _env()
    with pytest.raises(ValueError, match="encoder_ticks"):
        joint_pos_rel_quantized(env, asset_cfg=cfg, encoder_ticks=1)
    env.step_dt = 0
    with pytest.raises(ValueError, match="step_dt"):
        joint_vel_finite_difference(env, asset_cfg=cfg)
