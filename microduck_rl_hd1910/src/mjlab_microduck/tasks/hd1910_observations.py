"""HD-1910 encoder observations matching the position-difference runtime.

The Radxa runtime reads 4096-count position feedback and computes velocity
from two successive control-loop samples. It does not use the servo's
present-velocity register. Quantization here precedes differencing, so a tick
at 50 Hz contributes approximately 0.077 rad/s, as on the robot.

The encoder origin's fractional tick phase is not measured. Quantizing the
absolute model angle assumes zero fractional phase (at most half a tick of
position discrepancy); calibration-bias randomization remains a separate
environment event. These functions add no white noise or action filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .mdp import _backlash_encoder_ids


_DEFAULT_ENCODERS = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))


def joint_pos_rel_quantized(
    env,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ENCODERS,
    encoder_ticks: int = 4096,
) -> torch.Tensor:
    """Quantized output-side encoder angle minus the exact training HOME.

    Main and passive backlash positions are summed BEFORE quantization. The
    same function also handles plain models. Encoder-bias DR is included only
    when ``biased=True``; the passive backlash angle is never biased twice.
    ``asset_cfg`` must have the servo-only selection resolved by the manager.
    """
    if type(encoder_ticks) is not int or encoder_ticks < 2:
        raise ValueError("encoder_ticks must be an integer of at least 2")
    asset = env.scene[asset_cfg.name]
    main_ids, backlash_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    positions = asset.data.joint_pos_biased if biased else asset.data.joint_pos
    encoder = positions[:, main_ids] + asset.data.joint_pos[:, backlash_ids] * mask
    resolution = math.tau / encoder_ticks
    quantized = torch.round(encoder / resolution) * resolution
    return quantized - asset.data.default_joint_pos[:, main_ids]


@dataclass
class _EncoderHistory:
    position: torch.Tensor
    velocity: torch.Tensor
    step: torch.Tensor
    episode_step: torch.Tensor
    valid: torch.Tensor


def joint_vel_finite_difference(
    env,
    biased: bool = False,
    asset_cfg: SceneEntityCfg = _DEFAULT_ENCODERS,
    encoder_ticks: int = 4096,
) -> torch.Tensor:
    """Difference quantized positions over control steps, in radians/second.

    Uses ``env.step_dt`` (20 ms in the walking tasks), never the 5 ms physics
    substep. Repeated observation computations within a control step return
    the same cached velocity. Each reset starts with a zero velocity sample,
    while other worlds retain their histories. Missing observation steps use
    their full elapsed time instead of pretending to be consecutive samples.
    """
    dt = float(env.step_dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("control step_dt must be positive and finite")
    position = joint_pos_rel_quantized(env, biased, asset_cfg, encoder_ticks)
    key = (asset_cfg.name, str(asset_cfg.joint_ids), biased, encoder_ticks)
    histories = env.__dict__.setdefault("_hd1910_encoder_histories", {})
    step = int(env.common_step_counter)
    episode_step = env.episode_length_buf
    history = histories.get(key)
    if history is None:
        history = _EncoderHistory(
            position=position.clone(),
            velocity=torch.zeros_like(position),
            step=torch.full_like(episode_step, step),
            episode_step=episode_step.clone(),
            valid=torch.ones_like(episode_step, dtype=torch.bool),
        )
        histories[key] = history
        return history.velocity.clone()

    delta_steps = step - history.step
    fresh = (delta_steps != 0) | ~history.valid
    # Normal automatic resets are also recognized from the episode counter.
    # The explicit reset event handles manual resets at an unchanged global
    # counter, including a second reset of a world already at episode step 0.
    reset = ((episode_step < history.episode_step)
             | ((episode_step == 0) & (delta_steps > 0)))
    continuous = history.valid & (delta_steps > 0) & ~reset
    elapsed = delta_steps.clamp(min=1).to(position.dtype).unsqueeze(-1) * dt
    measured_velocity = (position - history.position) / elapsed
    measured_velocity = torch.where(
        continuous.unsqueeze(-1), measured_velocity, torch.zeros_like(position)
    )
    history.velocity = torch.where(fresh.unsqueeze(-1), measured_velocity, history.velocity)
    history.position = torch.where(fresh.unsqueeze(-1), position, history.position)
    history.step = torch.where(fresh, torch.full_like(history.step, step), history.step)
    history.episode_step = torch.where(fresh, episode_step, history.episode_step)
    history.valid.fill_(True)
    # Observation-manager scaling/noise must not mutate our cached signal.
    return history.velocity.clone()


def reset_hd1910_encoder_history(env, env_ids) -> None:
    """Invalidate selected histories before reset poses become observable."""
    selection = slice(None) if env_ids is None else env_ids
    for history in env.__dict__.get("_hd1910_encoder_histories", {}).values():
        history.valid[selection] = False
