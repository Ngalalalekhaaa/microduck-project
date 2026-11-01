"""Bounded HOME corrections with a consistent, unknown encoder zero offset.

The policy commands angles in the miscalibrated encoder coordinate system:
``q_reported = q_true + bias`` and ``q_target_true = HOME + correction - bias``.
The bias belongs to one episode and is never an actor observation. This first
experiment supports the plain 14-servo robot, without passive backlash hinges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg


def reported_joint_position(position: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Map physical joint angles to the incorrectly calibrated encoder frame."""
    return position + bias


def physical_position_target(
    home: torch.Tensor, command_offset: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """Map an encoder-frame HOME correction back into physical joint angles."""
    return home + command_offset - bias


@dataclass(kw_only=True)
class HomeCalibrationActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    target_names_expr: tuple[str, ...] = (r"^(?!passive_).*",)
    delta_limit_rad: float = math.radians(0.25)
    offset_limit_rad: float = math.radians(8.0)
    settle_seconds: float = 0.5
    bias_range_rad: tuple[float, float] = (-math.radians(3.0), math.radians(3.0))
    zero_bias_probability: float = 0.0
    observation_frame_dim: int = 49

    def __post_init__(self) -> None:
        for name in ("delta_limit_rad", "offset_limit_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        lo, hi = self.bias_range_rad
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("bias_range_rad must be a finite ordered pair")
        if not math.isfinite(self.settle_seconds) or self.settle_seconds < 0:
            raise ValueError("settle_seconds must be finite and nonnegative")
        if (not math.isfinite(self.zero_bias_probability)
                or not 0 <= self.zero_bias_probability <= 1):
            raise ValueError("zero_bias_probability must be between zero and one")
        if type(self.observation_frame_dim) is not int or self.observation_frame_dim not in (34, 49):
            raise ValueError("observation_frame_dim must be 34 or 49")
        if self.clip is not None:
            raise ValueError("HOME calibration uses delta/offset limits, not clip")

    def build(self, env) -> HomeCalibrationAction:
        return HomeCalibrationAction(self, env)


class HomeCalibrationAction(ActionTerm):
    """Integrate one bounded correction per policy step, not per physics step."""

    cfg: HomeCalibrationActionCfg

    def __init__(self, cfg: HomeCalibrationActionCfg, env):
        super().__init__(cfg, env)
        joint_ids, joint_names = self._entity.find_joints(cfg.target_names_expr)
        if len(joint_ids) != 14 or any("backlash" in n for n in self._entity.joint_names):
            raise ValueError("HOME calibration currently requires the plain 14-servo model")
        self.joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=self.device)
        self.joint_names = joint_names
        self.home = self._entity.data.default_joint_pos[:, self.joint_ids].clone()
        self.bias = torch.zeros_like(self.home)
        self.command_offset = torch.zeros_like(self.home)
        self._raw_actions = torch.zeros_like(self.home)
        self._processed_actions = self.home.clone()
        self.applied_delta = torch.zeros_like(self.home)
        self.control_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # These are deliberately synthetic current-sensor parameters. They do
        # not identify the HD1910 register, which still needs hardware validation.
        self.current_gain = torch.ones_like(self.home)
        self.current_bias_a = torch.zeros_like(self.home)
        self.current_noise_std_a = torch.zeros_like(self.home)
        self.current_quantization_a = torch.full_like(self.home, 0.0065)
        self.gravity_noise_std = torch.zeros(self.num_envs, 1, device=self.device)
        self.gyro_noise_std = torch.zeros_like(self.gravity_noise_std)
        self.observation_cache = torch.zeros(
            self.num_envs, cfg.observation_frame_dim, device=self.device
        )
        self.observation_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.observation_valid = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Physical position targets, after applying the unknown zero offset."""
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected actions of shape {tuple(self._raw_actions.shape)}")
        self._raw_actions.copy_(actions)
        delta = torch.nan_to_num(actions).clamp(-1.0, 1.0) * self.cfg.delta_limit_rad
        ready = self.control_steps * self._env.step_dt >= self.cfg.settle_seconds - 1e-8
        delta = torch.where(ready[:, None], delta, torch.zeros_like(delta))
        previous = self.command_offset.clone()
        self.command_offset.add_(delta).clamp_(
            -self.cfg.offset_limit_rad, self.cfg.offset_limit_rad
        )
        self.applied_delta.copy_(self.command_offset - previous)
        self._processed_actions.copy_(
            physical_position_target(self.home, self.command_offset, self.bias)
        )
        self.control_steps.add_(1)

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(
            self._processed_actions, joint_ids=self.joint_ids
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Clear selected episode histories while retaining newly sampled bias.

        mjlab calls reset events before ActionManager.reset. Bias and sensor
        nuisance therefore belong to the reset event and must survive this call.
        ObservationManager separately clears its frame stack for these worlds.
        """
        selection = slice(None) if env_ids is None else env_ids
        self.command_offset[selection] = 0.0
        self._raw_actions[selection] = 0.0
        self.applied_delta[selection] = 0.0
        self.control_steps[selection] = 0
        self._processed_actions[selection] = physical_position_target(
            self.home[selection], self.command_offset[selection], self.bias[selection]
        )
        self.observation_cache[selection] = 0.0
        self.observation_step[selection] = -1
        self.observation_valid[selection] = False
