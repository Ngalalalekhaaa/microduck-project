"""Feetech goal history for both plain and encoder-through-backlash BAM.

BAM initializes STS3215.q_target_smooth only when loading identification logs.
mjlab does not load those logs. Seed the limiter from the measured position on
the first compute after reset, after reset events have placed the joints.
"""

from dataclasses import dataclass

import torch
from mjlab.actuator.actuator import ActuatorCmd

from .friction_dr_bam import (
    BacklashEncoderBamActuator,
    FrictionDRBamActuator,
    FrictionDRBamActuatorCfg,
)


class _FeetechGoalHistory:
    """Cooperative mixin: preserve the selected BAM feedback implementation."""

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        # Robust HD1910 tasks vary the response of each physical servo. The
        # base BAM tensors are (world, 1); its voltage law also broadcasts
        # (world, joint) tensors, so opt into that shape only for these tasks.
        if self.cfg.per_joint_randomization:
            self.kp_scale = torch.ones_like(self._prev_motor_torque)
            self.kd_scale = torch.ones_like(self._prev_motor_torque)
            self.default_kp_scale = self.kp_scale.clone()
            self.default_kd_scale = self.kd_scale.clone()
            self.friction_scale = torch.ones_like(self._prev_motor_torque)
            self.default_friction_scale = self.friction_scale.clone()
            self.voltage_floor = torch.full_like(self.kp_scale[:, :1], self.cfg.vin_min)
        self._bam_model.actuator.q_target_smooth = torch.zeros_like(self._prev_motor_torque)
        self._goal_reset_pending = torch.ones(
            (data.nworld, 1), dtype=torch.bool, device=device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._goal_reset_pending.fill_(True)
        else:
            self._goal_reset_pending[env_ids] = True

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        actuator = self._bam_model.actuator
        # On backlash models get_command() supplies the output-side encoder
        # angle here; velocity stays motor-side for back-EMF and friction.
        actuator.q_target_smooth = torch.where(
            self._goal_reset_pending, cmd.pos, actuator.q_target_smooth
        )
        self._goal_reset_pending.zero_()
        if self.cfg.per_joint_randomization:
            # BAM accepts a broadcast tensor in clamp(). Keep the serialized
            # config scalar while using a per-world floor during the solve.
            nominal_floor = self.cfg.vin_min
            self.cfg.vin_min = self.voltage_floor
            try:
                return super().compute(cmd)
            finally:
                self.cfg.vin_min = nominal_floor
        return super().compute(cmd)


class FeetechBamActuator(_FeetechGoalHistory, FrictionDRBamActuator):
    """Feetech BAM with friction randomization and per-world goal reset."""


class FeetechBacklashBamActuator(_FeetechGoalHistory, BacklashEncoderBamActuator):
    """Feetech goal reset plus output-side encoder feedback through backlash."""


@dataclass(kw_only=True)
class FeetechBamActuatorCfg(FrictionDRBamActuatorCfg):
    per_joint_randomization: bool = False

    def build(self, entity, target_ids, target_names) -> FeetechBamActuator:
        return FeetechBamActuator(self, entity, target_ids, target_names)


@dataclass(kw_only=True)
class FeetechBacklashBamActuatorCfg(FeetechBamActuatorCfg):
    def build(self, entity, target_ids, target_names) -> FeetechBacklashBamActuator:
        return FeetechBacklashBamActuator(self, entity, target_ids, target_names)
