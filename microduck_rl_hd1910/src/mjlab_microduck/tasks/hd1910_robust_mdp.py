"""HD1910 robust walking MDP terms, isolated from the baseline tasks."""

from dataclasses import dataclass as _dataclass

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .mdp import (
    VelocityCommandCommandOnly,
    VelocityCommandCommandOnlyCfg,
    _backlash_encoder_ids,
    base_ang_vel_imu_misaligned,
    head_pose_tracking,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


# ==============================================================================
# HD1910 robust walking: deployed commands and physically effective DR
# ==============================================================================

@requires_model_fields("jnt_range")
def randomize_hd1910_backlash(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Scale each backlash width while keeping its +/- endpoints symmetric."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset.indexing.joint_ids[asset_cfg.joint_ids]
    default = env.sim.get_default_field("jnt_range")[joint_ids]
    scale = torch.empty((len(env_ids), len(joint_ids), 1), device=env.device).uniform_(*ranges)
    env.sim.model.jnt_range[env_ids[:, None], joint_ids] = default * scale


@requires_model_fields("actuator_forcerange")
def randomize_hd1910_response(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    kp_range: tuple[float, float],
    friction_range: tuple[float, float],
    effort_range: tuple[float, float],
    voltage_range: tuple[float, float],
    voltage_drop_range: tuple[float, float],
    voltage_floor: float,
    lag_range: tuple[int, int],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Reset independent servo responses and shared battery/bus parameters.

    All scales replace their defaults. Gains and friction vary per joint;
    battery voltage, load sag and command lag are shared by a robot. The
    voltage-drop gain is V/Nm times the sum of absolute motor torques, a
    robustness assumption rather than an identified electrical resistance.
    KD stays nominal: BAM's kd_scale is back-EMF, not the firmware D register.
    """
    from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if not len(env_ids):
        return
    asset = env.scene[asset_cfg.name]
    for actuator in asset.actuators:
        if not isinstance(actuator.cfg, FeetechBamActuatorCfg):
            raise TypeError("HD1910 response DR requires a Feetech BAM actuator")
        if not actuator.cfg.per_joint_randomization:
            raise ValueError("HD1910 response DR requires per_joint_randomization")
        count = len(actuator.ctrl_ids)

        def sample(ranges, shape):
            return torch.empty(shape, device=env.device).uniform_(*ranges)

        actuator.reset_gains(env_ids)
        actuator.set_gains(env_ids, kp_scale=sample(kp_range, (len(env_ids), count)))
        actuator.reset_friction_scale(env_ids)
        actuator.set_friction_scale(env_ids, sample(friction_range, (len(env_ids), count)))
        actuator.vin_tensor[env_ids] = sample(voltage_range, (len(env_ids), 1))
        actuator.vin_drop_gain[env_ids] = sample(voltage_drop_range, (len(env_ids), 1))
        actuator.voltage_floor[env_ids] = voltage_floor
        actuator.set_lags(torch.randint(lag_range[0], lag_range[1] + 1,
                                       (len(env_ids),), device=env.device), env_ids)
        ctrl_ids = actuator.global_ctrl_ids
        default = env.sim.get_default_field("actuator_forcerange")[ctrl_ids]
        scale = sample(effort_range, (len(env_ids), count, 1))
        env.sim.model.actuator_forcerange[env_ids[:, None], ctrl_ids] = default * scale


def hd1910_robust_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[dict],
) -> torch.Tensor:
    """Apply stages at absolute saved counters, without resetting PPO state.

    Mutates live managers. Each DR stage takes effect on subsequent resets;
    no physical parameters jump in the middle of an episode.
    """
    del env_ids
    index = 0
    for i, stage in enumerate(stages):
        if env.common_step_counter >= stage["step"]:
            index = i
    stage = stages[index]
    for name, params in stage["events"].items():
        env.event_manager.get_term_cfg(name).params.update(params)
    for name, params in stage.get("rewards", {}).items():
        env.reward_manager.get_term_cfg(name).params.update(params)
    for name, weight in stage.get("reward_weights", {}).items():
        env.reward_manager.get_term_cfg(name).weight = weight
    return torch.tensor(float(index), device=env.device)


def track_hd1910_planar_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """Track the raw body-frame vx/vy command; vertical bounce is priced separately."""
    cmd = env.command_manager.get_command(command_name)
    actual = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    return torch.exp(-torch.square(cmd[:, :2] - actual).sum(dim=1) / std**2)


def track_hd1910_yaw_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str = "twist",
) -> torch.Tensor:
    """Track raw commanded yaw rate without taxing roll/pitch gait motion."""
    cmd = env.command_manager.get_command(command_name)[:, 2]
    actual = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(cmd - actual) / std**2)


def hd1910_velocity_error_l1(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    yaw_scale: float = 0.15,
) -> torch.Tensor:
    """Nonnegative raw-command error cost, including at small nonzero commands."""
    cmd = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    linear = torch.abs(cmd[:, :2] - data.root_link_lin_vel_b[:, :2]).sum(dim=1)
    yaw = torch.abs(cmd[:, 2] - data.root_link_ang_vel_b[:, 2])
    return linear + yaw_scale * yaw


def hd1910_vertical_velocity_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_lin_vel_b[:, 2].square()


def hd1910_joint_limit_proximity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    margin_fraction: float = .1,
) -> torch.Tensor:
    """Measured output-angle cost in the outer physical joint-range margin.

    Commands retain their allowed overshoot for P=5 gravity compensation.
    The encoder can be beyond the motor angle by the backlash hinge angle.
    """
    asset = env.scene[asset_cfg.name]
    main_ids, backlash_ids, mask = _backlash_encoder_ids(env, asset, asset_cfg)
    position = asset.data.joint_pos[:, main_ids] + asset.data.joint_pos[:, backlash_ids] * mask
    limits = asset.data.default_joint_pos_limits[:, main_ids]
    margin = (limits[:, :, 1] - limits[:, :, 0]) * margin_fraction
    low = torch.relu((limits[:, :, 0] + margin - position) / margin)
    high = torch.relu((position - limits[:, :, 1] + margin) / margin)
    return (low.square() + high.square()).sum(dim=1)


def randomize_hd1910_gyro_bias(env: ManagerBasedRlEnv, env_ids, bias_range=(-.015, .015)) -> None:
    if not hasattr(env, "_hd1910_gyro_bias"):
        env._hd1910_gyro_bias = torch.zeros((env.num_envs, 3), device=env.device)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env._hd1910_gyro_bias[env_ids] = torch.empty((len(env_ids), 3), device=env.device).uniform_(*bias_range)


def hd1910_gyro_observation(env: ManagerBasedRlEnv, max_angle_deg: float = 6.) -> torch.Tensor:
    measured = base_ang_vel_imu_misaligned(env, max_angle_deg=max_angle_deg)
    return measured + getattr(env, "_hd1910_gyro_bias", 0.)


def hd1910_head_command_l2(
    env: ManagerBasedRlEnv,
    yaw_std: float = 0.35,
    roll_std: float = 0.15,
    command_name: str = "head_pose",
) -> torch.Tensor:
    """Nonnegative yaw/roll command error using the output-side encoder angle.

    Pitch retains the broader tracking reward and EMA bias cost, allowing the
    heavy head to move with a gait. A gentle nonzero head command changes the
    target of every head tracking term, avoiding conflicting HOME pressure.
    """
    if not hasattr(env, "_head_pose_neck_ids"):
        head_pose_tracking(env, command_name=command_name)
    data = env.scene["robot"].data
    ids = env._head_pose_neck_ids
    measured = data.joint_pos[:, ids] + (
        data.joint_pos[:, env._head_pose_bl_ids] * env._head_pose_bl_mask
    )
    command = env.command_manager.get_command(command_name)
    error = measured - data.default_joint_pos[:, ids] - command
    return (error[:, 2] / yaw_std).square() + (error[:, 3] / roll_std).square()


class HD1910RobustVelocityCommand(VelocityCommandCommandOnly):
    """Exclusive deployment buckets, with deliberate small-speed practice."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        count = len(env_ids)
        if not count:
            return
        bucket = torch.rand(count, device=self.device)
        stand = bucket < 0.20
        straight = (bucket >= 0.20) & (bucket < 0.60)
        turn = (bucket >= 0.60) & (bucket < 0.75)
        lateral = (bucket >= 0.75) & (bucket < 0.85)
        isolated = stand | straight | turn | lateral
        self.vel_command_b[env_ids[isolated]] = 0.0
        sign = torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)
        max_x = max(abs(v) for v in self.cfg.ranges.lin_vel_x)
        max_y = max(abs(v) for v in self.cfg.ranges.lin_vel_y)
        max_yaw = max(abs(v) for v in self.cfg.ranges.ang_vel_z)
        speed = torch.empty(count, device=self.device).uniform_(0.03, max_x)
        small_speed = torch.empty(count, device=self.device).uniform_(0.03, min(0.12, max_x))
        speed = torch.where(torch.rand(count, device=self.device) < 0.5, small_speed, speed)
        self.vel_command_b[env_ids[straight], 0] = (sign * speed)[straight]
        yaw = torch.empty(count, device=self.device).uniform_(0.2, max_yaw)
        self.vel_command_b[env_ids[turn], 2] = (sign * yaw)[turn]
        side = torch.empty(count, device=self.device).uniform_(0.03, max_y)
        slow_side = torch.empty(count, device=self.device).uniform_(0.03, min(.12, max_y))
        side = torch.where(torch.rand(count, device=self.device) < .5, slow_side, side)
        self.vel_command_b[env_ids[lateral], 1] = (sign * side)[lateral]
        self.is_standing_env[env_ids] = stand
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@_dataclass(kw_only=True)
class HD1910RobustVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    def build(self, env: ManagerBasedRlEnv) -> HD1910RobustVelocityCommand:
        return HD1910RobustVelocityCommand(self, env)
