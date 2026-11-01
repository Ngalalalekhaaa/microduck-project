"""Privileged-training HOME correction for a freely standing HD1910 duck.

This is a separate calibration policy family, not a hot-swappable 61D walking
policy. The actor sees eight 49D frames in v1, or eight 34D encoder/IMU frames
in the separate no-current v2 task; only the critic and
rewards see physical angles and calibration bias. No walking checkpoint is
loaded. Physical and sensor uncertainty ranges are assumptions, not hardware
identification results.
"""

from copy import deepcopy
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from . import hd1910_robust_mdp
from . import mdp as duck_mdp
from .home_calibration_actions import HomeCalibrationActionCfg
from .microduck_hd1910_env_cfg import make_microduck_hd1910_velocity_env_cfg


TASK_ID = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
NO_CURRENT_TASK_ID = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
FRAME_DIM = 49
NO_CURRENT_FRAME_DIM = 34
HISTORY_LENGTH = 8
ACTOR_DIM = FRAME_DIM * HISTORY_LENGTH
NO_CURRENT_ACTOR_DIM = NO_CURRENT_FRAME_DIM * HISTORY_LENGTH
CONTROL_DT = 0.05
BIAS_LIMIT_DEG = 3.0
DELTA_LIMIT_DEG = 0.25
OFFSET_LIMIT_DEG = 8.0
SERVO_JOINTS = (r"^(?!passive_).*",)
HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)")
LEG_BODIES = (
    "yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left",
    "bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right",
)


def make_microduck_home_calibration_env_cfg(
    play: bool = False,
    evaluation_profile: str | None = None,
    include_current: bool = True,
) -> ManagerBasedRlEnvCfg:
    """Build independent train/nominal/stress settings with identical I/O.

    Nominal removes nuisance randomization but retains unknown zero offsets;
    callers explicitly set the action bias range to zero for a HOME hold test.
    Play uses the training distribution, so it cannot hide failure by silently
    removing the calibration errors the policy is meant to correct.
    ``include_current=False`` creates the independent 272D actor contract;
    both actor and critic sensor frames then omit all current observations.
    """
    profile = evaluation_profile or "train"
    if profile not in ("train", "nominal", "stress"):
        raise ValueError(f"Unknown HOME calibration profile: {profile}")
    nominal = profile == "nominal"
    stress = profile == "stress"
    cfg = deepcopy(make_microduck_hd1910_velocity_env_cfg(play=False))
    cfg.scene.num_envs = 16 if play else 1024
    cfg.decimation = 10
    cfg.sim.mujoco.timestep = 0.005
    cfg.episode_length_s = 10.0
    cfg.commands = {}
    cfg.curriculum = {}
    cfg.actions = {
        "home_calibration": HomeCalibrationActionCfg(
            entity_name="robot",
            target_names_expr=SERVO_JOINTS,
            delta_limit_rad=math.radians(DELTA_LIMIT_DEG),
            offset_limit_rad=math.radians(OFFSET_LIMIT_DEG),
            bias_range_rad=(-math.radians(BIAS_LIMIT_DEG), math.radians(BIAS_LIMIT_DEG)),
            zero_bias_probability=0.0 if include_current else .1,
            observation_frame_dim=FRAME_DIM if include_current else NO_CURRENT_FRAME_DIM,
        ),
    }
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.per_joint_randomization = True
    actuator.delay_min_lag, actuator.delay_max_lag = 2, 8
    actuator.delay_hold_prob = 1.0
    actuator.delay_update_period = 0

    # Retain the same BAM expansion/contact implementation as the walking
    # recipe, but remove locomotion pushes, commands and observation-only bias.
    inherited = cfg.events
    cfg.events = {
        "expand_bam_friction_fields": inherited["expand_bam_friction_fields"],
        "reset_base": inherited["reset_base"],
        "foot_friction": inherited["foot_friction"],
        "randomize_armature": inherited["randomize_armature"],
    }
    cfg.events["reset_base"].params.update({
        "pose_range": {"x": (0., 0.), "y": (0., 0.), "z": (.125, .125),
                       "roll": (-.01, .01), "pitch": (-.01, .01), "yaw": (0., 0.)},
        "velocity_range": {},
    })
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["foot_friction"].params.update({
        "ranges": (1., 1.) if nominal else ((.5, 1.3) if stress else (.7, 1.2)),
        "shared_random": True,
    })
    cfg.events["randomize_armature"].params.update({
        "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS),
        "ranges": (1., 1.) if nominal else (.95, 1.05),
    })
    cfg.events["home_actuator_response"] = EventTermCfg(
        func=hd1910_robust_mdp.randomize_hd1910_response,
        mode="reset",
        params={
            "kp_range": (1., 1.) if nominal else ((.85, 1.15) if stress else (.95, 1.05)),
            "friction_range": (1., 1.) if nominal else ((.8, 1.2) if stress else (.9, 1.1)),
            "effort_range": (1., 1.) if nominal else (.95, 1.),
            "voltage_range": (7.4, 7.4) if nominal else ((6.8, 8.) if stress else (7.4, 8.)),
            "voltage_drop_range": (0., 0.) if nominal else (0., .2),
            "voltage_floor": 6.4 if stress else 7.0,
            "lag_range": (3, 3) if nominal else ((2, 8) if stress else (3, 6)),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    if not nominal:
        mass_range = (.9, 1.1) if stress else (.97, 1.03)
        com_limit = .004 if stress else .002
        for name, bodies in (("trunk", ("trunk_base",)), ("head", HEAD_BODIES), ("legs", LEG_BODIES)):
            cfg.events[f"home_{name}_inertia"] = EventTermCfg(
                func=dr.pseudo_inertia,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=bodies),
                    "alpha_range": tuple(math.log(v) / 2. for v in mass_range),
                    "t1_range": (-com_limit, com_limit),
                    "t2_range": (-com_limit, com_limit),
                    "t3_range": (-com_limit, com_limit),
                },
            )
    cfg.events["home_calibration_reset"] = EventTermCfg(
        func=duck_mdp.reset_home_calibration,
        mode="reset",
        params={
            "initial_position_noise_rad": .005,
            "current_gain_range": (1., 1.) if nominal else ((.8, 1.2) if stress else (.9, 1.1)),
            "current_bias_range_a": (0., 0.) if nominal else (-.02, .02),
            "current_noise_std_range_a": (0., 0.) if nominal else ((.0065, .015) if stress else (.003, .0065)),
            "current_quantization_range_a": (.0065, .0065),
            "imu_mounting_error_deg": 0. if nominal else (2. if stress else .5),
            "gravity_noise_std": 0. if nominal else .002,
            "gyro_noise_std": 0. if nominal else .005,
        },
    )
    # One observation term makes history flattening frame-major, oldest first.
    # No true joint position, hidden bias, force sensor or root position is in
    # the actor. Critic-only channels deliberately use the unbiased state.
    sensor_frame = (duck_mdp.home_calibration_observation if include_current
                    else duck_mdp.home_calibration_no_current_observation)
    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms={"sensor_frame": ObservationTermCfg(func=sensor_frame)},
            concatenate_terms=True,
            enable_corruption=False,
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms={
                "sensor_frame": ObservationTermCfg(func=sensor_frame),
                "true_joint_error": ObservationTermCfg(func=duck_mdp.home_calibration_true_joint_error),
                "calibration_bias": ObservationTermCfg(func=duck_mdp.home_calibration_bias),
                "joint_velocity": ObservationTermCfg(func=duck_mdp.home_calibration_joint_velocity),
                "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
                "true_gravity": ObservationTermCfg(func=mdp.projected_gravity),
                "foot_contact_forces": ObservationTermCfg(
                    func=duck_mdp.foot_contact_forces_safe,
                    params={"sensor_name": "feet_ground_contact"},
                ),
            },
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }
    cfg.rewards = {
        "true_home_tracking": RewardTermCfg(
            func=mdp.posture, weight=4.,
            params={"std": {r".*": math.radians(3.)},
                    "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS)},
        ),
        "true_home_error": RewardTermCfg(
            func=duck_mdp.home_calibration_pose_error, weight=-.25,
            params={"scale_rad": math.radians(3.)},
        ),
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.),
        "trunk_tilt": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.),
        "joint_motion": RewardTermCfg(
            func=mdp.joint_vel_l2, weight=-.002,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS)},
        ),
        "body_motion": RewardTermCfg(func=duck_mdp.home_calibration_body_motion_cost, weight=-.05),
        "delta_command": RewardTermCfg(func=duck_mdp.home_calibration_action_cost, weight=-.05),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-.005),
        "double_support": RewardTermCfg(func=duck_mdp.home_calibration_support_cost, weight=-.5),
        "failure": RewardTermCfg(func=mdp.is_terminated, weight=-500.),
    }
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": math.radians(25.)},
        ),
        "low_trunk": TerminationTermCfg(
            func=duck_mdp.root_height_below, params={"min_height": .085},
        ),
        "nan_state": TerminationTermCfg(
            func=duck_mdp.robot_state_is_nan,
            params={"sensor_names": ("feet_ground_contact",)},
        ),
    }
    return cfg


MicroduckHomeCalibrationRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": .3, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
        num_learning_epochs=5, num_mini_batches=4, learning_rate=3e-4,
        gamma=.995, lam=.95, entropy_coef=.003, desired_kl=.01,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    num_steps_per_env=32,
    max_iterations=3000,
    save_interval=100,
    experiment_name="microduck_hd1910_home_calibration",
    run_name="home_bias3deg_history8",
    logger="tensorboard",
    upload_model=False,
    seed=42,
)


MicroduckHomeCalibrationNoCurrentRlCfg = deepcopy(MicroduckHomeCalibrationRlCfg)
MicroduckHomeCalibrationNoCurrentRlCfg.experiment_name = "microduck_hd1910_home_calibration_no_current"
MicroduckHomeCalibrationNoCurrentRlCfg.run_name = "home_no_current_bias3deg_history8"
