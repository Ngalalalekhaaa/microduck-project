"""HD1910 continuation with measured-command tracking and bounded uncertainty.

The baseline task and the public BAM JSON remain independent of this recipe.
Ranges cover plausible replica differences and the reported battery sag; they
are training assumptions, not measurements or an identified servo model.
Encoder-bias DR shifts actor observations only. It approximates a sensing
offset and does not reproduce an incorrect calibration transform that would
also shift physical action targets when converting joint angles to ticks.
"""

from copy import deepcopy
from dataclasses import replace
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import hd1910_robust_mdp as robust_mdp
from . import hd1910_observations as encoder_obs
from . import mdp as microduck_mdp
from .microduck_hd1910_env_cfg import (
    MicroduckHD1910RlCfg,
    make_microduck_hd1910_velocity_env_cfg,
)


NUM_STEPS_PER_ENV = 24
CONTINUATION_START_STEP = 120_024
HEAD_COMMAND_RANGES = tuple(
    (-math.radians(degrees), math.radians(degrees)) for degrees in (10., 10., 20., 5.)
)
HEAD_ZERO_COMMAND_PROBABILITY = .8
_SERVO_JOINTS = (r"^(?!passive_).*",)
_HEAD_BODIES = ("neck", "neck_pitch", "yaw_roll_motion", "(bottom_head_shell|jaw_soft)")
_LEG_BODIES = (
    "yaw2roll", "hip_l", "upper_leg_left", "leg", "ankle_left",
    "bearing_roll", "hip_l_2", "upper_leg_right", "leg_2", "ankle_right",
)


def _inertial_params(mass_range, com_mm=(0.0, 0.0, 0.0)):
    # pseudo_inertia alpha scales mass AND inertia, maintaining a valid body.
    return {
        "alpha_range": tuple(math.log(v) / 2.0 for v in mass_range),
        "t1_range": (-com_mm[0] / 1000.0, com_mm[0] / 1000.0),
        "t2_range": (-com_mm[1] / 1000.0, com_mm[1] / 1000.0),
        "t3_range": (-com_mm[2] / 1000.0, com_mm[2] / 1000.0),
    }


def _stages():
    definitions = (
        # New iterations, mu, large/leg mass, trunk/head COM mm, P/friction,
        # effort ceiling, supply/sag/floor, delay steps, push m/s, tracking std.
        (0, (.65, 1.20), (.97, 1.03), (.98, 1.02), (3, 3, 2), (2, 2, 1),
         (.97, 1.03), (.90, 1.10), (.97, 1.), (7.4, 8.), (0., .15), 7., (3, 6), .10, .25, .70),
        (250, (.60, 1.25), (.95, 1.05), (.97, 1.03), (5, 4, 3), (3, 3, 2),
         (.94, 1.06), (.87, 1.13), (.95, 1.), (7.2, 8.), (0., .20), 6.8, (3, 7), .15, .22, .60),
        (750, (.50, 1.25), (.92, 1.08), (.96, 1.04), (6, 5, 3), (4, 4, 2),
         (.91, 1.09), (.83, 1.17), (.92, 1.), (7.0, 8.), (0., .23), 6.6, (2, 8), .20, .20, .55),
        (1500, (.45, 1.25), (.90, 1.10), (.95, 1.05), (8, 6, 4), (5, 5, 3),
         (.88, 1.12), (.80, 1.20), (.90, 1.), (6.8, 8.), (0., .25), 6.4, (2, 8), .25, .18, .50),
    )
    stages = []
    for index, (iteration, friction, large_mass, leg_mass, trunk_com, head_com,
         kp, joint_friction, effort, voltage, sag, floor, lag, push, lin_std, yaw_std) in enumerate(definitions):
        stages.append({
            "step": CONTINUATION_START_STEP + iteration * NUM_STEPS_PER_ENV,
            "events": {
                "foot_friction": {"ranges": friction},
                "robust_trunk_inertia": _inertial_params(large_mass, trunk_com),
                "robust_head_inertia": _inertial_params(large_mass, head_com),
                "robust_leg_inertia": _inertial_params(leg_mass),
                "robust_actuator_response": {
                    "kp_range": kp, "friction_range": joint_friction,
                    "effort_range": effort, "voltage_range": voltage,
                    "voltage_drop_range": sag, "voltage_floor": floor,
                    "lag_range": lag,
                },
                "push_robot": {"velocity_range": {"x": (-push, push), "y": (-push, push)}},
                "encoder_bias": {"bias_range": (-.035 - .005 * index, .035 + .005 * index)},
            },
            "rewards": {
                "track_linear_velocity": {"std": lin_std},
                "track_angular_velocity": {"std": yaw_std},
            },
            "reward_weights": {
                "action_rate_l2": (-.1, -.15, -.2, -.2)[index],
                "head_action_rate_l2": (-.03, -.05, -.08, -.1)[index],
            },
        })
    return stages


ROBUST_STAGES = _stages()


def make_microduck_hd1910_robust_velocity_env_cfg(
    play: bool = False,
    backlash: bool = False,
    evaluation_profile: str | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create training, nominal or final-DR stress physics.

    Training resumes the saved absolute counter; no warm-start reset is used.
    Play defaults to frozen final stress DR. Explicit evaluation profiles freeze
    curricula, and nominal keeps the same actuator/geometry with fixed physics.
    """
    profile = evaluation_profile or ("stress" if play else "train")
    if profile not in ("train", "nominal", "stress"):
        raise ValueError(f"Unknown HD1910 robust profile: {profile}")
    cfg = deepcopy(make_microduck_hd1910_velocity_env_cfg(play=False, backlash=backlash))
    cfg.curriculum = {}
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    actuator.per_joint_randomization = True
    # Allocate history through the final envelope; each reset explicitly samples
    # one robot's delay, held across that episode rather than changed every 5 ms.
    actuator.delay_min_lag, actuator.delay_max_lag = 2, 8
    actuator.delay_hold_prob = 1.0
    actuator.delay_update_period = 0

    for name in (
        "base_com", "randomize_com", "randomize_head_com", "randomize_mass_inertia",
        "randomize_joint_friction", "randomize_motor_gains", "randomize_joint_damping",
    ):
        cfg.events.pop(name, None)
    cfg.events["foot_friction"].mode = "reset"
    # Observation-bias approximation; target-to-tick calibration errors are a
    # separate unmodeled uncertainty until their actuator transform is measured.
    cfg.events["encoder_bias"].mode = "reset"
    cfg.events["foot_friction"].params["shared_random"] = True
    # Separate inertial terms avoid zero-mass backlash carrier bodies and the
    # old head selector's accidental right-hip randomization.
    for name, bodies in (
        ("robust_trunk_inertia", ("trunk_base",)),
        ("robust_head_inertia", _HEAD_BODIES),
        ("robust_leg_inertia", _LEG_BODIES),
    ):
        cfg.events[name] = EventTermCfg(
            func=dr.pseudo_inertia, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=bodies)},
        )
    cfg.events["robust_actuator_response"] = EventTermCfg(
        func=robust_mdp.randomize_hd1910_response, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.events["randomize_armature"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=_SERVO_JOINTS,
    )
    if backlash:
        cfg.events["robust_backlash"] = EventTermCfg(
            func=robust_mdp.randomize_hd1910_backlash, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"passive_.*_backlash",)),
                "ranges": (.75, 1.25),
            },
        )
    cfg.events["push_robot"].interval_range_s = (5.0, 9.0)
    cfg.events["reset_encoder_history"] = EventTermCfg(
        func=encoder_obs.reset_hd1910_encoder_history, mode="reset",
    )
    cfg.events["gyro_bias"] = EventTermCfg(
        func=robust_mdp.randomize_hd1910_gyro_bias, mode="reset",
        params={"bias_range": (-.015, .015)},
    )

    # The deployed actor derives velocity from consecutive encoder readings.
    # Quantization is shared by the position and finite-difference functions.
    actor = cfg.observations["actor"]
    actor.terms["base_ang_vel"].func = robust_mdp.hd1910_gyro_observation
    actor.terms["joint_pos"].func = encoder_obs.joint_pos_rel_quantized
    actor.terms["joint_vel"].func = encoder_obs.joint_vel_finite_difference
    actor.terms["joint_vel"].params["biased"] = True
    actor.terms["joint_vel"].delay_min_lag = 0
    actor.terms["joint_vel"].delay_max_lag = 0
    actor.terms["joint_pos"].noise.n_min = -.0005
    actor.terms["joint_pos"].noise.n_max = .0005
    actor.terms["joint_vel"].noise.n_min = -.10
    actor.terms["joint_vel"].noise.n_max = .10

    # Gentle head movements share training with the common deployed zero target.
    # Body commands stay zero; all 61 observation slots retain their order.
    cfg.commands["head_pose"].ranges = HEAD_COMMAND_RANGES
    cfg.commands["head_pose"].zero_command_prob = HEAD_ZERO_COMMAND_PROBABILITY
    cfg.commands["body_pose"].ranges = ((0., 0.),) * 6
    cfg.commands["body_pose"].zero_command_prob = 1.0
    twist = cfg.commands["twist"]
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.init_velocity_prob = 0.0
    cfg.commands["twist"] = robust_mdp.HD1910RobustVelocityCommandCfg(**vars(twist))

    cfg.rewards["track_linear_velocity"].func = robust_mdp.track_hd1910_planar_velocity
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].func = robust_mdp.track_hd1910_yaw_velocity
    cfg.rewards["track_angular_velocity"].weight = 2.5
    cfg.rewards["velocity_error_l1"] = RewardTermCfg(
        func=robust_mdp.hd1910_velocity_error_l1, weight=-2.0,
        params={"yaw_scale": .15},
    )
    cfg.rewards["vertical_velocity_l2"] = RewardTermCfg(
        func=robust_mdp.hd1910_vertical_velocity_l2, weight=-.2,
    )
    cfg.rewards["air_time"].weight = 1.5
    cfg.rewards["action_rate_l2"].weight = -.1
    cfg.rewards["head_action_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.neck_action_rate_l2, weight=-.03,
    )
    cfg.rewards["head_pose_tracking"].params["std"] = .5
    cfg.rewards["head_pose_bias"].weight = 1.5
    cfg.rewards["head_command_l2"] = RewardTermCfg(
        func=robust_mdp.hd1910_head_command_l2, weight=-.15,
    )
    cfg.rewards["joint_limit_proximity"] = RewardTermCfg(
        func=robust_mdp.hd1910_joint_limit_proximity, weight=-.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"(left_hip_yaw|right_hip_yaw|head_yaw|head_roll)",)),
            "margin_fraction": .1,
        },
    )
    for key in ("std_walking", "std_running"):
        cfg.rewards["pose"].params[key][r".*hip_roll.*"] = .1
    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(55)

    stages = deepcopy(ROBUST_STAGES)
    if backlash:
        for stage, limits in zip(stages, ((.9, 1.1), (.85, 1.15), (.8, 1.2), (.75, 1.25)), strict=True):
            stage["events"]["robust_backlash"] = {"ranges": limits}
    chosen = stages[-1] if profile == "stress" else stages[0]
    for name, params in chosen["events"].items():
        cfg.events[name].params.update(params)
    for name, params in chosen["rewards"].items():
        cfg.rewards[name].params.update(params)
    for name, weight in chosen["reward_weights"].items():
        cfg.rewards[name].weight = weight
    actuator.vin_min = chosen["events"]["robust_actuator_response"]["voltage_floor"]

    if profile == "train":
        cfg.curriculum["robustness"] = CurriculumTermCfg(
            func=robust_mdp.hd1910_robust_curriculum, params={"stages": stages},
        )
    elif profile == "nominal":
        for name in tuple(cfg.events):
            if name in chosen["events"] or name in ("randomize_armature", "encoder_bias", "gyro_bias"):
                cfg.events.pop(name)
        actuator.vin_range = None
        actuator.vin = 7.6
        actuator.vin_drop_gain_range = (.1, .1)
        actuator.vin_min = 6.4
        actuator.delay_min_lag = actuator.delay_max_lag = 4
        actuator.delay_hold_prob = 0.0
        actor.enable_corruption = False
        for name in ("base_ang_vel", "projected_gravity"):
            actor.terms[name].params["max_angle_deg"] = 0.0
            actor.terms[name].delay_min_lag = actor.terms[name].delay_max_lag = 0
        actor.terms["joint_pos"].params["biased"] = False
        actor.terms["joint_vel"].params["biased"] = False
    # Use the final reward definition in every frozen evaluation profile.
    if profile != "train":
        for name, params in stages[-1]["rewards"].items():
            cfg.rewards[name].params.update(params)
        for name, weight in stages[-1]["reward_weights"].items():
            cfg.rewards[name].weight = weight
    return cfg


MicroduckHD1910RobustRlCfg = replace(
    deepcopy(MicroduckHD1910RlCfg),
    experiment_name="microduck_hd1910_robust_flat",
    run_name="hd1910_robust_flat",
)
MicroduckHD1910RobustBacklashRlCfg = replace(
    deepcopy(MicroduckHD1910RobustRlCfg),
    experiment_name="microduck_hd1910_robust_backlash",
    run_name="hd1910_robust_backlash",
)
