"""Continue the original sit/stand actor on wide HD1910 replica uncertainty.

The deployment contract remains 61 raw observations -> 14 raw radian offsets.
Encoder bias shifts BOTH reported positions and physical position commands
through mjlab's JointPositionAction.apply_actions; it is not just pose noise.
All per-episode DR restores compiled defaults, and curricula edit live managers.
"""

from copy import deepcopy
from dataclasses import replace
import math

from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.actuator.feetech_bam import FeetechBamActuatorCfg
from . import hd1910_observations as encoder_obs
from . import hd1910_robust_mdp
from . import mdp
from .microduck_hd1910_env_cfg import HD1910_M6_JSON
from .microduck_sitstand_env_cfg import MicroduckSitStandRlCfg, make_microduck_sitstand_env_cfg

TASK_ID = "Mjlab-SitStand-Flat-MicroDuck-HD1910-Robust"
TILT_THRESHOLD_DEG = 15.
TILT_FAILURE_DEG = 35.
SERVO_JOINTS = (r"^(?!passive_).*",)


def _stages():
    stages = []
    # Additional iterations, trunk/head CoM mm, global/head mass, zero error deg,
    # feet mu, other P scale, ankle P scale, joint friction, supply/sag/floor/lag.
    definitions = (
        (0, (12, 12, 6), (7, 7, 4), (.95, 1.05), (.95, 1.05), 1.5,
         (.6, 1.4), (.95, 1.05), (.90, .95), (.85, 1.15), (7.0, 8.4), (0., .2), 6.5, (2, 6)),
        (500, (17, 17, 8), (10, 10, 6), (.90, 1.10), (.90, 1.15), 2.5,
         (.5, 1.5), (.9, 1.1), (.85, .95), (.75, 1.25), (6.8, 8.4), (0., .25), 6.3, (2, 7)),
        (1500, (22, 22, 10), (13, 13, 8), (.85, 1.15), (.80, 1.25), 4.,
         (.4, 1.6), (.85, 1.15), (.82, .95), (.65, 1.35), (6.5, 8.4), (0., .3), 6.0, (1, 8)),
        (2500, (25, 25, 12), (15, 15, 10), (.80, 1.20), (.75, 1.30), 5.,
         (.3, 1.6), (.8, 1.2), (.80, .95), (.6, 1.4), (6.4, 8.4), (0., .35), 5.8, (1, 8)),
    )
    for i, (iteration, trunk, head, mass, head_mass, bias, mu, kp, ankle,
            friction, voltage, sag, floor, lag) in enumerate(definitions):
        stages.append({
            "step": iteration * 24,
            "events": {
                "sitstand_inertia": {
                    "mass_range": mass, "head_mass_range": head_mass,
                    "trunk_com_range": tuple(v / 1000 for v in trunk),
                    "head_com_range": tuple(v / 1000 for v in head),
                },
                "sitstand_response": {
                    "kp_range": kp, "ankle_kp_range": ankle,
                    "friction_range": friction, "effort_range": (.9, 1.),
                    "voltage_range": voltage, "voltage_drop_range": sag,
                    "voltage_floor": floor, "lag_range": lag,
                },
                "foot_friction": {"ranges": mu},
                "encoder_bias": {"bias_range": (-math.radians(bias), math.radians(bias))},
                "push_robot": {"velocity_range": {
                    "x": (-.025 * i, .025 * i), "y": (-.025 * i, .025 * i),
                }},
            },
            "reward_weights": {
                "action_rate_l2": (-1., -1.1, -1.3, -1.5)[i],
                "head_action_rate_l2": (-.15, -.2, -.3, -.4)[i],
                "whole_motion_tilt": -8.,
            },
        })
    return stages


ROBUST_STAGES = _stages()


def make_microduck_hd1910_sitstand_robust_env_cfg(play=False, evaluation_profile=None):
    profile = evaluation_profile or ("stress" if play else "train")
    if profile not in ("train", "stress", "nominal"):
        raise ValueError(profile)
    cfg = deepcopy(make_microduck_sitstand_env_cfg(play=False))
    cfg.scene.num_envs = 16 if play else 512
    cfg.curriculum = {}
    # Preserve the mature source skill's regularizers, not the discovery ramp.
    cfg.rewards["action_rate_l2"].weight = -1.
    cfg.rewards["joint_torque_rate_l2"].weight = -.001
    cfg.rewards["descent_speed"].weight = 20.
    cfg.rewards["rise_speed"].weight = 5.
    cfg.rewards["whole_motion_tilt"] = RewardTermCfg(
        func=mdp.hd1910_sitstand_tilt_cost, weight=-8.,
        params={"threshold_deg": TILT_THRESHOLD_DEG, "scale_deg": 10.},
    )
    cfg.rewards["head_action_rate_l2"] = RewardTermCfg(
        func=mdp.neck_action_rate_l2, weight=-.15,
    )
    cfg.rewards["tilt_failure"] = RewardTermCfg(func=base_mdp.is_terminated, weight=-500.)
    cfg.terminations["tilt_failure"] = TerminationTermCfg(
        func=base_mdp.bad_orientation, params={"limit_angle": math.radians(TILT_FAILURE_DEG)},
    )
    # Shared source cfg must never be changed by this experimental factory.
    robot = cfg.scene.entities["robot"]
    robot.articulation.actuators = (FeetechBamActuatorCfg(
        json_path=str(HD1910_M6_JSON), target_names_expr=SERVO_JOINTS,
        kp_fw=5., per_joint_randomization=True,
        vin_range=(6.4, 8.4), vin_drop_gain_range=(0., .35), vin_min=5.8,
        delay_min_lag=1, delay_max_lag=8, delay_hold_prob=1., delay_update_period=0,
    ),)
    for name in ("base_com", "randomize_com", "randomize_head_com",
                 "randomize_mass_inertia", "randomize_joint_friction", "randomize_motor_gains"):
        cfg.events.pop(name, None)
    cfg.events["sitstand_inertia"] = EventTermCfg(
        func=mdp.randomize_hd1910_sitstand_inertia, mode="reset",
    )
    cfg.events["sitstand_response"] = EventTermCfg(
        func=mdp.randomize_hd1910_sitstand_response, mode="reset",
    )
    cfg.events["encoder_bias"].mode = "reset"
    cfg.events["encoder_bias"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SERVO_JOINTS)
    cfg.events["foot_friction"].mode = "reset"
    cfg.events["foot_friction"].params["shared_random"] = False
    cfg.events["randomize_armature"].params.update({
        "asset_cfg": SceneEntityCfg("robot", joint_names=SERVO_JOINTS), "ranges": (.8, 1.2),
    })
    cfg.events["push_robot"].interval_range_s = (5., 9.)
    cfg.events["reset_encoder_history"] = EventTermCfg(
        func=encoder_obs.reset_hd1910_encoder_history, mode="reset",
    )
    # Real deployment differences consecutive 4096-count encoder positions.
    actor = cfg.observations["actor"]
    actor.terms["joint_pos"].func = encoder_obs.joint_pos_rel_quantized
    actor.terms["joint_vel"].func = encoder_obs.joint_vel_finite_difference
    actor.terms["joint_vel"].params["biased"] = True
    actor.terms["joint_vel"].delay_min_lag = actor.terms["joint_vel"].delay_max_lag = 0
    actor.terms["joint_pos"].noise = actor.terms["joint_vel"].noise = None
    # Keep the shared command layout and train the exact deployed zero head input.
    cfg.commands["head_pose"].ranges = tuple(
        (-math.radians(d), math.radians(d)) for d in (5., 5., 10., 3.)
    )
    cfg.commands["head_pose"].zero_command_prob = .9
    chosen = deepcopy(ROBUST_STAGES[-1] if profile == "stress" else ROBUST_STAGES[0])
    if profile == "nominal":
        chosen["events"]["sitstand_inertia"].update({
            "mass_range": (1., 1.), "head_mass_range": (1., 1.),
            "trunk_com_range": (0., 0., 0.), "head_com_range": (0., 0., 0.),
        })
        chosen["events"]["sitstand_response"].update({
            "kp_range": (1., 1.), "ankle_kp_range": (.9, .9),
            "friction_range": (1., 1.), "effort_range": (1., 1.),
            "voltage_range": (7.6, 7.6), "voltage_drop_range": (0., 0.),
            "voltage_floor": 6., "lag_range": (4, 4),
        })
        chosen["events"]["foot_friction"]["ranges"] = (1., 1.)
        chosen["events"]["encoder_bias"]["bias_range"] = (0., 0.)
        cfg.events["randomize_armature"].params["ranges"] = (1., 1.)
        actor.enable_corruption = False
        for name in ("base_ang_vel", "projected_gravity"):
            actor.terms[name].params["max_angle_deg"] = 0.
            actor.terms[name].delay_min_lag = actor.terms[name].delay_max_lag = 0
    for name, params in chosen["events"].items():
        cfg.events[name].params.update(params)
    for name, weight in chosen["reward_weights"].items():
        cfg.rewards[name].weight = weight
    if profile == "train":
        cfg.curriculum["sitstand_robustness"] = CurriculumTermCfg(
            func=hd1910_robust_mdp.hd1910_robust_curriculum,
            params={"stages": deepcopy(ROBUST_STAGES)},
        )
    return cfg


MicroduckHD1910SitStandRobustRlCfg = replace(
    deepcopy(MicroduckSitStandRlCfg),
    experiment_name="microduck_hd1910_sitstand_robust", run_name="sitstand_1699_robust",
    logger="tensorboard", save_interval=100, max_iterations=5000,
)
MicroduckHD1910SitStandRobustRlCfg.actor.distribution_cfg.update({"init_std": .1})
MicroduckHD1910SitStandRobustRlCfg.algorithm.learning_rate = 1.e-4
MicroduckHD1910SitStandRobustRlCfg.algorithm.entropy_coef = .001
MicroduckHD1910SitStandRobustRlCfg.algorithm.desired_kl = .005
