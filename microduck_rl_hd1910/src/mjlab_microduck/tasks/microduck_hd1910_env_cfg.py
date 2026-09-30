"""Microduck walking with LuwuDynamics' public 1910 BAM M6 baseline.

The optional backlash model assumes +/-1 degree of play; this is not a
measurement of the replica. Geometry, mass, inertia, reward and observation
conventions remain those of this Microduck revision. See robot/hd1910/README.md.
"""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.actuator.feetech_bam import (
    FeetechBacklashBamActuatorCfg,
    FeetechBamActuatorCfg,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_BACKLASH_ROBOT_CFG

from .backlash import make_backlash_variant
from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg


HD1910_M6_JSON = Path(__file__).resolve().parents[1] / "robot" / "hd1910" / "1910_m6.json"


def make_microduck_hd1910_velocity_env_cfg(
    play: bool = False,
    backlash: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play)
    # The generic backlash wrapper replaces the WHOLE robot. Supply a private
    # walk-backlash copy, then replace its XL330 actuator with the Feetech one.
    if backlash:
        cfg = make_backlash_variant(cfg, deepcopy(MICRODUCK_WALK_BACKLASH_ROBOT_CFG))
    else:
        cfg.scene.entities["robot"] = deepcopy(cfg.scene.entities["robot"])
    robot = cfg.scene.entities["robot"]
    assert robot.articulation is not None
    actuator_cfg = FeetechBacklashBamActuatorCfg if backlash else FeetechBamActuatorCfg
    robot.articulation.actuators = (
        actuator_cfg(
            json_path=str(HD1910_M6_JSON),
            target_names_expr=(r"^(?!passive_).*",),
            kp_fw=5.0,
            vin_range=(7.4, 8.0),
            vin_drop_gain_range=(0.0, 0.2),
            vin_min=7.0,
            delay_min_lag=3,
            delay_max_lag=6,
        ),
    )
    return cfg


MicroduckHD1910RlCfg = replace(
    deepcopy(MicroduckRlCfg),
    experiment_name="microduck_hd1910_velocity",
    run_name="hd1910_m6_luwu",
    logger="tensorboard",
)

MicroduckHD1910BacklashRlCfg = replace(
    deepcopy(MicroduckHD1910RlCfg),
    experiment_name="microduck_hd1910_velocity_backlash",
    run_name="hd1910_m6_luwu_backlash_assumed_2deg",
)
