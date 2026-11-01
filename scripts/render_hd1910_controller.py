"""CPU MuJoCo BAM actuator setup for reproducible Microduck comparisons.

This module has no mjlab/torch dependency and does not change installed BAM.
Call ``controller.update()`` immediately before each ``mujoco.mj_step``. Set
``controller.q_target`` (or modify it in place) at the policy control rate.
After changing/resetting the simulated state, call ``controller.reset(data.qpos)``.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HD1910_JSON = (
    ROOT / "microduck_rl_hd1910/src/mjlab_microduck/robot/hd1910/1910_m6.json"
)
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)


class PhysicsStepDelay:
    """Fixed command lag in physics steps, initialized with the reset pose.

    Training randomizes lags in [3, 6]. A fixed lag makes video comparisons
    deterministic; four 5 ms steps are the default nominal 20 ms latency.
    A changed target is delivered after ``lag_steps`` complete updates.
    """

    def __init__(self, lag_steps: int):
        if not isinstance(lag_steps, (int, np.integer)) or lag_steps < 0:
            raise ValueError("lag_steps must be a non-negative integer")
        self.lag_steps = int(lag_steps)
        self._history: deque[np.ndarray] = deque()

    def reset(self, target: np.ndarray) -> None:
        self._history = deque(
            np.array(target, dtype=float, copy=True) for _ in range(self.lag_steps)
        )

    def update(self, target: np.ndarray) -> np.ndarray:
        self._history.append(np.array(target, dtype=float, copy=True))
        return self._history.popleft()


class ResetSafeBamController:
    """BAM controller with independent command history and reset-safe Feetech.

    BAM's CPU reset does not reset its timestamp or initialize the STS3215
    target limiter. Explicit copies also ensure neither policy writes nor the
    lag buffer can mutate MuJoCo's measured qpos or another history entry.
    Other BAM attributes (e.g. ``qpos_indexes``) remain available unchanged.
    """

    def __init__(self, controller, lag_steps: int):
        self._controller = controller
        self.delay = PhysicsStepDelay(lag_steps)
        self._q_target = np.zeros(len(controller.actuator), dtype=float)
        self.reset(controller.mujoco_data.qpos)

    def __getattr__(self, name):
        return getattr(self._controller, name)

    @property
    def q_target(self) -> np.ndarray:
        return self._q_target

    @q_target.setter
    def q_target(self, target) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != self._q_target.shape:
            raise ValueError(
                f"Expected {self._q_target.shape} joint target, got {target.shape}"
            )
        self._q_target = target.copy()

    @property
    def applied_q_target(self) -> np.ndarray:
        """Most recently delivered target, after command latency [rad]."""
        return self._controller.q_target

    def get_q_target(self, name: str) -> float:
        return self._q_target[self._controller.dof_to_q_target[name]]

    def set_q_target(self, name: str, target: float) -> None:
        self._q_target[self._controller.dof_to_q_target[name]] = target

    def reset(self, qpos) -> None:
        controller = self._controller
        controller.model.reset()
        controller.reset(qpos)
        target = np.asarray(qpos)[controller.qpos_indexes].copy()
        self._q_target = target.copy()
        controller.q_target = target.copy()
        controller.last_ts = float(controller.mujoco_data.time)
        # STS3215's limiter state is normally initialized by load_log(), which
        # is deliberately absent in deployment. Seed from the measured joints.
        if controller.model.actuator_name == "sts3215":
            controller.model.actuator.q_target_smooth = target.copy()
        self.delay.reset(target)
        controller.mujoco_data.ctrl[controller.act_indexes] = 0.0

    def update(self) -> None:
        self._controller.q_target = self.delay.update(self._q_target)
        self._controller.update()


def load_scene(xml_path, variant, timestep=0.005, lag_steps=4):
    """Return ``(model, data, controller, settings)`` for one nominal variant.

    ``variant`` is ``"hd1910"`` (also ``"actual"``) or ``"original"``.
    HD1910 uses the public LuwuDynamics 1910 M6 identification JSON and the
    fixed midpoint voltage/drop settings from the HD1910 task. Original uses
    upstream XL330 M6 with the midpoint of upstream's battery distribution.
    Geometry is supplied independently by ``xml_path``; it is never replaced.
    Pass ``lag_steps=0`` when the caller already models command latency.
    """
    from bam.model import load_model
    from bam.mujoco import MujocoController

    if timestep <= 0:
        raise ValueError("timestep must be positive")
    # Validate early, before loading a potentially large mesh scene.
    PhysicsStepDelay(lag_steps)
    variant = str(variant).lower().replace("-", "").replace("_", "")
    if variant in {"hd1910", "actual"}:
        variant = "hd1910"
        bam_model = load_model(str(HD1910_JSON))
        kp_fw, vin, vin_min = 5.0, 7.7, 7.0
        parameter_source = str(HD1910_JSON)
    elif variant == "original":
        bam_model = load_model(motor_name="xl330", model="m6")
        kp_fw, vin, vin_min = 200.0, 7.35, 6.0
        parameter_source = "bam.params/xl330/m6.json"
    else:
        raise ValueError(f"Unknown Microduck variant: {variant!r}")

    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = None
    vin_drop_gain = 0.1
    force_limit = vin * bam_model.kt.value / bam_model.R.value

    spec = mujoco.MjSpec.from_file(str(Path(xml_path).expanduser().resolve()))
    # EntityCfg applies FULL_COLLISION after loading the XML in training.
    # Mirror its named collision selection and foot contact priority here.
    collision_names = []
    for geom in spec.geoms:
        if geom.name and geom.name.endswith("_collision"):
            geom.contype = 1
            geom.conaffinity = 1
            is_foot = geom.name in {"left_foot_collision", "right_foot_collision"}
            geom.condim = 3 if is_foot else 1
            if is_foot:
                geom.friction = [1.0, *np.asarray(geom.friction)[1:]]
                geom.priority = 1
            collision_names.append(geom.name)
    joints = {joint.name: joint for joint in spec.joints}
    actuator_names = []
    for actuator in spec.actuators:
        target = actuator.target
        joint_name = target.name if hasattr(target, "name") else str(target)
        if joint_name.startswith("passive_"):
            continue
        if joint_name not in joints:
            raise ValueError(f"Actuator {actuator.name!r} is not attached to a joint")
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-force_limit, force_limit)
        actuator.ctrllimited = False
        actuator.gear = [1.0, 0, 0, 0, 0, 0]
        actuator_names.append(actuator.name)
        joint = joints[joint_name]
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = 0.0
        joint.solref_friction = STIFF_SOLREF_FRICTION
        joint.solimp_friction = STIFF_SOLIMP_FRICTION

    if not actuator_names:
        raise ValueError("Scene has no controllable non-passive joint actuators")
    model = spec.compile()
    model.opt.timestep = float(timestep)
    data = mujoco.MjData(model)
    bam_controller = MujocoController(
        bam_model, actuator_names, model, data,
        vin_drop_gain=vin_drop_gain, vin_min=vin_min,
    )
    controller = ResetSafeBamController(bam_controller, lag_steps)
    settings = {
        "variant": variant,
        "xml_path": str(Path(xml_path).expanduser().resolve()),
        "bam_model": "m6",
        "actuator": bam_model.actuator_name,
        "parameter_source": parameter_source,
        "kp_fw": kp_fw,
        "vin": vin,
        "vin_drop_gain": vin_drop_gain,
        "vin_min": vin_min,
        "max_current": None,
        "timestep": float(timestep),
        "lag_steps": int(lag_steps),
        "lag_seconds": float(timestep * lag_steps),
        "training_lag_range_steps": [3, 6],
        "kt": float(bam_model.kt.value),
        "R": float(bam_model.R.value),
        "armature": float(bam_model.actuator.get_extra_inertia()),
        "force_limit_nm": float(force_limit),
        "actuator_names": actuator_names,
        "collision_override": "FULL_COLLISION: named *_collision; feet condim3 friction1 priority1; others condim1",
        "collision_names": collision_names,
        "evaluation_scope": (
            "Nominal CPU evaluation with fixed battery/drop/latency and no "
            "domain randomization; unnamed XML contact flags are preserved, "
            "so this is not an exact replay of the Warp training environment."
        ),
    }
    return model, data, controller, settings
