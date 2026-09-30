"""CPU policy contract and calibrated HD-1910 joint conversions.

Contract checked against the exported HD flat/backlash model_3999 policies,
microduck_rl_hd1910's HOME_FRAME/robot_walk.xml, and duck-control/src/obs.rs.
The ONNX graph already contains observation normalization. Feed raw SI units.
No hardware access, policy action clipping, smoothing, or servo zero defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

OBS_SIZE = 61
ACTION_SIZE = 14
CONTROL_HZ = 50
CONTROL_DT = 1.0 / CONTROL_HZ
TICKS_PER_REV = 4096
RADIANS_PER_TICK = 2.0 * np.pi / TICKS_PER_REV
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "2026-09-24_20-26-30_xgoduck.onnx"

JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
# Official alpha ID mapping, with mouth ID 34 excluded. This is only a reference:
# physically identify and confirm every ID on this robot before enabling torque.
DEFAULT_SERVO_IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14)
SERVO_IDS = DEFAULT_SERVO_IDS

# Public XgoDuck HOME, consistent with this model's rounded metadata.
# Full user training configuration was not supplied; exact 24-degree values
# follow upstream xgoduck_constants.py. Physical Microduck calibration is retained.
HOME = np.array([
    0.0, -0.0873, -np.deg2rad(24), -0.0049, np.deg2rad(24), 0.3491, 0.3491, 0.0, 0.0,
    0.0, 0.0873, np.deg2rad(24), 0.0049, -np.deg2rad(24),
], dtype=np.float64)
HOME.setflags(write=False)

# Exact FK reference for the training robot_walk.xml (SHA-256 in docs/直腿参考几何核对.json).
# At q=0 the hip-to-knee offset is (-35.7771, *, -22) mm.
# Equal pitch/knee rotations put H/K/A on one vertical line in side projection.
STRAIGHT_ANGLE_RAD = float(np.arctan2(35.7771, 22.0))
STRAIGHT_REFERENCE = np.array([
    0, 0, -STRAIGHT_ANGLE_RAD, -STRAIGHT_ANGLE_RAD, 0, 0, 0, 0, 0,
    0, 0, STRAIGHT_ANGLE_RAD, STRAIGHT_ANGLE_RAD, 0,
], dtype=np.float64)
STRAIGHT_REFERENCE.setflags(write=False)

# Model anatomical limits in radians. These are not measured hardware limits and
# are not applied to policy outputs unless explicitly passed to action_to_targets.
# Training allows target overshoot and penalizes joint positions near their limits.
JOINT_LIMITS = np.deg2rad(np.array([
    [-25.0, 30.0], [-22.0, 22.0], [-90.0, 90.0], [-90.0, 90.0], [-90.0, 90.0],
    [-90.0, 60.0], [-90.0, 90.0], [-170.0, 170.0], [-25.0, 25.0],
    [-30.0, 25.0], [-22.0, 22.0], [-90.0, 90.0], [-90.0, 90.0], [-90.0, 90.0],
], dtype=np.float64))
JOINT_LIMITS.setflags(write=False)

# Joint positive rotation axes in the trunk frame WHEN ALL MODEL q == 0.
# Verified with mj_forward(robot_walk.xml). At other poses upstream joints rotate
# these axes. All XML joint-local axes are +Z, but their body frames differ.
# Trunk X forward, Y left, Z up; positive q follows the right-hand rule.
JOINT_POSITIVE_AXES_AT_ZERO = (
    (0, 0, -1), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 1, 0),
    (0, -1, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0),
    (0, 0, -1), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, -1, 0),
)

OBSERVATION_NAMES = (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions",
    "command", "head_command", "body_command",
)
COMMAND_NAMES = ("twist", "head_pose", "body_pose")


class PolicyContractError(ValueError):
    """A value or model does not satisfy the trained policy's contract."""


def _finite(value, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PolicyContractError(f"{name} must contain numbers") from exc
    if shape is not None and out.shape != shape:
        raise PolicyContractError(f"{name} must have shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise PolicyContractError(f"{name} contains non-finite values")
    return out


def _calibrated_arrays(value, zero_ticks, direction, name: str):
    value = _finite(value, name)
    zero = _finite(zero_ticks, "zero_ticks")
    sign = _finite(direction, "direction")
    if not np.all((sign == 1) | (sign == -1)):
        raise PolicyContractError("direction must be +1 or -1 for every joint")
    # Scalars can apply to a vector; reject accidental row/column broadcasting.
    for label, item in (("zero_ticks", zero), ("direction", sign)):
        if item.shape not in ((), value.shape):
            raise PolicyContractError(f"{label} must be scalar or match {name}'s shape")
    return value, zero, sign


def ticks_to_q(ticks, zero_ticks, direction) -> np.ndarray:
    """Single-turn raw counts -> absolute model q in radians.

    q = direction * (ticks - zero_ticks) * 2*pi/4096. Both calibration arguments
    are mandatory. zero_ticks may be fractional after capture at known HOME.
    No modulo wrapping; a crossing of the single-turn boundary must be rejected
    by the calling controller's continuity checks, not interpreted as motion.
    """
    ticks, zero, sign = _calibrated_arrays(ticks, zero_ticks, direction, "ticks")
    if np.any((ticks < 0) | (ticks > 4095) | (ticks != np.rint(ticks))):
        raise PolicyContractError("ticks must be integer encoder counts in [0, 4095]")
    return sign * (ticks - zero) * RADIANS_PER_TICK


def q_to_ticks(q, zero_ticks, direction) -> np.ndarray:
    """Absolute model q -> nearest integer single-turn count, never wrap or clip."""
    q, zero, sign = _calibrated_arrays(q, zero_ticks, direction, "q")
    with np.errstate(over="ignore", invalid="ignore"):
        ticks = np.rint(zero + sign * q / RADIANS_PER_TICK)
    if not np.all(np.isfinite(ticks)) or np.any((ticks < 0) | (ticks > 4095)):
        raise PolicyContractError("target is outside the calibrated single-turn encoder range")
    return ticks.astype(np.int64)


def policy_calibration_reference(reference: str = "home") -> np.ndarray:
    """Return the model angles for a PHYSICALLY IDENTIFIED calibration posture.

    'home' is the bent training stance. 'zero' is the MJCF q=0 geometry, not HOME
    and not servo center. Capturing a known posture gives:
      zero_ticks = ticks_ref - direction * q_ref * 4096/(2*pi).
    A capture cannot establish direction or whether the physical pose is correct.
    """
    if reference == "home":
        return HOME.copy()
    if reference == "zero":
        return np.zeros(ACTION_SIZE, dtype=np.float64)
    if reference == "straight":
        return STRAIGHT_REFERENCE.copy()
    raise PolicyContractError("calibration reference must be 'home' or 'zero'")


def build_obs61(
    gyro, gravity, q, qd, previous_action,
    twist=(0.0, 0.0, 0.0), head=(0.0, 0.0, 0.0, 0.0),
    body=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build raw float32[61]; no normalization, rescaling, noise or filtering.

    0:3   gyro in trunk axes, rad/s (X forward, Y left, Z up).
    3:6   projected gravity in SAME axes, unit vector; upright [0, 0, -1].
    6:20  q - exact HOME, rad, ordered as JOINT_NAMES.
    20:34 qd, rad/s, same order.
    34:48 previous raw policy action; initialize to zero after control reset.
    48:51 twist [forward m/s, left m/s, yaw rad/s].
    51:55 head offsets [neck_pitch, head_pitch, head_yaw, head_roll], rad.
    55:61 body offsets [x, y, z, roll, pitch, yaw], metres/radians.

    Current walking use supplies zero head/body offsets. Body x/y/yaw are
    unbound in the official runtime and remain zero. Do not add head/body
    commands again to the returned targets. Use estimated gravity, not raw
    accelerometer m/s^2. The IMU adapter owns its mounting transform.
    """
    g = _finite(gravity, "gravity", (3,))
    if not 0.95 <= np.linalg.norm(g) <= 1.05:
        raise PolicyContractError("gravity must be a unit projected-gravity vector")
    parts = (
        _finite(gyro, "gyro", (3,)), g,
        _finite(q, "q", (14,)) - HOME,
        _finite(qd, "qd", (14,)),
        _finite(previous_action, "previous_action", (14,)),
        _finite(twist, "twist", (3,)), _finite(head, "head", (4,)),
        _finite(body, "body", (6,)),
    )
    with np.errstate(over="ignore", invalid="ignore"):
        obs = np.concatenate(parts).astype(np.float32)
    _finite(obs, "float32 observation", (OBS_SIZE,))
    return obs


def action_to_targets(action, *, limits=None) -> np.ndarray:
    """q_target = HOME + raw action (scale 1 rad/action unit).

    Training has neither action clipping nor target filtering. This function
    preserves the raw output and does not update action history. Optionally
    pass measured conservative limits (14,2) to fail on any unsafe target.
    The caller must stop control on failure, not replace the action with HOME.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        target = HOME + _finite(action, "action", (ACTION_SIZE,))
    _finite(target, "target", (ACTION_SIZE,))
    if limits is not None:
        bounds = _finite(limits, "limits", (ACTION_SIZE, 2))
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise PolicyContractError("each lower joint limit must be below its upper limit")
        outside = np.flatnonzero((target < bounds[:, 0]) | (target > bounds[:, 1]))
        if len(outside):
            names = ", ".join(JOINT_NAMES[i] for i in outside)
            raise PolicyContractError(f"target outside configured joint limits: {names}")
    return target


def validate_metadata(metadata: Mapping[str, str]) -> None:
    """Refuse reordered, differently scaled or incompatible observation models."""
    for key, expected in (
        ("joint_names", JOINT_NAMES), ("observation_names", OBSERVATION_NAMES),
        ("command_names", COMMAND_NAMES),
    ):
        actual = tuple(part.strip() for part in metadata.get(key, "").split(","))
        if actual != expected:
            raise PolicyContractError(f"ONNX {key} does not match the HD-1910 policy contract")
    try:
        home = _finite([float(v) for v in metadata["default_joint_pos"].split(",")],
                       "ONNX default_joint_pos", (ACTION_SIZE,))
        scale = _finite([float(v) for v in metadata["action_scale"].split(",")],
                        "ONNX action_scale")
    except (KeyError, ValueError) as exc:
        raise PolicyContractError(f"invalid ONNX action/HOME metadata: {exc}") from exc
    # Exporter list_to_csv_str rounds numbers to 3 decimals. Never replace the
    # precise training HOME with this rounded metadata.
    if not np.allclose(home, HOME, rtol=0.0, atol=0.000501):
        raise PolicyContractError("ONNX HOME differs from the trained HOME pose")
    if scale.shape not in ((1,), (ACTION_SIZE,)) or not np.all(scale == 1.0):
        raise PolicyContractError("ONNX action scale must be 1 for every joint")


class OnnxPolicy:
    """Stateless, single-observation CPU inference for the bundled HD policies.

    No automatic previous_action state: the controller stores infer()'s raw
    result only after its safety checks and successful command write. A reset
    starts history at zero. Only numpy and onnxruntime are needed on the robot.
    """

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        validate_metadata(self.metadata)
        for items, name, width in (
            (self.session.get_inputs(), "obs", OBS_SIZE),
            (self.session.get_outputs(), "actions", ACTION_SIZE),
        ):
            if len(items) != 1:
                raise PolicyContractError(f"ONNX must have exactly one {name} tensor")
            node = items[0]
            if node.name != name or node.type != "tensor(float)" or node.shape != [1, width]:
                raise PolicyContractError(
                    f"ONNX {name} must be float32[1,{width}], got "
                    f"{node.name} {node.type} {node.shape}"
                )

    def infer(self, obs) -> np.ndarray:
        """Return raw float32[14], refusing non-finite input/output and bad shapes."""
        values = _finite(obs, "observation", (OBS_SIZE,))
        with np.errstate(over="ignore", invalid="ignore"):
            batch = values.astype(np.float32).reshape(1, OBS_SIZE)
        _finite(batch, "float32 observation", (1, OBS_SIZE))
        result = self.session.run(["actions"], {"obs": batch})[0]
        _finite(result, "ONNX actions", (1, ACTION_SIZE))
        if np.asarray(result).dtype != np.float32:
            raise PolicyContractError("ONNX actions must have dtype float32")
        return result[0].copy()
