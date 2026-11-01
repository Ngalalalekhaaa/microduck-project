"""Strict CPU-only inference for current-based and encoder/IMU HOME policies.

All sensors use the existing calibrated joint frame. The legacy current policy
requires amperes signed along each positive model joint axis. No unknown training
bias is supplied or subtracted here. The hardware controller owns freshness,
limits, scheduling, and deciding whether a candidate target may be written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .policy import HOME, JOINT_NAMES, PolicyContractError


FRAME_SIZE = 49
HISTORY_LENGTH = 8
OBS_SIZE = FRAME_SIZE * HISTORY_LENGTH
ACTION_SIZE = 14
TASK_ID = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
POLICY_FAMILY = "hd1910_home_calibration_v1"
NO_CURRENT_TASK_ID = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
NO_CURRENT_POLICY_FAMILY = "hd1910_home_calibration_no_current_v2"
NO_CURRENT_FRAME_SIZE = 34
NO_CURRENT_OBS_SIZE = NO_CURRENT_FRAME_SIZE * HISTORY_LENGTH
FORCE_RATIO_TASK_ID = "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"
FORCE_RATIO_POLICY_FAMILY = "hd1910_home_force_ratio_no_current_v1"
FORCE_RATIO_FRAME_SIZE = 48
OBSERVATION_SCHEMA = (
    ("reported_joint_position_minus_home", 0, 14, "rad"),
    ("signed_current_fraction", 14, 14, "dimensionless"),
    ("sum_absolute_current", 28, 1, "A"),
    ("projected_gravity", 29, 3, "unit_vector"),
    ("body_gyro", 32, 3, "rad/s"),
    ("command_offset_from_home", 35, 14, "rad"),
)
NO_CURRENT_OBSERVATION_SCHEMA = (
    ("reported_joint_position_minus_home", 0, 14, "rad"),
    ("projected_gravity", 14, 3, "unit_vector"),
    ("body_gyro", 17, 3, "rad/s"),
    ("command_offset_from_home", 20, 14, "rad"),
)
FORCE_RATIO_OBSERVATION_SCHEMA = NO_CURRENT_OBSERVATION_SCHEMA + (
    ("target_joint_torque_fraction", 34, 14, "dimensionless"),
)
_TEXT_CONTRACT = {
    "task_id": TASK_ID,
    "policy_family": POLICY_FAMILY,
    "current_normalization": "I/max(sum(abs(I)),current_normalization_epsilon_a)",
    "current_sign": "positive_motor_torque_along_positive_model_joint_axis",
    "history_order": "oldest_to_newest_frame_major_first_sample_repeated_on_reset",
    "observation_normalization": "baked_into_onnx",
    "action_semantics": "command_offset=clip(command_offset+clip(output,-1,1)*delta_limit_rad,-offset_limit_rad,offset_limit_rad)",
    "deployment_target": "existing_calibrated_HOME+command_offset; do_not_subtract_unknown_training_bias",
    "load_signal": "synthetic_signed_phase_current_A=BAM_motor_torque/kt; hardware_mapping_not_validated",
    "role": "simulation_candidate; current walking runtime has a different observation/action contract",
}
_NO_CURRENT_TEXT_CONTRACT = {
    **{key: value for key, value in _TEXT_CONTRACT.items()
       if key not in ("current_normalization", "current_sign")},
    "task_id": NO_CURRENT_TASK_ID,
    "policy_family": NO_CURRENT_POLICY_FAMILY,
    "load_signal": "none",
    "current_input": "none",
    "role": "simulation_candidate; encoder_and_imu_only_home_correction",
}
_FORCE_RATIO_TEXT_CONTRACT = {
    **_NO_CURRENT_TEXT_CONTRACT,
    "task_id": FORCE_RATIO_TASK_ID,
    "policy_family": FORCE_RATIO_POLICY_FAMILY,
    "role": "simulation_candidate; encoder_and_imu_only_home_force_ratio_correction",
    "objective": "joint_torque_magnitude_ratio",
    "force_ratio_definition": "abs(qfrc_actuator_servo)/sum(abs(qfrc_actuator_servo))",
    "force_ratio_input": "fixed_reference_target_only; no_measured_force_or_current",
}


class HomeCalibrationContractError(PolicyContractError):
    """A HOME artifact, observation, or action is incompatible or invalid."""


def _finite(value, name: str, shape: tuple[int, ...], *, dtype=np.float32) -> np.ndarray:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HomeCalibrationContractError(f"{name} must contain numeric values") from exc
    if values.shape != shape or not np.all(np.isfinite(values)):
        raise HomeCalibrationContractError(f"{name} must be finite with shape {shape}")
    return values


def _json(value, name: str):
    if not isinstance(value, str):
        raise HomeCalibrationContractError(f"{name} must be JSON text")
    try:
        return json.loads(value)
    except (ValueError, TypeError) as exc:
        raise HomeCalibrationContractError(f"Invalid {name}") from exc


@dataclass(frozen=True)
class HomeCalibrationContract:
    task_id: str
    policy_family: str
    run_path: str
    joint_names: tuple[str, ...]
    home_angles_rad: tuple[float, ...]
    control_dt: float
    actor_frame_dim: int
    history_length: int
    actor_dim: int
    encoder_ticks: int
    current_normalization_epsilon_a: float | None
    delta_limit_rad: float
    offset_limit_rad: float
    settle_seconds: float
    target_force_ratio: tuple[float, ...] | None = None

    @property
    def requires_current(self) -> bool:
        return self.policy_family == POLICY_FAMILY


def validate_home_contract(metadata: Mapping) -> HomeCalibrationContract:
    """Validate complete ONNX metadata or its typed JSON contract sidecar.

    The exporter's legacy default_joint_pos CSV has three decimal places;
    exact HOME comes from the mandatory full-precision home_angles_rad_json.
    A 61D walking model or an undocumented HOME model is never accepted.
    """
    if not isinstance(metadata, Mapping):
        raise HomeCalibrationContractError("HOME contract must be a mapping")
    force_ratio = metadata.get("policy_family") == FORCE_RATIO_POLICY_FAMILY
    no_current = force_ratio or metadata.get("policy_family") == NO_CURRENT_POLICY_FAMILY
    text_contract = (_FORCE_RATIO_TEXT_CONTRACT if force_ratio else
                     _NO_CURRENT_TEXT_CONTRACT if no_current else _TEXT_CONTRACT)
    if no_current and any(key in metadata for key in (
            "current_normalization", "current_sign", "current_normalization_epsilon_a")):
        raise HomeCalibrationContractError("No-current HOME contract must not contain current-input fields")
    for key, expected in text_contract.items():
        if metadata.get(key) != expected:
            raise HomeCalibrationContractError(f"HOME contract {key} is missing or incompatible")
    run_path = metadata.get("run_path")
    if not isinstance(run_path, str) or not run_path.strip():
        raise HomeCalibrationContractError("HOME contract requires a nonempty run_path")
    joint_names = metadata.get("joint_names")
    if isinstance(joint_names, str):
        joint_names = joint_names.split(",")
    if not isinstance(joint_names, (list, tuple)) or tuple(joint_names) != JOINT_NAMES:
        raise HomeCalibrationContractError("HOME contract joint_names/order differs")

    home = _finite(_json(metadata.get("home_angles_rad_json"), "home_angles_rad_json"),
                   "HOME angles", (14,))
    if not np.array_equal(home, HOME.astype(np.float32)):
        raise HomeCalibrationContractError("HOME contract exact reference angles differ")
    default_home = metadata.get("default_joint_pos")
    rounded_csv = isinstance(default_home, str)
    if rounded_csv:
        default_home = default_home.split(",")
    default_home = _finite(default_home, "default_joint_pos", (14,), dtype=np.float64)
    tolerance = 0.000501 if rounded_csv else 5e-8
    if not np.allclose(default_home, home, atol=tolerance, rtol=0):
        raise HomeCalibrationContractError("default_joint_pos disagrees with exact HOME")

    observation_schema = (FORCE_RATIO_OBSERVATION_SCHEMA if force_ratio else
                          NO_CURRENT_OBSERVATION_SCHEMA if no_current else OBSERVATION_SCHEMA)
    expected_schema = [
        {"name": name, "start": start, "size": size, "unit": unit}
        for name, start, size, unit in observation_schema
    ]
    schema = _json(metadata.get("observation_schema_json"), "observation_schema_json")
    if (schema != expected_schema or not isinstance(schema, list)
            or any(type(item.get("start")) is not int or type(item.get("size")) is not int
                   for item in schema)):
        raise HomeCalibrationContractError("HOME observation schema/order/units differ")

    frame_size = 48 if force_ratio else 34 if no_current else 49
    expected_numbers = {
        "control_dt": 0.05, "actor_frame_dim": frame_size, "history_length": 8,
        "actor_dim": frame_size * 8, "encoder_ticks": 4096,
        "delta_limit_rad": math.radians(0.25), "offset_limit_rad": math.radians(8),
        "settle_seconds": 0.5,
    }
    if not no_current:
        expected_numbers["current_normalization_epsilon_a"] = 0.02
    parsed = {}
    for key, expected in expected_numbers.items():
        try:
            raw = metadata[key]
            if isinstance(raw, (bool, list, tuple, dict)):
                raise ValueError("not a scalar number")
            value = float(raw)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise HomeCalibrationContractError(f"HOME contract requires numeric {key}") from exc
        if not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12):
            raise HomeCalibrationContractError(f"HOME contract {key} differs from supported value {expected}")
        parsed[key] = int(value) if isinstance(expected, int) else value
    if no_current:
        parsed["current_normalization_epsilon_a"] = None
    target = None
    if force_ratio:
        target_values = _finite(
            _json(metadata.get("target_force_ratio_json"), "target_force_ratio_json"),
            "target_force_ratio", (14,), dtype=np.float64)
        if np.any(target_values < 0) or not math.isclose(float(target_values.sum()), 1., rel_tol=0, abs_tol=1e-6):
            raise HomeCalibrationContractError("Target force ratio must be nonnegative and sum to one")
        target = tuple(float(value) for value in target_values)
    return HomeCalibrationContract(
        task_id=text_contract["task_id"], policy_family=text_contract["policy_family"], run_path=run_path,
        joint_names=JOINT_NAMES, home_angles_rad=tuple(float(q) for q in home),
        target_force_ratio=target, **parsed,
    )


def build_home_frame49(
    q_reported_rad, current_joint_a, gravity, gyro_rad_s, command_offset_rad,
    *, contract: HomeCalibrationContract,
) -> np.ndarray:
    """Build a raw float32 sensor frame in the exact training slot order.

    Quantize the absolute *reported* joint angle before subtracting HOME, as
    in training. Current is already measured/converted upstream: do not add
    synthetic training noise, rectify it, or quantize it a second time.
    """
    if not contract.requires_current or contract.actor_frame_dim != FRAME_SIZE:
        raise HomeCalibrationContractError("49D current observation requires the v1 current policy contract")
    q = _finite(q_reported_rad, "q_reported_rad", (14,))
    current = _finite(current_joint_a, "current_joint_a", (14,))
    g = _finite(gravity, "gravity", (3,))
    gyro = _finite(gyro_rad_s, "gyro_rad_s", (3,))
    offset = _finite(command_offset_rad, "command_offset_rad", (14,))
    if not 0.95 <= float(np.linalg.norm(g)) <= 1.05:
        raise HomeCalibrationContractError("gravity must be a unit projected-gravity vector")
    if np.any(np.abs(offset) > np.float32(contract.offset_limit_rad)):
        raise HomeCalibrationContractError("command_offset_rad exceeds the trained bound")
    resolution = np.float32(math.tau / contract.encoder_ticks)
    with np.errstate(over="ignore", invalid="ignore"):
        position = np.rint(q / resolution) * resolution - np.asarray(contract.home_angles_rad, dtype=np.float32)
        total = np.sum(np.abs(current), dtype=np.float32)
        ratio = current / np.maximum(total, np.float32(contract.current_normalization_epsilon_a))
        frame = np.concatenate((position, ratio, np.asarray([total], dtype=np.float32), g, gyro, offset))
    return _finite(frame, "HOME frame", (FRAME_SIZE,)).copy()


def build_home_frame34(
    q_reported_rad, gravity, gyro_rad_s, command_offset_rad,
    *, contract: HomeCalibrationContract,
) -> np.ndarray:
    """Encoder/IMU-only frame; neither current nor load enters the actor."""
    if contract.policy_family != NO_CURRENT_POLICY_FAMILY or contract.actor_frame_dim != NO_CURRENT_FRAME_SIZE:
        raise HomeCalibrationContractError("34D observation requires the v2 no-current policy contract")
    return _build_kinematic_frame(q_reported_rad, gravity, gyro_rad_s, command_offset_rad, contract=contract)


def _build_kinematic_frame(q_reported_rad, gravity, gyro_rad_s, command_offset_rad, *, contract):
    q = _finite(q_reported_rad, "q_reported_rad", (14,))
    g = _finite(gravity, "gravity", (3,))
    gyro = _finite(gyro_rad_s, "gyro_rad_s", (3,))
    offset = _finite(command_offset_rad, "command_offset_rad", (14,))
    if not 0.95 <= float(np.linalg.norm(g)) <= 1.05:
        raise HomeCalibrationContractError("gravity must be a unit projected-gravity vector")
    if np.any(np.abs(offset) > np.float32(contract.offset_limit_rad)):
        raise HomeCalibrationContractError("command_offset_rad exceeds the trained bound")
    resolution = np.float32(math.tau / contract.encoder_ticks)
    with np.errstate(over="ignore", invalid="ignore"):
        position = np.rint(q / resolution) * resolution - np.asarray(contract.home_angles_rad, dtype=np.float32)
        frame = np.concatenate((position, g, gyro, offset))
    return _finite(frame, "HOME frame", (NO_CURRENT_FRAME_SIZE,)).copy()


def build_home_frame48(
    q_reported_rad, gravity, gyro_rad_s, command_offset_rad,
    *, contract: HomeCalibrationContract,
) -> np.ndarray:
    """Kinematics plus a fixed reference target, with no measured force input."""
    if contract.policy_family != FORCE_RATIO_POLICY_FAMILY or contract.actor_frame_dim != FORCE_RATIO_FRAME_SIZE:
        raise HomeCalibrationContractError("48D observation requires the no-current force-ratio policy contract")
    sensors = _build_kinematic_frame(q_reported_rad, gravity, gyro_rad_s, command_offset_rad, contract=contract)
    target = _finite(contract.target_force_ratio, "target_force_ratio", (14,))
    if np.any(target < 0) or not math.isclose(float(target.sum()), 1., rel_tol=0, abs_tol=1e-6):
        raise HomeCalibrationContractError("Target force ratio must be nonnegative and sum to one")
    return np.concatenate((sensors, target)).astype(np.float32, copy=False)


class HomeObservationHistory:
    """Eight whole frames, oldest first; first sample repeats across the stack."""

    def __init__(self, contract: HomeCalibrationContract):
        self.contract = contract
        self._frames: np.ndarray | None = None

    def reset(self) -> None:
        self._frames = None

    def push(self, frame) -> np.ndarray:
        value = _finite(frame, "HOME frame", (self.contract.actor_frame_dim,))
        if self._frames is None:
            self._frames = np.repeat(value[None, :], self.contract.history_length, axis=0)
        else:
            self._frames[:-1] = self._frames[1:].copy()
            self._frames[-1] = value
        return self._frames.reshape(self.contract.actor_dim).copy()


class HomeCommandAccumulator:
    """Pure candidate command state; no hardware writes or zero-table edits.

    After reset, calls 1 through 10 hold HOME and call 11 may first change it,
    matching training's pre-increment control_steps test at dt=0.05 seconds.
    The hardware caller must run at this cadence and stop on missed deadlines.
    """

    def __init__(self, contract: HomeCalibrationContract):
        self.contract = contract
        self.reset()

    def reset(self) -> None:
        self._command_offset = np.zeros(ACTION_SIZE, dtype=np.float32)
        self._last_delta = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.control_steps = 0

    @property
    def command_offset(self) -> np.ndarray:
        return self._command_offset.copy()

    @property
    def last_delta(self) -> np.ndarray:
        return self._last_delta.copy()

    @property
    def is_settling(self) -> bool:
        elapsed = np.float32(self.control_steps) * np.float32(self.contract.control_dt)
        return bool(elapsed < np.float32(self.contract.settle_seconds - 1e-8))

    def advance(self, raw_action) -> np.ndarray:
        action = _finite(raw_action, "raw HOME action", (ACTION_SIZE,))
        previous = self._command_offset.copy()
        if not self.is_settling:
            delta = np.clip(action, -1, 1) * np.float32(self.contract.delta_limit_rad)
            limit = np.float32(self.contract.offset_limit_rad)
            self._command_offset = np.clip(previous + delta, -limit, limit)
        self._last_delta = self._command_offset - previous
        self.control_steps += 1
        # The real zero_ticks mapping already defines the reported frame.
        # There is deliberately no training-bias subtraction here.
        return np.asarray(self.contract.home_angles_rad, dtype=np.float32) + self._command_offset


class HomeCalibrationPolicy:
    """Stateless ONNX inference with mandatory metadata and CPU-only execution."""

    def __init__(self, model_path: str | Path, *, contract_path: str | Path | None = None):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        model_bytes = self.model_path.read_bytes()
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise HomeCalibrationContractError("HOME inference must use CPUExecutionProvider only")
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.contract = validate_home_contract(self.metadata)
        if contract_path is not None:
            try:
                sidecar = json.loads(Path(contract_path).read_text(encoding="utf-8"))
            except (ValueError, TypeError) as exc:
                raise HomeCalibrationContractError("Invalid HOME sidecar JSON") from exc
            if validate_home_contract(sidecar) != self.contract:
                raise HomeCalibrationContractError("HOME sidecar disagrees with embedded ONNX contract")
        for nodes, name, width in (
            (self.session.get_inputs(), "obs", self.contract.actor_dim),
            (self.session.get_outputs(), "actions", ACTION_SIZE),
        ):
            if len(nodes) != 1:
                raise HomeCalibrationContractError(f"HOME ONNX requires exactly one {name} tensor")
            node = nodes[0]
            if node.name != name or node.type != "tensor(float)" or node.shape != [1, width]:
                raise HomeCalibrationContractError(f"HOME ONNX {name} must be float32[1,{width}]")

    def infer(self, observation) -> np.ndarray:
        size = self.contract.actor_dim
        batch = _finite(observation, "HOME observation", (size,)).reshape(1, size)
        result = self.session.run(["actions"], {"obs": batch})[0]
        if np.asarray(result).dtype != np.float32:
            raise HomeCalibrationContractError("HOME ONNX output must have dtype float32")
        return _finite(result, "HOME ONNX actions", (1, ACTION_SIZE))[0].copy()

    def inspect(self) -> dict:
        """Load inspection plus one finite zero-input CPU inference; no motion."""
        action = self.infer(np.zeros(self.contract.actor_dim, dtype=np.float32))
        return {
            "model": str(self.model_path), "model_sha256": self.model_sha256,
            "providers": self.session.get_providers(),
            "input_shape": [1, self.contract.actor_dim], "output_shape": [1, ACTION_SIZE],
            "requires_current_input": self.contract.requires_current,
            "contract": asdict(self.contract), "zero_input_action": action.tolist(),
        }
