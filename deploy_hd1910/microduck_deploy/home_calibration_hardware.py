"""Hardware boundary for experimental HOME correction; no automatic calibration.

The deployed driver decodes register 69 as signed bit-15 sign/magnitude at
6.5 mA/count. This agrees with the repository's HD1910 register table, but is
not evidence that this robot's current scale or model-axis polarity was
measured. Keep the original word and require an explicit per-joint mapping.
Never infer missing current polarity from load, velocity, or position error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numbers
import struct
import time

import numpy as np

from .config import require_calibration
from .imu import ImuError
from .policy import JOINT_NAMES
from .servo import ServoBus, ServoConfigurationError, ServoError, ServoFeedback, ServoStatusError


class HomeCalibrationHardwareError(ValueError):
    """A measurement or explicit hardware contract cannot support correction."""


class HardwareSafetyError(HomeCalibrationHardwareError):
    """Explicit thermal/current fault; an already-active owner may release torque."""

    release_torque = True


@dataclass(frozen=True)
class RawCurrentFeedback(ServoFeedback):
    current_register_raw: int


class _RawCurrentBus(ServoBus):
    @staticmethod
    def _feedback(servo_id: int, raw: bytes) -> RawCurrentFeedback:
        feedback = ServoBus._feedback(servo_id, raw)
        return RawCurrentFeedback(
            **asdict(feedback), current_register_raw=struct.unpack_from("<H", raw, 13)[0]
        )


class ReadOnlyServoBus(_RawCurrentBus):
    """Allow protocol reads, but reject every servo register-writing operation."""

    def _sync_ram(self, address, size, values):
        raise ServoConfigurationError("HOME monitor bus is read-only; servo writes are forbidden")


class CalibrationServoBus(_RawCurrentBus):
    """Only write single-turn goals or turn torque OFF on already-held joints.

    No torque enable, RAM gain changes, EEPROM, or hardware zero calibration.
    The caller must validate its sensor mapping and held-HOME preconditions
    before sending goals; the bus never starts or prepares a standing robot.
    """

    def _sync_ram(self, address, size, values):
        if (address, size) == (40, 1):
            if any(bytes(value) != b"\x00" for value in values.values()):
                raise ServoConfigurationError("HOME correction permits torque OFF only")
        elif (address, size) == (42, 2):
            for value in values.values():
                value = bytes(value)
                if len(value) != 2 or struct.unpack("<H", value)[0] > 4095:
                    raise ServoConfigurationError("HOME correction requires single-turn position goals")
        else:
            raise ServoConfigurationError("HOME correction permits only goal 42 and torque-OFF 40 writes")
        super()._sync_ram(address, size, values)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise HomeCalibrationHardwareError(f"{name} must be a finite number")
    return float(value)


def _joints(cfg):
    joints = cfg.get("joints", [])
    if [joint.get("name") for joint in joints] != list(JOINT_NAMES):
        raise HomeCalibrationHardwareError("Expected the complete ordered 14-joint model mapping")
    ids = [joint.get("id") for joint in joints]
    if any(type(sid) is not int or not 0 <= sid <= 253 for sid in ids) or len(set(ids)) != 14:
        raise HomeCalibrationHardwareError("Servo IDs must be 14 distinct integers in 0..253")
    for joint in joints:
        if type(joint.get("direction")) is not int or joint["direction"] not in (-1, 1):
            raise HomeCalibrationHardwareError(f"{joint['name']}: joint direction must be explicitly +/-1")
    return joints


@dataclass(frozen=True)
class CurrentJointMapping:
    name: str
    id: int
    current_sign: int
    current_scale_a: float
    current_zero_raw: float
    verified: bool


@dataclass(frozen=True)
class CurrentMapping:
    """I_model = current_sign * current_scale_a * (decoded_word-current_zero_raw).

    current_sign already includes conversion into the positive MODEL joint
    axis. Encoder ``direction`` is NOT multiplied again. ``verified`` is an
    explicit configuration assertion; software does not perform or certify the
    electrical measurement. Unsigned magnitude data cannot supply signed input.
    """

    encoding: str
    joints: tuple[CurrentJointMapping, ...]

    @classmethod
    def from_dict(cls, spec, cfg, *, motion=False):
        joints = _joints(cfg)
        if not isinstance(spec, dict) or spec.get("schema") != 1:
            raise HomeCalibrationHardwareError("Current mapping requires schema 1")
        encoding = spec.get("encoding")
        if encoding not in ("sign_magnitude15", "twos_complement16", "unsigned16"):
            raise HomeCalibrationHardwareError("Current encoding must be explicit; load-derived signs are unsupported")
        rows = spec.get("joints", [])
        if not isinstance(rows, list) or len(rows) != 14:
            raise HomeCalibrationHardwareError("Current mapping requires all 14 joint entries")
        parsed = []
        for row, joint in zip(rows, joints):
            if (not isinstance(row, dict) or row.get("name") != joint["name"]
                    or type(row.get("id")) is not int or row["id"] != joint["id"]):
                raise HomeCalibrationHardwareError("Current mapping names/IDs must match this robot in model order")
            sign = row.get("current_sign")
            if type(sign) is not int or sign not in (-1, 1):
                raise HomeCalibrationHardwareError(f"{joint['name']}: current_sign must be explicitly +/-1")
            scale = _finite(row.get("current_scale_a"), "current_scale_a")
            if scale <= 0:
                raise HomeCalibrationHardwareError("current_scale_a must be positive amperes per decoded raw count")
            zero = _finite(row.get("current_zero_raw", 0.), "current_zero_raw")
            verified = row.get("verified")
            if type(verified) is not bool:
                raise HomeCalibrationHardwareError("Every current mapping entry requires an explicit verified boolean")
            parsed.append(CurrentJointMapping(joint["name"], joint["id"], sign, scale, zero, verified))
        result = cls(encoding, tuple(parsed))
        if motion:
            result.require_motion()
        return result

    @property
    def motion_ready(self):
        return self.encoding != "unsigned16" and all(joint.verified for joint in self.joints)

    def require_motion(self):
        if not self.motion_ready:
            raise HomeCalibrationHardwareError(
                "Motion requires verified signed current encoding, scale and model-axis polarity for all 14 joints; "
                "unsigned/load-derived current signs cannot be used"
            )

    def to_dict(self):
        return {"schema": 1, "encoding": self.encoding,
                "joints": [asdict(joint) for joint in self.joints]}

    def convert(self, raw_by_id):
        values = []
        for joint in self.joints:
            raw = raw_by_id.get(joint.id)
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral) or not 0 <= raw <= 65535:
                raise HomeCalibrationHardwareError(f"Servo {joint.id}: original current register word is unavailable/invalid")
            if self.encoding == "sign_magnitude15":
                decoded = -(raw & 0x7fff) if raw & 0x8000 else raw
            elif self.encoding == "twos_complement16":
                decoded = raw - 65536 if raw & 0x8000 else raw
            else:
                decoded = raw
            values.append(joint.current_sign * joint.current_scale_a * (decoded - joint.current_zero_raw))
        current = np.asarray(values, dtype=np.float64)
        if not np.isfinite(current).all():
            raise HomeCalibrationHardwareError("Mapped current contains nonfinite values")
        return current


def current_mapping_template(cfg):
    """An explicitly UNVERIFIED parser-based starting point, not a calibration."""
    return {
        "schema": 1, "encoding": "sign_magnitude15",
        "joints": [{"name": joint["name"], "id": joint["id"],
                    "current_sign": joint["direction"], "current_scale_a": .0065,
                    "current_zero_raw": 0., "verified": False}
                   for joint in _joints(cfg)],
    }


def validate_hold_configuration(configurations, cfg, *, expected_p=5):
    """Check an already-held pose without enabling torque or changing gains."""
    if type(expected_p) is not int or expected_p not in (5, 8):
        raise HomeCalibrationHardwareError("Expected RAM P must be 5 or 8")
    ids = [joint["id"] for joint in _joints(cfg)]
    if set(configurations) != set(ids):
        raise HomeCalibrationHardwareError("Held-HOME configuration must cover exactly the 14 body servos")
    for sid in ids:
        item = configurations[sid]
        if (item.get("register_profile") != "hls_2" or item.get("torque") != 1
                or item.get("mode") != 4 or item.get("angle_resolution") != 1):
            raise HomeCalibrationHardwareError(f"Servo {sid}: require already ON, HLS_2 mode 4, resolution 1")
        if item.get("ram_gains") != {"p": expected_p, "d": 0, "i": 0}:
            raise HomeCalibrationHardwareError(
                f"Servo {sid}: require existing RAM P={expected_p}/D=0/I=0; gains were not changed. "
                "请先托稳机器人，执行 relax 卸力，再以相同 --ram-p 重新回 HOME；不会自动切换持位增益")
        lower, upper = item.get("position_min"), item.get("position_max")
        if (type(lower) is not int or type(upper) is not int
                or not 0 <= lower < upper <= 4095):
            raise HomeCalibrationHardwareError(f"Servo {sid}: invalid single-turn hardware limits")


@dataclass(frozen=True)
class HardwareSample:
    q: np.ndarray
    current_a: np.ndarray | None
    gyro: np.ndarray
    gravity: np.ndarray
    diagnostics: dict
    feedback: dict[int, ServoFeedback]


def read_hardware_sample(
    bus, imu, cfg, *, current_mapping=None, sync=True, now=None,
    max_age_s=.10, max_skew_s=.05, imu_max_age_s=None,
):
    """Read one servo sweep and the latest already-body-frame IMU sample.

    With no current mapping, monitor mode returns raw diagnostic current and
    ``current_a=None``. An unverified mapping may be inspected, but must pass
    ``require_motion()`` before correction. No re-quantization of calibrated
    encoder angles and no second IMU axis rotation are applied here.

    Encoder/IMU-only policies also use ``current_a=None`` during correction.
    Their actor does not consume current; finite driver current, voltage,
    temperature, status and freshness remain mandatory safety telemetry.
    """
    joints = _joints(cfg)
    require_calibration(cfg, imu=False)
    maximum_imu_age = cfg["control"]["imu_max_age_s"] if imu_max_age_s is None else imu_max_age_s
    for name, value in (("max_age_s", max_age_s), ("max_skew_s", max_skew_s),
                        ("imu_max_age_s", maximum_imu_age)):
        if _finite(value, name) <= 0:
            raise HomeCalibrationHardwareError(f"{name} must be positive")
    ids = [joint["id"] for joint in joints]
    if current_mapping is not None:
        if not isinstance(current_mapping, CurrentMapping):
            raise HomeCalibrationHardwareError("current_mapping must be an explicitly parsed CurrentMapping")
        if [(item.name, item.id) for item in current_mapping.joints] != [(j["name"], j["id"]) for j in joints]:
            raise HomeCalibrationHardwareError("Current mapping belongs to a different robot joint mapping")
    try:
        feedback = bus.read_feedback_many(ids, sync=sync)
    except ServoStatusError as exc:
        error_type = HardwareSafetyError if exc.status & ((1 << 2) | (1 << 3)) else HomeCalibrationHardwareError
        raise error_type(str(exc)) from exc
    except ServoError as exc:
        raise HomeCalibrationHardwareError(str(exc)) from exc
    if set(feedback) != set(ids):
        raise HomeCalibrationHardwareError("Servo feedback does not contain exactly the requested 14 IDs")
    try:
        state = imu.latest(maximum_imu_age)
    except ImuError as exc:
        raise HomeCalibrationHardwareError(str(exc)) from exc
    measured_at = time.monotonic() if now is None else _finite(now, "now")
    voltage_bounds = cfg["control"]["servo_voltage_range"]
    if len(voltage_bounds) != 2:
        raise HomeCalibrationHardwareError("servo_voltage_range must have two bounds")
    lower, upper = (_finite(value, "servo_voltage_range") for value in voltage_bounds)
    if not 0 < lower < upper:
        raise HomeCalibrationHardwareError("servo_voltage_range must be positive and ordered")
    max_temperature = _finite(cfg["control"]["servo_max_temperature_c"], "servo_max_temperature_c")
    positions, ages, timestamps, raw_current, driver_current = [], [], [], {}, {}
    for joint in joints:
        sid = joint["id"]
        value = feedback[sid]
        if value.servo_id != sid:
            raise HomeCalibrationHardwareError(f"Servo {sid}: mismatched feedback ID")
        if value.status != 0:
            error_type = HardwareSafetyError if value.status & ((1 << 2) | (1 << 3)) else HomeCalibrationHardwareError
            raise error_type(f"Servo {sid}: device fault status {value.status}")
        tick = value.position_ticks
        if isinstance(tick, bool) or not isinstance(tick, numbers.Integral) or not 0 <= tick <= 4095:
            raise HomeCalibrationHardwareError(f"Servo {sid}: position is not a single-turn encoder count")
        voltage = _finite(value.voltage_v, f"Servo {sid} voltage")
        temperature = _finite(value.temperature_c, f"Servo {sid} temperature")
        if temperature >= max_temperature:
            raise HardwareSafetyError(f"Servo {sid}: temperature outside configured operating bounds")
        if not lower <= voltage <= upper:
            raise HomeCalibrationHardwareError(f"Servo {sid}: voltage outside configured operating bounds")
        timestamp = _finite(value.received_at, f"Servo {sid} timestamp")
        age = measured_at - timestamp
        if timestamp <= 0 or not 0 <= age <= max_age_s:
            raise HomeCalibrationHardwareError(f"Servo {sid}: stale or future-dated feedback ({age:.4f}s)")
        zero = _finite(joint["zero_tick"], f"Servo {sid} zero_tick")
        positions.append(joint["direction"] * (tick - zero) * math.tau / 4096)
        ages.append(age)
        timestamps.append(timestamp)
        raw_current[sid] = getattr(value, "current_register_raw", None)
        driver_current[sid] = _finite(value.current_ma, f"Servo {sid} diagnostic current")
    skew = max(timestamps) - min(timestamps)
    if skew > max_skew_s:
        raise HomeCalibrationHardwareError(f"Servo feedback sweep is too wide ({skew:.4f}s)")
    imu_timestamp = _finite(state.timestamp, "IMU timestamp")
    imu_age = measured_at - imu_timestamp
    if imu_timestamp <= 0 or not 0 <= imu_age <= maximum_imu_age:
        raise HomeCalibrationHardwareError(f"IMU sample is stale or future-dated ({imu_age:.4f}s)")
    gyro = np.asarray(state.base_ang_vel, dtype=np.float64)
    gravity = np.asarray(state.projected_gravity, dtype=np.float64)
    if (gyro.shape != (3,) or gravity.shape != (3,)
            or not np.isfinite(gyro).all() or not np.isfinite(gravity).all()
            or not .95 <= np.linalg.norm(gravity) <= 1.05):
        raise HomeCalibrationHardwareError("IMU must provide finite body gyro and approximately unit world-down gravity")
    current = None if current_mapping is None else current_mapping.convert(raw_current)
    return HardwareSample(
        q=np.asarray(positions), current_a=current, gyro=gyro.copy(), gravity=gravity.copy(),
        diagnostics={
            "current_mapping_verified": bool(current_mapping is not None and current_mapping.motion_ready),
            "current_mapping_status": ("missing" if current_mapping is None else
                                       ("verified_by_configuration" if current_mapping.motion_ready else "unverified")),
            "current_encoding": None if current_mapping is None else current_mapping.encoding,
            "current_register_raw_by_id": {str(sid): raw for sid, raw in raw_current.items()},
            "driver_current_ma_by_id": {str(sid): value for sid, value in driver_current.items()},
            "feedback_ages_s": ages, "feedback_skew_s": skew, "imu_age_s": imu_age,
            "imu_mount_verified": cfg.get("imu_mount_verified") is True,
            "received_at": measured_at,
            "note": "Current verification flags describe supplied configuration, not a calibration performed by this program.",
        },
        feedback=dict(feedback),
    )
