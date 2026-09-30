"""Linux LSM6DSV16X/TR acquisition and body-frame gravity estimation.

Register definitions and sensitivities are from ST DS13510 Rev 4 (sections
5.1, 9.14--9.22, 9.28--9.41) and ST's official lsm6dsv16x-pid driver at
2808e5cd6b85f91b66758e1dd0faab5f043aba07:
https://www.st.com/resource/en/datasheet/lsm6dsv16x.pdf
https://github.com/STMicroelectronics/lsm6dsv16x-pid

No board pin or bus number is assumed. Mounting rotation maps sensor vectors
to the model's body coordinates. An upright, stationary body must report
acceleration approximately [0, 0, +9.80665] and projected gravity [0, 0, -1].
This is a software complementary filter, not the sensor's SFLP firmware.
Host monotonic timestamps indicate receipt, not hardware sample time. Use one
continuous acquisition thread; the device and estimator are not thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import glob
import math
import re
import struct
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np

GRAVITY = 9.80665
WHO_AM_I = 0x0F
DEVICE_ID = 0x70
CTRL1, CTRL2, CTRL3, CTRL4 = 0x10, 0x11, 0x12, 0x13
CTRL6, CTRL8 = 0x15, 0x17
STATUS_REG, OUTX_L_G = 0x1E, 0x22
ODR_CODES = {60: 5, 120: 6, 240: 7, 480: 8, 960: 9}
ACCEL_SCALES = {2: (0, 0.061), 4: (1, 0.122), 8: (2, 0.244), 16: (3, 0.488)}
GYRO_SCALES = {
    125: (0, 4.375), 250: (1, 8.75), 500: (2, 17.5),
    1000: (3, 35.0), 2000: (4, 70.0), 4000: (12, 140.0),
}


class ImuError(RuntimeError):
    """Identification, transport, calibration, or attitude estimation failure."""


class ImuTimeout(ImuError, TimeoutError):
    """Fresh data or reset completion did not arrive within the deadline."""


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length 3")
    return result.copy()


@dataclass(frozen=True)
class ImuConfig:
    transport: str = "i2c"
    i2c_bus: int | None = None
    i2c_address: int | None = None
    spi_bus: int | None = None
    spi_device: int | None = None
    spi_max_speed_hz: int = 1_000_000
    spi_mode: int = 0
    odr_hz: int = 120
    accel_range_g: int = 4
    gyro_range_dps: int = 500
    mounting_rotation: Any = field(default_factory=lambda: np.eye(3).tolist())
    mount_verified: bool = False
    gyro_bias_rad_s: Any = (0.0, 0.0, 0.0)
    complementary_tau_s: float = 0.5
    accel_rejection_fraction: float = 0.2
    max_sample_gap_s: float = 0.1

    def __post_init__(self) -> None:
        if self.transport not in ("i2c", "spi"):
            raise ValueError("transport must be i2c or spi")
        for name in ("i2c_bus", "spi_bus", "spi_device"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer or null")
        if self.i2c_address is not None and (
            type(self.i2c_address) is not int or self.i2c_address not in (0x6A, 0x6B)
        ):
            raise ValueError("i2c_address must be 0x6A, 0x6B, or null")
        if self.transport == "spi" and (self.spi_bus is None or self.spi_device is None):
            raise ValueError("SPI requires explicit spi_bus and spi_device")
        if self.spi_mode not in (0, 3):
            raise ValueError("LSM6DSV16X supports SPI mode 0 or 3")
        if type(self.spi_max_speed_hz) is not int or not 0 < self.spi_max_speed_hz <= 10_000_000:
            raise ValueError("spi_max_speed_hz must be in 1..10000000")
        for value, allowed, name in (
            (self.odr_hz, ODR_CODES, "odr_hz"),
            (self.accel_range_g, ACCEL_SCALES, "accel_range_g"),
            (self.gyro_range_dps, GYRO_SCALES, "gyro_range_dps"),
        ):
            if value not in allowed:
                raise ValueError(f"unsupported {name}: {value}; choose {list(allowed)}")
        rotation = np.asarray(self.mounting_rotation, dtype=np.float64)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6, rtol=0)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)):
            raise ValueError("mounting_rotation must be a proper orthonormal 3x3 rotation")
        _vector(self.gyro_bias_rad_s, "gyro_bias_rad_s")
        if type(self.mount_verified) is not bool:
            raise ValueError("mount_verified must be a JSON boolean")
        for name in ("complementary_tau_s", "max_sample_gap_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.accel_rejection_fraction) or not 0 < self.accel_rejection_fraction < 1:
            raise ValueError("accel_rejection_fraction must be between 0 and 1")
        if self.max_sample_gap_s < 2 / self.odr_hz:
            raise ValueError("max_sample_gap_s must allow at least two sensor periods")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ImuConfig:
        values = dict(values)
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown IMU settings: {sorted(unknown)}")
        if isinstance(values.get("i2c_address"), str):
            values["i2c_address"] = int(values["i2c_address"], 0)
        return cls(**values)


class RegisterTransport(Protocol):
    def read(self, register: int, length: int) -> bytes: ...
    def write(self, register: int, data: bytes) -> None: ...
    def close(self) -> None: ...


class I2cTransport:
    def __init__(self, bus: Any, address: int):
        self.bus, self.address = bus, address

    def read(self, register: int, length: int) -> bytes:
        return bytes(self.bus.read_i2c_block_data(self.address, register, length))

    def write(self, register: int, data: bytes) -> None:
        self.bus.write_i2c_block_data(self.address, register, list(data))

    def close(self) -> None:
        self.bus.close()


class SpiTransport:
    """Four-wire SPI; auto increment is CTRL3.IF_INC, not an address flag."""

    def __init__(self, device: Any):
        self.device = device

    def read(self, register: int, length: int) -> bytes:
        return bytes(self.device.xfer2([register | 0x80] + [0] * length)[1:])

    def write(self, register: int, data: bytes) -> None:
        self.device.xfer2([register & 0x7F] + list(data))

    def close(self) -> None:
        self.device.close()


def discover_i2c(
    *, bus_factory: Callable[[int], Any] | None = None,
    device_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Read WHO_AM_I at 0x6A/0x6B on /dev/i2c-*; never configure a device.

    The register-address phase is part of a normal register read. There is no
    all-address scan or data-register write. Nonmatching IDs are not selected.
    """
    paths = sorted(glob.glob("/dev/i2c-*") if device_paths is None else device_paths)
    paths = [path for path in paths if re.fullmatch(r"/dev/i2c-\d+", path)]
    result: dict[str, Any] = {"buses": paths, "matches": [], "errors": []}
    if not paths:
        return result
    if bus_factory is None:
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise ImuError("I2C requires the smbus2 package") from exc
        bus_factory = SMBus
    for path in paths:
        bus_number = int(path.rsplit("-", 1)[1])
        try:
            bus = bus_factory(bus_number)
        except OSError as exc:
            result["errors"].append({"path": path, "i2c_address": None, "error": str(exc)})
            continue
        try:
            for address in (0x6A, 0x6B):
                try:
                    identity = bus.read_byte_data(address, WHO_AM_I)
                    if identity == DEVICE_ID:
                        result["matches"].append({
                            "i2c_bus": bus_number, "i2c_address": address,
                            "who_am_i": identity,
                        })
                except OSError as exc:
                    result["errors"].append({"path": path, "i2c_address": address, "error": str(exc)})
        finally:
            bus.close()
    return result


def open_imu(config: ImuConfig | Mapping[str, Any]) -> Lsm6dsv16x:
    """Open a bus without configuring the sensor; call initialize next."""
    if not isinstance(config, ImuConfig):
        config = ImuConfig.from_dict(config)
    if config.transport == "i2c":
        if config.i2c_bus is None or config.i2c_address is None:
            discovery = discover_i2c()
            matches = [item for item in discovery["matches"]
                       if (config.i2c_bus is None or item["i2c_bus"] == config.i2c_bus)
                       and (config.i2c_address is None or item["i2c_address"] == config.i2c_address)]
            if len(matches) != 1:
                raise ImuError(f"expected exactly one matching LSM6DSV16X; discovery={discovery}")
            config = replace(config, i2c_bus=matches[0]["i2c_bus"], i2c_address=matches[0]["i2c_address"])
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise ImuError("I2C requires the smbus2 package") from exc
        return Lsm6dsv16x(I2cTransport(SMBus(config.i2c_bus), config.i2c_address), config)
    try:
        import spidev
    except ImportError as exc:
        raise ImuError("SPI requires the spidev package") from exc
    device = spidev.SpiDev()
    try:
        device.open(config.spi_bus, config.spi_device)
        device.mode = config.spi_mode
        device.max_speed_hz = config.spi_max_speed_hz
        device.bits_per_word = 8
        device.lsbfirst = False
        device.threewire = False
    except Exception:
        device.close()
        raise
    return Lsm6dsv16x(SpiTransport(device), config)


@dataclass(frozen=True)
class ImuSample:
    timestamp: float
    gyro_rad_s: np.ndarray  # Body frame, before software bias subtraction.
    accel_m_s2: np.ndarray  # Specific force, body frame.


@dataclass(frozen=True)
class ImuState:
    timestamp: float
    base_ang_vel: np.ndarray  # Body-frame radians/second, bias corrected.
    projected_gravity: np.ndarray  # Unit world-down vector in the body frame.
    accel_m_s2: np.ndarray


class GravityEstimator:
    """Integrate body-frame gyro, correct roll/pitch using specific force.

    Gravity alone cannot observe yaw. Sustained linear acceleration near 1 g
    can bias the estimate; norm rejection does not remove that limitation.
    """

    def __init__(self, config: ImuConfig):
        self.config = config
        self.gravity: np.ndarray | None = None
        self.timestamp: float | None = None

    def update(self, gyro_rad_s: Any, accel_m_s2: Any, timestamp: float) -> np.ndarray:
        gyro = _vector(gyro_rad_s, "gyro_rad_s")
        accel = _vector(accel_m_s2, "accel_m_s2")
        if not math.isfinite(timestamp):
            raise ImuError("nonfinite IMU timestamp")
        norm = float(np.linalg.norm(accel))
        accept_accel = abs(norm - GRAVITY) <= GRAVITY * self.config.accel_rejection_fraction
        if self.gravity is None:
            if not accept_accel:
                raise ImuError("cannot initialize gravity: hold IMU still with acceleration near 1 g")
            gravity = -accel / norm
        else:
            dt = timestamp - self.timestamp
            if dt <= 0 or dt > self.config.max_sample_gap_s:
                raise ImuTimeout(f"invalid IMU sample interval: {dt:.6f} s")
            # g_body = R_world_from_body.T @ g_world, hence dg/dt = -omega x g.
            angle = float(np.linalg.norm(gyro)) * dt
            gravity = self.gravity.copy()
            if angle > 1e-12:
                axis = -gyro / np.linalg.norm(gyro)
                gravity = (gravity * math.cos(angle)
                           + np.cross(axis, gravity) * math.sin(angle)
                           + axis * np.dot(axis, gravity) * (1 - math.cos(angle)))
            if accept_accel:
                weight = -math.expm1(-dt / self.config.complementary_tau_s)
                gravity = (1 - weight) * gravity - weight * accel / norm
            length = float(np.linalg.norm(gravity))
            if length < 1e-6:
                raise ImuError("gravity estimate became degenerate")
            gravity /= length
        self.gravity, self.timestamp = gravity, timestamp
        return gravity.copy()


class Lsm6dsv16x:
    def __init__(
        self, transport: RegisterTransport, config: ImuConfig,
        *, clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.transport, self.config = transport, config
        self._clock, self._sleep = clock, sleep
        self._rotation = np.asarray(config.mounting_rotation, dtype=np.float64).copy()
        self.gyro_bias_rad_s = _vector(config.gyro_bias_rad_s, "gyro_bias_rad_s")
        self._estimator = GravityEstimator(config)
        self._initialized = False
        self._last_sample_time: float | None = None

    def _read(self, register: int, length: int = 1) -> bytes:
        try:
            data = self.transport.read(register, length)
        except OSError as exc:
            raise ImuError(f"IMU register 0x{register:02x} read failed: {exc}") from exc
        if len(data) != length:
            raise ImuError(f"short IMU read at 0x{register:02x}: {len(data)} != {length}")
        return data

    def _write(self, register: int, value: int) -> None:
        try:
            self.transport.write(register, bytes([value]))
        except OSError as exc:
            raise ImuError(f"IMU register 0x{register:02x} write failed: {exc}") from exc

    def initialize(self, timeout_s: float = 0.2) -> None:
        """Identify, reset, set high-performance ODR/scale, then verify writes.

        Reading is allowed before mount_verified so the mount can be checked.
        The actuator arming code must require mount_verified independently.
        """
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self._initialized = False
        identity = self._read(WHO_AM_I)[0]
        if identity != DEVICE_ID:
            raise ImuError(f"WHO_AM_I=0x{identity:02x}; LSM6DSV16X requires 0x70")
        # Match ST's sw_reset sequence; all writes are volatile control registers.
        self._write(CTRL1, 0)
        self._write(CTRL2, 0)
        self._write(CTRL3, 1)
        deadline = self._clock() + timeout_s
        while self._read(CTRL3)[0] & 1:
            if self._clock() >= deadline:
                raise ImuTimeout("LSM6DSV16X software reset timed out")
            self._sleep(0.001)
        settings = {
            CTRL3: 0x44,  # BDU + IF_INC.
            CTRL4: 0x08,  # Mask data-ready while sensor filters settle.
            CTRL6: GYRO_SCALES[self.config.gyro_range_dps][0],
            CTRL8: ACCEL_SCALES[self.config.accel_range_g][0],
            CTRL1: ODR_CODES[self.config.odr_hz],
            CTRL2: ODR_CODES[self.config.odr_hz],
        }
        for register, value in settings.items():
            self._write(register, value)
        for register, value in settings.items():
            if self._read(register)[0] != value:
                raise ImuError(f"configuration readback failed at 0x{register:02x}")
        self._sleep(0.05)  # Gyro turn-on is typically 30 ms; DRDY_MASK also applies.
        self._read(OUTX_L_G, 12)  # Drain any data held during startup.
        self._last_sample_time = None
        self._estimator = GravityEstimator(self.config)
        self._initialized = True

    def read_sample(self, timeout_s: float = 0.1) -> ImuSample:
        """Wait for BOTH fresh gyro and accel data; never reuse cached data.

        After a long pause, discard the pending BDU-held sample and wait for
        the next update. Linux bus calls themselves obey kernel, not Python,
        timeouts; the caller should also enforce a latest-state age watchdog.
        """
        if not self._initialized:
            raise ImuError("initialize the IMU before reading")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        start = self._clock()
        if self._last_sample_time is not None and start - self._last_sample_time > self.config.max_sample_gap_s:
            self._read(OUTX_L_G, 12)
        deadline = start + timeout_s
        while True:
            ready = self._read(STATUS_REG)[0] & 0x03
            if self._clock() >= deadline:
                raise ImuTimeout("timed out waiting for fresh accelerometer and gyroscope data")
            if ready == 0x03:
                break
            self._sleep(min(0.001, 0.1 / self.config.odr_hz))
        payload = self._read(OUTX_L_G, 12)
        timestamp = self._clock()
        if timestamp >= deadline:
            raise ImuTimeout("IMU data read exceeded its deadline")
        raw = np.asarray(struct.unpack("<6h", payload), dtype=np.float64)
        # Raw gyro then raw accelerometer: registers 0x22 through 0x2D.
        gyro = raw[:3] * GYRO_SCALES[self.config.gyro_range_dps][1] * math.pi / 180_000
        accel = raw[3:] * ACCEL_SCALES[self.config.accel_range_g][1] * GRAVITY / 1000
        self._last_sample_time = timestamp
        return ImuSample(timestamp, self._rotation @ gyro, self._rotation @ accel)

    def read_state(self, timeout_s: float = 0.1) -> ImuState:
        sample = self.read_sample(timeout_s)
        gyro = sample.gyro_rad_s - self.gyro_bias_rad_s
        gravity = self._estimator.update(gyro, sample.accel_m_s2, sample.timestamp)
        return ImuState(sample.timestamp, gyro, gravity, sample.accel_m_s2)

    def calibrate_gyro(self, sample_count: int = 240, timeout_s: float = 0.2) -> np.ndarray:
        """Estimate stationary gyro bias in BODY coordinates before starting a reader.

        Physically hold the robot still. Norm/variance checks reject ordinary
        motion, but six-axis IMUs cannot distinguish tiny constant yaw from
        bias. This does not calibrate accelerometer offsets or redefine level.
        Failure leaves the previous gyro bias unchanged.
        """
        if type(sample_count) is not int or sample_count < 20:
            raise ValueError("gyro calibration requires at least 20 samples")
        samples = [self.read_sample(timeout_s) for _ in range(sample_count)]
        gyro = np.stack([sample.gyro_rad_s for sample in samples])
        accel = np.stack([sample.accel_m_s2 for sample in samples])
        elapsed = samples[-1].timestamp - samples[0].timestamp
        if elapsed < 0.5 * (sample_count - 1) / self.config.odr_hz:
            raise ImuError("calibration samples arrived too quickly to be fresh")
        accel_norm = np.linalg.norm(accel, axis=1)
        bias = gyro.mean(axis=0)
        if (np.max(np.abs(accel_norm - GRAVITY)) > 0.15 * GRAVITY
                or np.max(accel.std(axis=0)) > 0.15
                or np.max(gyro.std(axis=0)) > 0.015
                or np.max(np.linalg.norm(gyro, axis=1)) > 0.1):
            raise ImuError("gyro calibration rejected motion or acceleration away from 1 g; hold robot still")
        self.gyro_bias_rad_s = bias
        self._estimator = GravityEstimator(self.config)
        return bias.copy()

    def close(self) -> None:
        self._initialized = False
        self.transport.close()
