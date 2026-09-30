"""CPU checks of the ST register contract and the body-frame estimator."""

import math
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy import imu


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Registers:
    def __init__(self, clock, raw=(10, -20, 30, 0, 0, 8197)):
        self.clock = clock
        self.regs = {imu.WHO_AM_I: imu.DEVICE_ID}
        self.raw = raw
        self.writes = []
        self.reads = []
        self.next_ready = 0
        self.status_mask = 3
        self.reset_stuck = False
        self.closed = False

    def read(self, register, length):
        self.reads.append((register, length))
        if register == imu.CTRL3 and self.regs.get(register) == 1 and not self.reset_stuck:
            self.regs[register] = 0x44
        if register == imu.STATUS_REG:
            return bytes([self.status_mask if self.clock() >= self.next_ready else 0])
        if register == imu.OUTX_L_G:
            self.next_ready = self.clock() + 1 / 120
            return struct.pack("<6h", *self.raw)
        return bytes([self.regs.get(register, 0)])

    def write(self, register, data):
        self.writes.append((register, data))
        self.regs[register] = data[0]

    def close(self):
        self.closed = True


def device(config=None, raw=(10, -20, 30, 0, 0, 8197)):
    clock = Clock()
    registers = Registers(clock, raw)
    sensor = imu.Lsm6dsv16x(registers, config or imu.ImuConfig(), clock=clock, sleep=clock.sleep)
    sensor.initialize()
    return sensor, registers, clock


def test_config_from_json_keeps_mount_unverified_and_never_assumes_bus():
    config = imu.ImuConfig.from_dict({"i2c_address": "0x6b"})
    assert config.i2c_address == 0x6B
    assert config.i2c_bus is None
    assert not config.mount_verified
    with pytest.raises(ValueError, match="unknown IMU"):
        imu.ImuConfig.from_dict({"i2c_pin": 3})


@pytest.mark.parametrize("settings", [
    {"mounting_rotation": np.diag([1, 1, -1])},
    {"mounting_rotation": np.eye(3) * 2},
    {"mounting_rotation": [[1, 0, 0], [0, 1, 0]]},
    {"mounting_rotation": np.eye(3) * float("nan")},
    {"mount_verified": "true"},
    {"gyro_bias_rad_s": [0, 0, float("nan")]},
    {"i2c_bus": -1}, {"i2c_bus": True}, {"i2c_address": 0x68},
    {"transport": "spi"}, {"spi_mode": 1},
    {"spi_max_speed_hz": 11_000_000}, {"odr_hz": 100},
    {"accel_range_g": 3}, {"gyro_range_dps": 300},
    {"max_sample_gap_s": 0.001}, {"complementary_tau_s": float("nan")},
])
def test_config_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        imu.ImuConfig(**settings)


def test_initialization_identifies_before_writing_and_checks_configuration():
    sensor, registers, _ = device()
    assert registers.reads[0] == (0x0F, 1)
    assert registers.writes[:3] == [(0x10, b"\x00"), (0x11, b"\x00"), (0x12, b"\x01")]
    assert {register: data[0] for register, data in registers.writes[3:]} == {
        0x12: 0x44, 0x13: 0x08, 0x15: 2, 0x17: 1, 0x10: 6, 0x11: 6,
    }
    sensor.close()
    assert registers.closed
    with pytest.raises(imu.ImuError, match="initialize"):
        sensor.read_sample()


def test_wrong_part_never_receives_configuration_writes():
    clock = Clock()
    registers = Registers(clock)
    registers.regs[imu.WHO_AM_I] = 0x6C
    sensor = imu.Lsm6dsv16x(registers, imu.ImuConfig(), clock=clock, sleep=clock.sleep)
    with pytest.raises(imu.ImuError, match="WHO_AM_I=0x6c"):
        sensor.initialize()
    assert not registers.writes


def test_reset_timeout_is_bounded():
    clock = Clock()
    registers = Registers(clock)
    registers.reset_stuck = True
    sensor = imu.Lsm6dsv16x(registers, imu.ImuConfig(), clock=clock, sleep=clock.sleep)
    with pytest.raises(imu.ImuTimeout, match="reset"):
        sensor.initialize(timeout_s=0.01)
    assert 0.01 <= clock.now < 0.012


def test_configuration_readback_mismatch_fails():
    clock = Clock()
    registers = Registers(clock)
    original_write = registers.write

    def ignore_accel_range(register, data):
        if register != imu.CTRL8:
            original_write(register, data)

    registers.write = ignore_accel_range
    sensor = imu.Lsm6dsv16x(registers, imu.ImuConfig(), clock=clock, sleep=clock.sleep)
    with pytest.raises(imu.ImuError, match="readback.*0x17"):
        sensor.initialize()


def test_signed_little_endian_si_conversion_and_mounting_rotation():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    raw = (-32768, 1000, 32767, -1000, 2000, 3000)
    sensor, _, _ = device(imu.ImuConfig(mounting_rotation=rotation), raw)
    sample = sensor.read_sample()
    np.testing.assert_allclose(sample.gyro_rad_s, rotation @ (np.array(raw[:3]) * 0.0175 * math.pi / 180))
    np.testing.assert_allclose(sample.accel_m_s2, rotation @ (np.array(raw[3:]) * 0.000122 * imu.GRAVITY))


@pytest.mark.parametrize("ready_mask", [0, 1, 2])
def test_waits_for_both_fresh_sensor_status_bits(ready_mask):
    sensor, registers, clock = device()
    registers.status_mask = ready_mask
    start = clock()
    with pytest.raises(imu.ImuTimeout, match="fresh"):
        sensor.read_sample(timeout_s=0.02)
    assert 0.02 <= clock() - start < 0.022


def test_repeat_values_still_need_fresh_status_and_long_pause_discards_held_sample():
    sensor, registers, clock = device()
    a = sensor.read_sample()
    b = sensor.read_sample()
    assert b.timestamp - a.timestamp >= 1 / 120
    np.testing.assert_array_equal(a.gyro_rad_s, b.gyro_rad_s)
    count = registers.reads.count((imu.OUTX_L_G, 12))
    clock.sleep(0.2)
    start = clock()
    c = sensor.read_sample()
    assert c.timestamp - start >= 1 / 120
    assert registers.reads.count((imu.OUTX_L_G, 12)) - count == 2


def test_short_and_failed_reads_are_errors():
    sensor, registers, _ = device()
    registers.read = lambda register, length: b""
    with pytest.raises(imu.ImuError, match="short IMU read"):
        sensor.read_sample()

    def fail(register, length):
        raise OSError("bus disconnected")

    registers.read = fail
    with pytest.raises(imu.ImuError, match="bus disconnected"):
        sensor.read_sample()


def test_gravity_initialization_preserves_real_tilt_and_sign():
    estimator = imu.GravityEstimator(imu.ImuConfig())
    accel = np.array([0.5, 0, math.sqrt(3) / 2]) * imu.GRAVITY
    np.testing.assert_allclose(estimator.update([0, 0, 0], accel, 0), -accel / imu.GRAVITY)
    upright = imu.GravityEstimator(imu.ImuConfig())
    np.testing.assert_array_equal(upright.update([0, 0, 0], [0, 0, imu.GRAVITY], 0), [0, 0, -1])
    with pytest.raises(imu.ImuError, match="initialize gravity"):
        imu.GravityEstimator(imu.ImuConfig()).update([0, 0, 0], [0, 0, 0], 0)


def test_body_frame_gyro_rotation_has_correct_sign_and_rejects_acceleration():
    estimator = imu.GravityEstimator(imu.ImuConfig())
    estimator.update([0, 0, 0], [0, 0, imu.GRAVITY], 0)
    # Positive body roll rotates projected world-down toward negative body y.
    for i in range(1, 101):
        gravity = estimator.update([math.pi / 2, 0, 0], [0, 0, 2 * imu.GRAVITY], i / 100)
    np.testing.assert_allclose(gravity, [0, -1, 0], atol=1e-12)


def test_complementary_correction_converges_without_redefining_upright():
    estimator = imu.GravityEstimator(imu.ImuConfig())
    estimator.update([0, 0, 0], [0, 0, imu.GRAVITY], 0)
    accel = np.array([0.5, 0, math.sqrt(3) / 2]) * imu.GRAVITY
    for i in range(1, 501):
        gravity = estimator.update([0, 0, 0], accel, i / 100)
    np.testing.assert_allclose(gravity, -accel / imu.GRAVITY, atol=3e-5)
    assert np.linalg.norm(gravity) == pytest.approx(1)


@pytest.mark.parametrize("timestamp", [0, -0.01, 0.2])
def test_estimator_rejects_nonmonotonic_or_stale_time(timestamp):
    estimator = imu.GravityEstimator(imu.ImuConfig())
    estimator.update([0, 0, 0], [0, 0, imu.GRAVITY], 0)
    with pytest.raises(imu.ImuTimeout):
        estimator.update([0, 0, 0], [0, 0, imu.GRAVITY], timestamp)


def test_static_gyro_calibration_corrects_bias_and_keeps_tilt():
    sensor, _, _ = device(raw=(10, -20, 30, 4098, 0, 7098))
    bias = sensor.calibrate_gyro(sample_count=30)
    np.testing.assert_allclose(bias, np.array([10, -20, 30]) * 0.0175 * math.pi / 180)
    state = sensor.read_state()
    np.testing.assert_allclose(state.base_ang_vel, [0, 0, 0], atol=1e-15)
    expected = -np.array([4098, 0, 7098]) / np.linalg.norm([4098, 0, 7098])
    np.testing.assert_allclose(state.projected_gravity, expected)


@pytest.mark.parametrize("motion", ["gyro_mean", "gyro_variance", "accel_norm", "accel_variance"])
def test_calibration_rejects_motion_without_changing_existing_bias(motion):
    original = np.array([0.001, 0.002, 0.003])
    sensor, _, _ = device(imu.ImuConfig(gyro_bias_rad_s=original))
    sequence = []
    for index in range(30):
        gyro = [0.2 if motion == "gyro_mean" else 0, 0, 0]
        accel = [0, 0, imu.GRAVITY]
        if motion == "gyro_variance":
            gyro[0] = 0.03 * (-1) ** index
        if motion == "accel_norm":
            accel[2] *= 1.3
        if motion == "accel_variance":
            accel[0] = 0.3 * (-1) ** index
        sequence.append(imu.ImuSample(index / 120, np.array(gyro), np.array(accel)))
    samples = iter(sequence)
    sensor.read_sample = lambda timeout_s: next(samples)
    with pytest.raises(imu.ImuError, match="calibration rejected"):
        sensor.calibrate_gyro(30)
    np.testing.assert_array_equal(sensor.gyro_bias_rad_s, original)


def test_discovery_only_reads_whoami_at_the_two_legal_addresses():
    accesses = []
    closed = []

    class Bus:
        def __init__(self, bus):
            self.number = bus
            if bus == 9:
                raise PermissionError("permission denied")

        def read_byte_data(self, address, register):
            accesses.append((self.number, address, register))
            if self.number == 3 and address == 0x6A:
                return 0x70
            if address == 0x6B:
                raise OSError("no response")
            return 0x6C

        def close(self):
            closed.append(self.number)

    result = imu.discover_i2c(bus_factory=Bus, device_paths=["/dev/i2c-3", "/dev/i2c-9", "/dev/i2c-4", "/dev/i2c-bad"])
    assert result["matches"] == [{"i2c_bus": 3, "i2c_address": 0x6A, "who_am_i": 0x70}]
    assert accesses == [(3, 0x6A, 0x0F), (3, 0x6B, 0x0F), (4, 0x6A, 0x0F), (4, 0x6B, 0x0F)]
    assert closed == [3, 4]
    assert any("permission denied" in item["error"] for item in result["errors"])


def test_no_bus_discovery_does_not_need_hardware_modules():
    assert imu.discover_i2c(device_paths=[]) == {"buses": [], "matches": [], "errors": []}


def test_auto_discovery_selects_only_one_matching_part(monkeypatch):
    monkeypatch.setattr(imu, "discover_i2c", lambda: {"matches": [
        {"i2c_bus": 7, "i2c_address": 0x6B, "who_am_i": 0x70}], "errors": [], "buses": []})
    opened = []
    monkeypatch.setitem(sys.modules, "smbus2", SimpleNamespace(SMBus=lambda bus: opened.append(bus) or object()))
    sensor = imu.open_imu({})
    assert opened == [7]
    assert sensor.config.i2c_address == 0x6B
    monkeypatch.setattr(imu, "discover_i2c", lambda: {"matches": [
        {"i2c_bus": 7, "i2c_address": 0x6B}, {"i2c_bus": 8, "i2c_address": 0x6A}], "errors": [], "buses": []})
    with pytest.raises(imu.ImuError, match="exactly one"):
        imu.open_imu({})


def test_spi_wire_protocol_uses_bit7_for_read_and_no_multiread_flag():
    transfers = []

    class Spi:
        def xfer2(self, values):
            transfers.append(values)
            return [0] + list(range(len(values) - 1))

        def close(self):
            pass

    transport = imu.SpiTransport(Spi())
    assert transport.read(0x22, 12) == bytes(range(12))
    transport.write(0x12, b"\x44")
    assert transfers == [[0xA2] + [0] * 12, [0x12, 0x44]]


def test_i2c_transport_uses_register_block_access():
    calls = []
    bus = SimpleNamespace(
        read_i2c_block_data=lambda *args: calls.append(("read", args)) or [1, 2],
        write_i2c_block_data=lambda *args: calls.append(("write", args)),
        close=lambda: calls.append(("close", ())),
    )
    transport = imu.I2cTransport(bus, 0x6A)
    assert transport.read(0x22, 2) == b"\x01\x02"
    transport.write(0x12, b"\x44")
    transport.close()
    assert calls == [("read", (0x6A, 0x22, 2)), ("write", (0x6A, 0x12, [0x44])), ("close", ())]
