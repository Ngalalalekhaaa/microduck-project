"""HOME hardware contracts, exercised with byte packets and fakes only."""

from copy import deepcopy
from dataclasses import replace
import math
import struct
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import default_config
from microduck_deploy.home_calibration_hardware import (
    CalibrationServoBus, CurrentMapping, HardwareSafetyError,
    HomeCalibrationHardwareError, RawCurrentFeedback, ReadOnlyServoBus,
    current_mapping_template, read_hardware_sample, validate_hold_configuration,
)
from microduck_deploy.imu import ImuState
from microduck_deploy.servo import ServoBus, ServoConfigurationError, ServoStatusError, ServoTimeout


def _config():
    cfg = default_config()
    cfg["calibration"] = {"pose": "home"}
    cfg["imu_mount_verified"] = True
    for index, joint in enumerate(cfg["joints"]):
        joint.update(id=index + 1, direction=-1 if index < 13 else 1,
                     zero_tick=2048.25, mapping_verified=True)
    return cfg


def _mapping(cfg, *, verified=False, encoding="sign_magnitude15"):
    spec = current_mapping_template(cfg)
    spec["encoding"] = encoding
    for joint in spec["joints"]:
        joint["verified"] = verified
    return CurrentMapping.from_dict(spec, cfg)


def _hardware(cfg):
    feedback = {
        joint["id"]: RawCurrentFeedback(
            servo_id=joint["id"], position_ticks=2050, velocity_raw=0,
            load_raw=900, voltage_v=7.4, temperature_c=30, status=0, moving=0,
            goal_ticks=2050, current_ma=-13., received_at=99.98 + index * .0002,
            current_register_raw=0x8002,
        ) for index, joint in enumerate(cfg["joints"])
    }
    state = ImuState(99.995, np.array([.1, -.2, .3]),
                     np.array([0., 0., -1.]), np.array([0., 0., 9.80665]))
    bus = SimpleNamespace(read_feedback_many=lambda ids, sync: dict(feedback))
    imu = SimpleNamespace(latest=lambda max_age: state)
    return bus, imu, feedback, state


def test_raw_current_is_retained_from_same_feedback_block_and_driver_is_signed():
    payload = bytearray(15)
    struct.pack_into("<HHH", payload, 0, 2048, 0, 900)
    payload[6:8] = bytes([74, 30])
    struct.pack_into("<HH", payload, 11, 2048, 0x8002)
    state = ReadOnlyServoBus._feedback(1, bytes(payload))
    assert state.current_register_raw == 0x8002
    assert state.current_ma == -13.
    assert state.load_raw == 900  # conflicting sign cannot change current sign


def test_readonly_bus_rejects_all_register_writes_before_parent_dispatch(monkeypatch):
    writes = []
    monkeypatch.setattr(ServoBus, "_sync_ram", lambda *args: writes.append(args))
    bus = object.__new__(ReadOnlyServoBus)
    bus._io = threading.RLock()
    with pytest.raises(ServoConfigurationError, match="read-only"):
        bus.sync_positions({1: 2048})
    for enabled in (True, False):
        with pytest.raises(ServoConfigurationError, match="read-only"):
            bus.set_torque([1], enabled, verify=False)
    for address, size, value in ((50, 3, b"\x05\x00\x00"), (31, 2, b"\x00\x00")):
        with pytest.raises(ServoConfigurationError):
            bus._sync_ram(address, size, {1: value})
    assert writes == []


def test_correction_bus_allows_goals_and_torque_off_but_no_enable_gains_or_eeprom(monkeypatch):
    writes = []
    monkeypatch.setattr(ServoBus, "_sync_ram",
                        lambda self, address, size, values: writes.append((address, size, values)))
    bus = object.__new__(CalibrationServoBus)
    bus._io = threading.RLock()
    bus.sync_positions({1: 2050})
    bus.set_torque([1], False, verify=False)
    assert writes == [(42, 2, {1: struct.pack("<H", 2050)}), (40, 1, {1: b"\x00"})]
    with pytest.raises(ServoConfigurationError, match="OFF only"):
        bus.set_torque([1], True, verify=False)
    for address, size, value in ((50, 3, b"\x05\x00\x00"), (31, 2, b"\x00\x00"),
                                 (42, 2, struct.pack("<H", 4096))):
        with pytest.raises(ServoConfigurationError):
            bus._sync_ram(address, size, {1: value})
    assert len(writes) == 2


def test_current_profile_requires_explicit_verified_signed_model_axis_mapping():
    cfg = _config()
    spec = current_mapping_template(cfg)
    assert all(row["verified"] is False for row in spec["joints"])
    mapping = CurrentMapping.from_dict(spec, cfg)
    assert not mapping.motion_ready
    with pytest.raises(HomeCalibrationHardwareError, match="verified signed"):
        mapping.require_motion()
    for row in spec["joints"]:
        row["verified"] = True
    assert CurrentMapping.from_dict(spec, cfg, motion=True).motion_ready
    spec["encoding"] = "unsigned16"
    with pytest.raises(HomeCalibrationHardwareError, match="unsigned"):
        CurrentMapping.from_dict(spec, cfg, motion=True)
    spec["encoding"] = "sign_from_load"
    with pytest.raises(HomeCalibrationHardwareError, match="load-derived"):
        CurrentMapping.from_dict(spec, cfg)


@pytest.mark.parametrize("encoding,word", [("sign_magnitude15", 0x8002), ("twos_complement16", 0xfffe)])
def test_current_conversion_applies_model_axis_sign_once_and_never_uses_load(encoding, word):
    cfg = _config()
    mapping = _mapping(cfg, verified=True, encoding=encoding)
    result = mapping.convert({joint["id"]: word for joint in cfg["joints"]})
    np.testing.assert_allclose(result[:13], .013)
    assert result[13] == pytest.approx(-.013)


def test_current_profile_rejects_wrong_robot_and_invalid_scales():
    cfg = _config()
    for field, value in (("id", 22), ("current_sign", True), ("current_scale_a", 0.),
                         ("current_scale_a", float("nan")), ("verified", "true")):
        spec = current_mapping_template(cfg)
        spec["joints"][0][field] = value
        with pytest.raises(HomeCalibrationHardwareError):
            CurrentMapping.from_dict(spec, cfg)
    mapping = _mapping(cfg)
    with pytest.raises(HomeCalibrationHardwareError, match="unavailable"):
        mapping.convert({})


def test_monitor_reads_raw_diagnostics_without_mapping_and_labels_unverified_mapping():
    cfg = _config()
    bus, imu, feedback, state = _hardware(cfg)
    sample = read_hardware_sample(bus, imu, cfg, now=100.)
    assert sample.current_a is None
    assert sample.diagnostics["current_mapping_status"] == "missing"
    assert not sample.diagnostics["current_mapping_verified"]
    assert sample.diagnostics["current_register_raw_by_id"]["1"] == 0x8002
    assert sample.feedback[1] is feedback[1]
    expected_q = np.array([joint["direction"] for joint in cfg["joints"]]) * 1.75 * math.tau / 4096
    np.testing.assert_allclose(sample.q, expected_q)
    np.testing.assert_array_equal(sample.gyro, state.base_ang_vel)
    np.testing.assert_array_equal(sample.gravity, state.projected_gravity)
    sample = read_hardware_sample(bus, imu, cfg, current_mapping=_mapping(cfg), now=100.)
    np.testing.assert_allclose(sample.current_a[:13], .013)
    assert sample.diagnostics["current_mapping_status"] == "unverified"
    assert not sample.diagnostics["current_mapping_verified"]


@pytest.mark.parametrize("field,value,match", [
    ("received_at", 99., "stale"), ("received_at", 101., "future"),
    ("status", 1, "status"), ("position_ticks", 4096, "single-turn"),
    ("voltage_v", 9., "voltage"), ("temperature_c", 60, "temperature"),
    ("current_ma", float("nan"), "finite"), ("servo_id", 99, "ID"),
])
def test_invalid_servo_samples_cannot_reach_policy(field, value, match):
    cfg = _config()
    bus, imu, feedback, _ = _hardware(cfg)
    feedback[1] = replace(feedback[1], **{field: value})
    with pytest.raises(HomeCalibrationHardwareError, match=match) as caught:
        read_hardware_sample(bus, imu, cfg, now=100.)
    assert bool(getattr(caught.value, "release_torque", False)) == (field == "temperature_c")


def test_sweep_skew_and_imu_freshness_and_gravity_are_checked():
    cfg = _config()
    bus, imu, feedback, state = _hardware(cfg)
    feedback[1] = replace(feedback[1], received_at=99.90)
    with pytest.raises(HomeCalibrationHardwareError, match="sweep"):
        read_hardware_sample(bus, imu, cfg, now=100., max_age_s=.2)
    feedback[1] = replace(feedback[1], received_at=99.98)
    for new_state, match in ((replace(state, timestamp=99.), "stale"),
                             (replace(state, timestamp=101.), "future"),
                             (replace(state, projected_gravity=np.zeros(3)), "unit"),
                             (replace(state, base_ang_vel=np.array([np.nan, 0., 0.])), "finite")):
        imu.latest = lambda max_age, value=new_state: value
        with pytest.raises(HomeCalibrationHardwareError, match=match):
            read_hardware_sample(bus, imu, cfg, now=100.)


@pytest.mark.parametrize("error,release", [(ServoTimeout("late"), False),
                                          (ServoStatusError(1, 1), False),
                                          (ServoStatusError(1, 1 << 2), True),
                                          (ServoStatusError(1, 1 << 3), True)])
def test_only_explicit_temperature_or_current_fault_requests_release(error, release):
    cfg = _config()
    bus, imu, _, _ = _hardware(cfg)

    def fail(ids, sync):
        raise error

    bus.read_feedback_many = fail
    with pytest.raises(HomeCalibrationHardwareError) as caught:
        read_hardware_sample(bus, imu, cfg, now=100.)
    assert isinstance(caught.value, HardwareSafetyError) == release


@pytest.mark.parametrize('expected_p', [5, 8])
def test_preheld_configuration_accepts_only_requested_gains_without_writes(expected_p):
    cfg = _config()
    configurations = {joint["id"]: {"register_profile": "hls_2", "torque": 1,
                                    "mode": 4, "angle_resolution": 1,
                                    "ram_gains": {"p": expected_p, "d": 0, "i": 0},
                                    "position_min": 0, "position_max": 4095}
                      for joint in cfg["joints"]}
    validate_hold_configuration(configurations, cfg, expected_p=expected_p)
    if expected_p == 5:
        validate_hold_configuration(configurations, cfg)  # Existing default.
    else:
        with pytest.raises(HomeCalibrationHardwareError, match='relax'):
            validate_hold_configuration(configurations, cfg)
    original = deepcopy(configurations)
    for field, value in (("torque", 0), ("mode", 0), ("angle_resolution", 2),
                         ("ram_gains", {"p": 16, "d": 0, "i": 0}),
                         ("ram_gains", {"p": expected_p, "d": 1, "i": 0}),
                         ("ram_gains", {"p": expected_p, "d": 0, "i": 1}), ("position_max", 0)):
        invalid = deepcopy(configurations)
        invalid[1][field] = value
        with pytest.raises(HomeCalibrationHardwareError):
            validate_hold_configuration(invalid, cfg, expected_p=expected_p)
    assert configurations == original


@pytest.mark.parametrize('expected_p', [True, 5., 0, 6, 16, 32, '8'])
def test_hold_configuration_rejects_unapproved_gain_selection(expected_p):
    with pytest.raises(HomeCalibrationHardwareError, match='5 or 8'):
        validate_hold_configuration({}, _config(), expected_p=expected_p)
