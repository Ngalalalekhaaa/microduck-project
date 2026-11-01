"""Ground-stance identification flow against deterministic plants, no hardware."""

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy.home_assist import HomeAssistError, pose_for_trim
from microduck_deploy.policy import HOME, JOINT_LIMITS


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("imu_home_assist_test", SCRIPTS / "imu_home_assist.py")
flow = importlib.util.module_from_spec(spec)
original_path = sys.path[:]
try:
    # The production entry point receives its own scripts directory from Python.
    sys.path.insert(0, str(SCRIPTS))
    spec.loader.exec_module(flow)
finally:
    sys.path[:] = original_path


def test_readonly_bus_rejects_every_motor_write_before_serial_access():
    bus = object.__new__(flow.ReadOnlyBus)
    with patch.object(flow.PositionOnlyBus, '_send', return_value='read') as send:
        for instruction in (3, 4, 5, 131, 254):
            with pytest.raises(ValueError, match='只读模式'):
                bus._send(bytes([255, 255, 1, 2, instruction, 0]))
        send.assert_not_called()
        for instruction in (2, 130):
            assert bus._send(bytes([255, 255, 1, 2, instruction, 0])) == 'read'
        assert send.call_count == 2


class FakePlant:
    """No servo/IMU API; attitude = initial bias + local response @ trim."""

    def __init__(self, base_deg=(3.0, -2.0), jacobian=((0.2, 1.0), (-1.0, 0.1))):
        self.base = np.deg2rad(base_deg)
        self.jacobian = np.asarray(jacobian, dtype=float)
        self.trim = np.zeros(2)
        self.commands = []
        self.samples = 0

    def move_trim(self, trim):
        self.trim = np.asarray(trim, dtype=float).copy()
        self.commands.append(self.trim.copy())

    def attitude(self):
        return self.base + self.jacobian @ self.trim

    def sample(self):
        self.samples += 1
        angles = self.attitude()
        return {"attitude_rad": angles.copy(), "attitude_deg": np.rad2deg(angles),
                "measured_q_rad": pose_for_trim(self.trim), "joint_check": {"fake": True}}

    def check_target(self, q):
        # Simulate target conversion without an encoder zero or motor packet.
        assert np.isfinite(q).all()
        assert np.all(q >= JOINT_LIMITS[:, 0]) and np.all(q <= JOINT_LIMITS[:, 1])
        return np.rint(2048 + 4096 * (q - HOME) / (2 * np.pi)).astype(int)


def test_reachable_2x2_response_converges_and_preserves_report_and_training_home():
    plant = FakePlant()
    before_home = HOME.copy()
    report = {"calibration_changed": False, "policy_changed": False, "sentinel": "preserve"}
    flow.trim_ground_stance(plant, report)

    np.testing.assert_allclose(report["attitude_response"], plant.jacobian, atol=1e-12)
    assert np.linalg.norm(report["after"]["attitude_rad"]) <= np.deg2rad(1.5)
    assert len(report["probes"]) == 4
    assert 1 <= len(report["iterations"]) <= 12
    assert np.max(np.abs(report["trim_rad"])) <= np.deg2rad(4)
    np.testing.assert_allclose(report["target_q_rad"], pose_for_trim(plant.trim))
    np.testing.assert_array_equal(HOME, before_home)
    assert report["sentinel"] == "preserve"
    assert report["calibration_changed"] is False and report["policy_changed"] is False
    # A final report must serialize without destroying collected NumPy data.
    encoded = json.dumps(report, default=flow.json_default, allow_nan=False)
    assert '"sentinel": "preserve"' in encoded


def test_already_level_does_not_probe_or_move():
    plant = FakePlant(base_deg=(0.4, -0.4))
    report = {}
    flow.trim_ground_stance(plant, report)
    assert plant.commands == []
    assert "无需微调" in report["result"]
    assert report["after"] is report["before"]


@pytest.mark.parametrize("response", [np.zeros((2, 2)), np.array([[0.0, 1.0], [0.0, 0.0]])])
def test_clamped_or_degenerate_response_rejects_and_keeps_probe_evidence(response):
    plant = FakePlant(jacobian=response)
    report = {"calibration_changed": False, "sentinel": "preserve"}
    with pytest.raises(HomeAssistError, match="姿态响应太弱"):
        flow.trim_ground_stance(plant, report)
    np.testing.assert_array_equal(plant.trim, np.zeros(2))
    assert len(report["probes"]) == 4
    assert report["iterations"] == []
    np.testing.assert_allclose(report["attitude_response"], response, atol=1e-12)
    assert report["sentinel"] == "preserve"
    assert report["calibration_changed"] is False


def test_contact_response_change_rolls_back_worsening_first_trial():
    class ChangedContact(FakePlant):
        def attitude(self):
            # Initial plus/minus probes are valid. Ground contact changes only
            # after returning to baseline, reversing the first trial's response.
            sign = -2.0 if self.samples >= 7 else 1.0
            return self.base + sign * self.jacobian @ self.trim

    plant = ChangedContact()
    report = {"sentinel": "preserve"}
    with pytest.raises(ValueError, match="已撤销本步"):
        flow.trim_ground_stance(plant, report)
    np.testing.assert_array_equal(plant.trim, np.zeros(2))
    assert len(report["probes"]) == 4
    assert report["iterations"] == []
    assert np.max(np.abs(plant.commands[-2])) > 0
    np.testing.assert_array_equal(plant.commands[-1], np.zeros(2))
    assert report["sentinel"] == "preserve"


def test_bounded_trim_cannot_report_unreachable_bias_as_level():
    plant = FakePlant(base_deg=(10.0, -10.0), jacobian=np.eye(2))
    report = {}
    flow.trim_ground_stance(plant, report)
    np.testing.assert_allclose(plant.trim, np.deg2rad((-4.0, 4.0)), atol=1e-12)
    assert np.linalg.norm(report["after"]["attitude_rad"]) > np.deg2rad(1.5)
    assert "次数上限" in report["result"]
    assert "不能据此认定" in report["result"]


def test_large_initial_tilt_is_reported_without_any_probe_motion():
    plant = FakePlant(base_deg=(13.0, 0.0))
    report = {"sentinel": "preserve"}
    with pytest.raises(ValueError, match="初始倾角超过"):
        flow.trim_ground_stance(plant, report)
    assert plant.commands == []
    assert "before" in report
    assert report["sentinel"] == "preserve"
