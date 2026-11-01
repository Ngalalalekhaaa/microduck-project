"""Known geometry and plant responses for HOME assistance; no hardware access."""
import numpy as np
import pytest

from microduck_deploy.home_assist import (
    HomeAssistError, TRIM_BASIS, central_difference_jacobian,
    gravity_to_roll_pitch, pose_for_trim, q_check, solve_trim_step,
)
from microduck_deploy.policy import HOME, JOINT_NAMES


def gravity_for_attitude(roll, pitch):
    # Independently expand Ry(pitch) Rx(roll), transpose, then multiply down.
    return np.array([
        np.sin(pitch),
        -np.sin(roll) * np.cos(pitch),
        -np.cos(roll) * np.cos(pitch),
    ])


@pytest.mark.parametrize("roll_deg,pitch_deg", [
    (0, 0), (15, 0), (-15, 0), (0, 20), (0, -20), (12, -17),
])
def test_gravity_recovers_signed_body_attitude(roll_deg, pitch_deg):
    attitude = np.deg2rad([roll_deg, pitch_deg])
    np.testing.assert_allclose(
        gravity_to_roll_pitch(gravity_for_attitude(*attitude)),
        attitude, atol=1e-12,
    )


def test_gravity_accepts_small_normalization_error_but_rejects_no_gravity():
    np.testing.assert_allclose(gravity_to_roll_pitch([0, 0, -1.01]), [0, 0])
    for gravity in ([0, 0, 0], [0, 0, -9.81], [1, 0, 0], [0, np.nan, -1]):
        with pytest.raises(HomeAssistError):
            gravity_to_roll_pitch(gravity)


def test_stance_basis_changes_only_selected_leg_joints():
    adjustment = np.deg2rad([2.0, -1.0])
    pose = pose_for_trim(adjustment)
    unchanged = [0, 2, 3, 5, 6, 7, 8, 9, 11, 12]
    np.testing.assert_array_equal(pose[unchanged], HOME[unchanged])
    np.testing.assert_allclose(
        pose[[4, 13, 1, 10]] - HOME[[4, 13, 1, 10]],
        np.deg2rad([2.0, -2.0, -1.0, -1.0]), atol=1e-15,
    )
    assert not TRIM_BASIS.flags.writeable


def test_joint_check_reports_tracking_without_recalibration():
    q = HOME.copy()
    q[4] += np.deg2rad(3)
    result = q_check(q, ids=list(range(1, 15)))
    assert result["max_abs_error_deg"] == pytest.approx(3)
    assert result["rms_error_deg"] == pytest.approx(3 / np.sqrt(14))
    assert result["joints"][4]["id"] == 5
    assert result["joints"][4]["name"] == JOINT_NAMES[4]
    assert result["joints"][4]["error_deg"] == pytest.approx(3)
    np.testing.assert_array_equal(pose_for_trim([0, 0]), HOME)


def test_probe_rows_become_response_columns_and_reduce_linear_plant_error():
    plant = np.array([[0.15, -0.85], [-0.95, 0.1]])
    initial = np.deg2rad([2, -1.5])
    probe = np.deg2rad(1)
    plus = np.stack([initial + plant[:, i] * probe for i in range(2)])
    minus = np.stack([initial - plant[:, i] * probe for i in range(2)])
    jacobian = central_difference_jacobian(plus, minus, probe)
    np.testing.assert_allclose(jacobian, plant, atol=1e-13)
    trim = np.zeros(2)
    for _ in range(8):
        measured = initial + plant @ trim
        step = solve_trim_step(jacobian, -measured, trim)
        assert np.max(np.abs(step)) <= np.deg2rad(0.5) + 1e-12
        trim += step
    assert np.linalg.norm(initial + plant @ trim) < np.deg2rad(0.01)


@pytest.mark.parametrize("jacobian", [
    np.zeros((2, 2)), np.array([[1, 1], [1, 1]]), np.eye(2) * 0.001,
    np.diag([5.0, 0.1]), np.eye(2) * 11,
])
def test_unresponsive_ambiguous_or_excessive_probe_cannot_be_inverted(jacobian):
    with pytest.raises(HomeAssistError):
        solve_trim_step(jacobian, [0.02, 0.02], [0, 0])


def test_step_and_total_adjustments_are_bounded_and_can_move_back_inward():
    current = np.deg2rad([3.9, -4.0])
    step = solve_trim_step(np.eye(2), np.deg2rad([50, -50]), current)
    np.testing.assert_allclose(current + step, np.deg2rad([4, -4]))
    assert np.max(np.abs(step)) <= np.deg2rad(0.5) + 1e-12
    inward = solve_trim_step(np.eye(2), np.deg2rad([-50, 50]), current + step)
    assert inward[0] < 0 < inward[1]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_cannot_become_motion_targets(bad):
    with pytest.raises(HomeAssistError):
        pose_for_trim([0, bad])
    with pytest.raises(HomeAssistError):
        q_check(np.full(14, bad))
    with pytest.raises(HomeAssistError):
        central_difference_jacobian([[0, bad], [0, 0]], np.zeros((2, 2)), 0.01)
    with pytest.raises(HomeAssistError):
        solve_trim_step(np.eye(2), [0, bad], [0, 0])
    with pytest.raises(HomeAssistError):
        solve_trim_step(np.eye(2), [0, 0], [0, bad])
    with pytest.raises(HomeAssistError):
        solve_trim_step(np.eye(2), [0, 0], [0, 0], damping=bad)


def test_invalid_shapes_probe_sizes_and_existing_trim_are_rejected():
    with pytest.raises(HomeAssistError):
        pose_for_trim([0, 0, 0])
    with pytest.raises(HomeAssistError):
        central_difference_jacobian(np.zeros((2, 2)), np.zeros((2, 2)), 0)
    with pytest.raises(HomeAssistError):
        central_difference_jacobian(np.zeros((2, 2)), np.zeros((2, 2)), np.deg2rad(6))
    with pytest.raises(HomeAssistError):
        solve_trim_step(np.eye(2), [0, 0], np.deg2rad([5, 0]))
