"""Evaluation must not let later episodes conceal a calibration failure."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_home_calibration.py"
SPEC = importlib.util.spec_from_file_location("home_calibration_check", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_first_episode_metrics_freeze_failed_world():
    metrics = module.FirstEpisodeMetrics(np.zeros((2, 2)))
    metrics.update(
        np.deg2rad([[1., 2.], [3., 4.]]), [26., 1.], [.12, .12],
        [["fell_over"], []], .05,
    )
    metrics.update(
        np.deg2rad([[0., 0.], [5., 6.]]), [0., 2.], [.01, .11],
        [[], []], .1,
    )
    report = metrics.report(completed_horizon=True)
    assert report["first_episode_horizon_survival_fraction"] == .5
    assert report["first_episode_stop_time_s"] == [.05, None]
    assert report["observed_first_episode_duration_s"] == [.05, .1]
    assert report["valid_samples_per_env"] == [1, 2]
    np.testing.assert_allclose(report["endpoint_error_deg_per_env_joint"], [[1, 2], [5, 6]])
    np.testing.assert_allclose(report["trajectory_mae_deg_per_joint"], [3, 4])
    np.testing.assert_allclose(report["trajectory_rmse_deg_per_joint"], np.sqrt([35 / 3, 56 / 3]))
    np.testing.assert_allclose(report["minimum_height_m_per_env"], [.12, .11])
    np.testing.assert_allclose(report["survivor_endpoint_mae_deg_per_joint"], [5, 6])


def test_nonfinite_state_is_failure_and_aborted_horizon_is_not_survival():
    metrics = module.FirstEpisodeMetrics(np.zeros((2, 2)))
    metrics.update([[float("nan"), 0], [0, 0]], [0, 0], [.12, .12], [[], []], .05)
    report = module._json_safe(metrics.report(completed_horizon=False))
    assert report["first_episode_stop_reasons"] == [["nonfinite_state"], []]
    assert report["first_episode_horizon_survival_fraction"] is None
    assert report["endpoint_error_deg_per_env_joint"][0][0] is None
    assert report["valid_samples_per_env"] == [0, 1]


def test_invalid_initial_error_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        module.FirstEpisodeMetrics(np.array([[float("nan")]]))
