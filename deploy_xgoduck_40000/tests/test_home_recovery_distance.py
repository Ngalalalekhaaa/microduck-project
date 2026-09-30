import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from supported_home import plan_straight_recovery, path_ticks
from microduck_deploy.config import load_config, calibration_arrays
from microduck_deploy.policy import HOME, q_to_ticks


def test_reported_right_knee_109_degree_recovery_is_planned_slowly():
    cfg = load_config(ROOT / 'robot.json')
    home = q_to_ticks(HOME, *calibration_arrays(cfg))
    start = home.copy()
    start[12] = 1758
    plan = plan_straight_recovery(cfg, start, target='home')
    assert plan['duration_s'] > 20
    assert plan['peak_target_speed_deg_s'] <= 8
    path = np.stack([start, *path_ticks(start, home, plan['steps'])])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 4
    np.testing.assert_array_equal(path[-1], home)
    assert np.all(np.diff(path[:, 12]) >= 0)


@pytest.mark.parametrize('tick', [0, 4095])
def test_single_turn_endpoints_keep_path_inside_encoder_range(tick):
    cfg = load_config(ROOT / 'robot.json')
    start = np.full(14, tick)
    plan = plan_straight_recovery(cfg, start, target='home')
    path = np.stack([start, *path_ticks(start, plan['home_ticks'], plan['steps'])])
    assert np.all((0 <= path) & (path <= 4095))
    assert np.max(np.abs(np.diff(path, axis=0))) <= 4


@pytest.mark.parametrize('tick', [-1, 4096, float('nan'), 2000.5])
def test_invalid_feedback_is_not_used_as_recovery_start(tick):
    cfg = load_config(ROOT / 'robot.json')
    with pytest.raises(ValueError):
        plan_straight_recovery(cfg, np.full(14, tick), target='home')
