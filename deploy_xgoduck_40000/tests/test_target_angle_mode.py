import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from microduck_deploy.config import load_config, calibration_arrays
from microduck_deploy.policy import HOME, q_to_ticks, PolicyContractError
from microduck_deploy.runtime import validate_targets, target_angle_warnings, StopRequested

ROOT = Path(__file__).resolve().parents[1]


def test_recorded_rejected_frame_passes_unchanged_only_when_enabled():
    path = ROOT.parent / 'reports/xgoduck_deploy_40000/trials/motion_20260930_111651_499457.session.json'
    recorded = json.loads(path.read_text())['rejected_policy_step']
    cfg = load_config(ROOT / 'robot.json')
    target, previous = recorded['target'], recorded['previous_target']
    with pytest.raises(StopRequested, match='right_hip_yaw'):
        validate_targets(target, previous, cfg)
    cfg['control']['log_only_target_angles'] = True
    before = copy.deepcopy(target)
    ticks = validate_targets(target, previous, cfg)
    np.testing.assert_array_equal(ticks, q_to_ticks(target, *calibration_arrays(cfg)))
    assert target == before
    assert any('right_hip_yaw' in item for item in target_angle_warnings(target, previous, cfg))


def test_large_target_jump_logged_without_clipping_but_invalid_targets_rejected():
    cfg = load_config(ROOT / 'robot.json')
    cfg['control']['log_only_target_angles'] = True
    target = HOME.copy()
    target[3] += np.deg2rad(60)
    expected = q_to_ticks(target, *calibration_arrays(cfg))
    np.testing.assert_array_equal(validate_targets(target, HOME, cfg), expected)
    assert any('单周期' in item for item in target_angle_warnings(target, HOME, cfg))
    with pytest.raises(StopRequested):
        validate_targets(np.full(14, np.nan), HOME, cfg)
    with pytest.raises(PolicyContractError):
        validate_targets(np.full(14, 100.0), HOME, cfg)


def test_switch_reaches_policy_only_and_requires_walk_confirmation():
    spec = importlib.util.spec_from_file_location('flow', ROOT / 'scripts/home_then_policy.py')
    flow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flow)
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: True, 'python', seconds=60, vx=.1,
                             log_only_target_angles=True) == 0
    assert '--log-only-target-angles' not in run.call_args_list[0].args[0]
    assert '--log-only-target-angles' in run.call_args_list[1].args[0]
    run.reset_mock()
    assert flow.run_sequence(run, lambda: False, 'python', seconds=60, vx=.1,
                             log_only_target_angles=True) == 130
    assert run.call_count == 1
