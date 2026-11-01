"""Explicit feedback margin for the head pitch is isolated to forward rolls."""
import json
import math

import numpy as np
import pytest

from microduck_deploy.config import load_config, DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG
from microduck_deploy.policy import HOME, JOINT_LIMITS, JOINT_NAMES
from microduck_deploy.runtime import check_posture, StopRequested
from test_roulade_runtime import enabled_config, state


HEAD = JOINT_NAMES.index('head_pitch')


def cfg105():
    cfg = enabled_config()
    cfg['control']['roulade_head_pitch_max_deg'] = 105.
    return cfg


def test_old_default_and_missing_option_keep_original_feedback_bounds():
    cfg = enabled_config()
    cfg['control'].pop('roulade_head_pitch_max_deg', None)
    q = HOME.copy(); q[HEAD] = math.radians(99.404296875)
    assert DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG == pytest.approx(96.87549354156988)
    with pytest.raises(StopRequested, match='head_pitch'):
        check_posture(q, state(), cfg, policy_mode='roulade')


@pytest.mark.parametrize('mode', ['walking', 'stopping', 'sit', 'stand', 'ground_pick'])
def test_override_cannot_expand_head_feedback_bound_outside_roll(mode):
    q = HOME.copy(); q[HEAD] = math.radians(99.404296875)
    with pytest.raises(StopRequested, match='head_pitch'):
        check_posture(q, state(), cfg105(), policy_mode=mode)


def test_roll_allows_recorded_994_degree_state_but_stops_above_105():
    q = HOME.copy();q[HEAD] = math.radians(99.404296875)
    check_posture(q, state(), cfg105(), policy_mode='roulade')
    q[HEAD] = math.radians(105.)
    check_posture(q, state(), cfg105(), policy_mode='roulade')
    q[HEAD] = math.radians(105.01)
    with pytest.raises(StopRequested, match='实测105.01.*105.00'):
        check_posture(q, state(), cfg105(), policy_mode='roulade')


@pytest.mark.parametrize('joint', range(14))
def test_other_joint_upper_and_every_lower_bound_remain_enforced(joint):
    q = HOME.copy();q[joint] = JOINT_LIMITS[joint, 0] - .121
    with pytest.raises(StopRequested, match=JOINT_NAMES[joint]):
        check_posture(q, state(), cfg105(), policy_mode='roulade')
    if joint != HEAD:
        q = HOME.copy();q[joint] = JOINT_LIMITS[joint, 1] + .121
        with pytest.raises(StopRequested, match=JOINT_NAMES[joint]):
            check_posture(q, state(), cfg105(), policy_mode='roulade')


@pytest.mark.parametrize('value', [True, False, None, '105', [], {}, float('nan'),
                                 float('inf'), 0, -105, 96., 110.01])
def test_invalid_head_limit_rejected_in_config_and_runtime(tmp_path, value):
    cfg=cfg105();cfg['control']['roulade_head_pitch_max_deg']=value
    path=tmp_path/'robot.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='roulade_head_pitch_max_deg'):
        load_config(path)
    with pytest.raises(StopRequested, match='roulade_head_pitch_max_deg'):
        check_posture(HOME, state(), cfg, policy_mode='roulade')


def test_head_option_does_not_mutate_global_limits_or_mask_roll_motion_bounds():
    original = JOINT_LIMITS.copy()
    q = HOME.copy();q[HEAD] = math.radians(104.)
    check_posture(q, state(math.pi), cfg105(), policy_mode='roulade')
    assert np.array_equal(original, JOINT_LIMITS)
    with pytest.raises(StopRequested, match='角速度'):
        check_posture(q, state(gyro=(0.,12.1,0.)), cfg105(), policy_mode='roulade')


@pytest.mark.parametrize('value', [DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG, 100., 105., 110.])
def test_valid_head_limit_is_loaded_without_changing_calibration(tmp_path, value):
    cfg=enabled_config();cfg['control']['roulade_head_pitch_max_deg']=value
    path=tmp_path/'robot.json';path.write_text(json.dumps(cfg))
    assert load_config(path)==cfg
