"""Assembly-reference and Xbox HOME checks, with no serial or motor access.

Synthetic geometry below is a mathematical test fixture, never a robot profile.
"""
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import supported_home as supported
from microduck_deploy import config, homing, runtime
from microduck_deploy.policy import HOME, JOINT_LIMITS, q_to_ticks, ticks_to_q
from microduck_deploy.servo import ServoFeedback


def assembly_config(*, left_knee_deg=-60):
    from microduck_deploy.assembly import record_assembly
    cfg = config.default_config()
    reference = HOME.copy()
    reference[3] = np.deg2rad(left_knee_deg)
    for i, joint in enumerate(cfg['joints']):
        joint.update(id=i + 1, direction=1 if i % 2 == 0 else -1,
                     mapping_verified=True)
    profile = {
        'schema': 1, 'type': 'microduck_assembly_geometry',
        'geometry_source': 'SYNTHETIC TEST ONLY, NOT A ROBOT ASSEMBLY DATUM',
        'hole_pitch_deg': 45,
        'joints': [dict(name=j['name'], id=j['id'], direction=j['direction'],
                       center_tick=2048, base_q_deg=float(np.rad2deg(reference[i])),
                       hole_steps=0, datum='SYNTHETIC TEST ONLY', verified=True)
                   for i, j in enumerate(cfg['joints'])],
    }
    record_assembly(cfg, profile)
    return cfg, reference


def test_assembly_reference_produces_home_and_preserves_config():
    cfg, reference = assembly_config()
    before = copy.deepcopy(cfg)
    plan = homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=12)
    assert plan['from_pose'] == 'assembly'
    assert plan['duration_s'] == 12
    np.testing.assert_array_equal(plan['start_ticks'], np.full(14, 2048))
    expected = q_to_ticks(HOME, *config.calibration_arrays(cfg))
    np.testing.assert_array_equal(plan['home_ticks'], expected)
    assert plan['peak_target_speed_deg_s'] <= 12
    assert cfg == before


def test_initial_acceptance_rejects_six_degree_misalignment():
    cfg, reference = assembly_config()
    reference[3] += np.deg2rad(6)
    with pytest.raises(ValueError, match='偏离装配标定参考超过5'):
        homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=12)


@pytest.mark.parametrize('seconds', [0, 7.99, 30.01, float('inf'), float('nan')])
def test_reference_duration_limits_apply_to_assembly(seconds):
    cfg, reference = assembly_config()
    with pytest.raises(ValueError, match='8..30秒'):
        homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=seconds)


def test_large_reference_displacement_requires_slower_move():
    cfg, reference = assembly_config(left_knee_deg=-80)
    with pytest.raises(ValueError, match='峰值速度超过12'):
        homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=8)
    plan = homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=30)
    assert plan['peak_target_speed_deg_s'] <= 12


@pytest.mark.parametrize('daily', [False, True])
def test_mutated_geometry_cannot_be_used_for_either_home_path(daily):
    cfg, reference = assembly_config()
    cfg['calibration']['assembly_profile']['joints'][0]['base_q_deg'] += 1
    with pytest.raises(ValueError):
        supported.plan_supported_move(cfg, np.full(14, 2048),
                                      log_only_feedback=daily)


def test_strict_script_uses_assembly_reference_and_refuses_straight():
    cfg, reference = assembly_config()
    plan = supported.plan_supported_move(cfg, np.full(14, 2048))
    assert plan['from_pose'] == 'assembly'
    assert plan['duration_s'] == 12
    for daily in (False, True):
        with pytest.raises(ValueError, match='不支持 --target straight'):
            supported.plan_supported_move(cfg, np.full(14, 2048), target='straight',
                                          log_only_feedback=daily)


@pytest.mark.parametrize('neck_offset_ticks', [0, -45])
def test_initial_large_neck_reference_automatically_extends_move(neck_offset_ticks):
    from microduck_deploy.assembly import record_assembly
    cfg, _ = assembly_config()
    profile = copy.deepcopy(cfg['calibration']['assembly_profile'])
    for row in profile['joints']:
        row['base_q_deg'] = 0.0
    profile['joints'][5]['base_q_deg'] = -89.0
    record_assembly(cfg, profile)
    start = np.full(14, 2048)
    start[5] += neck_offset_ticks  # Also cover a measured start within ±5°.
    q = ticks_to_q(start, *config.calibration_arrays(cfg))
    plan = supported.plan_supported_move(cfg, start)
    minimum = 1.5 * float(np.rad2deg(np.max(np.abs(HOME - q)))) / 12.0
    assert plan['from_pose'] == 'assembly'
    assert 13 < plan['duration_s'] < 14
    assert minimum <= plan['duration_s'] < minimum + .02
    assert plan['peak_target_speed_deg_s'] <= 12
    steps = round(plan['duration_s'] * 50)
    path = np.stack([start, *supported.path_ticks(start, plan['home_ticks'], steps)])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 4
    np.testing.assert_array_equal(path[-1], plan['home_ticks'])


def test_daily_home_need_not_start_at_assembly_reference():
    cfg, reference = assembly_config()
    q = HOME.copy()
    q[3] += np.deg2rad(25)
    start = q_to_ticks(q, *config.calibration_arrays(cfg))
    with pytest.raises(ValueError, match='偏离装配标定参考'):
        supported.plan_supported_move(cfg, start)
    plan = supported.plan_supported_move(cfg, start, log_only_feedback=True)
    assert plan['duration_s'] == 12
    assert plan['peak_target_speed_deg_s'] <= 12
    path = np.stack([start, *supported.path_ticks(start, plan['home_ticks'], plan['steps'])])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 4
    angles = np.stack([ticks_to_q(row, *config.calibration_arrays(cfg)) for row in path])
    assert np.all(angles >= JOINT_LIMITS[:, 0])
    assert np.all(angles <= JOINT_LIMITS[:, 1])


def test_daily_home_keeps_model_limits():
    cfg, reference = assembly_config()
    q = HOME.copy()
    q[0] = JOINT_LIMITS[0, 1] + np.deg2rad(2)
    start = q_to_ticks(q, *config.calibration_arrays(cfg))
    with pytest.raises(ValueError, match='模型关节限位'):
        supported.plan_supported_move(cfg, start, log_only_feedback=True)


def test_daily_large_displacement_extends_timing_without_fast_target_steps():
    cfg, reference = assembly_config()
    q = HOME.copy()
    q[7] = np.deg2rad(160)
    start = q_to_ticks(q, *config.calibration_arrays(cfg))
    plan = supported.plan_supported_move(cfg, start, log_only_feedback=True)
    assert 19 < plan['duration_s'] < 21
    assert plan['peak_target_speed_deg_s'] <= 12
    path = np.stack([start, *supported.path_ticks(start, plan['home_ticks'], plan['steps'])])
    assert np.max(np.abs(np.diff(path, axis=0))) <= 4
    np.testing.assert_array_equal(path[-1], plan['home_ticks'])


@pytest.mark.parametrize('tick', [-1, 4096, 2048.5, float('nan')])
def test_daily_home_rejects_non_single_turn_feedback(tick):
    cfg, reference = assembly_config()
    with pytest.raises(ValueError, match='single-turn integer'):
        supported.plan_supported_move(cfg, np.full(14, tick), log_only_feedback=True)


def test_reference_home_hardware_limits_still_apply():
    cfg, reference = assembly_config()
    plan = homing.plan_home(cfg, reference, from_pose='assembly', move_seconds=12)
    configurations = {sid: dict(torque=0, mode=4, angle_resolution=1,
                                position_min=0, position_max=4095)
                      for sid in range(1, 15)}
    configurations[4]['position_min'] = 2048
    with pytest.raises(ValueError, match='硬件限位'):
        homing.check_servo_limits(plan, configurations)


def test_runtime_accepts_assembly_reference_without_loading_policy():
    from test_straight_homing import straight_rig
    cfg, reference = assembly_config()
    cfg['imu_mount_verified'] = True
    with straight_rig(cfg=cfg) as rig, patch.object(
            runtime, 'OnnxPolicy', side_effect=AssertionError('No policy for HOME')):
        rig.bus.positions = dict(zip(rig.bus.ids, [2048] * 14))
        rig.bus.goals = dict(rig.bus.positions)
        rig.bus.start_q = reference.copy()
        rig.ctl.policy = None
        rig.ctl.execute(seconds=.08, motion=True, home_only=True,
                        from_pose='assembly', move_seconds=12)
        assert rig.ctl.session['home_plan']['from_pose'] == 'assembly'
        assert len(rig.ctl.session['home_samples']) == 600
        np.testing.assert_allclose(
            ticks_to_q([rig.bus.positions[sid] for sid in rig.bus.ids],
                       *config.calibration_arrays(cfg)),
            HOME, atol=np.pi / 4096)


@pytest.mark.parametrize('overcurrent', [False, True])
def test_xbox_flag_keeps_strict_assembly_feedback_and_reports_effective_mode(
        tmp_path, monkeypatch, overcurrent):
    cfg, reference = assembly_config()
    writes = []

    class Bus:
        def __init__(self, **options):
            self.communication_retries = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read_configuration(self, sid):
            return dict(torque=0, mode=4, angle_resolution=1,
                        position_min=0, position_max=4095)

        def read_feedback_many(self, ids, *, sync):
            now = supported.time.monotonic()
            return {sid: ServoFeedback(sid, 2048, 0, 0, 7.4, 30, 0,
                                      0, 2048, 900 if overcurrent else 0, now)
                    for sid in ids}

        def sync_positions(self, targets):
            writes.append(('positions', targets))
            raise AssertionError('Test must not move')

        def set_torque(self, ids, enabled, *, verify):
            writes.append(('torque', enabled))
            raise AssertionError('Test must not enable torque')

    argv = ['supported_home.py', '--log-only-feedback']
    if overcurrent:
        argv.append('--move')
    monkeypatch.setattr(supported, 'DEPLOY', tmp_path)
    monkeypatch.setattr(supported.sys, 'argv', argv)
    monkeypatch.setattr(supported, 'load_config', lambda _: cfg)
    monkeypatch.setattr(supported, 'PositionOnlyBus', Bus)
    monkeypatch.setattr(supported.signal, 'signal', lambda *args: None)
    if overcurrent:
        with pytest.raises(ValueError, match='voltage/temperature/current'):
            supported.main()
    else:
        supported.main()
        report = json.loads(next((tmp_path / 'logs').glob('supported_home_*.json')).read_text())
        assert report['log_only_feedback_requested'] is True
        assert report['log_only_feedback'] is False
        assert report['software_current_stop_ma'] == 800
    assert writes == []
