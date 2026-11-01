"""A selected measured endpoint must drive both walking inputs and outputs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from microduck_deploy.config import default_config, record_reference, calibration_arrays
from microduck_deploy.homing import plan_home
from microduck_deploy.policy import HOME, JOINT_LIMITS, build_obs61, action_to_targets, q_to_ticks, ticks_to_q


def script(name):
    path = Path(__file__).parents[1] / 'scripts' / (name + '.py')
    spec = importlib.util.spec_from_file_location('saved_home_test_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


save = script('save_home_for_walking')
supported = script('supported_home')
flow = script('home_then_policy')
trial = script('policy_trial')


@pytest.fixture
def source():
    cfg = default_config()
    cfg['imu_mount_verified'] = True
    for i, joint in enumerate(cfg['joints']):
        joint.update(id=(list(range(1, 10)) + list(range(11, 16)))[i],
                     direction=(-1 if i % 2 else 1), mapping_verified=True)
    original = np.arange(14) * 10 + 1900
    record_reference(cfg, original, 'home', HOME.copy())
    measured = original + np.arange(-7, 7)
    goals = measured + 10  # Must never replace the explicitly selected actual positions.
    report = {'completed':True, 'mode':'run', 'exit_state':save.HELD_EXIT,
              'duration_elapsed_s':10.02,
              'joint_mapping':[{k:j[k] for k in ('id','name','zero_tick','direction')} for j in cfg['joints']],
              'final_goal_ticks':{str(j['id']):int(t) for j,t in zip(cfg['joints'],goals)},
              'final_torque_readback':{str(j['id']):1 for j in cfg['joints']},
              'end_joint_positions':{'available':True,'fresh_readback':True,
                  'source':'normal_final_readback','exit_state_at_capture':save.HELD_EXIT,
                  'age_at_end_s':.004,
                  'joints':[{'id':j['id'],'name':j['name'],'actual_ticks':int(t),
                             'goal_ticks':int(g),'last_sent_goal_ticks':int(g),'status':0}
                            for j,t,g in zip(cfg['joints'],measured,goals)]}}
    return cfg,report,measured


def derive(cfg, report):
    return save.configuration_from_endpoint(cfg, report, source_path='/explicit/ground.json', source_sha256='a'*64)


def test_uses_measured_endpoint_preserves_original_and_maps_both_policy_directions(source):
    cfg,report,measured = source
    before = deepcopy(cfg)
    rebased = derive(cfg,report)
    assert cfg == before
    assert rebased['model'] == cfg['model'] and rebased['control'] == cfg['control']
    assert rebased['calibration']['home_ticks'] == measured.tolist()
    q = ticks_to_q(measured,*calibration_arrays(rebased))
    assert np.allclose(q,HOME,rtol=0,atol=1e-12)
    obs = build_obs61(np.zeros(3), [0,0,-1], q, np.zeros(14), np.zeros(14))
    assert obs.shape == (61,) and np.allclose(obs[6:20],0,atol=1e-7)
    assert q_to_ticks(action_to_targets(np.zeros(14)),*calibration_arrays(rebased)).tolist() == measured.tolist()
    action = np.linspace(-.02,.02,14)
    assert np.allclose(ticks_to_q(q_to_ticks(action_to_targets(action),*calibration_arrays(rebased)),
                                 *calibration_arrays(rebased)) - HOME, action, atol=np.pi/4096)


@pytest.mark.parametrize('problem', ['failed','short','old','missing','mixed_mapping','goal','torque','status','float_tick','outside'])
def test_refuses_failed_stale_or_incompatible_sources(source,problem):
    cfg,report,_ = source
    if problem=='failed':report['completed']=False
    if problem=='short':report['duration_elapsed_s']=5
    if problem=='old':report['end_joint_positions']['fresh_readback']=False
    if problem=='missing':report['end_joint_positions']['joints'].pop()
    if problem=='mixed_mapping':report['joint_mapping'][0]['zero_tick']+=1
    if problem=='goal':report['end_joint_positions']['joints'][0]['goal_ticks']+=1
    if problem=='torque':report['final_torque_readback']['1']=0
    if problem=='status':report['end_joint_positions']['joints'][0]['status']=1
    if problem=='float_tick':report['end_joint_positions']['joints'][0]['actual_ticks']+=.1
    if problem=='outside':report['end_joint_positions']['joints'][0]['actual_ticks']+=150
    with pytest.raises(ValueError):derive(cfg,report)


def test_home_plan_uses_selected_position_and_keeps_slow_bounded_path(source):
    cfg,report,measured = source
    rebased = derive(cfg,report)
    start = measured + 20
    plan = supported.plan_supported_move(rebased,start,log_only_feedback=True)
    assert plan['home_ticks'] == measured.tolist()
    assert plan['duration_s'] >= 6 and plan['peak_target_speed_deg_s'] <= 12
    path = np.array([start,*supported.path_ticks(start,measured,plan['steps'])])
    assert np.max(np.abs(np.diff(path,axis=0))) <= 4
    with pytest.raises(ValueError):supported.plan_supported_move(rebased,start,target='straight')
    rebased['calibration']['captured_ticks'][0] += 1
    with pytest.raises(ValueError):supported.plan_supported_move(rebased,start)


def assert_supported_path(plan, start, end):
    """The path must stay between its actual encoder endpoints without wrapping."""
    start, end = np.asarray(start), np.asarray(end)
    assert plan['start_ticks'] == start.tolist()
    assert plan['home_ticks'] == end.tolist()
    assert plan['duration_s'] >= 6
    assert plan['duration_s'] == plan['steps'] / 50
    assert plan['peak_target_speed_deg_s'] <= 12
    path = np.array([start, *supported.path_ticks(start, end, plan['steps'])])
    assert np.array_equal(path[-1], end)
    assert np.all(path >= np.minimum(start, end))
    assert np.all(path <= np.maximum(start, end))
    changes = np.diff(path, axis=0)
    assert np.all(changes * np.sign(end - start) >= 0)
    assert np.max(np.abs(changes)) <= 4


@pytest.mark.parametrize('posture', ['near', 'far', 'zero', 'maximum', 'mixed_edges'])
def test_saved_home_recovery_accepts_far_single_turn_postures(source, posture):
    cfg, report, measured = source
    rebased = derive(cfg, report)
    starts = {'near': measured + 20,
              'far': measured + 900,
              'zero': np.zeros(14, dtype=int),
              'maximum': np.full(14, 4095),
              'mixed_edges': np.tile([0, 4095], 7)}
    start = starts[posture]
    if posture != 'near':
        q = ticks_to_q(start, *calibration_arrays(rebased))
        assert np.max(np.abs(q - HOME)) > rebased['control']['max_start_distance_rad']
        assert np.any((q < JOINT_LIMITS[:, 0]) | (q > JOINT_LIMITS[:, 1]))
    plan = supported.plan_supported_move(rebased, start, log_only_feedback=True)
    assert_supported_path(plan, start, measured)
    if posture == 'near':
        assert plan['duration_s'] == 6
    else:
        assert plan['duration_s'] > 6


def test_saved_home_recovery_extends_beyond_thirty_seconds_without_encoder_wrap(source):
    cfg, _, _ = source
    destination = np.tile([5, 4090], 7)
    start = np.tile([4095, 0], 7)
    record_reference(cfg, destination, 'home', HOME.copy())
    plan = supported.plan_supported_move(cfg, start)
    assert plan['duration_s'] > 30
    assert_supported_path(plan, start, destination)


@pytest.mark.parametrize('bad_tick', [-1, 4096, np.nan, np.inf, 2000.5])
def test_saved_home_recovery_still_rejects_invalid_encoder_readings(source, bad_tick):
    cfg, report, measured = source
    start = measured.astype(float)
    start[0] = bad_tick
    with pytest.raises(ValueError, match='single-turn integer'):
        supported.plan_supported_move(derive(cfg, report), start)


@pytest.mark.parametrize('problem', ['short_start', 'ids', 'directions', 'reference', 'invalid_saved_tick'])
def test_saved_home_recovery_still_requires_matching_complete_reference(source, problem):
    cfg, report, start = source
    rebased = derive(cfg, report)
    if problem == 'short_start':
        start = start[:-1]
    elif problem == 'ids':
        rebased['calibration']['joint_ids'][0] = 99
    elif problem == 'directions':
        rebased['calibration']['directions'][0] *= -1
    elif problem == 'reference':
        rebased['calibration']['reference_q_rad'][0] += 0.01
    elif problem == 'invalid_saved_tick':
        rebased['calibration']['captured_ticks'][0] = np.nan
    with pytest.raises(ValueError):
        supported.plan_supported_move(rebased, start)


def test_walking_startup_still_requires_near_home_after_supported_recovery(source):
    cfg, report, measured = source
    rebased = derive(cfg, report)
    q = ticks_to_q(measured + 900, *calibration_arrays(rebased))
    with pytest.raises(ValueError, match='启动姿态离HOME太远'):
        plan_home(rebased, q)
    assert plan_home(rebased, HOME)['home_ticks'] == measured.tolist()


def test_custom_configuration_reaches_both_motion_stages_only_after_confirmation():
    calls = []
    def ready():
        assert len(calls)==1
        return True
    assert flow.run_sequence(lambda args:calls.append(args) or 0,ready,'python',
        seconds=0,vx=0,config_path=Path('/robot/new_home.json'),keyboard_args=['--keyboard'])==0
    for args in calls:
        assert args[args.index('--config')+1] == '/robot/new_home.json'
    assert '--calibrate-imu' in calls[1]
    args = trial.parse_args(['--config','/robot/new_home.json','--calibrate-imu'])
    assert args.config == Path('/robot/new_home.json')


def test_cli_refuses_to_overwrite_input_configuration(source,tmp_path):
    cfg,report,_ = source
    config=tmp_path/'robot.json';config.write_text(json.dumps(cfg))
    run=tmp_path/'ground.json';run.write_text(json.dumps(report))
    before=config.read_bytes()
    with pytest.raises(ValueError,match='new .json'):
        save.main(['--config',str(config),'--source-report',str(run),'--output',str(config)])
    assert config.read_bytes()==before
