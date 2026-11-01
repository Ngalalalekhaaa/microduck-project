"""Offline: use a successful 10-second run's actual endpoint as software HOME.

Writes a separate walking configuration, never motors or the input config.
The reference is user-selected; it is not independently measured mechanical HOME.
Encoder observations and policy targets both use the resulting calibration.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from microduck_deploy.config import calibration_arrays, load_config, record_reference, require_calibration
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks, ticks_to_q

HELD_EXIT = 'last_goal_readback_verified_torque_kept_on'


def configuration_from_endpoint(cfg, report, *, source_path, source_sha256):
    require_calibration(cfg)
    if ([j['name'] for j in cfg['joints']] != list(JOINT_NAMES)
            or report.get('completed') is not True or report.get('error')
            or report.get('exit_state') != HELD_EXIT
            or report.get('mode') not in ('run', 'observe_existing_goal', 'move_to_saved_goal')):
        raise ValueError('Require a successful policy/follow session with verified held final goals')
    duration = report.get('duration_elapsed_s')
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration < 9.9):
        raise ValueError('Require a completed 10-second session')
    mapping = report.get('joint_mapping')
    expected = [{k:j[k] for k in ('id', 'name', 'zero_tick', 'direction')} for j in cfg['joints']]
    if mapping != expected:
        raise ValueError('Source calibration/ID/direction mapping differs from the current configuration')
    snapshot = report.get('end_joint_positions', {})
    age = snapshot.get('age_at_end_s')
    if (snapshot.get('available') is not True or snapshot.get('fresh_readback') is not True
            or snapshot.get('source') != 'normal_final_readback'
            or snapshot.get('exit_state_at_capture') != HELD_EXIT
            or isinstance(age, bool) or not isinstance(age, (int, float))
            or not math.isfinite(age) or not 0 <= age <= .1):
        raise ValueError('Require fresh normal final readback, not old fallback positions')
    rows = snapshot.get('joints', [])
    if len(rows) != 14:
        raise ValueError('Final positions must cover exactly 14 joints')
    ids = [j['id'] for j in cfg['joints']]
    goals = report.get('final_goal_ticks', {})
    torque = report.get('final_torque_readback', {})
    if set(goals) != set(map(str, ids)) or set(torque) != set(map(str, ids)):
        raise ValueError('Require final goal and torque readbacks for all 14 joints')
    ticks = []
    for row, joint in zip(rows, cfg['joints']):
        sid = str(joint['id'])
        actual = row.get('actual_ticks')
        goal = row.get('goal_ticks')
        if (row.get('id') != joint['id'] or row.get('name') != joint['name']
                or type(actual) is not int or not 0 <= actual <= 4095
                or type(goal) is not int or not 0 <= goal <= 4095
                or goal != goals[sid] or goal != row.get('last_sent_goal_ticks')
                or type(torque[sid]) is not int or torque[sid] != 1 or row.get('status') != 0):
            raise ValueError(f'ID{sid}: invalid or inconsistent final position/goal/torque')
        ticks.append(actual)
    old_q = ticks_to_q(ticks, *calibration_arrays(cfg))
    if np.max(np.abs(old_q - HOME)) > np.deg2rad(8) + math.pi / 4096 + 1e-9:
        raise ValueError('Selected endpoint exceeds the HOME correction envelope')
    result = deepcopy(cfg)
    record_reference(result, ticks, 'home', HOME.copy())
    result['home_endpoint_reference'] = {
        'source_report': str(source_path), 'source_sha256': source_sha256,
        'position_field': 'end_joint_positions.joints.actual_ticks',
        'reference_kind': 'user_selected_software_HOME',
        'independent_mechanical_accuracy_verified': False,
        'source_target_reached': report.get('target_reached'),
        'previous_joint_mapping': expected,
        'previous_calibration': deepcopy(cfg['calibration']),
        'old_home_ticks': q_to_ticks(HOME, *calibration_arrays(cfg)).tolist(),
        'new_home_ticks': ticks,
    }
    if q_to_ticks(HOME, *calibration_arrays(result)).tolist() != ticks:
        raise ValueError('New HOME does not reproduce the selected encoder positions exactly')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'robot.json')
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    raw = args.source_report.read_bytes()
    selected = configuration_from_endpoint(cfg, json.loads(raw),
        source_path=args.source_report.resolve(), source_sha256=hashlib.sha256(raw).hexdigest())
    if args.output.suffix.lower() != '.json' or args.output.exists():
        raise ValueError('Output must be a new .json file; input configurations are never overwritten')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'config':str(args.output.resolve()),
                      'new_home_ticks':selected['calibration']['home_ticks'],
                      'walking_model':selected['model'],
                      'ram_gains':{k:selected['control']['ram_'+k] for k in ('p','d','i')},
                      'hardware_opened':False, 'actuator_writes':0},ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
