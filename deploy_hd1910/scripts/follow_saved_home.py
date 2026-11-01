"""Supported follow-up of a completed HOME run's saved target; no RL or IMU.

Default is a read-only preview. With --move, the operator must support the
trunk and keep the feet clear. Existing matching held targets are observed
without rewriting them. Relaxed joints approach the saved target slowly.
Success keeps torque ON; failure after takeover attempts verified torque OFF.
This temporary target never becomes a calibrated mechanical HOME.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from microduck_deploy.config import calibration_arrays, load_config, require_calibration
from microduck_deploy.home_calibration_hardware import validate_hold_configuration
from microduck_deploy.home_calibration_runtime import (
    _joint_position_snapshot, _write_end_positions, _write_report,
)
from microduck_deploy.policy import HOME, JOINT_LIMITS, JOINT_NAMES, ticks_to_q
from home_calibration import exclusive_hardware, stop_signals
from supported_home import PositionOnlyBus

DT = .05
MAX_STEP_TICKS = 2  # 0.17578125 degrees, below the 0.25-degree policy step.
HELD_EXIT = 'last_goal_readback_verified_torque_kept_on'


class FollowSavedBus(PositionOnlyBus):
    """The default/preview instance rejects all writes at the packet boundary."""

    writes_allowed = False

    def _sync_ram(self, address, size, values):
        if not self.writes_allowed:
            raise ValueError('Saved-target preview is read-only')
        return super()._sync_ram(address, size, values)


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _ticks(value, label):
    if type(value) is not int or not 0 <= value <= 4095:
        raise ValueError(f'{label} must be an integer single-turn encoder position')
    return value


def _check_output_paths(report_path, protected):
    path = Path(report_path)
    if path.suffix.lower() != '.json':
        raise ValueError('--report must end in .json')
    outputs = [path, path.with_suffix(path.suffix + '.tmp'),
               path.with_suffix('.positions.csv'), path.with_suffix('.positions.csv.tmp')]
    for output in outputs:
        if output.exists() or any(output.resolve() == item.resolve() or
                                  (output.exists() and item.exists() and output.samefile(item))
                                  for item in map(Path, protected)):
            raise ValueError('Output paths must be new and must not overwrite the source, configuration or model')


def _mapping(cfg):
    require_calibration(cfg, imu=False)
    if [joint['name'] for joint in cfg['joints']] != list(JOINT_NAMES):
        raise ValueError('Configuration must contain the 14 model joints in model order')
    return [{key: joint[key] for key in ('name', 'id', 'zero_tick', 'direction')}
            for joint in cfg['joints']]


def _bounded_positions(ticks, cfg, configurations=None, *, saved_target=False):
    ids = [joint['id'] for joint in cfg['joints']]
    if set(ticks) != set(ids):
        raise ValueError('Positions must cover exactly the 14 body joints')
    values = [_ticks(ticks[sid], f'ID{sid} position') for sid in ids]
    q = ticks_to_q(values, *calibration_arrays(cfg))
    if not np.isfinite(q).all() or np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        raise ValueError('Position exceeds model joint limits')
    # Saved commands were rounded to whole encoder counts. Permit at most
    # half a count beyond the continuous policy's +/-8-degree envelope.
    if saved_target and np.max(np.abs(q - HOME)) > math.radians(8) + math.pi / 4096 + 1e-9:
        raise ValueError('Saved target exceeds HOME +/-8 degrees (including encoder rounding)')
    if configurations is not None:
        for sid, tick in zip(ids, values):
            item = configurations[sid]
            if not item['position_min'] <= tick <= item['position_max']:
                raise ValueError(f'ID{sid}: position exceeds hardware joint limits')
    return q


def load_saved_target(source_report, cfg):
    """Load only an explicit, successful ground-policy report; never find latest."""
    path = Path(source_report).resolve()
    raw = path.read_bytes()
    report = json.loads(raw)
    expected_mapping = _mapping(cfg)
    if (report.get('mode') != 'run' or report.get('completed') is not True
            or report.get('exit_state') != HELD_EXIT or report.get('error')
            or report.get('watchdog_triggered') is True):
        raise ValueError('Source must be a completed, successful run with verified held final goals')
    mapping = report.get('joint_mapping')
    if not isinstance(mapping, list) or len(mapping) != 14:
        raise ValueError('Source must record all 14 joint calibrations')
    for recorded, current in zip(mapping, expected_mapping):
        if (not isinstance(recorded, dict)
                or any(recorded.get(key) != current[key] for key in current)
                or type(recorded.get('id')) is not int
                or type(recorded.get('direction')) is not int):
            raise ValueError('Source joint ID/name/zero/direction differs from current calibration')
        _finite(recorded['zero_tick'], 'Recorded zero_tick')
    contract = report.get('model', {}).get('contract', {})
    home = np.asarray(contract.get('home_angles_rad'), dtype=float)
    if (contract.get('joint_names') != list(JOINT_NAMES) or home.shape != (14,)
            or not np.isfinite(home).all() or not np.allclose(home, HOME, rtol=0, atol=1e-7)):
        raise ValueError('Source model HOME/joint contract does not match this deployment')
    ids = [joint['id'] for joint in cfg['joints']]
    keys = {str(sid) for sid in ids}
    raw_goals = report.get('final_goal_ticks', {})
    if not isinstance(raw_goals, dict) or set(raw_goals) != keys:
        raise ValueError('Source final_goal_ticks must contain exactly the 14 body IDs')
    goals = {sid: _ticks(raw_goals[str(sid)], f'ID{sid} saved goal') for sid in ids}
    torque = report.get('final_torque_readback', {})
    if (set(torque) != keys
            or any(type(value) is not int or value != 1 for value in torque.values())):
        raise ValueError('Source must include verified final torque ON for all 14 joints')
    snapshot = report.get('end_joint_positions', {})
    if (snapshot.get('available') is not True or snapshot.get('fresh_readback') is not True
            or snapshot.get('source') != 'normal_final_readback'
            or snapshot.get('exit_state_at_capture') != HELD_EXIT):
        raise ValueError('Source lacks a successful fresh final position readback')
    age = _finite(snapshot.get('age_at_end_s'), 'Source final readback age')
    if not 0 <= age <= .1:
        raise ValueError('Source final readback was stale at its own session end')
    rows = snapshot.get('joints', [])
    if not isinstance(rows, list) or len(rows) != 14:
        raise ValueError('Source final readback must contain all 14 joints')
    for row, joint in zip(rows, cfg['joints']):
        sid = joint['id']
        if (type(row.get('id')) is not int or row['id'] != sid or row.get('name') != joint['name']
                or _ticks(row.get('goal_ticks'), 'Final readback goal') != goals[sid]
                or _ticks(row.get('last_sent_goal_ticks'), 'Last sent goal') != goals[sid]
                or row.get('status') != 0):
            raise ValueError(f'ID{sid}: final target/readback/joint mapping is inconsistent')
        _ticks(row.get('actual_ticks'), 'Source actual position')
    target_q = _bounded_positions(goals, cfg, saved_target=True)
    return {'source_report': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'goal_ticks': goals, 'target_q': target_q.tolist(),
            'joint_mapping': expected_mapping, 'raw_report': report}


def _configuration(configurations, cfg, expected_p):
    ids = [joint['id'] for joint in cfg['joints']]
    if set(configurations) != set(ids):
        raise ValueError('Configuration must cover all 14 body servos')
    for sid, item in configurations.items():
        if (item.get('register_profile') != 'hls_2' or item.get('mode') != 4
                or item.get('angle_resolution') != 1 or type(item.get('torque')) is not int
                or item['torque'] not in (0, 1)):
            raise ValueError(f'ID{sid}: require HLS_2 mode 4, resolution 1 and known torque state')
        low, high = item.get('position_min'), item.get('position_max')
        if type(low) is not int or type(high) is not int or not 0 <= low < high <= 4095:
            raise ValueError(f'ID{sid}: invalid single-turn hardware limits')
    states = {item['torque'] for item in configurations.values()}
    if states not in ({0}, {1}):
        raise ValueError('Mixed torque states are refused; support the robot and relax first')
    enabled = states == {1}
    if enabled:
        validate_hold_configuration(configurations, cfg, expected_p=expected_p)
    elif any(item.get('ram_gains', {}).get('i') != 0 for item in configurations.values()):
        raise ValueError('Relaxed start requires existing RAM I=0')
    return enabled


def _read_checked(bus, cfg, configurations, clock, expected_goals=None):
    ids = [joint['id'] for joint in cfg['joints']]
    began = clock()
    feedback = bus.read_feedback_many(ids, sync=True)
    now = clock()
    if now - began > .1:
        raise ValueError('Servo read cycle exceeded 100 ms')
    if set(feedback) != set(ids):
        raise ValueError('Feedback must contain exactly the 14 body IDs')
    lower, upper = cfg['control']['servo_voltage_range']
    lower, upper = _finite(lower, 'Minimum voltage'), _finite(upper, 'Maximum voltage')
    maximum_temperature = _finite(cfg['control']['servo_max_temperature_c'], 'Maximum temperature')
    if not 0 < lower < upper or maximum_temperature <= 0:
        raise ValueError('Invalid voltage/temperature bounds')
    timestamps = []
    actual = {}
    for sid in ids:
        value = feedback[sid]
        if value.servo_id != sid or value.status != 0:
            raise ValueError(f'ID{sid}: mismatched ID or servo fault status {value.status}')
        age = now - _finite(value.received_at, f'ID{sid} timestamp')
        if value.received_at <= 0 or not 0 <= age <= .1:
            raise ValueError(f'ID{sid}: stale or future-dated feedback')
        timestamps.append(value.received_at)
        voltage = _finite(value.voltage_v, f'ID{sid} voltage')
        temperature = _finite(value.temperature_c, f'ID{sid} temperature')
        _finite(value.current_ma, f'ID{sid} current')  # Deliberately log-only, not zeroed.
        if not lower <= voltage <= upper or temperature >= maximum_temperature:
            raise ValueError(f'ID{sid}: voltage/temperature outside operating bounds')
        actual[sid] = _ticks(value.position_ticks, f'ID{sid} actual')
        goal = _ticks(value.goal_ticks, f'ID{sid} goal')
        if expected_goals is not None and goal != expected_goals[sid]:
            raise ValueError(f'ID{sid}: goal readback differs from expected target')
    if max(timestamps) - min(timestamps) > .05:
        raise ValueError('Servo feedback sweep exceeds 50 ms')
    _bounded_positions(actual, cfg, configurations)
    return feedback


def plan_follow(start, target):
    """At least six seconds, with every quantized command step <=2 ticks."""
    start, target = np.asarray(start, dtype=int), np.asarray(target, dtype=int)
    if np.max(np.abs(start - target)) * 360 / 4096 > 15:
        raise ValueError('Starting position is more than 15 degrees from the saved goal')
    steps = max(120, math.ceil(1.5 * float(np.max(np.abs(target - start))) / MAX_STEP_TICKS))
    while True:
        t = np.arange(steps + 1) / steps
        path = np.rint(start + (t * t * (3 - 2 * t))[:, None] * (target - start)).astype(int)
        if np.max(np.abs(np.diff(path, axis=0))) <= MAX_STEP_TICKS:
            return path, steps * DT
        steps += 1


def run_follow(cfg, source, bus, *, move=False, supported_confirmed=False, seconds=10.,
               expected_p=8, report_path=None, clock=time.monotonic, sleep=time.sleep,
               stop_event=None):
    if type(expected_p) is not int or expected_p not in (5, 8):
        raise ValueError('Expected RAM P must be 5 or 8')
    if not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError('Duration must be greater than zero and at most 10 seconds')
    if move and not supported_confirmed:
        raise ValueError('--move requires --supported-confirmed: support the trunk with feet clear')
    # Revalidate the exact source at the hardware boundary, including changes
    # between CLI parsing and lock acquisition. There is no latest-file fallback.
    checked = load_saved_target(source['source_report'], cfg)
    if checked['source_sha256'] != source['source_sha256'] or checked['goal_ticks'] != source['goal_ticks']:
        raise ValueError('Source report changed after validation')
    if report_path is not None:
        protected = [Path(checked['source_report'])]
        source_model = checked['raw_report'].get('model', {}).get('model')
        if source_model:
            protected.append(Path(source_model))
        _check_output_paths(report_path, protected)
    ids = [joint['id'] for joint in cfg['joints']]
    targets = dict(checked['goal_ticks'])
    stop = stop_event if stop_event is not None else threading.Event()
    report = {'task': 'supported_follow_saved_home', 'mode': 'preview', 'completed': False,
              'move_requested': move, 'supported_confirmed': supported_confirmed,
              'source_report': checked['source_report'], 'source_sha256': checked['source_sha256'],
              'joint_mapping': _mapping(cfg), 'saved_goal_ticks': targets,
              'requested_ram_gains': {'p': expected_p, 'd': 0, 'i': 0},
              'duration_requested_s': seconds, 'control_dt': DT,
              'current_check': 'log-only', 'current_stop_ma': None,
              'uses_model': False, 'uses_imu': False, 'eeprom_writes': False,
              'calibration_file_changed': False, 'physical_accuracy_measured': False,
              'goal_writes': 0, 'trajectory_goal_writes': 0, 'samples': [],
              'target_reached': False, 'exit_state': 'no_motor_writes'}
    claimed, failure, started = False, None, None
    feedback, last_observed, expected = None, None, targets.copy()
    successful_final = None
    path_out = Path(report_path) if report_path is not None else None

    def check_stop():
        if stop.is_set():
            raise ValueError('Stop requested')

    def read(expected_goals=None):
        nonlocal feedback, last_observed
        feedback = _read_checked(bus, cfg, configurations, clock, expected_goals)
        last_observed = clock()
        return feedback

    def send(goal):
        nonlocal expected
        check_stop()
        if last_observed is None or clock() - last_observed > .1:
            raise ValueError('Feedback too old to send a target')
        _bounded_positions(goal, cfg, configurations)
        bus.sync_positions(goal)
        expected = dict(goal)
        report['goal_writes'] += 1

    try:
        configurations = {sid: bus.read_configuration(sid) for sid in ids}
        report['configuration_before'] = configurations
        enabled = _configuration(configurations, cfg, expected_p)
        _bounded_positions(targets, cfg, configurations, saved_target=True)
        read()
        start = {sid: feedback[sid].position_ticks for sid in ids}
        if max(abs(start[sid] - targets[sid]) for sid in ids) * 360 / 4096 > 15:
            raise ValueError('Starting position is more than 15 degrees from the saved goal')
        if enabled and any(feedback[sid].goal_ticks != targets[sid] for sid in ids):
            raise ValueError('Already ON with different goals; support the robot, relax, then retry this saved report')
        trajectory, ramp_s = plan_follow([start[sid] for sid in ids], [targets[sid] for sid in ids])
        if not enabled and ramp_s + .5 > seconds:
            raise ValueError(f'Need at least {ramp_s + .5:.2f} seconds for the bounded trajectory and final observation')
        operation = 'observe_existing_goal' if enabled else 'move_to_saved_goal'
        report.update(planned_mode=operation, initial_torque_enabled=enabled,
                      start_ticks=start, trajectory_seconds=0. if enabled else ramp_s,
                      max_step_ticks=MAX_STEP_TICKS)
        check_stop()
        if not move:
            report['preview_passed'] = True
            return report
        # All rejection checks above are strictly read-only. Existing held
        # targets need no writes, but the requested observation owns cleanup.
        claimed = True
        bus.writes_allowed = True
        report['mode'] = operation
        if not enabled:
            send(start)
            bus.soft_gain_p = expected_p
            bus.soft_gain_write_allowed = True
            try:
                bus.set_pid_ram(ids, expected_p, 0, 0, verify=True)
            finally:
                bus.soft_gain_write_allowed = False
            for sid in ids:
                if bus.read(sid, 50, 3) != bytes((expected_p, 0, 0)):
                    raise ValueError(f'ID{sid}: RAM gain readback mismatch')
            read(start)
            if max(abs(feedback[sid].position_ticks - start[sid]) for sid in ids) > 2:
                raise ValueError('Supported pose moved during preload; torque was not enabled')
            check_stop()
            bus.set_torque(ids, True, verify=True)
        else:
            expected = targets.copy()
        started = clock()
        horizon = started + seconds
        next_at = started + (DT if not enabled else 0.)
        trajectory_index = 1
        stable_since = None
        while True:
            remaining = next_at - clock()
            if remaining > 0:
                sleep(remaining)
            check_stop()
            if clock() >= horizon:
                break
            cycle = clock()
            if last_observed is not None and cycle - last_observed > .15:
                raise ValueError('Servo observation gap exceeded 150 ms')
            read(expected)
            check_stop()
            if clock() >= horizon:
                break
            if not enabled and trajectory_index < len(trajectory):
                goal = dict(zip(ids, map(int, trajectory[trajectory_index])))
                if max(abs(goal[sid] - expected[sid]) for sid in ids) > MAX_STEP_TICKS:
                    raise ValueError('Saved-target trajectory step exceeds two encoder ticks')
                dispatch_started = clock()
                send(goal)
                report['trajectory_goal_writes'] += 1
                trajectory_index += 1
                next_at = dispatch_started + DT
            else:
                next_at = cycle + DT
            errors = [abs(feedback[sid].position_ticks - targets[sid]) * 360 / 4096 for sid in ids]
            if max(errors) <= 1 and expected == targets:
                stable_since = clock() if stable_since is None else stable_since
            else:
                stable_since = None
            report['samples'].append({
                'elapsed_s': clock() - started,
                'actual_ticks': {sid: feedback[sid].position_ticks for sid in ids},
                'goal_ticks_readback': {sid: feedback[sid].goal_ticks for sid in ids},
                'last_sent_goal_ticks': expected.copy(),
                'current_ma': {sid: feedback[sid].current_ma for sid in ids},
                'voltage_v': {sid: feedback[sid].voltage_v for sid in ids},
                'temperature_c': {sid: feedback[sid].temperature_c for sid in ids},
            })
            if clock() - cycle > .1:
                raise ValueError('Follow cycle exceeded 100 ms')
        if expected != targets or (not enabled and trajectory_index != len(trajectory)):
            raise ValueError('Time budget expired before the saved target was sent; no catch-up commands')
        check_stop()
        after = {sid: bus.read_configuration(sid) for sid in ids}
        validate_hold_configuration(after, cfg, expected_p=expected_p)
        for sid in ids:
            for key in ('phase', 'eeprom_gains', 'position_min', 'position_max'):
                if after[sid].get(key) != configurations[sid].get(key):
                    raise ValueError(f'ID{sid}: hardware configuration changed: {key}')
        read(targets)
        check_stop()
        captured = clock()
        reached_now = max(abs(feedback[sid].position_ticks - targets[sid]) * 360 / 4096 for sid in ids) <= 1
        stable_s = captured - stable_since if reached_now and stable_since is not None else 0.
        report.update(completed=True, configuration_after=after,
                      final_goal_ticks=targets, final_torque_readback={sid: after[sid]['torque'] for sid in ids},
                      target_reached=bool(reached_now and stable_s >= .5),
                      target_reached_criterion='All 14 reported joints within 1 degree of saved goal for >=0.5 seconds of observations',
                      final_observed_stable_s=stable_s, exit_state=HELD_EXIT)
        successful_final = (dict(feedback), captured)
        return report
    except BaseException as exc:
        failure = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
        if claimed:
            try:
                bus.set_torque(ids, False, verify=True)
                report['exit_state'] = 'torque_off_verified'
            except Exception as cleanup_error:
                report['exit_state'] = 'torque_off_unconfirmed'
                report['torque_off_error'] = str(cleanup_error)
        raise
    finally:
        report['duration_elapsed_s'] = clock() - started if started is not None else None
        snapshot = None
        try:
            if successful_final is not None:
                observed, captured = successful_final
                origin, fresh = 'normal_final_readback', True
            elif claimed:
                observed = bus.read_feedback_many(ids, sync=True)
                captured = clock()
                if (set(observed) != set(ids)
                        or any(not math.isfinite(f.received_at)
                               or not 0 <= captured - f.received_at <= .1 for f in observed.values())):
                    raise ValueError('Final cleanup readback incomplete or stale')
                origin, fresh = 'after_failure_cleanup_readback', True
            else:
                observed, captured = feedback, last_observed
                origin, fresh = 'preview_readback', False
            if observed is not None:
                snapshot = _joint_position_snapshot(observed, cfg, HOME, expected,
                            source=origin, captured_at=captured, started=started, fresh=fresh)
                snapshot.update(exit_state_at_capture=report['exit_state'],
                                age_at_end_s=clock() - captured)
        except Exception as snapshot_error:
            report['end_position_readback_error'] = str(snapshot_error)
        if snapshot is None and feedback is not None:
            try:
                snapshot = _joint_position_snapshot(feedback, cfg, HOME, expected,
                    source='last_observed_sample', captured_at=last_observed,
                    started=started, fresh=False)
                snapshot.update(exit_state_at_capture=report['exit_state'],
                                age_at_end_s=clock() - last_observed,
                                readback_error=report.get('end_position_readback_error'))
            except Exception as fallback_error:
                report['end_position_fallback_error'] = str(fallback_error)
        if snapshot is None:
            columns = ('actual_ticks', 'actual_rad', 'actual_deg', 'home_relative_deg',
                       'goal_ticks', 'goal_deg', 'tracking_error_deg', 'current_ma',
                       'voltage_v', 'temperature_c', 'status', 'received_at_monotonic_s')
            snapshot = {'available': False, 'source': 'unavailable', 'fresh_readback': False,
                        'captured_at_monotonic_s': None, 'elapsed_s': None,
                        'exit_state_at_capture': report['exit_state'], 'age_at_end_s': None,
                        'readback_error': report.get('end_position_readback_error', report.get('error')),
                        'reference': 'Current encoder calibration; no position readback available',
                        'joints': [{**dict.fromkeys(columns), 'id': joint['id'],
                                    'name': joint['name'], 'last_sent_goal_ticks': None}
                                   for joint in cfg['joints']]}
        report['end_joint_positions'] = snapshot
        if path_out is not None:
            try:
                path_out.parent.mkdir(parents=True, exist_ok=True)
                if report.get('end_joint_positions') is not None:
                    report['end_positions_csv'] = str(_write_end_positions(path_out, report['end_joint_positions']))
            except Exception as csv_error:
                report['end_positions_csv_error'] = str(csv_error)
            try:
                _write_report(path_out, report)
            except Exception as report_error:
                # A logging error must never hide the original servo failure.
                print(f'Could not write follow report: {report_error}', file=sys.stderr)
                if failure is None:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'robot.json')
    parser.add_argument('--supported-confirmed', action='store_true')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--ram-p', type=int, choices=(5, 8), default=8)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 10:
        parser.error('--seconds must be finite, greater than zero and at most 10')
    if args.move and not args.supported_confirmed:
        parser.error('--move requires --supported-confirmed: support the trunk and keep feet clear')
    report_path = args.report or args.config.resolve().parent / 'logs' / (
        'follow_saved_home_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    try:
        cfg = load_config(args.config)
        source = load_saved_target(args.source_report, cfg)
        source_model = source['raw_report'].get('model', {}).get('model')
        protected = [args.config.resolve(), args.source_report.resolve()]
        if source_model:
            protected.append(Path(source_model).resolve())
        for value in cfg.values():
            if isinstance(value, str) and value.lower().endswith('.onnx'):
                protected.append((args.config.resolve().parent / value).resolve())
        _check_output_paths(report_path, protected)
        stop = threading.Event()
        with stop_signals(stop), exclusive_hardware(cfg, args.config):
            serial_options = dict(cfg['serial'])
            serial_options['timeout'] = max(.05, serial_options['timeout'])
            with FollowSavedBus(**serial_options) as bus:
                result = run_follow(cfg, source, bus, move=args.move,
                                    supported_confirmed=args.supported_confirmed,
                                    seconds=args.seconds, expected_p=args.ram_p,
                                    report_path=report_path, stop_event=stop)
        print(json.dumps({key: result.get(key) for key in (
            'mode', 'planned_mode', 'completed', 'preview_passed', 'goal_writes',
            'trajectory_goal_writes', 'target_reached', 'final_observed_stable_s',
            'exit_state', 'end_positions_csv')}, ensure_ascii=False))
        print(f'Report: {report_path}')
        if args.move:
            print('Saved target remains held. target_reached refers only to this temporary saved target, not mechanical HOME accuracy.')
        else:
            print('Read-only preview: no goals, gains or torque were written.')
        return 0
    except Exception as exc:
        print(f'Saved-target follow stopped: {exc}', file=sys.stderr)
        print(f'Report if hardware preflight started: {report_path}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
