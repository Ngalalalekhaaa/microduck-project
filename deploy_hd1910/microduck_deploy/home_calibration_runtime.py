"""Bounded HOME command correction, separate from the 50 Hz walking runtime.

No torque enable, automatic homing, gain changes, or calibration-file writes.
The motion mode takes over an already held HOME; success leaves the last goal
held. A software failure stops new goals. Physical danger or an explicitly
selected release policy requests torque OFF, whose delivery is verified when
communication permits. A software watchdog is not a hardware safety guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from .config import calibration_arrays, require_calibration
from .home_calibration_hardware import read_hardware_sample, validate_hold_configuration
from .home_calibration_policy import (
    FORCE_RATIO_POLICY_FAMILY, HomeCommandAccumulator, HomeObservationHistory,
    build_home_frame34, build_home_frame48, build_home_frame49,
)
from .policy import HOME, JOINT_LIMITS, q_to_ticks, ticks_to_q
from .servo import ServoStatusError


class CalibrationStop(RuntimeError):
    def __init__(self, message, *, release_torque=False):
        super().__init__(message)
        self.release_torque = release_torque


@dataclass(frozen=True)
class RunLimits:
    max_cycle_s: float = .05
    max_sample_gap_s: float = .10
    max_start_error_deg: float = 5.
    max_start_tilt_deg: float = 10.
    max_tilt_deg: float = 25.
    max_start_gyro_rad_s: float = .4
    max_gyro_rad_s: float = 2.
    max_tracking_error_deg: float = 6.
    max_current_ma: float = 800.
    watchdog_s: float = .15


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, default=_json_default, ensure_ascii=False,
                                    indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _joint_position_snapshot(feedback, cfg, home, expected_ticks, *, source,
                             captured_at, started, fresh):
    """Reported encoder coordinates, not independently measured mechanical HOME."""
    ids = [joint['id'] for joint in cfg['joints']]
    if set(feedback) != set(ids):
        raise ValueError('End position readback must contain all 14 joints')
    zero, direction = calibration_arrays(cfg)
    actual = np.asarray([feedback[sid].position_ticks for sid in ids])
    goals = np.asarray([feedback[sid].goal_ticks for sid in ids])
    q = ticks_to_q(actual, zero, direction)
    goal_q = ticks_to_q(goals, zero, direction)
    if not np.isfinite([actual, goals, q, goal_q]).all():
        raise ValueError('Non-finite end position readback')
    rows = []
    for i, joint in enumerate(cfg['joints']):
        sid = joint['id']
        value = feedback[sid]
        rows.append({
            'id': sid, 'name': joint['name'],
            'actual_ticks': int(actual[i]), 'actual_rad': float(q[i]),
            'actual_deg': float(np.rad2deg(q[i])),
            'home_relative_deg': float(np.rad2deg(q[i] - home[i])),
            'goal_ticks': int(goals[i]), 'goal_deg': float(np.rad2deg(goal_q[i])),
            'tracking_error_deg': float(np.rad2deg(q[i] - goal_q[i])),
            'last_sent_goal_ticks': (int(expected_ticks[sid])
                                     if expected_ticks[sid] is not None else None),
            'current_ma': value.current_ma if math.isfinite(value.current_ma) else None,
            'voltage_v': value.voltage_v if math.isfinite(value.voltage_v) else None,
            'temperature_c': value.temperature_c if math.isfinite(value.temperature_c) else None,
            'status': value.status, 'received_at_monotonic_s': value.received_at,
        })
    return {
        'available': True, 'source': source, 'fresh_readback': fresh,
        'captured_at_monotonic_s': captured_at,
        'elapsed_s': captured_at - started if started is not None else None,
        'reference': 'Current encoder calibration; not verified mechanical zero offsets',
        'joints': rows,
    }


def _write_end_positions(path, snapshot):
    """Keep provenance in each CSV row so a fallback cannot look like a final read."""
    csv_path = path.with_suffix('.positions.csv')
    metadata = ['source', 'fresh_readback', 'captured_at_monotonic_s', 'elapsed_s',
                'exit_state_at_capture', 'age_at_end_s', 'readback_error']
    columns = metadata + ['id', 'name', 'actual_ticks', 'actual_rad', 'actual_deg',
               'home_relative_deg', 'goal_ticks', 'goal_deg', 'tracking_error_deg',
               'last_sent_goal_ticks', 'current_ma', 'voltage_v', 'temperature_c',
               'status', 'received_at_monotonic_s']
    temporary = csv_path.with_suffix(csv_path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        provenance = {key: snapshot.get(key) for key in metadata}
        for row in snapshot.get('joints', []):
            writer.writerow({**provenance, **row})
    temporary.replace(csv_path)
    return csv_path


def _tilt(gravity):
    return math.degrees(math.acos(float(np.clip(-gravity[2], -1., 1.))))


def validate_target(target, previous, cfg, configurations, contract):
    """Reject, never clip, mechanical/firmware limits after policy saturation."""
    q = np.asarray(target, dtype=float)
    old = np.asarray(previous, dtype=float)
    home = np.asarray(contract.home_angles_rad)
    if q.shape != (14,) or old.shape != (14,) or not np.isfinite([q, old]).all():
        raise CalibrationStop('Invalid target shape or non-finite command')
    if np.any(np.abs(q - home) > contract.offset_limit_rad + 1e-7):
        raise CalibrationStop('Accumulated correction exceeds model limit')
    if np.any(np.abs(q - old) > contract.delta_limit_rad + 1e-7):
        raise CalibrationStop('Per-step correction exceeds model limit')
    if np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1]):
        raise CalibrationStop('Correction exceeds model joint limits')
    zero, direction = calibration_arrays(cfg)
    ticks = q_to_ticks(q, zero, direction)
    rounded_q = ticks_to_q(ticks, zero, direction)
    if np.any(rounded_q < JOINT_LIMITS[:, 0]) or np.any(rounded_q > JOINT_LIMITS[:, 1]):
        raise CalibrationStop('Rounded target exceeds model joint limits')
    previous_ticks = q_to_ticks(old, zero, direction)
    # Quantization can add at most one count to an ideal per-step displacement.
    allowed_counts = math.ceil(contract.delta_limit_rad * 4096 / math.tau)
    if np.any(np.abs(ticks - previous_ticks) > allowed_counts):
        raise CalibrationStop('Rounded target jump exceeds command slew limit')
    for tick, joint in zip(ticks, cfg['joints']):
        c = configurations[joint['id']]
        if not c['position_min'] <= int(tick) <= c['position_max']:
            raise CalibrationStop(f"ID{joint['id']}: target exceeds firmware limits")
    return {j['id']: int(t) for j, t in zip(cfg['joints'], ticks)}


def _check_motion_sample(sample, expected_ticks, cfg, limits, *, initial=False,
                         requires_current=True, enforce_current_limit=True):
    tilt = _tilt(sample.gravity)
    tilt_limit = limits.max_start_tilt_deg if initial else limits.max_tilt_deg
    if tilt > tilt_limit:
        raise CalibrationStop(f'Trunk tilt {tilt:.2f} degrees exceeds {tilt_limit}',
                              release_torque=not initial)
    gyro_limit = limits.max_start_gyro_rad_s if initial else limits.max_gyro_rad_s
    if np.linalg.norm(sample.gyro) > gyro_limit:
        raise CalibrationStop('Trunk angular velocity exceeds calibration limit',
                              release_torque=not initial)
    if initial and np.max(np.abs(sample.q - HOME)) > math.radians(limits.max_start_error_deg):
        raise CalibrationStop('Reported pose is too far from HOME; no automatic homing')
    if np.any(sample.q < JOINT_LIMITS[:, 0]) or np.any(sample.q > JOINT_LIMITS[:, 1]):
        raise CalibrationStop('Reported joint position exceeds model limits')
    if requires_current and sample.current_a is None:
        raise CalibrationStop('Motion sample lacks verified mapped current')
    for index, joint in enumerate(cfg['joints']):
        sid = joint['id']
        feedback = sample.feedback[sid]
        # V1 uses its verified encoding (which may differ from the driver).
        # V2 excludes current from the actor. Its optional host threshold uses
        # the driver's absolute current; log-only still requires valid measured
        # telemetry and never replaces it with zero.
        current_ma = (sample.current_a[index] * 1000 if requires_current
                      else feedback.current_ma)
        if not math.isfinite(current_ma):
            raise CalibrationStop(f'ID{sid}: invalid motor-current safety feedback')
        if enforce_current_limit and abs(current_ma) > limits.max_current_ma:
            raise CalibrationStop(f'ID{sid}: motor current exceeds limit', release_torque=True)
        if abs(feedback.goal_ticks - expected_ticks[sid]) > 1:
            raise CalibrationStop(f'ID{sid}: goal readback differs from last sent target')
        if abs(feedback.position_ticks - expected_ticks[sid]) * 360 / 4096 > limits.max_tracking_error_deg:
            raise CalibrationStop(f'ID{sid}: excessive tracking error')


def run_session(cfg, policy, bus, imu, current_mapping, *, mode='monitor', seconds=10.,
                report_path, stop_event=None, limits=None, fault_action='hold',
                current_check='stop', expected_p=5,
                clock=time.monotonic, sleep=time.sleep):
    """Run at most one 10 s episode; injected interfaces support no-hardware tests.

    Monitor never integrates hypothetical actions into the known-command input:
    that input is reconstructed from actual goal readback. Run never resets an
    episode automatically, and never turns on an initially relaxed servo.
    """
    if mode not in ('monitor', 'run') or fault_action not in ('hold', 'release'):
        raise ValueError('Invalid mode or fault action')
    if current_check not in ('stop', 'log-only'):
        raise ValueError('Invalid current check; expected stop or log-only')
    if type(expected_p) is not int or expected_p not in (5, 8):
        raise ValueError('Expected RAM P must be 5 or 8')
    if not math.isfinite(seconds) or not 0 < seconds <= 10:
        raise ValueError('A HOME session must last >0 and <=10 seconds')
    require_calibration(cfg)
    moving = mode == 'run'
    contract = policy.contract
    requires_current = contract.requires_current
    enforce_current_limit = current_check == 'stop'
    limits = limits or RunLimits()
    for value in asdict(limits).values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Runtime limits must be positive and finite')
    if moving and requires_current:
        if current_mapping is None:
            raise ValueError('Motion requires a verified signed-current mapping')
        current_mapping.require_motion()
    stop = stop_event if stop_event is not None else threading.Event()
    history = HomeObservationHistory(contract)
    accumulator = HomeCommandAccumulator(contract)
    ids = [joint['id'] for joint in cfg['joints']]
    zero, direction = calibration_arrays(cfg)
    previous_target = np.asarray(contract.home_angles_rad, dtype=float)
    expected_ticks = dict(zip(ids, q_to_ticks(previous_target, zero, direction).tolist()))
    report = {
        'mode': mode, 'model': policy.inspect(), 'duration_requested_s': seconds,
        'control_dt': contract.control_dt, 'limits': asdict(limits),
        'current_mapping_verified': bool(current_mapping and current_mapping.motion_ready),
        'requires_current_input': requires_current,
        'current_safety_signal': 'verified_mapped_current_a' if requires_current else 'driver_current_ma_abs',
        'current_check': current_check,
        'current_stop_ma': limits.max_current_ma if enforce_current_limit else None,
        'expected_ram_gains': {'p': expected_p, 'd': 0, 'i': 0},
        'training_nominal_ram_p': 5,
        'gain_override_from_training': expected_p != 5,
        'fault_action': fault_action, 'goal_write_attempts': 0, 'goal_writes_sent': 0,
        'eeprom_writes': False, 'calibration_file_changed': False,
        'physical_accuracy_measured': False, 'completed': False,
        'exit_state': 'no_motor_writes', 'samples': 0,
        'time_budget_reached': False,
        'joint_mapping': [
            {key: joint[key] for key in ('name', 'id', 'zero_tick', 'direction')}
            for joint in cfg['joints']
        ],
    }
    lock = threading.Lock()
    watchdog_stop = threading.Event()
    watchdog_fault = threading.Event()
    last_heartbeat = [clock()]
    claimed = [False]
    release_attempted = [False]
    failure = None
    started = None
    last_observed = None
    last_observed_at = None
    successful_final = None

    def release():
        release_attempted[0] = True
        try:
            bus.set_torque(ids, False, verify=True)
            report['exit_state'] = 'torque_off_verified'
        except Exception as exc:
            report['exit_state'] = 'torque_off_unconfirmed'
            report['torque_off_error'] = str(exc)

    def watch():
        while not watchdog_stop.wait(.01):
            with lock:
                if claimed[0] and clock() - last_heartbeat[0] > limits.watchdog_s:
                    watchdog_fault.set()
                    if fault_action == 'release':
                        release()
                    return

    def sample():
        nonlocal last_observed, last_observed_at
        result = read_hardware_sample(bus, imu, cfg,
                                      current_mapping=current_mapping if requires_current else None,
                                      max_age_s=limits.max_cycle_s,
                                      max_skew_s=limits.max_cycle_s,
                                      imu_max_age_s=min(.10, cfg['control']['imu_max_age_s']))
        last_observed, last_observed_at = result, clock()
        return result

    def check_stop():
        if watchdog_fault.is_set():
            raise CalibrationStop('Software watchdog expired; no more target writes')
        if stop.is_set():
            raise CalibrationStop('Stop requested; no more target writes')

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples_path = path.with_suffix('.samples.jsonl')
    report['samples_path'] = str(samples_path)
    watcher = threading.Thread(target=watch, daemon=True, name='home-calibration-watchdog')
    try:
        configurations = {sid: bus.read_configuration(sid) for sid in ids}
        report['configuration_before'] = configurations
        if moving:
            validate_hold_configuration(configurations, cfg, expected_p=expected_p)
            validate_target(previous_target, previous_target, cfg, configurations, contract)
            initial = sample()
            _check_motion_sample(initial, expected_ticks, cfg, limits, initial=True,
                                 requires_current=requires_current,
                                 enforce_current_limit=enforce_current_limit)
        check_stop()
        started = clock()
        last_sample = started
        next_sample_at = started
        next_write_at = started
        horizon = started + seconds
        watcher.start()
        steps = math.ceil(seconds / contract.control_dt)
        with samples_path.open('w', buffering=1) as stream:
            for step in range(steps):
                scheduled = max(started + step * contract.control_dt, next_sample_at)
                wait_s = scheduled - clock()
                if wait_s > 0:
                    sleep(wait_s)
                check_stop()
                begun = clock()
                if begun >= horizon:
                    report['time_budget_reached'] = True
                    break
                if begun - scheduled > limits.max_cycle_s:
                    raise CalibrationStop('Scheduler missed a control period; no catch-up writes')
                if begun - last_sample > limits.max_sample_gap_s:
                    raise CalibrationStop('Sensor sampling gap exceeded limit')
                next_sample_at = begun + contract.control_dt
                observed = sample()
                timings = {'sensor_read_s': clock() - begun, 'inference_s': None,
                           'command_wait_s': None, 'command_write_s': None,
                           'sample_to_dispatch_s': None, 'dispatch_elapsed_s': None}
                last_sample = begun
                if moving:
                    _check_motion_sample(observed, expected_ticks, cfg, limits,
                                         requires_current=requires_current,
                                         enforce_current_limit=enforce_current_limit)
                    offset = accumulator.command_offset.copy()
                else:
                    goals = [observed.feedback[sid].goal_ticks for sid in ids]
                    offset = ticks_to_q(goals, zero, direction) - np.asarray(contract.home_angles_rad)
                raw = target = None
                reason = None
                if requires_current and observed.current_a is None:
                    reason = 'No current mapping; recording raw sensors only'
                elif np.max(np.abs(offset)) > contract.offset_limit_rad + 1e-7:
                    reason = 'Existing goal is outside HOME correction range; inference skipped'
                else:
                    if requires_current:
                        frame = build_home_frame49(observed.q, observed.current_a, observed.gravity,
                                                  observed.gyro, offset, contract=contract)
                    elif contract.policy_family == FORCE_RATIO_POLICY_FAMILY:
                        frame = build_home_frame48(observed.q, observed.gravity, observed.gyro,
                                                  offset, contract=contract)
                    else:
                        frame = build_home_frame34(observed.q, observed.gravity, observed.gyro,
                                                  offset, contract=contract)
                    inference_started = clock()
                    raw = policy.infer(history.push(frame))
                    timings['inference_s'] = clock() - inference_started
                    check_stop()
                    if clock() >= horizon:
                        report['time_budget_reached'] = True
                        break
                    if moving:
                        target = accumulator.advance(raw)
                    else:
                        target = np.asarray(contract.home_angles_rad) + np.clip(
                            offset + np.clip(raw, -1, 1) * contract.delta_limit_rad,
                            -contract.offset_limit_rad, contract.offset_limit_rad)
                if moving:
                    if target is None:
                        raise CalibrationStop('A complete model observation could not be built')
                    targets = validate_target(target, previous_target, cfg, configurations, contract)
                    # A late wake-up must not shorten the next command period.
                    # Wait outside the watchdog lock, then recheck freshness and
                    # the original sample-to-write deadline before sending.
                    wait_started = clock()
                    wait_s = next_write_at - wait_started
                    if wait_s > 0:
                        sleep(wait_s)
                    timings['command_wait_s'] = clock() - wait_started
                    # Check the IMU again after inference and use the original
                    # cycle deadline, so a slow model cannot write stale actions.
                    imu.latest(min(.10, cfg['control']['imu_max_age_s']))
                    with lock:
                        check_stop()
                        if clock() >= horizon:
                            report['time_budget_reached'] = True
                            break
                        if clock() - begun > limits.max_cycle_s:
                            raise CalibrationStop('Inference/read cycle exceeded 50 ms; target not sent')
                        claimed[0] = True  # A failed/partial first write is also ownership.
                        report['goal_write_attempts'] += 1
                        dispatch_started = clock()
                        bus.sync_positions(targets)
                        write_completed = clock()
                        report['goal_writes_sent'] += 1
                        report['exit_state'] = 'last_goal_sent_torque_unchanged'
                        expected_ticks = targets
                        previous_target = np.asarray(target).copy()
                        last_heartbeat[0] = write_completed
                        # Enforce 20 Hz between dispatch starts. Basing the next
                        # dispatch on write completion adds serial latency every
                        # cycle, eventually consuming the sample-to-write budget.
                        next_write_at = dispatch_started + contract.control_dt
                        # Let sampling follow the dispatch phase when cadence
                        # waiting or its IMU/lock overhead delays dispatch. An
                        # independent sample clock would accumulate that delay
                        # until an otherwise fast control cycle times out.
                        # Serial write duration is intentionally excluded.
                        next_sample_at = max(
                            next_sample_at,
                            begun + contract.control_dt + dispatch_started - wait_started)
                        timings['command_write_s'] = write_completed - dispatch_started
                        timings['sample_to_dispatch_s'] = dispatch_started - begun
                        timings['dispatch_elapsed_s'] = dispatch_started - started
                row = {'step': step, 'elapsed_s': clock() - started,
                       'q_reported_rad': observed.q, 'current_joint_a': observed.current_a,
                       'gravity': observed.gravity, 'gyro_rad_s': observed.gyro,
                       'known_command_offset_rad': offset, 'raw_action': raw,
                       'proposed_target_rad': target, 'inference_skipped': reason,
                       'diagnostics': observed.diagnostics, 'goals_sent': moving,
                       'cycle_s': clock() - begun, 'timings': timings}
                stream.write(json.dumps(row, default=_json_default, allow_nan=False) + '\n')
                report['samples'] += 1
                if clock() - begun > limits.max_cycle_s:
                    raise CalibrationStop('Control cycle exceeded 50 ms; stopping further commands')
            # Read back the final sent goal as well; broadcast write success is
            # not proof of reception. This also completes the final control step.
            due = max(horizon, next_sample_at, next_write_at)
            wait_s = due - clock()
            if wait_s > 0:
                sleep(wait_s)
            check_stop()
            if moving:
                final_started = clock()
                final = sample()
                _check_motion_sample(final, expected_ticks, cfg, limits,
                                     requires_current=requires_current,
                                     enforce_current_limit=enforce_current_limit)
                # A goal readback alone does not establish that torque is ON.
                # Bound the complete final sample + 14 torque reads separately
                # from a normal inference cycle, without extending the watchdog.
                final_torque = {}
                for sid in ids:
                    check_stop()
                    if clock() - final_started > limits.max_sample_gap_s:
                        raise CalibrationStop('Final readback exceeded its deadline')
                    torque = bus.read(sid, 40, 1)
                    if torque != b'\x01':
                        raise CalibrationStop(f'ID{sid}: final torque is not verified ON')
                    final_torque[sid] = 1
                with lock:
                    check_stop()
                    if clock() - final_started > limits.max_sample_gap_s:
                        raise CalibrationStop('Final readback exceeded its deadline')
                    report['final_q_reported_rad'] = final.q
                    report['final_diagnostics'] = final.diagnostics
                    # Only a successfully sent command is an applied correction;
                    # inference may have consumed the final wall-clock budget.
                    correction = previous_target - np.asarray(contract.home_angles_rad)
                    report['command_correction_rad'] = correction
                    report['command_correction_deg'] = np.rad2deg(correction)
                    report['final_goal_ticks'] = expected_ticks
                    report['final_torque_readback'] = final_torque
                    report['exit_state'] = ('last_goal_readback_verified_torque_kept_on'
                                            if claimed[0] else 'existing_goal_readback_verified_no_motor_writes')
                    report['candidate_only'] = 'Command compensation, not a verified mechanical zero offset'
                    report['completed'] = True
                    successful_final = final
                    watchdog_stop.set()
            else:
                check_stop()
                report['completed'] = True
    except BaseException as exc:
        failure = exc
        report['error'] = f'{type(exc).__name__}: {exc}'
        with lock:
            if claimed[0] and not release_attempted[0]:
                physical_status = (isinstance(exc, ServoStatusError)
                                   and bool(exc.status & ((1 << 2) | (1 << 3))))
                if fault_action == 'release' or getattr(exc, 'release_torque', False) or physical_status:
                    release()
                else:
                    report['exit_state'] = 'commands_stopped_last_goal_retained_unverified'
    finally:
        if started is not None:
            report['duration_elapsed_s'] = clock() - started
        watchdog_stop.set()
        if watcher.is_alive():
            watcher.join(timeout=.5)
        report['watchdog_triggered'] = watchdog_fault.is_set()
        report['command_correction_last_sent_rad'] = previous_target - np.asarray(contract.home_angles_rad)
        if moving:
            # Fault cleanup (including any torque-OFF) finishes before diagnostic
            # reads. These reads do not depend on a working IMU and never write
            # targets, gains, torque, or calibration. Preserve the original error.
            snapshot_started = clock()
            last_sent_ticks = (expected_ticks if report['goal_writes_sent'] else
                               {sid: None for sid in ids})
            snapshot = {'available': False, 'source': 'unavailable',
                        'fresh_readback': False, 'joints': []}
            if last_observed is not None:
                try:
                    prior = _joint_position_snapshot(
                        last_observed.feedback, cfg, contract.home_angles_rad, last_sent_ticks,
                        source='last_observed_sample', captured_at=last_observed_at,
                        started=started, fresh=False)
                    if failure is not None:
                        report['last_observed_joint_positions'] = prior
                    snapshot = prior
                except Exception as exc:
                    report['last_observed_position_error'] = repr(exc)
            try:
                if successful_final is not None:
                    values = successful_final.feedback
                    captured = last_observed_at
                    source = 'normal_final_readback'
                elif watcher.is_alive():
                    raise RuntimeError('Watchdog has not exited; skip concurrent serial readback')
                else:
                    values = bus.read_feedback_many(ids, sync=True)
                    captured = clock()
                    if (set(values) != set(ids)
                            or any(not math.isfinite(v.received_at)
                                   or not 0 <= captured - v.received_at <= limits.max_sample_gap_s
                                   for v in values.values())):
                        raise ValueError('Incomplete or stale post-stop position readback')
                    source = 'post_stop_readback'
                snapshot = _joint_position_snapshot(
                    values, cfg, contract.home_angles_rad, last_sent_ticks,
                    source=source, captured_at=captured, started=started, fresh=True)
            except Exception as exc:
                snapshot['readback_error'] = repr(exc)
            snapshot['exit_state_at_capture'] = report['exit_state']
            snapshot['age_at_end_s'] = (max(0., clock() - snapshot['captured_at_monotonic_s'])
                                        if snapshot.get('available') else None)
            if not snapshot['available']:
                # Failed communication must not make a missing joint look like
                # a zero position or silently remove it from the 14-joint report.
                snapshot['joints'] = [
                    {'id': joint['id'], 'name': joint['name'],
                     'actual_ticks': None, 'actual_rad': None, 'actual_deg': None,
                     'home_relative_deg': None, 'goal_ticks': None, 'goal_deg': None,
                     'tracking_error_deg': None,
                     'last_sent_goal_ticks': last_sent_ticks[joint['id']],
                     'current_ma': None, 'voltage_v': None, 'temperature_c': None,
                     'status': None, 'received_at_monotonic_s': None}
                    for joint in cfg['joints']
                ]
            report['end_joint_positions'] = snapshot
            report['end_position_read_duration_s'] = clock() - snapshot_started
            try:
                report['end_positions_csv'] = str(_write_end_positions(path, snapshot))
            except Exception as exc:
                report['end_positions_csv_error'] = repr(exc)
        try:
            _write_report(path, report)
        except Exception as exc:
            if failure is None:
                raise
            report['report_write_error'] = repr(exc)
        if moving:
            try:
                print(json.dumps({
                    'event': 'end_joint_positions', 'report': str(path),
                    'csv': report.get('end_positions_csv'),
                    'report_write_error': report.get('report_write_error'),
                    **report['end_joint_positions'],
                }, default=_json_default, ensure_ascii=False, allow_nan=False), flush=True)
            except Exception:
                pass  # A closed terminal must not mask the saved run/fault report.
    if failure is not None:
        raise failure
    return report
