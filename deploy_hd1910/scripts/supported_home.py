"""One supported, suspended transition to a saved straight reference or HOME.

The operator must support the trunk with the feet clear. No balance controller,
IMU, RL policy, EEPROM writes, or automatic calibration. Optional soft RAM gains
are restored after failure and retained for holding after success. Success leaves
the requested pose held; any failure attempts torque-OFF. Default: preview.
"""
import argparse
from contextlib import ExitStack
import dataclasses
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY))
import numpy as np
from microduck_deploy.config import load_config, calibration_arrays, require_calibration
from microduck_deploy.homing import plan_home, validate_reference_calibration
from microduck_deploy.policy import ticks_to_q, q_to_ticks, HOME, STRAIGHT_REFERENCE, JOINT_LIMITS
from microduck_deploy.servo import ServoBus, ServoTimeout, ServoProtocolError


def exclusive_hardware(cfg, config_path):
    # Only the dedicated calibration preparation mode acquires this lock.
    # Legacy home_then_policy already owns it while launching this script.
    from home_calibration import exclusive_hardware as acquire
    return acquire(cfg, config_path)


class PositionOnlyBus(ServoBus):
    soft_gain_write_allowed = False
    soft_gain_p = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.communication_retries = []

    def _retry_read(self, operation, label):
        # Repeat complete read transactions only. Never replay a position/torque
        # write and never reuse partial feedback from a failed transaction.
        for attempt in range(1, 4):
            try:
                return operation()
            except (ServoTimeout, ServoProtocolError) as exc:
                self.communication_retries.append({'operation':label,'attempt':attempt,'error':str(exc)})
                if attempt == 3:
                    raise type(exc)(f'{label} failed after 3 attempts: {exc}') from exc
                print(json.dumps({'event':'read_retry','operation':label,'attempt':attempt,'error':str(exc)}),flush=True)
                time.sleep(.005)

    def read(self, servo_id, address, count):
        return self._retry_read(lambda: ServoBus.read(self,servo_id,address,count),
                                f'READ ID{servo_id} address{address}')

    def read_feedback_many(self, ids, *, sync=False):
        ids = list(ids)
        if not sync:
            return ServoBus.read_feedback_many(self,ids,sync=False)
        return self._retry_read(lambda: ServoBus.read_feedback_many(self,ids,sync=True),
                                'SYNC_READ ' + ','.join(map(str,ids)))

    def _sync_ram(self, address, size, values):
        soft = (self.soft_gain_write_allowed and (address,size)==(50,3)
                and self.soft_gain_p in (5, 8)
                and all(v == bytes((self.soft_gain_p,0,0)) for v in values.values()))
        if (address, size) not in ((40, 1), (42, 2)) and not soft:
            raise ValueError('Only position and torque RAM writes are allowed')
        return super()._sync_ram(address, size, values)


def path_ticks(start, end, steps=600):
    start, end = np.asarray(start), np.asarray(end)
    for k in range(1, steps + 1):
        t = k / steps
        s = t * t * (3 - 2 * t)
        yield np.rint(start + s * (end - start)).astype(int)


def plan_straight_recovery(cfg, start, *, target='straight'):
    """Supported recovery to an already verified pose, not an RL startup.

    HOME recovery accepts single-turn positions outside training limits and
    lengthens the trajectory for large displacements. Straight recovery retains
    its original geometric checks. Hardware limits are checked in main().
    """
    validate_reference_calibration(cfg, 'straight')
    zero, direction = calibration_arrays(cfg)
    start = np.asarray(start, dtype=float)
    if (start.shape != (14,) or not np.all(np.isfinite(start))
            or np.any(start != np.rint(start)) or np.any(start < 0) or np.any(start > 4095)):
        raise ValueError('Recovery start must contain 14 single-turn integer readings')
    start = start.astype(int)
    end_raw = np.asarray(cfg['calibration']['captured_ticks'])
    if end_raw.shape != (14,) or not np.all(end_raw == np.rint(end_raw)):
        raise ValueError('Saved reference must contain 14 exact encoder readings')
    end = end_raw.astype(int) if target == 'straight' else q_to_ticks(HOME, zero, direction)
    if start.shape != (14,) or np.any(start < 0) or np.any(start > 4095):
        raise ValueError('Recovery start must be within single-turn range')
    distance_deg = float(np.max(np.abs(end-start))*360/4096)
    if target == 'straight' and np.max(np.abs(end-start)) > 910:
        raise ValueError('Recovery displacement exceeds 80 degrees')
    q = ticks_to_q(start, zero, direction)
    if np.any(STRAIGHT_REFERENCE < JOINT_LIMITS[:,0]) or np.any(STRAIGHT_REFERENCE > JOINT_LIMITS[:,1]):
        raise ValueError('Saved recovery destination outside model limits')
    if target == 'straight':
        path = np.stack([start, *path_ticks(start, end)])
        angles = np.stack([ticks_to_q(row, zero, direction) for row in path])
        violation = np.maximum(JOINT_LIMITS[:,0]-angles, 0) + np.maximum(angles-JOINT_LIMITS[:,1], 0)
        if np.any(np.diff(violation, axis=0) > 1e-10):
            raise ValueError('Recovery path would increase model-limit violation')
    # HOME uses a 6 s minimum and 16 deg/s peak; larger displacements extend
    # the move. Keep the original timing for straight-reference recovery.
    minimum_seconds, peak_speed = (6.0, 16.0) if target == 'home' else (12.0, 8.0)
    steps = max(math.ceil(minimum_seconds * 50), math.ceil(1.5 * distance_deg / peak_speed * 50))
    duration = steps / 50
    return {'target': target, 'home_ticks': end.tolist(),
            'start_ticks': start.tolist(), 'start_q_rad': q.tolist(),
            'duration_s': duration, 'steps': steps,
            'peak_target_speed_deg_s': distance_deg*1.5/duration}


def plan_assembly_recovery(cfg, start):
    """Daily supported HOME recovery using verified assembly geometry.

    Unlike the existing straight-calibration recovery, this keeps both model
    endpoints in range and retains the strict feedback checks in main(). The
    initial assembly-reference acceptance remains plan_home(from_pose=assembly).
    """
    validate_reference_calibration(cfg, 'assembly')
    start = np.asarray(start, dtype=float)
    if (start.shape != (14,) or not np.all(np.isfinite(start))
            or np.any(start != np.rint(start)) or np.any(start < 0) or np.any(start > 4095)):
        raise ValueError('Recovery start must contain 14 single-turn integer readings')
    zero, direction = calibration_arrays(cfg)
    q = ticks_to_q(start, zero, direction)
    endpoints = np.stack([q, HOME])
    if np.any(endpoints < JOINT_LIMITS[:, 0]) or np.any(endpoints > JOINT_LIMITS[:, 1]):
        raise ValueError('装配标定回HOME路径端点超出模型关节限位')
    end = q_to_ticks(HOME, zero, direction)
    distance_deg = float(np.max(np.abs(end - start)) * 360 / 4096)
    steps = max(600, math.ceil(1.5 * distance_deg / 12.0 * 50))
    duration = steps / 50
    if duration > 30:
        raise ValueError('装配标定回HOME所需时间超过30秒；先托住并调整到接近HOME')
    return {'target': 'home', 'calibration_pose': 'assembly',
            'recovery': 'supported_daily', 'home_ticks': end.tolist(),
            'start_ticks': start.astype(int).tolist(), 'start_q_rad': q.tolist(),
            'duration_s': duration, 'steps': steps,
            'peak_target_speed_deg_s': distance_deg * 1.5 / duration}


def plan_supported_move(cfg, start, *, target='home', log_only_feedback=False,
                        daily_recovery=False):
    """Select reference acceptance or daily recovery without mixing references."""
    pose = cfg.get('calibration', {}).get('pose')
    if pose == 'home':
        if target != 'home':
            raise ValueError('HOME reference has no measured straight pose')
        require_calibration(cfg, imu=False)
        start = np.asarray(start, dtype=float)
        if (start.shape != (14,) or not np.isfinite(start).all()
                or np.any(start != np.rint(start)) or np.any(start < 0) or np.any(start > 4095)):
            raise ValueError('HOME recovery requires 14 single-turn integer readings')
        zero, direction = calibration_arrays(cfg)
        saved = cfg['calibration']
        recorded = np.asarray(saved.get('captured_ticks'), dtype=float)
        reference = np.asarray(saved.get('reference_q_rad'), dtype=float)
        if (saved.get('joint_ids') != [j['id'] for j in cfg['joints']]
                or saved.get('directions') != direction.tolist()
                or recorded.shape != (14,) or reference.shape != (14,)
                or not np.isfinite(recorded).all() or np.any(recorded != np.rint(recorded))
                or np.any(recorded < 0) or np.any(recorded > 4095)
                or not np.allclose(reference, HOME, rtol=0, atol=1e-10)
                or not np.allclose(ticks_to_q(recorded, zero, direction), HOME, rtol=0, atol=1e-10)):
            raise ValueError('Saved HOME reference is inconsistent with joint calibration')
        q = ticks_to_q(start, zero, direction)
        # This is supported recovery, so the walking startup distance/model
        # checks in plan_home do not apply. Interpolate within the single turn;
        # main checks both endpoints against each servo's hardware limits.
        end = recorded.astype(int)
        distance_deg = float(np.max(np.abs(end - start)) * 360 / 4096)
        steps = max(300, math.ceil(1.5 * distance_deg / 12.0 * 50))
        duration = steps / 50
        return {'target': 'home', 'calibration_pose': 'home',
                'recovery': 'supported_daily', 'home_ticks': end.tolist(),
                'start_ticks': start.astype(int).tolist(), 'start_q_rad': q.tolist(),
                'duration_s': duration, 'steps': steps,
                'peak_target_speed_deg_s': distance_deg * 1.5 / duration}
    if pose == 'assembly':
        if target != 'home':
            raise ValueError('装配标定没有直腿实测参考；不支持 --target straight')
        if log_only_feedback or daily_recovery:
            return plan_assembly_recovery(cfg, start)
        q = ticks_to_q(start, *calibration_arrays(cfg))
        distance_deg = float(np.rad2deg(np.max(np.abs(HOME - q))))
        # Use the measured start, including its allowed reference tolerance.
        # Smoothstep peak speed is 1.5 * displacement / duration; round the
        # minimum duration up to the actual 50 Hz trajectory grid.
        steps = max(600, math.ceil(1.5 * distance_deg / 12.0 * 50))
        return plan_home(cfg, q, from_pose='assembly', move_seconds=steps / 50)
    if target == 'straight' or log_only_feedback or daily_recovery:
        return plan_straight_recovery(cfg, start, target=target)
    return plan_home(cfg, ticks_to_q(start, *calibration_arrays(cfg)),
                     from_pose='straight', move_seconds=12)


def verify_feedback(feedback, ids, last, start, end, now, *, log_only=False):
    for idx, sid in enumerate(ids):
        f = feedback[sid]
        if now - f.received_at > .1 or f.status:
            raise ValueError(f'ID{sid}: stale/error feedback')
        if log_only:
            continue
        if not 5.0 <= f.voltage_v <= 8.4 or f.temperature_c >= 55 or abs(f.current_ma) > 800:
            raise ValueError(f'ID{sid}: voltage/temperature/current outside supported-test bounds: {f.voltage_v}V, {f.temperature_c}C, {f.current_ma}mA; actual={f.position_ticks}, commanded={int(last[idx])}')
        if abs(f.position_ticks - int(last[idx])) > 57:
            raise ValueError(f'ID{sid}: tracking error >5 degrees')
        if not min(start[idx], end[idx]) - 57 <= f.position_ticks <= max(start[idx], end[idx]) + 57:
            raise ValueError(f'ID{sid}: outside planned path')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--config', type=Path, default=DEPLOY/'robot.json')
    parser.add_argument('--target', choices=('home', 'straight'), default='home')
    parser.add_argument('--calibration-hold', action='store_true',
                        help='Supported daily HOME recovery with the requested RAM P,D=0,I=0; strict feedback by default, or existing straight-calibration behavior with --log-only-feedback')
    parser.add_argument('--ram-p', type=int, choices=(5, 8), default=None,
                        help='Only with --calibration-hold: requested volatile P gain, default 5; D/I remain zero')
    parser.add_argument('--soft-gains', action='store_true', help='Use verified volatile P=5,D=0,I=0; requires all joints relaxed')
    parser.add_argument('--log-only-feedback', action='store_true', help='Daily supported HOME recovery; straight calibration logs added thresholds only, assembly calibration retains all feedback thresholds')
    parser.add_argument('--jaw-command-fd', type=int, default=None, help=argparse.SUPPRESS)
    a = parser.parse_args(argv)
    if a.ram_p is not None and not a.calibration_hold:
        parser.error('--ram-p requires --calibration-hold')
    a.ram_p = 5 if a.ram_p is None else a.ram_p
    if a.calibration_hold and (a.target != 'home' or a.jaw_command_fd is not None):
        parser.error('--calibration-hold requires --target home and no jaw command input')
    cfg = load_config(a.config)
    require_calibration(cfg, imu=False)
    # Xbox requests daily recovery through this existing flag. Assembly-based
    # robots retain the voltage/current/tracking checks even for that request.
    log_only_feedback = a.log_only_feedback and cfg['calibration'].get('pose') != 'assembly'
    ids = [j['id'] for j in cfg['joints']]
    from microduck_deploy.jaw import JawCommandReader, JawController, jaw_ids
    jaw_source = JawCommandReader(a.jaw_command_fd) if a.jaw_command_fd is not None else None
    motion_ids = ids + (jaw_ids(cfg) if jaw_source is not None else [])
    # require_calibration already validates all 14 unique body IDs. Preserve
    # model joint order; a robot may use mouth ID 10 and right-leg IDs 11..15.
    zero, direction = calibration_arrays(cfg)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())
    report = {'task': f'supported move to {a.target}; not free standing', 'samples': [],
              'gains_changed': False, 'eeprom_writes': False, 'uses_imu': False,
              'software_current_stop_ma': None if log_only_feedback else 800,
              'calibration_hold': a.calibration_hold,
              'requested_ram_gains': {'p': a.ram_p, 'd': 0, 'i': 0} if a.calibration_hold else None,
              'config': str(a.config.resolve()),
              'log_only_feedback': log_only_feedback,
              'log_only_feedback_requested': a.log_only_feedback}
    log = a.config.resolve().parent/'logs'/('supported_home_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    log.parent.mkdir(exist_ok=True)
    serial_options = dict(cfg['serial'])
    serial_options['timeout'] = max(.05, serial_options['timeout'])
    report['serial_timeout_s'] = serial_options['timeout']
    with ExitStack() as stack:
        if a.calibration_hold:
            stack.enter_context(exclusive_hardware(cfg, a.config))
        bus = stack.enter_context(PositionOnlyBus(**serial_options))
        jaw = JawController(cfg, bus, command_source=jaw_source) if len(motion_ids) > len(ids) else None

        def send_targets(body_targets, feedback, *, cycle_started=None):
            targets = dict(body_targets)
            if jaw is not None:
                bus.soft_gain_write_allowed = True
                try:
                    targets.update(jaw.targets(feedback=feedback))
                finally:
                    bus.soft_gain_write_allowed = False
            if cycle_started is not None:
                if stop.is_set():
                    raise ValueError('Stopped before target write')
                if time.monotonic()-cycle_started > .2:
                    raise ValueError('Control cycle exceeded 200ms')
            if targets:
                bus.sync_positions(targets)
        configs = {sid: bus.read_configuration(sid) for sid in ids}
        for sid, c in configs.items():
            if a.calibration_hold:
                if (c.get('mode') != 4 or c.get('angle_resolution') != 1
                        or type(c.get('torque')) is not int or c['torque'] not in (0,1)):
                    raise ValueError(f'ID{sid}: calibration HOME requires mode 4, resolution 1, and known torque state')
                lower, upper = c.get('position_min'), c.get('position_max')
                if (type(lower) is not int or type(upper) is not int
                        or not 0 <= lower < upper <= 4095):
                    raise ValueError(f'ID{sid}: calibration HOME requires valid single-turn hardware limits')
            else:
                assert c['mode'] == 4 and c['angle_resolution'] == 1 and c['torque'] in (0,1), f'ID{sid}: mode/resolution/torque'
                assert 0 <= c['position_min'] < c['position_max'] <= 4095
            if a.soft_gains and not a.calibration_hold:
                assert c['torque']==0 and c['ram_gains']['i']==0, 'Soft start requires torque OFF and I=0'
        change_gains = a.soft_gains
        already_enabled = False
        if a.calibration_hold:
            # All preflight rejection paths precede the motion/failure-cleanup
            # scope: a held robot is never disabled to change its gains.
            states = {c['torque'] for c in configs.values()}
            if states not in ({0}, {1}):
                raise ValueError('Calibration HOME preparation requires all body servos OFF or all ON; mixed torque states are refused')
            if any(c.get('register_profile') != 'hls_2' for c in configs.values()):
                raise ValueError('Calibration HOME preparation requires the HLS_2 register profile')
            already_enabled = states == {1}
            if already_enabled:
                from microduck_deploy.home_calibration_hardware import validate_hold_configuration
                validate_hold_configuration(configs, cfg, expected_p=a.ram_p)
                change_gains = False
            else:
                if any(c['ram_gains']['i'] != 0 for c in configs.values()):
                    raise ValueError('Calibration HOME soft start requires existing RAM I=0')
                change_gains = True
            report['initial_torque_enabled'] = already_enabled
        feedback = bus.read_feedback_many(motion_ids, sync=True)
        start = np.array([feedback[sid].position_ticks for sid in ids],dtype=int)
        plan = plan_supported_move(cfg, start, target=a.target,
                                   log_only_feedback=a.log_only_feedback,
                                   daily_recovery=a.calibration_hold)
        end = np.array(plan['home_ticks'],dtype=int)
        steps = int(plan.get('steps', round(plan['duration_s']*50)))
        duration = steps / 50
        for i,sid in enumerate(ids):
            in_limits = configs[sid]['position_min'] <= min(start[i],end[i]) <= max(start[i],end[i]) <= configs[sid]['position_max']
            if a.calibration_hold and not in_limits:
                raise ValueError(f'ID{sid}: current position to HOME path exceeds hardware limits')
            assert in_limits
        # Planning may take >100ms on the SBC: validate a new sample, not its old input.
        feedback = bus.read_feedback_many(motion_ids, sync=True)
        if max(abs(feedback[sid].position_ticks-start[i]) for i,sid in enumerate(ids)) > 8:
            raise ValueError('Reference moved during planning')
        verify_feedback(feedback, ids, start, start, end, time.monotonic(), log_only=log_only_feedback)
        report.update(plan=plan, configuration_before=configs, start_feedback={sid:dataclasses.asdict(f) for sid,f in feedback.items()})
        log.write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({'event':'plan','target':a.target,'move':a.move,'start':start.tolist(),'end':end.tolist(),'seconds':duration,'peak_deg_s':plan['peak_target_speed_deg_s'],'log':str(log)}),flush=True)
        if not a.move:
            return
        success = False
        started = time.monotonic()
        last = start.copy()
        try:
            if stop.is_set():
                raise ValueError('Stopped before motion')
            if jaw is not None:
                jaw.preflight()
                bus.soft_gain_write_allowed = True
                try:
                    jaw.warmup()
                finally:
                    bus.soft_gain_write_allowed = False
            fresh = bus.read_feedback_many(motion_ids,sync=True)
            if max(abs(fresh[sid].position_ticks-start[i]) for i,sid in enumerate(ids)) > 8:
                raise ValueError('Reference moved during planning')
            if a.calibration_hold:
                verify_feedback(fresh, ids, start, start, end, time.monotonic(), log_only=log_only_feedback)
            if change_gains:
                # Save the original settings before the first write.
                report['gains_changed'] = True
                log.write_text(json.dumps(report,ensure_ascii=False,indent=2))
                bus.soft_gain_p = a.ram_p
                bus.soft_gain_write_allowed = True
                bus.set_pid_ram(ids,a.ram_p,0,0,verify=True)
                bus.soft_gain_write_allowed = False
                fresh=bus.read_feedback_many(motion_ids,sync=True)
                if max(abs(fresh[sid].position_ticks-start[i]) for i,sid in enumerate(ids)) > 8:
                    raise ValueError('Reference moved while preparing soft gains')
                if a.calibration_hold:
                    verify_feedback(fresh, ids, start, start, end, time.monotonic(), log_only=log_only_feedback)
            send_targets(dict(zip(ids,map(int,start))), fresh)
            if a.calibration_hold:
                for i, sid in enumerate(ids):
                    if bus.read_motion_profile(sid)['goal_position_ticks'] != int(start[i]):
                        raise ValueError(f'ID{sid}: preloaded current-position goal readback differs; torque enable refused')
                fresh = bus.read_feedback_many(motion_ids,sync=True)
                verify_feedback(fresh, ids, start, start, end, time.monotonic(), log_only=log_only_feedback)
                if max(abs(fresh[sid].position_ticks-start[i]) for i,sid in enumerate(ids)) > 8:
                    raise ValueError('Reference moved while verifying preloaded goals')
                if stop.is_set():
                    raise ValueError('Stopped before torque enable')
            if not (a.calibration_hold and already_enabled):
                bus.set_torque(ids,True,verify=True)
            next_target_at = time.monotonic() + .02
            for k,goal in enumerate(path_ticks(start,end,steps=steps),1):
                cycle = time.monotonic()
                if stop.is_set() or cycle-started > max(40, duration*2+10):
                    raise ValueError('Stopped or overall deadline exceeded')
                feedback = bus.read_feedback_many(motion_ids,sync=True)
                verify_feedback(feedback,ids,last,start,end,time.monotonic(), log_only=log_only_feedback)
                if np.max(np.abs(goal-last)) > 4:
                    raise ValueError('Target step exceeded limit')
                # Read feedback during the interval before the next command;
                # do not add an unconditional 20 ms after the serial read.
                stop.wait(max(0, next_target_at-time.monotonic()))
                if stop.is_set():
                    raise ValueError('Stopped before target write')
                if time.monotonic()-cycle > .2:
                    raise ValueError('Control cycle exceeded 200ms')
                send_targets(dict(zip(ids,map(int,goal))), feedback)
                next_target_at = time.monotonic() + .02
                last=goal
                if k%10==0:
                    report['samples'].append({'step':k,'elapsed':time.monotonic()-started,
                        'command':goal.tolist(),'actual':[feedback[sid].position_ticks for sid in ids],
                        'current_ma':[feedback[sid].current_ma for sid in ids],
                        'voltage_v':[feedback[sid].voltage_v for sid in ids],
                        'temperature_c':[feedback[sid].temperature_c for sid in ids]})
                if k%100==0:
                    print(json.dumps({'event':'progress','step':k,'of':steps}),flush=True)
                if time.monotonic()-cycle > .2:
                    raise ValueError('Control cycle exceeded 200ms')
            # Allow 1 s to settle at the last target, maintaining feedback checks.
            for _ in range(50):
                cycle = time.monotonic()
                if stop.is_set():raise ValueError('Stopped during settle')
                feedback=bus.read_feedback_many(motion_ids,sync=True)
                verify_feedback(feedback,ids,end,start,end,time.monotonic(), log_only=log_only_feedback)
                if jaw is not None:
                    send_targets({}, feedback, cycle_started=cycle)
                stop.wait(max(0, .02-(time.monotonic()-cycle)))
            errors=[feedback[sid].position_ticks-int(end[i]) for i,sid in enumerate(ids)]
            reached = max(map(abs,errors)) <= 23
            if not reached and not log_only_feedback:
                raise ValueError(f'Target error exceeds 2 degrees: {errors}')
            after={sid:bus.read_configuration(sid) for sid in ids}
            for i,(sid,c) in enumerate(after.items()):
                if a.calibration_hold and c['torque'] != 1:
                    raise ValueError(f'ID{sid}: final HOME torque is not ON')
                assert c['torque']==1
                if a.calibration_hold and c['motion_profile']['goal_position_ticks'] != int(end[i]):
                    raise ValueError(f'ID{sid}: final goal readback is not calibrated HOME')
                for key in ('mode','angle_resolution','phase','eeprom_gains','ram_gains','position_min','position_max'):
                    expected = {'p':a.ram_p,'d':0,'i':0} if key=='ram_gains' and change_gains else configs[sid][key]
                    if a.calibration_hold and c[key] != expected:
                        raise ValueError(f'ID{sid}: final configuration changed: {key}')
                    assert c[key]==expected,f'ID{sid}: configuration changed'
            if stop.is_set(): raise ValueError('Stopped during final verification')
            success=True
            report.update(result=f'{a.target} interpolation complete; torque remains ON',target_reached=reached,final_errors_ticks=errors,configuration_after=after)
            print(json.dumps({'event':'complete','target':a.target,'target_reached':reached,'target_ticks':end.tolist(),'actual_ticks':[feedback[sid].position_ticks for sid in ids],'errors_ticks':errors,'holding':True}),flush=True)
        except BaseException as exc:
            report.update(result='aborted',error=repr(exc),last_command_ticks=last.tolist(),
                          fault_feedback={sid:dataclasses.asdict(f) for sid,f in feedback.items()})
            raise
        finally:
            if not success:
                failures=[]
                for attempt in range(3):
                    try:
                        bus.set_torque(motion_ids,False,verify=True)
                        report['failure_cleanup']='torque OFF verified'
                        if report['gains_changed']:
                            # Restore only the captured RAM gains, with torque confirmed OFF.
                            for sid in ids:
                                original=configs[sid]['ram_gains']
                                values=bytes(original[k] for k in ('p','d','i'))
                                ServoBus._sync_ram(bus,50,3,{sid:values})
                                restored = bus.read(sid,50,3)
                                if a.calibration_hold and restored != values:
                                    raise ValueError(f'ID{sid}: restored RAM gain readback differs')
                                assert restored==values
                            report['failure_gain_cleanup']='original RAM gains restored'
                        break
                    except Exception as exc:
                        failures.append(repr(exc))
                else:
                    report['failure_cleanup_errors']=failures
                    print('TORQUE-OFF NOT VERIFIED: disconnect servo power',flush=True)
                if jaw is not None:
                    try:
                        jaw.disable(restore_gains=False)
                    except Exception as exc:
                        report['jaw_cleanup_error'] = repr(exc)
            if jaw_source is not None:
                jaw_source.close()
            report['communication_retries'] = bus.communication_retries
            log.write_text(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
