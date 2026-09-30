"""Explicit, bounded ankle motion test; no policy or EEPROM changes.

Run only after the operator has confirmed both feet are suspended.
Default invocation plans only. --move enables ID 5 then ID 14, separately.
"""
import argparse
import dataclasses
import json
import signal
import threading
import time
from pathlib import Path

from microduck_deploy.servo import ServoBus


class Abort(RuntimeError):
    pass


def targets(sid, start, lo, hi):
    if sid not in (5, 14) or not 0 <= lo < hi <= 4095:
        raise Abort('Unexpected ID or firmware limits')
    end = start + (114 if sid == 5 else -114)
    if not lo + 16 <= min(start, end) <= max(start, end) <= hi - 16:
        raise Abort('Test would approach an encoder/firmware limit')
    return start, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move', action='store_true')
    args = parser.parse_args()
    if not args.move:
        print(json.dumps({'plan_only': True, 'ids': [5, 14],
                          'offset_ticks': [114, -114], 'ramp_seconds': 2.0}))
        return
    stop = threading.Event()
    finished = threading.Event()
    shared = {'active': None, 'heartbeat': time.monotonic()}
    report = {'ids': [5, 14], 'eeprom_writes': False, 'events': [], 'samples': []}
    deadline = time.monotonic() + 20
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: stop.set())

    def emit(event):
        report['events'].append(event)
        print(json.dumps(event), flush=True)

    def check_stop():
        if stop.is_set() or time.monotonic() > deadline:
            raise Abort('Stopped or overall deadline exceeded')

    cfg = json.load(open('robot.json'))['serial']
    if cfg['register_profile'] != 'hls_2':
        raise Abort('Expected HLS_2 register profile')
    with ServoBus(**cfg) as bus:
        def relax(sid):
            for attempt in range(3):
                try:
                    bus.set_torque([sid], False, verify=True)
                    return
                except Exception:
                    if attempt == 2:
                        raise

        def watchdog():
            while not finished.wait(.05):
                sid = shared['active']
                if sid is not None and (stop.is_set() or
                        time.monotonic() - shared['heartbeat'] > .5):
                    stop.set()
                    try:
                        bus.set_torque([sid], False, verify=False)
                    except Exception:
                        pass

        guard = threading.Thread(target=watchdog, daemon=True)
        guard.start()
        try:
            configs = {sid: bus.read_configuration(sid) for sid in (5, 14)}
            report['configuration_before'] = configs
            for sid, c in configs.items():
                if c['torque'] != 0 or c['mode'] != 4 or c['angle_resolution'] != 1:
                    raise Abort(f'ID{sid}: expected torque off, position mode 4, resolution 1')
            for sid in (5, 14):
                check_stop()
                fb = bus.read_feedback(sid)
                start, end = targets(sid, fb.position_ticks,
                                     configs[sid]['position_min'], configs[sid]['position_max'])
                emit({'event': 'planned', 'id': sid, 'start': start, 'target': end})
                last_target = start

                def read_checked(stage):
                    check_stop()
                    f = bus.read_feedback(sid)
                    shared['heartbeat'] = time.monotonic()
                    report['samples'].append({'id': sid, 'stage': stage,
                        'command': last_target, 'feedback': dataclasses.asdict(f)})
                    if not 6.0 <= f.voltage_v <= 8.4 or f.temperature_c >= 55:
                        raise Abort(f'ID{sid}: voltage/temperature outside test limits')
                    if abs(f.current_ma) > 500:
                        raise Abort(f'ID{sid}: elevated feedback current')
                    if abs(f.position_ticks - last_target) > 45:
                        raise Abort(f'ID{sid}: tracking error exceeded about 4 degrees')
                    if not min(start, end) - 24 <= f.position_ticks <= max(start, end) + 24:
                        raise Abort(f'ID{sid}: movement outside allowed test envelope')
                    return f

                read_checked('before_enable')
                shared['heartbeat'] = time.monotonic()
                shared['active'] = sid  # Preload itself may activate some firmware.
                bus.sync_positions({sid: start})
                if int.from_bytes(bus.read(sid, 42, 2), 'little') != start:
                    raise Abort(f'ID{sid}: preload goal was not confirmed')
                check_stop()
                bus.set_torque([sid], True, verify=True)
                shared['heartbeat'] = time.monotonic()
                emit({'event': 'enabled', 'id': sid})
                for stage, a, b in [('outward', start, end), ('return', end, start)]:
                    for step in range(1, 41):
                        check_stop()
                        read_checked(stage)
                        check_stop()
                        last_target = round(a + (b - a) * step / 40)
                        bus.sync_positions({sid: last_target})
                        shared['heartbeat'] = time.monotonic()
                        stop.wait(.05)
                    for _ in range(15):
                        read_checked(stage + '_hold')
                        stop.wait(.05)
                    actual = read_checked(stage + '_end')
                    if abs(actual.position_ticks - b) > 20:
                        raise Abort(f'ID{sid}: endpoint was not reached')
                    emit({'event': stage + '_complete', 'id': sid,
                          'target': b, 'actual': actual.position_ticks})
                relax(sid)
                shared['active'] = None
                emit({'event': 'torque_off_verified', 'id': sid})
            report['result'] = 'completed'
        except BaseException as exc:
            report['result'] = 'aborted'
            report['reason'] = str(exc)
            stop.set()
            raise
        finally:
            try:
                if shared['active'] is not None:
                    relax(shared['active'])
                    emit({'event': 'torque_off_verified', 'id': shared['active']})
                    shared['active'] = None
            except Exception as exc:
                report['cleanup_error'] = str(exc)
                print('Unable to verify torque off; disconnect servo power.', flush=True)
                raise
            finally:
                finished.set()
                guard.join(timeout=.5)
                Path('/tmp/microduck_ankles_motion_20260930.json').write_text(
                    json.dumps(report, indent=2) + '\n')
    emit({'event': 'completed', 'ids': [5, 14], 'torque': 'off'})


if __name__ == '__main__':
    main()
