"""Measure real feedback/IMU/policy costs without any servo writes."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_sysfs_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError, UnicodeError):
        return None


def read_telemetry():
    """Best-effort observations, collected outside the measured work interval."""
    thermal = Path('/sys/class/thermal')
    cooling = []
    try:
        devices = sorted(thermal.glob('cooling_device*'))
    except OSError:
        devices = None
    if devices is not None:
        for device in devices:
            try:
                kind = (device/'type').read_text().strip()
            except (OSError, UnicodeError):
                continue
            if 'cpufreq' in kind.lower():
                cooling.append({'name': device.name, 'type': kind,
                                'cur_state': read_sysfs_int(device/'cur_state'),
                                'max_state': read_sysfs_int(device/'max_state')})
    return {
        'cpu_freq_khz': read_sysfs_int('/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'),
        'temperature_millicelsius': read_sysfs_int(thermal/'thermal_zone0/temp'),
        'cpufreq_cooling': cooling or None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--legacy-serial', action='store_true')
    parser.add_argument('--no-gamepad', action='store_true',
                        help='Use a constant zero velocity command without opening a gamepad')
    parser.add_argument('--telemetry', action='store_true',
                        help='Save per-cycle timings and CPU/thermal observations in the report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 120:
        parser.error('--seconds must be 1..120')
    from microduck_deploy.config import load_config, calibration_arrays
    from microduck_deploy.servo import ServoBus
    from microduck_deploy.policy import OnnxPolicy, HOME, ticks_to_q, build_obs61
    from microduck_deploy.gamepad import GamepadReader, GamepadCommand
    from microduck_deploy.imu_process import ProcessImuReader

    class ReadOnlyBus(ServoBus):
        def _sync_ram(self, *a, **kw):
            raise AssertionError('Benchmark forbids all servo RAM writes')
        def write(self, *a, **kw):
            raise AssertionError('Benchmark forbids servo writes')

    cfg = load_config(ROOT/'robot.json')
    ids = [j['id'] for j in cfg['joints']]
    zero, direction = calibration_arrays(cfg)
    policy = OnnxPolicy(ROOT/cfg['model'])
    pads = []
    imu = None
    rows, errors = [], []
    try:
        if args.no_gamepad:
            command = lambda: (0., 0., 0.)
        else:
            # Reproduce the parent button listener and policy velocity listener.
            for _ in range(2): pads.append(GamepadReader())
            command = GamepadCommand(pads[-1])
        imu = ProcessImuReader(cfg, calibrate=False)
        with ReadOnlyBus(**cfg['serial']) as bus:
            if args.legacy_serial:
                bus._read_fd = None
            started = time.monotonic()
            deadline = started
            previous_q = HOME.copy()
            previous_time = started-.02
            action = np.zeros(14)
            while time.monotonic()-started < args.seconds:
                t0 = time.monotonic()
                try:
                    feedback = bus.read_feedback_many(ids, sync=True)
                    t1 = time.monotonic()
                    state = imu.latest(cfg['control']['imu_max_age_s'])
                    q = ticks_to_q([feedback[i].position_ticks for i in ids], zero, direction)
                    obs = build_obs61(state.base_ang_vel, state.projected_gravity, q,
                                      (q-previous_q)/max(.002,t1-previous_time), action,
                                      twist=np.asarray(command()))
                    t2 = time.monotonic()
                    action = policy.infer(obs)
                    t3 = time.monotonic()
                    # Approximate the deployed sample serialization without sending goals.
                    json.dumps({'q':q.tolist(),'action':action.tolist(),
                                'feedback':{k:vars(v) for k,v in feedback.items()}})
                    t4 = time.monotonic()
                    row = {'t': t0-started,
                           'read_ms':(t1-t0)*1000,'prepare_ms':(t2-t1)*1000,
                           'inference_ms':(t3-t2)*1000,'serialize_ms':(t4-t3)*1000,
                           'total_ms':(t4-t0)*1000}
                    if args.telemetry:
                        row['telemetry'] = read_telemetry()
                    rows.append(row)
                    previous_q, previous_time = q, t1
                except Exception as exc:
                    errors.append({'t':time.monotonic()-started,'error':repr(exc)})
                deadline += .02
                if deadline < time.monotonic(): deadline = time.monotonic()
                time.sleep(max(0,deadline-time.monotonic()))
    finally:
        if imu is not None: imu.close()
        for pad in pads: pad.close()
    metrics = {}
    if rows:
        for name in ('read_ms', 'prepare_ms', 'inference_ms', 'serialize_ms', 'total_ms'):
            vals = [r[name] for r in rows]
            metrics[name] = dict(zip(('p50','p95','p99','max'), map(float,np.percentile(vals,[50,95,99,100]))))
    report = {'legacy_serial':args.legacy_serial,'duration_s':args.seconds,
              'no_gamepad':args.no_gamepad,'telemetry':args.telemetry,
              'imu_max_age_s':cfg['control']['imu_max_age_s'],
              'samples':len(rows),'errors':errors,'metrics':metrics,
              'cycles_over_20ms':sum(r['total_ms']>.02*1000 for r in rows),
              'servo_writes':0}
    if args.telemetry:
        report['rows'] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # Keep the console summary short; the output file retains every measured row.
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'},
                     ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
