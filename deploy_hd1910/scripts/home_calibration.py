"""Inspect, monitor, or run a separately contracted HOME correction policy.

Default inspect is offline. Monitor never writes servo registers. Run requires
an already held, independently standing HOME. The legacy current-based model
also requires a verified current mapping; the encoder/IMU-only model does not.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from microduck_deploy.config import load_config, require_calibration
from microduck_deploy.home_calibration_hardware import (
    CalibrationServoBus, CurrentMapping, ReadOnlyServoBus, current_mapping_template,
)
from microduck_deploy.home_calibration_policy import HomeCalibrationPolicy
from microduck_deploy.home_calibration_runtime import RunLimits, run_session, _write_report
from microduck_deploy.imu_process import ProcessImuReader


@contextmanager
def exclusive_hardware(cfg, config_path):
    # The lock belongs to the existing deployment/config directory, including
    # when this script is running from an isolated staged release.
    logdir = Path(config_path).resolve().parent / 'logs'
    logdir.mkdir(exist_ok=True)
    with (logdir / 'supported_sequence.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another robot operation holds supported_sequence.lock') from exc
        result = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
        if result.returncode != 1:
            raise RuntimeError('Serial device is already used or cannot be checked: '
                               + result.stdout + result.stderr)
        # IMU readers must not reinitialize a sensor used by another process.
        bus = cfg['imu'].get('i2c_bus')
        paths = ([Path(f'/dev/i2c-{bus}')] if bus is not None
                 else list(Path('/dev').glob('i2c-*')))
        for path in paths:
            result = subprocess.run(['fuser', str(path)], capture_output=True, text=True)
            if result.returncode != 1:
                raise RuntimeError(f'IMU bus {path} is already used or cannot be checked')
        yield


@contextmanager
def stop_signals(event):
    saved = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        saved[sig] = signal.signal(sig, lambda *_: event.set())
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inspect', 'monitor', 'run'), default='inspect')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--contract', type=Path, help='Optional matching export contract JSON')
    parser.add_argument('--config', type=Path, default=ROOT / 'robot.json')
    parser.add_argument('--current-map', type=Path)
    parser.add_argument('--current-check', choices=('stop', 'log-only'), default='stop',
                        help='stop enforces the software current threshold; log-only records current without that threshold stop')
    parser.add_argument('--ram-p', type=int, choices=(5, 8), default=5,
                        help='Require this existing RAM P gain, with D=0/I=0; never change gains during takeover')
    parser.add_argument('--write-current-template', type=Path,
                        help='Write a separate UNVERIFIED mapping template; never modify robot.json')
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--standing-confirmed', action='store_true',
                        help='Operator confirms feet on floor, self-supported and already holding HOME')
    parser.add_argument('--fault-action', choices=('hold', 'release'), default='hold',
                        help='Software fault response after takeover; physical danger always requests release')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 10:
        parser.error('--seconds must be finite, greater than zero and at most 10')
    if args.mode == 'run' and not args.standing_confirmed:
        parser.error('run requires --standing-confirmed after preparing an independently standing HOME')
    if args.write_current_template and args.mode != 'inspect':
        parser.error('--write-current-template is only supported with offline inspect')
    cfg = load_config(args.config)
    require_calibration(cfg)
    policy = HomeCalibrationPolicy(args.model, contract_path=args.contract)
    if args.write_current_template:
        # Refuse overwriting a mapping that may contain measured calibration.
        with args.write_current_template.open('x') as stream:
            json.dump(current_mapping_template(cfg), stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    mapping = (CurrentMapping.from_dict(json.loads(args.current_map.read_text()), cfg,
                                       motion=args.mode == 'run' and policy.contract.requires_current)
               if args.current_map else None)
    if args.mode == 'run' and policy.contract.requires_current and mapping is None:
        parser.error('run requires --current-map with independently verified signed-current mappings')
    report_path = args.report or (args.config.resolve().parent / 'logs' /
                                 ('home_calibration_' + args.mode + '_' +
                                  datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json'))
    protected = [path.resolve() for path in
                 (args.config, args.model, args.contract, args.current_map) if path is not None]
    positions_path = report_path.with_suffix('.positions.csv')
    for output in (report_path, report_path.with_suffix('.samples.jsonl'),
                   report_path.with_suffix(report_path.suffix + '.tmp'),
                   positions_path, positions_path.with_suffix(positions_path.suffix + '.tmp')):
        if any(output.resolve() == path or
               (output.exists() and path.exists() and output.samefile(path))
               for path in protected):
            parser.error('Report paths must not overwrite configuration, model, contract or current mapping')
    inspection = policy.inspect()
    inspection.update({
        'mode': args.mode, 'config': str(args.config.resolve()),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'current_mapping_verified': bool(mapping and mapping.motion_ready),
        'current_check': args.current_check,
        'current_stop_ma': RunLimits().max_current_ma if args.current_check == 'stop' else None,
        'expected_ram_gains': {'p': args.ram_p, 'd': 0, 'i': 0},
        'training_nominal_ram_p': 5,
        'gain_override_from_training': args.ram_p != 5,
        'joint_ids': [joint['id'] for joint in cfg['joints']],
        'serial_device': cfg['serial']['port'], 'hardware_opened': False,
    })
    if args.mode == 'inspect':
        _write_report(report_path, inspection)
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        print(f'Offline inspection passed. Report: {report_path}')
        return 0
    stop = threading.Event()
    # Policy loading and CPU warm-up finish before sensor acquisition starts.
    # IMU mounting and configured gyro bias (zero if absent) are reused,
    # never re-learned while the robot is loaded.
    bus_type = CalibrationServoBus if args.mode == 'run' else ReadOnlyServoBus
    imu = None
    try:
        with stop_signals(stop), exclusive_hardware(cfg, args.config):
            with bus_type(**cfg['serial']) as bus:
                try:
                    imu = ProcessImuReader(cfg, calibrate=False)
                    result = run_session(cfg, policy, bus, imu, mapping, mode=args.mode,
                                         seconds=args.seconds, report_path=report_path,
                                         stop_event=stop, fault_action=args.fault_action,
                                         current_check=args.current_check, expected_p=args.ram_p)
                finally:
                    if imu is not None:
                        imu.close()
        print(f"Completed {result['samples']} samples. Exit state: {result['exit_state']}")
        print(f'Report: {report_path}')
        if args.mode == 'run':
            print('Final command remains held. This is a temporary command correction; robot.json is unchanged.')
        return 0
    except Exception as exc:
        print(f'HOME calibration stopped: {exc}', file=sys.stderr)
        print(f'Report if sampling started: {report_path}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
