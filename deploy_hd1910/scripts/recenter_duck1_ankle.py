#!/usr/bin/env python3
"""One-time duck1 ID15 encoder-reference maintenance; default is read-only prepare.

Only --apply --backup DIR changes hardware. It sends no goal or torque-ON
packet. This does not establish a mechanical/model zero: recapture straight
calibration afterwards. Old saved poses are intentionally not migrated.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microduck_deploy.config import require_mapping
from microduck_deploy.policy import HOME, JOINT_LIMITS, JOINT_NAMES, policy_calibration_reference
from microduck_deploy.servo import ServoBus, ServoTimeout, build_packet, decode_sign_magnitude

SID, TARGET = 15, 2048
BODY_IDS = list(range(1, 10)) + list(range(11, 16))
ALL_IDS = list(range(1, 16))
OPERATION = 'duck1-right-ankle-15-current-reference-to-2048-v1'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_config(raw):
    cfg = json.loads(raw)
    require_mapping(cfg)
    need([j['name'] for j in cfg['joints']] == list(JOINT_NAMES), 'Unexpected joint order')
    need([j['id'] for j in cfg['joints']] == BODY_IDS, 'Not the duck1 body ID mapping')
    need(all(j['direction'] == (1 if j['id'] == SID else -1) for j in cfg['joints']),
         'Direction mapping differs from the checked duck1 configuration')
    need(cfg.get('calibration') is None and all(j.get('zero_tick') is None for j in cfg['joints']),
         'Existing calibration/zero: this tool does not migrate calibrated robots')
    need(cfg['serial'].get('register_profile') == 'hls_2', 'Requires hls_2 profile')
    return cfg


def reject_saved_poses(config_path):
    poses = config_path.parent.parent / 'servo_web_hd1910' / 'poses'
    for path in poses.glob('*.json'):
        need(json.loads(path.read_text()).get('fake') is True,
             f'Real saved pose must be reviewed/archived before maintenance: {path}')


class MaintenanceBus(ServoBus):
    def __init__(self, *args, emit=None, **kwargs):
        self.emit = emit or (lambda event: None)
        self.allow_changes = False
        self.emergency_off_attempted = False
        self.warnings = []
        super().__init__(*args, **kwargs)

    def _send(self, packet):
        need(len(packet) >= 6, 'Invalid packet')
        sid, instruction, params = packet[2], packet[4], bytes(packet[5:-1])
        need(packet == build_packet(sid, instruction, params), 'Invalid packet checksum/length')
        need(sid in ALL_IDS, 'No broadcast or unknown ID')
        if instruction != 0x02:
            off = sid == SID and instruction == 0x03 and params == b'\x28\x00'
            lock = sid == SID and instruction == 0x03 and params in (b'\x37\x00', b'\x37\x01')
            calibrate = sid == SID and instruction == 0x0b and params == struct.pack('<H', TARGET)
            need(self.allow_changes and (off or lock or calibrate), 'Forbidden hardware write')
            if not off:
                self.check_off()
            event = {'event': 'write_packet', 'id': sid, 'instruction': instruction,
                     'params_hex': params.hex(), 'packet_hex': packet.hex()}
            if off:
                try:
                    self.emit(event)
                except BaseException as exc:
                    # A full disk must never prevent the emergency OFF packet.
                    self.warn(f'Emergency OFF event could not be persisted: {exc}')
            else:
                self.emit(event)
        return super()._send(packet)

    def command(self, instruction, params):
        with self._io:
            deadline = self._send(build_packet(SID, instruction, params))
            try:
                self._read_status(deadline, SID, 0)
            except ServoTimeout:
                # Never replay a write; decide by explicit readback instead.
                return None
        return 0

    def warn(self, message):
        self.warnings.append(message)
        try:
            print(f'维护告警: {message}', file=sys.stderr, flush=True)
        except BaseException:
            pass

    def emergency_off(self):
        # Exactly one OFF attempt for this operation, including uncertain ACKs.
        # Verification still runs if transport or event persistence failed.
        if not self.emergency_off_attempted:
            self.emergency_off_attempted = True
            try:
                self.command(0x03, b'\x28\x00')
            except BaseException as exc:
                self.warn(f'Emergency OFF send/ACK failed: {exc}')
        need(self.read(SID, 40, 1) == b'\x00', 'Emergency torque-OFF could not be confirmed')

    def check_off(self):
        try:
            torque = self.read(SID, 40, 1)
        except BaseException as original:
            try:
                self.emergency_off()
            except BaseException as cleanup:
                self.warn(f'Torque read failed and emergency OFF verification failed: {cleanup}')
            # Even successful emergency cleanup cannot turn this into success.
            raise original
        if torque != b'\x00':
            self.emergency_off()
            raise RuntimeError('Unexpected ID15 torque ON; emergency OFF sent; stop')

    def cleanup(self):
        original = None
        try:
            self.check_off()
        except BaseException as exc:
            original = exc
        try:
            # An unknown first read attempts OFF; a fresh read must confirm
            # before locking. Lock writes still require a durable event.
            self.check_off()
            self.command(0x03, b'\x37\x01')
            need(self.read(SID, 55, 1) == b'\x01', 'Relock failed')
            self.check_off()
        except BaseException as exc:
            if original is not None:
                self.warn(f'Additional cleanup failure: {exc}')
                raise original from exc
            raise
        if original is not None:
            raise original


def positions(bus, count=12):
    result = []
    for _ in range(count):
        need(bus.read(SID, 40, 1) == b'\x00', 'ID15 must stay torque OFF')
        result.append(bus.read_feedback(SID).position_ticks)
        time.sleep(.025)
    need(all(0 <= p <= 4095 for p in result), 'ID15 feedback outside single turn')
    need(max(result)-min(result) <= 8, f'ID15 moved during capture: {result}')
    return result


def snapshot(bus):
    configurations = {str(sid): bus.read_configuration(sid) for sid in ALL_IDS}
    for sid, row in configurations.items():
        need(row['torque'] == 0, f'ID{sid} must be torque OFF')
        need(row['firmware'] == [3, 46] and row['model_version'] == [10, 31], f'ID{sid}: unexpected firmware/model')
        need(row['mode'] == 4 and row['angle_resolution'] == 1, f'ID{sid}: unexpected mode/resolution')
        need(row['position_min'] == 0 and row['position_max'] == 4095 and row['lock'] == 1,
             f'ID{sid}: unexpected limits/lock')
    return {'configurations': configurations,
            'eeprom_hex': {str(sid): bus.read(sid, 0, 40).hex() for sid in ALL_IDS},
            'id15_registers_0_55_hex': bus.read(SID, 0, 56).hex(),
            'body_positions': {str(sid): bus.read_feedback(sid).position_ticks for sid in BODY_IDS},
            'id15_positions': positions(bus)}


def offset(raw):
    return decode_sign_magnitude(struct.unpack_from('<H', raw, 31)[0])


def range_report(cfg, snap):
    reference = policy_calibration_reference('straight')
    report = []
    for index, joint in enumerate(cfg['joints']):
        ticks = snap['body_positions'][str(joint['id'])]
        d = joint['direction']
        zero = ticks - d * float(reference[index]) * 4096 / (2 * 3.141592653589793)
        edges = sorted(zero + d * float(angle) * 4096 / (2 * 3.141592653589793)
                       for angle in JOINT_LIMITS[index])
        report.append({'id': joint['id'], 'name': joint['name'], 'current_tick': ticks,
                       'home_tick_if_current_pose_is_straight': zero+d*float(HOME[index])*4096/(2*3.141592653589793),
                       'full_model_tick_range_if_straight': edges,
                       'full_model_range_fits': edges[0] >= 0 and edges[1] <= 4095,
                       'will_change': joint['id'] == SID})
    return report


def prepare(config_path, bus, backup_root=None):
    raw = config_path.read_bytes()
    cfg = validate_config(raw)
    reject_saved_poses(config_path)
    need(not bus.allow_changes, 'Prepare must be read-only')
    snap = snapshot(bus)
    need(config_path.read_bytes() == raw, 'Configuration changed during prepare')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup = (backup_root or config_path.parent/'backups') / ('duck1-id15-recenter-'+stamp)
    backup.mkdir(parents=True, exist_ok=False)
    (backup/'robot.json.before').write_bytes(raw)
    state = {'operation': OPERATION, 'phase': 'prepared', 'config_path': str(config_path.resolve()),
             'config_sha256': sha(raw), 'snapshot': snap, 'target_tick': TARGET,
             'offset_before': offset(bytes.fromhex(snap['id15_registers_0_55_hex'])),
             'range_report': range_report(cfg, snap), 'events': [],
             'note': 'Range report assumes every joint is still in the straight reference pose; only ID15 will change.'}
    save(backup/'state.json', state)
    return backup, state


def apply(config_path, bus, backup):
    raw = config_path.read_bytes()
    validate_config(raw)
    reject_saved_poses(config_path)
    state_path = backup/'state.json'
    state = json.loads(state_path.read_text())
    need(state['operation'] == OPERATION and state['phase'] == 'prepared', 'Backup not prepared, or already attempted; do not replay')
    need(state['config_path'] == str(config_path.resolve()), 'Different config path')
    need(sha(raw) == state['config_sha256'] and (backup/'robot.json.before').read_bytes() == raw,
         'Config changed since prepare')
    live = snapshot(bus)
    need(live['eeprom_hex'] == state['snapshot']['eeprom_hex'], 'EEPROM changed since prepare')
    before = live['id15_positions']
    origin = state['snapshot']['id15_positions']
    need(max(abs(a-b) for a in before for b in origin) <= 8, 'ID15 moved since prepare; restore pose and prepare again')
    need(config_path.read_bytes() == raw, 'Config changed during preflight')
    old_raw = bytes.fromhex(live['id15_registers_0_55_hex'])

    def emit(event):
        state['events'].append({**event, 'monotonic': time.monotonic()})
        save(state_path, state)

    bus.emit = emit
    state.update(phase='applying', positions_before=before, registers_before=old_raw.hex())
    save(state_path, state)
    bus.allow_changes = True
    try:
        try:
            bus.command(0x03, b'\x37\x00')
            need(bus.read(SID, 55, 1) == b'\x00', 'Unlock failed')
            state['phase'] = 'calibration_pending'
            save(state_path, state)
            try:
                state['calibration_ack'] = bus.command(0x0b, struct.pack('<H', TARGET))
            finally:
                # First register read after calibration, even after ACK error.
                bus.check_off()
            after_positions = positions(bus)
            need(all(abs(p-TARGET) <= 8 for p in after_positions), 'Calibration readback differs from 2048')
            state.update(phase='hardware_calibrated', positions_after=after_positions,
                         torque_immediately_after_calibration=0)
            save(state_path, state)
        finally:
            original = sys.exc_info()[1]
            try:
                bus.cleanup()
            except BaseException as cleanup:
                if original is not None:
                    bus.warn(f'Cleanup also failed: {cleanup}')
                    raise original from cleanup
                raise
        after = snapshot(bus)
        need(all(abs(p-TARGET) <= 8 for p in after['id15_positions']),
             'ID15 moved before final verification')
        new_raw = bytes.fromhex(after['id15_registers_0_55_hex'])
        need(all(old_raw[i] == new_raw[i] for i in range(56) if i not in (31, 32, 42, 43)),
             'Unrelated ID15 register changed')
        for sid in ALL_IDS:
            old = bytes.fromhex(live['eeprom_hex'][str(sid)])
            new = bytes.fromhex(after['eeprom_hex'][str(sid)])
            need(all(old[i] == new[i] for i in range(40) if sid != SID or i not in (31, 32)),
                 f'Unexpected EEPROM change on ID{sid}')
        delta = offset(old_raw)-offset(new_raw)
        need(abs(delta-(TARGET-before[-1])) <= 8, 'Offset change disagrees with intended encoder shift')
        need(all(abs((p-before[-1])-delta) <= 8 for p in after_positions), 'Position delta disagrees with EEPROM offset')
        need(config_path.read_bytes() == raw, 'robot.json changed; do not start motion')
        state.update(phase='complete', effective_delta_ticks=delta, offset_after=offset(new_raw),
                     final_snapshot=after, all_torque_off=True, config_unchanged=True,
                     position_goal_writes=0, torque_on_writes=0,
                     next_step='Restore precise straight pose, then run ./run.sh calibrate --pose straight; do not directly enable torque.')
        save(state_path, state)
    except BaseException as exc:
        state.update(phase='failed', error=f'{type(exc).__name__}: {exc}',
                     maintenance_warnings=list(bus.warnings))
        try:
            save(state_path, state)
        except BaseException as log_error:
            bus.warn(f'Failure state could not be persisted: {log_error}; original failure: {exc}')
        raise
    finally:
        bus.allow_changes = False
    return state


def quiet(port):
    need(shutil.which('fuser') is not None, 'Install psmisc for the UART ownership check')
    result = subprocess.run(['fuser', port], capture_output=True, text=True)
    need(result.returncode == 1 and not result.stdout.strip() and not result.stderr.strip(),
         f'UART occupied or ownership check failed: {result.stdout} {result.stderr}')
    for service in ('microduck-xbox.service', 'robotd.service'):
        result = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True)
        need(result.stdout.strip() not in ('active', 'activating', 'reloading'), f'Stop {service} first')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'robot.json')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    cfg = validate_config(args.config.read_bytes())
    need(not args.apply or (args.backup is not None and args.backup.is_dir()), '--apply requires --backup from prepare')
    quiet(cfg['serial']['port'])
    with MaintenanceBus(**{**cfg['serial'], 'timeout': .1}) as bus:
        if args.apply:
            state = apply(args.config, bus, args.backup)
            print(json.dumps({k: state[k] for k in ('phase', 'effective_delta_ticks', 'all_torque_off', 'next_step')}, ensure_ascii=False))
        else:
            backup, state = prepare(args.config, bus)
            print(json.dumps({'phase': state['phase'], 'backup': str(backup),
                              'id15_positions': state['snapshot']['id15_positions'],
                              'target_tick': TARGET, 'hardware_writes': False,
                              'range_report': state['range_report']}, ensure_ascii=False, indent=2))
            print('准备完成，未写硬件。只有明确使用 --apply --backup 上述目录才会重设 ID15 编码器参考。')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'停止: {exc}', file=sys.stderr)
        sys.exit(1)
