#!/usr/bin/env python3
"""Relabel duck1's 15 currently supported positions as 2048; no motion targets.

Default: read-only prepare. Explicit --apply --backup DIR or
--prepare-and-apply performs one-time encoder reference maintenance. Physical model calibration remains unset. Maintain every
joint's current supported pose until final verification; do not enable torque.
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
from microduck_deploy.policy import JOINT_NAMES
from microduck_deploy.servo import ServoBus, ServoTimeout, build_packet, decode_sign_magnitude

IDS = list(range(1, 16))
BODY_IDS = list(range(1, 10)) + list(range(11, 16))
TARGET = 2048
OPERATION = 'duck1-all15-supported-current-reference-to-2048-v1'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_config(raw):
    cfg = json.loads(raw)
    require_mapping(cfg)
    need([j['name'] for j in cfg['joints']] == list(JOINT_NAMES), 'Unexpected body joint order')
    need([j['id'] for j in cfg['joints']] == BODY_IDS, 'Not the duck1 body ID mapping; mouth must be ID10')
    need(cfg.get('calibration') is None and all(j.get('zero_tick') is None for j in cfg['joints']),
         'Existing body calibration/zero: stop; old encoder references cannot be preserved by this tool')
    jaw = cfg.get('jaw') or {}
    need(not jaw.get('calibration') and jaw.get('closed_tick') is None and jaw.get('open_tick') is None,
         'Existing jaw calibration/endpoints: stop before relabelling mouth ID10')
    need(cfg['serial'].get('register_profile') == 'hls_2', 'Requires hls_2 profile')
    return cfg


def reject_saved_poses(config_path):
    for path in (config_path.parent.parent/'servo_web_hd1910'/'poses').glob('*.json'):
        need(json.loads(path.read_text()).get('fake') is True,
             f'Real saved pose must be reviewed/archived first: {path}')


class MaintenanceBus(ServoBus):
    def __init__(self, *args, emit=None, **kwargs):
        self.emit = emit or (lambda event: None)
        self.active_id = None
        self.emergency_off_attempted = set()
        self.warnings = []
        super().__init__(*args, **kwargs)

    def warn(self, message):
        self.warnings.append(message)
        try:
            print(f'维护告警: {message}', file=sys.stderr, flush=True)
        except BaseException:
            pass

    def _send(self, packet):
        need(len(packet) >= 6, 'Invalid packet')
        sid, instruction, params = packet[2], packet[4], bytes(packet[5:-1])
        need(packet == build_packet(sid, instruction, params), 'Invalid packet checksum/length')
        need(sid in IDS, 'No broadcast or unknown ID')
        if instruction != 0x02:
            off = instruction == 0x03 and params == b'\x28\x00'
            lock = instruction == 0x03 and params in (b'\x37\x00', b'\x37\x01')
            calibrate = instruction == 0x0b and params == struct.pack('<H', TARGET)
            need(sid == self.active_id and (off or lock or calibrate), 'Forbidden hardware write')
            if not off:
                self.check_off(sid)
            event = {'event': 'write_packet', 'id': sid, 'instruction': instruction,
                     'params_hex': params.hex(), 'packet_hex': packet.hex()}
            if off:
                try:
                    self.emit(event)
                except BaseException as exc:
                    self.warn(f'ID{sid} emergency OFF event could not be persisted: {exc}')
            else:
                self.emit(event)
        return super()._send(packet)

    def command(self, sid, instruction, params):
        with self._io:
            deadline = self._send(build_packet(sid, instruction, params))
            try:
                self._read_status(deadline, sid, 0)
            except ServoTimeout:
                return None  # Do not replay writes; explicit readback decides.
        return 0

    def emergency_off(self, sid):
        need(sid == self.active_id, 'Emergency OFF may only address the current servo')
        if sid not in self.emergency_off_attempted:
            self.emergency_off_attempted.add(sid)
            try:
                self.command(sid, 0x03, b'\x28\x00')
            except BaseException as exc:
                self.warn(f'ID{sid} emergency OFF send/ACK failed: {exc}')
        need(self.read(sid, 40, 1) == b'\x00', f'ID{sid} emergency OFF could not be confirmed')

    def check_off(self, sid):
        try:
            torque = self.read(sid, 40, 1)
        except BaseException as original:
            try:
                self.emergency_off(sid)
            except BaseException as cleanup:
                self.warn(f'ID{sid} torque read and emergency verification failed: {cleanup}')
            raise original
        if torque != b'\x00':
            self.emergency_off(sid)
            raise RuntimeError(f'Unexpected ID{sid} torque ON; emergency OFF sent; stop')

    def cleanup(self, sid):
        original = None
        try:
            self.check_off(sid)
        except BaseException as exc:
            original = exc
        try:
            self.check_off(sid)
            self.command(sid, 0x03, b'\x37\x01')
            need(self.read(sid, 55, 1) == b'\x01', f'ID{sid} relock failed')
            self.check_off(sid)
        except BaseException as exc:
            if original is not None:
                self.warn(f'ID{sid} additional cleanup failure: {exc}')
                raise original from exc
            raise
        if original is not None:
            raise original


def position_samples(bus, ids, count=12):
    samples = {str(sid): [] for sid in ids}
    for _ in range(count):
        for sid in ids:
            if sid == bus.active_id:
                bus.check_off(sid)
            else:
                need(bus.read(sid, 40, 1) == b'\x00', f'ID{sid} must be torque OFF')
            samples[str(sid)].append(bus.read_feedback(sid).position_ticks)
        time.sleep(.025)
    for sid, values in samples.items():
        need(max(values)-min(values) <= 8, f'ID{sid} moved during capture: {values}')
    # Multi-turn/negative raw feedback is retained verbatim before calibration.
    return samples


def snapshot(bus):
    configurations = {str(sid): bus.read_configuration(sid) for sid in IDS}
    for sid, row in configurations.items():
        need(row['torque'] == 0 and row['lock'] == 1, f'ID{sid} must be OFF and locked')
        need(row['firmware'] == [3, 46] and row['model_version'] == [10, 31], f'ID{sid}: unexpected firmware/model')
        need(row['mode'] == 4 and row['angle_resolution'] == 1, f'ID{sid}: unexpected mode/resolution')
        need(row['position_min'] == 0 and row['position_max'] == 4095, f'ID{sid}: unexpected limits')
    registers = {str(sid): bus.read(sid, 0, 56).hex() for sid in IDS}
    return {'configurations': configurations, 'registers_0_55_hex': registers,
            'eeprom_hex': {sid: bytes.fromhex(raw)[:40].hex() for sid, raw in registers.items()},
            'position_samples': position_samples(bus, IDS)}


def offset(raw):
    return decode_sign_magnitude(struct.unpack_from('<H', raw, 31)[0])


def verify_shift(before, after, raw_before, raw_after):
    """Check periodic encoder consistency; never wrap/modify a target or reading."""
    need(all(abs(p-TARGET) <= 8 for p in after), 'Raw calibration feedback differs from 2048')
    encoder_delta = offset(raw_before)-offset(raw_after)
    position_delta = after[-1]-before[-1]
    turns = round((position_delta-encoder_delta)/4096)
    need(all(abs((p-before[-1])-encoder_delta-turns*4096) <= 8 for p in after),
         'Position change disagrees with EEPROM offset, including whole-turn state')
    return {'offset_before': offset(raw_before), 'offset_after': offset(raw_after),
            'offset_reference_delta': encoder_delta, 'raw_position_delta': position_delta,
            'whole_turn_state_delta': turns}


def prepare(config_path, bus, backup_root=None):
    raw = config_path.read_bytes()
    cfg = validate_config(raw)
    reject_saved_poses(config_path)
    need(bus.active_id is None, 'Prepare must be read-only')
    snap = snapshot(bus)
    need(config_path.read_bytes() == raw, 'Config changed during prepare')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    backup = (backup_root or config_path.parent/'backups')/('duck1-all15-recenter-'+stamp)
    backup.mkdir(parents=True, exist_ok=False)
    (backup/'robot.json.before').write_bytes(raw)
    state = {'operation': OPERATION, 'phase': 'prepared', 'target_tick': TARGET,
             'ids': IDS, 'body_joint_ids': BODY_IDS, 'mouth_id': 10,
             'mapped_joints': cfg['joints'], 'config_path': str(config_path.resolve()),
             'config_sha256': sha(raw), 'snapshot': snap, 'events': [], 'results': {},
             'note': 'Current supported physical poses relabelled only; no model-angle calibration or motion goals.'}
    save(backup/'state.json', state)
    return backup, state


def process_one(bus, sid, config_path, config_bytes, state, state_path):
    # Check all torque states immediately before each servo's maintenance.
    for other in IDS:
        need(bus.read(other, 40, 1) == b'\x00', f'ID{other} no longer torque OFF')
    before = position_samples(bus, [sid])[str(sid)]
    prepared = state['snapshot']['position_samples'][str(sid)]
    need(max(abs(a-b) for a in before for b in prepared) <= 8, f'ID{sid} moved since prepare; stop')
    old_raw = bus.read(sid, 0, 56)
    need(old_raw[:40].hex() == state['snapshot']['eeprom_hex'][str(sid)], f'ID{sid} EEPROM changed before maintenance')
    need(old_raw[40] == 0 and old_raw[55] == 1, f'ID{sid} changed torque/lock')
    need(config_path.read_bytes() == config_bytes, 'Config changed during operation')
    state.update(phase='servo_pending', current_id=sid)
    state['results'][str(sid)] = {'phase': 'pending', 'positions_before': before,
                                 'registers_before': old_raw.hex()}
    save(state_path, state)
    bus.active_id = sid
    result = state['results'][str(sid)]
    try:
        try:
            bus.command(sid, 0x03, b'\x37\x00')
            need(bus.read(sid, 55, 1) == b'\x00', f'ID{sid} unlock failed')
            result['phase'] = 'calibration_pending'
            save(state_path, state)
            try:
                result['calibration_ack'] = bus.command(sid, 0x0b, struct.pack('<H', TARGET))
            finally:
                # First register read after calibration, including ACK failure.
                bus.check_off(sid)
            after = position_samples(bus, [sid])[str(sid)]
            need(all(abs(p-TARGET) <= 8 for p in after), f'ID{sid} raw calibration feedback differs from 2048')
            result.update(phase='hardware_calibrated', positions_after=after,
                          torque_immediately_after_calibration=0)
            save(state_path, state)
        finally:
            original = sys.exc_info()[1]
            try:
                bus.cleanup(sid)
            except BaseException as cleanup:
                if original is not None:
                    bus.warn(f'ID{sid} cleanup also failed: {cleanup}')
                    raise original from cleanup
                raise
        new_raw = bus.read(sid, 0, 56)
        need(all(old_raw[i] == new_raw[i] for i in range(56) if i not in (31, 32, 42, 43)),
             f'ID{sid} unrelated register changed')
        result.update(verify_shift(before, after, old_raw, new_raw))
        result.update(phase='complete', registers_after=new_raw.hex())
        save(state_path, state)
    finally:
        bus.active_id = None


def apply(config_path, bus, backup):
    raw = config_path.read_bytes()
    validate_config(raw)
    reject_saved_poses(config_path)
    state_path = backup/'state.json'
    state = json.loads(state_path.read_text())
    need(state['operation'] == OPERATION and state['phase'] == 'prepared', 'Backup not prepared or already attempted; do not replay')
    need(state['ids'] == IDS and state['config_path'] == str(config_path.resolve()), 'Backup scope/path differs')
    need(sha(raw) == state['config_sha256'] and (backup/'robot.json.before').read_bytes() == raw, 'Config changed since prepare')
    live = snapshot(bus)
    need(live['eeprom_hex'] == state['snapshot']['eeprom_hex'], 'EEPROM changed since prepare')
    for sid in IDS:
        need(max(abs(a-b) for a in live['position_samples'][str(sid)]
                 for b in state['snapshot']['position_samples'][str(sid)]) <= 8,
             f'ID{sid} moved since prepare')
    need(config_path.read_bytes() == raw, 'Config changed during preflight')

    def emit(event):
        state['events'].append({**event, 'monotonic': time.monotonic()})
        save(state_path, state)

    bus.emit = emit
    state.update(phase='applying')
    save(state_path, state)
    try:
        for sid in IDS:
            state.update(phase='servo_preflight', current_id=sid)
            save(state_path, state)
            process_one(bus, sid, config_path, raw, state, state_path)
        final = snapshot(bus)
        for sid in IDS:
            values = final['position_samples'][str(sid)]
            need(all(abs(p-TARGET) <= 8 for p in values), f'ID{sid} moved before final verification')
            old = bytes.fromhex(state['snapshot']['registers_0_55_hex'][str(sid)])
            new = bytes.fromhex(final['registers_0_55_hex'][str(sid)])
            need(all(old[i] == new[i] for i in range(56) if i not in (31, 32, 42, 43)),
                 f'ID{sid} unrelated register changed by final verification')
        need(config_path.read_bytes() == raw, 'robot.json changed; do not start motion')
        state.update(phase='complete', current_id=None, final_snapshot=final,
                     all_torque_off=True, all_locked=True, config_unchanged=True,
                     position_goal_writes=0, torque_on_writes=0,
                     next_step='Encoder references only. Re-establish a known mechanical pose and run software calibration; do not directly enable torque.')
        save(state_path, state)
    except BaseException as exc:
        # Only reached after the durable applying phase: hardware may already
        # have changed. Check every servo, including completed/non-current IDs.
        # An unreachable servo must not prevent OFF attempts on the others.
        for sid in IDS:
            bus.active_id = sid
            try:
                bus.check_off(sid)
            except BaseException as cleanup:
                bus.warn(f'Global torque cleanup ID{sid}: {cleanup}')
            finally:
                bus.active_id = None
        state.update(phase='failed', error=f'{type(exc).__name__}: {exc}', maintenance_warnings=list(bus.warnings))
        try:
            save(state_path, state)
        except BaseException as log_error:
            bus.warn(f'Failure state could not be persisted: {log_error}; original failure: {exc}')
        raise
    finally:
        bus.active_id = None
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--prepare-and-apply', action='store_true',
                      help='Create a fresh backup then explicitly apply within this session')
    parser.add_argument('--backup', type=Path)
    args = parser.parse_args()
    need(not (args.prepare_and_apply and args.backup is not None),
         '--prepare-and-apply creates its own backup; do not supply --backup')
    cfg = validate_config(args.config.read_bytes())
    need(not args.apply or (args.backup is not None and args.backup.is_dir()), '--apply requires --backup from prepare')
    quiet(cfg['serial']['port'])
    with MaintenanceBus(**{**cfg['serial'], 'timeout': .1}) as bus:
        if args.apply:
            state = apply(args.config, bus, args.backup)
            print(json.dumps({k: state[k] for k in ('phase', 'all_torque_off', 'all_locked', 'next_step')}, ensure_ascii=False))
        else:
            backup, state = prepare(args.config, bus)
            print(json.dumps({'phase': 'prepared', 'backup': str(backup), 'ids': IDS,
                              'positions': {sid: values[-1] for sid, values in state['snapshot']['position_samples'].items()},
                              'target_tick': TARGET, 'hardware_writes': False}, ensure_ascii=False, indent=2), flush=True)
            if args.prepare_and_apply:
                print(f'备份已保存：{backup}；开始执行明确请求的全15颗编码器参考重设。', flush=True)
                state = apply(args.config, bus, backup)
                print(json.dumps({k: state[k] for k in ('phase', 'all_torque_off', 'all_locked', 'next_step')}, ensure_ascii=False), flush=True)
            else:
                print('只读准备完成；保持全部15颗当前姿态。--apply --backup 才会改写编码器参考，不发送位置运动目标。')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f'停止: {exc}', file=sys.stderr)
        sys.exit(1)
