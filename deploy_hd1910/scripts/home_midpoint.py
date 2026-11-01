"""Return to the selected HOME, recenter 14 encoders, and migrate references.

Default is an offline preview. --move --apply --supported-confirmed is the
explicit one-shot operation. Mouth ID10 is read only. Finishes torque OFF.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microduck_deploy.config import calibration_arrays, load_config, require_calibration
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks, ticks_to_q
from microduck_deploy.servo import ServoBus, build_packet
from recenter_duck1_all import MaintenanceBus, need, offset, quiet, sha

IDS = [*range(1, 10), *range(11, 16)]
TARGET = 2048


def save(path, value):
    """Durably replace a JSON journal or configuration before the next write."""
    path = Path(path)
    temporary = path.with_name(path.name+'.midpoint.tmp')
    with temporary.open('w') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def valid_config(cfg):
    require_calibration(cfg, imu=False)
    need([j['id'] for j in cfg['joints']] == IDS, 'Requires the duck1 14-joint mapping')
    need([j['name'] for j in cfg['joints']] == list(JOINT_NAMES), 'Unexpected joint order')
    need(cfg['serial'].get('register_profile') == 'hls_2', 'Requires hls_2')
    need(cfg['calibration']['pose'] in ('home', 'straight', 'zero'),
         'This tool migrates home/straight/zero references only')
    need('encoder_reference_maintenance' not in cfg, 'Unfinished midpoint maintenance; inspect its backup first')


def shift_tick(value, delta):
    need(type(value) in (int, float) and math.isfinite(value), 'Invalid saved encoder value')
    result = value+delta
    need(0 <= value <= 4095 and 0 <= result <= 4095,
         'A saved reference would cross the encoder boundary; no wrapping is allowed')
    return result


def migrate_config(cfg, deltas, record):
    """Shift actual encoder coordinates and zeros together; physical HOME stays."""
    valid_config(cfg)
    need(set(deltas) == set(IDS) and all(type(d) is int for d in deltas.values()), 'Need 14 integer shifts')
    result = deepcopy(cfg)
    before = q_to_ticks(HOME, *calibration_arrays(cfg))
    for j in result['joints']:
        j['zero_tick'] += deltas[j['id']]
    cal = result['calibration']
    need(set(cal['joint_ids']) == set(IDS), 'Incomplete calibration IDs')
    for field in ('captured_ticks', 'home_ticks'):
        need(len(cal[field]) == 14, 'Incomplete calibration reference')
        cal[field] = [shift_tick(v, deltas[sid]) for sid, v in zip(cal['joint_ids'], cal[field])]
    endpoint = result.get('home_endpoint_reference')
    if endpoint and 'new_home_ticks' in endpoint:
        endpoint['new_home_ticks'] = [shift_tick(v, deltas[sid])
                                     for sid, v in zip(IDS, endpoint['new_home_ticks'])]
    result.setdefault('encoder_reference_migrations', []).append(deepcopy(record))
    after = q_to_ticks(HOME, *calibration_arrays(result))
    need(after.tolist() == [int(v)+deltas[sid] for sid, v in zip(IDS, before)], 'HOME migration mismatch')
    import numpy as np
    np.testing.assert_allclose(ticks_to_q(before, *calibration_arrays(cfg)),
                               ticks_to_q(after, *calibration_arrays(result)), atol=1e-12, rtol=0)
    return result


def migrate_pose(pose, deltas, record):
    result = deepcopy(pose)
    need(isinstance(result.get('rows'), list), 'Unknown saved pose format')
    changed = False
    for row in result['rows']:
        sid = row.get('id')
        if sid not in deltas:
            continue
        need(row.get('deg') is None, 'Saved degree-based pose needs a separate migration')
        row['pos'] = shift_tick(row['pos'], deltas[sid])
        changed = True
    if changed:
        result.setdefault('encoder_reference_migrations', []).append(deepcopy(record))
    return result


def inventory(base):
    """Only live profiles/poses; logs, backups and validation reports are history."""
    files = {}
    for path in sorted(base.glob('robot*.json')):
        cfg = json.loads(path.read_text())
        if not cfg.get('calibration') and path.name == 'robot.example.json':
            continue
        valid_config(cfg)
        need(not path.is_symlink(), 'Profile symlink requires explicit review')
        files[path] = ('config', path.read_bytes())
    need(base/'robot.json' in files, 'Missing robot.json')
    for path in sorted((base.parent/'servo_web_hd1910'/'poses').glob('*.json')):
        pose = json.loads(path.read_text())
        if pose.get('fake') is True:
            continue
        migrate_pose(pose, dict.fromkeys(IDS, 0), {})
        files[path] = ('pose', path.read_bytes())
    selection = base/'walking_home_selection.json'
    if selection.exists():
        need(len(json.loads(selection.read_text())['new_home_ticks']) == 14, 'Unknown HOME selection format')
        files[selection] = ('selection', selection.read_bytes())
    return files


class MidpointBus(MaintenanceBus):
    """Write only a precomputed snapshot offset and current-position OFF goal."""
    planned_offsets = None
    stage_targets = None

    def _send(self, packet):
        sid, instruction, params = packet[2], packet[4], bytes(packet[5:-1])
        need(packet == build_packet(sid, instruction, params), 'Invalid packet checksum/length')
        if sid == 254 and instruction == 130 and params == bytes((56,15,*IDS)):
            return ServoBus._send(self, packet)  # One read-only snapshot request.
        need(sid in IDS or (sid == 10 and instruction == 2), 'Mouth writes are forbidden')
        need(instruction != 11, 'Snapshot calibration writes a fixed offset; live-position calibration is forbidden')
        if instruction == 3 and len(params) == 3 and params[0] == 31:
            need(self.planned_offsets is not None and sid == self.active_id
                 and params[1:] == encode_offset(self.planned_offsets[sid]),
                 'Only the offset calculated from the HOME snapshot may be written')
            self.check_off(sid)
            need(self.read(sid,55,1)==b'\x00', 'Offset EEPROM must be unlocked')
            self.emit({'event':'write_snapshot_offset','id':sid,'packet_hex':packet.hex()})
            return ServoBus._send(self, packet)
        if instruction == 3 and len(params) == 3 and params[0] == 42:
            need(self.stage_targets is not None and sid == self.active_id
                 and struct.unpack('<H', params[1:])[0] == self.stage_targets[sid],
                 'Only the final current-position snapshot may be staged while OFF')
            self.check_off(sid)
            self.emit({'event':'stage_off_goal', 'id':sid, 'packet_hex':packet.hex()})
            return ServoBus._send(self, packet)
        return super()._send(packet)


def encode_offset(value):
    # HLS_2 position-offset register 31: signed magnitude, direction BIT15,
    # documented range -4095..4095. It is independent of current shaft motion.
    need(type(value) is int and -4095 <= value <= 4095, 'Snapshot offset exceeds HLS_2 register range')
    return struct.pack('<H', abs(value) | (0x8000 if value < 0 else 0))


def check_hardware(bus, torque):
    result = {}
    for sid in IDS:
        row = bus.read_configuration(sid)
        need(row['firmware'] == [3, 46] and row['model_version'] == [10, 31], f'ID{sid}: wrong firmware/model')
        need(row['mode'] == 4 and row['angle_resolution'] == 1 and row['lock'] == 1,
             f'ID{sid}: mode/resolution/lock changed')
        need(row['position_min'] == 0 and row['position_max'] == 4095 and row['torque'] == torque,
             f'ID{sid}: unexpected limits/torque')
        result[str(sid)] = bus.read(sid, 0, 56).hex()
    return result


def off_all(bus):
    failures = []
    for sid in IDS:
        bus.active_id = sid
        try:
            bus.command(sid, 3, b'\x28\x00')
            need(bus.read(sid, 40, 1) == b'\x00', f'ID{sid}: torque OFF failed')
        except BaseException as exc:
            failures.append(f'ID{sid}: {exc}')
        finally:
            bus.active_id = None
    need(not failures, '; '.join(failures))


def take_home_snapshot(bus, seconds, *, clock=time.monotonic, sleep=time.sleep):
    """Optional fixed hold, then exactly one reference frame; no motion gate."""
    started = clock()
    if seconds:
        print(f'回HOME后保持上力{seconds:g}秒，然后读取一次位置快照。',flush=True)
        while clock()-started < seconds:
            sleep(min(.05,seconds-(clock()-started)))
    feedback = bus.read_feedback_many(IDS,sync=True)
    positions = {sid:feedback[sid].position_ticks for sid in IDS}
    need(all(type(v) is int and 0 <= v <= 4095 for v in positions.values()), 'Invalid single-turn reference snapshot')
    return positions, {'requested_s':seconds,'elapsed_s':clock()-started,
                       'reference_sample_count':1,'sync_read':True,'motion_stability_checked':False,
                       'torque_held_on':True}


def apply_midpoint(base, cfg, files, bus, backup, state, *, hold_seconds=0.):
    state_path = backup/'state.json'
    marker = base/'logs/home_midpoint.pending.json'

    def emit(event):
        state['events'].append(event)
        save(state_path, state)

    bus.emit = emit
    try:
        registers = check_hardware(bus, 1)
        mouth_before = bus.read(10, 0, 56)
        home = dict(zip(IDS, q_to_ticks(HOME, *calibration_arrays(cfg)).tolist()))
        held, hold_report = take_home_snapshot(bus,hold_seconds)
        state.update(phase='home_held', held_ticks=held, home_targets=home, home_hold=hold_report)
        save(state_path, state)
        predicted = {sid:TARGET-held[sid] for sid in IDS}
        offsets = {sid:offset(bytes.fromhex(registers[str(sid)]))-predicted[sid] for sid in IDS}
        for value in offsets.values():
            encode_offset(value)
        record = {'utc':datetime.now(timezone.utc).isoformat(), 'backup':str(backup),
                  'kind':'hardware_midpoint_and_matching_software_reference_shift',
                  'ids':IDS, 'target_tick':TARGET, 'physical_home_preserved':True,
                  'reference':'one_torque_on_HOME_snapshot','snapshot_ticks':held}
        for path, (kind, raw) in files.items():
            need(path.read_bytes() == raw, f'File changed: {path}')
            value = json.loads(raw)
            if kind == 'config':
                migrate_config(value, predicted, record)
            elif kind == 'pose':
                migrate_pose(value, predicted, record)
            else:
                [shift_tick(v, predicted[sid]) for sid, v in zip(IDS, value['new_home_ticks'])]
        state.update(phase='prepared', registers_before=registers, planned_offsets=offsets,
                     mouth_before_hex=mouth_before.hex())
        save(state_path, state)
        # Existing require_calibration rejects every profile until migration
        # completes. This also protects independent home-calibration packages.
        expected_current = {}
        for path, (kind, raw) in files.items():
            if kind == 'config':
                pending = json.loads(raw)
                pending['calibration'] = None
                pending['encoder_reference_maintenance'] = {'phase':'pending', 'backup':str(backup)}
                for joint in pending['joints']:
                    joint['zero_tick'] = None
                save(path, pending)
            expected_current[path] = path.read_bytes()
        save(marker, {'backup':str(backup), 'phase':'pending'})
        state['phase'] = 'calibrating'
        save(state_path, state)
        print('已记录一次HOME快照；现在卸力，按快照写入固定偏移，关节移动不会改变参考。',flush=True)
        off_all(bus)
        bus.planned_offsets = offsets
        deltas = {}
        for sid in IDS:
            need(all(bus.read(other, 40, 1) == b'\x00' for other in IDS), 'Torque state changed')
            old = bus.read(sid, 0, 56)
            need(old[:40].hex() == registers[str(sid)][:80], f'ID{sid} EEPROM changed')
            state['current_id'] = sid
            state['results'][str(sid)] = {'phase':'pending','reference_snapshot_tick':held[sid],
                                        'planned_offset':offsets[sid],'registers_before':old.hex()}
            save(state_path, state)
            bus.active_id = sid
            try:
                bus.command(sid, 3, b'\x37\x00')
                need(bus.read(sid, 55, 1) == b'\x00', 'Unlock failed')
                try:
                    bus.command(sid,3,b'\x1f'+encode_offset(offsets[sid]))
                finally:
                    bus.check_off(sid)  # First register read after writing EEPROM.
                need(offset(bus.read(sid,0,56)) == offsets[sid], f'ID{sid}: offset readback failed')
            finally:
                bus.cleanup(sid)
                bus.active_id = None
            new = bus.read(sid, 0, 56)
            # A firmware-adjusted RAM goal is harmless while OFF; replace it
            # with the final current-position snapshot before finishing.
            need(all(old[i] == new[i] for i in range(56) if i not in (31,32,42,43)),
                 f'ID{sid}: unrelated register changed')
            delta = offset(old)-offset(new)
            need(delta == predicted[sid], f'ID{sid}: reference shift differs from the snapshot')
            deltas[sid] = delta
            state['results'][str(sid)].update(phase='complete',offset_before=offset(old),offset_after=offset(new),
                                             effective_delta_ticks=delta, registers_after=new.hex())
            save(state_path, state)
            print(json.dumps({'event':'midpoint','id':sid,'snapshot_maps_to_ticks':held[sid]+delta},ensure_ascii=False),flush=True)
        bus.planned_offsets = None
        check_hardware(bus, 0)
        need(bus.read(10, 0, 56) == mouth_before, 'Mouth registers changed')
        # Stage the new current goal while OFF, so an old RAM goal cannot
        # accidentally be used after an independent torque-ON command.
        final_feedback = bus.read_feedback_many(IDS,sync=True)
        final = {sid:final_feedback[sid].position_ticks for sid in IDS}
        need(all(type(v) is int and 0 <= v <= 4095 for v in final.values()), 'Current pose outside single-turn range; remain OFF')
        bus.stage_targets = final
        try:
            for sid in IDS:
                bus.active_id = sid
                bus.command(sid, 3, b'\x2a'+struct.pack('<H',final[sid]))
                need(struct.unpack('<H',bus.read(sid,42,2))[0] == final[sid], 'Goal staging failed')
        finally:
            bus.stage_targets = None
            bus.active_id = None
        record['effective_delta_ticks'] = {str(sid):value for sid, value in deltas.items()}
        updated = {}
        for path, (kind, raw) in files.items():
            data = json.loads(raw)
            if kind == 'config':
                data = migrate_config(data, deltas, record)
            elif kind == 'pose':
                data = migrate_pose(data, deltas, record)
            else:
                data['new_home_ticks'] = [shift_tick(v, deltas[sid]) for sid, v in zip(IDS, data['new_home_ticks'])]
                data.setdefault('encoder_reference_migrations', []).append(record)
            updated[path] = data
        state.update(phase='software_commit', effective_delta_ticks=record['effective_delta_ticks'])
        save(state_path, state)
        need(all(path.read_bytes() == expected for path, expected in expected_current.items()),
             'A live file changed during maintenance; stop without overwriting it')
        for path, data in updated.items():
            save(path, data)
        for path, (kind, _) in files.items():
            if kind == 'config':
                valid_config(load_config(path))
        need(all(bus.read(sid, 40, 1) == b'\x00' for sid in IDS), 'Final torque OFF verification failed')
        state.update(phase='complete', current_id=None, final_ticks=final,
                     snapshot_maps_to_ticks=dict.fromkeys(IDS,TARGET),
                     all_torque_off=True, mouth_unchanged=True, torque_on_writes=0,
                     migrated_files=[str(p) for p in updated])
        save(state_path, state)
        marker.unlink()
        print(json.dumps({'event':'complete','snapshot_midpoint':TARGET,'reference_sample_count':1,'all_torque_off':True,
                          'mouth_unchanged':True, 'backup':str(backup)},ensure_ascii=False), flush=True)
        return state
    except BaseException as exc:
        try:
            off_all(bus)
        except BaseException as cleanup:
            bus.warn(f'Torque OFF cleanup failed: {cleanup}')
        state.update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        try:
            save(state_path, state)
        except BaseException as log_error:
            bus.warn(f'Could not save failure journal: {log_error}')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'robot.home_actions_20261101.json')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--supported-confirmed', action='store_true')
    parser.add_argument('--hold-seconds', type=float, default=0.,
                        help='Optional fixed torque-on hold before taking one HOME snapshot, default 0')
    args = parser.parse_args(argv)
    if not math.isfinite(args.hold_seconds) or not 0 <= args.hold_seconds <= 30:
        parser.error('--hold-seconds must be in [0,30]')
    if any((args.move, args.apply, args.supported_confirmed)) and not all((args.move, args.apply, args.supported_confirmed)):
        parser.error('Execution requires --move --apply --supported-confirmed together')
    base = args.config.resolve().parent
    need(not (base/'logs/home_midpoint.pending.json').exists(), 'Unfinished maintenance; inspect the recorded backup, do not repeat')
    cfg = load_config(args.config)
    valid_config(cfg)
    files = inventory(base)
    need(args.config.resolve() in files, 'Selected HOME config is not a live robot profile')
    need(cfg['calibration']['pose'] == 'home', 'Choose the saved tuned HOME profile')
    if not args.apply:
        print(json.dumps({'mode':'offline_preview', 'hardware_opened':False, 'hardware_writes':0,
                          'home_config':str(args.config.resolve()), 'ids':IDS, 'mouth_id_unchanged':10,
                          'target_midpoint':TARGET, 'profiles_and_poses':[str(p) for p in files],
                          'home_hold_seconds':args.hold_seconds,
                          'reference_sample_count':1,'motion_stability_checked':False,
                          'steps':['supported HOME','one torque-on snapshot','compute fixed encoder offsets','torque OFF','write snapshot offsets',
                                   'migrate matching software references','verify and remain OFF']},ensure_ascii=False,indent=2))
        return 0
    (base/'logs').mkdir(exist_ok=True)
    with (base/'logs/supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        quiet(cfg['serial']['port'])
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backup = base/'.backups'/('home_midpoint_'+stamp)
        backup.mkdir(parents=True, exist_ok=False)
        original_files = {}
        for index, (path, (kind, raw)) in enumerate(files.items()):
            name = f'{index:02d}_{path.name}.before'
            with (backup/name).open('wb') as original:
                original.write(raw)
                original.flush()
                os.fsync(original.fileno())
            original_files[str(path)] = {'backup_file':name, 'sha256':sha(raw), 'kind':kind}
        state = {'phase':'starting', 'source_config':str(args.config.resolve()),
                 'files_before':original_files, 'events':[], 'results':{}, 'ids':IDS}
        save(backup/'state.json', state)
        print(json.dumps({'event':'backup', 'directory':str(backup),
                          'state_file':str(backup/'state.json')},ensure_ascii=False),flush=True)
        # Acquire no second maintenance lock in the existing HOME child.
        # Reset torque first so its established soft-start path is used.
        with MidpointBus(**{**cfg['serial'], 'timeout':.1}) as bus:
            off_all(bus)
        child = subprocess.Popen([sys.executable, str(base/'scripts/supported_home.py'),
            '--config', str(args.config.resolve()), '--target','home', '--soft-gains',
            '--log-only-feedback', '--move'])
        try:
            need(child.wait() == 0, 'Supported HOME failed; encoder references were not changed')
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            with MidpointBus(**{**cfg['serial'], 'timeout':.1}) as bus:
                off_all(bus)
            raise
        try:
            print('开始中位校准：读取一次HOME快照，完成后保持卸力。', flush=True)
            with MidpointBus(**{**cfg['serial'], 'timeout':.1}) as bus:
                apply_midpoint(base, cfg, files, bus, backup, state, hold_seconds=args.hold_seconds)
        except BaseException:
            try:
                with MidpointBus(**{**cfg['serial'], 'timeout':.1}) as bus:
                    off_all(bus)
            except BaseException as cleanup:
                print(f'卸力清理未能确认：{cleanup}', file=sys.stderr, flush=True)
            raise
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'中位校准停止：{exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
