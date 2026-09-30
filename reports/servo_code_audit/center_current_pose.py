"""Relabel the supported current pose as encoder midpoint, without goal writes.

Only READ, individual CALIBRATE(2048), lock and torque-OFF packets are allowed.
This is not a model-angle or HOME calibration. Stop servo-web before use.
"""
import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

WEB = Path.home() / 'servo_web_hd1910'
sys.path.insert(0, str(WEB))
import feetech

IDS = list(range(1, 15))


class OriginBus(feetech.FeetechBus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._io = threading.RLock()

    def _send(self, sid, instr, params=b''):
        assert sid in IDS, 'Unexpected servo ID'
        if instr != feetech.READ:
            off = instr == feetech.WRITE and bytes(params) == b'\x28\x00'
            allowed = off or (instr == feetech.WRITE and bytes(params) in (b'\x37\x00', b'\x37\x01')) or (instr == feetech.CALIBRATE and bytes(params) == b'\x00\x08')
            assert allowed, 'Goal/torque-ON/other register writes are forbidden'
            if not off:
                assert self.read_u8(sid, 40) == 0, 'Torque must be off'
        super()._send(sid, instr, params)


def check_off(bus, sid):
    torque = bus.read_u8(sid, 40)
    if torque != 0:
        bus.write_u8(sid, 40, 0)
        assert bus.read_u8(sid, 40) == 0, 'Emergency torque-OFF verification failed'
        raise RuntimeError(f'ID{sid} unexpected torque={torque}; switched off; batch aborted')


def sample(bus, sid, n=6):
    out = []
    for _ in range(n):
        check_off(bus, sid)
        s = bus.state(sid)
        assert s['err'] == 0 and s['status'] == 0, f'ID{sid} status error'
        out.append(s['pos'])
        time.sleep(.025)
    assert max(out) - min(out) <= 8, f'ID{sid} is moving: {out}'
    return out


def recenter_one(bus, sid, expected, emit):
    before = bus.dump(sid)['raw']
    assert before[0:2] == [3, 46] and before[2] == 0 and before[3:6] == [10, 31, sid]
    assert before[30] == 1 and before[33] == 4 and before[40] == 0
    assert before[9:13] == [0, 0, 255, 15]
    positions = sample(bus, sid)
    # User requested each currently observed pose, not a replay of the earlier snapshot.
    emit({'event': 'before', 'id': sid, 'registers': before, 'positions': positions,
          'change_since_snapshot': positions[-1] - expected})
    attempted = False
    try:
        attempted = True
        bus.unlock(sid)
        assert bus.read_u8(sid, 55) == 0
        check_off(bus, sid)
        ack = bus.calibrate_to(sid, 2048)
        # First read after calibration is torque. No target-position writes at all.
        check_off(bus, sid)
        emit({'event': 'calibration_ack', 'id': sid, 'ack': ack, 'torque': 0})
        readings = sample(bus, sid, 12)
        assert all(abs(p - 2048) <= 8 for p in readings), f'ID{sid} midpoint not verified: {readings}'
    finally:
        if attempted:
            # On failures force OFF, then lock. Never restore an ON state.
            if bus.read_u8(sid, 40) != 0:
                bus.write_u8(sid, 40, 0)
            assert bus.read_u8(sid, 40) == 0
            bus.lock(sid)
            assert bus.read_u8(sid, 55) == 1
            check_off(bus, sid)
    after = bus.dump(sid)['raw']
    # Firmware may update its own target when relabeling the position.
    stable = [i for i in range(56) if i not in (31, 32, 42, 43, 55)]
    assert all(before[i] == after[i] for i in stable), f'ID{sid} unrelated configuration changed'
    emit({'event': 'complete', 'id': sid, 'registers_after': after,
          'position': feetech.sign15(after[56] | after[57] << 8), 'torque': after[40],
          'goal_raw': after[42] | after[43] << 8, 'host_goal_writes': 0})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--backup', type=Path)
    p.add_argument('--ids', default='14')
    a = p.parse_args()
    ids = [int(i) for i in a.ids.split(',')]
    assert ids and len(set(ids)) == len(ids) and set(ids) <= set(IDS)
    bus = OriginBus('/dev/ttyS2', 1000000)
    try:
        if a.prepare:
            rows = {sid: {'registers': bus.dump(sid)['raw'], 'positions': sample(bus, sid)} for sid in IDS}
            backup = WEB / 'calib' / ('all-midpoint-' + time.strftime('%Y%m%d-%H%M%S'))
            backup.mkdir(parents=True, exist_ok=False)
            (backup/'before.json').write_text(json.dumps(rows, indent=2))
            config = Path.home()/'deploy_hd1910/robot.json'
            shutil.copy2(config, backup/'robot.json.before')
            shutil.copy2(WEB/'directions.json', backup/'directions.json.before')
            shutil.copytree(WEB/'poses', backup/'poses.before')
            print(json.dumps({'backup':str(backup),'positions':{sid:r['positions'][-1] for sid,r in rows.items()}}),flush=True)
            return
        assert a.backup and a.backup.is_dir(), 'Run --prepare first'
        before = json.loads((a.backup/'before.json').read_text())
        events_file = a.backup/'events.json'
        events = json.loads(events_file.read_text()) if events_file.exists() else []
        done = {e['id'] for e in events if e['event']=='complete'}
        assert not set(ids)&done, 'An ID already completed; do not recalibrate twice'
        config = Path.home()/'deploy_hd1910/robot.json'
        cfg = json.loads(config.read_text())
        for joint in cfg['joints']:
            joint['zero_tick'] = None
        cfg['calibration'] = None
        tmp = config.with_suffix('.json.tmp');tmp.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n');tmp.replace(config)
        # Preserve old raw poses but keep them out of the web playback list.
        archive = a.backup/'poses.archived';archive.mkdir(exist_ok=True)
        for pose in (WEB/'poses').glob('*.json'):
            if not json.loads(pose.read_text()).get('fake',False):shutil.move(str(pose),str(archive/pose.name))
        def emit(e):
            events.append(e);tmp=events_file.with_suffix('.tmp');tmp.write_text(json.dumps(events,indent=2));tmp.replace(events_file)
            print(json.dumps({k:v for k,v in e.items() if k not in ('registers','registers_after')}),flush=True)
        for sid in ids:
            try:
                recenter_one(bus,sid,before[str(sid)]['positions'][-1],emit)
            except Exception as exc:
                emit({'event':'failed','id':sid,'error':repr(exc)})
                raise
    finally:
        bus.close()


if __name__ == '__main__':
    main()
