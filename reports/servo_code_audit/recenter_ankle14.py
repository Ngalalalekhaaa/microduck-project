"""One-shot, torque-off origin adjustment for the observed HD-1910 ID 14.

No movement/torque/phase/mode/limit writes. Default invocation is read-only.
Requires the servo-web backend to be stopped before opening the UART.
"""
import argparse
import json
import shutil
import struct
import sys
import threading
import time
from pathlib import Path

WEB = Path.home() / 'servo_web_hd1910'
sys.path.insert(0, str(WEB))
import feetech


class CheckedBus(feetech.FeetechBus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each permitted write rechecks torque using a nested READ transaction.
        self._io = threading.RLock()

    def _send(self, sid, instr, params=b''):
        assert sid == 14, 'Only ID 14 is allowed'
        if instr != feetech.READ:
            allowed = ((instr == feetech.WRITE and
                        ((params[0] == 55 and bytes(params[1:]) in (b'\x00', b'\x01')) or
                         (params[0] == 42 and len(params) == 3 and 0 <= int.from_bytes(params[1:], 'little') <= 4095))) or
                       (instr == feetech.CALIBRATE and bytes(params) == b'\x00\x08') or
                       (instr == feetech.RESET_TURNS and not params))
            assert allowed, f'Forbidden packet {instr} {params!r}'
            assert self.read_u8(14, 40) == 0, 'Torque must remain off'
        super()._send(sid, instr, params)


def capture(bus):
    raw = bus.dump(14)['raw']
    assert raw[0:2] == [3, 46] and raw[2] == 0 and raw[3:6] == [10, 31, 14], 'Unexpected firmware/model/ID'
    assert raw[30] == 1 and raw[33] == 4 and raw[40] == 0, 'Unexpected resolution/mode/torque'
    assert raw[9:13] == [0, 0, 255, 15], 'Unexpected limits'
    samples = []
    for _ in range(12):
        state = bus.state(14)
        assert state['err'] == 0 and state['status'] == 0, 'Servo reports an error'
        samples.append(state['pos'])
        time.sleep(.025)
    assert max(samples) - min(samples) <= 8, 'Reference pose is not stationary'
    assert all(abs(p - 4353) <= 20 for p in samples), f'Reference pose changed: {samples}'
    return {'registers': raw, 'positions': samples}


def adjust(bus, before, emit):
    """Use the documented calibration command; do not guess an offset encoding."""
    touched = False
    try:
        assert bus.read_u8(14, 40) == 0
        assert abs(bus.state(14)['pos'] - before['positions'][-1]) <= 8
        touched = True
        bus.unlock(14)
        assert bus.read_u8(14, 55) == 0, 'Unlock did not take effect'
        ack = bus.calibrate_to(14, 2048)
        emit({'event': 'calibration_sent', 'ack': ack})
        samples = []
        for _ in range(12):
            time.sleep(.04)
            state = bus.state(14)
            assert state['err'] == 0 and state['status'] == 0
            samples.append(state['pos'])
        emit({'event': 'calibration_readback', 'positions': samples,
              'offset_raw': bus.read_u16(14, 31)})
        pos = samples[-1]
        # Some firmware retains accumulated turns across an origin adjustment.
        # Clear only if the within-turn position already matches the requested midpoint.
        if not 0 <= pos <= 4095 and abs((pos % 4096) - 2048) <= 8:
            with bus._io:
                bus._send(14, feetech.RESET_TURNS)
                try:
                    bus._read_status(14)
                except feetech.BusTimeout:
                    pass
            time.sleep(.15)
            pos = bus.state(14)['pos']
            emit({'event': 'cleared_accumulated_turns', 'position': pos})
        assert abs(pos - 2048) <= 8, f'Midpoint verification failed: {pos}'
        assert max(samples[-4:]) - min(samples[-4:]) <= 8, 'Pose moved during adjustment'
        bus.write_u16(14, 42, pos)
        assert bus.read_u16(14, 42) == pos, 'Target alignment failed'
    finally:
        if touched:
            bus.lock(14)
            assert bus.read_u8(14, 55) == 1, 'Lock verification failed'
            assert bus.read_u8(14, 40) == 0, 'Torque verification failed'
    after = bus.dump(14)['raw']
    unchanged = [i for i in range(56) if i not in (31, 32, 42, 43, 55)]
    assert all(after[i] == before['registers'][i] for i in unchanged), 'Unrelated register changed'
    emit({'event': 'complete', 'registers_after': after,
          'position': after[56] | after[57] << 8, 'torque': after[40]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.apply:
        parser.error("Disabled after observed torque=1 during the hardware trial; do not repeat without resolving the firmware behavior.")
    bus = CheckedBus('/dev/ttyS2', 1000000)
    try:
        before = capture(bus)
        print(json.dumps({'plan': 'ID14 current physical pose -> 2048; keep torque off', **before}), flush=True)
        if not args.apply:
            return
        backup = WEB / 'calib' / ('id14-origin-' + time.strftime('%Y%m%d-%H%M%S'))
        backup.mkdir(parents=True, exist_ok=False)
        (backup / 'before.json').write_text(json.dumps(before, indent=2))
        config = Path.home() / 'deploy_hd1910/robot.json'
        shutil.copy2(config, backup / 'robot.json.before')
        cfg = json.loads(config.read_text())
        # An origin change invalidates previous model-angle calibration.
        joint = next(j for j in cfg['joints'] if j['id'] == 14)
        assert joint['name'] == 'right_ankle'
        joint['zero_tick'] = None
        cfg['calibration'] = None
        temp = config.with_suffix('.json.tmp')
        temp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
        temp.replace(config)
        (backup / 'poses').mkdir()
        for pose in (WEB / 'poses').glob('*.json'):
            doc = json.loads(pose.read_text())
            if not doc.get('fake', False):
                shutil.move(str(pose), str(backup / 'poses' / pose.name))
        events = []
        def emit(event):
            events.append(event)
            (backup / 'events.json').write_text(json.dumps(events, indent=2))
            print(json.dumps(event), flush=True)
        print('BACKUP=' + str(backup), flush=True)
        try:
            adjust(bus, before, emit)
        except Exception as exc:
            emit({'event': 'failed', 'error': repr(exc)})
            raise
    finally:
        bus.close()


if __name__ == '__main__':
    main()
