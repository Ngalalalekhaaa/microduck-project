"""Read-only Xbox diagnostics: no servo or IMU access."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from microduck_deploy.gamepad import GamepadReader, axes_to_twist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--seconds', type=float, default=30)
    args = p.parse_args()
    if not 0 < args.seconds <= 600:
        p.error('--seconds must be in (0,600]')
    if args.list:
        from evdev import InputDevice, list_devices
        for node in list_devices():
            device = InputDevice(node)
            try:
                print(json.dumps({'path': node, 'name': device.name, 'uniq': device.uniq,
                                  'capabilities': device.capabilities()},
                                 ensure_ascii=False, default=str))
            finally:
                device.close()
        return
    reader = GamepadReader(args.device)
    print(json.dumps(reader.metadata, ensure_ascii=False))
    print('只读测试：摇杆和十字键↑↓←→；方向按钮规范为544/545/546/547。'
          'B键或Ctrl+C退出；不发送舵机动作。', flush=True)
    try:
        started = time.monotonic()
        while time.monotonic() - started < args.seconds:
            sample = reader.latest()
            out = dataclasses.asdict(sample)
            out['dpad_pressed'] = [name for code, name in
                                   ((544, '↑'), (545, '↓'), (546, '←'), (547, '→'))
                                   if code in sample.buttons]
            out['command_vx_vy_wz'] = axes_to_twist(sample.axes) if sample.connected else (0, 0, 0)
            print(json.dumps(out, ensure_ascii=False), flush=True)
            if sample.stop_requested or not sample.connected:
                break
            time.sleep(.1)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == '__main__':
    main()
