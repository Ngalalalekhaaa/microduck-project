"""Continuously scan Feetech IDs; only PING and READ packets are sent."""
import dataclasses
import fcntl
import json
import os
import signal
import struct
import termios
import time

from microduck_deploy.servo import ServoBus, ServoTimeout


running = True


def stop(_signal, _frame):
    global running
    running = False


def emit(event):
    print(json.dumps(event, ensure_ascii=False), flush=True)


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(sig, stop)

config = json.load(open('robot.json'))['serial']
config['timeout'] = 0.015
started = time.monotonic()
cycle = 0
with ServoBus(**config) as bus:
    modem = struct.unpack('i', fcntl.ioctl(
        bus._serial.fileno(), termios.TIOCMGET, bytes(4)))[0]
    if modem & 0x8000:
        raise RuntimeError('Internal UART loopback must be disabled')
    emit({'event': 'started', 'pid': os.getpid(), 'read_only': True,
          'scan_range': [0, 253], 'config': config})
    while running:
        cycle += 1
        found = []
        errors = []
        feedback = {}
        for sid in range(254):
            if not running:
                break
            try:
                bus.ping(sid)
                found.append(sid)
            except ServoTimeout:
                pass
            except Exception as exc:
                errors.append({'id': sid, 'type': type(exc).__name__,
                               'message': str(exc)})
            time.sleep(0.003)
        for sid in found:
            if not running:
                break
            try:
                feedback[sid] = dataclasses.asdict(bus.read_feedback(sid))
            except Exception as exc:
                feedback[sid] = {'error': str(exc)}
        emit({'event': 'scan', 'cycle': cycle,
              'elapsed_s': round(time.monotonic() - started, 1),
              'complete': running, 'found_ids': found,
              'feedback': feedback, 'errors': errors})
        for _ in range(10):
            if not running:
                break
            time.sleep(0.1)
emit({'event': 'stopped', 'cycles': cycle})
