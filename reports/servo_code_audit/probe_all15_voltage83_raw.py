"""Read-only FT-SCS probe using Linux termios, without PySerial or deploy code.

Only unicast PING and READ of position register 56 are generated. It does not
enable torque, send goals, change servo baud/IDs, or write any servo register.
Raw RX is kept, including echoes/noise; a received byte is not called success.
This is a diagnostic with long receive windows, not a real-time controller.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import select
import termios
import time
from contextlib import contextmanager


def request(sid: int, operation: str) -> bytes:
    if type(sid) is not int or not 0 <= sid <= 253:
        raise ValueError("Only individual servo IDs 0..253 are allowed")
    if operation == "ping":
        payload = [sid, 2, 1]
    elif operation == "position":
        payload = [sid, 4, 2, 0x38, 2]
    elif operation == "feedback":
        payload = [sid, 4, 2, 0x38, 15]
    else:
        raise ValueError("Only ping, position and feedback reads are allowed")
    return bytes([255, 255, *payload, 255 - (sum(payload) % 256)])


@contextmanager
def open_port(path: str, baudrate: int):
    speed = getattr(termios, "B" + str(baudrate), None)
    if speed is None:
        raise ValueError("Unsupported termios baudrate")
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attrs = termios.tcgetattr(fd)
        # Raw 8N1, receiver enabled, no software or hardware flow control.
        attrs[0], attrs[1], attrs[3] = 0, 0, 0
        attrs[2] = termios.CLOCAL | termios.CREAD | termios.CS8
        attrs[4] = attrs[5] = speed
        attrs[6][termios.VMIN] = attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
        yield fd
    finally:
        os.close(fd)


def exchange(fd: int, sid: int, operation: str, timeout: float = 0.1) -> dict:
    if not math.isfinite(timeout) or not 0 < timeout <= 2:
        raise ValueError("timeout must be in (0, 2] seconds")
    packet = request(sid, operation)
    started = time.monotonic()
    deadline = started + timeout
    sent = 0
    while sent < len(packet):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError("Timed out sending diagnostic request")
        try:
            sent += os.write(fd, packet[sent:])
        except BlockingIOError:
            continue
    # Do not use tcdrain(), which has no Python-level deadline on real UARTs.
    raw = bytearray()
    first_byte_ms = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            break
        try:
            chunk = os.read(fd, 1024)
        except BlockingIOError:
            continue
        if not chunk:
            break
        if first_byte_ms is None:
            first_byte_ms = round((time.monotonic() - started) * 1000, 3)
        raw.extend(chunk)
        if len(raw) >= 8192:
            raise RuntimeError("Unexpected continuous RX; stop and check port ownership")
    return {"id": sid, "operation": operation, "tx_hex": packet.hex(),
            "rx_hex": raw.hex(), "first_byte_ms": first_byte_ms,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--ids", type=int, nargs="+", default=list(range(1, 15)))
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--timeout", type=float, default=0.1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 10 or len(args.ids) != len(set(args.ids)):
        parser.error("repeat must be 1..10 and IDs must not be duplicated")
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 2:
        parser.error("timeout must be in (0, 2] seconds")
    for sid in args.ids:
        request(sid, "ping")  # Validate before opening any port.
    with open_port(args.port, args.baudrate) as fd:
        time.sleep(0.2)
        for sid in args.ids:
            for operation in ("ping", "position"):
                for _ in range(args.repeat):
                    print(json.dumps(exchange(fd, sid, operation, args.timeout)), flush=True)
                    time.sleep(0.05)



import struct

def counters(fd):
    return dict(zip(['cts','dsr','rng','dcd','rx','tx','frame','overrun','parity','brk','buf_overrun'],struct.unpack('20i',fcntl.ioctl(fd,0x545D,bytes(80)))))
result={'read_only':True,'user_reported_supply_v':8.3,'rows':[]}
with open_port('/dev/ttyS2',1000000) as fd:
    modem=struct.unpack('i',fcntl.ioctl(fd,termios.TIOCMGET,bytes(4)))[0]
    if modem & 0x8000:raise RuntimeError('Internal loopback unexpectedly enabled')
    result['before']=counters(fd)
    for sid in range(1,16):
        row=exchange(fd,sid,'feedback',0.4)
        raw=bytes.fromhex(row['rx_hex'])
        valid=len(raw)==21 and raw[:4]==bytes([255,255,sid,17]) and sum(raw[2:])%256==255
        row['valid_frame']=valid
        if valid:
            row['reply_status']=raw[4]
            row['voltage_v']=raw[11]/10
            row['temperature_c']=raw[12]
            row['feedback_status']=raw[14]
        result['rows'].append(row)
        time.sleep(0.1)
    result['after']=counters(fd)
print(json.dumps(result,indent=2))
