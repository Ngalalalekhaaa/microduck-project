"""Bounded UART BREAK for a manual DATA-to-GND voltage measurement.

No serial bytes, servo packets, or servo-register writes are generated.
Use only when the user has their voltmeter in place. On the inspected HAT,
TX low enables its hardware TTL driver. Electrical DATA voltage must be
measured externally; an accepted ioctl does not prove the physical level.
"""
import argparse
import fcntl
import json
import math
import os
import signal
import struct
import termios
import time

TIOCSBRK = getattr(termios, "TIOCSBRK", 0x5427)
TIOCCBRK = getattr(termios, "TIOCCBRK", 0x5428)


def emit(event):
    print(json.dumps(event), flush=True)


def hold_low(fd, seconds, *, ioctl=fcntl.ioctl, sleep=time.sleep, report=emit):
    if not math.isfinite(seconds) or not 1 <= seconds <= 30:
        raise ValueError("Hold duration must be 1..30 seconds")
    try:
        ioctl(fd, TIOCSBRK)
        report({"event": "break_enabled", "duration_s": seconds,
                "voltage_requires_meter": True})
        sleep(seconds)
    finally:
        ioctl(fd, TIOCCBRK)
        report({"event": "break_disabled"})


def stop(signum, frame):
    raise KeyboardInterrupt("Interrupted; releasing UART BREAK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hold-low", action="store_true")
    parser.add_argument("--seconds", type=float, default=30)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 30:
        parser.error("seconds must be 1..30")
    if not args.hold_low:
        emit({"event": "plan_only", "port": "/dev/ttyS2",
              "duration_s": args.seconds, "serial_bytes_written": 0})
        return
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    fd = os.open("/dev/ttyS2", os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        modem = struct.unpack("i", fcntl.ioctl(fd, termios.TIOCMGET, bytes(4)))[0]
        if modem & 0x8000:
            raise RuntimeError("UART internal loopback must be disabled")
        attrs = termios.tcgetattr(fd)
        if attrs[4:6] != [termios.B1000000, termios.B1000000]:
            raise RuntimeError("Unexpected UART speed; inspect configuration first")
        if attrs[2] & termios.CSIZE != termios.CS8 or attrs[2] & (termios.PARENB | termios.CSTOPB):
            raise RuntimeError("Expected 8N1 UART configuration")
        pending = struct.unpack("i", fcntl.ioctl(fd, termios.TIOCOUTQ, bytes(4)))[0]
        if pending:
            raise RuntimeError("UART has pending TX; stop other tests first")
        hold_low(fd, args.seconds)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
