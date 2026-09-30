"""Local Linux PTY tests. Never open a physical serial device or contact Radxa.

The peer checks literal requests and sends literal replies from the SCS manual;
it does not call the deployment packet builder to fabricate expected results.
PTYs exercise the local OS byte stream, not UART baud accuracy or HAT electronics.
"""
import importlib.util
import os
from pathlib import Path
import pty
import select
import sys
import termios
import threading
import time
import unittest
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy_hd1910"))
from microduck_deploy.servo import ServoBus, ServoTimeout, ServoProtocolError, build_packet
from probe_posix import open_port, exchange, request

spec = importlib.util.spec_from_file_location(
    "reference_feetech", ROOT / "microduck-replica/tools/servo-web/feetech.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)

# Protocol document sections 4.1 and 4.2, including its example value 1304.
PING = bytes.fromhex("ff ff 01 02 01 fb")
PING_REPLY = bytes.fromhex("ff ff 01 02 00 fc")
POSITION = bytes.fromhex("ff ff 01 04 02 38 02 be")
POSITION_REPLY = bytes.fromhex("ff ff 01 04 00 18 05 dd")


@contextmanager
def peer(exchanges):
    """exchanges = (literal TX, literal RX, response delay, chunk size)."""
    master, slave = pty.openpty()
    port = os.ttyname(slave)
    assert port.startswith("/dev/pts/")
    errors, received = [], []
    stop = threading.Event()

    def serve():
        try:
            for expected, response, delay, chunk_size in exchanges:
                incoming = bytearray()
                deadline = time.monotonic() + 2
                while len(incoming) < len(expected) and not stop.is_set():
                    if time.monotonic() > deadline:
                        raise AssertionError("No complete request on PTY")
                    if select.select([master], [], [], 0.01)[0]:
                        incoming.extend(os.read(master, len(expected) - len(incoming)))
                if stop.is_set():
                    return
                received.append(bytes(incoming))
                if bytes(incoming) != expected:
                    raise AssertionError(f"Wrong TX: {incoming.hex()} != {expected.hex()}")
                if stop.wait(delay):
                    return
                for offset in range(0, len(response), chunk_size):
                    if stop.is_set():
                        return
                    os.write(master, response[offset:offset + chunk_size])
                    if chunk_size < len(response) and stop.wait(0.001):
                        return
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield port, received
    finally:
        stop.set()
        worker.join(2)
        os.close(master)
        os.close(slave)
        if worker.is_alive():
            raise AssertionError("PTY peer did not terminate")
        if errors:
            raise errors[0]


class TransportTests(unittest.TestCase):
    def test_document_requests_and_reference_match_all_user_ids(self):
        self.assertEqual(build_packet(1, 1), PING)
        self.assertEqual(build_packet(1, 2, bytes([56, 2])), POSITION)
        for sid in range(1, 15):
            for instruction, params in [(1, b""), (2, bytes([56, 2])),
                                        (2, bytes([56, 15])), (2, bytes([0, 9]))]:
                with self.subTest(id=sid, instruction=instruction, params=params):
                    self.assertEqual(build_packet(sid, instruction, params),
                                     reference.FeetechBus._packet(sid, instruction, params))
            self.assertEqual(request(sid, "ping"), build_packet(sid, 1))
            self.assertEqual(request(sid, "position"), build_packet(sid, 2, bytes([56, 2])))

    def test_pyserial_settings_are_8n1_and_no_flow_control(self):
        with peer([]) as (port, _):
            with ServoBus(port, timeout=0.1) as bus:
                s = bus._serial
                self.assertEqual((s.baudrate, s.bytesize, s.parity, s.stopbits),
                                 (1_000_000, 8, "N", 1))
                self.assertFalse(s.xonxoff or s.rtscts or s.dsrdtr)
                attrs = termios.tcgetattr(s.fileno())
                self.assertTrue(attrs[2] & termios.CREAD)
                self.assertFalse(attrs[2] & termios.CRTSCTS)
                self.assertFalse(attrs[0] & (termios.IXON | termios.IXOFF))
                self.assertFalse(attrs[3] & (termios.ICANON | termios.ECHO))

    def check_driver(self, which, delay, fragment, timeout):
        cases = [(PING, PING_REPLY, delay, fragment),
                 (POSITION, POSITION_REPLY, delay, fragment)]
        with peer(cases) as (port, received):
            if which == "deploy":
                with ServoBus(port, timeout=timeout) as bus:
                    self.assertIs(bus.ping(1), True)
                    self.assertEqual(bus.read(1, 56, 2), bytes.fromhex("18 05"))
            else:
                bus = reference.FeetechBus(port, timeout=timeout)
                try:
                    self.assertEqual(bus.ping(1), 0)
                    self.assertEqual(bus.read(1, 56, 2), (0, bytes.fromhex("18 05")))
                finally:
                    bus.close()
            self.assertEqual(received, [PING, POSITION])

    def test_deploy_default_timeout_literal_replies(self):
        self.check_driver("deploy", delay=0.002, fragment=100, timeout=0.015)

    def test_reference_default_timeout_literal_replies(self):
        self.check_driver("reference", delay=0.002, fragment=100, timeout=0.015)

    def test_deploy_delayed_bytewise_replies(self):
        self.check_driver("deploy", delay=0.025, fragment=1, timeout=0.2)

    def test_reference_delayed_bytewise_replies(self):
        self.check_driver("reference", delay=0.025, fragment=1, timeout=0.2)

    def test_deploy_does_not_report_missing_reply_as_success(self):
        with peer([(PING, b"", 0, 1)]) as (port, received):
            with ServoBus(port, timeout=0.03) as bus:
                with self.assertRaises(ServoTimeout):
                    bus.ping(1)
            self.assertEqual(received, [PING])

    def test_reference_reports_missing_reply_as_none(self):
        with peer([(PING, b"", 0, 1)]) as (port, received):
            bus = reference.FeetechBus(port, timeout=0.03)
            try:
                self.assertIsNone(bus.ping(1))
            finally:
                bus.close()
            self.assertEqual(received, [PING])

    def test_deploy_rejects_corrupt_reply(self):
        corrupt = bytes.fromhex("ff ff 01 02 00 fd")
        with peer([(PING, corrupt, 0, 100)]) as (port, _):
            with ServoBus(port, timeout=0.1) as bus:
                with self.assertRaises(ServoProtocolError):
                    bus.ping(1)

    def test_deploy_explicit_echo_setting(self):
        with peer([(PING, PING + PING_REPLY, 0, 1)]) as (port, _):
            with ServoBus(port, timeout=0.2, discard_echo=True) as bus:
                self.assertIs(bus.ping(1), True)

    def test_posix_without_pyserial_literal_replies(self):
        with peer([(PING, PING_REPLY, 0.005, 1),
                   (POSITION, POSITION_REPLY, 0.005, 1)]) as (port, received):
            with open_port(port, 1_000_000) as fd:
                ping = exchange(fd, 1, "ping", 0.1)
                position = exchange(fd, 1, "position", 0.1)
            self.assertEqual(ping["rx_hex"], PING_REPLY.hex())
            self.assertEqual(position["rx_hex"], POSITION_REPLY.hex())
            self.assertEqual(received, [PING, POSITION])

    def test_posix_keeps_echo_and_noise_for_diagnosis(self):
        raw = b"noise" + PING + PING_REPLY
        with peer([(PING, raw, 0.005, 1)]) as (port, _):
            with open_port(port, 1_000_000) as fd:
                result = exchange(fd, 1, "ping", 0.1)
            self.assertEqual(result["rx_hex"], raw.hex())

    def test_probe_cannot_make_broadcast_or_motion_commands(self):
        for sid, operation in [(254, "ping"), (255, "ping"),
                               (1, "write"), (1, "torque"), (1, "position_goal")]:
            with self.subTest(id=sid, operation=operation), self.assertRaises(ValueError):
                request(sid, operation)


if __name__ == "__main__":
    unittest.main()
