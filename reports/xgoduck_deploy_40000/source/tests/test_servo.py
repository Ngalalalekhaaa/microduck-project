"""FT-SCS wire and register tests; no serial device or GPU is opened.

SPDX-License-Identifier: Apache-2.0
"""

import math
import struct
import unittest

from microduck_deploy.servo import (
    PING, READ, SYNC_READ, SYNC_WRITE, BROADCAST,
    ServoBus, ServoError, ServoTimeout, ServoProtocolError, ServoStatusError,
    ServoConfigurationError, build_packet, decode_sign_magnitude,
    ticks_to_radians, radians_to_ticks, velocity_to_rad_s,
)


def status(sid, params=b"", error=0):
    return build_packet(sid, error, params)


def feedback(position=2048, velocity=0, load=0, device_status=0):
    return (struct.pack("<HHH", position, velocity, load) +
            bytes((74, 31, 0, device_status, 1)) + struct.pack("<HH", 2100, 0x8002))


class MemorySerial:
    """Serial-shaped memory table; supports fragmented responses and wire faults."""
    timeout = 0.001

    def __init__(self, ids=(1, 2), *, fragment=1, echo=False):
        self.memory = {}
        self.fragment, self.echo = fragment, echo
        self.pending = bytearray()
        self.writes = []
        self.closed = False
        self.reply_filter = lambda reply: reply
        self.ignore_writes = False
        self.short_write = False
        for sid in ids:
            memory = bytearray(87)
            memory[:9] = bytes((3, 46, 0, 19, 10, sid, 0, sid, 1))
            memory[9:13] = struct.pack("<HH", 0, 4095)
            memory[18] = 4
            memory[21:24] = bytes((6, 20, 0))
            memory[30] = 1
            memory[33] = 4
            memory[50:53] = bytes((6, 20, 0))
            memory[55] = 1
            memory[56:71] = feedback()
            self.memory[sid] = memory

    def reset_input_buffer(self):
        self.pending.clear()

    def write(self, packet):
        self.writes.append(packet)
        if self.short_write:
            return len(packet) - 1
        assert packet[:2] == b"\xff\xff" and sum(packet[2:]) & 255 == 255
        assert packet[3] == len(packet) - 4
        sid, instruction, params = packet[2], packet[4], packet[5:-1]
        reply = b""
        if instruction == PING:
            reply = status(sid)
        elif instruction == READ:
            address, count = params
            reply = status(sid, bytes(self.memory[sid][address:address + count]))
        elif instruction == SYNC_READ:
            address, count = params[:2]
            reply = b"".join(status(i, bytes(self.memory[i][address:address + count])) for i in params[2:])
        elif instruction == SYNC_WRITE:
            assert sid == BROADCAST
            address, size = params[:2]
            for start in range(2, len(params), size + 1):
                target = params[start]
                if not self.ignore_writes:
                    self.memory[target][address:address + size] = params[start + 1:start + 1 + size]
        else:
            raise AssertionError(f"Forbidden instruction {instruction}")
        self.pending.extend((packet if self.echo else b"") + self.reply_filter(reply))
        return len(packet)

    def read(self, count):
        count = min(count, self.fragment)
        result = bytes(self.pending[:count])
        del self.pending[:count]
        return result

    def close(self):
        self.closed = True


class ServoTests(unittest.TestCase):
    def make_bus(self, **kwargs):
        transport = MemorySerial()
        bus = ServoBus(None, transport=transport, timeout=0.002, **kwargs)
        self.addCleanup(bus.close)
        return bus, transport

    def test_known_protocol_packets_and_fragmented_reads(self):
        self.assertEqual(build_packet(1, PING), bytes.fromhex("ff ff 01 02 01 fb"))
        self.assertEqual(build_packet(1, READ, bytes((56, 15))), bytes.fromhex("ff ff 01 04 02 38 0f b1"))
        bus, serial = self.make_bus()
        serial.reply_filter = lambda reply: b"noise" + reply
        self.assertTrue(bus.ping(1))
        self.assertEqual(bus.read_identity(1)["firmware"], [3, 46])
        self.assertEqual(bus.read_identity(1)["model_version"], [19, 10])

    def test_context_only_closes_transport(self):
        serial = MemorySerial()
        with ServoBus(None, transport=serial) as bus:
            self.assertFalse(serial.closed)
        self.assertTrue(serial.closed)
        self.assertEqual(serial.writes, [])
        with self.assertRaises(ServoError):
            bus.ping(1)

    def test_sign_magnitude_and_feedback(self):
        self.assertEqual(decode_sign_magnitude(0x8005), -5)
        self.assertEqual(decode_sign_magnitude(0x8000), 0)
        self.assertEqual(decode_sign_magnitude(0x403, 10), -3)
        bus, serial = self.make_bus()
        serial.memory[1][56:71] = feedback(0x8005, 0x800A, 0x403)
        state = bus.read_feedback(1)
        self.assertEqual((state.position_ticks, state.velocity_raw, state.load_raw), (-5, -10, -3))
        self.assertEqual((state.voltage_v, state.temperature_c, state.current_ma), (7.4, 31, -13.0))
        self.assertEqual(state.goal_ticks, 2100)
        self.assertGreater(state.received_at, 0)
        self.assertAlmostEqual(state.velocity_rad_s(0.732), -10 * 0.732 * math.tau / 60)

    def test_read_configuration_preserves_raw_profile_information(self):
        bus, serial = self.make_bus()
        config = bus.read_configuration(1)
        self.assertEqual(config["ram_gains"], {"p": 6, "d": 20, "i": 0})
        self.assertEqual((config["mode"], config["angle_resolution"], config["phase"]), (4, 1, 4))
        self.assertEqual(config["velocity_rpm_per_unit_nominal"], 0.732)
        self.assertFalse(config["velocity_unit_confirmed"])
        bus.register_profile = "sts"
        config = bus.read_configuration(1)
        self.assertIsNone(config["ram_gains"])
        self.assertEqual(config["velocity_rpm_per_unit_nominal"], 0.0146)
        serial.memory[1][2] = 1
        with self.assertRaises(ServoConfigurationError):
            bus.read_identity(1)

    def test_sync_positions_exact_bytes_and_no_ack_wait(self):
        bus, serial = self.make_bus()
        bus.sync_positions({1: 2048, 2: 1024})
        expected_params = bytes.fromhex("2a 02 01 00 08 02 00 04")
        self.assertEqual(serial.writes, [build_packet(254, 0x83, expected_params)])
        self.assertEqual(serial.memory[1][42:44], bytes((0, 8)))

    def test_motion_profile_reads_real_values_without_writes_or_unit_assumptions(self):
        bus, serial = self.make_bus()
        values = bytes((23,)) + struct.pack("<HHHH", 3120, 0x8007, 0x8064, 875)
        serial.memory[1][41:50] = values
        profile = bus.read_motion_profile(1)
        self.assertEqual(profile["acceleration_raw"], 23)
        self.assertEqual(profile["goal_position_ticks"], 3120)
        self.assertEqual(profile["goal_current_raw"], -7)
        self.assertEqual(profile["register_44_raw"], 0x8007)
        self.assertEqual(profile["goal_speed_raw"], -100)
        self.assertEqual(profile["torque_limit_raw"], 875)
        self.assertEqual(profile["registers_41_49_hex"], values.hex())
        self.assertFalse(profile["motion_semantics_confirmed"])
        self.assertEqual(serial.writes, [build_packet(1, READ, bytes((41, 9)))])
        self.assertEqual(bus.read_configuration(1)["motion_profile"], profile)
        self.assertTrue(all(packet[4] == READ for packet in serial.writes))
        bus.register_profile = "sts"
        self.assertIsNone(bus.read_motion_profile(1)["goal_current_raw"])

    def test_sync_feedback_handles_reordered_frames_and_requires_all_ids(self):
        bus, serial = self.make_bus()
        serial.reply_filter = lambda _: status(2, feedback()) + status(1, feedback())
        self.assertEqual(list(bus.read_feedback_many([1, 2], sync=True)), [1, 2])
        self.assertEqual(serial.writes[-1], build_packet(254, 0x82, bytes((56, 15, 1, 2))))
        serial.reply_filter = lambda _: status(1, feedback())
        with self.assertRaises(ServoTimeout):
            bus.read_feedback_many([1, 2], sync=True)

    def test_sync_feedback_rejects_duplicate_and_unrequested_ids(self):
        bus, serial = self.make_bus()
        for other in (1, 3):
            serial.reply_filter = lambda _, other=other: status(1, feedback()) + status(other, feedback())
            with self.assertRaises(ServoProtocolError):
                bus.read_feedback_many([1, 2], sync=True)

    def test_unicast_feedback_fallback(self):
        bus, serial = self.make_bus()
        self.assertEqual(set(bus.read_feedback_many([1, 2])), {1, 2})
        self.assertEqual([p[4] for p in serial.writes], [READ, READ])

    def test_torque_both_reply_levels_use_readback(self):
        bus, serial = self.make_bus()
        serial.memory[1][8] = 0
        bus.set_torque([1, 2], True)
        self.assertEqual([serial.memory[i][40] for i in (1, 2)], [1, 1])
        bus.set_torque([1, 2], False)
        self.assertEqual([serial.memory[i][40] for i in (1, 2)], [0, 0])
        serial.ignore_writes = True
        with self.assertRaises(ServoConfigurationError):
            bus.set_torque([1], True)

    def test_pid_ram_can_zero_and_restore_i_without_eeprom(self):
        bus, serial = self.make_bus()
        serial.memory[1][52] = 7
        before = bytes(serial.memory[1][:40])
        bus.set_pid_ram([1], 5, 0, 0)
        self.assertEqual(serial.memory[1][50:53], bytes((5, 0, 0)))
        bus.set_pid_ram([1], 6, 20, 7)
        self.assertEqual(serial.memory[1][50:53], bytes((6, 20, 7)))
        self.assertEqual(bytes(serial.memory[1][:40]), before)
        writes = [p for p in serial.writes if p[4] == SYNC_WRITE]
        self.assertTrue(all(p[5:7] == bytes((50, 3)) for p in writes))

    def test_pd_ram_leaves_i_unchanged_and_rejects_nonzero_i(self):
        bus, serial = self.make_bus()
        bus.set_pd_ram([1], 5, 0)
        self.assertEqual(serial.memory[1][50:53], bytes((5, 0, 0)))
        serial.memory[1][52] = 1
        with self.assertRaises(ServoConfigurationError):
            bus.set_pd_ram([1], 5, 0)

    def test_gain_prechecks_all_ids_before_any_write(self):
        for address, value in ((40, 1), (33, 0), (30, 2)):
            bus, serial = self.make_bus()
            serial.memory[2][address] = value
            with self.assertRaises(ServoConfigurationError):
                bus.set_pid_ram([1, 2], 5, 0, 0)
            self.assertFalse(any(p[4] == SYNC_WRITE for p in serial.writes))

    def test_ram_gains_profile_and_readback_fail_closed(self):
        bus, serial = self.make_bus(register_profile="sts")
        with self.assertRaises(ServoConfigurationError):
            bus.set_pid_ram([1], 5, 0, 0)
        self.assertEqual(serial.writes, [])
        bus.register_profile = "hls_2"
        serial.ignore_writes = True
        with self.assertRaises(ServoConfigurationError):
            bus.set_pid_ram([1], 5, 0, 0)

    def test_checksum_wrong_id_invalid_length_and_truncation(self):
        bus, serial = self.make_bus()
        malformed = (
            lambda p: p[:-1] + bytes((p[-1] ^ 1,)),
            lambda _: status(2),
            lambda _: b"\xff\xff\x01\x01\x00",
            lambda _: status(1, b"extra"),
        )
        for corrupt in malformed:
            serial.reply_filter = corrupt
            with self.assertRaises(ServoProtocolError):
                bus.ping(1)
        for truncate in (lambda _: b"", lambda p: p[:-1], lambda _: b"\xff"):
            serial.reply_filter = truncate
            with self.assertRaises(ServoTimeout):
                bus.ping(1)

    def test_packet_and_feedback_device_errors_are_not_swallowed(self):
        bus, serial = self.make_bus()
        serial.reply_filter = lambda _: status(1, error=4)
        with self.assertRaises(ServoStatusError) as result:
            bus.read_feedback(1)
        self.assertEqual(result.exception.status, 4)
        serial.reply_filter = lambda _: status(1, feedback(device_status=8))
        with self.assertRaises(ServoStatusError) as result:
            bus.read_feedback(1)
        self.assertEqual(result.exception.source, "feedback")

    def test_explicit_echo_validation_and_short_write(self):
        serial = MemorySerial(echo=True)
        with ServoBus(None, transport=serial, discard_echo=True) as bus:
            self.assertTrue(bus.ping(1))
            bus.sync_positions({1: 2048})
            serial.echo = False
            with self.assertRaises(ServoProtocolError):
                bus.ping(1)
        bus, serial = self.make_bus()
        serial.short_write = True
        with self.assertRaises(ServoProtocolError):
            bus.ping(1)

    def test_invalid_parameters_send_nothing(self):
        bus, serial = self.make_bus()
        actions = [lambda: bus.ping(254), lambda: bus.ping(True),
                   lambda: bus.sync_positions({1: -1}), lambda: bus.sync_positions({1: 4096}),
                   lambda: bus.sync_positions({1: 0.2}), lambda: bus.sync_positions({}),
                   lambda: bus.set_torque([1, 1], False), lambda: bus.set_torque([1], 128),
                   lambda: bus.set_pid_ram([1], 255, 0, 0), lambda: bus.read(1, 86, 2),
                   lambda: bus._sync_ram(5, 1, {1: b"\x02"})]
        for action in actions:
            with self.assertRaises((ValueError, ServoConfigurationError)):
                action()
        self.assertEqual(serial.writes, [])
        with self.assertRaises(ValueError):
            build_packet(1, READ, bytes(254))

    def test_software_zero_direction_and_explicit_velocity_units(self):
        self.assertAlmostEqual(ticks_to_radians(3072, 2048, 1), math.pi / 2)
        self.assertAlmostEqual(ticks_to_radians(3072, 2048, -1), -math.pi / 2)
        self.assertEqual(radians_to_ticks(math.pi / 2, 2048, -1), 1024)
        self.assertEqual(radians_to_ticks(0, 1234.4, 1), 1234)
        self.assertAlmostEqual(velocity_to_rad_s(50, 0.0146), 50 * 0.0146 * math.tau / 60)
        for angle, zero, direction in ((math.nan, 2048, 1), (5, 2048, 1), (0, 2048, 0)):
            with self.assertRaises(ValueError):
                radians_to_ticks(angle, zero, direction)


if __name__ == "__main__":
    unittest.main()
