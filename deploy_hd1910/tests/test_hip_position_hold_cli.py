"""CLI and raw-packet contracts for the static hip hold helper; no hardware."""
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import hip_position_hold as hold
from microduck_deploy.config import default_config, record_reference
from microduck_deploy.policy import STRAIGHT_REFERENCE
from microduck_deploy.servo import ServoBus, build_packet


def config():
    cfg = default_config()
    for sid, joint in zip(list(range(1, 10)) + list(range(11, 16)), cfg['joints']):
        joint.update(id=sid, mapping_verified=True, direction=1 if sid == 15 else -1)
    record_reference(cfg, np.full(14, 2048), 'straight', STRAIGHT_REFERENCE)
    cfg['imu_mount_verified'] = True
    return cfg


def sync_packet(address, payload, ids=(2, 3, 12, 13)):
    params = bytearray((address, len(payload)))
    for sid in ids:
        params.append(sid)
        params.extend(payload)
    return build_packet(254, 131, params)


class PacketTests(unittest.TestCase):
    def bus(self, moving):
        bus = object.__new__(hold.HipHoldBus)
        bus.allow_motion = moving
        return bus

    def test_default_allows_reads_but_no_writes(self):
        bus = self.bus(False)
        with patch.object(ServoBus, '_send', return_value=123.) as send:
            self.assertEqual(bus._send(build_packet(3, 2, bytes((50, 3)))), 123.)
            self.assertEqual(bus._send(build_packet(254, 130, bytes((56, 15, 2, 3, 12, 13)))), 123.)
            for packet in (sync_packet(40, b'\x00'), sync_packet(40, b'\x01'),
                           sync_packet(42, b'\x00\x08'), sync_packet(50, bytes((16, 0, 0)))):
                with self.subTest(packet=packet), self.assertRaises(ValueError):
                    bus._send(packet)
            self.assertEqual(send.call_count, 2)

    def test_only_selected_hips_torque_position_and_listed_pid_reach_transport(self):
        bus = self.bus(True)
        allowed = [sync_packet(40, bytes((enabled,)), ids=(sid,))
                   for sid in hold.HIP_IDS for enabled in (0, 1)]
        allowed += [sync_packet(42, b'\x00\x08')]
        allowed += [sync_packet(50, bytes((p, 0, 0))) for p in (5, 8, 12, 16)]
        with patch.object(ServoBus, '_send', return_value=123.) as send:
            for packet in allowed:
                with self.subTest(packet=packet):
                    bus._send(packet)
            self.assertEqual(send.call_count, len(allowed))

    def test_current_eeprom_other_joints_and_unlisted_gains_never_reach_transport(self):
        bus = self.bus(True)
        denied = [sync_packet(44, b'\x00\x00'), sync_packet(46, b'\x00\x00'),
                  sync_packet(48, b'\xff\x03'), sync_packet(31, b'\x00\x08'),
                  sync_packet(21, bytes((16, 0, 0))), sync_packet(55, b'\x00'),
                  sync_packet(40, b'\x02'), sync_packet(42, b'\x00\x10'),
                  sync_packet(42, b'\x00\x08', ids=(1,)),
                  sync_packet(42, b'\x00\x08', ids=(2, 3, 12, 14)),
                  sync_packet(42, b'\x00\x08', ids=(2, 2)),
                  sync_packet(42, b'\x00\x08', ids=()),
                  sync_packet(50, bytes((20, 0, 0))),
                  sync_packet(50, bytes((16, 1, 0))),
                  sync_packet(50, bytes((16, 0, 1))),
                  sync_packet(50, bytes((16, 0))),
                  build_packet(3, 3, bytes((50, 16, 0, 0))),
                  build_packet(3, 3, bytes((40, 1))), b'',
                  sync_packet(42, b'\x00\x08')[:-1]+b'\x00']
        with patch.object(ServoBus, '_send') as send:
            for packet in denied:
                with self.subTest(packet=packet), self.assertRaises(ValueError):
                    bus._send(packet)
            send.assert_not_called()


class FakeBus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MainTests(unittest.TestCase):
    def test_version_is_available_without_hardware_or_robot_config(self):
        with patch.object(hold, 'HipHoldBus') as bus, \
             patch.object(hold, 'load_config') as load, \
             patch('sys.stdout', new_callable=io.StringIO) as output:
            with self.assertRaises(SystemExit) as caught:
                hold.main(['--version'])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn(hold.VERSION, output.getvalue())
            bus.assert_not_called()
            load.assert_not_called()

    def test_move_requires_supported_and_p_is_restricted_before_serial_access(self):
        invalid = [['--move'], ['--p', '0'], ['--p', '6'], ['--p', '20'],
                   ['--d', '1'], ['--i', '1'], ['--seconds', '0'],
                   ['--seconds', '0.1'], ['--seconds', '31'], ['--seconds', 'nan']]
        with patch.object(hold, 'HipHoldBus') as bus, \
             patch('sys.stderr', new_callable=io.StringIO):
            for args in invalid:
                with self.subTest(args=args), self.assertRaises(SystemExit) as caught:
                    hold.main(args)
                self.assertEqual(caught.exception.code, 2)
            bus.assert_not_called()

    def test_default_measures_only_and_never_changes_robot_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'robot.json'
            path.write_text(json.dumps(config()))
            original = path.read_bytes()
            with patch.object(hold, 'ROOT', root), \
                 patch.object(hold, 'exclusive_serial', return_value=nullcontext()), \
                 patch.object(hold, 'stop_signals', return_value=nullcontext()), \
                 patch.object(hold, 'HipHoldBus', return_value=FakeBus()) as bus, \
                 patch.object(hold, 'HipPositionHold') as core, \
                 patch('sys.stdout', new_callable=io.StringIO):
                hold.main(['--config', str(path), '--seconds', '0.5'])
            self.assertFalse(bus.call_args.kwargs['allow_motion'])
            self.assertFalse(core.call_args.kwargs['moving'])
            core.return_value.preflight.assert_called_once_with()
            core.return_value.observe.assert_called_once_with(seconds=.5)
            core.return_value.run.assert_not_called()
            self.assertEqual(path.read_bytes(), original)

    def test_explicit_static_gain_choice_does_not_persist_robot_config(self):
        for p in (5, 8, 12, 16):
            with self.subTest(p=p), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root/'robot.json'
                path.write_text(json.dumps(config()))
                original = path.read_bytes()
                with patch.object(hold, 'ROOT', root), \
                     patch.object(hold, 'exclusive_serial', return_value=nullcontext()), \
                     patch.object(hold, 'stop_signals', return_value=nullcontext()), \
                     patch.object(hold, 'HipHoldBus', return_value=FakeBus()) as bus, \
                     patch.object(hold, 'HipPositionHold') as core, \
                     patch('sys.stdout', new_callable=io.StringIO):
                    hold.main(['--config', str(path), '--move', '--supported', '--p', str(p)])
                self.assertTrue(bus.call_args.kwargs['allow_motion'])
                self.assertTrue(core.call_args.kwargs['moving'])
                core.return_value.preflight.assert_called_once_with()
                core.return_value.run.assert_called_once_with(p=p)
                core.return_value.observe.assert_not_called()
                self.assertEqual(path.read_bytes(), original)

    def test_preflight_failure_never_enters_motion_or_changes_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'robot.json'
            path.write_text(json.dumps(config()))
            original = path.read_bytes()
            with patch.object(hold, 'ROOT', root), \
                 patch.object(hold, 'exclusive_serial', return_value=nullcontext()), \
                 patch.object(hold, 'stop_signals', return_value=nullcontext()), \
                 patch.object(hold, 'HipHoldBus', return_value=FakeBus()), \
                 patch.object(hold, 'HipPositionHold') as core, \
                 patch('sys.stdout', new_callable=io.StringIO):
                core.return_value.preflight.side_effect = ValueError('injected preflight failure')
                result = hold.main(['--config', str(path), '--move', '--supported'])
            self.assertEqual(result, 1)
            core.return_value.run.assert_not_called()
            core.return_value.observe.assert_not_called()
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
