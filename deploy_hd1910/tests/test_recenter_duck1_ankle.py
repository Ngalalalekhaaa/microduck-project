"""Maintenance safety on a simulated byte transport; never opens a UART."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest
from microduck_deploy.config import default_config
from microduck_deploy.servo import build_packet, decode_sign_magnitude

spec = importlib.util.spec_from_file_location('duck1_recenter', Path(__file__).parents[1]/'scripts/recenter_duck1_ankle.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def encode(value):
    return abs(value) | (0x8000 if value < 0 else 0)


class Wire:
    def __init__(self):
        self.reg = {}
        self.packets = []
        self.rx = bytearray()
        self.drop_calibration_ack = False
        self.turn_on_after_calibration = False
        self.mutate_other_after_calibration = False
        self.fail_torque_read_number_after_calibration = None
        self.torque_reads_after_calibration = 0
        self.calibrated = False
        for sid in m.ALL_IDS:
            raw = bytearray(87)
            raw[:6] = bytes([3, 46, 0, 10, 31, sid])
            raw[8] = 1
            struct.pack_into('<HH', raw, 9, 0, 4095)
            raw[30], raw[33], raw[55] = 1, 4, 1
            raw[62], raw[63] = 75, 30
            struct.pack_into('<H', raw, 56, 297 if sid == 15 else 2048)
            self.reg[sid] = raw

    def reset_input_buffer(self):
        self.rx.clear()

    def write(self, packet):
        self.packets.append(bytes(packet))
        sid, instruction, params = packet[2], packet[4], packet[5:-1]
        raw = self.reg[sid]
        data = b''
        if instruction == 2:
            if sid == 15 and params == b'\x28\x01' and self.calibrated:
                self.torque_reads_after_calibration += 1
                if self.torque_reads_after_calibration == self.fail_torque_read_number_after_calibration:
                    return len(packet)
            data = bytes(raw[params[0]:params[0]+params[1]])
        elif instruction == 3:
            raw[params[0]:params[0]+len(params)-1] = params[1:]
        elif instruction == 11:
            self.calibrated = True
            position = struct.unpack_from('<H', raw, 56)[0]
            old_offset = decode_sign_magnitude(struct.unpack_from('<H', raw, 31)[0])
            target = struct.unpack('<H', params)[0]
            struct.pack_into('<H', raw, 31, encode(old_offset+position-target))
            struct.pack_into('<H', raw, 56, target)
            if self.turn_on_after_calibration:
                raw[40] = 1
            if self.mutate_other_after_calibration:
                self.reg[1][21] ^= 1
            if self.drop_calibration_ack:
                return len(packet)
        else:
            raise AssertionError(f'Unexpected packet: {packet.hex()}')
        body = bytes([sid, len(data)+2, 0])+data
        self.rx += b'\xff\xff'+body+bytes([(~sum(body)) & 255])
        return len(packet)

    def read(self, count):
        value = bytes(self.rx[:count])
        del self.rx[:count]
        return value

    def close(self):
        pass

    def changes(self):
        return [p for p in self.packets if p[4] != 2]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(m.time, 'sleep', lambda _: None)
    cfg = default_config()
    for joint, sid in zip(cfg['joints'], m.BODY_IDS):
        joint.update(id=sid, direction=1 if sid == 15 else -1, mapping_verified=True, zero_tick=None)
    config = tmp_path/'deploy_hd1910'/'robot.json'
    config.parent.mkdir()
    config.write_text(json.dumps(cfg))
    wire = Wire()
    bus = m.MaintenanceBus(None, transport=wire, timeout=.002)
    yield config, wire, bus
    bus.close()


def prepare(setup):
    config, wire, bus = setup
    backup, state = m.prepare(config, bus)
    return config, wire, bus, backup, state


def test_prepare_is_only_reads_with_full_backup(setup):
    config, wire, bus, backup, state = prepare(setup)
    assert wire.changes() == []
    assert (backup/'robot.json.before').read_bytes() == config.read_bytes()
    assert state['phase'] == 'prepared'
    assert len(state['snapshot']['eeprom_hex']) == 15
    assert state['snapshot']['id15_positions'] == [297]*12
    assert len(bytes.fromhex(state['snapshot']['id15_registers_0_55_hex'])) == 56
    assert [row['id'] for row in state['range_report'] if row['will_change']] == [15]


def test_apply_exact_packets_preserves_config_and_checks_torque_first(setup):
    config, wire, bus, backup, _ = prepare(setup)
    original = config.read_bytes()
    result = m.apply(config, bus, backup)
    assert result['phase'] == 'complete'
    assert result['effective_delta_ticks'] == 1751
    assert config.read_bytes() == original
    expected = [build_packet(15, 3, b'\x37\x00'), build_packet(15, 11, b'\x00\x08'), build_packet(15, 3, b'\x37\x01')]
    assert wire.changes() == expected
    index = wire.packets.index(expected[1])
    assert wire.packets[index+1] == build_packet(15, 2, b'\x28\x01')
    assert [e['packet_hex'] for e in result['events']] == [p.hex() for p in expected]
    assert wire.reg[15][40] == 0 and wire.reg[15][55] == 1
    assert not bus.allow_changes


def test_replay_rejected_without_new_writes(setup):
    config, wire, bus, backup, _ = prepare(setup)
    m.apply(config, bus, backup)
    count = len(wire.packets)
    with pytest.raises(RuntimeError, match='already attempted'):
        m.apply(config, bus, backup)
    assert len(wire.packets) == count


@pytest.mark.parametrize('change,match', [('config', 'Config changed'), ('eeprom', 'EEPROM changed'), ('pose', 'moved since prepare'), ('torque', 'torque OFF')])
def test_changed_preconditions_prevent_all_writes(setup, change, match):
    config, wire, bus, backup, _ = prepare(setup)
    if change == 'config':
        config.write_text(config.read_text()+'\n')
    elif change == 'eeprom':
        wire.reg[2][21] ^= 1
    elif change == 'pose':
        struct.pack_into('<H', wire.reg[15], 56, 310)
    else:
        wire.reg[10][40] = 1
    with pytest.raises(RuntimeError, match=match):
        m.apply(config, bus, backup)
    assert wire.changes() == []


@pytest.mark.parametrize('existing', ['calibration', 'zero', 'ids', 'direction'])
def test_existing_calibration_or_wrong_robot_refused_before_hardware(setup, existing):
    config, wire, bus = setup
    cfg = json.loads(config.read_text())
    if existing == 'calibration':
        cfg['calibration'] = {'pose': 'straight'}
    elif existing == 'zero':
        cfg['joints'][0]['zero_tick'] = 0
    elif existing == 'ids':
        cfg['joints'][-1]['id'] = 10
    else:
        cfg['joints'][-1]['direction'] = -1
    config.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError):
        m.prepare(config, bus)
    assert wire.packets == []


def test_post_calibration_torque_anomaly_emergency_off_and_fail(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.turn_on_after_calibration = True
    with pytest.raises(RuntimeError, match='Unexpected ID15 torque ON'):
        m.apply(config, bus, backup)
    assert wire.changes() == [build_packet(15, 3, b'\x37\x00'), build_packet(15, 11, b'\x00\x08'), build_packet(15, 3, b'\x28\x00'), build_packet(15, 3, b'\x37\x01')]
    assert wire.reg[15][40] == 0 and wire.reg[15][55] == 1
    assert json.loads((backup/'state.json').read_text())['phase'] == 'failed'
    with pytest.raises(RuntimeError, match='already attempted'):
        m.apply(config, bus, backup)


def test_ack_timeout_never_replays_calibration_and_readback_decides(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.drop_calibration_ack = True
    result = m.apply(config, bus, backup)
    assert result['phase'] == 'complete'
    assert result['calibration_ack'] is None
    assert sum(p[4] == 11 for p in wire.packets) == 1


def test_other_eeprom_change_after_calibration_is_failure(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.mutate_other_after_calibration = True
    with pytest.raises(RuntimeError, match='Unexpected EEPROM change on ID1'):
        m.apply(config, bus, backup)
    assert wire.reg[15][40] == 0 and wire.reg[15][55] == 1
    assert json.loads((backup/'state.json').read_text())['phase'] == 'failed'


@pytest.mark.parametrize('sid,instruction,params', [
    (15, 3, b'\x2a\x00\x08'), (15, 3, b'\x28\x01'), (15, 3, b'\x1f\x00\x08'),
    (15, 3, b'\x32\x05'), (15, 11, b''), (15, 11, b'\x01\x08'),
    (14, 11, b'\x00\x08'), (10, 3, b'\x28\x00'), (14, 3, b'\x37\x00'),
    (15, 10, b''), (254, 2, b'\x28\x01'),
])
def test_packet_gate_forbids_goals_on_other_ids_resets_and_broadcast(setup, sid, instruction, params):
    _, wire, bus = setup
    bus.allow_changes = True
    with pytest.raises(RuntimeError):
        bus._send(build_packet(sid, instruction, params))
    assert wire.packets == []


def test_prepare_mode_gate_rejects_even_allowed_mutation(setup):
    _, wire, bus = setup
    with pytest.raises(RuntimeError, match='Forbidden hardware write'):
        bus._send(build_packet(15, 11, b'\x00\x08'))
    assert wire.packets == []


def test_unstable_capture_refused(setup, monkeypatch):
    config, wire, bus = setup
    original = bus.read_feedback
    count = 0
    def changing(sid):
        nonlocal count
        if sid == 15:
            count += 1
            struct.pack_into('<H', wire.reg[15], 56, 297+count)
        return original(sid)
    monkeypatch.setattr(bus, 'read_feedback', changing)
    with pytest.raises(RuntimeError, match='moved during capture'):
        m.prepare(config, bus)
    assert wire.changes() == []


def test_real_saved_pose_refused(setup):
    config, wire, bus = setup
    poses = config.parent.parent/'servo_web_hd1910'/'poses'
    poses.mkdir(parents=True)
    (poses/'old.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Real saved pose'):
        m.prepare(config, bus)
    assert wire.packets == []


@pytest.mark.parametrize('failed_read', [1, 14])
def test_unknown_torque_after_calibration_or_at_cleanup_attempts_off_once(setup, failed_read):
    config, wire, bus, backup, _ = prepare(setup)
    # Read 1 is immediately after 0x0B; read 14 is cleanup's first read
    # following that check and 12 stable torque/position samples.
    wire.fail_torque_read_number_after_calibration = failed_read
    with pytest.raises(m.ServoTimeout):
        m.apply(config, bus, backup)
    changes = wire.changes()
    assert changes == [build_packet(15, 3, b'\x37\x00'),
                       build_packet(15, 11, b'\x00\x08'),
                       build_packet(15, 3, b'\x28\x00'),
                       build_packet(15, 3, b'\x37\x01')]
    cal_index = wire.packets.index(changes[1])
    assert wire.packets[cal_index+1] == build_packet(15, 2, b'\x28\x01')
    assert wire.reg[15][40] == 0 and wire.reg[15][55] == 1
    assert json.loads((backup/'state.json').read_text())['phase'] == 'failed'
    assert not bus.allow_changes


@pytest.mark.parametrize('failure', ['on', 'unknown'])
def test_disk_failure_never_blocks_emergency_off_or_masks_original_fault(setup, monkeypatch, capsys, failure):
    config, wire, bus, backup, _ = prepare(setup)
    wire.turn_on_after_calibration = failure == 'on'
    wire.fail_torque_read_number_after_calibration = 1 if failure == 'unknown' else None
    original_save = m.save
    def fail_after_calibration(path, value):
        if wire.calibrated:
            raise OSError('mock disk full')
        return original_save(path, value)
    monkeypatch.setattr(m, 'save', fail_after_calibration)
    expected_error = RuntimeError if failure == 'on' else m.ServoTimeout
    with pytest.raises(expected_error) as raised:
        m.apply(config, bus, backup)
    assert 'mock disk full' not in str(raised.value)
    assert wire.changes() == [build_packet(15, 3, b'\x37\x00'),
                              build_packet(15, 11, b'\x00\x08'),
                              build_packet(15, 3, b'\x28\x00')]
    # Relock is intentionally blocked if its write event cannot be persisted.
    assert wire.reg[15][40] == 0 and wire.reg[15][55] == 0
    assert bus.emergency_off_attempted
    assert bus.warnings
    assert 'mock disk full' in capsys.readouterr().err
    assert 'Emergency OFF event could not be persisted' in '\n'.join(bus.warnings)
    # Last durable phase already prevents reusing this partially applied plan.
    assert json.loads((backup/'state.json').read_text())['phase'] == 'calibration_pending'
    assert not bus.allow_changes
