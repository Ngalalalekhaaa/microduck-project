"""All-15 midpoint maintenance on a protocol simulator; no serial/network use."""
import importlib.util
import json
from pathlib import Path
import struct

import pytest
from microduck_deploy.config import default_config
from microduck_deploy.servo import build_packet, decode_sign_magnitude

spec = importlib.util.spec_from_file_location('duck1_all_recenter', Path(__file__).parents[1]/'scripts/recenter_duck1_all.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def encode(value):
    return abs(value) | (0x8000 if value < 0 else 0)


class Wire:
    def __init__(self):
        self.reg = {}
        self.packets = []
        self.rx = bytearray()
        self.calibrated = set()
        self.drop_ack_id = None
        self.turn_on_id = None
        self.bad_target_id = None
        self.fail_torque_read = None  # (ID, ordinal after its calibration)
        self.torque_reads = {}
        self.move_next_after_id = None
        for sid in m.IDS:
            raw = bytearray(87)
            raw[:6] = bytes([3, 46, 0, 10, 31, sid])
            raw[8] = 1
            struct.pack_into('<HH', raw, 9, 0, 4095)
            raw[30], raw[33], raw[55] = 1, 4, 1
            raw[62], raw[63] = 75, 30
            struct.pack_into('<H', raw, 56, 1500+sid*17)
            self.reg[sid] = raw

    def reset_input_buffer(self):
        self.rx.clear()

    def write(self, packet):
        self.packets.append(bytes(packet))
        sid, instruction, params = packet[2], packet[4], packet[5:-1]
        raw = self.reg[sid]
        data = b''
        if instruction == 2:
            if params == b'\x28\x01' and sid in self.calibrated:
                self.torque_reads[sid] = self.torque_reads.get(sid, 0)+1
                if self.fail_torque_read == (sid, self.torque_reads[sid]):
                    return len(packet)
            data = bytes(raw[params[0]:params[0]+params[1]])
        elif instruction == 3:
            raw[params[0]:params[0]+len(params)-1] = params[1:]
        elif instruction == 11:
            self.calibrated.add(sid)
            current = decode_sign_magnitude(struct.unpack_from('<H', raw, 56)[0])
            old_offset = decode_sign_magnitude(struct.unpack_from('<H', raw, 31)[0])
            target = struct.unpack('<H', params)[0]
            # Model observed firmware reference behavior: the calibration
            # itself can clear accumulated turns; the host never sends RESET.
            new_offset = old_offset+(current % 4096)-target
            struct.pack_into('<H', raw, 31, encode(new_offset))
            struct.pack_into('<H', raw, 56, target+20 if self.bad_target_id == sid else target)
            if self.turn_on_id == sid:
                raw[40] = 1
            if self.move_next_after_id == sid:
                struct.pack_into('<H', self.reg[sid+1], 56, 3500)
            if self.drop_ack_id == sid:
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
    cfg['imu_mount_verified'] = True
    cfg['imu']['i2c_bus'] = 3
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


def expected_writes(sid):
    return [build_packet(sid, 3, b'\x37\x00'), build_packet(sid, 11, b'\x00\x08'), build_packet(sid, 3, b'\x37\x01')]


def test_prepare_only_reads_every_servo_and_backs_up_multiturn_raw(setup):
    config, wire, bus = setup
    for sid, position in ((1, 5432), (2, 4861), (10, -100)):
        struct.pack_into('<H', wire.reg[sid], 56, encode(position))
    _, _, _, backup, state = prepare(setup)
    assert wire.changes() == []
    assert (backup/'robot.json.before').read_bytes() == config.read_bytes()
    assert state['ids'] == m.IDS
    assert state['body_joint_ids'] == m.BODY_IDS
    assert state['mouth_id'] == 10
    assert state['snapshot']['position_samples']['1'] == [5432]*12
    assert state['snapshot']['position_samples']['2'] == [4861]*12
    assert state['snapshot']['position_samples']['10'] == [-100]*12
    assert all(len(bytes.fromhex(value)) == 56 for value in state['snapshot']['registers_0_55_hex'].values())
    assert all(len(bytes.fromhex(value)) == 40 for value in state['snapshot']['eeprom_hex'].values())


def test_all_fifteen_success_exactly_45_writes_no_goals_or_on_and_config_unchanged(setup):
    config, wire, bus, backup, state = prepare(setup)
    before = config.read_bytes()
    result = m.apply(config, bus, backup)
    assert result['phase'] == 'complete'
    assert wire.changes() == [packet for sid in m.IDS for packet in expected_writes(sid)]
    assert len(wire.changes()) == 45
    assert config.read_bytes() == before
    assert result['all_torque_off'] and result['all_locked'] and result['config_unchanged']
    assert set(result['results']) == {str(sid) for sid in m.IDS}
    for sid in m.IDS:
        assert result['final_snapshot']['position_samples'][str(sid)] == [2048]*12
        index = wire.packets.index(expected_writes(sid)[1])
        assert wire.packets[index+1] == build_packet(sid, 2, b'\x28\x01')
        assert wire.reg[sid][40] == 0 and wire.reg[sid][55] == 1
    assert bus.active_id is None


def test_multiturn_known_hardware_fixture_and_other_outside_single_turn(setup):
    config, wire, bus = setup
    for sid, position in ((1, 4358), (2, 5432), (3, 4861), (4, -100)):
        struct.pack_into('<H', wire.reg[sid], 56, encode(position))
    _, _, _, backup, _ = prepare(setup)
    result = m.apply(config, bus, backup)
    old_case = result['results']['1']
    assert old_case['offset_before'] == 0
    assert old_case['offset_after'] == -1786
    assert old_case['raw_position_delta'] == -2310
    assert old_case['offset_reference_delta'] == 1786
    assert old_case['whole_turn_state_delta'] == -1
    assert result['results']['5']['whole_turn_state_delta'] == 0
    assert all(values == [2048]*12 for values in result['final_snapshot']['position_samples'].values())
    assert all(p[4] != 10 for p in wire.packets)


@pytest.mark.parametrize('bad_id', [1, 10, 15])
def test_failure_stops_later_ids_and_locks_failed_servo(setup, bad_id):
    config, wire, bus, backup, _ = prepare(setup)
    wire.bad_target_id = bad_id
    with pytest.raises(RuntimeError, match='raw calibration feedback'):
        m.apply(config, bus, backup)
    assert wire.changes() == [p for sid in range(1, bad_id+1) for p in expected_writes(sid)]
    assert wire.reg[bad_id][55] == 1 and wire.reg[bad_id][40] == 0
    state = json.loads((backup/'state.json').read_text())
    assert state['phase'] == 'failed'
    assert state['current_id'] == bad_id
    assert len(state['results']) == bad_id
    before = len(wire.packets)
    with pytest.raises(RuntimeError, match='already attempted'):
        m.apply(config, bus, backup)
    assert len(wire.packets) == before


def test_drift_between_servos_refused_before_next_servo_writes(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.move_next_after_id = 3
    with pytest.raises(RuntimeError, match='ID4 moved since prepare'):
        m.apply(config, bus, backup)
    assert wire.changes() == [p for sid in (1, 2, 3) for p in expected_writes(sid)]
    assert json.loads((backup/'state.json').read_text())['current_id'] == 4


@pytest.mark.parametrize('failure', ['on', 'unknown', 'cleanup_unknown'])
def test_torque_fault_sends_one_off_stops_batch_and_locks(setup, failure):
    config, wire, bus, backup, _ = prepare(setup)
    sid = 3
    wire.turn_on_id = sid if failure == 'on' else None
    wire.fail_torque_read = (sid, 1 if failure == 'unknown' else 14) if failure != 'on' else None
    expected_error = RuntimeError if failure == 'on' else m.ServoTimeout
    with pytest.raises(expected_error):
        m.apply(config, bus, backup)
    assert wire.changes() == ([p for prior in (1, 2) for p in expected_writes(prior)] +
                              [expected_writes(sid)[0], expected_writes(sid)[1],
                               build_packet(sid, 3, b'\x28\x00'), expected_writes(sid)[2]])
    assert wire.reg[sid][40] == 0 and wire.reg[sid][55] == 1
    assert wire.calibrated == {1, 2, 3}
    assert bus.active_id is None


@pytest.mark.parametrize('failure', ['on', 'unknown'])
def test_disk_failure_cannot_prevent_current_servo_emergency_off(setup, monkeypatch, capsys, failure):
    config, wire, bus, backup, _ = prepare(setup)
    sid = 10
    wire.turn_on_id = sid if failure == 'on' else None
    wire.fail_torque_read = (sid, 1) if failure == 'unknown' else None
    original = m.save
    def fail(path, value):
        if sid in wire.calibrated:
            raise OSError('mock disk full')
        return original(path, value)
    monkeypatch.setattr(m, 'save', fail)
    expected_error = RuntimeError if failure == 'on' else m.ServoTimeout
    with pytest.raises(expected_error) as raised:
        m.apply(config, bus, backup)
    assert 'mock disk full' not in str(raised.value)
    assert wire.changes()[-3:] == [expected_writes(sid)[0], expected_writes(sid)[1], build_packet(sid, 3, b'\x28\x00')]
    assert wire.reg[sid][40] == 0
    assert wire.calibrated == set(range(1, sid+1))
    assert 'mock disk full' in capsys.readouterr().err
    assert bus.warnings and bus.active_id is None
    assert json.loads((backup/'state.json').read_text())['phase'] != 'prepared'


def test_ack_loss_is_not_retried(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.drop_ack_id = 7
    result = m.apply(config, bus, backup)
    assert result['phase'] == 'complete'
    assert result['results']['7']['calibration_ack'] is None
    assert wire.changes() == [p for sid in m.IDS for p in expected_writes(sid)]


def test_successful_backup_not_replayed(setup):
    config, wire, bus, backup, _ = prepare(setup)
    m.apply(config, bus, backup)
    before = len(wire.packets)
    with pytest.raises(RuntimeError, match='already attempted'):
        m.apply(config, bus, backup)
    assert len(wire.packets) == before


@pytest.mark.parametrize('change', ['calibration', 'zero', 'jaw'])
def test_existing_calibration_refused_without_hardware_access(setup, change):
    config, wire, bus = setup
    cfg = json.loads(config.read_text())
    if change == 'calibration':
        cfg['calibration'] = {'pose': 'straight'}
    elif change == 'zero':
        cfg['joints'][0]['zero_tick'] = 0
    else:
        cfg['jaw'] = {'closed_tick': 100}
    config.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError, match='Existing'):
        m.prepare(config, bus)
    assert wire.packets == []


@pytest.mark.parametrize('sid, instruction, params', [
    (10, 3, b'\x2a\x00\x08'), (10, 3, b'\x28\x01'), (10, 3, b'\x1f\x00\x08'),
    (10, 10, b''), (10, 11, b''), (10, 11, b'\x01\x08'),
    (9, 11, b'\x00\x08'), (11, 3, b'\x28\x00'), (254, 2, b'\x28\x01'),
])
def test_packet_scope_prevents_other_ids_goals_on_offset_and_reset(setup, sid, instruction, params):
    _, wire, bus = setup
    bus.active_id = 10
    with pytest.raises(RuntimeError):
        bus._send(build_packet(sid, instruction, params))
    assert wire.packets == []


def test_changed_eeprom_or_config_refuses_apply_before_writes(setup):
    config, wire, bus, backup, _ = prepare(setup)
    wire.reg[8][21] ^= 1
    with pytest.raises(RuntimeError, match='EEPROM changed'):
        m.apply(config, bus, backup)
    assert wire.changes() == []
    wire.reg[8][21] ^= 1
    config.write_text(config.read_text()+'\n')
    with pytest.raises(RuntimeError, match='Config changed'):
        m.apply(config, bus, backup)
    assert wire.changes() == []


def test_unstable_prepare_aborts_without_any_hardware_write(setup, monkeypatch):
    config, wire, bus = setup
    original = bus.read_feedback
    count = 0
    def moving(sid):
        nonlocal count
        if sid == 8:
            count += 1
            struct.pack_into('<H', wire.reg[8], 56, 1800+count)
        return original(sid)
    monkeypatch.setattr(bus, 'read_feedback', moving)
    with pytest.raises(RuntimeError, match='ID8 moved during capture'):
        m.prepare(config, bus)
    assert wire.changes() == []
    assert bus.active_id is None


@pytest.mark.parametrize('trigger', [3, 15])
def test_previous_servo_on_at_next_precheck_or_final_snapshot_is_globally_unloaded(setup, monkeypatch, trigger):
    config, wire, bus, backup, _ = prepare(setup)
    original = wire.write
    def inject(packet):
        result = original(packet)
        if packet[2] == trigger and packet[4] == 11:
            wire.reg[1][40] = 1
        return result
    monkeypatch.setattr(wire, 'write', inject)
    with pytest.raises(RuntimeError, match='ID1'):
        m.apply(config, bus, backup)
    assert wire.calibrated == set(range(1, trigger+1))
    assert wire.changes() == ([p for sid in range(1, trigger+1) for p in expected_writes(sid)] +
                              [build_packet(1, 3, b'\x28\x00')])
    assert all(wire.reg[sid][40] == 0 for sid in m.IDS)
    assert bus.active_id is None
    state = json.loads((backup/'state.json').read_text())
    assert state['phase'] == 'failed' and 'ID1' in state['error']


def test_global_cleanup_unknown_servo_does_not_block_other_off_attempts(setup, monkeypatch):
    config, wire, bus, backup, _ = prepare(setup)
    original = wire.write
    def inject(packet):
        sid, instruction, params = packet[2], packet[4], packet[5:-1]
        if 3 in wire.calibrated and sid == 1 and instruction == 2 and params == b'\x28\x01':
            # Remain unreachable for every torque read, including OFF readback.
            wire.packets.append(bytes(packet))
            return len(packet)
        result = original(packet)
        if sid == 3 and instruction == 11:
            wire.reg[1][40] = 1
            wire.reg[2][40] = 1
        return result
    monkeypatch.setattr(wire, 'write', inject)
    with pytest.raises(m.ServoTimeout):
        m.apply(config, bus, backup)
    assert wire.calibrated == {1, 2, 3}
    assert wire.changes()[-2:] == [build_packet(1, 3, b'\x28\x00'), build_packet(2, 3, b'\x28\x00')]
    assert wire.reg[1][40] == 0 and wire.reg[2][40] == 0
    assert wire.changes().count(build_packet(1, 3, b'\x28\x00')) == 1
    assert any('Global torque cleanup ID1' in warning for warning in bus.warnings)
    # The last servo is still checked after an earlier servo cannot be verified.
    assert wire.packets[-1] == build_packet(15, 2, b'\x28\x01')
    assert bus.active_id is None


@pytest.mark.parametrize('explicit', [False, True])
def test_cli_default_only_prepares_and_explicit_combined_mode_prepares_then_applies(setup, monkeypatch, capsys, explicit):
    config, _, bus = setup
    backup = config.parent/'backups'/'fresh'
    calls = []
    state = {'snapshot': {'position_samples': {str(sid): [1900]*12 for sid in m.IDS}}}
    def fake_prepare(path, passed_bus):
        assert path == config and passed_bus is bus
        calls.append('prepare')
        return backup, state
    def fake_apply(path, passed_bus, selected_backup):
        assert path == config and passed_bus is bus and selected_backup == backup
        assert calls == ['prepare']
        calls.append('apply')
        return {'phase': 'complete', 'all_torque_off': True, 'all_locked': True, 'next_step': 'software calibration'}
    monkeypatch.setattr(m, 'quiet', lambda _: None)
    monkeypatch.setattr(m, 'MaintenanceBus', lambda **_: bus)
    monkeypatch.setattr(m, 'prepare', fake_prepare)
    monkeypatch.setattr(m, 'apply', fake_apply)
    monkeypatch.setattr(m.sys, 'argv', ['recenter_duck1_all.py', '--config', str(config)]+(['--prepare-and-apply'] if explicit else []))
    m.main()
    assert calls == (['prepare', 'apply'] if explicit else ['prepare'])
    assert str(backup) in capsys.readouterr().out


def test_cli_combined_mode_never_applies_after_prepare_failure(setup, monkeypatch):
    config, _, bus = setup
    calls = []
    def fail_prepare(*args):
        calls.append('prepare')
        raise RuntimeError('unstable pose')
    monkeypatch.setattr(m, 'quiet', lambda _: None)
    monkeypatch.setattr(m, 'MaintenanceBus', lambda **_: bus)
    monkeypatch.setattr(m, 'prepare', fail_prepare)
    monkeypatch.setattr(m, 'apply', lambda *args: calls.append('apply'))
    monkeypatch.setattr(m.sys, 'argv', ['recenter_duck1_all.py', '--config', str(config), '--prepare-and-apply'])
    with pytest.raises(RuntimeError, match='unstable pose'):
        m.main()
    assert calls == ['prepare']


@pytest.mark.parametrize('extra,error', [(['--apply'], SystemExit), (['--backup', '/tmp/no-backup'], RuntimeError)])
def test_cli_combined_mode_rejects_conflicting_apply_or_backup(setup, monkeypatch, extra, error):
    config, _, _ = setup
    monkeypatch.setattr(m.sys, 'argv', ['recenter_duck1_all.py', '--config', str(config), '--prepare-and-apply']+extra)
    monkeypatch.setattr(m, 'quiet', lambda _: pytest.fail('must not inspect/open hardware'))
    with pytest.raises(error):
        m.main()
