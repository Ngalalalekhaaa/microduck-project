import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

import robot_profile
import server

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(robot_profile, 'DEPLOY_CONFIG', None)
    monkeypatch.setattr(server, 'READ_ONLY', False)
    monkeypatch.setattr(server, 'IDS', list(range(1, 15)))
    monkeypatch.setattr(server, 'PRESENT', list(range(1, 15)))
    monkeypatch.setattr(server, 'BUS', server.FakeBus(list(range(1, 15))))
    monkeypatch.setattr(server, 'POSES_DIR', str(tmp_path / 'poses'))
    monkeypatch.setattr(server, 'LOG_DIR', str(tmp_path / 'logs'))
    monkeypatch.setattr(server, 'DIRS_FILE', str(tmp_path / 'directions.json'))


def config_file(tmp_path, calibrated=True):
    joints = [{'name': name, 'id': sid, 'direction': -1 if calibrated else None,
               'zero_tick': 1900.25 if calibrated else None,
               'mapping_verified': calibrated}
              for sid, name in enumerate(robot_profile.BODY_NAMES, 1)]
    p = tmp_path / 'robot.json'
    p.write_text(json.dumps({'joints': joints}))
    return p


def test_model_and_backend_map_agree():
    model = json.loads((HERE / 'model/model.json').read_text())
    seen = {}
    for body in model['bodies']:
        joint = body.get('joint')
        if joint and joint.get('id') is not None:
            seen[joint['id']] = joint['name']
    assert seen == server.JOINT_NAMES
    assert {i: server.JOINT_NAMES[i] for i in range(1, 15)} == dict(enumerate(robot_profile.BODY_NAMES, 1))
    assert server.JOINT_NAMES[15] == 'mouth'
    assert server.parse_ids(server.parse_args([]).ids) == list(range(1, 15))


def test_unverified_directions_are_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DIRS_FILE', str(HERE / 'directions.json'))
    robot_profile.configure(config_file(tmp_path, False))
    assert server.load_dirs() == {}
    assert robot_profile.calibration({}, False)['ready_ids'] == []


def test_reads_verified_calibration_without_writing(tmp_path):
    p = config_file(tmp_path)
    original = p.read_bytes()
    robot_profile.configure(p)
    dirs = server.load_dirs()
    cal = robot_profile.calibration(dirs)
    assert cal['ready_ids'] == list(range(1, 15))
    assert cal['zero_ticks']['5'] == 1900.25
    dirs[5] = 1
    assert 5 not in robot_profile.calibration(dirs)['ready_ids']
    assert p.read_bytes() == original


def test_rejects_wrong_deploy_mapping(tmp_path):
    p = config_file(tmp_path)
    doc = json.loads(p.read_text())
    doc['joints'][0]['id'] = 20
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='ID 1'):
        robot_profile.configure(p)


def test_pose_saves_actual_positions_and_never_moves():
    server.BUS.torque = {sid: 0 for sid in server.IDS}
    for sid in server.IDS:
        server.BUS.pos[sid] = 1000 + sid
        server.BUS.goal[sid] = 3000
    fn = server.save_pose('参考姿态')
    result = json.loads((Path(server.POSES_DIR) / fn).read_text())
    assert [r['id'] for r in result['rows']] == list(range(1, 15))
    assert [r['pos'] for r in result['rows']] == list(range(1001, 1015))
    assert result['rows'][4]['joint'] == 'left_ankle'
    assert result['rows'][13]['joint'] == 'right_ankle'
    assert all(v == 0 for v in server.BUS.torque.values())
    assert all(v == 3000 for v in server.BUS.goal.values())
    assert server.save_pose('参考姿态') != fn


def test_partial_pose_does_not_create_file(monkeypatch):
    server.PRESENT = list(range(1, 14))
    states = server.BUS.states(server.IDS)
    states[14] = None
    monkeypatch.setattr(server.BUS, 'states', lambda ids: states)
    with pytest.raises(ValueError, match='14'):
        server.save_pose('不完整')
    assert not Path(server.POSES_DIR).exists()


def test_simulated_poses_do_not_appear_for_real_bus(monkeypatch):
    server.save_pose('模拟姿态')
    assert len(server.list_poses()) == 1
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    assert server.list_poses() == []


def test_websocket_reports_mapping_and_calibration():
    with TestClient(server.app) as client:
        info = client.get('/api/info').json()
        assert info['names']['5'] == 'left_ankle'
        with client.websocket_connect('/ws') as ws:
            hello = ws.receive_json()
            assert hello['type'] == 'hello'
            assert hello['ids'] == list(range(1, 15))
            assert hello['names']['14'] == 'right_ankle'
            assert hello['calibration']['fake'] is True


def test_calibration_command_does_not_write_deploy_or_servo(tmp_path, monkeypatch):
    p = config_file(tmp_path, False)
    original = p.read_bytes()
    robot_profile.configure(p)
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    with pytest.raises(ValueError, match='软件零位'):
        server.handle({'op': 'calib_mid_all'})
    assert p.read_bytes() == original


def test_real_position_stream_preserves_other_registers(tmp_path, monkeypatch):
    robot_profile.configure(config_file(tmp_path, False))
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    writes = []
    monkeypatch.setattr(server.BUS, 'sync_write', lambda addr, size, items: writes.append((addr, size, items)))
    server.handle({'op': 'goals_stream', 'goals': {'5': 1200, '14': 2400}, 'dt': .05})
    server.handle({'op': 'stream_end', 'ids': [5, 14]})
    server.handle({'op': 'release', 'ids': [5, 14]})
    assert writes == [(42, 2, [(5, b'\xb0\x04'), (14, b'\x60\x09')])]
    for bad in (-1, 4096, True, 1.5):
        with pytest.raises(ValueError, match='刻度'):
            server.handle({'op': 'goal', 'id': 5, 'pos': bad})
    for op in ('goals_profile', 'write_reg', 'set_id'):
        with pytest.raises(ValueError, match='保留现有'):
            server.handle({'op': op})
    assert len(writes) == 1


def test_readonly_blocks_motion_but_allows_relax_and_capture(monkeypatch):
    monkeypatch.setattr(server, 'HOLD_ENABLED', False)
    monkeypatch.setattr(server, 'READ_ONLY', True)
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    original = dict(server.BUS.goal)
    for cmd in ({'op':'goal','id':14,'pos':2048}, {'op':'torque','on':1}, {'op':'calibrate','id':14}, {'op':'goals_stream','goals':{'14':2048}}):
        with pytest.raises(ValueError, match='标定读数模式'):
            server.handle(cmd)
    server.handle({'op':'torque','on':0,'id':14})
    assert server.BUS.torque[14] == 0
    assert server.BUS.goal == original
    server.save_pose('参考刻度')
    assert len(server.list_poses()) == 1


def test_page_log_allowed_in_readonly_without_servo_writes(monkeypatch):
    monkeypatch.setattr(server, 'READ_ONLY', True)
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    messages = []
    monkeypatch.setattr(server, 'log', lambda *args: messages.append(args))
    monkeypatch.setattr(server, 'BUS', None)
    assert server.handle({'op':'page_log', 'msg':'页面已连接'}) is None
    assert messages[0][0] == '页面已连接'


def protocol_bus(monkeypatch, auto_enable=True):
    import feetech
    from sim_bus import SimSerial
    serial = SimSerial([1, 14], {1: 1800, 14: 2200})
    for servo in serial.servos.values():
        servo.r[30] = 1
        servo.goal = 3900
        if auto_enable:
            old = servo.write
            def write(addr, data, servo=servo, old=old):
                old(addr, data)
                if addr == 42:
                    servo.r[40] = 1
            servo.write = write
    bus = feetech.FeetechBus(None, ser=serial)
    monkeypatch.setattr(server, 'BUS', bus)
    monkeypatch.setattr(server, 'IDS', [1, 14])
    monkeypatch.setattr(server, 'PRESENT', [1, 14])
    monkeypatch.setattr(server, 'is_fake', lambda: False)
    return bus, serial


def test_hold_handles_goal_write_that_enables_torque(monkeypatch):
    bus, serial = protocol_bus(monkeypatch)
    monkeypatch.setattr(server, 'READ_ONLY', True)
    monkeypatch.setattr(server, 'HOLD_ENABLED', True)
    result = server.handle({'op': 'torque', 'on': 1})
    assert result['bad'] == {}
    assert result['aligned'] == {'1': 1800, '14': 2200}
    for servo in serial.servos.values():
        assert servo.goal == servo.phys
        assert servo.r[40] == 1
    with pytest.raises(ValueError, match='标定读数模式'):
        server.handle({'op': 'goal', 'id': 14, 'pos': 2500})
    off = server.handle({'op': 'torque', 'on': 0})
    assert off['bad'] == {} and off['on'] == 0
    assert all(servo.r[40] == 0 for servo in serial.servos.values())


def test_failed_alignment_switches_off_auto_enabled_and_previous_servos(monkeypatch):
    bus, serial = protocol_bus(monkeypatch)
    read = bus.read_u16
    monkeypatch.setattr(bus, 'read_u16', lambda sid, addr: 17 if (sid, addr) == (14, 42) else read(sid, addr))
    on, aligned, bad = server.torque_on_aligned([1, 14])
    assert on == [] and set(bad) == {1, 14}
    assert all(servo.r[40] == 0 for servo in serial.servos.values())
    assert all(servo.goal == servo.phys for servo in serial.servos.values())


def test_out_of_range_position_does_not_send_goals(monkeypatch):
    bus, serial = protocol_bus(monkeypatch)
    serial.servos[14].phys = 4353
    on, aligned, bad = server.torque_on_aligned([1, 14])
    assert on == [] and aligned == {} and 14 in bad
    assert all(servo.r[40] == 0 and servo.goal == 3900 for servo in serial.servos.values())


def test_unknown_mode_refuses_torque_and_goal_writes(monkeypatch):
    bus, serial = protocol_bus(monkeypatch)
    serial.servos[1].r[33] = 1
    on, aligned, bad = server.torque_on_aligned([1, 14])
    assert not on and 1 in bad
    assert all(servo.r[40] == 0 and servo.goal == 3900 for servo in serial.servos.values())
