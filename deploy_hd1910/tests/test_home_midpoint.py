"""Calibrated all-body midpoint migration against the real packet parser."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from microduck_deploy.config import (
    calibration_arrays, default_config, load_config, record_reference, require_calibration,
)
from microduck_deploy.policy import HOME, q_to_ticks, ticks_to_q
from test_recenter_duck1_all import Wire

SPEC = importlib.util.spec_from_file_location('home_midpoint_test', Path(__file__).parents[1]/'scripts/home_midpoint.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def encode(value):
    return abs(value)|(0x8000 if value<0 else 0)


class SnapshotWire(Wire):
    """Firmware offset writes change coordinates even while the shaft moves."""
    drop_after_off = None
    wrong_offset_id = None
    unrelated_change_id = None

    def changes(self):
        return [p for p in self.packets if p[4] not in (2,130)]

    def write(self,packet):
        sid,instruction,params=packet[2],packet[4],packet[5:-1]
        if sid==254 and instruction==130:
            assert params==bytes((56,15,*m.IDS))
            self.packets.append(bytes(packet))
            for joint in m.IDS:
                data=bytes(self.reg[joint][56:71]);body=bytes([joint,len(data)+2,0])+data
                self.rx+=b'\xff\xff'+body+bytes([(~sum(body))&255])
            return len(packet)
        if instruction==3 and params[0]==31:
            raw=self.reg[sid]
            old_offset=m.offset(bytes(raw));old_position=struct.unpack_from('<H',raw,56)[0]
            result=super().write(packet)
            if self.wrong_offset_id==sid:
                actual=m.offset(bytes(raw))+1
                struct.pack_into('<H',raw,31,encode(actual))
            new_offset=m.offset(bytes(raw))
            struct.pack_into('<H',raw,56,encode(old_position+old_offset-new_offset))
            self.calibrated.add(sid)
            if self.turn_on_id==sid:raw[40]=1
            if self.unrelated_change_id==sid:raw[18]^=1
            if self.drop_ack_id==sid:del self.rx[-6:]
            return result
        was_on=(instruction==3 and params==b'\x28\x00' and self.reg[sid][40]==1)
        result=super().write(packet)
        if was_on and self.drop_after_off and sid in self.drop_after_off:
            old=struct.unpack_from('<H',self.reg[sid],56)[0]
            struct.pack_into('<H',self.reg[sid],56,encode(old+self.drop_after_off[sid]))
        return result


@pytest.fixture
def rig(tmp_path, monkeypatch):
    import recenter_duck1_all
    monkeypatch.setattr(recenter_duck1_all.time, 'sleep', lambda _:None)
    original_snapshot = m.take_home_snapshot
    now = [0.]
    def simulated_snapshot(*args):
        def sleep(seconds):
            now[0] += seconds
        return original_snapshot(*args, clock=lambda:now[0], sleep=sleep)
    monkeypatch.setattr(m,'take_home_snapshot',simulated_snapshot)
    cfg = default_config()
    for j, sid in zip(cfg['joints'], m.IDS):
        j.update(id=sid, direction=1 if sid == 15 else -1, mapping_verified=True)
    cfg['imu_mount_verified'] = True
    reference = [2045,2191,1620,1203,1696,1784,1781,2084,1988,2042,2056,2474,2760,1812]
    record_reference(cfg, reference, 'home', HOME)
    base = tmp_path/'deploy_hd1910'
    (base/'logs').mkdir(parents=True)
    for name in ('robot.json','robot.home_actions_20261101.json','robot.home_actions_com_20261101.json'):
        (base/name).write_text(json.dumps(cfg))
    web = base.parent/'servo_web_hd1910'/'poses'
    web.mkdir(parents=True)
    (web/'home.json').write_text(json.dumps({'rows':[{'id':sid,'pos':tick,'deg':None}
                                                     for sid,tick in zip(m.IDS,reference)]}))
    (base/'walking_home_selection.json').write_text(json.dumps({'new_home_ticks':reference}))
    files = m.inventory(base)
    wire = SnapshotWire()
    for sid,tick in zip(m.IDS,reference):
        raw = wire.reg[sid]
        raw[40] = 1
        raw[50:53] = bytes([5,0,0])
        struct.pack_into('<H', raw, 56, tick)
        struct.pack_into('<H', raw, 42, tick)
    # The mouth may be ON: this maintenance must neither switch it OFF nor calibrate it.
    wire.reg[10][40] = 1
    bus = m.MidpointBus(None, transport=wire, timeout=.002)
    backup = base/'.backups'/'test'
    backup.mkdir(parents=True)
    state = dict(events=[],results={},ids=m.IDS,phase='starting')
    try:
        yield base,cfg,files,wire,bus,backup,state
    finally:
        bus.close()


def run(r):
    base,cfg,files,wire,bus,backup,state = r
    return m.apply_midpoint(base,cfg,files,bus,backup,state,hold_seconds=0.)


def assert_off(wire):
    assert all(wire.reg[sid][40] == 0 for sid in m.IDS)
    assert all(not (p[4]==3 and p[5:7]==b'\x28\x01') for p in wire.packets)
    assert all(p[2] != 10 for p in wire.changes())


def test_complete_migrates_every_live_reference_and_keeps_physical_home(rig):
    base,cfg,files,wire,bus,backup,state = rig
    before = q_to_ticks(HOME,*calibration_arrays(cfg))
    mouth = bytes(wire.reg[10])
    result = run(rig)
    assert result['phase']=='complete'
    assert not (base/'logs/home_midpoint.pending.json').exists()
    for name in ('robot.json','robot.home_actions_20261101.json','robot.home_actions_com_20261101.json'):
        updated = load_config(base/name)
        assert q_to_ticks(HOME,*calibration_arrays(updated)).tolist() == [2048]*14
        np.testing.assert_allclose(ticks_to_q(before,*calibration_arrays(cfg)),
                                   ticks_to_q([2048]*14,*calibration_arrays(updated)),atol=1e-12)
    pose = json.loads((base.parent/'servo_web_hd1910/poses/home.json').read_text())
    assert [row['pos'] for row in pose['rows']] == [2048]*14
    assert json.loads((base/'walking_home_selection.json').read_text())['new_home_ticks'] == [2048]*14
    assert wire.calibrated == set(m.IDS)
    assert bytes(wire.reg[10]) == mouth
    for sid in m.IDS:
        assert struct.unpack_from('<H',wire.reg[sid],42)[0] == 2048
        assert wire.reg[sid][55] == 1
    assert_off(wire)


def test_slight_home_tracking_error_preserves_reference_and_centers_actual_pose(rig):
    base,cfg,files,wire,bus,backup,state = rig
    old_home = cfg['calibration']['home_ticks'][3]
    struct.pack_into('<H',wire.reg[4],56,old_home+20)
    run(rig)
    updated = load_config(base/'robot.json')
    assert updated['calibration']['home_ticks'][3] == 2028
    assert struct.unpack_from('<H',wire.reg[4],56)[0] == 2048
    qold=ticks_to_q(cfg['calibration']['home_ticks'],*calibration_arrays(cfg))
    qnew=ticks_to_q(updated['calibration']['home_ticks'],*calibration_arrays(updated))
    np.testing.assert_allclose(qold,qnew,atol=1e-12)


@pytest.mark.parametrize('failure',['bad_offset','turn_on','lost_read','unrelated_register'])
def test_partial_failure_leaves_off_invalidates_old_profiles_and_preserves_journal(rig,failure):
    base,cfg,files,wire,bus,backup,state=rig
    if failure=='bad_offset':wire.wrong_offset_id=4
    elif failure=='turn_on':wire.turn_on_id=4
    elif failure=='lost_read':wire.fail_torque_read=(4,1)
    else:wire.unrelated_change_id=4
    with pytest.raises(Exception):run(rig)
    assert_off(wire)
    assert (base/'logs/home_midpoint.pending.json').exists()
    for path,(kind,raw) in files.items():
        if kind=='config':
            with pytest.raises(ValueError):require_calibration(load_config(path),imu=False)
    saved=json.loads((backup/'state.json').read_text())
    assert saved['phase']=='failed'
    assert wire.reg[4][55]==1


def test_lost_calibration_ack_uses_readback_without_replaying_write(rig):
    wire=rig[3]
    wire.drop_ack_id=4
    run(rig)
    assert sum(p[2]==4 and p[4]==3 and p[5]==31 for p in wire.packets)==1
    assert not any(p[4]==11 for p in wire.packets)
    assert_off(wire)


def test_single_snapshot_does_not_impose_home_tracking_tolerance(rig):
    base,cfg,files,wire,bus,backup,state=rig
    struct.pack_into('<H',wire.reg[4],56,1800)
    run(rig)
    cfg_new=load_config(base/'robot.json')
    assert cfg_new['calibration']['home_ticks'][3]==1203+2048-1800
    assert struct.unpack_from('<H',wire.reg[4],56)[0]==2048
    assert_off(wire)


def test_stage_goal_write_allowlist_cannot_enable_or_move_a_joint(rig):
    bus=rig[4]
    bus.active_id=4
    with pytest.raises(RuntimeError):bus.command(4,3,b'\x28\x01')
    with pytest.raises(RuntimeError):bus.command(4,3,b'\x2a\x00\x08')
    bus.active_id=10
    with pytest.raises(RuntimeError):bus.command(10,11,b'\x00\x08')


def test_pose_boundary_and_degree_fields_stop_migration_before_hardware():
    deltas=dict.fromkeys(m.IDS,0);deltas[4]=845
    for row in ({'id':4,'pos':3900,'deg':None},{'id':4,'pos':1203,'deg':30}):
        with pytest.raises(RuntimeError):m.migrate_pose({'rows':[row]},deltas,{})


def test_offline_preview_never_opens_devices(rig,monkeypatch,capsys):
    base=rig[0]
    def forbidden(*a,**k):raise AssertionError('No hardware in preview')
    monkeypatch.setattr(m,'MidpointBus',forbidden)
    assert m.main(['--config',str(base/'robot.home_actions_20261101.json')])==0
    out=json.loads(capsys.readouterr().out)
    assert out['hardware_opened'] is False and out['hardware_writes']==0


@pytest.mark.parametrize('flags',[['--move'],['--apply'],['--move','--apply']])
def test_execution_requires_all_explicit_flags(flags):
    with pytest.raises(SystemExit):m.main(flags)


def test_optional_hold_then_one_reference_snapshot_without_stability_window(rig):
    base,cfg,files,wire,bus,backup,state=rig
    home=dict(zip(m.IDS,cfg['calibration']['home_ticks']))
    held,report=m.take_home_snapshot(bus,5.)
    assert report['elapsed_s'] >= 5.
    assert report['torque_held_on'] is True
    assert report['motion_stability_checked'] is False
    assert report['reference_sample_count']==1
    assert sum(p[4]==130 for p in wire.packets)==1
    assert held==home
    assert not wire.changes()
    assert all(wire.reg[sid][40]==1 for sid in m.IDS)


def test_apply_performs_requested_hold_before_first_off_and_calibration(rig):
    base,cfg,files,wire,bus,backup,state=rig
    result=m.apply_midpoint(base,cfg,files,bus,backup,state,hold_seconds=5.)
    assert result['home_hold']['elapsed_s'] >= 5.
    assert result['home_hold']['requested_s'] == 5.
    assert result['phase']=='complete'
    assert_off(wire)


@pytest.mark.parametrize('value',['-1','31','nan','inf'])
def test_invalid_hold_durations_are_rejected_without_hardware(value):
    with pytest.raises(SystemExit):m.main(['--hold-seconds',value])


def test_post_snapshot_droop_does_not_change_midpoint_reference_or_stop_calibration(rig):
    base,cfg,files,wire,bus,backup,state=rig
    wire.drop_after_off={4:-300,7:-180,13:120}
    result=run(rig)
    assert result['phase']=='complete'
    assert result['home_hold']['reference_sample_count']==1
    # Each reference snapshot maps to 2048; the current relaxed pose may differ.
    assert result['snapshot_maps_to_ticks']==dict.fromkeys(m.IDS,2048)
    for sid,movement in wire.drop_after_off.items():
        assert result['final_ticks'][sid]==2048+movement
        assert struct.unpack_from('<H',wire.reg[sid],42)[0]==2048+movement
    assert q_to_ticks(HOME,*calibration_arrays(load_config(base/'robot.json'))).tolist()==[2048]*14
    # Exactly one sync reference before OFF; the later frame is current-goal staging.
    first_off=next(i for i,p in enumerate(wire.packets) if p[4]==3 and p[5:7]==b'\x28\x00')
    assert sum(p[4]==130 for p in wire.packets[:first_off])==1
    assert sum(p[4]==130 for p in wire.packets)==2
    assert not any(p[4]==11 for p in wire.packets)
    assert_off(wire)


def test_offset_allowlist_requires_the_exact_snapshot_value_and_off_state(rig):
    bus=rig[4]
    bus.active_id=4
    with pytest.raises(RuntimeError):bus.command(4,3,b'\x1f'+m.encode_offset(-845))
    bus.planned_offsets={4:-845}
    with pytest.raises(RuntimeError):bus.command(4,3,b'\x1f'+m.encode_offset(-844))
    for value in (-4096,4096):
        with pytest.raises(RuntimeError):m.encode_offset(value)


def test_real_robot_signed_offsets_include_valid_values_above_2047(rig):
    wire=rig[3]
    struct.pack_into('<H',wire.reg[13],31,encode(1671))
    struct.pack_into('<H',wire.reg[6],31,encode(-1397))
    run(rig)
    assert m.offset(bytes(wire.reg[13]))==2097
    assert m.offset(bytes(wire.reg[6]))==-1661
    assert_off(wire)
