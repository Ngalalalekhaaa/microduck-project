"""Geometry calibration is offline, explicit, and used consistently by control."""
import copy
import io
import json
from contextlib import redirect_stdout
from unittest.mock import Mock

import numpy as np
import pytest

from microduck_deploy.__main__ import main
from microduck_deploy.assembly import (assembly_template, compile_assembly,
                                       record_assembly, assembly_reference)
from microduck_deploy.config import default_config, require_calibration, save_config
from microduck_deploy.policy import HOME, ticks_to_q, q_to_ticks


def fixture():
    """Synthetic known geometry only; these values are NOT a hardware profile."""
    cfg = default_config()
    for sid, joint in enumerate(cfg['joints'], 1):
        joint.update(id=sid, direction=1, mapping_verified=True)
    profile = assembly_template(cfg)
    profile['geometry_source'] = 'synthetic unit-test geometry; not for a robot'
    for row in profile['joints']:
        row.update(base_q_deg=0.0, hole_steps=0,
                   datum='synthetic fixture with known model zero at encoder center', verified=True)
    return cfg, profile


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('steps', [-1, 1])
def test_one_mounting_step_changes_offset_by_exactly_512_counts(direction, steps):
    cfg, profile = fixture()
    cfg['joints'][4]['direction'] = direction
    profile['joints'][4].update(direction=direction, hole_steps=steps)
    record_assembly(cfg, profile)
    assert cfg['joints'][4]['zero_tick'] == 2048 - direction * steps * 512
    ticks, qref = assembly_reference(cfg)
    assert ticks.tolist() == [2048] * 14
    assert np.rad2deg(qref[4]) == steps * 45
    zero = [j['zero_tick'] for j in cfg['joints']]
    signs = [j['direction'] for j in cfg['joints']]
    np.testing.assert_allclose(ticks_to_q(ticks, zero, signs), qref, atol=1e-14)
    actual_home = ticks_to_q(cfg['calibration']['home_ticks'], zero, signs)
    np.testing.assert_allclose(actual_home, HOME, atol=np.deg2rad(360 / 4096 / 2))


def test_non_45_degree_geometry_is_preserved_not_snapped():
    cfg, profile = fixture()
    cfg['joints'][6]['direction'] = -1
    profile['joints'][6].update(direction=-1, base_q_deg=12.0, hole_steps=1)
    report = record_assembly(cfg, profile)
    row = report['joints'][6]
    assert row['reference_deg'] == 57
    assert row['zero_tick'] == pytest.approx(2048 + 57 * 4096 / 360)
    assert row['home_tick'] == round(2048 - (np.rad2deg(HOME[6]) - 57) * 4096 / 360)
    assert 'captured_ticks' not in cfg['calibration']
    assert cfg['calibration']['hardware_read'] is False


@pytest.mark.parametrize(('field', 'bad'), [
    ('base_q_deg', None), ('base_q_deg', float('nan')), ('base_q_deg', float('inf')),
    ('base_q_deg', True), ('hole_steps', None), ('hole_steps', .5), ('hole_steps', True),
    ('hole_steps', 8), ('center_tick', 0), ('center_tick', 2048.0),
    ('direction', -1), ('direction', True), ('id', 99), ('id', True),
    ('datum', ''), ('datum', None), ('verified', False), ('verified', 1),
])
def test_unknown_or_inconsistent_geometry_never_partially_changes_config(field, bad):
    cfg, profile = fixture()
    before = copy.deepcopy(cfg)
    profile['joints'][0][field] = bad
    with pytest.raises(ValueError):
        record_assembly(cfg, profile)
    assert cfg == before


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'reorder', 'unknown_field', 'pitch', 'source'])
def test_wrong_profile_structure_is_rejected(change):
    cfg, profile = fixture()
    if change == 'missing': profile['joints'].pop()
    if change == 'duplicate': profile['joints'][1] = copy.deepcopy(profile['joints'][0])
    if change == 'reorder': profile['joints'].reverse()
    if change == 'unknown_field': profile['joints'][0]['base_angle'] = 0
    if change == 'pitch': profile['hole_pitch_deg'] = 90
    if change == 'source': profile['geometry_source'] = None
    with pytest.raises(ValueError):
        compile_assembly(cfg, profile)


def test_entire_model_travel_must_fit_even_when_reference_and_home_fit():
    cfg, profile = fixture()
    # HOME yaw=0 and reference yaw=20 both fit, but -170..170 would cross zero.
    profile['joints'][7]['base_q_deg'] = 20
    with pytest.raises(ValueError, match='跨出单圈'):
        record_assembly(cfg, profile)


def test_wrong_45_degree_branch_cannot_be_hidden_with_modulo():
    cfg, profile = fixture()
    profile['joints'][0]['hole_steps'] = 7
    with pytest.raises(ValueError, match='参考角超出'):
        record_assembly(cfg, profile)


@pytest.mark.parametrize('change', ['zero', 'profile', 'mapping', 'home', 'reference', 'model_home', 'verified'])
def test_saved_calibration_is_rechecked_before_policy_use(change):
    cfg, profile = fixture()
    record_assembly(cfg, profile)
    if change == 'zero': cfg['joints'][0]['zero_tick'] += 1
    if change == 'profile': cfg['calibration']['assembly_profile']['joints'][0]['base_q_deg'] += 1
    if change == 'mapping': cfg['joints'][0]['direction'] = -1
    if change == 'home': cfg['calibration']['home_ticks'][0] += 1
    if change == 'reference': cfg['calibration']['reference_q_rad'][0] += .01
    if change == 'model_home': cfg['calibration']['model_home_rad'][0] += .01
    if change == 'verified': cfg['calibration']['assembly_profile']['joints'][0]['verified'] = False
    with pytest.raises(ValueError):
        require_calibration(cfg, imu=False)


def test_assembly_snapshot_does_not_alias_the_input_profile():
    cfg, profile = fixture()
    record_assembly(cfg, profile)
    profile['joints'][0]['base_q_deg'] = 10
    assembly_reference(cfg)
    assert cfg['calibration']['assembly_profile']['joints'][0]['base_q_deg'] == 0


def cli_paths(tmp_path):
    cfg, profile = fixture()
    config_path, profile_path = tmp_path/'robot.json', tmp_path/'assembly.json'
    save_config(config_path, cfg)
    profile_path.write_text(json.dumps(profile), encoding='utf-8')
    return cfg, config_path, profile_path


def no_hardware(monkeypatch):
    bus = Mock(side_effect=AssertionError('offline command attempted serial access'))
    monkeypatch.setattr('microduck_deploy.servo.ServoBus', bus)
    return bus


def test_cli_template_preserves_unknowns_and_refuses_overwrite(tmp_path, monkeypatch):
    cfg, config_path, _ = cli_paths(tmp_path)
    path = tmp_path/'template.json'
    bus = no_hardware(monkeypatch)
    with redirect_stdout(io.StringIO()):
        assert main(['--config', str(config_path), 'assembly-template', '--output', str(path)]) == 0
    template = json.loads(path.read_text())
    assert template['geometry_source'] is None
    assert all(j['base_q_deg'] is None and j['hole_steps'] is None and not j['verified']
               for j in template['joints'])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        main(['--config', str(config_path), 'assembly-template', '--output', str(path)])
    assert path.read_bytes() == before
    bus.assert_not_called()


def test_cli_preview_is_offline_and_apply_preserves_non_calibration_settings(tmp_path, monkeypatch):
    original, config_path, profile_path = cli_paths(tmp_path)
    report_path = tmp_path/'plan.json'
    before = config_path.read_bytes()
    bus = no_hardware(monkeypatch)
    args = ['--config', str(config_path), 'assembly-calibrate', '--profile', str(profile_path)]
    with redirect_stdout(io.StringIO()):
        assert main(args + ['--output', str(report_path)]) == 0
    assert config_path.read_bytes() == before
    assert not list(tmp_path.glob('robot.json.bak.*'))
    report = json.loads(report_path.read_text())
    assert report['hardware_read'] is False and report['motion_sent'] is False
    assert len(report['joints']) == 14
    with redirect_stdout(io.StringIO()):
        assert main(args + ['--apply']) == 0
    saved = json.loads(config_path.read_text())
    assembly_reference(saved)
    for key in set(original) - {'joints', 'calibration'}:
        assert saved[key] == original[key]
    for before_joint, after_joint in zip(original['joints'], saved['joints']):
        assert {k:v for k,v in after_joint.items() if k != 'zero_tick'} == {
            k:v for k,v in before_joint.items() if k != 'zero_tick'}
    backups = list(tmp_path.glob('robot.json.bak.*'))
    assert len(backups) == 1 and backups[0].read_bytes() == before
    assert not saved['imu_mount_verified']
    bus.assert_not_called()


@pytest.mark.parametrize('destination', ['config', 'profile'])
def test_report_cannot_overwrite_inputs_even_with_apply(tmp_path, monkeypatch, destination):
    _, config_path, profile_path = cli_paths(tmp_path)
    before_config, before_profile = config_path.read_bytes(), profile_path.read_bytes()
    bus = no_hardware(monkeypatch)
    path = config_path if destination == 'config' else profile_path
    with pytest.raises(ValueError, match='不能覆盖'):
        main(['--config', str(config_path), 'assembly-calibrate', '--profile', str(profile_path),
              '--apply', '--output', str(path)])
    assert config_path.read_bytes() == before_config and profile_path.read_bytes() == before_profile
    bus.assert_not_called()


def test_unconfirmed_geometry_fails_before_hardware_or_config_write(tmp_path, monkeypatch):
    _, config_path, profile_path = cli_paths(tmp_path)
    profile = json.loads(profile_path.read_text())
    profile['joints'][-1]['verified'] = False
    profile_path.write_text(json.dumps(profile))
    before = config_path.read_bytes()
    bus = no_hardware(monkeypatch)
    with pytest.raises(ValueError, match='尚未确认'):
        main(['--config', str(config_path), 'assembly-calibrate', '--profile', str(profile_path), '--apply'])
    assert config_path.read_bytes() == before
    assert not list(tmp_path.glob('robot.json.bak.*'))
    bus.assert_not_called()
