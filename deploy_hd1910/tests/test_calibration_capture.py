"""Rejected calibration retains observations without changing software/hardware state."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy.__main__ import main
from microduck_deploy.calibration_capture import capture_report
from microduck_deploy.config import default_config, record_reference, save_config
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks


BODY_IDS = list(range(1, 10)) + list(range(11, 16))


def config():
    cfg = default_config()
    for j, sid in zip(cfg['joints'], BODY_IDS):
        j.update(id=sid, direction=1 if sid == 15 else -1, mapping_verified=True)
    return cfg


class ReadOnlyBus:
    """No hardware-write methods exist, so CLI mutation attempts fail the test."""
    def __init__(self, samples):
        self.samples = np.asarray(samples)
        self.calls = 0
        self.configuration_ids = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read_configuration(self, sid):
        self.configuration_ids.append(sid)
        return {'torque': 0, 'angle_resolution': 1}

    def read_feedback_many(self, ids, *, sync):
        assert ids == BODY_IDS and sync is True
        values = self.samples[self.calls]
        self.calls += 1
        return {sid: SimpleNamespace(position_ticks=int(tick)) for sid, tick in zip(ids, values)}


def run_capture(path, cfg, samples, *, sample_count=None):
    bus = ReadOnlyBus(samples)
    stream = io.StringIO()
    argv = ['--config', str(path), 'calibrate', '--pose', 'straight']
    if sample_count is not None:
        argv.extend(['--samples', str(sample_count)])
    with patch('microduck_deploy.__main__.load_config', return_value=cfg), \
         patch('microduck_deploy.servo.ServoBus', return_value=bus), \
         patch('microduck_deploy.__main__.time.sleep') as sleep, redirect_stdout(stream):
        try:
            result = main(argv)
        finally:
            assert bus.calls == (12 if sample_count is None else sample_count)
            assert bus.configuration_ids == BODY_IDS
            assert sleep.call_count == (0 if sample_count == 1 else 12)
    return result, stream.getvalue()


def log(path):
    files = list((path.parent/'logs').glob('calibration_capture_*.json'))
    assert len(files) == 1
    return files[0], json.loads(files[0].read_text())


@pytest.mark.parametrize('bad_tick,expected_fragment', [(0, 'HOME未取整刻度'), (-1, '参考刻度-1')])
def test_failed_capture_preserves_every_sample_and_config(tmp_path, bad_tick, expected_fragment):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    before_file, before_cfg = path.read_bytes(), deepcopy(cfg)
    samples = np.full((12, 14), 2048)
    samples[:, -1] = bad_tick
    with pytest.raises(ValueError) as raised:
        run_capture(path, cfg, samples)
    saved_path, captured = log(path)
    assert str(saved_path) in str(raised.value)
    assert 'ID15 right_ankle' in str(raised.value)
    assert expected_fragment in str(raised.value)
    assert captured['raw_samples'] == samples.tolist()
    assert captured['joint_ids'] == BODY_IDS and 10 not in captured['joint_ids']
    assert captured['rounded_mean_ticks'] == samples[0].tolist()
    assert captured['spans_ticks'] == [0]*14
    assert captured['sample_count'] == 12
    assert captured['sample_mode'] == 'stable_average'
    assert captured['stability_checked'] is True
    assert captured['joints'][-1]['issues']
    assert captured['utc'].endswith('+00:00')
    assert cfg == before_cfg and path.read_bytes() == before_file
    assert not list(tmp_path.glob('robot.json.bak.*'))


def test_unstable_capture_is_retained_before_rejection(tmp_path):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    before = path.read_bytes()
    samples = np.full((12, 14), 2048)
    samples[-1, 0] += 9
    with pytest.raises(ValueError, match='ID1 left_hip_yaw'):
        run_capture(path, cfg, samples)
    _, captured = log(path)
    assert captured['raw_samples'] == samples.tolist()
    assert captured['spans_ticks'][0] == 9
    assert path.read_bytes() == before


def test_success_logs_first_and_continues_standard_software_calibration(tmp_path):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    samples = np.tile(np.arange(14)+2048, (12, 1))
    samples[0, 0] += 1
    actual_save = save_config
    def verify_log_before_save(destination, value):
        _, captured = log(path)
        assert captured['raw_samples'] == samples.tolist()
        return actual_save(destination, value)
    with patch('microduck_deploy.__main__.save_config', side_effect=verify_log_before_save):
        result, output = run_capture(path, cfg, samples)
    assert result == 0 and '原始标定采样已保存' in output
    saved_path, captured = log(path)
    saved = json.loads(path.read_text())
    assert saved['calibration']['captured_ticks'] == captured['rounded_mean_ticks']
    assert saved['calibration']['joint_ids'] == BODY_IDS
    np.testing.assert_array_equal(saved['calibration']['home_ticks'],
                                  np.rint(captured['home_ticks_unrounded']).astype(int))
    assert len(list(tmp_path.glob('robot.json.bak.*'))) == 1
    assert all(not row['issues'] for row in captured['joints'])


@pytest.mark.parametrize('direction', [-1, 1])
def test_report_uses_signed_reference_to_home_delta_without_mutating_config(direction):
    cfg = config()
    for joint in cfg['joints']:
        joint['direction'] = direction
    original = deepcopy(cfg)
    samples = np.full((12, 14), 2048)
    report = capture_report(cfg, samples, 'straight')
    expected = 2048 + direction*(HOME-STRAIGHT_REFERENCE)*4096/(2*np.pi)
    np.testing.assert_allclose(report['home_ticks_unrounded'], expected, rtol=0, atol=1e-12)
    assert cfg == original
    assert report['directions'] == [direction]*14


def test_candidate_zero_outside_single_turn_is_allowed_if_reference_and_home_fit(tmp_path):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    samples = np.full((12, 14), 2048)
    samples[:, 2] = 500   # Left hip-pitch software zero is negative; HOME still fits.
    samples[:, 11] = 3700 # Right hip-pitch zero exceeds 4095; HOME still fits.
    result, _ = run_capture(path, cfg, samples)
    assert result == 0
    _, captured = log(path)
    assert captured['candidate_zero_ticks'][2] < 0
    assert captured['candidate_zero_ticks'][11] > 4095
    assert captured['joints'][2]['issues'] == []
    assert captured['joints'][11]['issues'] == []
    saved = json.loads(path.read_text())
    zero = [j['zero_tick'] for j in saved['joints']]
    direction = [j['direction'] for j in saved['joints']]
    assert all(0 <= tick <= 4095 for tick in q_to_ticks(HOME, zero, direction))


def test_remote_patch_fixture_targets_only_the_calibration_subsection():
    root = Path(__file__).parents[1]
    patch_spec = json.loads((Path(__file__).parent/'fixtures/calibration_capture_cli_patch.json').read_text())
    source = (root/'microduck_deploy/__main__.py').read_text()
    single_spec = json.loads((Path(__file__).parent/'fixtures/calibration_single_snapshot_cli_patch.json').read_text())
    for replacement in reversed(single_spec['replacements']):
        assert source.count(replacement['new']) == 1
        source = source.replace(replacement['new'], replacement['old'])
    assert source.count(patch_spec['new']) == 1
    assert source.count(patch_spec['old']) == 0
    original = source.replace(patch_spec['new'], patch_spec['old'])
    assert original.count(patch_spec['old']) == 1
    assert original.replace(patch_spec['old'], patch_spec['new']) == source
    assert 'save_config(args.config,cfg)' in patch_spec['old']
    assert "elif args.command == 'calibrate':" not in patch_spec['old']


def test_single_snapshot_reads_once_without_stability_claim(tmp_path):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    # If the CLI reads a second time it encounters a different pose, and the
    # interface-call assertion fails. Only the first observation is requested.
    samples = np.tile(np.arange(14)+2048, (12, 1))
    samples[1:, 0] += 87
    result, _ = run_capture(path, cfg, samples, sample_count=1)
    assert result == 0
    _, captured = log(path)
    assert captured['sample_count'] == 1
    assert captured['sample_mode'] == 'single_snapshot'
    assert captured['stability_checked'] is False
    assert captured['raw_samples'] == [samples[0].tolist()]
    assert captured['rounded_mean_ticks'] == samples[0].tolist()
    assert '不保证14关节严格同时采样' in captured['note']
    assert '未检查姿态稳定性' in captured['note']
    saved = json.loads(path.read_text())
    assert saved['calibration']['captured_ticks'] == samples[0].tolist()
    assert saved['calibration']['joint_ids'] == BODY_IDS


@pytest.mark.parametrize('bad_tick,fragment', [(0, 'HOME未取整刻度'), (5432, '参考刻度5432')])
def test_single_snapshot_still_rejects_reference_and_home_bounds(tmp_path, bad_tick, fragment):
    cfg = config()
    original = deepcopy(cfg)
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    before = path.read_bytes()
    samples = np.full((1, 14), 2048)
    samples[0, -1] = bad_tick
    with pytest.raises(ValueError, match=fragment):
        run_capture(path, cfg, samples, sample_count=1)
    _, captured = log(path)
    assert captured['sample_count'] == 1
    assert captured['raw_samples'] == samples.tolist()
    assert captured['joints'][-1]['issues']
    assert cfg == original and path.read_bytes() == before
    assert not list(tmp_path.glob('robot.json.bak.*'))


def test_explicit_twelve_samples_keeps_stability_rejection(tmp_path):
    cfg = config()
    path = tmp_path/'robot.json'
    save_config(path, cfg)
    before = path.read_bytes()
    samples = np.full((12, 14), 2048)
    samples[-1, 6] += 87
    with pytest.raises(ValueError, match='ID7 head_pitch'):
        run_capture(path, cfg, samples, sample_count=12)
    _, captured = log(path)
    assert captured['sample_count'] == 12
    assert captured['spans_ticks'][6] == 87
    assert captured['stability_checked'] is True
    assert path.read_bytes() == before


@pytest.mark.parametrize('sample_count', ['0', '2', '11', '13', '-1', '1.0', 'bad'])
def test_invalid_cli_sample_counts_fail_before_hardware(sample_count):
    with patch('microduck_deploy.servo.ServoBus') as factory, \
         pytest.raises(SystemExit) as raised:
        main(['calibrate', '--pose', 'straight', '--samples', sample_count])
    assert raised.value.code == 2
    factory.assert_not_called()


@pytest.mark.parametrize('shape', [(0, 14), (2, 14), (11, 14), (13, 14), (1, 13), (14,)])
def test_report_rejects_unsupported_sample_shapes(shape):
    with pytest.raises(ValueError, match='1或12组'):
        capture_report(config(), np.full(shape, 2048), 'straight')


def test_single_snapshot_exact_remote_patch_roundtrip():
    source = (Path(__file__).parents[1]/'microduck_deploy/__main__.py').read_text()
    spec = json.loads((Path(__file__).parent/'fixtures/calibration_single_snapshot_cli_patch.json').read_text())
    assert spec['expected_source_sha256'] == '190982c00726f4c1d794d9aba65a333e09ac7d1b6206a6dbf3e97a75aca08756'
    before = source
    for replacement in reversed(spec['replacements']):
        assert before.count(replacement['new']) == 1
        before = before.replace(replacement['new'], replacement['old'])
    restored = before
    for replacement in spec['replacements']:
        assert restored.count(replacement['old']) == 1
        restored = restored.replace(replacement['old'], replacement['new'])
    assert restored == source
