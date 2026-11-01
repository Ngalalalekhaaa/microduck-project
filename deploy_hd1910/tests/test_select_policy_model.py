import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('select_policy_model', ROOT/'scripts/select_policy_model.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


@pytest.fixture
def configured_robot(tmp_path):
    from microduck_deploy.config import default_config
    cfg = default_config()
    cfg['imu']['i2c_bus'] = 3
    cfg['imu']['i2c_address'] = 0x6a
    cfg['imu_mount_verified'] = True
    cfg['calibration'] = {'pose': 'straight', 'captured_ticks': list(range(2000, 2014))}
    for i, joint in enumerate(cfg['joints']):
        joint.update(id=i+1, direction=1 if i in (4, 13) else -1,
                     zero_tick=1900.5+i, mapping_verified=True)
    (tmp_path/'robot.json').write_text(json.dumps(cfg))
    return tmp_path, cfg


def test_selection_validates_model_and_preserves_every_other_config(configured_robot, monkeypatch):
    root, original = configured_robot
    validations = []
    monkeypatch.setattr(selector, 'OnnxPolicy', validations.append)
    selector.select_model('models/hd1910_backlash_30000.onnx', 0, .25, root=root)
    changed = json.loads((root/'robot.json').read_text())
    assert validations == [root/'models/hd1910_backlash_30000.onnx']
    assert changed == {**original, 'model': 'models/hd1910_backlash_30000.onnx'}
    backups = list(root.glob('robot.json.bak.*'))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == original
    selector.select_model('models/hd1910_backlash_30000.onnx', 0, .25, root=root)
    assert len(list(root.glob('robot.json.bak.*'))) == 1


def test_absolute_model_keeps_absolute_config_path(configured_robot, monkeypatch):
    root, _ = configured_robot
    validations = []
    monkeypatch.setattr(selector, 'OnnxPolicy', validations.append)
    model = '/tmp/new model.onnx'
    selector.select_model(model, 60, .2, root=root)
    assert validations == [Path(model)]
    assert json.loads((root/'robot.json').read_text())['model'] == model


def test_incompatible_model_leaves_configuration_untouched(configured_robot, monkeypatch):
    root, original = configured_robot
    def reject(_):
        raise ValueError('incompatible HOME')
    monkeypatch.setattr(selector, 'OnnxPolicy', reject)
    with pytest.raises(ValueError, match='incompatible HOME'):
        selector.select_model('bad.onnx', 0, .25, root=root)
    assert json.loads((root/'robot.json').read_text()) == original
    assert not list(root.glob('robot.json.bak.*'))


@pytest.mark.parametrize('model,seconds,speed', [
    ('model.onnx', -1, .25), ('model.onnx', 1, .25), ('model.onnx', float('nan'), .25),
    ('model.onnx', 0, 0), ('model.onnx', 0, .5), ('model.onnx', 0, float('inf')),
    ('', 0, .25), ('model.pt', 0, .25), ('bad\nname.onnx', 0, .25),
    ('bad\0name.onnx', 0, .25),
])
def test_bad_arguments_are_rejected_before_validation_or_save(model, seconds, speed, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('invalid arguments accessed model or configuration')
    for name in ('OnnxPolicy', 'load_config', 'save_config'):
        monkeypatch.setattr(selector, name, unexpected)
    with pytest.raises(ValueError):
        selector.select_model(model, seconds, speed)


@pytest.mark.parametrize('selection_fails', [False, True])
def test_wrapper_preserves_button_workflow_and_never_runs_after_selection_failure(tmp_path, selection_fails):
    shutil.copy2(ROOT/'xbox_model_hd1910.sh', tmp_path/'xbox_model_hd1910.sh')
    (tmp_path/'.venv/bin').mkdir(parents=True)
    # Replace Python itself in this isolated copy: no policy or hardware opens.
    python = tmp_path/'.venv/bin/python'
    python.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
with Path('calls.jsonl').open('a') as output:
    output.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1] == 'scripts/select_policy_model.py' and os.environ.get('SELECT_FAIL') == '1':
    sys.exit(2)
''')
    python.chmod(0o755)
    env = {**os.environ, 'SELECT_FAIL': '1' if selection_fails else '0'}
    result = subprocess.run(['bash', str(tmp_path/'xbox_model_hd1910.sh'),
                             'models/new model.onnx'], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    assert calls[0] == ['scripts/select_policy_model.py', 'models/new model.onnx', '0', '0.3']
    if selection_fails:
        assert result.returncode == 2
        assert len(calls) == 1
    else:
        assert result.returncode == 0
        assert calls[1] == ['scripts/home_then_policy.py', '--move', '--seconds', '0',
                            '--vx', '0', '--ramp-seconds', '3', '--log-only-target-angles',
                            '--gamepad', '--gamepad-buttons', '--max-vx', '0.3',
                            '--max-vy', '0.3', '--max-wz', '1.0']
