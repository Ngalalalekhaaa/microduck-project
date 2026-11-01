"""Action-policy warmup preserves each exported command contract before device IO."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'action_policy_trial', Path(__file__).resolve().parents[1] / 'scripts/policy_trial.py')
trial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


@pytest.fixture
def cfg():
    return {
        'sitstand_model': 'models/sit.onnx',
        'ground_pick_model': 'models/ground.onnx',
        'roulade_model': 'models/roll.onnx',
        'control': {'roulade_enabled': True},
    }


@pytest.fixture
def loader():
    policies = []

    def load(path, *, policy_kind):
        policy = SimpleNamespace(path=path, kind=policy_kind, observations=[])

        def infer(observation):
            assert observation.shape == (61,)
            assert observation.dtype == np.float32
            assert np.isfinite(observation).all()
            policy.observations.append(observation.copy())
            return np.zeros(14, dtype=np.float32)

        policy.infer = infer
        policies.append(policy)
        return policy

    result = Mock(side_effect=load)
    result.policies = policies
    return result


def assert_upright_home_observations(policy):
    for observation in policy.observations:
        np.testing.assert_array_equal(observation[:6], [0, 0, 0, 0, 0, -1])
        np.testing.assert_array_equal(observation[6:48], 0)
        np.testing.assert_array_equal(observation[51:], 0)


def test_disabled_actions_do_not_read_config_or_construct_models(loader):
    assert trial.load_action_policies({}, enabled=False, policy_loader=loader) == (None, None, None)
    loader.assert_not_called()


def test_enabled_empty_configuration_has_no_optional_models(loader):
    assert trial.load_action_policies({'control': {}}, enabled=True, policy_loader=loader) == (None, None, None)
    loader.assert_not_called()


def test_all_actions_use_explicit_policy_kinds_paths_and_distinct_command_contracts(cfg, loader):
    sit, ground, roll = trial.load_action_policies(cfg, enabled=True, policy_loader=loader)
    assert [p.kind for p in loader.policies] == ['sitstand', 'ground_pick', 'roulade']
    assert [p.path for p in loader.policies] == [trial.ROOT / cfg[key] for key in (
        'sitstand_model', 'ground_pick_model', 'roulade_model')]
    assert [sit, ground, roll] == loader.policies
    assert [len(p.observations) for p in loader.policies] == [2, 9, 3]
    np.testing.assert_array_equal(np.array(sit.observations)[:, 48:51], [[0, 0, 0], [1, 0, 0]])
    phase = np.linspace(0, 2 * np.pi, 9)
    expected = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(9)))
    np.testing.assert_allclose(np.array(ground.observations)[:, 48:51], expected, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(np.array(ground.observations)[:, 48:50], axis=1), 1)
    np.testing.assert_array_equal(np.array(roll.observations)[:, 48:51], 0)
    for policy in loader.policies:
        assert_upright_home_observations(policy)


@pytest.mark.parametrize('enabled_flag', [None, False])
def test_roulade_model_presence_does_not_enable_roulade(cfg, loader, enabled_flag):
    if enabled_flag is None:
        cfg['control'].pop('roulade_enabled')
    else:
        cfg['control']['roulade_enabled'] = enabled_flag
    sit, ground, roll = trial.load_action_policies(cfg, enabled=True, policy_loader=loader)
    assert sit is not None and ground is not None and roll is None
    assert [p.kind for p in loader.policies] == ['sitstand', 'ground_pick']


@pytest.mark.parametrize('kind, message', [
    ('ground_pick', '俯身动作需要站坐权重'),
    ('roulade', '滚翻需要站坐权重'),
])
def test_actions_that_return_to_stand_require_sitstand_weights(cfg, loader, kind, message):
    cfg['sitstand_model'] = None
    if kind == 'roulade':
        cfg['ground_pick_model'] = None
    with pytest.raises(ValueError, match=message):
        trial.load_action_policies(cfg, enabled=True, policy_loader=loader)
    loader.assert_not_called()


def test_sitstand_can_be_used_without_ground_or_roll_models(cfg, loader):
    cfg.pop('ground_pick_model')
    cfg.pop('roulade_model')
    cfg['control']['roulade_enabled'] = False
    sit, ground, roll = trial.load_action_policies(cfg, enabled=True, policy_loader=loader)
    assert sit.kind == 'sitstand' and ground is None and roll is None
    assert len(sit.observations) == 2


def test_absolute_model_paths_are_preserved_and_default_onnx_loader_is_used(cfg, loader, monkeypatch, tmp_path):
    cfg['sitstand_model'] = str(tmp_path / 'sit.onnx')
    cfg['ground_pick_model'] = None
    cfg['control']['roulade_enabled'] = False
    monkeypatch.setattr('microduck_deploy.policy.OnnxPolicy', loader)
    sit, _, _ = trial.load_action_policies(cfg, enabled=True)
    assert sit.path == tmp_path / 'sit.onnx'
    loader.assert_called_once_with(tmp_path / 'sit.onnx', policy_kind='sitstand')


def test_model_contract_or_warmup_failure_prevents_a_partially_loaded_result(cfg):
    failure = ValueError('wrong exported command contract')
    loader = Mock(return_value=SimpleNamespace(infer=Mock(side_effect=failure)))
    with pytest.raises(ValueError, match='wrong exported command contract'):
        trial.load_action_policies(cfg, enabled=True, policy_loader=loader)
    loader.assert_called_once_with(trial.ROOT / cfg['sitstand_model'], policy_kind='sitstand')
