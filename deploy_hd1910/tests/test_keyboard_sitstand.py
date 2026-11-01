"""PTY model selection and real sit/stand routing without hardware access."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from microduck_deploy.keyboard_sitstand import load_sitstand_policies, sitstand_options
from microduck_deploy.runtime import StopRequested
from test_runtime import Policy, calibrated_config, rig
from test_terminal_actions import terminal, ready, tick, advance
from test_terminal_action_runtime import attach_policies, on_infer, assert_unloaded, samples

MODELS = {key: 'sitstand-' + key for key in 'ghimn'}
WALKING = {key: 'walking-' + key for key in 'abcdef'}
OPTIONS = {'sitstand_models': MODELS, 'default_sitstand_key': 'n', 'single_key': True}


@pytest.mark.parametrize('key', list(MODELS))
def test_letter_only_selects_and_seven_toggles_both_postures(key):
    with terminal(**OPTIONS) as t:
        ready(t)
        tick(t, line='j')
        assert advance(t, 1.2)[0] == pytest.approx(.15)
        t.write(key.upper().encode())
        assert tick(t) == (0., 0., 0.)
        assert t.command.sitstand_model_key == key
        assert t.command.policy_mode == 'walking'
        tick(t, line='7')
        assert t.command.policy_mode == 'stopping'
        tick(t, .61)
        assert t.command.policy_mode == 'sit'
        tick(t, line='7')
        assert t.command.policy_mode == 'stand'
        assert t.command.sitstand_model_key == key


def test_seated_selection_never_switches_under_load_and_uses_current_model_to_rise():
    with terminal(**OPTIONS) as t:
        ready(t)
        tick(t, line='g7')
        tick(t, .61)
        assert t.command.policy_mode == 'sit'
        tick(t, line='h')
        assert t.command.sitstand_model_key == 'g'
        assert t.command.policy_mode == 'sit'
        tick(t, line='g')
        assert t.command.policy_mode == 'sit'
        tick(t, line='7')
        assert t.command.sitstand_model_key == 'g'
        tick(t, line='i')
        assert t.command.sitstand_model_key == 'g'
        tick(t, 3.1)
        tick(t, line='i')
        assert t.command.sitstand_model_key == 'i'
        assert t.command.policy_mode == 'stand'
        tick(t, line='7')  # Standing policy changed; retain its adaptation guard.
        assert t.command.policy_mode == 'stand'
        tick(t, 3.1)
        tick(t, line='7')
        assert t.command.policy_mode == 'sit'


@pytest.mark.parametrize('action,mode', [('9', 'ground_pick'), ('r', 'roulade')])
def test_letters_during_actions_are_consumed_without_queueing(action, mode):
    with terminal(**OPTIONS) as t:
        ready(t)
        tick(t, line=action)
        tick(t, line='g')
        tick(t, .61)
        assert t.command.policy_mode == mode
        tick(t, line='h')
        assert t.command.sitstand_model_key == 'n'
        if mode == 'roulade':
            t.command.finish_roulade()
        else:
            tick(t, 4.01)
        tick(t, 3.1)
        assert t.command.sitstand_model_key == 'n'


def test_walking_switch_speed_keys_and_sitstand_selection_are_independent():
    with terminal(**OPTIONS, walking_models=WALKING) as t:
        ready(t)
        tick(t, line='g')
        tick(t, line='b')
        tick(t, line='h')
        assert t.command.sitstand_model_key == 'g'
        tick(t, .61)
        assert t.command.walking_model_key == 'b'
        assert t.command.policy_mode == 'walking'
        for key, speed in (('j', .15), ('k', .20), ('l', .25), ('1', .30)):
            tick(t, line=key)
            assert advance(t, 1.2)[0] == pytest.approx(speed)
        assert t.command.walking_model_key == 'b' and t.command.sitstand_model_key == 'g'


def make_cfg(tmp_path):
    options = {}
    for key in MODELS:
        p = tmp_path / (key + '.onnx')
        p.write_bytes(key.encode())
        options[key] = {'label': MODELS[key], 'model': p.name,
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    return {'sitstand_model': 'n.onnx', 'keyboard_sitstand_default': 'n',
            'keyboard_sitstand_models': options}


def test_preload_all_five_with_sitstand_contract_and_reuse_default(tmp_path):
    cfg = make_cfg(tmp_path)
    created = []

    class Session:
        def __init__(self, path, *, policy_kind):
            assert policy_kind == 'sitstand'
            self.path, self.observations = path, []
            created.append(self)

        def infer(self, obs):
            self.observations.append(obs.copy())
            return np.zeros(14)

    base = Session(tmp_path / 'n.onnx', policy_kind='sitstand')
    sessions = load_sitstand_policies(cfg, tmp_path, base_policy=base, policy_loader=Session)
    assert len(created) == 5 and sessions['n'] is base
    for session in created:
        obs = np.array(session.observations)
        assert obs.shape == (2, 61)
        np.testing.assert_array_equal(obs[:, 6:48], 0)
        np.testing.assert_array_equal(obs[:, 48:51], [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(obs[:, 51:], 0)


def test_hash_failure_before_any_session_is_constructed(tmp_path):
    cfg = make_cfg(tmp_path)
    (tmp_path / 'n.onnx').write_bytes(b'corrupt')

    def never(*args, **kwargs):
        raise AssertionError('should validate whole set before loading')

    with pytest.raises(ValueError, match='坐站模型 n 校验失败'):
        load_sitstand_policies(cfg, tmp_path, policy_loader=never)


@pytest.mark.parametrize('mutate', [
    lambda c: c['keyboard_sitstand_models'].pop('g'),
    lambda c: c['keyboard_sitstand_models']['h'].update(model='h.pt'),
    lambda c: c['keyboard_sitstand_models']['i'].update(sha256='bad'),
    lambda c: c.update(keyboard_sitstand_default='x'),
    lambda c: c.update(sitstand_model='wrong.onnx'),
])
def test_invalid_mapping_fails_before_hardware(tmp_path, mutate):
    cfg = make_cfg(tmp_path)
    mutate(cfg)
    with pytest.raises(ValueError):
        sitstand_options(cfg)


@pytest.mark.parametrize('key', list(MODELS))
def test_real_controller_routes_selected_model_for_both_7_commands_and_preserves_history(key):
    cfg = calibrated_config()
    cfg['keyboard_sitstand_models'] = {k: {'model': k + '.onnx'} for k in MODELS}
    with rig(cfg=cfg) as r, terminal(clock=r.clock.monotonic, **OPTIONS) as t:
        attach_policies(r)
        policies = {}
        for index, k in enumerate(MODELS):
            p = Policy(r.clock)
            p.output[:] = .02 + index * .01
            p.model_path = Path('/models') / (k + '.onnx')
            policies[k] = p
        r.ctl.keyboard_sitstand_policies = policies
        selected = policies[key]
        on_infer(r.policy, lambda obs, count: t.write((key + '7').encode()) if count == 1 else None)

        def sitstand(obs, count):
            if count == 1:
                t.write(b'7')
            elif count == 3:
                t.write(b'q')

        on_infer(selected, sitstand)
        with pytest.raises(StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=3, motion=True, command_provider=t.command)
        obs = np.array(selected.observations)
        np.testing.assert_array_equal(obs[:, 48:51], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(obs[0, 34:48], .01)
        np.testing.assert_allclose(obs[1, 34:48], selected.output)
        assert all(not p.observations for k, p in policies.items() if k != key)
        events = r.ctl.session['policy_model_transitions']
        assert events[-1]['to'] == 'sitstand:keyboard_' + key
        assert events[-1]['model_path'] == '/models/' + key + '.onnx'
        assert samples(r)[-1]['policy_mode'] == 'stand'
        assert_unloaded(r)
