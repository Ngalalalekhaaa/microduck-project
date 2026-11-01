"""Immediate model selection, preloading and real controller routing without IO."""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from microduck_deploy.keyboard_walking import load_walking_policies, walking_options
from microduck_deploy.runtime import StopRequested
from test_runtime import Policy, calibrated_config, rig
from test_terminal_actions import terminal, ready, tick, advance
from test_terminal_action_runtime import attach_policies, samples, assert_unloaded


MODELS = {key: 'walking-' + key for key in 'abcdef'}


@pytest.mark.parametrize('key', list('bcdef'))
def test_switch_waits_on_previous_model_then_starts_new_model_at_zero(key):
    with terminal(single_key=True, walking_models=MODELS) as t:
        ready(t)
        tick(t, line='1')
        assert advance(t, 1.2)[0] > .2
        t.write(key.upper().encode())
        assert tick(t) == (0., 0., 0.)
        assert t.command.policy_mode == 'stopping'
        assert t.command.walking_model_key == 'a'
        tick(t, .3, line='17r9c')  # Busy input must not queue motion/actions.
        assert t.command.walking_model_key == 'a'
        tick(t, .31)
        assert t.command.walking_model_key == key
        assert t.command.policy_mode == 'walking'
        assert advance(t, 1.) == (0., 0., 0.)
        tick(t, line='1')
        assert advance(t, 1.2)[0] > .2


def test_initial_a_same_key_stops_velocity_and_does_not_sit():
    with terminal(single_key=True, walking_models=MODELS) as t:
        ready(t)
        tick(t, line='1')
        advance(t, 1.2)
        assert tick(t, line='a') == (0., 0., 0.)
        assert t.command.walking_model_key == 'a'
        assert t.command.policy_mode == 'walking'


@pytest.mark.parametrize('action,mode', [('7', 'sit'), ('9', 'ground_pick'), ('r', 'roulade')])
def test_model_keys_do_not_interrupt_sitting_or_existing_actions(action, mode):
    with terminal(single_key=True, walking_models=MODELS) as t:
        ready(t)
        tick(t, line=action)
        tick(t, .61)
        assert t.command.policy_mode == mode
        tick(t, line='bcdef')
        assert t.command.policy_mode == mode
        assert t.command.walking_model_key == 'a'
        if action == '7':
            tick(t, line='7')
            assert t.command.policy_mode == 'stand'
            tick(t, line='c')
            assert t.command.walking_model_key == 'a'
            tick(t, 3.1)
            tick(t, line='b')
            tick(t, .61)
            assert t.command.policy_mode == 'walking'
            assert t.command.walking_model_key == 'b'


def test_quit_during_model_switch_is_immediate():
    with terminal(single_key=True, walking_models=MODELS) as t:
        ready(t)
        tick(t, line='b')
        t.write(b'q')
        assert '终端请求退出' in t.command.check_stop()
        assert not t.command._active


def configuration(tmp_path):
    options = {}
    for key in MODELS:
        path = tmp_path / (key + '.onnx')
        path.write_bytes(key.encode())
        options[key] = {'label': MODELS[key], 'model': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return {'model': 'a.onnx', 'keyboard_walking_models': options}


def test_preload_every_session_with_walking_contract_and_reuse_base(tmp_path):
    cfg = configuration(tmp_path)
    loaded = []

    class Session:
        def __init__(self, path, *, policy_kind):
            assert policy_kind == 'walking'
            self.path, self.observations = path, []
            loaded.append(self)

        def infer(self, obs):
            self.observations.append(obs.copy())
            return np.zeros(14)

    base = Session(tmp_path / 'a.onnx', policy_kind='walking')
    sessions = load_walking_policies(cfg, tmp_path, base_policy=base, policy_loader=Session)
    assert sessions['a'] is base
    assert len(loaded) == 6 and set(sessions) == set('abcdef')
    for session in loaded:
        obs = np.array(session.observations)
        assert obs.shape == (3, 61)
        np.testing.assert_array_equal(obs[:, 6:48], 0)
        np.testing.assert_array_equal(obs[:, 51:], 0)
        np.testing.assert_allclose(obs[:, 48:51], [[0, 0, 0], [.3, 0, 0], [0, 0, .5]])


def test_corrupt_last_model_fails_before_constructing_any_session(tmp_path):
    cfg = configuration(tmp_path)
    (tmp_path / 'f.onnx').write_bytes(b'changed')

    def never(*args, **kwargs):
        raise AssertionError('session constructed before validating all models')

    with pytest.raises(ValueError, match='行走模型 f 校验失败'):
        load_walking_policies(cfg, tmp_path, policy_loader=never)


@pytest.mark.parametrize('change', [
    lambda c: c['keyboard_walking_models'].pop('f'),
    lambda c: c['keyboard_walking_models']['b'].update(model='b.pt'),
    lambda c: c['keyboard_walking_models']['c'].update(sha256='bad'),
    lambda c: c.update(model='other.onnx'),
])
def test_malformed_mapping_is_rejected(tmp_path, change):
    cfg = configuration(tmp_path)
    change(cfg)
    with pytest.raises(ValueError):
        walking_options(cfg)


def test_real_controller_switches_both_directions_and_keeps_previous_raw_actions():
    cfg = calibrated_config()
    cfg['keyboard_walking_models'] = {key: {'model': key + '.onnx'} for key in MODELS}
    with rig(cfg=cfg) as r, terminal(clock=r.clock.monotonic, single_key=True,
                                   walking_models=MODELS) as t:
        attach_policies(r)
        policies = {}
        records = []
        for index, key in enumerate(MODELS):
            p = r.policy if key == 'a' else Policy(r.clock)
            p.output[:] = .01 * (index + 1)
            p.model_path = Path('/models') / (key + '.onnx')
            original = p.infer

            def infer(obs, *, delegate=original, selected=key):
                records.append((selected, obs.copy()))
                value = delegate(obs)
                count = len(delegate.__self__.observations)
                if selected == 'a' and count == 1:
                    t.write(b'f')
                elif selected == 'f' and count == 1:
                    t.write(b'a')
                elif selected == 'a' and t.command.walking_model_key == 'a' and any(k == 'f' for k, _ in records):
                    t.write(b'q')
                return value

            p.infer = infer
            policies[key] = p
        r.ctl.keyboard_walking_policies = policies
        with pytest.raises(StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=3, motion=True, command_provider=t.command)
        assert [row[0] for row in records].count('f') >= 25
        for (previous_key, _), (_, obs) in zip(records, records[1:]):
            np.testing.assert_allclose(obs[34:48], policies[previous_key].output)
        np.testing.assert_array_equal(np.array([o[48:51] for _, o in records]), 0)
        transitions = r.ctl.session['policy_model_transitions']
        assert [row['to'] for row in transitions] == ['walking:keyboard_a', 'walking:keyboard_f', 'walking:keyboard_a']
        assert transitions[1]['model_path'] == '/models/f.onnx'
        assert {row['policy_id'] for row in samples(r)} == {'walking:keyboard_a', 'walking:keyboard_f'}
        assert_unloaded(r)
