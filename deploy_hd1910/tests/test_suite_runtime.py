"""Policy-session routing and preloading for the opt-in D-pad model suite."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import ROOT, default_config, load_config
from microduck_deploy.gamepad import GamepadStop
from microduck_deploy.runtime import StopRequested
from test_runtime import Policy, calibrated_config, rig


def suite_config():
    cfg = calibrated_config()
    cfg.update(updated_walking_model='models/updated.onnx',
               sitstand_friction_model='models/friction.onnx',
               sitstand_com_model='models/com.onnx', roulade_model='models/roll.onnx')
    cfg['control'].update(model_suite_buttons_enabled=True, roulade_enabled=True)
    return cfg


class Source:
    def __init__(self, steps):
        self.steps = list(steps)
        self.index = 0
        self.policy_mode = 'walking'
        self.walking_variant, self.sitstand_variant = 'original', 'default'

    def __call__(self):
        mode, walking, sitstand = self.steps[min(self.index, len(self.steps) - 1)]
        self.index += 1
        self.walking_variant, self.sitstand_variant = walking, sitstand
        if mode == 'B':
            raise GamepadStop('B')
        self.policy_mode = mode
        return (.3, .2, .5)

    def finish_roulade(self):
        self.policy_mode = 'stand'


def install_policies(r):
    records = []
    specifications = (
        ('walking:original', 'policy', .01),
        ('sitstand:default', 'sitstand_policy', .02),
        ('walking:updated', 'updated_walking_policy', .03),
        ('sitstand:friction', 'sitstand_friction_policy', .04),
        ('sitstand:com', 'sitstand_com_policy', .05),
        ('roulade', 'roulade_policy', .06),
    )
    for label, attr, value in specifications:
        policy = r.policy if attr == 'policy' else Policy(r.clock)
        policy.output[:] = value
        policy.model_path = Path('/mock') / (label.replace(':', '_') + '.onnx')
        original = policy.infer

        def infer(obs, *, name=label, delegate=original):
            records.append((name, obs.copy()))
            return delegate(obs)

        policy.infer = infer
        setattr(r.ctl, attr, policy)
    return records


def test_real_controller_routes_all_variants_and_preserves_every_raw_action():
    steps = [
        ('walking', 'original', 'default'),
        ('stopping', 'original', 'default'),
        ('stand', 'updated', 'default'),
        ('walking', 'updated', 'default'),
        ('stopping', 'updated', 'default'),
        ('sit', 'updated', 'friction'),
        ('stand', 'updated', 'friction'),
        ('sit', 'updated', 'com'),
        ('stand', 'updated', 'com'),
        ('walking', 'original', 'com'),
        ('B', 'bad', 'bad'),
    ]
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.5, motion=True, command_provider=Source(steps), command_ramp_s=.2)
        expected = ['walking:original', 'walking:original', 'sitstand:default',
                    'walking:updated', 'walking:updated', 'sitstand:friction',
                    'sitstand:friction', 'sitstand:com', 'sitstand:com', 'walking:original']
        assert [name for name, _ in records] == expected
        for (_, obs), previous in zip(records, [0, .01, .01, .02, .03, .03, .04, .04, .05, .05]):
            np.testing.assert_allclose(obs[34:48], previous)
        for index in (1, 2, 4, 6, 8):
            np.testing.assert_array_equal(records[index][1][48:51], 0)
        for index in (5, 7):
            np.testing.assert_array_equal(records[index][1][48:51], [1, 0, 0])
        samples = [json.loads(line) for line in r.path.read_text().splitlines()
                   if json.loads(line)['kind'] == 'sample']
        assert [row['policy_id'] for row in samples] == expected
        assert [row['walking_variant'] for row in samples] == [step[1] for step in steps[:-1]]
        assert [row['sitstand_variant'] for row in samples] == [step[2] for step in steps[:-1]]
        assert [event['to'] for event in r.ctl.session['policy_model_transitions']] == [
            'sitstand:default', 'walking:updated', 'sitstand:friction', 'sitstand:com', 'walking:original']
        assert not r.ctl.enabled
        assert any(e[0] == 'torque' and e[2] is False for e in r.bus.events)


def test_real_suite_up_command_keeps_old_gait_until_stop_then_guards_stand():
    from microduck_deploy.gamepad import DPAD_UP_BUTTON, PadSample, SuiteGamepadCommand

    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        sample = PadSample(r.clock.now, True, {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {}, ())
        holder = SimpleNamespace(sample=sample)
        command = SuiteGamepadCommand(SimpleNamespace(latest=lambda: holder.sample),
                                      clock=r.clock.monotonic, roulade_enabled=True)

        class TimedPad:
            started = None

            def __getattr__(self, name):
                return getattr(command, name)

            def __call__(self):
                if self.started is None:
                    self.started = r.clock.now
                elapsed = r.clock.now - self.started
                counts = {DPAD_UP_BUTTON: int(elapsed >= .04)}
                holder.sample = PadSample(
                    r.clock.now, True,
                    {'ABS_X': 0., 'ABS_Y': -1. if elapsed >= 3.8 else 0., 'ABS_RX': 0.}, {},
                    (305,) if elapsed >= 4. else (), button_press_counts=counts)
                return command()

        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=4.5, motion=True, command_provider=TimedPad())
        samples = [json.loads(line) for line in r.path.read_text().splitlines()
                   if json.loads(line)['kind'] == 'sample']
        stopping = [row for row in samples if row['policy_mode'] == 'stopping']
        standing = [row for row in samples if row['policy_mode'] == 'stand']
        assert stopping and {row['policy_id'] for row in stopping} == {'walking:original'}
        assert stopping[-1]['t'] - stopping[0]['t'] >= .56
        assert standing and {row['policy_id'] for row in standing} == {'sitstand:default'}
        assert standing[-1]['t'] - standing[0]['t'] >= 2.98
        assert samples[-1]['policy_id'] == 'walking:updated'
        first_updated = next(obs for name, obs in records if name == 'walking:updated')
        np.testing.assert_allclose(first_updated[34:48], .02)
        assert not r.ctl.enabled


def test_same_mode_model_switch_is_visible_in_session_and_sample_logs():
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        source = Source([('stand', 'original', 'friction'), ('stand', 'original', 'com'),
                         ('B', 'original', 'com')])
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.2, motion=True, command_provider=source)
        assert [name for name, _ in records] == ['sitstand:friction', 'sitstand:com']
        np.testing.assert_allclose(records[1][1][34:48], .04)
        assert [event['to'] for event in r.ctl.session['policy_transitions']] == ['stand']
        transitions = r.ctl.session['policy_model_transitions']
        assert [event['to'] for event in transitions] == ['sitstand:friction', 'sitstand:com']
        assert transitions[1]['model_path'] == '/mock/sitstand_com.onnx'
        saved = json.loads(r.path.with_suffix('.session.json').read_text())
        assert saved['policy_model_transitions'] == transitions


def test_selected_com_can_enter_roll_without_changing_roll_command_or_history():
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        source = Source([('stand', 'updated', 'com'), ('roulade', 'updated', 'com'),
                         ('B', 'updated', 'com')])
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.2, motion=True, command_provider=source)
        assert [name for name, _ in records] == ['sitstand:com', 'roulade']
        np.testing.assert_allclose(records[-1][1][34:48], .05)
        np.testing.assert_array_equal(records[-1][1][48:], np.zeros(13))
        assert not r.ctl.enabled


@pytest.mark.parametrize('walking,sitstand', [('unknown', 'default'), (None, 'default'),
                                            ([], 'default'), ('original', 'unknown'),
                                            ('original', None), ('original', [])])
def test_unknown_variant_stops_before_any_inference(walking, sitstand):
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        with pytest.raises(StopRequested, match='模型选择无效'):
            r.ctl.execute(seconds=.1, motion=True,
                          command_provider=Source([('walking', walking, sitstand)]))
        assert not records and not r.ctl.enabled


@pytest.mark.parametrize('attr,mode,walking,sitstand', [
    ('updated_walking_policy', 'walking', 'updated', 'default'),
    ('updated_walking_policy', 'stopping', 'updated', 'default'),
    ('sitstand_friction_policy', 'sit', 'original', 'friction'),
    ('sitstand_com_policy', 'stand', 'original', 'com'),
])
def test_missing_selected_model_stops_before_inference(attr, mode, walking, sitstand):
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        setattr(r.ctl, attr, None)
        with pytest.raises(StopRequested, match='未加载所选'):
            r.ctl.execute(seconds=.1, motion=True,
                          command_provider=Source([(mode, walking, sitstand)]))
        assert not records and not r.ctl.enabled


def test_disabled_suite_cannot_be_bypassed_by_provider_variant_attributes():
    with rig() as r:
        records = install_policies(r)
        with pytest.raises(StopRequested, match='未显式启用'):
            r.ctl.execute(seconds=.1, motion=True,
                          command_provider=Source([('walking', 'updated', 'default')]))
        assert not records and not r.ctl.enabled


@pytest.mark.parametrize('mode,walking,sitstand', [('walking', 'updated', 'default'),
                                               ('stopping', 'updated', 'default'),
                                               ('sit', 'original', 'friction'),
                                               ('stand', 'original', 'com'),
                                               ('roulade', 'updated', 'com')])
def test_b_precedes_invalid_variant_selection_in_every_suite_policy(mode, walking, sitstand):
    with rig(cfg=suite_config()) as r:
        records = install_policies(r)
        source = Source([(mode, walking, sitstand), ('B', 'invalid', 'invalid')])
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.2, motion=True, command_provider=source)
        assert len(records) == 1 and not r.ctl.enabled


def test_suite_default_is_disabled_and_old_config_with_no_new_keys_loads(tmp_path):
    cfg = calibrated_config()
    assert default_config()['control']['model_suite_buttons_enabled'] is False
    for key in ('updated_walking_model', 'sitstand_friction_model', 'sitstand_com_model'):
        assert cfg.pop(key) is None
    cfg['control'].pop('model_suite_buttons_enabled')
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    assert load_config(path)['schema'] == 1


@pytest.mark.parametrize('key', ['updated_walking_model', 'sitstand_friction_model',
                               'sitstand_com_model', 'sitstand_model', 'roulade_model'])
def test_enabled_suite_requires_every_model_path(tmp_path, key):
    cfg = suite_config()
    cfg[key] = None
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize('value', ['true', 'false', 1, None])
def test_suite_switch_requires_a_json_boolean(tmp_path, value):
    cfg = suite_config()
    cfg['control']['model_suite_buttons_enabled'] = value
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='model_suite_buttons_enabled.*布尔'):
        load_config(path)


def test_enabled_suite_also_requires_explicit_roulade_opt_in(tmp_path):
    cfg = suite_config()
    cfg['control']['roulade_enabled'] = False
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='方向键模型切换'):
        load_config(path)


@pytest.mark.parametrize('key', ['updated_walking_model', 'sitstand_friction_model', 'sitstand_com_model'])
def test_optional_suite_model_paths_must_be_onnx_even_when_disabled(tmp_path, key):
    cfg = calibrated_config()
    cfg[key] = 'model.pt'
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=key):
        load_config(path)


def trial_module():
    spec = importlib.util.spec_from_file_location('suite_policy_trial', ROOT / 'scripts/policy_trial.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_preloads_correct_contracts_and_warms_all_commands_without_devices():
    calls = []

    class Session:
        def __init__(self, path, policy_kind):
            self.path, self.kind, self.observations = path, policy_kind, []
            calls.append(self)

        def infer(self, obs):
            self.observations.append(obs.copy())
            return np.zeros(14, dtype=np.float32)

    sessions = trial_module().load_suite_policies(suite_config(), gamepad=True, policy_loader=Session)
    assert list(sessions) == calls
    assert [p.kind for p in calls] == ['walking', 'sitstand', 'sitstand']
    assert [p.path.name for p in calls] == ['updated.onnx', 'friction.onnx', 'com.onnx']
    np.testing.assert_array_equal(np.array(calls[0].observations)[:, 48:], np.zeros((3, 13)))
    for session in calls[1:]:
        np.testing.assert_array_equal(np.array(session.observations)[:, 48:51], [[0, 0, 0], [1, 0, 0]])
        np.testing.assert_array_equal(np.array(session.observations)[:, 34:48], np.zeros((2, 14)))


@pytest.mark.parametrize('gamepad,enabled', [(False, True), (True, False), (False, False)])
def test_suite_preloader_is_inactive_for_existing_or_non_gamepad_entry(gamepad, enabled):
    cfg = suite_config()
    cfg['control']['model_suite_buttons_enabled'] = enabled

    def unexpected_load(*args, **kwargs):
        raise AssertionError('optional policy loaded while disabled')

    assert trial_module().load_suite_policies(cfg, gamepad=gamepad, policy_loader=unexpected_load) == (None, None, None)
