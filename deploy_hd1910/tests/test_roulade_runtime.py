"""Hardware-free one-shot, interruption, and measured roll recovery tests."""
import dataclasses
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from microduck_deploy.config import default_config, load_config
from microduck_deploy.gamepad import (
    GamepadStop, PadSample, RouladeGamepadCommand, ROULADE_BUTTON,
    GROUND_PICK_BUTTON, SITSTAND_BUTTON,
)
from microduck_deploy.policy import HOME, JOINT_LIMITS, PolicyContractError, validate_metadata
from microduck_deploy.runtime import RouladeMonitor, StopRequested, check_posture
from test_runtime import Policy, calibrated_config, rig
from test_sitstand_policy import sitstand_metadata


ZERO = (0., 0., 0.)


class Pad:
    def __init__(self, **options):
        self.now = 10.
        self.sample = PadSample(self.now, True, {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {}, ())
        self.command = RouladeGamepadCommand(
            SimpleNamespace(latest=lambda: self.sample), clock=lambda: self.now, **options)

    def tick(self, dt=.02, **changes):
        self.now += dt
        changes.setdefault('timestamp', self.now)
        self.sample = dataclasses.replace(self.sample, **changes)
        return self.command()

    def press(self, button, **changes):
        counts = dict(self.sample.button_press_counts)
        counts[button] = counts.get(button, 0) + 1
        return self.tick(buttons=(button,), button_press_counts=counts, **changes)

    def start(self):
        self.tick()
        assert self.press(ROULADE_BUTTON) == ZERO
        assert self.command.policy_mode == 'stopping'
        assert self.tick(.61) == ZERO
        assert self.command.policy_mode == 'roulade'


def state(angle=0., gyro=(0., 0., 0.)):
    return SimpleNamespace(projected_gravity=np.array([math.sin(angle), 0., -math.cos(angle)]),
                           base_ang_vel=np.array(gyro, dtype=float))


def enabled_config():
    cfg = calibrated_config()
    cfg['roulade_model'] = 'models/roll.onnx'
    cfg['control']['roulade_enabled'] = True
    return cfg


def test_roll_is_opt_in_in_config_and_gamepad():
    assert default_config()['control']['roulade_enabled'] is False
    pad = Pad()
    pad.tick()
    pad.press(ROULADE_BUTTON)
    pad.tick(10)
    assert pad.command.policy_mode == 'walking'


@pytest.mark.parametrize('changes', [
    {'roulade_model': None}, {'sitstand_model': None},
    {'roulade_model': 'model.pt'},
    {'control.roulade_enabled': 'true'},
    {'control.roulade_timeout_s': 0}, {'control.roulade_timeout_s': 11},
    {'control.roulade_lateral_limit_deg': 180},
    {'control.roulade_max_angular_speed_rad_s': float('nan')},
])
def test_roll_config_rejects_missing_contract_and_unbounded_protection(tmp_path, changes):
    cfg = enabled_config()
    for key, value in changes.items():
        if key.startswith('control.'):
            cfg['control'][key.split('.')[1]] = value
        else:
            cfg[key] = value
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_old_config_without_roll_keys_remains_loadable(tmp_path):
    cfg = calibrated_config()
    cfg.pop('roulade_model')
    cfg['control'].pop('roulade_enabled')
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    assert load_config(path)['schema'] == 1


@pytest.mark.parametrize('value', ['false', 'true', 0, 1, None, [], {}])
def test_target_log_only_cannot_be_enabled_by_non_boolean_json(tmp_path, value):
    cfg = calibrated_config()
    cfg['control']['log_only_target_angles'] = value
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='log_only_target_angles.*布尔'):
        load_config(path)


@pytest.mark.parametrize('value', [False, True])
def test_explicit_boolean_target_log_mode_stays_compatible(tmp_path, value):
    cfg = calibrated_config()
    cfg['control']['log_only_target_angles'] = value
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    assert load_config(path)['control']['log_only_target_angles'] is value


def test_roll_does_not_confuse_ground_pick_with_identical_tensor_shapes():
    metadata = {**sitstand_metadata(), 'command_names': 'twist',
                'run_path': 'logs/microduck_roulade_hd1910/model_1600.pt'}
    validate_metadata(metadata, policy_kind='roulade')
    for run_path in ('', 'logs/ground_pick_hd1910/model_2249.pt'):
        with pytest.raises(PolicyContractError, match='run_path'):
            validate_metadata({**metadata, 'run_path': run_path}, policy_kind='roulade')


def test_lb_stops_rolls_once_then_runtime_returns_to_stand_with_rise_guard():
    pad = Pad(roulade_enabled=True)
    pad.start()
    for button in (ROULADE_BUTTON, SITSTAND_BUTTON, GROUND_PICK_BUTTON):
        pad.press(button)
        assert pad.command.policy_mode == 'roulade'
    pad.command.finish_roulade()
    assert pad.command.policy_mode == 'stand'
    pad.press(ROULADE_BUTTON, axes={'ABS_X': 0., 'ABS_Y': -1., 'ABS_RX': 0.})
    pad.tick(2.9)
    assert pad.command.policy_mode == 'stand'
    pad.tick(.2)
    assert pad.command.policy_mode == 'walking'
    # Held LB never queues or restarts a roll when the guard ends.
    pad.tick(.1)
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=())
    pad.press(ROULADE_BUTTON)
    assert pad.command.policy_mode == 'stopping'


def test_held_startup_lb_and_presses_during_guard_are_consumed():
    pad = Pad(roulade_enabled=True)
    pad.press(ROULADE_BUTTON)
    assert pad.command.policy_mode == 'walking'
    pad.tick()
    assert pad.command.policy_mode == 'walking'
    pad.tick(buttons=())
    pad.press(ROULADE_BUTTON)
    assert pad.command.policy_mode == 'stopping'


@pytest.mark.parametrize('first', [SITSTAND_BUTTON, GROUND_PICK_BUTTON])
def test_y_and_enabled_rb_win_simultaneous_lb(first):
    pad = Pad(roulade_enabled=True, ground_pick_enabled=True)
    pad.tick()
    pad.tick(button_press_counts={ROULADE_BUTTON: 1, first: 1})
    assert pad.command.policy_mode == 'stopping'
    pad.tick(.61)
    assert pad.command.policy_mode == ('sit' if first == SITSTAND_BUTTON else 'ground_pick')


def test_rb_disabled_if_no_ground_model_is_loaded():
    pad = Pad(roulade_enabled=True, ground_pick_enabled=False)
    pad.tick()
    pad.press(GROUND_PICK_BUTTON)
    assert pad.command.policy_mode == 'walking'


@pytest.mark.parametrize('pending', [True, False])
@pytest.mark.parametrize('fault', ['B', 'disconnect', 'stale', 'future'])
def test_roll_interruption_takes_priority_over_timers_and_buttons(pending, fault):
    pad = Pad(roulade_enabled=True)
    pad.tick()
    pad.press(ROULADE_BUTTON)
    if not pending:
        pad.tick(.61)
    changes = {'buttons': (305,)} if fault == 'B' else (
        {'connected': False} if fault == 'disconnect' else
        {'timestamp': pad.now - 1. if fault == 'stale' else pad.now + 1.})
    with pytest.raises(GamepadStop):
        pad.tick(**changes)


def test_roll_timeout_never_assumes_success_or_restarts():
    pad = Pad(roulade_enabled=True)
    pad.start()
    with pytest.raises(GamepadStop, match='超时'):
        pad.tick(5.01)
    assert pad.command.policy_mode == 'roulade'


def test_roll_posture_checks_allow_sagittal_inversion_but_keep_other_bounds():
    cfg = enabled_config()
    assert check_posture(HOME, state(math.pi), cfg, policy_mode='roulade') == pytest.approx(180)
    for mode in ('walking', 'stand', 'ground_pick'):
        with pytest.raises(StopRequested, match='倾斜'):
            check_posture(HOME, state(math.pi), cfg, policy_mode=mode)
    sideways = state()
    sideways.projected_gravity = [0., math.sin(math.radians(36)), -math.cos(math.radians(36))]
    with pytest.raises(StopRequested, match='横向'):
        check_posture(HOME, sideways, cfg, policy_mode='roulade')
    with pytest.raises(StopRequested, match='角速度'):
        check_posture(HOME, state(gyro=(0., 12.1, 0.)), cfg, policy_mode='roulade')
    q = HOME.copy()
    q[3] = JOINT_LIMITS[3, 1] + .13
    with pytest.raises(StopRequested, match='测得关节角'):
        check_posture(q, state(math.pi), cfg, policy_mode='roulade')


def test_monitor_requires_inversion_full_forward_progress_and_settling():
    cfg = enabled_config()
    monitor = RouladeMonitor(0., state(), cfg)
    # Two seconds at pi rad/s yields one forward turn; start/end ramps lose
    # only a small fraction of a turn, still above the 330-degree threshold.
    for i in range(1, 101):
        t = i * .02
        sample = state(math.pi * t, (0., math.pi, 0.))
        tilt = check_posture(HOME, sample, cfg, policy_mode='roulade')
        assert not monitor.update(t, sample, tilt)
    assert monitor.saw_inversion and math.degrees(monitor.angle) > 350
    for i in range(1, 13):
        assert not monitor.update(2 + i * .02, state(), 0.)
    assert monitor.update(2.28, state(), 0.)
    assert monitor.snapshot(2.28)['phase'] == 'settling'


def test_monitor_does_not_claim_stationary_upright_start_is_completion():
    monitor = RouladeMonitor(0., state(), enabled_config())
    for i in range(1, 250):
        assert not monitor.update(i * .02, state(), 0.)
    with pytest.raises(StopRequested, match='超时'):
        monitor.update(5., state(), 0.)


def test_monitor_allows_small_backward_windup_but_stops_a_reverse_roll():
    cfg = enabled_config()
    monitor = RouladeMonitor(0., state(), cfg)
    for i in range(1, 26):
        t = i * .02
        sample = state(-math.pi * t, (0., -math.pi, 0.))
        tilt = check_posture(HOME, sample, cfg, policy_mode='roulade')
        assert not monitor.update(t, sample, tilt)
    assert -90 < math.degrees(monitor.angle) < -85
    sample = state(-math.pi * .52, (0., -math.pi, 0.))
    tilt = check_posture(HOME, sample, cfg, policy_mode='roulade')
    with pytest.raises(StopRequested, match='反向转动'):
        monitor.update(.52, sample, tilt)


class Commands:
    def __init__(self, modes=('roulade',)):
        self.modes, self.index, self.policy_mode = modes, 0, 'walking'

    def __call__(self):
        mode = self.modes[min(self.index, len(self.modes) - 1)]
        self.index += 1
        if mode == 'B':
            raise GamepadStop('B')
        self.policy_mode = mode
        return (.3, .2, .5)

    def finish_roulade(self):
        self.policy_mode = 'stand'


def test_runtime_uses_zero_roll_command_and_preserves_raw_previous_action():
    with rig(cfg=enabled_config()) as r:
        roll = Policy(r.clock)
        r.policy.output[:] = .01
        roll.output[:] = .02
        r.ctl.roulade_policy, r.ctl.sitstand_policy = roll, Policy(r.clock)
        source = Commands(('walking', 'roulade', 'B'))
        with pytest.raises(GamepadStop):
            r.ctl.execute(seconds=.3, motion=True, command_provider=source)
        assert len(roll.observations) == 1
        np.testing.assert_array_equal(roll.observations[0][48:], np.zeros(13))
        np.testing.assert_allclose(roll.observations[0][34:48], .01)
        assert not r.ctl.enabled
        assert any(e[0] == 'torque' and e[2] is False for e in r.bus.events)


@pytest.mark.parametrize('enabled,loaded', [(False, True), (True, False)])
def test_runtime_rejects_unconfigured_roll_before_inference(enabled, loaded):
    cfg = enabled_config()
    cfg['control']['roulade_enabled'] = enabled
    with rig(cfg=cfg) as r:
        roll = Policy(r.clock)
        r.ctl.roulade_policy = roll if loaded else None
        r.ctl.sitstand_policy = Policy(r.clock)
        with pytest.raises(StopRequested, match='滚翻'):
            r.ctl.execute(seconds=.2, motion=True, command_provider=Commands())
        assert not roll.observations
        assert not r.ctl.enabled


def test_runtime_rejects_early_policy_switch_and_preserves_unload():
    with rig(cfg=enabled_config()) as r:
        r.ctl.roulade_policy, r.ctl.sitstand_policy = Policy(r.clock), Policy(r.clock)
        with pytest.raises(StopRequested, match='未确认恢复'):
            r.ctl.execute(seconds=.2, motion=True, command_provider=Commands(('roulade', 'walking')))
        assert not r.ctl.enabled


def test_runtime_measured_full_rotation_hands_control_to_stand_and_logs_progress():
    with rig(cfg=enabled_config()) as r:
        roll, stand = Policy(r.clock), Policy(r.clock)
        roll.output[:] = .01
        stand.output[:] = .02
        r.ctl.roulade_policy, r.ctl.sitstand_policy = roll, stand

        class Source:
            policy_mode = 'roulade'
            started = None
            finished = False

            def __call__(self):
                if self.started is None:
                    self.started = r.clock.now
                return ZERO

            def finish_roulade(self):
                self.finished = True
                self.policy_mode = 'stand'

        source = Source()
        original_latest = r.imu.latest

        def measured_imu(max_age):
            original = original_latest(max_age)
            age = 0. if source.started is None else r.clock.now - source.started
            sample = (state(age * math.pi, (0., math.pi, 0.))
                      if 0 < age < 2. else state())
            return dataclasses.replace(original, projected_gravity=sample.projected_gravity,
                                       base_ang_vel=sample.base_ang_vel)

        r.imu.latest = measured_imu
        r.ctl.execute(seconds=2.9, motion=True, command_provider=source)
        assert source.finished and len(roll.observations) > 100 and stand.observations
        np.testing.assert_array_equal(stand.observations[0][48:], np.zeros(13))
        np.testing.assert_allclose(stand.observations[0][34:48], .01)
        rows = [json.loads(row) for row in r.path.read_text().splitlines()]
        recovered = next(row for row in rows if row.get('policy_mode') == 'stand')
        assert recovered['roulade_progress']['saw_inversion']
        assert recovered['roulade_progress']['rotation_deg'] >= 330


@pytest.mark.parametrize('fault', ['imu', 'serial', 'encoder'])
def test_roll_keeps_sensor_and_encoder_fail_stop_checks(fault):
    with rig(cfg=enabled_config()) as r:
        roll = Policy(r.clock)
        r.ctl.roulade_policy, r.ctl.sitstand_policy = roll, Policy(r.clock)
        original_infer = roll.infer

        def after_first_inference(obs):
            result = original_infer(obs)
            if fault == 'imu':
                r.imu.fail_when = lambda: True
            elif fault == 'serial':
                r.bus.fail_read_at = r.bus.read_count + 1
            else:
                # The next bus read reports an impossible one-cycle jump.
                original_read = r.bus.read_feedback_many

                def jumped(ids, sync):
                    values = original_read(ids, sync)
                    sid = ids[0]
                    values[sid] = dataclasses.replace(values[sid], position_ticks=values[sid].position_ticks + 400)
                    return values

                r.bus.read_feedback_many = jumped
            return result

        roll.infer = after_first_inference
        with pytest.raises((StopRequested, OSError)):
            r.ctl.execute(seconds=.2, motion=True, command_provider=Commands())
        assert len(roll.observations) == 1
        assert not r.ctl.enabled
