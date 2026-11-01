import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from microduck_deploy.imu import ImuError

spec=importlib.util.spec_from_file_location(
    'policy_trial',Path(__file__).resolve().parents[1]/'scripts/policy_trial.py')
trial=importlib.util.module_from_spec(spec)
spec.loader.exec_module(trial)


def test_transient_startup_failure_retries_without_any_controller():
    sensor=object()
    factory=Mock(side_effect=[ImuError('sample stale'),sensor])
    sleep=Mock()
    assert trial.start_imu({},True,factory,sleep=sleep) is sensor
    assert factory.call_count==2
    sleep.assert_called_once_with(1)


def test_persistent_imu_failure_is_bounded_and_preserves_cause():
    factory=Mock(side_effect=ImuError('calibration rejected motion'))
    sleep=Mock()
    with pytest.raises(ImuError,match='calibration rejected motion'):
        trial.start_imu({},True,factory,sleep=sleep)
    assert factory.call_count==3
    assert sleep.call_count==2


def test_programming_errors_are_not_silently_retried():
    factory=Mock(side_effect=ValueError('invalid configuration'))
    with pytest.raises(ValueError):
        trial.start_imu({},True,factory,sleep=Mock())
    assert factory.call_count==1


def test_walk_policy_defaults_to_three_second_bias_and_one_second_alignment():
    args = trial.parse_args(['--calibrate-imu', '--gamepad'])
    assert args.imu_calibration_seconds == 3
    assert args.policy_home_seconds == 1
    assert (args.max_vx,args.max_vy,args.max_wz) == (.3,.3,1.)
    factory = Mock(return_value=object())
    trial.start_imu({}, True, factory, calibration_seconds=args.imu_calibration_seconds)
    factory.assert_called_once_with({}, calibrate=True, calibration_seconds=3.)


def test_saved_bias_does_not_request_new_stationary_sampling():
    factory = Mock(return_value=object())
    trial.start_imu({}, False, factory, calibration_seconds=3.)
    factory.assert_called_once_with({}, calibrate=False)


@pytest.mark.parametrize('seconds', ['0', '120'])
def test_policy_duration_no_longer_has_sixty_second_limit(seconds):
    args = trial.parse_args(['--calibrate-imu', '--gamepad', '--seconds', seconds,
                             '--ramp-seconds', '3'])
    assert args.seconds == float(seconds)
    assert args.ramp_seconds == 3


@pytest.mark.parametrize('seconds', ['-1', 'nan', 'inf'])
def test_policy_duration_still_rejects_invalid_values(seconds):
    with pytest.raises(SystemExit):
        trial.parse_args(['--calibrate-imu', '--seconds', seconds])


def test_timed_zero_keeps_controller_unlimited_and_preserves_startup_options():
    args = trial.parse_args(['--calibrate-imu', '--move', '--seconds', '0',
                             '--vx', '0.3', '--zero-after-seconds', '5', '--ramp-seconds', '2'])
    assert args.seconds == 0
    assert args.zero_after_seconds == 5
    assert args.vx == .3
    assert args.ramp_seconds == 2
    assert args.imu_calibration_seconds == 3
    assert args.policy_home_seconds == 1
    assert trial.parse_args(['--calibrate-imu', '--seconds', '5']).zero_after_seconds is None


@pytest.mark.parametrize('extra', [
    ['--zero-after-seconds', '0'],
    ['--zero-after-seconds', '-1'],
    ['--zero-after-seconds', 'nan'],
    ['--zero-after-seconds', 'inf'],
    ['--zero-after-seconds', '5', '--seconds', '5'],
    ['--zero-after-seconds', '5', '--gamepad'],
    ['--zero-after-seconds', '5', '--ramp-seconds', '6'],
])
def test_timed_zero_rejects_ambiguous_or_invalid_runtime(extra):
    with pytest.raises(SystemExit):
        trial.parse_args(['--calibrate-imu', '--seconds', '0', *extra])


def test_keyboard_has_zero_start_and_uses_configured_direction_speeds():
    args = trial.parse_args(['--calibrate-imu', '--keyboard', '--seconds', '0', '--vx', '0',
                             '--max-vx', '.3', '--max-vy', '.2', '--max-wz', '.5'])
    assert args.keyboard and not args.gamepad
    assert args.vx == 0 and args.seconds == 0
    assert (args.max_vx, args.max_vy, args.max_wz) == (.3, .2, .5)


@pytest.mark.parametrize('extra', [
    ['--gamepad'], ['--seconds', '5'], ['--vx', '.3'],
    ['--zero-after-seconds', '5'], ['--ramp-seconds', '2'],
])
def test_keyboard_rejects_competing_command_sources_and_timers(extra):
    with pytest.raises(SystemExit):
        trial.parse_args(['--calibrate-imu', '--keyboard', '--seconds', '0', '--vx', '0', *extra])


def test_preflight_uses_configured_imu_age_without_opening_real_hardware(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch
    import numpy as np
    from test_runtime import rig
    with rig() as r:
        r.cfg['control']['imu_max_age_s'] = .10
        r.imu.imu = SimpleNamespace(gyro_bias_rad_s=np.zeros(3))
        bus_context = MagicMock()
        bus_context.__enter__.return_value = r.bus
        monkeypatch.setattr(trial.sys, 'argv', ['policy_trial.py', '--calibrate-imu'])
        monkeypatch.setattr('microduck_deploy.config.load_config', lambda _: r.cfg)
        monkeypatch.setattr('microduck_deploy.policy.OnnxPolicy', lambda _: r.policy)
        monkeypatch.setattr('microduck_deploy.imu_process.ProcessImuReader', lambda *a, **kw: r.imu)
        monkeypatch.setattr('microduck_deploy.servo.ServoBus', lambda **kw: bus_context)
        monkeypatch.setattr('microduck_deploy.runtime.Controller', lambda cfg, **kw: r.ctl)
        with patch.object(r.imu, 'latest', wraps=r.imu.latest) as latest:
            trial.main()
        latest.assert_called_once_with(.10)
        assert not any(event[0] in ('positions', 'torque', 'gains') for event in r.bus.events)
