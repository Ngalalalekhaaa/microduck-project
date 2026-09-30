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
