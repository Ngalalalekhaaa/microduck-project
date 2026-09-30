import importlib.util
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
import io

import pytest

spec = importlib.util.spec_from_file_location(
    'home_then_policy', Path(__file__).resolve().parents[1]/'scripts/home_then_policy.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


def test_home_failure_never_starts_policy_or_prompts():
    run, ready = Mock(return_value=1), Mock()
    assert flow.run_sequence(run, ready, 'python', seconds=30, vx=.3) == 1
    assert run.call_count == 1
    ready.assert_not_called()


def test_missing_physical_confirmation_never_starts_policy():
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: False, 'python', seconds=30, vx=.3) == 130
    assert run.call_count == 1


def test_home_holds_until_confirmation_then_exact_requested_policy():
    calls = []
    def run(command):
        calls.append(command)
        return 0
    def ready():
        assert len(calls) == 1
        assert '--target' in calls[0] and 'home' in calls[0]
        assert '--soft-gains' in calls[0]
        return True
    assert flow.run_sequence(run, ready, 'python', seconds=30, vx=.3) == 0
    assert calls[1][2:] == ['--calibrate-imu', '--seconds', '30', '--vx', '0.3', '--move']


def test_policy_failure_is_not_retried_and_existing_hold_keeps_gains():
    run = Mock(side_effect=[0, 1])
    assert flow.run_sequence(run, lambda: True, 'python', seconds=30, vx=.3, soft_gains=False) == 1
    assert run.call_count == 2
    assert '--soft-gains' not in run.call_args_list[0].args[0]


def test_ramp_option_reaches_policy_only():
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: True, 'python', seconds=30, vx=.05, ramp_seconds=3) == 0
    assert '--ramp-seconds' not in run.call_args_list[0].args[0]
    assert run.call_args_list[1].args[0][-2:] == ['--ramp-seconds', '3']


def test_jump_override_reaches_policy_only():
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: True, 'python', seconds=10, vx=.2,
                             max_target_jump_deg=45) == 0
    assert '--max-target-jump-deg' not in run.call_args_list[0].args[0]
    assert run.call_args_list[1].args[0][-2:] == ['--max-target-jump-deg', '45']


@pytest.mark.parametrize('answer', ['WALK\n', 'walk\n', ' Walk \n'])
def test_walk_confirmation_ignores_case(answer):
    with patch.object(flow.sys, 'stdin', io.StringIO(answer)), patch.object(
            flow.select, 'select', return_value=([True], [], [])):
        assert flow.wait_for_feet()


def test_eof_does_not_start_policy():
    with patch.object(flow.sys, 'stdin', io.StringIO('')), patch.object(
            flow.select, 'select', return_value=([True], [], [])):
        assert not flow.wait_for_feet()


def test_support_wait_requires_off_not_arbitrary_input():
    with patch.object(flow.sys, 'stdin', io.StringIO('walk\noff\n')), patch.object(
            flow.select, 'select', return_value=([True], [], [])) as select_ready:
        flow.wait_for_support()
    assert select_ready.call_count==2


def test_support_wait_does_not_unload_immediately_on_stdin_eof():
    with patch.object(flow.sys, 'stdin', io.StringIO('')), patch.object(
            flow.time, 'monotonic', side_effect=[0, 0, 0, 60]), patch.object(
            flow.select, 'select', return_value=([True], [], [])):
        flow.wait_for_support()


def test_interruption_between_home_and_policy_propagates_for_cleanup():
    run = Mock(return_value=0)
    def ready():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        flow.run_sequence(run, ready, 'python', seconds=30, vx=.3)
    assert run.call_count == 1
