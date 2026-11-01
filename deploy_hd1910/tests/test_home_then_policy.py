import importlib.util
from pathlib import Path
from unittest.mock import Mock, MagicMock
from unittest.mock import patch
import io
import dataclasses
import time
from types import SimpleNamespace

import pytest
from microduck_deploy.gamepad import PadSample, GamepadStop, GamepadError

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


def test_imu_calibration_entry_is_launched_only_after_walk_confirmation():
    events = []
    def run(command):
        events.append('imu' if '--calibrate-imu' in command else 'home')
        return 0
    def ready():
        events.append('WALK')
        return True
    assert flow.run_sequence(run, ready, 'python', seconds=60, vx=0) == 0
    assert events == ['home', 'WALK', 'imu']


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


def test_timed_zero_reaches_one_continuous_policy_after_home_and_confirmation():
    run = Mock(return_value=0)
    ready = Mock(return_value=True)
    assert flow.run_sequence(run, ready, 'python', seconds=0, vx=.3,
                             ramp_seconds=2, zero_after_seconds=5) == 0
    ready.assert_called_once()
    assert run.call_count == 2
    assert '--zero-after-seconds' not in run.call_args_list[0].args[0]
    command = run.call_args_list[1].args[0]
    assert command[command.index('--seconds')+1] == '0'
    assert command[command.index('--zero-after-seconds')+1] == '5'
    assert command[command.index('--ramp-seconds')+1] == '2'


def test_timed_zero_preview_describes_continuation_without_accessing_devices(monkeypatch, capsys):
    monkeypatch.setattr(flow.sys, 'argv', ['home_then_policy.py', '--seconds', '0',
                                          '--vx', '.3', '--zero-after-seconds', '5'])
    assert flow.main() == 0
    out = capsys.readouterr().out
    assert '5秒后速度指令归零' in out
    assert '继续运行模型' in out
    assert '未访问硬件' in out


@pytest.mark.parametrize('extra', [
    ['--zero-after-seconds', '0'],
    ['--zero-after-seconds', '-1'],
    ['--zero-after-seconds', 'nan'],
    ['--zero-after-seconds', 'inf'],
    ['--zero-after-seconds', '5', '--seconds', '5'],
    ['--zero-after-seconds', '5', '--gamepad'],
    ['--zero-after-seconds', '5', '--ramp-seconds', '6'],
])
def test_timed_zero_invalid_arguments_fail_before_home(monkeypatch, extra):
    monkeypatch.setattr(flow.sys, 'argv', ['home_then_policy.py', '--move', '--seconds', '0', *extra])
    with pytest.raises(SystemExit) as exc:
        flow.main()
    assert exc.value.code == 2


def test_keyboard_options_only_reach_policy_child():
    run, ready = Mock(return_value=0), Mock(return_value=True)
    keyboard_args = ['--keyboard', '--max-vx', '.3', '--max-vy', '.2', '--max-wz', '.5']
    assert flow.run_sequence(run, ready, 'python', seconds=0, vx=0,
                             keyboard_args=keyboard_args, log_only_target_angles=True) == 0
    ready.assert_called_once()
    assert '--keyboard' not in run.call_args_list[0].args[0]
    command = run.call_args_list[1].args[0]
    assert command[-len(keyboard_args):] == keyboard_args
    assert '--log-only-target-angles' in command
    assert command[command.index('--vx') + 1] == '0'


def test_keyboard_preview_is_zero_start_and_does_not_require_a_tty(monkeypatch, capsys):
    monkeypatch.setattr(flow.sys, 'argv', ['home_then_policy.py', '--keyboard',
                                          '--seconds', '0', '--vx', '0'])
    assert flow.main() == 0
    out = capsys.readouterr().out
    assert '零速度启动模型' in out and '0停步保持模型' in out and '未访问硬件' in out


@pytest.mark.parametrize('extra', [
    ['--gamepad'], ['--seconds', '5'], ['--vx', '.3'],
    ['--zero-after-seconds', '5'], ['--ramp-seconds', '2'],
])
def test_keyboard_invalid_options_fail_before_any_home_move(monkeypatch, extra):
    monkeypatch.setattr(flow.sys, 'argv', ['home_then_policy.py', '--move', '--keyboard',
                                          '--seconds', '0', '--vx', '0', *extra])
    with pytest.raises(SystemExit) as exc:
        flow.main()
    assert exc.value.code == 2


def test_jump_override_reaches_policy_only():
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: True, 'python', seconds=10, vx=.2,
                             max_target_jump_deg=45) == 0
    assert '--max-target-jump-deg' not in run.call_args_list[0].args[0]
    assert run.call_args_list[1].args[0][-2:] == ['--max-target-jump-deg', '45']


def test_target_logging_mode_reaches_policy_after_home():
    run = Mock(return_value=0)
    assert flow.run_sequence(run, lambda: True, 'python', seconds=60, vx=.2,
                             log_only_target_angles=True) == 0
    assert '--log-only-target-angles' not in run.call_args_list[0].args[0]
    assert '--log-only-target-angles' in run.call_args_list[1].args[0]


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


def test_gamepad_options_reach_policy_only():
    run = Mock(return_value=0)
    flags = ['--gamepad', '--max-vx', '.2', '--gamepad-device', '/dev/input/event4']
    assert flow.run_sequence(run, lambda: True, 'python', seconds=60, vx=0,
                             gamepad_args=flags) == 0
    assert '--gamepad' not in run.call_args_list[0].args[0]
    assert run.call_args_list[1].args[0][-len(flags):] == flags


def sample(**kwargs):
    return PadSample(time.monotonic(), True,
                     {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {}, (), **kwargs)


def test_button_wait_ignores_earlier_press_and_accepts_new_brief_press():
    reader = Mock()
    reader.latest.side_effect = [sample(button_press_counts={315: 1}),
                                 sample(button_press_counts={315: 1}),
                                 sample(button_press_counts={315: 2})]
    with patch.object(flow.time, 'sleep'):
        assert flow.wait_for_gamepad_button(reader, 315, timeout=1)
    assert reader.latest.call_count == 3


def test_held_menu_button_does_not_advance_without_another_press():
    held = dataclasses.replace(sample(button_press_counts={315: 1}), buttons=(315,))
    reader = Mock(latest=Mock(return_value=held))
    assert not flow.wait_for_gamepad_button(reader, 315, timeout=.04)


def test_start_press_with_deflected_stick_requires_new_press_after_centering():
    deflected = dataclasses.replace(sample(button_press_counts={315: 1}),
                                   axes={'ABS_X': .5, 'ABS_Y': 0., 'ABS_RX': 0.})
    reader = Mock()
    reader.latest.side_effect = [sample(), deflected,
                                 sample(button_press_counts={315: 1}),
                                 sample(button_press_counts={315: 2})]
    with patch.object(flow.time, 'sleep'):
        assert flow.wait_for_gamepad_button(reader, 315, timeout=1)
    assert reader.latest.call_count == 4


@pytest.mark.parametrize('waiting_button', [304, 315])
def test_b_interrupts_either_button_wait(waiting_button):
    reader = Mock()
    reader.latest.side_effect = [sample(), sample(stop_requested=True)]
    with pytest.raises(GamepadStop):
        flow.wait_for_gamepad_button(reader, waiting_button, timeout=1)


def test_b_interrupts_running_child_for_parent_cleanup():
    child = Mock()
    child.poll.return_value = None
    reader = Mock(latest=Mock(return_value=sample(stop_requested=True)))
    with pytest.raises(GamepadStop):
        flow.wait_for_child(child, reader)
    child.wait.assert_not_called()


def test_b_interrupts_imu_failure_hold_without_keyboard_input():
    reader = Mock(latest=Mock(return_value=sample(stop_requested=True)))
    with pytest.raises(GamepadStop):
        flow.wait_for_support(reader=reader)


def test_disconnected_reader_cannot_start_home_or_policy():
    reader = Mock(latest=Mock(return_value=dataclasses.replace(sample(), connected=False)))
    with pytest.raises(GamepadError):
        flow.wait_for_gamepad_button(reader, 304, timeout=1)


@pytest.mark.parametrize('stop_stage', [None, 'home', 'policy'])
def test_main_button_sequence_and_b_cleanup_order(tmp_path, monkeypatch, stop_stage):
    """Run the real orchestrator with fake devices/processes; no hardware opens."""
    from contextlib import ExitStack
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    import supported_home
    events = []
    active = [None]
    stopped = [False]
    starts = [0]
    cfg = {'serial': {'port': '/dev/test', 'timeout': .015},
           'joints': [{'id': i} for i in range(1, 15)]}
    bus = Mock()
    bus.read_configuration.return_value = {'torque': 0}
    def unload(*args, **kwargs):
        events.append('unload')
        active[0] = None
    bus.set_torque.side_effect = unload
    factory = MagicMock()
    factory.return_value.__enter__.return_value = bus
    reader = Mock(metadata={})
    def latest():
        stop = active[0] == stop_stage and stop_stage is not None and not stopped[0]
        if stop:
            stopped[0] = True
        return sample(stop_requested=stop)
    reader.latest.side_effect = latest
    def button(_, code, **kwargs):
        starts[0] += 1
        if starts[0] > 2:
            raise KeyboardInterrupt
        events.append('A')
        return True
    def ready(reader):
        events.append('Menu')
        return True
    def popen(command, **kwargs):
        stage = 'policy' if '--calibrate-imu' in command else 'home'
        events.append(stage)
        active[0] = stage
        child = Mock(returncode=0)
        child.poll.return_value = None if stage == stop_stage and not stopped[0] else 0
        child.send_signal.side_effect = lambda sig: events.append('interrupt-'+stage)
        child.wait.side_effect = lambda **kwargs: events.append('joined-'+stage) or 0
        return child
    with ExitStack() as stack:
        stack.enter_context(patch.object(flow, 'ROOT', tmp_path))
        stack.enter_context(patch.object(flow, 'WEB', tmp_path/'no-web'))
        stack.enter_context(patch.object(flow.sys, 'argv', ['home_then_policy.py', '--move',
            '--gamepad', '--gamepad-buttons', '--vx', '0', '--seconds', '60']))
        stack.enter_context(patch('microduck_deploy.config.load_config', return_value=cfg))
        stack.enter_context(patch('microduck_deploy.gamepad.GamepadReader', return_value=reader))
        stack.enter_context(patch.object(supported_home, 'PositionOnlyBus', factory))
        stack.enter_context(patch.object(flow.fcntl, 'flock'))
        stack.enter_context(patch.object(flow.subprocess, 'run', return_value=SimpleNamespace(returncode=1)))
        stack.enter_context(patch.object(flow.subprocess, 'Popen', side_effect=popen))
        stack.enter_context(patch.object(flow, 'wait_for_gamepad_button', side_effect=button))
        stack.enter_context(patch.object(flow, 'wait_for_gamepad_walk', side_effect=ready))
        stack.enter_context(patch.object(flow, 'verify_home_hold', return_value=True))
        assert flow.main() == 130
    if stop_stage is None:
        assert events == ['A', 'home', 'Menu', 'policy', 'unload',
                          'A', 'home', 'Menu', 'policy', 'unload', 'unload']
    elif stop_stage == 'home':
        assert events == ['A', 'home', 'interrupt-home', 'joined-home', 'unload',
                          'A', 'home', 'Menu', 'policy', 'unload', 'unload']
    else:
        assert events == ['A', 'home', 'Menu', 'policy', 'interrupt-policy', 'joined-policy', 'unload',
                          'A', 'home', 'Menu', 'policy', 'unload', 'unload']
    assert bus.set_torque.call_count == 3
    assert all(call.args == (list(range(1, 15)), False) and call.kwargs == {'verify': True}
               for call in bus.set_torque.call_args_list)


def test_home_ready_a_requests_new_home_without_starting_policy():
    with patch.object(flow, 'wait_for_gamepad_action', return_value=304):
        with pytest.raises(flow.RehomeRequested):
            flow.wait_for_gamepad_walk(Mock())


def test_measured_home_hold_is_required_before_enabling_menu():
    from test_runtime import rig
    with rig() as r:
        r.bus.initial_torque = 1
        factory = MagicMock()
        factory.return_value.__enter__.return_value = r.bus
        assert flow.verify_home_hold(r.cfg, factory)
        r.bus.positions[r.bus.ids[3]] += 200
        assert not flow.verify_home_hold(r.cfg, factory)
        assert not any(event[0] in ('positions', 'torque') for event in r.bus.events)
