"""Exercise mouth ownership and stop ordering through the real parent main."""
import copy
import importlib.util
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from microduck_deploy.gamepad import PadSample
from microduck_deploy.config import default_config


spec = importlib.util.spec_from_file_location(
    'jaw_flow_parent', Path(__file__).resolve().parents[1] / 'scripts/home_then_policy.py')
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)


@pytest.mark.parametrize('stop_stage', [None, 'home', 'policy'])
def test_parent_hands_off_held_x_and_joins_before_unloading_all_servos(
        tmp_path, monkeypatch, stop_stage):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    import supported_home
    import microduck_deploy.config as config_module
    import microduck_deploy.gamepad as gamepad_module
    import microduck_deploy.jaw as jaw_module
    import microduck_deploy.servo as servo_module

    cfg = default_config()
    cfg['serial']['port'] = '/dev/fake-mouth-flow'
    for sid, joint in enumerate(cfg['joints'], 1):
        joint.update(id=sid, zero_tick=1900 + sid, direction=(-1 if sid % 2 else 1),
                     mapping_verified=True)
    cfg['calibration'] = {'captured_ticks': list(range(1901, 1915)), 'reference': 'straight'}
    cfg['jaw'] = {'enabled': True, 'id': 15, 'mapping_verified': True,
                  'closed_tick': 344, 'open_tick': 51, 'button_code': 307}
    original_cfg = copy.deepcopy(cfg)
    events, messages, idle_commands = [], {}, []
    owner = {'idle': None, 'position': None, 'child': None}

    class Reader:
        metadata = {}
        count = 0
        stopped = False

        def latest(self):
            return PadSample(time.monotonic(), True,
                             {'ABS_X': 0., 'ABS_Y': 0., 'ABS_RX': 0.}, {},
                             (307,) if self.count else (), self.stopped,
                             button_press_counts={307: self.count})

        def close(self):
            events.append('reader-close')

    reader = Reader()

    class IdleBus:
        def __init__(self, **options):
            assert owner['idle'] is None and owner['position'] is None
            assert owner['child'] is None or owner['child'].poll() is not None
            owner['idle'] = self
            events.append('idle-open')

        def close(self):
            assert owner['idle'] is self
            owner['idle'] = None
            events.append('idle-close')

        def sync_positions(self, targets):
            assert owner['idle'] is self and set(targets) == {15}
            events.append(('idle-goal', targets[15]))

    class PositionBus:
        def __init__(self, **options):
            assert owner['idle'] is None and owner['position'] is None
            assert owner['child'] is None or owner['child'].poll() is not None
            owner['position'] = self
            events.append('position-open')

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            assert owner['position'] is self
            owner['position'] = None
            events.append('position-close')

        def read_configuration(self, sid):
            assert sid in range(1, 15)
            return {'torque': 0}

        def set_torque(self, ids, enabled, *, verify):
            assert ids == list(range(1, 16)) and enabled is False and verify
            events.append('unload-1..15')

    class Jaw:
        def __init__(self, configuration, bus):
            assert configuration is cfg and owner['idle'] is bus
            self.open = None

        def preflight(self):
            events.append('jaw-preflight')

        def targets(self, *, command=None):
            idle_commands.append(command)
            if command is not None:
                self.open = command
            assert self.open is not None
            return {15: 51 if self.open else 344}

    class Child:
        def __init__(self, stage, read_fd):
            self.stage, self.fd = stage, os.dup(read_fd)
            os.set_blocking(self.fd, False)
            self.returncode = None
            self.pid = 123456

        def poll(self):
            return self.returncode

        def receive(self):
            try:
                return os.read(self.fd, 4096)
            except BlockingIOError:
                return b''

        def finish(self):
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None
            self.returncode = 0

        def send_signal(self, signal):
            events.append('interrupt-' + self.stage)

        def wait(self, timeout=None):
            events.append('joined-' + self.stage)
            self.finish()
            return 0

    def popen(command, *, pass_fds, **kwargs):
        assert owner['idle'] is None and owner['position'] is None
        stage = 'policy' if '--calibrate-imu' in command else 'home'
        index = command.index('--jaw-command-fd')
        read_fd = int(command[index + 1])
        assert pass_fds == (read_fd,)
        child = Child(stage, read_fd)
        owner['child'] = child
        events.append('popen-' + stage)
        return child

    original_wait = flow.wait_for_child

    def wait_for_child(child, pad, *, tick):
        assert pad is reader and owner['idle'] is None
        messages[child.stage] = [child.receive()]
        if child.stage == stop_stage:
            reader.stopped = True
            # The real monitor raises B; the real parent must close its pipe,
            # signal/join this still-running child, then open the unload bus.
            return original_wait(child, pad, tick=tick)
        tick()  # X is still held from the preceding stage.
        assert child.receive() == b'' and owner['idle'] is None
        reader.count += 1  # A new brief X press while this child owns motion.
        tick()
        messages[child.stage].append(child.receive())
        tick()
        assert child.receive() == b'' and owner['idle'] is None
        child.finish()
        return 0

    starts = 0

    def wait_for_a(pad, button, *, tick):
        nonlocal starts
        starts += 1
        if starts > 1:
            raise KeyboardInterrupt
        tick()  # Baseline old button counts before accepting a new X.
        reader.count += 1
        tick()
        tick()  # A held X leaves the command unchanged.
        return True

    def wait_for_menu(pad, *, tick):
        assert pad is reader
        # The HOME child's latest desired CLOSED state is retained. Holding
        # the same X across child exit and Menu cannot toggle it a second time.
        tick()
        tick()
        assert idle_commands[-2:] == [False, None]
        return True

    def verify_home(configuration, factory):
        assert configuration is cfg and owner['idle'] is None
        assert owner['child'].poll() == 0
        return True

    def exit_after_b_release(pad):
        raise KeyboardInterrupt

    monkeypatch.setattr(flow, 'ROOT', tmp_path)
    monkeypatch.setattr(flow, 'WEB', tmp_path / 'no-web')
    monkeypatch.setattr(flow.sys, 'argv', ['home_then_policy.py', '--move', '--gamepad',
                         '--gamepad-buttons', '--vx', '0', '--seconds', '60'])
    monkeypatch.setattr(config_module, 'load_config', lambda path: cfg)
    monkeypatch.setattr(gamepad_module, 'GamepadReader', lambda path: reader)
    monkeypatch.setattr(servo_module, 'ServoBus', IdleBus)
    monkeypatch.setattr(supported_home, 'PositionOnlyBus', PositionBus)
    monkeypatch.setattr(jaw_module, 'JawController', Jaw)
    monkeypatch.setattr(flow.fcntl, 'flock', lambda *args: None)
    monkeypatch.setattr(flow.subprocess, 'run', lambda *args, **kwargs:
                        SimpleNamespace(returncode=1))
    monkeypatch.setattr(flow.subprocess, 'Popen', popen)
    monkeypatch.setattr(flow, 'wait_for_gamepad_button', wait_for_a)
    monkeypatch.setattr(flow, 'wait_for_gamepad_walk', wait_for_menu)
    monkeypatch.setattr(flow, 'verify_home_hold', verify_home)
    monkeypatch.setattr(flow, 'wait_for_child', wait_for_child)
    monkeypatch.setattr(flow, 'wait_for_b_release', exit_after_b_release)

    assert flow.main() == 130
    assert cfg == original_cfg
    assert owner['idle'] is None and owner['position'] is None
    assert idle_commands[:2] == [True, None]
    assert events.index('idle-close') < events.index('popen-home')
    assert 'unload-1..15' in events
    if stop_stage == 'home':
        assert messages == {'home': [b'1\n']}
        assert 'popen-policy' not in events
    else:
        assert messages['home'] == [b'1\n', b'0\n']
        # The parent reopened only 15 while waiting for Menu, then closed it
        # before the policy child inherited the existing CLOSED request.
        policy_at = events.index('popen-policy')
        assert events[policy_at - 1] == 'idle-close'
        assert messages['policy'][0] == b'0\n'
        if stop_stage is None:
            assert messages['policy'] == [b'0\n', b'1\n']
    if stop_stage is not None:
        interrupt_at = events.index('interrupt-' + stop_stage)
        join_at = events.index('joined-' + stop_stage)
        unload_at = events.index('unload-1..15')
        assert interrupt_at < join_at < unload_at
        assert events[join_at + 1] == 'position-open'
