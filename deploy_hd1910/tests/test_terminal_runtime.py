"""Real terminal input routed through a simulated motor/IMU control loop."""
import os
import pty
import select
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from microduck_deploy import runtime
from microduck_deploy.terminal_command import TerminalCommand
from test_runtime import rig


@contextmanager
def terminal(clock):
    master, slave = pty.openpty()
    source = TerminalCommand(SimpleNamespace(isatty=lambda: True, fileno=lambda: slave), clock=clock)
    def send(line):
        os.write(master, line.encode())
        assert select.select([slave], [], [], .5)[0]
    try:
        yield source, send, master
    finally:
        source.close()
        os.close(slave)
        try:
            os.close(master)
        except OSError:
            pass


def test_numeric_input_has_one_reader_and_zero_keeps_policy_and_torque_active():
    with rig() as r, terminal(r.clock.monotonic) as (source, send, _):
        send('1\n')  # Must be discarded during HOME; never applied to the first sample.
        r.policy.output[:] = .01
        original = r.policy.infer
        def infer(obs):
            assert not any(e[0] == 'torque' and not e[2] for e in r.bus.events)
            result = original(obs)
            count = len(r.policy.observations)
            if count == 1:
                send('1\n')
            elif count == 5:
                send('0\n')
            elif count == 10:
                send('q\n')
            elif count > 12:
                pytest.fail('Keyboard q was consumed or ignored')
            return result
        r.policy.infer = infer
        # The original generic stdin reader must never steal keyboard input.
        fake_stdin = SimpleNamespace(isatty=lambda: True,
                                     readline=lambda: pytest.fail('Second stdin reader used'))
        with patch.object(runtime.sys, 'stdin', fake_stdin):
            with pytest.raises(runtime.StopRequested):
                r.ctl.execute(seconds=0, motion=True, command_provider=source)
        obs = np.asarray(r.policy.observations)
        assert len(obs) == 10
        np.testing.assert_array_equal(obs[0, 48:51], 0)
        assert np.all(obs[1:5, 48] > 0)
        np.testing.assert_array_equal(obs[5:, 48:51], 0)
        np.testing.assert_allclose(obs[1:, 34:48], .01)
        assert r.ctl.session['command_source'] == 'terminal'
        assert r.ctl.enabled is False
        off = [e for e in r.bus.events if e[0] == 'torque' and not e[2]]
        assert len(off) == 1 and off[0][1] >= r.policy.times[-1]


def test_lost_terminal_stops_controller_and_unloads_after_initial_zero():
    with rig() as r, terminal(r.clock.monotonic) as (source, send, master):
        original = r.policy.infer
        def infer(obs):
            result = original(obs)
            if len(r.policy.observations) == 2:
                os.close(master)
            elif len(r.policy.observations) > 4:
                pytest.fail('Lost terminal kept controller alive')
            return result
        r.policy.infer = infer
        with pytest.raises(runtime.StopRequested):
            r.ctl.execute(seconds=0, motion=True, command_provider=source)
        assert r.ctl.enabled is False
        assert any(e[0] == 'torque' and not e[2] for e in r.bus.events)


def test_keyboard_q_during_home_never_runs_policy():
    with rig() as r, terminal(r.clock.monotonic) as (source, send, _):
        send('q\n')
        with pytest.raises(runtime.StopRequested):
            r.ctl.execute(seconds=0, motion=True, command_provider=source)
        assert not r.policy.observations
        assert r.ctl.enabled is False
