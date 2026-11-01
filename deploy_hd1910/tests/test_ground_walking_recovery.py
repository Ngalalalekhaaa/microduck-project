"""Regression for keyboard 9 -> implicit experimental stand -> motor OFF."""
from pathlib import Path

import numpy as np
import pytest

from microduck_deploy.runtime import StopRequested
from test_runtime import Policy, calibrated_config, rig
from test_terminal_actions import terminal, ready, tick, ground
from test_terminal_action_runtime import attach_policies, on_infer, samples, assert_unloaded


def test_direction_during_recovery_is_discarded_and_fresh_six_slews_after_three_seconds():
    with terminal(single_key=True) as t:
        ready(t)
        ground(t)
        tick(t, 4.01)
        assert t.command.policy_mode == 'walking'
        assert tick(t, .1, line='6') == (0., 0., 0.)
        assert tick(t, 2.8, line='6') == (0., 0., 0.)
        assert tick(t, .2) == (0., 0., 0.)
        assert tick(t, 1.) == (0., 0., 0.)
        assert tick(t, line='6') == pytest.approx((0., 0., -.02))


def test_real_controller_returns_to_selected_d_and_never_invokes_selected_sitstand_n():
    cfg = calibrated_config()
    cfg['keyboard_walking_models'] = {key: {'model': key + '.onnx'} for key in 'abcdef'}
    cfg['keyboard_sitstand_models'] = {key: {'model': key + '.onnx'} for key in 'ghimn'}
    with rig(cfg=cfg) as r, terminal(
            clock=r.clock.monotonic, single_key=True,
            walking_models={key: key for key in 'abcdef'},
            sitstand_models={key: key for key in 'ghimn'}, default_sitstand_key='n') as t:
        sit, ground_policy, roll = attach_policies(r)
        selected_walk = Policy(r.clock)
        selected_walk.output[:] = .05
        selected_walk.model_path = Path('/models/d.onnx')
        r.ctl.keyboard_walking_policies = {key: r.policy for key in 'abcdef'}
        r.ctl.keyboard_walking_policies['d'] = selected_walk
        r.ctl.keyboard_sitstand_policies = {key: sit for key in 'ghimn'}
        sit.fail = AssertionError('Keyboard 9 must not invoke sitstand holding')
        state = {'recovery_started': None, 'fresh_six_sent': False}
        on_infer(r.policy, lambda obs, count: t.write(b'd') if count == 1 else None)

        def walking(obs, count):
            if count == 1:
                t.write(b'9')
            if not ground_policy.observations:
                return
            state.setdefault('recovery_observations', []).append((r.clock.now, obs.copy()))
            if state['recovery_started'] is None:
                state['recovery_started'] = r.clock.now
                state['first_recovery_obs'] = obs.copy()
                t.write(b'6')  # Discarded during the three-second zero hold.
            elif r.clock.now - state['recovery_started'] >= 3.08 and not state['fresh_six_sent']:
                t.write(b'6')
                state['fresh_six_sent'] = True
            elif r.clock.now - state['recovery_started'] > 3.6:
                t.write(b'q')

        on_infer(selected_walk, walking)
        with pytest.raises(StopRequested, match='终端请求退出'):
            r.ctl.execute(seconds=12, motion=True, command_provider=t.command)
        rows = samples(r)
        ground_end = max(i for i, row in enumerate(rows) if row['policy_mode'] == 'ground_pick')
        recovery = rows[ground_end + 1:]
        assert recovery and {row['policy_id'] for row in recovery} == {'walking:keyboard_d'}
        assert not sit.observations and not roll.observations
        first_recovery_obs = state['first_recovery_obs']
        np.testing.assert_allclose(first_recovery_obs[34:48], ground_policy.output)
        # Inspect every policy input directly: the bounded async log writer can
        # drop frames when a mock clock runs 9 simulated seconds in milliseconds.
        for stamp, obs in state['recovery_observations']:
            if stamp - state['recovery_started'] < 3.:
                np.testing.assert_array_equal(obs[48:51], 0)
        turning = [(stamp, obs) for stamp, obs in state['recovery_observations'] if obs[50] < 0]
        assert turning and turning[0][0] - state['recovery_started'] > 3.
        assert -.021 <= turning[0][1][50] < 0
        assert all(np.max(np.abs(np.array(row['command'])[:2])) == 0 for row in recovery)
        assert_unloaded(r)
