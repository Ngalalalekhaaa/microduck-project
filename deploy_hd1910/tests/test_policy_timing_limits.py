"""Policy jitter budgets remain separate from servo and HOME deadlines."""
import dataclasses
import json

import pytest

from microduck_deploy import config, runtime
from test_runtime import calibrated_config, rig


def relaxed_config():
    cfg = calibrated_config()
    cfg['control'].update(imu_max_age_s=.10, max_policy_cycle_s=.08,
                          max_policy_sample_gap_s=.10)
    return cfg


def duck1_config():
    cfg = relaxed_config()
    cfg['control']['policy_overrun_threshold_s'] = .030
    return cfg


def replay_cycle_times(r, durations_ms, *, subsequent_ms=21.5):
    """Replay measured processing durations after each live target send."""
    durations = iter(durations_ms)
    read = r.ctl._read
    send = r.bus.sync_positions
    started = None
    sent_ms = []

    def read_cycle(*args, **kwargs):
        nonlocal started
        if r.ctl.session.get('status') == 'running':
            started = r.clock.now
        return read(*args, **kwargs)

    def send_cycle(positions):
        send(positions)
        if r.ctl.session.get('status') == 'running':
            assert not any(event[0] == 'torque' and not event[2]
                           for event in r.bus.events)
            duration = next(durations, subsequent_ms) / 1000
            elapsed = r.clock.now - started
            assert duration >= elapsed
            r.clock.advance(duration - elapsed)
            sent_ms.append((r.clock.now - started) * 1000)

    r.ctl._read = read_cycle
    r.bus.sync_positions = send_cycle
    return sent_ms


def add_one_imu_delay(r, delay=.04, *, observation_count=1):
    original = r.imu.latest
    delayed = False

    def latest(max_age):
        nonlocal delayed
        if (r.ctl.session.get('status') == 'running'
                and len(r.policy.observations) == observation_count and not delayed):
            r.clock.advance(delay)
            delayed = True
        return original(max_age)

    r.imu.latest = latest


def samples(r):
    return [row for line in r.path.read_text().splitlines()
            if (row := json.loads(line))['kind'] == 'sample']


def test_one_47ms_read_and_60ms_gap_continues_policy_without_intermediate_unload():
    with rig(cfg=relaxed_config()) as r:
        r.bus.read_s = .007
        add_one_imu_delay(r)
        original = r.policy.infer

        def infer(obs):
            assert not any(event[0] == 'torque' and not event[2] for event in r.bus.events)
            return original(obs)

        r.policy.infer = infer
        result = r.ctl.execute(seconds=.3, motion=True)
        rows = samples(r)
        delayed = [row for row in rows if row['this_read_ms'] > 40]
        assert len(delayed) == 1
        assert delayed[0]['this_read_ms'] == pytest.approx(47)
        assert delayed[0]['sample_gap_ms'] == pytest.approx(60)
        assert rows[-1]['t'] > delayed[0]['t'] + .1
        assert result['samples'] == len(rows) > 5
        assert r.ctl.session['max_policy_cycle_s'] == .08
        assert r.ctl.session['max_policy_sample_gap_s'] == .10
        assert r.cfg['control']['max_cycle_s'] == .04
        assert r.cfg['control']['frequency_hz'] == 50
        assert r.cfg['control']['max_overruns'] == 5
        assert r.ctl.session['status'] == 'completed'
        assert len([event for event in r.bus.events if event[0] == 'torque' and not event[2]]) == 1
        saved = json.loads(r.path.with_suffix('.session.json').read_text())
        assert saved['max_policy_cycle_s'] == .08
        assert saved['max_policy_sample_gap_s'] == .10


def test_policy_sampling_gap_over_100ms_still_stops_before_inference():
    with rig(cfg=relaxed_config()) as r:
        wait = r.ctl.stop.wait
        delayed = False

        def delayed_wait(timeout=None):
            nonlocal delayed
            result = wait(timeout)
            if r.ctl.session.get('status') == 'running' and not delayed:
                r.clock.advance(.09)
                delayed = True
            return result

        r.ctl.stop.wait = delayed_wait
        with pytest.raises(runtime.StopRequested, match='采样间隔异常'):
            r.ctl.execute(seconds=.3, motion=True)
        fault = r.ctl.session['sampling_timing_fault']
        assert fault['sample_gap_ms'] > 100
        assert fault['limit_ms'] == 100
        assert fault['policy_target_sent_this_step'] is False
        assert r.policy.observations == []
        assert r.ctl.enabled is False


@pytest.mark.parametrize('slow_send', [False, True])
def test_policy_cycle_over_80ms_logs_whether_target_was_sent(slow_send):
    with rig(cfg=relaxed_config()) as r:
        if slow_send:
            original = r.bus.sync_positions

            def send(positions):
                if r.ctl.session.get('status') == 'running':
                    r.clock.advance(.081)
                original(positions)

            r.bus.sync_positions = send
        else:
            r.policy.inference_s = .081
        with pytest.raises(runtime.StopRequested, match='超时|用时'):
            r.ctl.execute(seconds=.3, motion=True)
        fault = r.ctl.session['policy_cycle_timing_fault']
        assert fault['cycle_ms'] > 80
        assert fault['limit_ms'] == 80
        assert fault['policy_target_sent_this_step'] is slow_send
        assert fault['stage'] == ('after_target_send' if slow_send else 'before_target_send')
        assert r.ctl.enabled is False


def test_servo_read_deadline_stays_at_40ms():
    with rig(cfg=relaxed_config()) as r:
        r.bus.read_s = .041
        with pytest.raises(runtime.StopRequested, match='读取舵机反馈超过控制周期时限'):
            r.ctl.execute(seconds=.2, motion=True)
        assert r.ctl.session['servo_read_timing_fault']['limit_ms'] == 40
        assert not any(event[0] in ('positions', 'torque') for event in r.bus.events)


def test_servo_feedback_freshness_stays_at_40ms():
    with rig(cfg=relaxed_config()) as r:
        original = r.bus.read_feedback_many

        def stale(ids, sync):
            values = original(ids, sync)
            return {sid: dataclasses.replace(value, received_at=r.clock.now - .041)
                    for sid, value in values.items()}

        r.bus.read_feedback_many = stale
        with pytest.raises(runtime.StopRequested, match='反馈过期'):
            r.ctl.execute(seconds=.2, motion=True)
        assert not any(event[0] in ('positions', 'torque') for event in r.bus.events)


def test_home_motion_retains_the_40ms_deadline():
    with rig(cfg=relaxed_config()) as r:
        original = r.imu.latest

        def delayed_during_home(max_age):
            if 'home_samples' in r.ctl.session:
                r.clock.advance(.04)
            return original(max_age)

        r.imu.latest = delayed_during_home
        with pytest.raises(runtime.StopRequested, match='回HOME传感器读取超时'):
            r.ctl.execute(seconds=.2, motion=True)
        assert r.policy.observations == []
        assert r.ctl.enabled is False


def test_home_only_hold_ignores_policy_overrides():
    with rig(cfg=relaxed_config()) as r:
        add_one_imu_delay(r, observation_count=0)
        with pytest.raises(runtime.StopRequested, match='采样间隔异常'):
            r.ctl.execute(seconds=.2, motion=True, home_only=True)
        assert r.ctl.session['max_policy_cycle_s'] == .04
        assert r.ctl.session['max_policy_sample_gap_s'] == .05
        assert r.ctl.session['sampling_timing_fault']['limit_ms'] == 50
        assert r.ctl.enabled is False


def test_default_policy_still_rejects_the_same_single_read_delay():
    with rig() as r:
        r.bus.read_s = .007
        add_one_imu_delay(r)
        with pytest.raises(runtime.StopRequested, match='采样间隔异常'):
            r.ctl.execute(seconds=.3, motion=True)
        assert r.ctl.session['sampling_timing_fault']['sample_gap_ms'] == pytest.approx(60)
        assert r.ctl.session['sampling_timing_fault']['limit_ms'] == 50
        assert r.ctl.session['max_policy_cycle_s'] == .04


def test_five_consecutive_over_20ms_cycles_still_stop():
    with rig(cfg=relaxed_config()) as r:
        r.bus.read_s = .007
        r.policy.inference_s = .015
        with pytest.raises(runtime.StopRequested, match='连续控制周期超过20ms'):
            r.ctl.execute(seconds=1, motion=True)
        assert r.ctl.session['control_timing_fault']['consecutive_overruns'] == 5
        assert r.ctl.session['control_timing_fault']['threshold_ms'] == 20
        assert r.ctl.session['policy_overrun_threshold_s'] == .020
        assert r.ctl.enabled is False


def test_duck1_replays_reported_21ms_cycles_and_keeps_torque_until_requested_end():
    reported_ms = [21.07, 21.35, 21.37, 21.04, 21.67]
    with rig(cfg=duck1_config()) as r:
        sent_ms = replay_cycle_times(r, reported_ms)
        result = r.ctl.execute(seconds=1, motion=True)
        rows = samples(r)
        assert sent_ms[:5] == pytest.approx(reported_ms)
        assert [row['cycle_ms'] for row in rows[:5]] == pytest.approx(reported_ms)
        assert result['samples'] == len(rows) == len(sent_ms) > 40
        assert all(20 < row['cycle_ms'] < 30 for row in rows)
        assert 'control_timing_fault' not in r.ctl.session
        assert r.ctl.session['status'] == 'completed'
        assert r.ctl.session['policy_overrun_threshold_s'] == .030
        assert r.cfg['control']['frequency_hz'] == 50
        assert r.cfg['control']['max_overruns'] == 5
        off = [event for event in r.bus.events if event[0] == 'torque' and not event[2]]
        assert len(off) == 1
        assert off[0][1] >= r.policy.times[-1]
        saved = json.loads(r.path.with_suffix('.session.json').read_text())
        assert saved['policy_overrun_threshold_s'] == .030


def test_duck1_five_consecutive_over_30ms_cycles_still_stop_and_unload():
    overrun_ms = [31.1, 32.2, 33.3, 34.4, 35.5]
    with rig(cfg=duck1_config()) as r:
        sent_ms = replay_cycle_times(r, overrun_ms)
        with pytest.raises(runtime.StopRequested, match='连续控制周期超过30ms'):
            r.ctl.execute(seconds=1, motion=True)
        assert sent_ms == pytest.approx(overrun_ms)
        fault = r.ctl.session['control_timing_fault']
        assert fault['consecutive_overruns'] == fault['max_overruns'] == 5
        assert fault['threshold_ms'] == 30
        assert fault['cycle_ms'] == pytest.approx(overrun_ms[-1])
        assert fault['policy_target_sent_this_step'] is True
        assert len(r.policy.observations) == 5
        assert r.ctl.enabled is False
        assert any(event[0] == 'torque' and not event[2] for event in r.bus.events)


def test_cycle_below_30ms_resets_consecutive_counter_even_when_above_20ms():
    repeated_ms = [31., 32., 33., 34., 29.] * 3
    with rig(cfg=duck1_config()) as r:
        sent_ms = replay_cycle_times(r, repeated_ms)
        result = r.ctl.execute(seconds=1, motion=True)
        assert sent_ms[:len(repeated_ms)] == pytest.approx(repeated_ms)
        assert result['samples'] == len(sent_ms) > len(repeated_ms)
        assert 'control_timing_fault' not in r.ctl.session
        assert r.ctl.session['status'] == 'completed'


def test_home_only_consecutive_limit_stays_at_20ms_with_duck1_override():
    with rig(cfg=duck1_config()) as r:
        sent_ms = replay_cycle_times(r, [21.07, 21.35, 21.37, 21.04, 21.67])
        with pytest.raises(runtime.StopRequested, match='连续控制周期超过20ms'):
            r.ctl.execute(seconds=1, motion=True, home_only=True)
        assert len(sent_ms) == 5
        assert r.policy.observations == []
        assert r.ctl.session['policy_overrun_threshold_s'] == .020
        fault = r.ctl.session['control_timing_fault']
        assert fault['consecutive_overruns'] == 5
        assert fault['threshold_ms'] == 20
        assert r.ctl.enabled is False
        assert any(event[0] == 'torque' and not event[2] for event in r.bus.events)


@pytest.mark.parametrize('home_only', [False, True])
@pytest.mark.parametrize('override', [
    {'policy_overrun_threshold_s': 0}, {'policy_overrun_threshold_s': -.01},
    {'policy_overrun_threshold_s': .019}, {'policy_overrun_threshold_s': .041},
    {'policy_overrun_threshold_s': float('nan')},
    {'policy_overrun_threshold_s': float('inf')},
    {'policy_overrun_threshold_s': True}, {'policy_overrun_threshold_s': False},
    {'policy_overrun_threshold_s': '.03'}, {'policy_overrun_threshold_s': None},
    {'policy_overrun_threshold_s': .030, 'max_policy_cycle_s': .025},
])
def test_invalid_overrun_threshold_rejected_before_device_access(override, home_only):
    cfg = relaxed_config()
    cfg['control'].update(override)
    with rig(owned=True, cfg=cfg) as r:
        with pytest.raises(ValueError, match='policy_overrun_threshold_s'):
            r.ctl.execute(seconds=.2, motion=True, home_only=home_only)
        r.factory.assert_not_called()
        assert r.bus.events == []
        assert not r.path.with_suffix('.session.json').exists()


@pytest.mark.parametrize('threshold', [.020, .030, .040])
def test_load_config_preserves_valid_policy_overrun_threshold(threshold, tmp_path):
    cfg = relaxed_config()
    cfg['control']['policy_overrun_threshold_s'] = threshold
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    loaded = config.load_config(path)
    assert config.policy_overrun_threshold(loaded['control']) == threshold
    assert loaded['control']['frequency_hz'] == 50


def test_load_config_rejects_invalid_policy_overrun_threshold(tmp_path):
    cfg = relaxed_config()
    cfg['control']['policy_overrun_threshold_s'] = .041
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='policy_overrun_threshold_s'):
        config.load_config(path)


@pytest.mark.parametrize('override', [
    {'max_policy_cycle_s': 0}, {'max_policy_cycle_s': -.01},
    {'max_policy_cycle_s': .101}, {'max_policy_cycle_s': float('nan')},
    {'max_policy_cycle_s': True}, {'max_policy_cycle_s': '.08'},
    {'max_policy_sample_gap_s': 0}, {'max_policy_sample_gap_s': .121},
    {'max_policy_sample_gap_s': float('inf')}, {'max_policy_sample_gap_s': False},
    {'max_policy_sample_gap_s': .079},
])
def test_invalid_policy_limits_rejected_before_device_access(override):
    cfg = relaxed_config()
    cfg['control'].update(override)
    with rig(cfg=cfg) as r:
        with pytest.raises(ValueError, match='max_policy_'):
            r.ctl.execute(seconds=.2, motion=True)
        assert r.bus.events == []
        assert not r.path.with_suffix('.session.json').exists()


def test_load_config_validates_overrides_without_adding_new_defaults(tmp_path):
    cfg = config.default_config()
    assert 'max_policy_cycle_s' not in cfg['control']
    assert 'max_policy_sample_gap_s' not in cfg['control']
    assert 'policy_overrun_threshold_s' not in cfg['control']
    assert config.policy_overrun_threshold(cfg['control']) == .020
    assert config.policy_timing_limits(cfg['control']) == (.04, .05)
    cfg['control']['max_cycle_s'] = .12
    assert config.policy_timing_limits(cfg['control']) == (.12, .13)
    cfg = relaxed_config()
    path = tmp_path / 'robot.json'
    path.write_text(json.dumps(cfg))
    assert config.policy_timing_limits(config.load_config(path)['control']) == (.08, .1)
    cfg['control']['max_policy_sample_gap_s'] = .04
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='不得小于'):
        config.load_config(path)
