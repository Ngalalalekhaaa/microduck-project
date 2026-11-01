"""HOME settle retains the existing body hold and writes only mouth targets."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import supported_home as supported
import microduck_deploy.jaw as jaw_module
from microduck_deploy.config import calibration_arrays
from microduck_deploy.policy import HOME, q_to_ticks
from microduck_deploy.servo import ServoFeedback
from test_home_recovery_distance import recovery_config
from test_runtime import Clock, TimedEvent


@pytest.mark.parametrize('mouth_active,late_prepare', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('body_ids,mouth_id', [
    (list(range(1, 15)), 15),
    (list(range(1, 10)) + list(range(11, 16)), 10),
])
def test_settle_never_resends_body_goals_and_checks_deadline(
        tmp_path, monkeypatch, mouth_active, late_prepare, body_ids, mouth_id):
    cfg = recovery_config()
    for joint, sid in zip(cfg['joints'], body_ids):
        joint['id'] = sid
    cfg['calibration']['joint_ids'] = body_ids.copy()
    cfg['jaw'] = {'enabled': True, 'id': mouth_id, 'mapping_verified': True,
                  'closed_tick': 344, 'open_tick': 51}
    original_cfg = copy.deepcopy(cfg)
    home = q_to_ticks(HOME, *calibration_arrays(cfg))
    clock, writes = Clock(), []
    gains = {'p': 6, 'd': 20, 'i': 0}

    class Bus:
        soft_gain_write_allowed = False

        def __init__(self, **options):
            self.torque = 0
            self.communication_retries = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def read_configuration(self, sid):
            return {'torque': self.torque, 'mode': 4, 'angle_resolution': 1,
                    'position_min': 0, 'position_max': 4095, 'phase': 0,
                    'ram_gains': dict(gains), 'eeprom_gains': dict(gains)}

        def read_feedback_many(self, ids, *, sync):
            assert ids == body_ids + [mouth_id] and sync
            clock.advance(.001)
            ticks = dict(zip(body_ids, map(int, home))) | {mouth_id: 344}
            return {sid: ServoFeedback(sid, ticks[sid], 0, 0, 7.4, 30, 0,
                        0, ticks[sid], 0, clock.now) for sid in ids}

        def sync_positions(self, targets):
            assert targets
            writes.append(dict(targets))

        def set_torque(self, ids, enabled, *, verify):
            self.torque = int(enabled)

    class Jaw:
        def __init__(self, configuration, bus, command_source):
            self.calls = 0

        def preflight(self):
            pass

        def warmup(self):
            pass

        def targets(self, *, feedback):
            self.calls += 1
            if late_prepare and self.calls == 3:  # First settle iteration.
                clock.advance(.201)
            return {mouth_id: 344} if mouth_active else {}

        def disable(self, *, restore_gains):
            pass

    class Source:
        def __init__(self, fd):
            pass

        def close(self):
            pass

    monkeypatch.setattr(supported, 'DEPLOY', tmp_path)
    monkeypatch.setattr(supported.sys, 'argv', ['supported_home.py', '--move',
                        '--log-only-feedback', '--jaw-command-fd', '987'])
    monkeypatch.setattr(supported, 'load_config', lambda path: cfg)
    monkeypatch.setattr(supported, 'PositionOnlyBus', Bus)
    monkeypatch.setattr(supported, 'plan_straight_recovery', lambda *args, **kwargs:
                        {'home_ticks': home.tolist(), 'duration_s': .02,
                         'steps': 1, 'peak_target_speed_deg_s': 0})
    monkeypatch.setattr(jaw_module, 'JawController', Jaw)
    monkeypatch.setattr(jaw_module, 'JawCommandReader', Source)
    monkeypatch.setattr(supported.time, 'monotonic', clock.monotonic)
    monkeypatch.setattr(supported.threading, 'Event', lambda: TimedEvent(clock))
    monkeypatch.setattr(supported.signal, 'signal', lambda *args: None)

    if late_prepare:
        with pytest.raises(ValueError, match='200ms'):
            supported.main()
    else:
        supported.main()
    # Preload and one interpolated step are the only body goal packets. The
    # settled body continues its previous hardware hold without repeated writes.
    assert len([targets for targets in writes if set(targets).intersection(body_ids)]) == 2
    assert all(set(targets) == {mouth_id} for targets in writes[2:])
    assert len(writes[2:]) == (50 if mouth_active and not late_prepare else 0)
    assert cfg == original_cfg
