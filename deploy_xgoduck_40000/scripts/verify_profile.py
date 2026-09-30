"""Offline contract checks. Does not open serial or I2C devices."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify():
    import numpy as np
    from microduck_deploy.config import load_config, calibration_arrays, require_calibration
    from microduck_deploy.policy import HOME, OnnxPolicy, build_obs61, action_to_targets, q_to_ticks
    from microduck_deploy.homing import validate_reference_calibration, plan_home

    cfg = load_config(ROOT / 'robot.json')
    require_calibration(cfg)
    validate_reference_calibration(cfg, 'straight')
    model = ROOT / cfg['model']
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != manifest['model_sha256']:
        raise ValueError('新模型 SHA256 不匹配')
    policy = OnnxPolicy(model)
    obs = build_obs61([0, 0, 0], [0, 0, -1], HOME, np.zeros(14),
                      np.zeros(14), twist=(0.1, 0, 0))
    np.testing.assert_array_equal(obs[6:20], np.zeros(14))
    np.testing.assert_array_equal(action_to_targets(np.zeros(14)), HOME)
    action = policy.infer(obs)
    assert action.shape == (14,) and np.all(np.isfinite(action))
    z, d = calibration_arrays(cfg)
    ticks = q_to_ticks(HOME, z, d).tolist()
    assert plan_home(cfg, HOME)['home_ticks'] == ticks
    from supported_home import plan_straight_recovery, DEPLOY
    assert DEPLOY == ROOT
    assert plan_straight_recovery(cfg, cfg['calibration']['captured_ticks'],
                                 target='home')['home_ticks'] == ticks
    print(json.dumps({'model': str(model), 'sha256': digest, 'home_ticks': ticks,
                      'offline_check': 'passed', 'hardware_access': False}, ensure_ascii=False), flush=True)
    return cfg


if __name__ == '__main__':
    verify()
