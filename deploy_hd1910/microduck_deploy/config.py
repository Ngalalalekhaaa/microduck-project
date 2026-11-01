"""Local calibration; no EEPROM writes and no assumed encoder zero."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .policy import JOINT_NAMES, JOINT_LIMITS, DEFAULT_SERVO_IDS, HOME, policy_calibration_reference

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG = math.degrees(
    JOINT_LIMITS[JOINT_NAMES.index('head_pitch'), 1] + .12)


def roulade_head_pitch_max_deg(control):
    """Explicit positive feedback bound for head_pitch during a roll only.

    This software observation bound does not establish a measured physical
    travel limit or change the model, servo EEPROM, zero, or target commands.
    Missing configuration preserves the established model bound + 0.12 rad.
    """
    value = control.get('roulade_head_pitch_max_deg', DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG <= value <= 110.):
        raise ValueError('control.roulade_head_pitch_max_deg 必须为 '
                         f'[{DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG:.6f},110] 度内的有限数字')
    return float(value)


def policy_timing_limits(control):
    """Optional policy budgets, separate from servo/HOME max_cycle_s.

    Absent keys retain the original budgets, including older configurations.
    Explicit policy overrides are bounded and must allow a full policy cycle.
    """
    cycle = control.get('max_policy_cycle_s', float(control['max_cycle_s']))
    gap = control.get('max_policy_sample_gap_s', float(control['max_cycle_s']) + .01)
    for key, value, upper in (
            ('max_policy_cycle_s', cycle, .10),
            ('max_policy_sample_gap_s', gap, .12)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
                or (key in control and value > upper)):
            raise ValueError(f'control.{key} 必须为(0,{upper}]内的有限数字')
    if gap < cycle:
        raise ValueError('control.max_policy_sample_gap_s 不得小于 max_policy_cycle_s')
    return float(cycle), float(gap)


def policy_overrun_threshold(control):
    """Consecutive policy overrun threshold; the 50 Hz schedule is unchanged."""
    if 'policy_overrun_threshold_s' not in control:
        return .020
    value = control['policy_overrun_threshold_s']
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not .020 <= value <= .040):
        raise ValueError('control.policy_overrun_threshold_s 必须为[0.020,0.040]内的有限数字')
    if value > policy_timing_limits(control)[0]:
        raise ValueError('control.policy_overrun_threshold_s 不得大于策略单周期时限')
    return float(value)


def default_config():
    return {
        "schema": 1,
        "model": "models/hd1910_flat.onnx",
        "sitstand_model": "models/hd1910_sitstand_20261015.onnx",
        "roulade_model": None,
        "updated_walking_model": None,
        "sitstand_friction_model": None,
        "sitstand_com_model": None,
        "serial": {"port": "/dev/ttyS2", "baudrate": 1000000, "timeout": 0.015,
                   "discard_echo": False, "register_profile": "hls_2"},
        "imu": {"transport": "i2c", "i2c_bus": None, "i2c_address": None,
                "mounting_rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "imu_mount_verified": False,
        "joints": [
            {"name": name, "id": int(sid), "direction": None, "zero_tick": None,
             "mapping_verified": False}
            for name, sid in zip(JOINT_NAMES, DEFAULT_SERVO_IDS)
        ],
        "calibration": None,
        "control": {
            "frequency_hz": 50, "home_duration_s": 3.0,
            "max_start_distance_rad": 0.60, "home_tolerance_rad": 0.15,
            "tilt_limit_deg": 35.0, "imu_max_age_s": 0.06,
            "max_cycle_s": 0.04, "max_overruns": 5,
            "servo_voltage_range": [4.0, 8.4], "servo_max_temperature_c": 60,
            "max_target_jump_rad": 0.6,
            "ram_p": 5, "ram_d": 0, "ram_i": 0,
            "roulade_enabled": False,
            "model_suite_buttons_enabled": False,
            "roulade_head_pitch_max_deg": DEFAULT_ROULADE_HEAD_PITCH_MAX_DEG,
        },
    }


def load_config(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"没有配置文件 {path}；先运行 init。")
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != 1:
        raise ValueError("不支持的配置版本")
    for key in ('sitstand_model', 'ground_pick_model', 'roulade_model',
                'updated_walking_model', 'sitstand_friction_model', 'sitstand_com_model'):
        model = cfg.get(key)
        if model is not None and (not isinstance(model, str) or not model.strip()
                                  or Path(model).suffix.lower() != '.onnx'):
            raise ValueError(f'{key} 须为 ONNX 文件路径或 null')
    if cfg.get('ground_pick_model') and not cfg.get('sitstand_model'):
        raise ValueError('俯身动作需要 sitstand_model 用于完成后的站立保持')
    if not isinstance(cfg['control'].get('log_only_target_angles', False), bool):
        raise ValueError('control.log_only_target_angles 必须为布尔值')
    enabled = cfg['control'].get('roulade_enabled', False)
    if not isinstance(enabled, bool):
        raise ValueError('control.roulade_enabled 必须为布尔值')
    if enabled and (not cfg.get('roulade_model') or not cfg.get('sitstand_model')):
        raise ValueError('滚翻启用需要 roulade_model 和 sitstand_model')
    suite_enabled = cfg['control'].get('model_suite_buttons_enabled', False)
    if not isinstance(suite_enabled, bool):
        raise ValueError('control.model_suite_buttons_enabled 必须为布尔值')
    if suite_enabled:
        required = ('updated_walking_model', 'sitstand_friction_model',
                    'sitstand_com_model', 'sitstand_model', 'roulade_model')
        missing = [key for key in required if not cfg.get(key)]
        if missing or not enabled:
            raise ValueError('方向键模型切换需要配置 ' + ', '.join(required)
                             + '，并启用 control.roulade_enabled')
    for key, default, minimum, maximum in (
        ('roulade_timeout_s', 5., 0., 10.),
        ('roulade_lateral_limit_deg', 35., 0., 60.),
        ('roulade_max_angular_speed_rad_s', 12., 0., 20.),
        ('roulade_completion_tilt_deg', 25., 0., 35.),
        ('roulade_settle_seconds', .25, 0., 2.),
    ):
        value = float(cfg['control'].get(key, default))
        if not math.isfinite(value) or not minimum < value <= maximum:
            raise ValueError(f'control.{key} 必须在({minimum},{maximum}]内')
    roulade_head_pitch_max_deg(cfg['control'])
    joints = cfg.get("joints", [])
    if [x.get("name") for x in joints] != list(JOINT_NAMES):
        raise ValueError("joints 顺序/名称与模型不匹配；不能删除嘴部之外的14个关节。")
    ids = [x["id"] for x in joints]
    if any(type(x) is not int or not 0 <= x <= 253 for x in ids):
        raise ValueError("舵机ID必须在0..253内")
    confirmed_ids = [j['id'] for j in joints if j.get('mapping_verified')]
    if len(set(confirmed_ids)) != len(confirmed_ids):
        raise ValueError("已确认关节的ID不能重复")
    if cfg["control"]["frequency_hz"] != 50:
        raise ValueError("这份模型固定以50Hz运行")
    for key in ("home_duration_s", "max_start_distance_rad", "home_tolerance_rad",
                "tilt_limit_deg", "imu_max_age_s", "max_cycle_s", "max_target_jump_rad"):
        v = float(cfg["control"][key])
        if not math.isfinite(v) or v <= 0:
            raise ValueError(f"control.{key} 必须为正有限数")
    ground_tilt = float(cfg['control'].get('ground_pick_tilt_limit_deg', 50.))
    if not math.isfinite(ground_tilt) or not 0 < ground_tilt <= 90:
        raise ValueError('control.ground_pick_tilt_limit_deg 必须在(0,90]内')
    policy_timing_limits(cfg['control'])
    policy_overrun_threshold(cfg['control'])
    return cfg


def save_config(path, cfg):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + ".bak." + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def apply_joint_ids(cfg, profile):
    """Apply a complete name-to-ID map; never infer directions or encoder zeros."""
    if not isinstance(profile, dict) or set(profile) != set(JOINT_NAMES):
        raise ValueError("ID表必须恰好包含模型的14个关节名称")
    ids = [profile[name] for name in JOINT_NAMES]
    if any(type(sid) is not int or not 0 <= sid <= 253 for sid in ids):
        raise ValueError("舵机ID必须为0..253内的整数")
    if len(set(ids)) != len(ids):
        raise ValueError("ID表中的14个ID必须互不相同")
    # Validate the entire profile before changing any mapping or calibration.
    changed = [j['name'] for j in cfg['joints'] if j['id'] != profile[j['name']]]
    for joint in cfg['joints']:
        if joint['name'] in changed:
            joint.update(id=profile[joint['name']], direction=None, zero_tick=None,
                         mapping_verified=False)
    if changed:
        cfg['calibration'] = None
    return changed


def require_mapping(cfg):
    missing = [j["name"] for j in cfg["joints"]
               if not j.get("mapping_verified") or j.get("direction") not in (-1, 1)]
    if missing:
        raise ValueError("先逐关节核对ID/正方向并运行 map-joint：" + ", ".join(missing))
    if len({j['id'] for j in cfg['joints']}) != 14:
        raise ValueError("14个已确认关节的ID必须互不相同")


def require_calibration(cfg, *, imu=True):
    require_mapping(cfg)
    if not cfg.get("calibration"):
        raise ValueError("尚未标定；运行 calibrate --pose straight/zero/home，或按已核对几何运行 assembly-calibrate --apply。")
    z = np.asarray([j.get("zero_tick") for j in cfg["joints"]], dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("关节零位缺失或不是有限数")
    if cfg['calibration'].get('pose') == 'assembly':
        from .assembly import assembly_reference
        assembly_reference(cfg)
    if imu and not cfg.get("imu_mount_verified"):
        raise ValueError("先用 imu 检查前后/左右倾斜方向，再用 imu-map 保存已核对的轴映射。")


def calibration_arrays(cfg):
    return (np.asarray([j["zero_tick"] for j in cfg["joints"]], dtype=float),
            np.asarray([j["direction"] for j in cfg["joints"]], dtype=float))


def record_reference(cfg, ticks, pose, reference):
    require_mapping(cfg)
    ticks = np.asarray(ticks, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if (ticks.shape != (14,) or reference.shape != (14,)
            or not np.all(np.isfinite(ticks)) or not np.all(np.isfinite(reference))
            or np.any(ticks != np.rint(ticks))):
        raise ValueError("标定需要14个有效角度")
    if np.any(ticks < 0) or np.any(ticks > 4095):
        raise ValueError("当前只支持单圈0..4095读数；不能对负数/多圈读数取模后继续。")
    if not np.allclose(reference, policy_calibration_reference(pose), rtol=0, atol=1e-10):
        raise ValueError("参考角与所选姿态不一致；不能把直腿姿态当成ZERO或HOME")
    direction = np.asarray([j["direction"] for j in cfg["joints"]])
    zeros = ticks - direction * reference * (4096 / (2 * np.pi))
    home_ticks = zeros + direction * HOME * (4096 / (2 * np.pi))
    if np.any(home_ticks < 0) or np.any(home_ticks > 4095):
        raise ValueError("HOME将跨过编码器单圈边界；需调整机械安装/专门校零，不能取模跳过边界。")
    for joint, value in zip(cfg["joints"], zeros):
        joint["zero_tick"] = float(value)
    cfg["calibration"] = {"pose": pose, "captured_ticks": ticks.tolist(),
                          "reference_q_rad": reference.tolist(),
                          "joint_ids": [j["id"] for j in cfg["joints"]],
                          "directions": direction.tolist(),
                          "home_ticks": np.rint(home_ticks).astype(int).tolist(),
                          "utc": datetime.now(timezone.utc).isoformat()}


def axes_rotation(axes):
    if len(axes) != 3:
        raise ValueError("需要3个轴，例如 +x +y +z")
    rotation = np.zeros((3, 3))
    for row, axis in enumerate(axes):
        if axis not in ("+x", "-x", "+y", "-y", "+z", "-z"):
            raise ValueError("轴格式为+x/-x/+y/-y/+z/-z")
        rotation[row, "xyz".index(axis[1])] = 1 if axis[0] == "+" else -1
    if not np.allclose(rotation @ rotation.T, np.eye(3)) or np.linalg.det(rotation) < 0.99:
        raise ValueError("轴必须互不重复且构成右手坐标系；不要用镜像变换")
    return rotation.tolist()
