"""Local calibration; no EEPROM writes and no assumed encoder zero."""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .policy import JOINT_NAMES, DEFAULT_SERVO_IDS, HOME, policy_calibration_reference

ROOT = Path(__file__).resolve().parents[1]


def default_config():
    return {
        "schema": 1,
        "model": "models/hd1910_flat.onnx",
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
            "servo_voltage_range": [7.0, 8.4], "servo_max_temperature_c": 60,
            "max_target_jump_rad": 0.6,
            "ram_p": 5, "ram_d": 0, "ram_i": 0,
        },
    }


def load_config(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"没有配置文件 {path}；先运行 init。")
    cfg = json.loads(path.read_text())
    if cfg.get("schema") != 1:
        raise ValueError("不支持的配置版本")
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
        raise ValueError("尚未标定；按已知姿态摆好，再运行 calibrate --pose straight、zero 或 home。")
    z = np.asarray([j.get("zero_tick") for j in cfg["joints"]], dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("关节零位缺失或不是有限数")
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
