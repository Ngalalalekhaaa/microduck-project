"""Derive software calibration from a verified mechanical assembly datum.

No serial access, motor commands, or EEPROM changes. Unknown encoder-to-hole
phase is deliberately left unset in templates: eight holes establish a pitch,
not the angle of those holes at encoder count 2048.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np

from .policy import HOME, JOINT_LIMITS, JOINT_NAMES, q_to_ticks

PROFILE_TYPE = "microduck_assembly_geometry"
TICKS_PER_DEGREE = 4096.0 / 360.0


def assembly_template(cfg):
    """Copy this robot's mapping, never guess a hole phase or mounting step."""
    return {
        "schema": 1,
        "type": PROFILE_TYPE,
        "geometry_source": None,
        "hole_pitch_deg": 45,
        "joints": [
            {"name": j["name"], "id": j["id"], "direction": j.get("direction"),
             "center_tick": 2048, "base_q_deg": None, "hole_steps": None,
             "datum": None, "verified": False}
            for j in cfg["joints"]
        ],
    }


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 未填写：需要可追溯的几何/中位孔型基准")
    return value


def _number(value, name):
    if (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f"{name} 必须为已确定的有限数字，不能用 null/猜测值代替")
    return float(value)


def _profile_hash(profile):
    encoded = json.dumps(profile, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compile_assembly(cfg, profile):
    """Validate all 14 datums and return a calibration plus reviewable targets.

    base_q_deg is the model joint angle at center_tick with mounting step 0.
    It already includes encoder/hole phase, case orientation and link geometry.
    hole_steps counts 45-degree increments in the MODEL-positive direction;
    direction independently maps increasing encoder counts to model angle.
    Neither angle nor encoder counts are silently reduced modulo a turn.
    """
    from .config import require_mapping

    require_mapping(cfg)
    if not isinstance(profile, dict):
        raise ValueError("装配几何文件必须是 JSON 对象")
    allowed = {"schema", "type", "geometry_source", "hole_pitch_deg", "joints"}
    if set(profile) != allowed:
        raise ValueError("装配几何顶层字段不完整或未知；请使用 assembly-template")
    if type(profile["schema"]) is not int or profile["schema"] != 1 or profile["type"] != PROFILE_TYPE:
        raise ValueError("不支持的装配几何版本/类型")
    _text(profile["geometry_source"], "geometry_source")
    if _number(profile["hole_pitch_deg"], "hole_pitch_deg") != 45.0:
        raise ValueError("此入口只支持已确认的8孔、45°安装档位")
    rows = profile["joints"]
    if (not isinstance(rows, list) or len(rows) != 14
            or any(not isinstance(row, dict) for row in rows)
            or [row.get("name") for row in rows] != list(JOINT_NAMES)):
        raise ValueError("装配表必须按模板顺序包含14个模型关节，不能重复、遗漏或包含嘴部")
    if [j.get("name") for j in cfg["joints"]] != list(JOINT_NAMES):
        raise ValueError("robot.json 关节顺序与模型不一致")

    reference_deg = []
    fields = {"name", "id", "direction", "center_tick", "base_q_deg", "hole_steps", "datum", "verified"}
    for joint, row in zip(cfg["joints"], rows):
        name = joint["name"]
        if set(row) != fields:
            raise ValueError(f"{name}: 字段不完整或未知；请使用 assembly-template")
        if type(row["id"]) is not int or row["id"] != joint["id"]:
            raise ValueError(f"{name}: 装配表ID与已确认映射不一致")
        if (type(row["direction"]) is not int or row["direction"] not in (-1, 1)
                or row["direction"] != joint["direction"]):
            raise ValueError(f"{name}: direction须为已实测且与robot.json一致的±1")
        if type(row["center_tick"]) is not int or row["center_tick"] != 2048:
            raise ValueError(f"{name}: 此装配入口以2048为中立刻度，不修改内部零偏")
        base = _number(row["base_q_deg"], f"{name}.base_q_deg")
        steps = row["hole_steps"]
        if type(steps) is not int or not -7 <= steps <= 7:
            raise ValueError(f"{name}: hole_steps须为-7..7的有符号整数档，不能把角度四舍五入成档号")
        _text(row["datum"], f"{name}.datum")
        if row["verified"] is not True:
            raise ValueError(f"{name}: 中位孔型、装配基准、实际档位尚未确认(verified)")
        reference_deg.append(base + 45.0 * steps)

    reference = np.deg2rad(reference_deg)
    direction = np.asarray([j["direction"] for j in cfg["joints"]], dtype=float)
    ticks = np.asarray([row["center_tick"] for row in rows], dtype=int)
    bad = np.flatnonzero((reference < JOINT_LIMITS[:, 0] - 1e-12)
                         | (reference > JOINT_LIMITS[:, 1] + 1e-12))
    if len(bad):
        raise ValueError("装配参考角超出模型关节限位；检查基准/方向/档位，不取模：" +
                         ", ".join(f"{JOINT_NAMES[i]}={reference_deg[i]:.4f}°" for i in bad))
    zero = ticks - direction * np.asarray(reference_deg) * TICKS_PER_DEGREE
    home_ticks = q_to_ticks(HOME, zero, direction)
    raw_limits = zero[:, None] + direction[:, None] * JOINT_LIMITS * (4096.0 / (2 * np.pi))
    raw_min, raw_max = np.min(raw_limits, axis=1), np.max(raw_limits, axis=1)
    bad = np.flatnonzero((raw_min < 0) | (raw_max > 4095))
    if len(bad):
        raise ValueError("模型关节行程会跨出单圈0..4095；需重新选择装配档/基准，不能取模：" +
                         ", ".join(JOINT_NAMES[i] for i in bad))

    calibration = {
        "pose": "assembly", "source": "verified_assembly_geometry",
        "reference_ticks": ticks.tolist(), "reference_q_rad": reference.tolist(),
        "joint_ids": [j["id"] for j in cfg["joints"]], "directions": direction.astype(int).tolist(),
        "home_ticks": home_ticks.tolist(), "model_home_rad": HOME.tolist(),
        "assembly_profile": copy.deepcopy(profile), "profile_sha256": _profile_hash(profile),
        "hardware_read": False,
    }
    table = [
        {"name": name, "id": joint["id"], "direction": joint["direction"],
         "center_tick": int(ticks[i]), "base_q_deg": float(rows[i]["base_q_deg"]),
         "hole_steps": rows[i]["hole_steps"], "reference_deg": float(reference_deg[i]),
         "zero_tick": float(zero[i]), "home_deg": float(np.rad2deg(HOME[i])),
         "home_tick": int(home_ticks[i]), "move_deg": float(np.rad2deg(HOME[i] - reference[i])),
         "model_range_deg": np.rad2deg(JOINT_LIMITS[i]).tolist(),
         "model_range_ticks": [float(raw_min[i]), float(raw_max[i])]}
        for i, (name, joint) in enumerate(zip(JOINT_NAMES, cfg["joints"]))
    ]
    return {"calibration": calibration, "joints": table, "hardware_read": False,
            "motion_sent": False, "eeprom_written": False,
            "note": "几何计算不证明实机中位孔型、装配档位或运动路径无碰撞；须完成实体核对及只读home-plan。"}


def record_assembly(cfg, profile):
    """Apply a fully validated calculation to an in-memory robot config."""
    report = compile_assembly(cfg, profile)
    for joint, row in zip(cfg["joints"], report["joints"]):
        joint["zero_tick"] = row["zero_tick"]
    cfg["calibration"] = copy.deepcopy(report["calibration"])
    cfg["calibration"]["utc"] = datetime.now(timezone.utc).isoformat()
    return report


def assembly_reference(cfg):
    """Recompute saved geometry before using it for control or observations."""
    saved = cfg.get("calibration")
    if not isinstance(saved, dict) or saved.get("pose") != "assembly":
        raise ValueError("缺少装配几何标定；先运行 assembly-calibrate")
    report = compile_assembly(cfg, saved.get("assembly_profile"))
    expected = report["calibration"]
    for key in ("source", "profile_sha256", "joint_ids", "directions", "hardware_read"):
        if saved.get(key) != expected[key]:
            raise ValueError(f"装配标定{key}与当前几何/ID/方向不一致；重新生成标定")
    for key in ("reference_ticks", "reference_q_rad", "home_ticks", "model_home_rad"):
        try:
            value = np.asarray(saved.get(key), dtype=float)
            valid = (value.shape == (14,) and np.all(np.isfinite(value))
                     and np.allclose(value, expected[key], rtol=0, atol=1e-10))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"装配标定{key}与当前几何/模型不一致；重新生成标定")
    zero = np.asarray([j.get("zero_tick") for j in cfg["joints"]], dtype=float)
    if not np.allclose(zero, [row["zero_tick"] for row in report["joints"]], rtol=0, atol=1e-10):
        raise ValueError("软件零位与装配几何不一致；不要手改zero_tick或冒充straight标定")
    return np.asarray(expected["reference_ticks"], dtype=int), np.asarray(expected["reference_q_rad"])
