"""Read-only planning and bounded permission to move from a known reference.

This verifies saved numeric relationships, not the actual mechanical geometry.
No hardware access, policy inference, EEPROM write, or implicit motor enable.
"""
from __future__ import annotations

import math
import numpy as np

from .config import calibration_arrays, require_calibration
from .policy import HOME, JOINT_NAMES, JOINT_LIMITS, STRAIGHT_REFERENCE, q_to_ticks, ticks_to_q

REFERENCE_TOLERANCE_RAD = float(np.deg2rad(5.0))
HOME_TRACKING_LIMIT_RAD = float(np.deg2rad(10.0))
REFERENCE_MIN_SECONDS = 8.0
REFERENCE_MAX_SPEED_RAD_S = float(np.deg2rad(12.0))


def validate_reference_calibration(cfg, from_pose):
    if from_pose != "straight":
        raise ValueError("参考姿态入口只支持 --from-pose straight")
    require_calibration(cfg, imu=False)
    saved = cfg["calibration"]
    ids = [j["id"] for j in cfg["joints"]]
    zero, direction = calibration_arrays(cfg)
    if (saved.get("pose") != "straight" or saved.get("joint_ids") != ids
            or saved.get("directions") != direction.tolist()):
        raise ValueError("缺少匹配的直腿标定记录或ID/方向已变；重新 calibrate --pose straight")
    ref = np.asarray(saved.get("reference_q_rad"), dtype=float)
    ticks = np.asarray(saved.get("captured_ticks"), dtype=float)
    if (ref.shape != (14,) or ticks.shape != (14,)
            or not np.allclose(ref, STRAIGHT_REFERENCE, rtol=0, atol=1e-10)
            or not np.allclose(ticks_to_q(ticks, zero, direction), ref, rtol=0, atol=1e-10)):
        raise ValueError("直腿参考角、记录刻度与软件零位不一致；重新标定，不要手改记录")


def plan_home(cfg, q, *, from_pose=None, move_seconds=None):
    require_calibration(cfg, imu=False)
    q = np.asarray(q, dtype=float)
    if q.shape != (14,) or not np.all(np.isfinite(q)):
        raise ValueError("当前位置必须是14个有限角度")
    if from_pose is not None:
        validate_reference_calibration(cfg, from_pose)
        distance = np.abs(q - STRAIGHT_REFERENCE)
        bad = np.flatnonzero(distance > REFERENCE_TOLERANCE_RAD)
        if len(bad):
            raise ValueError("当前姿态偏离直腿标定参考超过5°：" + ", ".join(
                f"{JOINT_NAMES[i]} {np.rad2deg(distance[i]):.2f}°" for i in bad))
        duration = REFERENCE_MIN_SECONDS if move_seconds is None else float(move_seconds)
        if not math.isfinite(duration) or not REFERENCE_MIN_SECONDS <= duration <= 30:
            raise ValueError("从直腿移到HOME的 --move-seconds 必须在8..30秒之间")
    else:
        if np.max(np.abs(q - HOME)) > cfg["control"]["max_start_distance_rad"]:
            raise ValueError("启动姿态离HOME太远；摆到接近HOME，或使用已标定的 --from-pose straight")
        duration = float(cfg["control"]["home_duration_s"] if move_seconds is None else move_seconds)
        if not math.isfinite(duration) or not 0 < duration <= 30:
            raise ValueError("回HOME时长必须在0..30秒之间")
    # Both sets are convex intervals. Valid endpoints imply valid smoothstep
    # interpolation, and runtime still validates every outgoing target.
    endpoints = np.stack([q, HOME])
    if np.any(endpoints < JOINT_LIMITS[:, 0]) or np.any(endpoints > JOINT_LIMITS[:, 1]):
        raise ValueError("回HOME路径端点超出模型关节限位")
    zero, direction = calibration_arrays(cfg)
    start_ticks, home_ticks = q_to_ticks(endpoints[0], zero, direction), q_to_ticks(HOME, zero, direction)
    peak = 1.5 * float(np.max(np.abs(HOME - q))) / duration
    if from_pose is not None and peak > REFERENCE_MAX_SPEED_RAD_S:
        raise ValueError("回HOME目标峰值速度超过12°/s；增大 --move-seconds")
    return {"from_pose": from_pose, "duration_s": duration,
            "start_q_rad": q.tolist(), "home_q_rad": HOME.tolist(),
            "start_ticks": start_ticks.tolist(), "home_ticks": home_ticks.tolist(),
            "peak_target_speed_deg_s": float(np.rad2deg(peak)),
            "joints": [{"name": name, "id": joint["id"], "direction": joint["direction"],
                        "current_tick": int(start_ticks[i]), "home_tick": int(home_ticks[i]),
                        "current_deg": float(np.rad2deg(q[i])), "home_deg": float(np.rad2deg(HOME[i])),
                        "move_deg": float(np.rad2deg(HOME[i] - q[i]))}
                       for i, (name, joint) in enumerate(zip(JOINT_NAMES, cfg["joints"]))]}


def check_servo_limits(plan, configurations):
    """Validate both path endpoints against enabled single-turn firmware limits."""
    for row in plan["joints"]:
        sid = row["id"]
        conf = configurations[sid]
        if conf["torque"] != 0 or conf["mode"] != 4 or conf["angle_resolution"] != 1:
            raise ValueError(f"ID{sid}需要先卸力，且mode=4、angle_resolution=1")
        lo, hi = conf.get("position_min"), conf.get("position_max")
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= 4095:
            raise ValueError(f"ID{sid}单圈硬件角限位不明确：{lo}..{hi}；先inspect核对，不自动改写")
        if not all(lo <= row[key] <= hi for key in ("current_tick", "home_tick")):
            raise ValueError(f"ID{sid}当前到HOME路径超出舵机硬件限位{lo}..{hi}")
