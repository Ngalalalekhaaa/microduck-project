"""Math for checking HOME and adjusting a standing body's roll and pitch.

The two adjustment values describe a commanded stance. They do not estimate
joint encoder zeros, identify assembly offsets, or change the policy contract.
This module has no hardware or file access.
"""
from __future__ import annotations

import math
import numpy as np

from .policy import HOME, JOINT_LIMITS, JOINT_NAMES


class HomeAssistError(ValueError):
    """Measurements or an adjustment cannot support this calculation."""


# Columns: ankle pitch, then hip roll, in model joint coordinates.
TRIM_BASIS = np.zeros((14, 2), dtype=np.float64)
TRIM_BASIS[4, 0], TRIM_BASIS[13, 0] = 1.0, -1.0
TRIM_BASIS[1, 1], TRIM_BASIS[10, 1] = 1.0, 1.0
TRIM_BASIS.setflags(write=False)


def _finite(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HomeAssistError(f"{name} 必须是有限数值") from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise HomeAssistError(f"{name} 必须是形状为 {shape} 的有限数值")
    return array


def _positive(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HomeAssistError(f"{name} 必须是正的有限数值") from exc
    if not math.isfinite(result) or result <= 0:
        raise HomeAssistError(f"{name} 必须是正的有限数值")
    return result


def gravity_to_roll_pitch(gravity):
    """Return [roll, pitch] in radians for a body-frame world-down vector.

Axes are X forward, Y left, Z up. For R = Rz(yaw) Ry(pitch) Rx(roll),
gravity is R.T @ [0, 0, -1]. An upright body therefore gives [0, 0, -1].
Gravity does not determine yaw. Near a 90-degree pitch, roll is undefined.
"""
    gravity = _finite(gravity, (3,), "重力向量")
    length = float(np.linalg.norm(gravity))
    if not 0.95 <= length <= 1.05:
        raise HomeAssistError("重力向量必须是单位向量，模长在 0.95..1.05 内")
    gravity = gravity / length
    yz_length = float(np.hypot(gravity[1], gravity[2]))
    if yz_length < 1e-6:
        raise HomeAssistError("俯仰接近 90°，重力无法确定侧倾角")
    return np.array([
        math.atan2(-float(gravity[1]), -float(gravity[2])),
        math.atan2(float(gravity[0]), yz_length),
    ])


def q_check(q, *, target=HOME, ids=None):
    """Summarize feedback versus a target using the existing calibration.

Small errors here establish encoder tracking only; they do not prove that
the real link geometry agrees with the simulation or that both feet are flat.
"""
    q = _finite(q, (14,), "关节反馈角度")
    target = _finite(target, (14,), "关节目标角度")
    if ids is not None:
        if len(ids) != 14 or any(type(sid) is not int for sid in ids):
            raise HomeAssistError("关节 ID 必须是 14 个整数")
    errors = np.rad2deg(q - target)
    joints = []
    for index, name in enumerate(JOINT_NAMES):
        row = {
            "name": name,
            "actual_deg": float(np.rad2deg(q[index])),
            "target_deg": float(np.rad2deg(target[index])),
            "error_deg": float(errors[index]),
            "within_model_limits": bool(
                JOINT_LIMITS[index, 0] <= q[index] <= JOINT_LIMITS[index, 1]
            ),
        }
        if ids is not None:
            row["id"] = ids[index]
        joints.append(row)
    return {
        "joints": joints,
        "max_abs_error_deg": float(np.max(np.abs(errors))),
        "rms_error_deg": float(np.sqrt(np.mean(errors ** 2))),
        "meaning": "使用现有零点换算的跟踪误差，不能验证真实机械零点",
    }


def pose_for_trim(trim):
    """Return HOME plus the two stance adjustments, with other joints fixed."""
    trim = _finite(trim, (2,), "站姿调节量")
    return HOME + TRIM_BASIS @ trim


def central_difference_jacobian(attitude_plus, attitude_minus, probe_rad):
    """Estimate d[roll,pitch]/d[ankle_pitch,hip_roll] from symmetric probes.

Each input is (2, 2): row 0 contains the [roll, pitch] measured while probing
the ankle parameter; row 1 contains those angles for the hip-roll parameter.
The returned matrix has attitude rows and adjustment columns, in rad/rad.
"""
    plus = _finite(attitude_plus, (2, 2), "正向探测姿态")
    minus = _finite(attitude_minus, (2, 2), "反向探测姿态")
    probe = _positive(probe_rad, "探测角度")
    if probe > np.deg2rad(5):
        raise HomeAssistError("局部姿态探测角度不能超过 5°")
    # A roll near +/- pi must not appear to jump through an entire revolution.
    difference = np.arctan2(np.sin(plus - minus), np.cos(plus - minus))
    jacobian = difference.T / (2.0 * probe)
    if not np.all(np.isfinite(jacobian)):
        raise HomeAssistError("姿态响应矩阵出现非有限数值")
    return jacobian


def solve_trim_step(
    jacobian, attitude_error, current_trim, *, damping=0.05,
    max_step_rad=float(np.deg2rad(0.5)),
    max_total_rad=float(np.deg2rad(4.0)),
    min_sensitivity=0.05, max_sensitivity=10.0, max_condition=30.0,
):
    """Return a bounded adjustment delta using damped least squares.

attitude_error is target attitude minus measured attitude, in [roll, pitch].
The caller applies current_trim + returned_delta. The step is scaled together
to preserve its direction, then projected into each parameter's total bounds.
An unresponsive or ambiguous probe is rejected rather than inverted.
"""
    jacobian = _finite(jacobian, (2, 2), "姿态响应矩阵")
    error = _finite(attitude_error, (2,), "目标减实测姿态误差")
    current = _finite(current_trim, (2,), "当前站姿调节量")
    damping = _positive(damping, "求解正则系数")
    step_limit = _positive(max_step_rad, "单步调节上限")
    total_limit = _positive(max_total_rad, "累计调节上限")
    minimum = _positive(min_sensitivity, "最小姿态响应")
    maximum = _positive(max_sensitivity, "最大姿态响应")
    condition_limit = _positive(max_condition, "最大响应条件数")
    if maximum <= minimum or condition_limit < 1:
        raise HomeAssistError("姿态响应检查参数不一致")
    if np.any(np.abs(current) > total_limit + 1e-12):
        raise HomeAssistError("当前站姿调节量已超过累计上限")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    largest, smallest = map(float, singular_values)
    if smallest < minimum:
        raise HomeAssistError(
            f"姿态响应太弱或无法区分两个方向：最小响应 {smallest:.4g}；"
            "检查躯干是否被固定、双脚是否接地或发生打滑"
        )
    if largest > maximum:
        raise HomeAssistError(
            f"姿态响应过大 {largest:.4g}，探测期间可能晃动或接触状态改变"
        )
    condition = largest / smallest
    if condition > condition_limit:
        raise HomeAssistError(
            f"两个调节方向难以区分：响应条件数 {condition:.1f} "
            f"超过 {condition_limit:.1f}"
        )
    try:
        delta = np.linalg.solve(
            jacobian.T @ jacobian + damping ** 2 * np.eye(2),
            jacobian.T @ error,
        )
    except np.linalg.LinAlgError as exc:
        raise HomeAssistError("无法求解姿态响应矩阵") from exc
    if not np.all(np.isfinite(delta)):
        raise HomeAssistError("姿态调节结果不是有限数值")
    peak = float(np.max(np.abs(delta)))
    if peak > step_limit:
        delta *= step_limit / peak
    next_trim = np.clip(current + delta, -total_limit, total_limit)
    return next_trim - current
