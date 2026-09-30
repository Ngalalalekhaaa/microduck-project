"""Read the deployment calibration without changing it or any servo registers."""
import json
import math
from pathlib import Path

BODY_NAMES = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)
DEPLOY_CONFIG = None


def configure(path):
    global DEPLOY_CONFIG
    DEPLOY_CONFIG = Path(path).expanduser() if path else None
    if DEPLOY_CONFIG is not None and DEPLOY_CONFIG.exists():
        read_joints()  # Reject an incorrect ID/name map before opening the UART.


def read_joints():
    if DEPLOY_CONFIG is None or not DEPLOY_CONFIG.exists():
        return {}
    raw = json.loads(DEPLOY_CONFIG.read_text())
    joints = raw.get('joints', [])
    if len(joints) != 14:
        raise ValueError('部署配置必须包含身体的 14 个关节')
    by_name = {j['name']: j for j in joints}
    if set(by_name) != set(BODY_NAMES):
        raise ValueError('部署配置的关节名称不匹配')
    result = {}
    for sid, name in enumerate(BODY_NAMES, 1):
        j = by_name[name]
        if type(j.get('id')) is not int or j['id'] != sid:
            raise ValueError(f'{name} 应为 ID {sid}，配置中为 {j.get("id")}')
        result[sid] = j
    return result


def configured_directions():
    return {sid: j['direction'] for sid, j in read_joints().items()
            if type(j.get('direction')) is int and j['direction'] in (-1, 1)}


def calibration(directions, fake=False):
    if fake:
        return {'ready_ids': list(range(1, 16)),
                'zero_ticks': {str(i): 2048 for i in range(1, 16)},
                'message': '模拟模式：编号 1–14，嘴部 15；不会控制实体舵机。',
                'fake': True}
    try:
        joints = read_joints()
        ready = []
        zeros = {}
        for sid, j in joints.items():
            zero = j.get('zero_tick')
            direction = j.get('direction')
            valid = (type(zero) in (int, float) and math.isfinite(zero)
                     and type(direction) is int and direction in (-1, 1)
                     and directions.get(sid) == direction
                     and j.get('mapping_verified') is True)
            if valid:
                ready.append(sid)
                zeros[str(sid)] = zero
        message = (f'已读取部署标定：{len(ready)}/14 个关节。'
                   if len(ready) == 14 else
                   f'编号已适配；标定 {len(ready)}/14。可保存实际刻度；未标定的 3D 关节仅显示相对变化，不代表实物姿态。')
        return {'ready_ids': ready, 'zero_ticks': zeros, 'message': message,
                'fake': False, 'source': str(DEPLOY_CONFIG) if DEPLOY_CONFIG else None}
    except (ValueError, KeyError, OSError) as exc:
        return {'ready_ids': [], 'zero_ticks': {}, 'fake': False,
                'message': f'标定配置不可用：{exc}。预设姿态不可发送。'}
