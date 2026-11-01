"""Bounded stationary leveling of duck1 using only hips 2,3,12,13.

Default: measure, with no servo writes. Motion requires --move and
--ground-supported. Stop/exit retains the last sent targets and torque;
servos may finish the last small step. This is not a balance controller.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from microduck_deploy.config import load_config, require_calibration, calibration_arrays, save_config
from microduck_deploy.hip_level import selected_indices, hip_targets, level_hips
from microduck_deploy.home_assist import gravity_to_roll_pitch
from microduck_deploy.imu import GRAVITY
from microduck_deploy.imu_process import ProcessImuReader
from microduck_deploy.policy import HOME, q_to_ticks, ticks_to_q
from supported_home import PositionOnlyBus

HIP_IDS = frozenset((2, 3, 12, 13))
PERIOD = .1  # One encoder tick per 100 ms: <=0.879 degrees/s per joint.
MAX_RUN_SECONDS = 420
ALGORITHM_VERSION = 'adaptive-encoder-v2'
STOP_REASONS = {
    'insufficient_joint_probe_response': '探测已到允许幅度，但两侧舵机没有都产生足够实际位移',
    'weak_body_probe_response': '舵机已实际移动，但机身倾角变化仍无法与测量噪声区分',
    'reversed_joint_probe_response': '探测时出现与目标方向相反的实际关节位移',
    'reversed_joint_trim_response': '微调时出现与目标方向相反的实际关节位移',
    'joint_response_stalled': '已累计小步修正，舵机实际位移仍不足',
    'body_response_stalled': '舵机已跟随，但机身倾角没有继续改善',
    'baseline_did_not_return': '返回原目标后，机身倾角未回到基线；可能存在回差或接触变化',
    'inconsistent_probe_response': '正反方向的机身响应不一致，无法可靠计算微调方向',
    'offset_limit': '已达到允许的累计微调范围',
    'response_worsened': '机身倾角变差，停止继续修正',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class HipOnlyBus(PositionOnlyBus):
    """Packet-level boundary: reads or goals for exactly four selected hips."""
    def __init__(self, *args, allow_motion=False, **kwargs):
        self.allow_motion = allow_motion
        super().__init__(*args, **kwargs)

    def _send(self, packet):
        instruction = packet[4]
        if instruction not in (2, 130):
            data = packet[7:-1]
            allowed = (self.allow_motion and instruction == 131
                       and packet[2] == 254 and packet[5:7] == bytes((42, 2))
                       and len(data) == 12 and set(data[::3]) == HIP_IDS
                       and all(int.from_bytes(data[i+1:i+3], 'little') <= 4095
                               for i in range(0, len(data), 3)))
            if not allowed:
                raise ValueError('仅允许读取，或向2/3/12/13号写位置；禁止扭矩、增益和EEPROM写入')
        return super()._send(packet)


@contextmanager
def stop_signals():
    def stopped(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, stopped)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def check_stop():
    if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline()
        if not line or line.strip().lower() in ('q', 'stop', 'exit'):
            raise KeyboardInterrupt


@contextmanager
def exclusive_serial(cfg):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
        if owner.returncode != 1:
            raise ValueError('串口被占用；先关闭网页/其他控制程序：'+owner.stdout+owner.stderr)
        yield


def stationary_measurement(states):
    """Estimate gravity from fresh stationary accelerometer windows, in body axes."""
    if len(states) < 8:
        raise ValueError('有效IMU独立样本不足8个')
    stamps = np.array([s.timestamp for s in states], dtype=float)
    accel = np.array([s.accel_m_s2 for s in states], dtype=float)
    gyro = np.array([s.base_ang_vel for s in states], dtype=float)
    if (accel.shape != (len(states), 3) or gyro.shape != accel.shape
            or not np.isfinite(accel).all() or not np.isfinite(gyro).all()
            or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0)
            or stamps[-1]-stamps[0] < .6):
        raise ValueError('IMU样本不是连续、独立、有限的静止数据')
    norms = np.linalg.norm(accel, axis=1)
    if (np.any(np.abs(norms/GRAVITY-1) > .12)
            or np.max(np.linalg.norm(gyro, axis=1)) > .15):
        raise ValueError('检测到晃动或加速度偏离1g；停止继续调整，保留上力')
    gravity = -accel.mean(axis=0)
    gravity /= np.linalg.norm(gravity)
    angles = np.rad2deg([gravity_to_roll_pitch(-a/n) for a, n in zip(accel, norms)])
    roll, pitch = np.rad2deg(gravity_to_roll_pitch(gravity))
    std = np.std(angles, axis=0)
    middle = len(angles)//2
    drift = np.max(np.abs(angles[:middle].mean(axis=0)-angles[middle:].mean(axis=0)))
    if np.max(std) > .20 or drift > .30:
        raise ValueError('静止窗口内姿态仍在晃动/漂移；未继续调整')
    return {'pitch_deg': float(pitch), 'roll_deg': float(roll),
            'pitch_std_deg': float(std[1]), 'roll_std_deg': float(std[0]),
            'gravity': gravity.tolist(), 'samples': len(states),
            'window_s': float(stamps[-1]-stamps[0])}


class StancePlant:
    def __init__(self, cfg, bus, imu, report, *, moving=False, config_path=None):
        self.cfg, self.bus, self.imu, self.report = cfg, bus, imu, report
        self.moving, self.config_path = moving, config_path
        self.config_hash = digest(config_path) if config_path else None
        self.ids = [j['id'] for j in cfg['joints']]
        self.home = dict(zip(self.ids, map(int, q_to_ticks(HOME, *calibration_arrays(cfg)))))
        self.targets = self.home.copy()
        self.offset = np.zeros(2, dtype=int)
        self.hardware = {}
        self.started = time.monotonic()
        self.report['last_sent_targets'] = None
        self.report['pending_targets'] = None
        self.report['writes'] = []
        self.report['measurement_windows'] = []

    def preflight(self):
        self.hardware = {sid: self.bus.read_configuration(sid) for sid in self.ids}
        self.report['hardware_config'] = {str(sid): copy.deepcopy(self.hardware[sid])
                                          for sid in sorted(HIP_IDS)}
        for sid, hw in self.hardware.items():
            if hw['mode'] != 4 or hw['angle_resolution'] != 1:
                raise ValueError(f'ID{sid}必须使用单圈位置模式')
        if self.moving:
            for pitch in (-44, 44):
                for roll in (-44, 44):
                    self.validate_targets(hip_targets(self.cfg, (pitch, roll)))
        rows, _ = self.read()
        self.report['initial_feedback'] = rows

    def validate_targets(self, targets):
        if set(targets) != HIP_IDS:
            raise ValueError('目标必须且只能包含2/3/12/13号')
        for sid, tick in targets.items():
            hw = self.hardware[sid]
            if not hw['position_min'] <= tick <= hw['position_max']:
                raise ValueError(f'ID{sid}超出硬件单圈限位')

    def read(self):
        began = time.monotonic()
        check_stop()
        if time.monotonic()-self.started > MAX_RUN_SECONDS:
            raise ValueError(f'达到{MAX_RUN_SECONDS}秒时限，停止继续调整并保留上力')
        if self.config_path and digest(self.config_path) != self.config_hash:
            raise ValueError('运行期间robot.json发生变化，停止继续调整')
        readings = {}
        feedback = self.bus.read_feedback_many(self.ids, sync=True)
        for sid in self.ids:
            f = feedback[sid]
            torque = self.bus.read(sid, 40, 1)[0]
            goal = int.from_bytes(self.bus.read(sid, 42, 2), 'little')
            readings[str(sid)] = {'tick': f.position_ticks, 'goal': goal, 'torque': torque}
            if f.status or not 0 <= f.position_ticks <= 4095 or torque not in (0, 1):
                raise ValueError(f'ID{sid}反馈状态异常')
            if self.moving:
                if torque != 1 or goal != self.targets[sid]:
                    raise ValueError(f'ID{sid}未保持预期HOME/微调目标；先托稳回HOME，勿同时运行网页')
                if abs(f.position_ticks-self.targets[sid])*360/4096 > 8:
                    raise ValueError(f'ID{sid}跟踪偏差超过8°，停止继续调整')
                low, high = self.cfg['control']['servo_voltage_range']
                if not low <= f.voltage_v <= high or f.temperature_c >= self.cfg['control']['servo_max_temperature_c']:
                    raise ValueError(f'ID{sid}电压/温度超出当前配置范围')
        state = self.imu.latest(self.cfg['control']['imu_max_age_s'])
        now = time.monotonic()
        if now-began > .2:
            raise ValueError('舵机读取超过200ms，停止继续调整；本步未发送新目标')
        age = now-state.timestamp
        if not math.isfinite(age) or not 0 <= age <= self.cfg['control']['imu_max_age_s']:
            raise ValueError('IMU样本时间戳过期或无效')
        accel = np.asarray(state.accel_m_s2, dtype=float)
        gyro = np.asarray(state.base_ang_vel, dtype=float)
        if accel.shape != (3,) or gyro.shape != (3,) or not np.isfinite([*accel, *gyro]).all():
            raise ValueError('IMU数据无效')
        norm = float(np.linalg.norm(accel))
        if not .8*GRAVITY < norm < 1.2*GRAVITY:
            raise ValueError('IMU加速度明显偏离1g')
        tilt = math.degrees(math.acos(float(np.clip(accel[2]/norm, -1, 1))))
        if self.moving and (tilt >= 12 or np.linalg.norm(gyro) > .5):
            raise ValueError('机身倾斜/转动过大，停止继续调整并保留上力')
        return readings, state

    def sample(self):
        # Discard the settling interval, then acquire a new nonoverlapping window.
        states = []
        joint_rows = []
        started = time.monotonic()
        previous = None
        while time.monotonic()-started < 2.4:
            tick = time.monotonic()
            readings, state = self.read()
            if previous is not None and state.timestamp < previous:
                raise ValueError('IMU时间戳倒退')
            if tick-started >= 1.2 and (previous is None or state.timestamp > previous):
                states.append(state)
                joint_rows.append(readings)
            previous = state.timestamp
            time.sleep(max(0, PERIOD-(time.monotonic()-tick)))
        observation = stationary_measurement(states)
        # Use the SAME accepted timestamps for IMU and encoder summaries.
        # Probe identification must see actual joint travel, not just new goals.
        feedback = {}
        for sid in self.ids:
            key = str(sid)
            ticks = [row[key]['tick'] for row in joint_rows]
            feedback[key] = {'mean_tick': float(np.mean(ticks)),
                             'min_tick': int(min(ticks)), 'max_tick': int(max(ticks)),
                             'goal': joint_rows[-1][key]['goal'],
                             'torque': joint_rows[-1][key]['torque']}
        observation['joint_feedback'] = feedback
        observation['joint_ticks'] = {str(sid): feedback[str(sid)]['mean_tick']
                                      for sid in sorted(HIP_IDS)}
        observation['joint_directions'] = {str(j['id']): j['direction']
                                           for j in self.cfg['joints'] if j['id'] in HIP_IDS}
        self.report['measurement_windows'].append({
            'offset_ticks': self.offset.tolist(), **copy.deepcopy(observation)})
        print(f"前后倾={observation['pitch_deg']:+.3f}°  左右倾={observation['roll_deg']:+.3f}°  "
              f"修正刻度(pitch,roll)={self.offset.tolist()}", flush=True)
        print('髋关节实际/目标：'+', '.join(
            f"ID{sid} {feedback[str(sid)]['mean_tick']:.1f}/{feedback[str(sid)]['goal']}"
            for sid in sorted(HIP_IDS)), flush=True)
        return observation

    def move(self, offset):
        if not self.moving:
            raise ValueError('未启用--move，拒绝写目标')
        desired = np.asarray(offset, dtype=int)
        hip_targets(self.cfg, offset)  # Validate integers and cumulative bounds first.
        while not np.array_equal(self.offset, desired):
            began = time.monotonic()
            self.read()  # Current goals, torque, contact motion and config before every write.
            next_offset = self.offset + np.sign(desired-self.offset).astype(int)
            targets = hip_targets(self.cfg, tuple(map(int, next_offset)))
            self.validate_targets(targets)
            check_stop()
            if time.monotonic()-began > .2:
                raise ValueError('本步读取超过200ms，未发送新目标')
            self.report['pending_targets'] = targets.copy()
            self.bus.sync_positions(targets)
            self.targets.update(targets)
            self.offset = next_offset
            self.report['last_sent_targets'] = targets.copy()
            self.report['last_sent_offset_ticks'] = self.offset.tolist()
            self.report['writes'].append({'elapsed_s': time.monotonic()-self.started,
                                          'targets': targets.copy()})
            self.report['pending_targets'] = None
            for sid, target in targets.items():
                if int.from_bytes(self.bus.read(sid, 42, 2), 'little') != target:
                    raise ValueError(f'ID{sid}目标回读不一致，停止继续写入')
            time.sleep(max(0, PERIOD-(time.monotonic()-began)))


def home_candidate(cfg, offsets, report):
    """Optional stance-derived HOME shift; this does not prove anatomical zero."""
    candidate = copy.deepcopy(cfg)
    indices = selected_indices(cfg)
    previous = q_to_ticks(HOME, *calibration_arrays(cfg))
    targets = hip_targets(cfg, offsets)
    stamp = datetime.now(timezone.utc).isoformat()
    records = candidate['calibration'].setdefault('manual_home_adjustments', [])
    for sid, index in indices.items():
        delta = targets[sid]-int(previous[index])
        candidate['joints'][index]['zero_tick'] += delta
        candidate['calibration']['captured_ticks'][index] += delta
        if not 0 <= candidate['calibration']['captured_ticks'][index] <= 4095:
            raise ValueError(f'ID{sid}保存后的参考刻度越过单圈边界')
        records.append({'id': sid, 'old_home_tick': int(previous[index]),
                        'new_home_tick': targets[sid], 'delta_ticks': delta, 'utc': stamp,
                        'method': 'IMU stationary stance trim; not independent kinematic zero identification',
                        'report': report})
    candidate['calibration']['home_ticks'] = q_to_ticks(HOME, *calibration_arrays(candidate)).tolist()
    old_reference = ticks_to_q(cfg['calibration']['captured_ticks'], *calibration_arrays(cfg))
    new_reference = ticks_to_q(candidate['calibration']['captured_ticks'], *calibration_arrays(candidate))
    if not np.allclose(old_reference, new_reference, rtol=0, atol=1e-10):
        raise ValueError('原有参考姿态关系变化，拒绝保存')
    return candidate


def save_success(plant, report, log_path, tolerance):
    if report.get('status') != 'converged' or report.get('pending_offset_ticks') is not None:
        raise ValueError('只有完整收敛结果才能保存HOME')
    if list(plant.offset) != report['offset_ticks']:
        raise ValueError('实际发送的修正与完成记录不一致')
    fresh = plant.sample()
    if max(abs(fresh['pitch_deg']), abs(fresh['roll_deg'])) > tolerance:
        raise ValueError('保存前复测未达标，未修改HOME')
    rows, _ = plant.read()
    for sid in HIP_IDS:
        if abs(rows[str(sid)]['tick']-plant.targets[sid]) > 23:
            raise ValueError(f'ID{sid}保存前跟踪误差超过约2°，未修改HOME')
    candidate = home_candidate(plant.cfg, report['offset_ticks'], str(log_path))
    if digest(plant.config_path) != plant.config_hash:
        raise ValueError('配置已变化，拒绝覆盖')
    if not any(report['offset_ticks']):
        report['save_note'] = '无需修正，原配置未改'
        return
    expected = hashlib.sha256((json.dumps(candidate, ensure_ascii=False, indent=2)+'\n').encode()).hexdigest()
    try:
        save_config(plant.config_path, candidate)
    finally:
        # An interrupt can occur immediately after the atomic replacement.
        # Report what actually reached disk, without attempting robot motion.
        try:
            actual = digest(plant.config_path)
        except OSError:
            actual = None
        if actual == expected:
            report['calibration_saved'] = True
            report['saved_config_sha256'] = actual
            report['saved_home_ticks'] = candidate['calibration']['home_ticks']
        elif actual != plant.config_hash:
            report['calibration_save_state'] = 'unknown; inspect robot.json and its backup'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=ALGORITHM_VERSION)
    parser.add_argument('--config', type=Path, default=ROOT/'robot.json')
    parser.add_argument('--move', action='store_true', help='只向2/3/12/13号发送小步微调目标')
    parser.add_argument('--ground-supported', action='store_true', help='确认双脚在水平防滑地面自然承重，有防倒保护且躯干可轻微转动')
    parser.add_argument('--save-home', action='store_true', help='仅达标后备份并保存四颗髋关节的软件HOME修正')
    parser.add_argument('--tolerance-deg', type=float, default=.5)
    args = parser.parse_args(argv)
    if args.move and not args.ground_supported:
        parser.error('--move必须同时提供--ground-supported，并满足实际承重条件')
    if args.save_home and not args.move:
        parser.error('--save-home需要--move')
    if not math.isfinite(args.tolerance_deg) or not .2 <= args.tolerance_deg <= 1:
        parser.error('--tolerance-deg须在0.2..1之间')
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    require_calibration(cfg)
    selected_indices(cfg)
    report = {'started': datetime.now(timezone.utc).isoformat(), 'config_path': str(config_path),
              'algorithm_version': ALGORITHM_VERSION,
              'config_sha256': digest(config_path), 'move': args.move, 'save_requested': args.save_home,
              'selected_ids': sorted(HIP_IDS), 'calibration_saved': False,
              'stop_behavior': 'No further goal writes; retain last sent targets and torque.'}
    log_path = ROOT/'logs'/('hip_level_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    log_path.parent.mkdir(exist_ok=True)
    reader = None
    code = 0
    print('只调整2/3/12/13号；其余关节须保持HOME。目标是机身重力投影接近[0,0,-1]。', flush=True)
    print('自适应编码器探测v2：9→18→27→36→44刻度，确认两侧实际移动后辨识响应；不改变舵机增益。', flush=True)
    print('Ctrl+C或q回车：停止继续发目标，保留最后目标和上力。不会自动卸力。', flush=True)
    try:
        with stop_signals(), exclusive_serial(cfg):
            if digest(config_path) != report['config_sha256']:
                raise ValueError('启动期间配置发生变化')
            reader = ProcessImuReader(cfg, calibrate=False)
            options = dict(cfg['serial'])
            options['timeout'] = max(.05, options['timeout'])
            with HipOnlyBus(**options, allow_motion=args.move) as bus:
                plant = StancePlant(cfg, bus, reader, report, moving=args.move, config_path=config_path)
                plant.preflight()
                if args.move:
                    level_hips(plant, report, tolerance_deg=args.tolerance_deg)
                    if report['status'] == 'converged':
                        if args.save_home:
                            save_success(plant, report, log_path, args.tolerance_deg)
                        print('前后/左右倾角均连续达标，保留当前目标与上力。', flush=True)
                    else:
                        code = 2
                        reason = str(report['reason'])
                        detail = STOP_REASONS.get(reason, reason)
                        print(f'未达标，已停止继续调整并保留上力：{detail}（{reason}）；未保存HOME。', flush=True)
                else:
                    report.update(status='measured', measurement=plant.sample())
                    print('测量完成；未发送任何舵机写指令。', flush=True)
    except KeyboardInterrupt:
        code = 130
        report.update(status='interrupted', error='operator stop')
        print('已停止继续写目标；最后的小步可能尚在完成，保持支撑。未发送卸力。', flush=True)
    except Exception as exc:
        code = 1
        report.update(status='stopped', error=str(exc))
        print('停止：'+str(exc)+'。未发送卸力；请保持防倒支撑。', flush=True)
    finally:
        if reader is not None:
            reader.close()
        log_path.write_text(json.dumps(report, default=json_value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print('报告：'+str(log_path), flush=True)
        if report['calibration_saved']:
            print('已备份robot.json并只保存四颗髋关节HOME修正；其余关节、IMU和模型未改。', flush=True)
        elif report.get('calibration_save_state'):
            print('配置保存状态未确认，请检查robot.json和备份；未发送恢复动作。', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
