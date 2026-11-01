"""Foreground 50 Hz controller with measured inputs and fail-stop checks."""
from __future__ import annotations

import dataclasses
import json
import math
import select
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (ROOT, calibration_arrays, require_calibration, roulade_head_pitch_max_deg,
                     policy_timing_limits, policy_overrun_threshold)
from .policy import (HOME, JOINT_NAMES, JOINT_LIMITS, OnnxPolicy, build_obs61,
                     ticks_to_q, q_to_ticks, action_to_targets)
from .servo import ServoBus, ServoTimeout
from .imu import ImuConfig, open_imu
from .background_log import BackgroundLineLog
from .gamepad import COMMAND_LIMITS
from .homing import (plan_home, check_servo_limits, validate_reference_calibration,
                     HOME_TRACKING_LIMIT_RAD)


class StopRequested(RuntimeError):
    pass


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(type(value).__name__)


def log_path(prefix):
    path = ROOT / "logs" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jsonl"
    path.parent.mkdir(exist_ok=True)
    return path


class ImuReader:
    """Continuously integrate the IMU at its ODR; controller consumes latest state."""
    def __init__(self, cfg, *, calibrate=True):
        self.imu = open_imu(ImuConfig.from_dict(cfg["imu"]))
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.state = None
        self.error = None
        self.thread = None
        try:
            self.imu.initialize()
            if calibrate:
                print("保持机身静止约2秒，测量陀螺仪偏置……", flush=True)
                self.imu.calibrate_gyro(sample_count=240)
            self.thread = threading.Thread(target=self._loop, name="imu-reader", daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 2
            while self.state is None and self.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.latest(0.2)
        except BaseException:
            self.close()
            raise

    def _loop(self):
        try:
            while not self.stop.is_set():
                value = self.imu.read_state(timeout_s=0.05)
                with self.lock:
                    self.state = value
        except Exception as exc:
            self.error = exc

    def latest(self, max_age):
        if self.error is not None:
            raise StopRequested(f"IMU失效: {self.error}")
        with self.lock:
            value = self.state
        if value is None or time.monotonic() - value.timestamp > max_age:
            raise StopRequested("IMU数据过期；停止发送动作")
        if not np.all(np.isfinite(value.base_ang_vel)) or not np.all(np.isfinite(value.projected_gravity)):
            raise StopRequested("IMU输出包含非有限值")
        return value

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=0.3)
        self.imu.close()


def check_feedback(feedback, cfg, *, extra_ids=()):
    ids = [j["id"] for j in cfg["joints"]] + list(extra_ids)
    if set(feedback) != set(ids):
        raise StopRequested("没有收到完整舵机反馈")
    lo, hi = cfg["control"]["servo_voltage_range"]
    now = time.monotonic()
    for sid, value in feedback.items():
        if now - value.received_at > cfg["control"]["max_cycle_s"]:
            raise StopRequested(f"舵机{sid}反馈过期")
        if value.status:
            raise StopRequested(f"舵机{sid}报告状态错误0x{value.status:02x}")
        if not lo <= value.voltage_v <= hi:
            raise StopRequested(f"舵机{sid}电压{value.voltage_v:.1f}V超出配置范围{lo}..{hi}V")
        if value.temperature_c >= cfg["control"]["servo_max_temperature_c"]:
            raise StopRequested(f"舵机{sid}温度{value.temperature_c}°C过高")
        if not 0 <= value.position_ticks <= 4095:
            raise StopRequested(f"舵机{sid}不是单圈0..4095读数；拒绝取模")


def check_posture(q, state, cfg, *, policy_mode='walking'):
    q = np.asarray(q, dtype=float)
    gravity = np.asarray(state.projected_gravity, dtype=float)
    gyro = np.asarray(state.base_ang_vel, dtype=float)
    if (q.shape != (14,) or not np.all(np.isfinite(q))
            or gravity.shape != (3,) or not np.all(np.isfinite(gravity))
            or gyro.shape != (3,) or not np.all(np.isfinite(gyro))
            or not 0.95 <= np.linalg.norm(gravity) <= 1.05):
        raise StopRequested("姿态或关节数据不是有效有限值/单位重力向量")
    tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
    # Ground-pick intentionally leans the trunk (34.5 degrees in rehearsal).
    # Its own bound is active only while this explicitly selected action runs.
    if policy_mode == 'roulade':
        if cfg['control'].get('roulade_enabled', False) is not True:
            raise StopRequested('滚翻未显式启用')
        # Pitch must pass through inversion during a roll. Keep sagittal
        # plane and angular-speed bounds instead of disabling posture checks.
        lateral = math.degrees(math.asin(float(np.clip(abs(gravity[1]), 0, 1))))
        if lateral > cfg['control'].get('roulade_lateral_limit_deg', 35.):
            raise StopRequested(f'滚翻横向倾斜{lateral:.1f}°超过阈值')
        if np.linalg.norm(gyro) > cfg['control'].get('roulade_max_angular_speed_rad_s', 12.):
            raise StopRequested('滚翻角速度超过阈值')
    else:
        limit = (cfg['control'].get('ground_pick_tilt_limit_deg', 50.)
                 if policy_mode == 'ground_pick' else cfg['control']['tilt_limit_deg'])
        if tilt > limit:
            raise StopRequested(f"机身倾斜{tilt:.1f}°超过阈值")
    lower, upper = JOINT_LIMITS[:, 0] - .12, JOINT_LIMITS[:, 1] + .12
    if policy_mode == 'roulade':
        # Only the explicitly configured positive head-pitch feedback bound
        # changes; all lower limits and all other joints retain their bounds.
        try:
            upper[JOINT_NAMES.index('head_pitch')] = math.radians(roulade_head_pitch_max_deg(cfg['control']))
        except ValueError as exc:
            raise StopRequested(str(exc)) from exc
    bad = np.flatnonzero((q < lower) | (q > upper))
    if len(bad):
        joints = cfg.get('joints', [])
        details = ', '.join(
            f'{JOINT_NAMES[i]}（ID{joints[i].get("id", "?") if i < len(joints) else "?"}）'
            f'实测{math.degrees(q[i]):.2f}°，允许'
            f'[{math.degrees(lower[i]):.2f}, {math.degrees(upper[i]):.2f}]°'
            for i in bad)
        raise StopRequested("测得关节角超出模型机械范围；" + details + "；检查零位/方向/装配")
    return tilt


class RouladeMonitor:
    """Bound one roll using measured IMU progress and settled upright recovery.

    Training accumulates +body-Y rotation with a contact gate. Hardware has
    no contact sensors: this monitor cannot certify head contact or support.
    Its purpose is a bounded exit, never a claim of trained-task success.
    """

    def __init__(self, stamp, state, cfg):
        self.control = cfg['control']
        gravity = np.asarray(state.projected_gravity)
        tilt = math.degrees(math.acos(float(np.clip(-gravity[2], -1, 1))))
        if tilt > self.control.get('roulade_completion_tilt_deg', 25.) or np.linalg.norm(state.base_ang_vel) > 1.5:
            raise StopRequested('滚翻必须从稳定直立状态开始')
        self.started = self.previous_stamp = stamp
        self.previous_pitch_rate = float(state.base_ang_vel[1])
        self.angle = 0.
        self.saw_inversion = False
        self.settled_since = None
        self.phase = 'rolling'

    def update(self, stamp, state, tilt):
        age = stamp - self.started
        if age >= self.control.get('roulade_timeout_s', 5.):
            raise StopRequested('滚翻超时且未确认稳定站立')
        dt = stamp - self.previous_stamp
        if not 0 <= dt <= self.control['max_cycle_s'] + .01:
            raise StopRequested('滚翻姿态采样间隔异常')
        gyro = np.asarray(state.base_ang_vel)
        self.angle += .5 * (self.previous_pitch_rate + float(gyro[1])) * dt
        self.previous_pitch_rate = float(gyro[1])
        self.previous_stamp = stamp
        self.saw_inversion |= float(state.projected_gravity[2]) > .5
        # A small backward wind-up is possible. A quarter-turn in reverse
        # is no longer a forward-roll attempt and must not run until timeout.
        if self.angle < -math.pi / 2:
            raise StopRequested('滚翻反向转动超过90°')
        if self.angle > math.radians(450):
            raise StopRequested('滚翻超过一次转动范围')
        recovered = (self.saw_inversion and self.angle >= math.radians(330)
                     and tilt <= self.control.get('roulade_completion_tilt_deg', 25.)
                     and np.linalg.norm(gyro) <= 1.5)
        if recovered:
            self.phase = 'settling'
            if self.settled_since is None:
                self.settled_since = stamp
        else:
            self.phase = 'recovering' if self.saw_inversion else 'rolling'
            self.settled_since = None
        return (self.settled_since is not None and
                stamp - self.settled_since >= self.control.get('roulade_settle_seconds', .25))

    def snapshot(self, stamp):
        return {'elapsed_s': stamp - self.started, 'phase': self.phase,
                'rotation_deg': math.degrees(self.angle),
                'saw_inversion': bool(self.saw_inversion)}


def target_angle_warnings(target, previous_target, cfg):
    """Record target bounds and jumps without clipping the policy output."""
    target = np.asarray(target, dtype=float)
    if target.shape != (14,) or not np.all(np.isfinite(target)):
        raise StopRequested("策略目标不是14个有限角度")
    previous_target = np.asarray(previous_target, dtype=float)
    if previous_target.shape != (14,) or not np.all(np.isfinite(previous_target)):
        raise StopRequested("上一周期目标不是14个有限角度")
    warnings = []
    # Training permits target overshoot. These bounds can either stop a trial
    # or produce log entries; neither mode clips the raw policy output.
    limits = np.asarray(cfg.get("target_limits_rad", JOINT_LIMITS), dtype=float)
    if limits.shape != (14, 2) or not np.all(np.isfinite(limits)) or np.any(limits[:, 0] >= limits[:, 1]):
        raise StopRequested("目标角度限制配置无效")
    bad = np.flatnonzero((target < limits[:, 0]) | (target > limits[:, 1]))
    if len(bad):
        warnings.append("策略目标超出首次试机角度范围: " + ", ".join(
            f"{JOINT_NAMES[i]} 目标{np.rad2deg(target[i]):.2f}°，"
            f"允许[{np.rad2deg(limits[i,0]):.2f}, {np.rad2deg(limits[i,1]):.2f}]°"
            for i in bad))
    delta = np.abs(target - previous_target)
    if np.max(delta) > cfg["control"]["max_target_jump_rad"]:
        i = int(np.argmax(delta))
        warnings.append(f"单周期目标变化过大：{JOINT_NAMES[i]} "
                            f"{np.rad2deg(delta[i]):.2f}°，上限"
                            f"{np.rad2deg(cfg['control']['max_target_jump_rad']):.2f}°")
    return warnings


def validate_targets(target, previous_target, cfg, *, return_warnings=False):
    warnings = target_angle_warnings(target, previous_target, cfg)
    if warnings and not cfg['control'].get('log_only_target_angles', False):
        raise StopRequested(warnings[0] + '；目标未发送')
    zero, direction = calibration_arrays(cfg)
    ticks = q_to_ticks(target, zero, direction)
    return (ticks, warnings) if return_warnings else ticks


def constrain_posture_target(target, cfg):
    """Saturate small sitstand goal overshoots at calibrated single-turn bounds.

    The sitstand network briefly asks the right knee for 4097 ticks on this
    robot. Saturating a goal at 4095 preserves single-turn encoding; it never
    wraps the measured angle or alters calibration. Larger (>5 degree) goal
    errors still stop control. The caller keeps raw action history unchanged.
    """
    target = np.asarray(target, dtype=float)
    if target.shape != (14,) or not np.isfinite(target).all():
        raise StopRequested('站坐目标不是14个有限角度')
    zero, direction = calibration_arrays(cfg)
    ends = direction[:, None] * (np.array([0., 4095.]) - zero[:, None]) * (2*np.pi/4096)
    limited = np.clip(target, ends.min(axis=1), ends.max(axis=1))
    error = np.abs(limited-target)
    if np.max(error) > math.radians(5):
        # Diagnose the original continuous encoder goal without calling
        # q_to_ticks, which intentionally rejects a single-turn overshoot.
        raw_ticks = zero + direction * target * (4096 / (2*np.pi))
        details = ', '.join(
            f'{JOINT_NAMES[i]}（ID{cfg["joints"][i]["id"]}）'
            f'原始目标{raw_ticks[i]:.2f}ticks，允许[0,4095]ticks，'
            f'越界{np.rad2deg(error[i]):.2f}°'
            for i in np.flatnonzero(error > math.radians(5)))
        raise StopRequested('站坐目标超出单圈可表示范围超过5°：' + details + '；目标未发送')
    names = [JOINT_NAMES[i] for i in np.flatnonzero(error > 1e-10)]
    return limited, names


class Controller:
    def __init__(self, cfg, *, bus=None, imu_reader=None, policy=None, jaw_source=None,
                 sitstand_policy=None, ground_pick_policy=None, roulade_policy=None,
                 updated_walking_policy=None, sitstand_friction_policy=None,
                 sitstand_com_policy=None, keyboard_walking_policies=None,
                 keyboard_sitstand_policies=None):
        self.cfg = cfg
        self.bus = bus
        self.imu = imu_reader
        self.policy = policy
        self.sitstand_policy = sitstand_policy
        self.ground_pick_policy = ground_pick_policy
        self.roulade_policy = roulade_policy
        self.updated_walking_policy = updated_walking_policy
        self.sitstand_friction_policy = sitstand_friction_policy
        self.sitstand_com_policy = sitstand_com_policy
        self.keyboard_walking_policies = dict(keyboard_walking_policies or {})
        self.keyboard_sitstand_policies = dict(keyboard_sitstand_policies or {})
        self.stop = threading.Event()
        self.enabled = False
        self.heartbeat = 0.0
        self.ids = [j["id"] for j in cfg["joints"]]
        from .jaw import jaw_ids
        self.jaw_ids = jaw_ids(cfg) if jaw_source is not None else []
        self.motion_ids = self.ids + self.jaw_ids
        self.jaw_source = jaw_source
        self.jaw = None
        self.old_gains = {}
        self.watchdog = None
        self._signals = {}
        self.reason = None
        self.session = {}
        self.from_pose = None
        self.move_seconds = None
        self.take_over_hold = False
        self._stdin_provider = None

    def _selected_policy_variants(self, command_provider):
        walking = getattr(command_provider, 'walking_variant', 'original')
        sitstand = getattr(command_provider, 'sitstand_variant', 'default')
        if not isinstance(walking, str) or walking not in ('original', 'updated'):
            raise StopRequested('行走模型选择无效')
        if not isinstance(sitstand, str) or sitstand not in ('default', 'friction', 'com'):
            raise StopRequested('坐站模型选择无效')
        if ((walking != 'original' or sitstand != 'default') and
                self.cfg['control'].get('model_suite_buttons_enabled', False) is not True):
            raise StopRequested('方向键模型切换未显式启用')
        walk_policy = self.policy if walking == 'original' else self.updated_walking_policy
        sit_policy = {'default': self.sitstand_policy, 'friction': self.sitstand_friction_policy,
                      'com': self.sitstand_com_policy}[sitstand]
        # HOME-only operation deliberately never opens the original ONNX.
        # A selected optional variant, however, must already be preloaded.
        if walking == 'updated' and walk_policy is None:
            raise StopRequested(f'未加载所选行走模型：{walking}')
        if sitstand != 'default' and sit_policy is None:
            raise StopRequested(f'未加载所选坐站模型：{sitstand}')
        key = getattr(command_provider, 'walking_model_key', None)
        if key is not None:
            options = self.cfg.get('keyboard_walking_models') or {}
            if (not isinstance(key, str) or key not in 'abcdef' or len(key) != 1
                    or key not in options or key not in self.keyboard_walking_policies
                    or walking != 'original'):
                raise StopRequested('键盘行走模型未配置或未预加载')
            walk_policy = self.keyboard_walking_policies[key]
            if walk_policy is None:
                raise StopRequested('键盘行走模型会话无效')
            walking = f'keyboard_{key}'
        key = getattr(command_provider, 'sitstand_model_key', None)
        if key is not None:
            options = self.cfg.get('keyboard_sitstand_models') or {}
            if (not isinstance(key, str) or key not in 'ghimn' or len(key) != 1
                    or key not in options or key not in self.keyboard_sitstand_policies
                    or sitstand != 'default'):
                raise StopRequested('键盘坐站模型未配置或未预加载')
            sit_policy = self.keyboard_sitstand_policies[key]
            if sit_policy is None:
                raise StopRequested('键盘坐站模型会话无效')
            sitstand = f'keyboard_{key}'
        return walking, sitstand, walk_policy, sit_policy

    def _home_plan(self, q):
        try:
            return plan_home(self.cfg, q, from_pose=self.from_pose, move_seconds=self.move_seconds)
        except ValueError as exc:
            raise StopRequested(str(exc)) from exc

    def _check_home_feedback(self, q, previous_q, previous_target):
        if np.max(np.abs(q - previous_q)) > 0.12:
            raise StopRequested("回HOME反馈角度跳变超过0.12rad；停止，检查跨圈/方向/支撑")
        error = np.abs(q - previous_target)
        if np.max(error) > HOME_TRACKING_LIMIT_RAD:
            index = int(np.argmax(error))
            raise StopRequested(f"回HOME跟踪误差超过10°：{JOINT_NAMES[index]} "
                                f"{np.rad2deg(error[index]):.2f}°；检查卡住、方向和速度限制")

    def _watchdog(self):
        while not self.stop.wait(0.02):
            if self.enabled and time.monotonic() - self.heartbeat > 0.15:
                self.reason = "控制循环超过150ms未完成，软件看门狗尝试卸力"
                self.stop.set()
                try:
                    self.bus.set_torque(self.motion_ids, False, verify=False)
                except Exception:
                    pass
                return

    def _install_signals(self):
        if threading.current_thread() is not threading.main_thread():
            return
        def stop(signum, frame):
            self.reason = f"收到停止信号{signum}"
            self.stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._signals[sig] = signal.signal(sig, stop)

    def _check_stop(self):
        if self.stop.is_set():
            raise StopRequested(self.reason or "已停止")
        if self._stdin_provider is not None:
            reason = self._stdin_provider.check_stop()
            if reason:
                raise StopRequested(reason)
            return
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() in ("q", "stop", "exit"):
                raise StopRequested("用户输入停止")

    def _read(self, *, strict=True):
        read_started = time.monotonic()
        try:
            values = self.bus.read_feedback_many(self.motion_ids, sync=True)
        except ServoTimeout:
            # A single lost batch must not cause an automatic motion restart.
            # Retry only this read; keep the previous target and the existing
            # control-period deadline. Never assemble a batch from cached data.
            self._check_stop()
            if time.monotonic() - read_started >= self.cfg['control']['max_cycle_s']:
                raise
            self.session['servo_read_retries'] = self.session.get('servo_read_retries', 0) + 1
            self.stop.wait(.001)
            self._check_stop()
            values = self.bus.read_feedback_many(self.motion_ids, sync=True)
        read_elapsed = time.monotonic() - read_started
        if read_elapsed > self.cfg['control']['max_cycle_s']:
            self.session['servo_read_timing_fault'] = {
                'read_ms': read_elapsed*1000,
                'limit_ms': self.cfg['control']['max_cycle_s']*1000,
                'reply_ids': list(values),
            }
            raise StopRequested('读取舵机反馈超过控制周期时限')
        if strict:
            try:
                check_feedback(values, self.cfg, extra_ids=self.jaw_ids)
            except StopRequested as exc:
                # Preserve the failing batch rather than only the prior good frame.
                self.session['fault_feedback'] = values
                self.session['fault_feedback_reason'] = str(exc)
                raise
        ticks = [values[sid].position_ticks for sid in self.ids]
        zero, direction = calibration_arrays(self.cfg)
        q = ticks_to_q(ticks, zero, direction)
        try:
            state = self.imu.latest(self.cfg["control"]["imu_max_age_s"])
        except Exception as exc:
            # Preserve this frame's already-read feedback on IMU failure. No
            # extra bus IO, retry, stale sample or change to fail-stop behavior.
            self.session['imu_fault'] = {
                'observed_at_monotonic_s': time.monotonic(),
                'error': str(exc), 'feedback': values,
            }
            raise
        self.session["last_measurement"] = {"q": q, "gravity": state.projected_gravity,
                                            "gyro": state.base_ang_vel, "feedback": values}
        return q, state, values

    def _prepare_motors(self, q):
        self._check_stop()
        if self.cfg["serial"]["register_profile"] != "hls_2":
            raise StopRequested("策略运行要求核实过的HD1910 HLS_2寄存器配置")
        configs = {sid: self.bus.read_configuration(sid) for sid in self.ids}
        self.session["servo_configuration_before_motion"] = configs
        plan = self._home_plan(q)
        for sid, conf in configs.items():
            expected_torque = 1 if self.take_over_hold else 0
            if conf["torque"] != expected_torque:
                raise StopRequested(f"舵机{sid}已上力；先托住机器人运行 relax")
            if conf["mode"] != 4 or conf["angle_resolution"] != 1:
                raise StopRequested(f"舵机{sid}需要mode=4、angle_resolution=1，当前为{conf}")
            if self.take_over_hold:
                expected = {key:self.cfg['control']['ram_'+key] for key in ('p','d','i')}
                if conf['ram_gains'] != expected:
                    raise StopRequested(f'舵机{sid}当前增益不符合已上力接管条件')
        if self.from_pose is not None:
            check_servo_limits(plan, configs)
        self.old_gains = {sid: conf["ram_gains"] for sid, conf in configs.items()}
        p = self.cfg["control"]
        if not self.take_over_hold:
            self.bus.set_pid_ram(self.ids, p["ram_p"], p["ram_d"], p["ram_i"])
        self._check_stop()
        q, state, _ = self._read()
        check_posture(q, state, self.cfg)
        plan = self._home_plan(q)
        if self.from_pose is not None:
            check_servo_limits(plan, configs)
        self.session["home_plan"] = plan
        zero, direction = calibration_arrays(self.cfg)
        preload = dict(zip(self.ids, q_to_ticks(q, zero, direction).tolist()))
        self._check_stop()
        # HD firmware may react to goal writes even after torque-off. Treat the
        # very first goal packet as motion, including partial/broken writes.
        self.heartbeat = time.monotonic()
        self.enabled = True
        self.watchdog = threading.Thread(target=self._watchdog, name="control-watchdog", daemon=True)
        self.watchdog.start()
        self.bus.sync_positions(preload)
        # A broadcast goal has no acknowledgement. Confirm register 42 before
        # torque-on so a lost preload cannot expose an old stored target.
        for sid, target in preload.items():
            actual = int.from_bytes(self.bus.read(sid, 42, 2), 'little')
            if actual != target:
                raise StopRequested(f"舵机{sid}当前角预装目标未读回确认，拒绝上力")
        self._check_stop()
        self.heartbeat = time.monotonic()
        self.bus.set_torque(self.ids, True, verify=True)
        self.heartbeat = time.monotonic()
        return q

    def _send(self, target, previous_target, *, cycle_started=None, cycle_limit_s=None,
              validated_ticks=None):
        self._check_stop()
        # The policy loop already validates this exact target and retains its
        # warnings. Reuse its encoder result; HOME and other callers validate
        # here as before. Never reuse a result across control iterations.
        ticks = (validate_targets(target, previous_target, self.cfg)
                 if validated_ticks is None else validated_ticks)
        targets = dict(zip(self.ids, ticks.tolist()))
        if self.jaw is not None:
            feedback = self.session.get('last_measurement', {}).get('feedback')
            targets.update(self.jaw.targets(feedback=feedback))
            self._check_stop()
            limit = self.cfg['control']['max_cycle_s'] if cycle_limit_s is None else cycle_limit_s
            if cycle_started is not None and time.monotonic() - cycle_started > limit:
                if cycle_limit_s is not None:
                    self.session['policy_cycle_timing_fault'] = {
                        'stage': 'before_jaw_target_send',
                        'cycle_ms': (time.monotonic() - cycle_started) * 1000,
                        'limit_ms': limit * 1000, 'policy_target_sent_this_step': False}
                raise StopRequested('嘴部准备后控制周期已超时；拒绝发送过时身体目标')
            check_feedback(feedback, self.cfg, extra_ids=self.jaw_ids)
        # Validation or other preparation can consume the remaining budget.
        # Check again at the write boundary even when there is no jaw.
        self._check_stop()
        limit = self.cfg['control']['max_cycle_s'] if cycle_limit_s is None else cycle_limit_s
        if cycle_started is not None and time.monotonic() - cycle_started > limit:
            if cycle_limit_s is not None:
                self.session['policy_cycle_timing_fault'] = {
                    'stage': 'before_target_send',
                    'cycle_ms': (time.monotonic() - cycle_started) * 1000,
                    'limit_ms': limit * 1000, 'policy_target_sent_this_step': False}
            raise StopRequested('目标准备后控制周期已超时；拒绝发送过时目标')
        self.bus.sync_positions(targets)
        self.heartbeat = time.monotonic()

    def _home(self, start):
        p = self.cfg["control"]
        plan = self._home_plan(start)
        previous = start.copy()
        previous_q = start.copy()
        steps = max(1, math.ceil(plan["duration_s"] * 50))
        self.session["home_samples"] = []
        print(f"缓慢回HOME：{steps/50:.2f}秒，峰值目标速度约"
              f"{plan['peak_target_speed_deg_s']:.2f}°/s；保持支撑。", flush=True)
        next_target_at = time.monotonic() + 0.02
        for step in range(1, steps + 1):
            self._check_stop()
            begun = time.monotonic()
            q, state, _ = self._read()
            check_posture(q, state, self.cfg)
            self._check_home_feedback(q, previous_q, previous)
            alpha = step / steps
            alpha = alpha * alpha * (3 - 2 * alpha)
            target = start + alpha * (HOME - start)
            # A late read must not trigger a burst of catch-up targets. Keep
            # each target's send start >=20ms after the preceding send ends; delays
            # extend the transition instead of compressing its trajectory.
            self.stop.wait(max(0, next_target_at - time.monotonic()))
            if time.monotonic() - begun > p["max_cycle_s"]:
                raise StopRequested("回HOME传感器读取超时；拒绝发送过时目标")
            self._send(target, previous, cycle_started=begun)
            next_target_at = time.monotonic() + 0.02
            self.session["home_samples"].append({"step": step, "q": q, "target": target})
            previous_q = q.copy()
            previous = target
            if step % 50 == 0:
                print(f"回HOME {step/steps:.0%}，最大跟踪误差"
                      f"{np.rad2deg(np.max(np.abs(q-target))):.2f}°", flush=True)
            if time.monotonic() - begun > p["max_cycle_s"]:
                raise StopRequested("回HOME期间控制周期过长")
        q, state, _ = self._read()
        check_posture(q, state, self.cfg)
        self._check_home_feedback(q, previous_q, previous)
        if np.max(np.abs(q - HOME)) > p["home_tolerance_rad"]:
            raise StopRequested("未到达HOME容差；检查支撑、零位、增益、速度限制和装配")
        return q

    def execute(self, *, seconds=5.0, twist=(0, 0, 0), motion=False, home_only=False,
                from_pose=None, move_seconds=None, take_over_hold=False, command_ramp_s=0,
                command_provider=None, zero_command_after_s=None):
        require_calibration(self.cfg, imu=motion)
        if command_provider is not None and (not callable(command_provider) or home_only):
            raise ValueError('实时速度源须可调用，并且只用于策略执行')
        stdin_provider = command_provider if getattr(command_provider, 'owns_stdin', False) is True else None
        if stdin_provider is not None and any(not callable(getattr(stdin_provider, name, None))
                                              for name in ('check_stop', 'start_policy')):
            raise ValueError('终端速度源缺少停止检查或策略启动接口')
        if take_over_hold and (not motion or from_pose is not None):
            raise ValueError('上力接管只用于接近HOME的短时运动入口')
        self.take_over_hold = bool(take_over_hold)
        if (from_pose is not None or move_seconds is not None) and not (motion and home_only):
            raise ValueError("参考姿态与慢移时长选项只用于home；不能用它放宽策略run的启动范围")
        if from_pose is not None:
            validate_reference_calibration(self.cfg, from_pose)
        if move_seconds is not None:
            minimum = 8 if from_pose is not None else 0
            if not math.isfinite(move_seconds) or not minimum <= move_seconds <= 30 or move_seconds <= 0:
                raise ValueError(f"回HOME时长需为{minimum}..30秒内的正数")
        self.from_pose, self.move_seconds = from_pose, move_seconds
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("运行时长须为非负有限数；0表示持续运行直到停止")
        if zero_command_after_s is not None:
            if not math.isfinite(zero_command_after_s) or zero_command_after_s <= 0:
                raise ValueError('速度归零时间须为有限正数')
            if seconds != 0 or home_only or command_provider is not None:
                raise ValueError('定时归零只用于seconds=0的固定速度策略执行')
        if (not math.isfinite(command_ramp_s) or command_ramp_s < 0
                or (seconds > 0 and command_ramp_s > seconds)
                or (zero_command_after_s is not None and command_ramp_s > zero_command_after_s)):
            raise ValueError('指令渐增时间须为非负有限数，且不能超过运行或速度归零时间')
        twist = np.asarray(twist, dtype=float)
        if twist.shape != (3,) or not np.all(np.isfinite(twist)) or np.any(np.abs(twist) > COMMAND_LIMITS):
            raise ValueError("试机指令范围：|vx|<=0.4、|vy|<=0.3、|wz|<=1.0")
        policy_cycle_s, policy_gap_s = policy_timing_limits(self.cfg['control'])
        overrun_threshold_s = policy_overrun_threshold(self.cfg['control'])
        if home_only:
            policy_cycle_s = self.cfg['control']['max_cycle_s']
            policy_gap_s = policy_cycle_s + .01
            overrun_threshold_s = .020
        self._stdin_provider = stdin_provider
        path = log_path("motion" if motion else "read_only")
        session_path = path.with_suffix('.session.json')
        self.session = {"started": datetime.now().astimezone().isoformat(),
                        "status": "initializing", "config": self.cfg,
                        "motion": motion, "twist": twist, "from_pose": from_pose,
                        "move_seconds": move_seconds, "home_only": home_only,
                        "take_over_hold": self.take_over_hold, "command_ramp_s": command_ramp_s,
                        "policy_seconds": seconds, "unlimited_duration": seconds == 0,
                        "max_policy_cycle_s": policy_cycle_s,
                        "max_policy_sample_gap_s": policy_gap_s,
                        "policy_overrun_threshold_s": overrun_threshold_s,
                        "zero_command_after_s": zero_command_after_s,
                        "zero_command_transition": None}
        self.session['command_source'] = ('terminal' if stdin_provider is not None else
                                          'live' if command_provider is not None else 'fixed')
        self.session['sitstand_enabled'] = self.sitstand_policy is not None
        self.session['policy_transitions'] = []
        self.session['policy_model_transitions'] = []
        session_path.write_text(json.dumps(self.session, default=json_default, ensure_ascii=False, indent=2)+'\n')
        stop_hint = ('Ctrl+C 可停止；控制就绪后 q 无需回车即可退出卸力。'
                     if getattr(stdin_provider, 'single_key', False)
                     else 'Ctrl+C 或输入 q 后回车停止。')
        print(f"日志: {path}\n{stop_hint}", flush=True)
        self._install_signals()
        owned_bus = self.bus is None
        owned_imu = self.imu is None
        sample_log = None
        try:
            if self.bus is None:
                self.bus = ServoBus(**self.cfg["serial"])
            if motion and self.jaw_ids:
                from .jaw import JawController
                self.jaw = JawController(self.cfg, self.bus, command_source=self.jaw_source)
                self.jaw.preflight()
                self.jaw.warmup()
            # Ordinary startup requires torque OFF. Explicit hold takeover uses
            # an already initialized IMU and matching RAM gains, without dropping
            # the supported robot between its current hold and policy control.
            if self.take_over_hold and self.imu is None:
                raise StopRequested('上力接管需要已初始化的IMU；不得在接管时重新校偏')
            if motion:
                for sid in self.ids:
                    if self.bus.read_configuration(sid)["torque"] != (1 if self.take_over_hold else 0):
                        raise StopRequested(f"舵机{sid}已上力；请先托住并运行relax")
            # Loading ONNX/native modules can block Python threads long enough
            # to time out live sensor acquisition. Finish it before starting IMU.
            if self.policy is None and not home_only:
                model = Path(self.cfg["model"])
                self.policy = OnnxPolicy(model if model.is_absolute() else ROOT / model)
            if self.imu is None:
                if self.cfg.get('imu_process', False):
                    from .imu_process import ProcessImuReader
                    self.imu = ProcessImuReader(self.cfg)
                else:
                    self.imu = ImuReader(self.cfg)
                device = getattr(self.imu, 'imu', None)
                if device is not None:
                    self.session['imu_configuration'] = dataclasses.asdict(device.config)
                    self.session['gyro_bias_rad_s'] = device.gyro_bias_rad_s
            q, state, _ = self._read()
            if motion:
                check_posture(q, state, self.cfg)
                self._home_plan(q)
                q = self._prepare_motors(q)
                q = self._home(q)
            if stdin_provider is not None:
                reason = stdin_provider.start_policy()
                if reason:
                    raise StopRequested(reason)
                print(getattr(stdin_provider, 'ready_message',
                      '数字控制已就绪：模型从零速度开始。每条命令输入后回车：'
                      '1前进/2后退/3左移/4右移/5左转/6右转；0停步保持模型；q退出卸力。'),
                      flush=True)
            previous_q = q.copy()
            previous_time = time.monotonic()
            previous_action = np.zeros(14)
            previous_target = HOME.copy() if motion else q.copy()
            started = time.monotonic()
            self.session['policy_start_monotonic_s'] = started
            deadline = started
            overruns = 0
            count = 0
            previous_loop_timing = None
            previous_policy_mode = 'walking'
            previous_policy_id = 'walking:original'
            roulade_monitor = None
            self.session['status'] = 'running'
            sample_log = BackgroundLineLog(path)
            with sample_log as log:
                log.write(json.dumps({"kind": "header", "motion": motion, "twist": twist,
                                      "config": self.cfg, "qd_source": "position_finite_difference"},
                                     default=json_default, ensure_ascii=False) + "\n")
                # Obtain a full control-period difference for the first velocity observation.
                self.stop.wait(0.02)
                deadline = time.monotonic()
                while seconds == 0 or time.monotonic() - started < seconds:
                    stop_check_started = time.monotonic()
                    self._check_stop()
                    begun = time.monotonic()
                    q, state, feedback = self._read()
                    stamp = time.monotonic()
                    elapsed = stamp - previous_time
                    if not 0.002 <= elapsed <= policy_gap_s:
                        self.session['sampling_timing_fault'] = {
                            't': stamp - started,
                            'sample_gap_ms': elapsed * 1000,
                            'limit_ms': policy_gap_s * 1000,
                            # Includes the prior inference/send/logging, wait and scheduling.
                            'gap_before_this_read_ms': (begun - previous_time) * 1000,
                            # Includes feedback/IMU retrieval and their validation.
                            'this_read_ms': (stamp - begun) * 1000,
                            'policy_target_sent_this_step': False,
                            'before_read_stop_check_ms': (begun-stop_check_started)*1000,
                            'previous_loop_timing': previous_loop_timing,
                        }
                        raise StopRequested(f"关节采样间隔异常: {elapsed*1000:.1f}ms")
                    if np.max(np.abs(q - previous_q)) > 0.45:
                        raise StopRequested("反馈角度跳变；可能过零、掉包或标定错误")
                    qd = (q - previous_q) / elapsed
                    # Ramp the velocity request, not the policy actions or servo clock.
                    requested_command = (twist if command_provider is None else
                                         np.asarray(command_provider(), dtype=float))
                    if zero_command_after_s is not None and stamp - started >= zero_command_after_s:
                        requested_command = np.zeros(3)
                        if self.session['zero_command_transition'] is None:
                            self.session['zero_command_transition'] = {
                                't': stamp - started, 'from': twist.copy(),
                                'to': requested_command.copy()}
                            print('前进计时结束：速度指令已归零，模型继续运行；Ctrl+C或q回车停止并卸力。',
                                  flush=True)
                    if (requested_command.shape != (3,) or not np.all(np.isfinite(requested_command))
                            or np.any(np.abs(requested_command) > COMMAND_LIMITS)):
                        raise StopRequested('实时速度命令无效或超出vx/vy/wz运行范围')
                    fraction = 1.0 if command_ramp_s == 0 else min(1.0, max(0.0, (stamp-started)/command_ramp_s))
                    command = requested_command * (fraction*fraction*(3-2*fraction))
                    # A posture flag is a binary policy command, not velocity.
                    # Keep it outside velocity bounds/ramping and retain the
                    # shared previous raw action when changing policy sessions.
                    policy_mode = getattr(command_provider, 'policy_mode', 'walking')
                    if policy_mode not in ('walking', 'stopping', 'sit', 'stand', 'ground_pick', 'roulade'):
                        raise StopRequested('手柄姿态模式无效')
                    walking_variant, sitstand_variant, walking_policy, sitstand_policy = self._selected_policy_variants(command_provider)
                    roulade_progress = None
                    if policy_mode == 'roulade':
                        if self.cfg['control'].get('roulade_enabled', False) is not True:
                            raise StopRequested('滚翻未显式启用')
                        if self.roulade_policy is None or self.sitstand_policy is None:
                            raise StopRequested('滚翻需要滚翻权重及站坐权重')
                        finish_roll = getattr(command_provider, 'finish_roulade', None)
                        if not callable(finish_roll):
                            raise StopRequested('滚翻控制器缺少完成回调')
                        roll_tilt = check_posture(q, state, self.cfg, policy_mode='roulade')
                        if roulade_monitor is None:
                            roulade_monitor = RouladeMonitor(stamp, state, self.cfg)
                        completed = roulade_monitor.update(stamp, state, roll_tilt)
                        roulade_progress = roulade_monitor.snapshot(stamp)
                        if completed:
                            finish_roll()
                            policy_mode = getattr(command_provider, 'policy_mode', None)
                            if policy_mode != 'stand':
                                raise StopRequested('滚翻结束后未切回站立保持')
                            walking_variant, sitstand_variant, walking_policy, sitstand_policy = self._selected_policy_variants(command_provider)
                            roulade_monitor = None
                    elif roulade_monitor is not None:
                        raise StopRequested('滚翻未确认恢复前拒绝切换动作')
                    active_policy = walking_policy
                    policy_id = f'walking:{walking_variant}'
                    policy_config_key = 'model' if walking_variant == 'original' else 'updated_walking_model'
                    if policy_mode in ('sit', 'stand'):
                        if sitstand_policy is None:
                            raise StopRequested('未加载站坐权重；拒绝姿态切换')
                        command = np.array([float(policy_mode == 'sit'), 0., 0.])
                        active_policy = sitstand_policy
                        policy_id = f'sitstand:{sitstand_variant}'
                        policy_config_key = ('sitstand_model' if sitstand_variant == 'default'
                                             else f'sitstand_{sitstand_variant}_model')
                    elif policy_mode == 'stopping':
                        command = np.zeros(3)
                    elif policy_mode == 'ground_pick':
                        if self.ground_pick_policy is None:
                            raise StopRequested('未加载俯身权重；拒绝动作切换')
                        command = np.asarray(getattr(command_provider, 'ground_pick_command', None), dtype=float)
                        if (command.shape != (3,) or not np.isfinite(command).all()
                                or abs(command[2]) > 1e-7
                                or abs(np.linalg.norm(command[:2]) - 1.) > 1e-5):
                            raise StopRequested('俯身相位指令无效；需要[cos(phase),sin(phase),0]')
                        active_policy = self.ground_pick_policy
                        policy_id, policy_config_key = 'ground_pick', 'ground_pick_model'
                    elif policy_mode == 'roulade':
                        command = np.zeros(3)
                        active_policy = self.roulade_policy
                        policy_id, policy_config_key = 'roulade', 'roulade_model'
                    tilt = check_posture(q, state, self.cfg, policy_mode=policy_mode) if motion else None
                    if policy_mode != previous_policy_mode:
                        self.session['policy_transitions'].append({
                            't': stamp-started, 'from': previous_policy_mode, 'to': policy_mode})
                        previous_policy_mode = policy_mode
                    if policy_id != previous_policy_id:
                        self.session['policy_model_transitions'].append({
                            't': stamp-started, 'from': previous_policy_id, 'to': policy_id,
                            'mode': policy_mode,
                            'model_path': str(getattr(active_policy, 'model_path',
                                                      self.cfg.get(policy_config_key)))})
                        previous_policy_id = policy_id
                    obs = build_obs61(state.base_ang_vel, state.projected_gravity,
                                      q, qd, previous_action, twist=command)
                    inference_started = time.monotonic()
                    action = np.zeros(14) if home_only else active_policy.infer(obs)
                    inference_finished = time.monotonic()
                    target = HOME.copy() if home_only else action_to_targets(action)
                    raw_target = target.copy()
                    constrained_joints = []
                    if policy_mode in ('sit', 'stand') and not home_only:
                        try:
                            target, constrained_joints = constrain_posture_target(target, self.cfg)
                        except StopRequested as exc:
                            # This constraint runs before validate_targets.
                            # Preserve the rejected raw network output too.
                            self.session['rejected_policy_step'] = {
                                'policy_mode': policy_mode,
                                'policy_id': policy_id,
                                't': stamp - started, 'reason': str(exc), 'q': q,
                                'qd': qd, 'obs': obs, 'observation': obs,
                                'action': action, 'raw_target': raw_target,
                                'target': target, 'previous_action': previous_action,
                                'previous_target': previous_target,
                                'command': command, 'sent': False,
                            }
                            raise
                    if motion and home_only and np.max(np.abs(q - HOME)) > self.cfg["control"]["home_tolerance_rad"]:
                        raise StopRequested("HOME保持期间偏离目标；停止并检查支撑、卡滞和增益")
                    if time.monotonic() - begun > policy_cycle_s:
                        self.session['policy_cycle_timing_fault'] = {
                            'stage': 'before_target_send', 't': stamp - started,
                            'cycle_ms': (time.monotonic() - begun) * 1000,
                            'limit_ms': policy_cycle_s * 1000,
                            'read_ms': (stamp - begun) * 1000,
                            'inference_ms': (inference_finished - inference_started) * 1000,
                            'policy_target_sent_this_step': False}
                        raise StopRequested("传感器读取/推理已超时；拒绝发送过时目标")
                    target_error = None
                    try:
                        target_ticks, warnings = validate_targets(
                            target, previous_target, self.cfg, return_warnings=True)
                        if self.cfg['control'].get('log_only_target_angles', False):
                            target_error = '; '.join(warnings) or None
                    except StopRequested as exc:
                        if motion:
                            self.session['rejected_policy_step'] = {
                                'policy_mode': policy_mode,
                                'policy_id': policy_id,
                                't': stamp - started, 'reason': str(exc), 'q': q,
                                'qd': qd, 'obs': obs, 'observation': obs,
                                'action': action, 'raw_target': raw_target,
                                'target': target, 'previous_action': previous_action,
                                'previous_target': previous_target,
                                'command': command, 'sent': False,
                            }
                            raise
                        target_error = str(exc)
                    send_started = time.monotonic()
                    if motion:
                        self._send(target, previous_target, cycle_started=begun,
                                   cycle_limit_s=policy_cycle_s, validated_ticks=target_ticks)
                    computation_s = time.monotonic() - begun
                    if computation_s > policy_cycle_s:
                        self.session['policy_cycle_timing_fault'] = {
                            'stage': 'after_target_send', 't': stamp - started,
                            'cycle_ms': computation_s * 1000,
                            'limit_ms': policy_cycle_s * 1000,
                            'read_ms': (stamp - begun) * 1000,
                            'inference_ms': (inference_finished - inference_started) * 1000,
                            'policy_target_sent_this_step': bool(motion)}
                        raise StopRequested(f"单周期用时{computation_s*1000:.1f}ms过长")
                    overruns = overruns + 1 if computation_s > overrun_threshold_s else 0
                    if overruns >= self.cfg["control"]["max_overruns"]:
                        self.session['control_timing_fault'] = {
                            't': stamp-started, 'consecutive_overruns': overruns,
                            'threshold_ms': overrun_threshold_s*1000,
                            'max_overruns': self.cfg["control"]["max_overruns"],
                            'cycle_ms': computation_s*1000,
                            'read_ms': (stamp-begun)*1000,
                            'inference_ms': (inference_finished-inference_started)*1000,
                            'send_ms': (time.monotonic()-send_started)*1000,
                            'policy_target_sent_this_step': bool(motion),
                        }
                        raise StopRequested(f"连续控制周期超过{overrun_threshold_s*1000:g}ms；"
                                            "先解决Radxa时序再上力")
                    logging_started = time.monotonic()
                    sample_json = json.dumps({"kind": "sample", "t": stamp - started,
                              "q": q, "qd": qd, "gyro": state.base_ang_vel,
                              "gravity": state.projected_gravity, "action": action,
                              "target": target, "command": command, "cycle_ms": computation_s*1000,
                              "requested_command": requested_command,
                              "policy_mode": policy_mode,
                              "policy_id": policy_id,
                              "walking_variant": walking_variant,
                              "sitstand_variant": sitstand_variant,
                              "roulade_progress": roulade_progress,
                              "raw_target": raw_target,
                              "posture_encoder_limited_joints": constrained_joints,
                              "inference_ms": (inference_finished-inference_started)*1000,
                              "prepare_policy_ms": (inference_started-stamp)*1000,
                              "send_ms": (logging_started-send_started)*1000,
                              "sample_gap_ms": elapsed * 1000,
                              "gap_before_this_read_ms": (begun - previous_time) * 1000,
                              "this_read_ms": (stamp - begun) * 1000,
                              "tilt_deg": tilt, "target_check": target_error,
                              "feedback": feedback}, default=json_default) + "\n"
                    serialization_finished = time.monotonic()
                    log.write(sample_json)
                    logging_finished = time.monotonic()
                    count += 1
                    if count % 25 == 0:
                        print(f"{stamp-started:.1f}s 周期{computation_s*1000:.1f}ms "
                              f"gravity={np.round(state.projected_gravity,3)}"
                              + (f" cmd={np.round(command,3)}" if command_provider is not None else "")
                              + (f" 目标检查:{target_error}" if target_error else ""), flush=True)
                    console_finished = time.monotonic()
                    previous_q, previous_time = q, stamp
                    # Policy observes its previous raw output, not HOME-relative
                    # clipped/smoothed motor commands. home_only never uses policy output.
                    previous_action = np.zeros(14) if home_only else action
                    previous_target = target
                    deadline += 0.02
                    if deadline < time.monotonic():
                        deadline = time.monotonic()
                    wait_started = time.monotonic()
                    wait_requested = max(0, deadline - wait_started)
                    self.stop.wait(wait_requested)
                    previous_loop_timing = {
                        'sample_t': stamp-started,
                        'processing_after_read_ms': (logging_started-stamp)*1000,
                        'serialize_and_write_ms': (logging_finished-logging_started)*1000,
                        'serialize_ms': (serialization_finished-logging_started)*1000,
                        'write_ms': (logging_finished-serialization_finished)*1000,
                        'console_ms': (console_finished-logging_finished)*1000,
                        'bookkeeping_ms': (wait_started-console_finished)*1000,
                        'wait_requested_ms': wait_requested*1000,
                        'wait_elapsed_ms': (time.monotonic()-wait_started)*1000,
                    }
            return {"samples": count, "log": str(path), "motion": motion}
        finally:
            failure = sys.exc_info()[1]
            # Restore the operator's terminal even if later device/log cleanup
            # fails, including direct Controller callers outside policy_trial.
            close_stdin = getattr(stdin_provider, 'close', None)
            if callable(close_stdin):
                try:
                    close_stdin()
                except Exception as exc:
                    print(f'恢复终端失败: {exc}', file=sys.stderr)
            self.session['status'] = 'stopped' if failure is not None else 'completed'
            self.session['stop_reason'] = str(failure) if failure is not None else self.reason
            self.session['finished'] = datetime.now().astimezone().isoformat()
            self.stop.set()
            if self.enabled:
                # Best effort on a broken cable cannot guarantee de-energization.
                for attempt in range(3):
                    try:
                        self.bus.set_torque(self.motion_ids, False, verify=attempt == 0)
                        break
                    except Exception as exc:
                        print(f"卸力尝试{attempt+1}失败: {exc}；必要时断开舵机电源。", file=sys.stderr)
                self.enabled = False
            if self.jaw is not None:
                try:
                    self.jaw.disable()
                except Exception as exc:
                    print(f'嘴部卸力/增益恢复失败: {exc}', file=sys.stderr)
            for sid, gains in self.old_gains.items():
                try:
                    self.bus.set_pid_ram([sid], gains["p"], gains["d"], gains["i"])
                except Exception as exc:
                    print(f"舵机{sid} RAM增益恢复失败: {exc}；断电重启恢复EPROM值。", file=sys.stderr)
            if self.watchdog is not None:
                self.watchdog.join(timeout=0.3)
            if owned_imu and self.imu is not None:
                try:
                    self.imu.close()
                except Exception as exc:
                    print(f"关闭IMU失败: {exc}", file=sys.stderr)
            if owned_bus and self.bus is not None:
                try:
                    self.bus.close()
                except Exception as exc:
                    print(f"关闭串口失败: {exc}", file=sys.stderr)
            for sig, handler in self._signals.items():
                signal.signal(sig, handler)
            if sample_log is not None:
                self.session['log_writer'] = sample_log.finish()
                stats = self.session['log_writer']
                if stats['error'] or stats['dropped_lines'] or stats['writer_running']:
                    print(f"日志写入未完整完成: {stats}", file=sys.stderr)
            try:
                session_path.write_text(json.dumps(self.session, default=json_default, ensure_ascii=False, indent=2)+'\n')
            except Exception as exc:
                print(f"保存会话日志失败: {exc}", file=sys.stderr)
