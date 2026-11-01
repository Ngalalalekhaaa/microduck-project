"""Check calibrated HOME; optionally measure and trim a supported ground stance.

This cannot recover unknown joint installation offsets from a trunk IMU.
It never changes encoder calibration, IMU mounting, training HOME or policy.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from microduck_deploy.config import load_config, require_calibration, calibration_arrays
from microduck_deploy.home_assist import (
    gravity_to_roll_pitch, q_check, pose_for_trim,
    central_difference_jacobian, solve_trim_step,
)
from microduck_deploy.imu_process import ProcessImuReader
from microduck_deploy.policy import HOME, JOINT_LIMITS, q_to_ticks, ticks_to_q
from microduck_deploy.runtime import check_feedback, json_default
from supported_home import PositionOnlyBus, path_ticks


class ReadOnlyBus(PositionOnlyBus):
    def _send(self, packet):
        if packet[4] not in (2, 130):
            raise ValueError('只读模式拒绝舵机写指令')
        return super()._send(packet)


@contextmanager
def stop_signals():
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    saved = {sig: signal.signal(sig, interrupt)
             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def confirm_ground():
    print('\n双脚放到平整、不打滑的地面，自然承重；旁边扶护，但不要把躯干固定住。\n'
          'IMU须牢固装在躯干，安装方向已核对。准备好后输入 GROUND 并回车。\n'
          '120秒未确认会停止并卸力；Ctrl+C也会卸力。', flush=True)
    readable, _, _ = select.select([sys.stdin], [], [], 120)
    if not readable or sys.stdin.readline().strip().upper() != 'GROUND':
        raise ValueError('未确认承重站姿')


@contextmanager
def exclusive_serial(cfg):
    """Share the existing sequence lock and suspend only the known servo web UI."""
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    web = Path.home()/'servo_web_hd1910'
    stopped = False
    with (logs/'supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            pidfile = web/'server.pid'
            if pidfile.exists():
                pid = int(pidfile.read_text().strip())
                proc = Path('/proc')/str(pid)
                if proc.exists():
                    if b'server.py' not in (proc/'cmdline').read_bytes() or (proc/'cwd').resolve() != web:
                        raise ValueError('网页PID与预期程序不符，未停止它')
                    os.kill(pid, signal.SIGTERM)
                    stopped = True
                    for _ in range(50):
                        if not proc.exists():
                            break
                        time.sleep(.1)
            usage = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
            if usage.returncode != 1:
                raise ValueError('串口仍被占用：'+usage.stdout+usage.stderr)
            yield
        finally:
            if stopped:
                with (web/'logs/console.log').open('ab', buffering=0) as out:
                    server = subprocess.Popen(['bash', 'run.sh'], cwd=web,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
                (web/'server.pid').write_text(str(server.pid)+'\n')
                print('网页调试台已恢复。', flush=True)


class StanceMeasurements:
    def __init__(self, cfg, bus, imu):
        self.cfg, self.bus, self.imu = cfg, bus, imu
        self.ids = [j['id'] for j in cfg['joints']]
        self.zero, self.direction = calibration_arrays(cfg)
        self.hardware = {}
        self.last_target = HOME.copy()

    def read(self):
        feedback = self.bus.read_feedback_many(self.ids, sync=True)
        check_feedback(feedback, self.cfg)
        q = ticks_to_q([feedback[i].position_ticks for i in self.ids], self.zero, self.direction)
        state = self.imu.latest(self.cfg['control']['imu_max_age_s'])
        return q, state

    def sample(self, *, settle_s=1, duration_s=.6, require_still=True):
        attitudes, positions, gyros, accelerations, stamps = [], [], [], [], []
        started = time.monotonic()
        while time.monotonic()-started < settle_s+duration_s:
            q, state = self.read()
            if time.monotonic()-started >= settle_s and (not stamps or state.timestamp != stamps[-1]):
                attitudes.append(gravity_to_roll_pitch(state.projected_gravity))
                positions.append(q)
                gyros.append(state.base_ang_vel)
                accelerations.append(np.linalg.norm(state.accel_m_s2))
                stamps.append(state.timestamp)
            time.sleep(.02)
        if len(stamps) < 10:
            raise ValueError('有效IMU静止样本不足')
        attitude = np.mean(attitudes, axis=0)
        std_deg = np.rad2deg(np.std(attitudes, axis=0))
        if require_still and (max(std_deg) > .6 or np.mean(np.linalg.norm(gyros, axis=1)) > .15
                              or not 8.3 < np.mean(accelerations) < 11.3):
            raise ValueError('站姿尚未稳定；停止自动微调，检查晃动/打滑/扶持')
        return {'attitude_rad': attitude, 'attitude_deg': np.rad2deg(attitude),
                'attitude_std_deg': std_deg,
                'joint_check': q_check(np.mean(positions, axis=0), target=self.last_target, ids=self.ids),
                'measured_q_rad': np.mean(positions, axis=0)}

    def check_target(self, q):
        if (np.any(q < JOINT_LIMITS[:, 0]) or np.any(q > JOINT_LIMITS[:, 1])
                or not np.isfinite(q).all()):
            raise ValueError('站姿微调目标超出模型机械范围')
        ticks = q_to_ticks(q, self.zero, self.direction)
        for i, sid in enumerate(self.ids):
            c = self.hardware[sid]
            if not c['position_min'] <= ticks[i] <= c['position_max']:
                raise ValueError(f'ID{sid}站姿微调越过硬件位置范围')
        return ticks

    def move_trim(self, trim):
        target = pose_for_trim(trim)
        end = self.check_target(target)
        start = self.check_target(self.last_target)
        # At most one degree/s, and only small offsets around the nominal HOME.
        seconds = max(1., 1.5*float(np.rad2deg(np.max(np.abs(target-self.last_target)))))
        for goal in path_ticks(start, end, steps=math.ceil(seconds*50)):
            q, state = self.read()
            if np.max(np.abs(q-self.last_target)) > np.deg2rad(10):
                raise ValueError('关节跟踪偏离当前站姿超过10°')
            if np.max(np.abs(gravity_to_roll_pitch(state.projected_gravity))) > np.deg2rad(20):
                raise ValueError('机身明显倾斜；停止自动微调')
            self.bus.sync_positions(dict(zip(self.ids, map(int, goal))))
            time.sleep(.02)
        self.last_target = target


def trim_ground_stance(measurements, report, *, iterations=12):
    u = np.zeros(2)
    before = measurements.sample()
    report['before'] = before
    print('HOME实测：'+json.dumps(before, default=json_default, ensure_ascii=False), flush=True)
    if np.max(np.abs(before['attitude_deg'])) > 12:
        raise ValueError('初始倾角超过12°，先检查机械参考和IMU安装，暂不自动微调')
    if np.linalg.norm(before['attitude_rad']) <= np.deg2rad(1.5):
        report.update(result='机身已接近目标倾角，无需微调', trim_rad=u, after=before)
        return
    # Identify response on this robot, rather than assuming joint signs or contact.
    probe = np.deg2rad(1.5)
    plus, minus = [], []
    report['probes'] = []
    for column in range(2):
        step = np.zeros(2)
        step[column] = probe
        for sign, samples in ((1, plus), (-1, minus)):
            measurements.move_trim(sign*step)
            observation = measurements.sample()
            samples.append(observation['attitude_rad'])
            report['probes'].append({'trim_rad': sign*step, 'measurement': observation})
        measurements.move_trim(u)
    jacobian = central_difference_jacobian(plus, minus, probe)
    report['attitude_response'] = jacobian
    current = measurements.sample()
    best_u, best = u.copy(), current
    misses = 0
    report['iterations'] = []
    for k in range(iterations):
        error = -current['attitude_rad']  # training HOME has a level trunk frame
        if np.linalg.norm(error) <= np.deg2rad(1.5):
            break
        delta = solve_trim_step(jacobian, error, u)
        if np.max(np.abs(delta)) < np.deg2rad(.05):
            break
        trial = u+delta
        measurements.move_trim(trial)
        observation = measurements.sample()
        if np.linalg.norm(observation['attitude_rad']) > np.linalg.norm(current['attitude_rad'])+np.deg2rad(.4):
            report['rejected_step'] = {'trim_rad': trial, 'measurement': observation,
                                       'restored_trim_rad': u.copy()}
            measurements.move_trim(u)
            raise ValueError('倾角没有按实测响应改善；已撤销本步，检查接触或扶持')
        u, current = trial, observation
        if np.linalg.norm(current['attitude_rad']) < np.linalg.norm(best['attitude_rad'])-np.deg2rad(.05):
            best_u, best, misses = u.copy(), current, 0
        else:
            misses += 1
        report['iterations'].append({'step': k+1, 'trim_rad': u.copy(), 'measurement': current})
        print(f'微调{k+1}：前后/左右参数={np.round(np.rad2deg(u),2)}°，'
              f'机身roll/pitch={np.round(current["attitude_deg"],2)}°', flush=True)
        if misses >= 2:
            break
    if np.linalg.norm(best['attitude_rad']) < np.linalg.norm(current['attitude_rad']):
        measurements.move_trim(best_u)
        u, current = best_u, measurements.sample()
    report.update(trim_rad=u, after=current,
        result='已减小机身倾斜' if np.linalg.norm(current['attitude_rad']) <= np.deg2rad(1.5)
               else '已到微调范围或次数上限；不能据此认定真实关节已标定',
        target_q_rad=pose_for_trim(u),
        target_ticks=measurements.check_target(pose_for_trim(u)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--move', action='store_true', help='慢速回HOME，随后保持上力')
    parser.add_argument('--level', action='store_true', help='落脚后用IMU做小范围站姿微调')
    args = parser.parse_args()
    if args.level and not args.move:
        parser.error('--level需要--move；默认仅测量')
    cfg = load_config(ROOT/'robot.json')
    require_calibration(cfg)
    print('本工具检查已标定关节并辅助调姿；无法自动识别14关节的真实安装零位。', flush=True)
    report = {'started': datetime.now().astimezone().isoformat(),
              'move': args.move, 'level': args.level, 'calibration_changed': False,
              'policy_changed': False, 'training_home_rad': HOME}
    log = ROOT/'logs'/('imu_home_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    log.parent.mkdir(exist_ok=True)
    motion_owned = False
    success = False
    child = None
    with exclusive_serial(cfg):
        reader = ProcessImuReader(cfg, calibrate=False)
        try:
            if args.move:
                print('先托住躯干和头部、双脚悬空；现在开始慢速HOME。', flush=True)
                # Reuse the existing verified movement path and soft RAM gains.
                with PositionOnlyBus(**cfg['serial']) as probe_bus:
                    configs = [probe_bus.read_configuration(j['id']) for j in cfg['joints']]
                torques = {c['torque'] for c in configs}
                if torques not in ({0}, {1}):
                    raise ValueError('当前扭矩状态混合，先统一状态')
                command = [sys.executable, str(ROOT/'scripts/supported_home.py'),
                           '--target', 'home', '--move', '--log-only-feedback']
                if torques == {0}:
                    command.append('--soft-gains')
                motion_owned = True
                child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
                if child.wait():
                    raise ValueError('慢速HOME未完成')
                if args.level:
                    confirm_ground()
            bus_type = PositionOnlyBus if args.move else ReadOnlyBus
            with bus_type(**cfg['serial']) as bus:
                measurements = StanceMeasurements(cfg, bus, reader)
                if args.level:
                    measurements.hardware = {sid: bus.read_configuration(sid) for sid in measurements.ids}
                    if any(c['torque'] != 1 or c['mode'] != 4 or c['angle_resolution'] != 1
                           for c in measurements.hardware.values()):
                        raise ValueError('站姿微调需要位置模式和全部上力')
                    trim_ground_stance(measurements, report)
                else:
                    report.update(result='已完成测量', measurement=measurements.sample(require_still=False))
                    print(json.dumps(report['measurement'], default=json_default, ensure_ascii=False, indent=2))
                success = True
        except BaseException as exc:
            report.update(result='stopped', error=repr(exc))
            raise
        finally:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if motion_owned and not success:
                try:
                    with PositionOnlyBus(**cfg['serial']) as bus:
                        bus.set_torque([j['id'] for j in cfg['joints']], False, verify=True)
                    print('流程停止，已卸力；请继续托住机器人。', flush=True)
                except Exception as exc:
                    report['torque_off_error'] = repr(exc)
                    print('无法确认卸力，请手动断开舵机电源：'+str(exc), flush=True)
            reader.close()
            log.write_text(json.dumps(report, default=json_default, ensure_ascii=False, indent=2)+'\n')
            print('报告：'+str(log), flush=True)
    if args.move:
        print('完成，保持当前姿态上力。零位和训练HOME未更改。\n'
              '微调结果只用于本次站姿；启动现有行走命令仍会回训练HOME。', flush=True)
    return 0


if __name__ == '__main__':
    try:
        with stop_signals():
            raise SystemExit(main())
    except KeyboardInterrupt:
        print('用户停止。', flush=True)
        raise SystemExit(130)
