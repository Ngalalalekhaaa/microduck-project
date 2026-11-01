"""Supported static HOME tracking for duck1's four hips; default is read-only.

Only IDs 2/3/12/13 can receive volatile P/D/I, torque or position writes.
No calibration, EEPROM, current-limit or walking-control configuration is edited.
The operator must support the body throughout each torque-off gain transition.
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

from microduck_deploy.config import calibration_arrays, load_config, require_calibration
from microduck_deploy.hip_level import selected_indices
from microduck_deploy.policy import HOME, q_to_ticks
from microduck_deploy.servo import ServoBus
from supported_home import PositionOnlyBus

VERSION = 'hip-position-hold-v1'
HIP_IDS = frozenset((2, 3, 12, 13))
P_CHOICES = (5, 8, 12, 16)
PERIOD = .1
MAX_SECONDS = 120
MAX_AGE = .2
MAX_ERROR_TICKS = 8 * 4096 / 360
MAX_CURRENT_MA = 800
REACHED_TICKS = 6


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


class HipHoldBus(PositionOnlyBus):
    """Packet gate remains effective even if a caller bypasses public methods."""
    def __init__(self, *args, allow_motion=False, **kwargs):
        self.allow_motion = allow_motion
        super().__init__(*args, **kwargs)

    def _send(self, packet):
        packet = bytes(packet)
        if (len(packet) < 6 or packet[:2] != b'\xff\xff'
                or packet[3] != len(packet)-4 or sum(packet[2:]) % 256 != 255):
            raise ValueError('舵机数据包格式错误')
        instruction = packet[4]
        if instruction not in (2, 130):
            allowed = False
            if self.allow_motion and instruction == 131 and packet[2] == 254 and len(packet) >= 9:
                address, size = packet[5:7]
                body = packet[7:-1]
                if ((address, size) in ((40, 1), (42, 2), (50, 3))
                        and body and len(body) % (size+1) == 0):
                    ids = list(body[::size+1])
                    values = [body[i+1:i+1+size] for i in range(0, len(body), size+1)]
                    allowed = len(set(ids)) == len(ids) and set(ids) <= HIP_IDS
                    if address == 40:
                        allowed = allowed and all(value in (b'\x00', b'\x01') for value in values)
                    elif address == 42:
                        allowed = allowed and all(int.from_bytes(value, 'little') <= 4095 for value in values)
                    else:
                        allowed = allowed and all(value[0] in P_CHOICES and value[1:] == b'\x00\x00'
                                                  for value in values)
            if not allowed:
                raise ValueError('只允许2/3/12/13号的目标、扭矩和P=5/8/12/16,D=I=0运行参数')
        return super()._send(packet)

    def _sync_ram(self, address, size, values):
        # Parent PositionOnlyBus admits only P=5. The stricter packet gate above
        # admits this utility's four known profiles; ServoBus retains RAM bounds.
        return ServoBus._sync_ram(self, address, size, values)


@contextmanager
def exclusive_serial(cfg):
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/supported_sequence.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
        if owner.returncode != 1:
            raise ValueError('串口被占用，请先关闭网页/其他控制程序：'+owner.stdout+owner.stderr)
        yield


@contextmanager
def stop_signals():
    def stopped(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, stopped) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
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


class HipPositionHold:
    def __init__(self, cfg, bus, report, *, moving=False, config_path=None,
                 persist=None, clock=None, sleep=None, stop=None):
        self.cfg, self.bus, self.report = cfg, bus, report
        self.moving, self.config_path = moving, config_path
        self.clock, self.sleep, self.stop = clock or time.monotonic, sleep or time.sleep, stop or check_stop
        self.persist = persist or (lambda: None)
        self.original_cfg = copy.deepcopy(cfg)
        self.config_hash = digest(config_path) if config_path else None
        self.ids = [joint['id'] for joint in cfg['joints']]
        self.home = dict(zip(self.ids, map(int, q_to_ticks(HOME, *calibration_arrays(cfg)))))
        self.targets = self.home.copy()
        self.torques = dict.fromkeys(self.ids, 1)
        self.hardware = {}
        self.touched = []
        self.started = self.clock()
        self.cleanup_started = None
        self.ready = False
        report.update(version=VERSION, selected_ids=sorted(HIP_IDS), move=moving,
                      status='initializing', actions=[], telemetry=[], cleanup={},
                      calibration_saved=False, robot_config_changed=False,
                      eeprom_writes=False, current_limit_writes=False)

    def _guard(self, cleanup=False):
        if cleanup:
            if self.clock()-self.cleanup_started > 30:
                raise ValueError('参数恢复超过30秒')
            return
        self.stop()
        if self.clock()-self.started > MAX_SECONDS:
            raise ValueError('达到120秒操作时限')
        if self.cfg != self.original_cfg or (self.config_path and digest(self.config_path) != self.config_hash):
            raise ValueError('robot.json在运行期间发生变化')

    def _persist(self, cleanup=False):
        try:
            self.persist()
        except Exception as exc:
            if not cleanup:
                raise
            self.report.setdefault('report_write_errors', []).append(repr(exc))

    def _intent(self, event, *, cleanup=False, **values):
        self.report['actions'].append({'event': event, 'elapsed_s': self.clock()-self.started, **values})
        self._persist(cleanup=cleanup)
        self._guard(cleanup=cleanup)

    def _feedback_row(self, sid, feedback, torque, goal, now):
        numeric = (feedback.position_ticks, feedback.velocity_raw, feedback.current_ma,
                   feedback.voltage_v, feedback.temperature_c, feedback.received_at)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f'ID{sid}反馈含无效数字')
        if (feedback.status or not 0 <= feedback.position_ticks <= 4095
                or feedback.position_ticks != int(feedback.position_ticks)
                or torque not in (0, 1) or not 0 <= goal <= 4095):
            raise ValueError(f'ID{sid}状态/单圈位置异常')
        if not 0 <= now-feedback.received_at <= MAX_AGE:
            raise ValueError(f'ID{sid}反馈超过200ms或时间戳无效')
        return {'tick': int(feedback.position_ticks), 'goal': goal, 'torque': torque,
                'velocity_raw': feedback.velocity_raw, 'current_ma': feedback.current_ma,
                'voltage_v': feedback.voltage_v, 'temperature_c': feedback.temperature_c,
                'received_at': feedback.received_at}

    def _electrical(self, sid, row):
        low, high = self.cfg['control']['servo_voltage_range']
        if (not low <= row['voltage_v'] <= high
                or row['temperature_c'] >= self.cfg['control']['servo_max_temperature_c']
                or abs(row['current_ma']) >= MAX_CURRENT_MA):
            raise ValueError(f'ID{sid}电压/温度/电流超限: {row["voltage_v"]}V, '
                             f'{row["temperature_c"]}°C, {row["current_ma"]}mA')

    def _snapshot(self, *, enforce=True):
        self._guard()
        began = self.clock()
        feedback = self.bus.read_feedback_many(self.ids, sync=True)
        rows = {}
        for sid in self.ids:
            torque = self.bus.read(sid, 40, 1)[0]
            goal = int.from_bytes(self.bus.read(sid, 42, 2), 'little')
            rows[sid] = self._feedback_row(sid, feedback[sid], torque, goal, self.clock())
        if self.clock()-began > MAX_AGE:
            raise ValueError('整组舵机读取超过200ms')
        self._guard()
        for sid, row in rows.items():
            if sid in HIP_IDS:
                self._electrical(sid, row)
            if self.moving and enforce:
                if row['torque'] != self.torques[sid] or row['goal'] != self.targets[sid]:
                    raise ValueError(f'ID{sid}扭矩/目标与预期不一致，不能并行运行其他控制程序')
                if sid in HIP_IDS:
                    hw = self.hardware[sid]
                    if not hw['position_min'] <= row['tick'] <= hw['position_max']:
                        raise ValueError(f'ID{sid}实际位置超出硬件单圈限位')
                    if abs(row['tick']-self.home[sid]) > MAX_ERROR_TICKS:
                        raise ValueError(f'ID{sid}实际位置偏离原HOME超过8°')
        self.report['last_feedback'] = {str(sid): row for sid, row in rows.items()}
        self.report['telemetry'].append({'elapsed_s': self.clock()-self.started,
                                         'hips': {str(sid): rows[sid] for sid in sorted(HIP_IDS)}})
        return rows

    def _fresh(self, rows, cleanup=False):
        self._guard(cleanup=cleanup)
        now = self.clock()
        if any(not 0 <= now-row['received_at'] <= MAX_AGE for row in rows.values()):
            raise ValueError('写入前的舵机采样超过200ms')

    def preflight(self):
        require_calibration(self.cfg, imu=False)
        selected_indices(self.cfg)
        if self.cfg['serial']['register_profile'] != 'hls_2':
            raise ValueError('仅支持已确认的HD1910 HLS_2配置')
        self._guard()
        self.hardware = {sid: self.bus.read_configuration(sid) for sid in self.ids}
        for sid, hw in self.hardware.items():
            if (hw['mode'] != 4 or hw['angle_resolution'] != 1
                    or not 0 <= hw['position_min'] < hw['position_max'] <= 4095
                    or not hw['position_min'] <= self.home[sid] <= hw['position_max']):
                raise ValueError(f'ID{sid}模式、分辨率或单圈限位不符合要求')
            if self.moving and sid in HIP_IDS:
                gains = hw['ram_gains']
                if gains not in ({'p': p, 'd': 0, 'i': 0} for p in P_CHOICES):
                    raise ValueError(f'ID{sid}原RAM参数不属于可恢复的P=5/8/12/16,D=I=0范围')
        rows = self._snapshot()
        self.report.update(hardware_before=copy.deepcopy(self.hardware),
                           initial_feedback={str(sid): row for sid, row in rows.items()},
                           original_home=self.home.copy(), original_gains={sid: self.hardware[sid]['ram_gains']
                                                                         for sid in sorted(HIP_IDS)},
                           status='ready')
        # No hardware writes are possible before the complete backup is saved.
        self._persist()
        self.ready = True

    def _single(self, sid, *, cleanup=False):
        self._guard(cleanup=cleanup)
        began = self.clock()
        feedback = self.bus.read_feedback(sid)
        torque = self.bus.read(sid, 40, 1)[0]
        goal = int.from_bytes(self.bus.read(sid, 42, 2), 'little')
        row = self._feedback_row(sid, feedback, torque, goal, self.clock())
        if self.clock()-began > MAX_AGE:
            raise ValueError(f'ID{sid}恢复检查超过200ms')
        self._electrical(sid, row)
        hw = self.hardware[sid]
        if (not hw['position_min'] <= row['tick'] <= hw['position_max']
                or abs(row['tick']-self.home[sid]) > MAX_ERROR_TICKS):
            raise ValueError(f'ID{sid}实际位置不在本次安全范围')
        return row

    def _seed_hold(self, sid, row, *, cleanup=False):
        tick = row['tick']
        hw = self.hardware[sid]
        if (not hw['position_min'] <= tick <= hw['position_max']
                or abs(tick-self.home[sid]) > MAX_ERROR_TICKS):
            raise ValueError(f'ID{sid}不能以超出本次范围的位置恢复保持')
        if row['torque'] != 0:
            raise ValueError(f'ID{sid}写入当前位置前未确认卸力')
        self._intent('seed_current_position', cleanup=cleanup, id=sid, target=tick)
        self._fresh({sid: row}, cleanup=cleanup)
        self.bus.sync_positions({sid: tick})
        self.targets[sid] = tick
        # A position write can enable HD1910 torque; never assume it stays OFF.
        fresh = self._single(sid, cleanup=cleanup)
        if fresh['goal'] != tick or abs(fresh['tick']-tick) > 2:
            raise ValueError(f'ID{sid}当前姿态目标回读不匹配或支撑移动超过2刻度')
        if fresh['torque'] == 0:
            self._intent('enable_seeded_hold', cleanup=cleanup, id=sid)
            self._fresh({sid: fresh}, cleanup=cleanup)
            self.bus.set_torque([sid], True, verify=True)
        self.torques[sid] = 1
        fresh = self._single(sid, cleanup=cleanup)
        if fresh['torque'] != 1 or fresh['goal'] != tick or abs(fresh['tick']-tick) > 2:
            raise ValueError(f'ID{sid}重新上力未确认')
        return fresh

    def _change_gain(self, sid, p):
        rows = self._snapshot()
        self._intent('begin_gain_transition', id=sid, old=self.hardware[sid]['ram_gains'],
                     requested={'p': p, 'd': 0, 'i': 0})
        self._fresh(rows)
        self.touched.append(sid)  # Before ambiguous torque-off/gain writes.
        self.bus.set_torque([sid], False, verify=True)
        self.torques[sid] = 0
        self._snapshot()
        self._intent('set_ram_gain', id=sid, p=p, d=0, i=0)
        self.bus.set_pid_ram([sid], p, 0, 0, verify=True)
        fresh = self._snapshot()[sid]
        if abs(fresh['tick']-rows[sid]['tick']) > MAX_ERROR_TICKS:
            raise ValueError(f'ID{sid}卸力后移动超过8°，请保持支撑')
        self._seed_hold(sid, fresh)
        self._snapshot()
        if self.bus.read(sid, 50, 3) != bytes((p, 0, 0)):
            raise ValueError(f'ID{sid}RAM参数复核失败')

    def _ramp_home(self):
        while any(self.targets[sid] != self.home[sid] for sid in HIP_IDS):
            began = self.clock()
            rows = self._snapshot()
            targets = {sid: self.targets[sid] + (1 if self.home[sid] > self.targets[sid]
                                                else -1 if self.home[sid] < self.targets[sid] else 0)
                       for sid in sorted(HIP_IDS)}
            self._fresh(rows)
            self.report['actions'].append({'event': 'home_step', 'elapsed_s': self.clock()-self.started,
                                           'targets': targets.copy()})
            self.bus.sync_positions(targets)
            self.targets.update(targets)
            self._snapshot()  # Verify the target registers, not just transmission.
            self.sleep(max(0, PERIOD-(self.clock()-began)))

    def _window(self, seconds, *, enforce=True):
        samples = []
        started = self.clock()
        while self.clock()-started < seconds:
            began = self.clock()
            rows = self._snapshot(enforce=enforce)
            samples.append(rows)
            self.sleep(max(0, PERIOD-(self.clock()-began)))
        summary = {}
        for sid in sorted(HIP_IDS):
            ticks = [row[sid]['tick'] for row in samples]
            middle = max(1, len(ticks)//2)
            first, second = ticks[:middle], ticks[middle:] or ticks[-1:]
            summary[str(sid)] = {'home': self.home[sid], 'actual_mean': sum(ticks)/len(ticks),
                                 'actual_min': min(ticks), 'actual_max': max(ticks),
                                 'max_error_ticks': max(abs(tick-self.home[sid]) for tick in ticks),
                                 'span_ticks': max(ticks)-min(ticks),
                                 'drift_ticks': abs(sum(first)/len(first)-sum(second)/len(second)),
                                 'torque': samples[-1][sid]['torque'], 'goal': samples[-1][sid]['goal'],
                                 'current_ma': samples[-1][sid]['current_ma']}
        stable = all(row['span_ticks'] <= 4 and row['drift_ticks'] <= 2 for row in summary.values())
        all_held = all(row[sid]['goal'] == self.home[sid] and row[sid]['torque'] == 1
                       for row in samples for sid in self.ids)
        reached = all_held and all(row['max_error_ticks'] <= REACHED_TICKS for row in summary.values())
        result = {'seconds': self.clock()-started, 'sample_count': len(samples), 'hips': summary,
                  'stable': stable, 'all_body_goals_held': all_held,
                  'target_reached': reached, 'passed': stable and reached}
        self.report['measurement'] = result
        return result

    def observe(self, *, seconds=5):
        if not self.ready:
            raise ValueError('必须先完成只读预检')
        if not math.isfinite(seconds) or not .2 <= seconds <= 30:
            raise ValueError('只读采样时长必须为0.2..30秒')
        measured = self._window(seconds, enforce=False)
        self.report['status'] = 'read_only_reached' if measured['passed'] else 'read_only_not_reached'
        self._persist()
        return measured['passed']

    def _verify_final_configuration(self, p):
        after = {sid: self.bus.read_configuration(sid) for sid in self.ids}
        for sid, actual in after.items():
            expected = copy.deepcopy(self.hardware[sid])
            expected['torque'] = 1
            if sid in HIP_IDS:
                expected['ram_gains'] = {'p': p, 'd': 0, 'i': 0}
            if actual != expected:
                raise ValueError(f'ID{sid}最终配置与预期不一致')
        self.report['hardware_after'] = after
        self.report['gains_verified'] = True

    def _restore(self):
        self.cleanup_started = self.clock()
        for sid in reversed(self.touched):
            state = self.report['cleanup'][str(sid)] = {'status': 'restoring'}
            try:
                self._intent('restore_torque_off', cleanup=True, id=sid)
                self.bus.set_torque([sid], False, verify=True)
                self.torques[sid] = 0
                self._single(sid, cleanup=True)
                gain = self.hardware[sid]['ram_gains']
                self._intent('restore_original_gain', cleanup=True, id=sid, gain=gain)
                self.bus.set_pid_ram([sid], gain['p'], gain['d'], gain['i'], verify=True)
                row = self._single(sid, cleanup=True)
                held = self._seed_hold(sid, row, cleanup=True)
                if self.bus.read(sid, 50, 3) != bytes(gain[key] for key in ('p', 'd', 'i')):
                    raise ValueError('恢复原RAM参数回读失败')
                state.update(status='original_gains_restored_holding_current_position',
                             gain=copy.deepcopy(gain), held_tick=held['tick'], goal=held['goal'], torque=1)
            except BaseException as exc:
                state.update(status='restoration_failed', error=repr(exc))
                for _ in range(2):
                    try:
                        self.bus.set_torque([sid], False, verify=True)
                        self.torques[sid] = 0
                        state.update(status='left_torque_off', torque=0)
                        break
                    except BaseException as off_error:
                        state.setdefault('torque_off_errors', []).append(repr(off_error))
                else:
                    state.update(status='torque_unknown', torque=None)
            self._persist(cleanup=True)

    def run(self, *, p=16):
        if not self.moving or not self.ready:
            raise ValueError('执行需要--move、支撑声明和完整预检')
        if isinstance(p, bool) or p not in P_CHOICES:
            raise ValueError('P只允许5、8、12、16，D和I固定为0')
        self.report['requested_ram_gains'] = {'p': p, 'd': 0, 'i': 0}
        try:
            for sid in sorted(HIP_IDS):
                self._change_gain(sid, p)
            self._ramp_home()
            self._window(1.)  # Allow loaded motion to settle before measuring.
            measured = self._window(2.)
            if not measured['stable']:
                raise ValueError('到位采样发现持续晃动/漂移，恢复原RAM参数')
            self._verify_final_configuration(p)
            self._guard()
            self.report['status'] = ('holding_home' if measured['passed']
                                     else 'gains_applied_but_target_not_reached')
            self._persist()
            return measured['passed']
        except BaseException as exc:
            self.report.update(status='aborted', error=repr(exc))
            if self.touched:
                self._restore()
            self._persist(cleanup=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'robot.json')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--supported', action='store_true', help='确认全程托稳躯干，允许四颗髋关节逐颗短暂卸力')
    parser.add_argument('--p', type=int, choices=P_CHOICES, default=16)
    parser.add_argument('--seconds', type=float, default=5, help='只读状态采样时长，默认5秒')
    parser.add_argument('--version', action='version', version=VERSION)
    args = parser.parse_args(argv)
    if args.move and not args.supported:
        parser.error('--move必须同时提供--supported；全过程必须托稳机器人')
    if not math.isfinite(args.seconds) or not .2 <= args.seconds <= 30:
        parser.error('--seconds必须为0.2..30秒')
    path = args.config.resolve()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'config_path': str(path),
              'mode': 'supported_motion' if args.move else 'read_only'}
    log = ROOT/'logs'/('hip_position_hold_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    result = 1
    try:
        cfg = load_config(path)
        report['config_sha256'] = digest(path)
        with exclusive_serial(cfg), stop_signals():
            options = dict(cfg['serial'])
            options['timeout'] = max(.05, float(options.get('timeout', .05)))
            with HipHoldBus(**options, allow_motion=args.move) as bus:
                task = HipPositionHold(cfg, bus, report, moving=args.move, config_path=path,
                                       persist=lambda: save_report(log, report))
                task.preflight()
                if args.move:
                    print(f'仅四颗髋关节切换运行参数P={args.p},D=I=0；逐颗短暂卸力，请全程托稳。', flush=True)
                    passed = task.run(p=args.p)
                else:
                    print('只读检查，不改参数、不发送动作。', flush=True)
                    passed = task.observe(seconds=args.seconds)
                print(json.dumps(report.get('measurement', {}), ensure_ascii=False, indent=2), flush=True)
                if args.move:
                    print(f'四颗髋关节RAM参数已回读确认P={args.p},D=I=0；robot.json未改。', flush=True)
                    print('已保持原HOME目标；实际到位检查'+('通过。' if passed else '未通过，暂不运行自动调平。'), flush=True)
                result = 0 if passed else 2
    except KeyboardInterrupt:
        report.setdefault('error', 'KeyboardInterrupt')
        print('已中止。若已开始参数切换，会尝试恢复已处理关节的原参数并保持当前位置；请继续托稳。', flush=True)
        result = 130
    except Exception as exc:
        report.setdefault('error', repr(exc))
        print(f'停止：{exc}', file=sys.stderr, flush=True)
        result = 1
    finally:
        for sid, state in report.get('cleanup', {}).items():
            print(f'ID{sid}恢复状态：{state["status"]}，扭矩={state.get("torque", "未知")}', flush=True)
        save_report(log, report)
        print(f'报告：{log}', flush=True)
    return result


if __name__ == '__main__':
    sys.exit(main())
