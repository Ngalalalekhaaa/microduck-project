"""Independent mouth motion on the existing bus; never a policy joint.

Endpoints are absolute encoder ticks. Their degree conversion describes the
servo encoder, not the mechanical angle of the beak. Only an explicit command
arms the mouth; constructing or preflighting this controller performs no writes.
"""
from __future__ import annotations

import copy
import math
import os
import time
from collections.abc import Mapping


X_BUTTON = 307  # Linux BTN_X / BTN_NORTH; configurable for the actual HID map.
SOFT_GAINS = {'p': 5, 'd': 0, 'i': 0}


class JawError(RuntimeError):
    pass


def jaw_config(cfg):
    settings = cfg.get('jaw')
    if settings is None:
        return None
    if not isinstance(settings, dict) or type(settings.get('enabled', False)) is not bool:
        raise ValueError('jaw.enabled 必须是布尔值')
    if not settings.get('enabled', False):
        return None
    result = copy.deepcopy(settings)
    sid = result.get('id')
    if type(sid) is not int or not 0 <= sid <= 253:
        raise ValueError('嘴部舵机ID必须是0..253的整数')
    if sid in [joint['id'] for joint in cfg.get('joints', ())]:
        raise ValueError('嘴部舵机ID不能与14个策略关节重复')
    if result.get('mapping_verified') is not True:
        raise ValueError('先手动核对嘴部舵机ID并确认 mapping_verified')
    for name in ('closed_tick', 'open_tick'):
        tick = result.get(name)
        if type(tick) is not int or not 0 <= tick <= 4095:
            raise ValueError(f'jaw.{name} 必须是0..4095的整数刻度')
    distance = abs(result['open_tick'] - result['closed_tick'])
    if distance == 0 or distance > 2048:
        raise ValueError('嘴部开闭刻度须不同，且不能跨编码器单圈边界')
    duration = result.setdefault('transition_seconds', .8)
    if (isinstance(duration, bool) or not isinstance(duration, (int, float))
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError('jaw.transition_seconds 必须是正有限数')
    button = result.setdefault('button_code', X_BUTTON)
    if type(button) is not int or button not in (307, 308):
        raise ValueError('jaw.button_code 须为已核对的Xbox X事件码307或308')
    return result


def jaw_ids(cfg):
    settings = jaw_config(cfg)
    return [] if settings is None else [settings['id']]


class JawButtonToggle:
    """Count press edges in the parent's one reader, including short presses."""
    def __init__(self, reader, button_code=X_BUTTON, *, clock=time.monotonic):
        self.reader, self.button_code, self.clock = reader, button_code, clock
        self.desired_open = None
        self._seen = None

    def baseline(self, sample=None):
        sample = self.reader.latest() if sample is None else sample
        self._seen = sample.button_press_counts.get(self.button_code, 0)

    def reset(self):
        self.desired_open, self._seen = None, None

    def poll(self):
        sample = self.reader.latest()
        count = sample.button_press_counts.get(self.button_code, 0)
        if (sample.stop_requested or not sample.connected
                or not 0 <= self.clock() - sample.timestamp <= .3):
            self.baseline(sample)
            return None
        if self._seen is None or count < self._seen:
            self.baseline(sample)
            return None
        presses = count - self._seen
        self._seen = count
        if presses == 0:
            return None
        # The first new X press requests OPEN. Two presses between polls cancel.
        current = False if self.desired_open is None else self.desired_open
        self.desired_open = not current if presses % 2 else current
        return self.desired_open


class JawCommandReader:
    """Receive the parent's latest requested state through an inherited pipe."""
    def __init__(self, fd):
        self.fd, self._buffer, self._eof = fd, b'', False
        os.set_blocking(fd, False)

    def poll(self):
        if self.fd is None:
            raise JawError('嘴部控制通道已关闭')
        if self._eof:
            raise JawError('嘴部控制父进程已断开，停止动作')
        while True:
            try:
                chunk = os.read(self.fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                self._eof = True
                break
            self._buffer += chunk
            if len(self._buffer) > 65536:
                raise JawError('嘴部控制通道数据过长')
        if self._eof:
            # Loss of supervision takes precedence over a queued final command.
            raise JawError('嘴部控制父进程已断开，停止动作')
        latest = None
        while b'\n' in self._buffer:
            line, self._buffer = self._buffer.split(b'\n', 1)
            if line not in (b'0', b'1'):
                raise JawError('嘴部控制通道收到无效命令')
            latest = line == b'1'
        if self._eof and latest is None:
            raise JawError('嘴部控制父进程已断开，停止动作')
        return latest

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)


class JawController:
    """Read/arm one servo, then add its bounded trajectory to the owner's writes.

    ``targets`` does not send trajectory packets: the bus owner merges its result
    with its existing targets. Initial preload and torque changes are deliberately
    performed through this same bus. Call ``disable`` during the owner's cleanup,
    including when an attempted preload or gain change raises an exception.
    """
    def __init__(self, cfg, bus, command_source=None, clock=time.monotonic):
        self.cfg, self.bus = cfg, bus
        self.settings = jaw_config(cfg)
        self.id = None if self.settings is None else self.settings['id']
        self.ids = [] if self.id is None else [self.id]
        self.command_source, self.clock = command_source, clock
        self.prepared, self.active = False, False
        self.desired_open = None
        self.configuration, self.old_gains = None, None
        self.gains_changed = False
        self._target = self._start = self._goal = None
        self._elapsed = self._duration = 0.
        self._last_time = None

    def _feedback(self, feedback=None):
        if feedback is None:
            value = self.bus.read_feedback(self.id)
        elif isinstance(feedback, Mapping):
            if self.id not in feedback:
                raise JawError(f'缺少嘴部舵机{self.id}反馈')
            value = feedback[self.id]
        else:
            value = feedback
        if value.servo_id != self.id:
            raise JawError('嘴部反馈的舵机ID不匹配')
        control = self.cfg['control']
        if not 0 <= self.clock() - value.received_at <= control['max_cycle_s']:
            raise JawError(f'嘴部舵机{self.id}反馈过期')
        if value.status:
            raise JawError(f'嘴部舵机{self.id}状态错误0x{value.status:02x}')
        lo, hi = control['servo_voltage_range']
        if not lo <= value.voltage_v <= hi:
            raise JawError(f'嘴部舵机{self.id}电压{value.voltage_v}V超出{lo}..{hi}V')
        if value.temperature_c >= control['servo_max_temperature_c']:
            raise JawError(f'嘴部舵机{self.id}温度过高')
        if type(value.position_ticks) is not int or not 0 <= value.position_ticks <= 4095:
            raise JawError(f'嘴部舵机{self.id}不是单圈0..4095反馈')
        if self.configuration is not None and not (
                self.configuration['position_min'] <= value.position_ticks
                <= self.configuration['position_max']):
            raise JawError('嘴部当前位置超出舵机配置位置范围')
        return value

    def _configuration(self):
        conf = self.bus.read_configuration(self.id)
        if self.cfg['serial'].get('register_profile') != 'hls_2':
            raise JawError('嘴部控制要求已核实的HD1910 HLS_2寄存器配置')
        if conf['mode'] != 4 or conf['angle_resolution'] != 1:
            raise JawError('嘴部舵机需要mode=4、angle_resolution=1')
        if conf['torque'] not in (0, 1):
            raise JawError('嘴部舵机扭矩状态无效')
        lo, hi = conf['position_min'], conf['position_max']
        if not 0 <= lo < hi <= 4095 or any(
                not lo <= self.settings[name] <= hi for name in ('closed_tick', 'open_tick')):
            raise JawError('嘴部开闭端点超出舵机配置位置范围')
        if conf['torque'] == 1 and conf['ram_gains'] != SOFT_GAINS:
            raise JawError('已上力嘴部舵机只允许接管P5/D0/I0保持')
        self.configuration = conf
        return conf

    def preflight(self):
        if self.settings is None:
            return None
        self._configuration()
        value = self._feedback()
        return {'id': self.id, 'position_ticks': value.position_ticks,
                'torque': self.configuration['torque'],
                'closed_tick': self.settings['closed_tick'],
                'open_tick': self.settings['open_tick']}

    def warmup(self):
        """Optionally prepare RAM gains outside a timed loop; never send goals.

        This is an explicit operation, separate from read-only ``preflight``.
        The owner must call ``disable`` on failures to restore modified gains.
        """
        if self.settings is None:
            return
        conf = self._configuration()
        self._feedback()
        if self.old_gains is None:
            self.old_gains = dict(conf['ram_gains'])
        if conf['torque'] == 0 and conf['ram_gains'] != SOFT_GAINS:
            self.gains_changed = True
            self._set_gains(SOFT_GAINS)
            self.configuration['ram_gains'] = dict(SOFT_GAINS)

    def _set_gains(self, gains):
        # supported_home's narrow PositionOnlyBus allows RAM gains only in this
        # explicit preparation/restoration window. Ordinary ServoBus needs none.
        limited = hasattr(self.bus, 'soft_gain_write_allowed')
        old_allowed = self.bus.soft_gain_write_allowed if limited else None
        if limited:
            self.bus.soft_gain_write_allowed = True
        try:
            self.bus.set_pid_ram(self.ids, gains['p'], gains['d'], gains['i'], verify=True)
        finally:
            if limited:
                self.bus.soft_gain_write_allowed = old_allowed

    def _prepare(self, feedback=None):
        # Recheck torque at the first command; a preflight may be much earlier.
        if self.configuration is None:
            conf = self._configuration()
        else:
            conf = dict(self.configuration)
            conf['torque'] = self.bus.read(self.id, 40, 1)[0]
            if conf['torque'] not in (0, 1):
                raise JawError('嘴部舵机扭矩状态无效')
            if conf['torque'] == 1 and conf['ram_gains'] != SOFT_GAINS:
                raise JawError('已上力嘴部舵机只允许接管P5/D0/I0保持')
        if self.old_gains is None:
            self.old_gains = dict(conf['ram_gains'])
        if conf['torque'] == 0 and conf['ram_gains'] != SOFT_GAINS:
            self.gains_changed = True
            self._set_gains(SOFT_GAINS)
        # Read again after gain operations, unless the owner supplied a fresh
        # synchronized batch that is still within the same control deadline.
        value = self._feedback(feedback)
        start = value.position_ticks
        lo, hi = sorted((self.settings['open_tick'], self.settings['closed_tick']))
        if not lo - 32 <= start <= hi + 32:
            raise JawError('嘴部当前位置远离标定范围；先卸力手动摆回开闭区间')
        self.active = True  # Even a failed goal write may already cause motion.
        self.bus.sync_positions({self.id: start})
        actual = int.from_bytes(self.bus.read(self.id, 42, 2), 'little')
        if actual != start:
            raise JawError('嘴部当前角预装目标未读回确认，拒绝上力')
        if conf['torque'] == 0:
            self.bus.set_torque(self.ids, True, verify=True)
        self.prepared = True
        self._target = float(start)
        self._last_time = self.clock()
        return value

    def _advance(self):
        now = self.clock()
        # A paused loop resumes the path gradually instead of jumping to its end.
        elapsed = min(.04, max(0., now - self._last_time))
        self._last_time = now
        self._elapsed = min(self._duration, self._elapsed + elapsed)
        alpha = 1. if self._duration == 0 else self._elapsed / self._duration
        alpha = alpha * alpha * (3 - 2 * alpha)
        self._target = self._start + alpha * (self._goal - self._start)

    def targets(self, feedback=None, command=None):
        if self.settings is None:
            return {}
        if command is None and self.command_source is not None:
            command = self.command_source.poll()
        if command is not None and type(command) is not bool:
            raise ValueError('嘴部命令必须是bool或None')
        if not self.prepared:
            if command is None:
                return {}
            self._prepare(feedback)
        else:
            self._feedback(feedback)
            if self._goal is not None:
                self._advance()
        if command is not None and (command != self.desired_open or self._goal is None):
            self.desired_open = command
            self._start = self._target
            self._goal = float(self.settings['open_tick' if command else 'closed_tick'])
            span = abs(self.settings['open_tick'] - self.settings['closed_tick'])
            # Preserve the same peak speed for reversals and for an initial
            # hand-set position slightly outside the calibrated interval.
            self._duration = self.settings['transition_seconds'] * abs(self._goal - self._start) / span
            self._elapsed = 0.
        return {self.id: int(round(self._target))}

    def disable(self, verify=True, *, restore_gains=True):
        if self.settings is None:
            return
        self.bus.set_torque(self.ids, False, verify=verify)
        self.active = self.prepared = False
        self.desired_open = None
        self._target = self._start = self._goal = None
        if self.configuration is not None:
            self.configuration['torque'] = 0
        if restore_gains and self.gains_changed and self.old_gains is not None:
            self._set_gains(self.old_gains)
            self.gains_changed = False
            if self.configuration is not None:
                self.configuration['ram_gains'] = dict(self.old_gains)
