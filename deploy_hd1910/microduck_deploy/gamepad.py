"""Read Xbox controls through Linux evdev; supply velocity commands to the policy."""
from __future__ import annotations

import dataclasses
import math
import re
import subprocess
import threading
import time


COMMAND_LIMITS = (.4, .3, 1.0)  # vx/vy in m/s; yaw in rad/s.
DEFAULT_GAMEPAD_LIMITS = (.3, .3, 1.0)
MAX_BACKWARD_VX = .3  # Match the deployed controller's backward limit.
HOME_BUTTON = 304  # Xbox A / BTN_SOUTH.
WALK_BUTTON = 315  # Xbox Menu / BTN_START.
SITSTAND_BUTTON = 308  # Xbox Y / BTN_WEST (X is Linux code 307).
GROUND_PICK_BUTTON = 311  # Xbox RB / BTN_TR.
GROUND_PICK_SECONDS = 4.0
ROULADE_BUTTON = 310  # Xbox LB / BTN_TL; requires explicit configuration.
DPAD_UP_BUTTON = 544
DPAD_DOWN_BUTTON = 545
DPAD_LEFT_BUTTON = 546
DPAD_RIGHT_BUTTON = 547
DPAD_BUTTONS = frozenset((DPAD_UP_BUTTON, DPAD_DOWN_BUTTON, DPAD_LEFT_BUTTON, DPAD_RIGHT_BUTTON))


class GamepadError(RuntimeError):
    pass


class GamepadStop(GamepadError):
    """The operator pressed B to end this policy session."""


def add_gamepad_args(parser):
    parser.add_argument('--gamepad', action='store_true', help='Xbox摇杆实时速度命令')
    parser.add_argument('--gamepad-device', default='auto')
    parser.add_argument('--max-vx', type=float, default=DEFAULT_GAMEPAD_LIMITS[0])
    parser.add_argument('--max-vy', type=float, default=DEFAULT_GAMEPAD_LIMITS[1])
    parser.add_argument('--max-wz', type=float, default=DEFAULT_GAMEPAD_LIMITS[2])


def validate_gamepad_args(parser, args):
    for name, bound in zip(('max_vx', 'max_vy', 'max_wz'), COMMAND_LIMITS):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 < value <= bound:
            parser.error(f'--{name.replace("_", "-")}须在(0,{bound}]内')


def gamepad_options(args):
    if not args.gamepad:
        return []
    return ['--gamepad', '--gamepad-device', args.gamepad_device,
            '--max-vx', str(args.max_vx), '--max-vy', str(args.max_vy),
            '--max-wz', str(args.max_wz)]


@dataclasses.dataclass(frozen=True)
class PadSample:
    timestamp: float
    connected: bool
    axes: dict[str, float]
    raw_axes: dict[str, int]
    buttons: tuple[int, ...]
    stop_requested: bool = False
    error: str | None = None
    button_press_counts: dict[int, int] = dataclasses.field(default_factory=dict)


def normalize_axis(value, minimum, maximum, deadzone=.12):
    """Normalize signed/unsigned stick ranges and rescale outside the dead zone."""
    if maximum <= minimum or not 0 <= deadzone < 1:
        raise GamepadError('无效的摇杆范围或死区')
    center = (minimum + maximum) / 2
    position = max(-1., min(1., (value - center) / ((maximum - minimum) / 2)))
    if abs(position) <= deadzone:
        return 0.
    return math.copysign((abs(position) - deadzone) / (1 - deadzone), position)


def axes_to_twist(axes, *, max_vx=DEFAULT_GAMEPAD_LIMITS[0], max_vy=DEFAULT_GAMEPAD_LIMITS[1], max_wz=DEFAULT_GAMEPAD_LIMITS[2]):
    # Robot frame: X forward, Y left, positive yaw turns left.
    longitudinal_limit = min(max_vx, MAX_BACKWARD_VX) if axes['ABS_Y'] > 0 else max_vx
    return (-axes['ABS_Y'] * longitudinal_limit, -axes['ABS_X'] * max_vy,
            -axes['ABS_RX'] * max_wz)


def bluez_connected(address):
    """Check the BLE link too: a stale UHID node can outlive a disconnect."""
    if not re.fullmatch(r'[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}', address or ''):
        return True  # USB input devices do not have a Bluetooth MAC.
    result = subprocess.run(['busctl', '--system', 'get-property', 'org.bluez',
        '/org/bluez/hci0/dev_' + address.upper().replace(':', '_'),
        'org.bluez.Device1', 'Connected'], capture_output=True, text=True, timeout=.25)
    return result.returncode == 0 and result.stdout.strip() == 'b true'


def open_xbox_device(path='auto'):
    try:
        from evdev import InputDevice, list_devices
    except ImportError as exc:
        raise GamepadError("缺少evdev：.venv/bin/python -m pip install 'evdev>=1.9,<2'") from exc
    if path != 'auto':
        return InputDevice(path)
    matches = []
    for node in list_devices():
        try:
            device = InputDevice(node)
        except OSError:
            continue
        caps = device.capabilities(absinfo=False)
        if ('xbox' in device.name.lower() or device.info.vendor == 0x045E) and {
                0, 1}.issubset(caps.get(3, [])) and ({2, 3} & set(caps.get(3, []))) and 305 in caps.get(1, []):
            matches.append(device)
        else:
            device.close()
    if len(matches) != 1:
        # BlueZ can retain an old UHID node after re-pairing the same controller.
        # Multiple physical pads still need an explicit selection.
        addresses = {device.uniq.lower() for device in matches if device.uniq}
        if matches and len(addresses) == 1 and all(device.uniq for device in matches):
            selected = max(matches, key=lambda device: int(device.path.rsplit('event', 1)[1]))
            for device in matches:
                if device is not selected:
                    device.close()
            return selected
        for device in matches:
            device.close()
        raise GamepadError('未发现唯一的Xbox手柄；先配对并运行test_xbox.py --list，'
                           '多个手柄时用--gamepad-device指定event路径')
    return matches[0]


class GamepadReader:
    """Poll kernel state in a thread; stationary sticks do not falsely expire."""
    AXES = {'ABS_X': 0, 'ABS_Y': 1, 'ABS_RX': 3}
    B_BUTTON = 305  # BTN_EAST, the Xbox B button.

    def __init__(self, path='auto', *, deadzone=.12, factory=open_xbox_device):
        self.device = factory(path)
        self.deadzone = deadzone
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.thread = None
        self.stop_requested = False
        self.button_press_counts = {}
        self.previous_buttons = set()
        self.last_link_check = float('-inf')
        try:
            # Linux's generic Bluetooth HID driver maps this Xbox's right stick
            # to ABS_Z/RZ; xpad/xpadneo commonly use ABS_RX/RY instead.
            caps = self.device.capabilities(absinfo=False).get(3, [])
            self.hat_codes = tuple(code for code in (16, 17) if code in caps)
            self._hat_values = {code: 0 for code in self.hat_codes}
            self._dpad_key_down = set()
            self._dpad_down = set()
            right_x = 3 if 3 in caps else 2
            self.axis_codes = {'ABS_X': 0, 'ABS_Y': 1, 'ABS_RX': right_x}
            self.axis_info = {name: self.device.absinfo(code) for name, code in self.axis_codes.items()}
            if any(info is None for info in self.axis_info.values()):
                raise GamepadError('手柄缺少ABS_X、ABS_Y或ABS_RX；先核对输入映射')
            self.metadata = {'path': self.device.path, 'name': self.device.name,
                             'uniq': self.device.uniq,
                             'axis_codes': self.axis_codes,
                             'dpad_hat_codes': self.hat_codes,
                             'dpad_button_codes': sorted(DPAD_BUTTONS),
                             'axes': {name: {'min': info.min, 'max': info.max}
                                      for name, info in self.axis_info.items()}}
            self.sample = self._poll()
            self.thread = threading.Thread(target=self._run, name='xbox-reader', daemon=True)
            self.thread.start()
        except BaseException:
            self.device.close()
            raise

    def _sync_dpad(self):
        """Count rising edges of the combined hat/key state exactly once.

        Some drivers expose ABS_HAT0X/Y, others BTN_DPAD_*, and some expose
        both. Replay events before reconciling current state so a quick tap
        ending between polls is retained without treating a hold as repeat.
        """
        down = set(self._dpad_key_down)
        x, y = self._hat_values.get(16, 0), self._hat_values.get(17, 0)
        if x:
            down.add(DPAD_LEFT_BUTTON if x < 0 else DPAD_RIGHT_BUTTON)
        if y:
            down.add(DPAD_UP_BUTTON if y < 0 else DPAD_DOWN_BUTTON)
        for button in down - self._dpad_down:
            self.button_press_counts[button] = self.button_press_counts.get(button, 0) + 1
        self._dpad_down = down

    def _poll(self):
        now = time.monotonic()
        if now - self.last_link_check >= .2:
            self.last_link_check = now
            try:
                connected = bluez_connected(self.device.uniq)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise GamepadError('无法确认手柄蓝牙连接') from exc
            if not connected:
                raise GamepadError('Xbox蓝牙连接已断开；残留input节点不代表仍连接')
        # Drain events too: this retains a brief B press between two state polls.
        try:
            events = list(self.device.read())
        except BlockingIOError:
            events = []
        pressed_events = set()
        for event in events:
            if event.type == 3 and event.code in self.hat_codes:
                self._hat_values[event.code] = event.value
                self._sync_dpad()
            elif event.type == 1 and event.code in DPAD_BUTTONS:
                if event.value == 0:
                    self._dpad_key_down.discard(event.code)
                elif event.value == 1:
                    self._dpad_key_down.add(event.code)
                self._sync_dpad()
            elif event.type == 1 and event.value == 1:
                pressed_events.add(event.code)
                self.button_press_counts[event.code] = self.button_press_counts.get(event.code, 0) + 1
                if event.code == self.B_BUTTON:
                    self.stop_requested = True
        actual_buttons = set(self.device.active_keys())
        self._dpad_key_down = actual_buttons & DPAD_BUTTONS
        for code in self.hat_codes:
            info = self.device.absinfo(code)
            if info is None:
                raise GamepadError('手柄方向键信息已不可用')
            self._hat_values[code] = info.value
        self._sync_dpad()
        buttons = tuple(sorted((actual_buttons - DPAD_BUTTONS) | self._dpad_down))
        # active_keys also catches a held button on the first device read.
        for button in set(buttons) - self.previous_buttons - pressed_events - DPAD_BUTTONS:
            self.button_press_counts[button] = self.button_press_counts.get(button, 0) + 1
        self.previous_buttons = set(buttons)
        if self.B_BUTTON in buttons:
            self.stop_requested = True
        info = {name: self.device.absinfo(code) for name, code in self.axis_codes.items()}
        if any(value is None for value in info.values()):
            raise GamepadError('手柄轴信息已不可用')
        raw = {name: value.value for name, value in info.items()}
        axes = {name: normalize_axis(raw[name], self.axis_info[name].min,
                                    self.axis_info[name].max, self.deadzone) for name in raw}
        return PadSample(time.monotonic(), True, axes, raw, buttons, self.stop_requested,
                         button_press_counts=dict(self.button_press_counts))

    def _run(self):
        while not self.halt.wait(.02):
            try:
                sample = self._poll()
            except (OSError, GamepadError) as exc:
                sample = PadSample(time.monotonic(), False, {}, {}, (),
                                   self.stop_requested, str(exc))
                with self.lock:
                    self.sample = sample
                return
            with self.lock:
                self.sample = sample

    def latest(self):
        with self.lock:
            return self.sample

    def close(self):
        self.halt.set()
        if self.thread is not None:
            self.thread.join(timeout=.5)
        self.device.close()


class GamepadCommand:
    """Slew velocity requests only; policy actions remain unfiltered."""
    def __init__(self, reader, *, max_vx=DEFAULT_GAMEPAD_LIMITS[0], max_vy=DEFAULT_GAMEPAD_LIMITS[1],
                 max_wz=DEFAULT_GAMEPAD_LIMITS[2], clock=time.monotonic):
        limits = (max_vx, max_vy, max_wz)
        if any(not math.isfinite(v) or v <= 0 or v > bound
               for v, bound in zip(limits, COMMAND_LIMITS)):
            raise GamepadError('手柄速度上限须满足0<vx<=0.4、0<vy<=0.3、0<wz<=1.0')
        self.reader, self.limits, self.clock = reader, limits, clock
        self.last_time = clock()
        self.previous = (0., 0., 0.)
        self.disconnected_reported = False

    def __call__(self):
        now = self.clock()
        sample = self.reader.latest()
        if sample.stop_requested:
            raise GamepadStop('Xbox B键：结束策略并卸力')
        if not sample.connected or not 0 <= now - sample.timestamp <= .3:
            if not self.disconnected_reported:
                print('手柄输入断开或过期：速度命令归零，策略继续运行；重新连接后请重启手柄控制。', flush=True)
                self.disconnected_reported = True
            self.previous, self.last_time = (0., 0., 0.), now
            return self.previous
        return self._slew_velocity(sample, now)

    def _slew_velocity(self, sample, now):
        desired = axes_to_twist(sample.axes, max_vx=self.limits[0],
                                max_vy=self.limits[1], max_wz=self.limits[2])
        # At most one second from zero to each configured maximum.
        elapsed = min(.1, max(0., now - self.last_time))
        self.previous = tuple(old + max(-rate * elapsed, min(rate * elapsed, want - old))
                              for old, want, rate in zip(self.previous, desired, self.limits))
        self.last_time = now
        return self.previous


class SitStandGamepadCommand(GamepadCommand):
    """Use Y for posture selection while returning only walking velocities.

    ``policy_mode`` tells the runtime which policy and posture flag to use.
    Stop the gait before sitting, and keep sticks disabled during sitting and
    the first seconds of a rise. A centered stick keeps the standing policy.
    """

    def __init__(self, reader, *, stop_seconds=.6, rise_seconds=3., **kwargs):
        super().__init__(reader, **kwargs)
        if any(not math.isfinite(value) or value < 0
               for value in (stop_seconds, rise_seconds)):
            raise GamepadError('坐站切换等待时间须为非负有限数')
        self.stop_seconds, self.rise_seconds = stop_seconds, rise_seconds
        self._policy_mode = 'walking'
        self._transition_until = None
        self._baseline_pending = True
        self._last_y_count = 0
        self._y_was_down = False
        self._baseline_y(reader.latest())

    @property
    def policy_mode(self):
        return self._policy_mode

    def _baseline_y(self, sample):
        self._last_y_count = sample.button_press_counts.get(SITSTAND_BUTTON, 0)
        self._y_was_down = SITSTAND_BUTTON in sample.buttons

    def _fresh_y_presses(self, sample):
        if self._baseline_pending:
            # The reader can run during HOME/startup. Do not execute a Y
            # press accumulated before the policy loop's first active call.
            self._baseline_y(sample)
            self._baseline_pending = False
            return 0
        down = SITSTAND_BUTTON in sample.buttons
        count = sample.button_press_counts.get(SITSTAND_BUTTON)
        if count is None:
            presses = int(down and not self._y_was_down)
            self._last_y_count += presses
        else:
            presses = max(0, count - self._last_y_count)
            self._last_y_count = count
        self._y_was_down = down
        return presses

    def _zero_velocity(self, now):
        self.previous, self.last_time = (0., 0., 0.), now
        return self.previous

    def _toggle_posture(self, now):
        if self._policy_mode == 'walking':
            self._policy_mode = 'stopping'
            self._transition_until = now + self.stop_seconds
        elif self._policy_mode in {'stopping', 'sit'}:
            self._policy_mode = 'stand'
            self._transition_until = now + self.rise_seconds
        else:
            self._policy_mode = 'sit'
            self._transition_until = None
        self._zero_velocity(now)

    def __call__(self):
        now = self.clock()
        sample = self.reader.latest()
        if sample.stop_requested:
            raise GamepadStop('Xbox B键：结束策略并卸力')
        if not sample.connected or not 0 <= now - sample.timestamp <= .3:
            if not self.disconnected_reported:
                print('手柄输入断开或过期：速度命令归零，坐站模式保持；重新连接后需重新按Y切换。', flush=True)
                self.disconnected_reported = True
            self._baseline_pending = True
            return self._zero_velocity(now)
        self.disconnected_reported = False
        if self._policy_mode == 'stopping' and now >= self._transition_until:
            self._policy_mode = 'sit'
            self._transition_until = None
        for _ in range(self._fresh_y_presses(sample)):
            self._toggle_posture(now)
        if self._policy_mode == 'stopping' and now >= self._transition_until:
            self._policy_mode = 'sit'
            self._transition_until = None
        if self._policy_mode in {'stopping', 'sit'}:
            return self._zero_velocity(now)
        if self._policy_mode == 'stand':
            desired = axes_to_twist(sample.axes, max_vx=self.limits[0],
                                    max_vy=self.limits[1], max_wz=self.limits[2])
            if now < self._transition_until or not any(desired):
                return self._zero_velocity(now)
            self._policy_mode = 'walking'
            self._transition_until = None
            self._zero_velocity(now)
        return self._slew_velocity(sample, now)


class ActionGamepadCommand(SitStandGamepadCommand):
    """Add one complete RB ground action to the existing Y posture controls.

    ``ground_pick_command`` is the three-value command expected by the ground
    policy: one full unit-circle phase over four seconds, plus a zero value.
    During gait stop and ground motion, further Y/RB presses are consumed and
    discarded. Loss of controller input zeros the walking command; an action
    already requested still completes and returns to the standing policy.
    """

    def __init__(self, reader, *, ground_pick_enabled=True, **kwargs):
        super().__init__(reader, **kwargs)
        self.ground_pick_enabled = ground_pick_enabled
        self._ground_pick_pending = False
        self._ground_pick_started = None
        self._rb_baseline_pending = True
        self._last_rb_count = 0
        self._rb_was_down = False
        self._baseline_rb(reader.latest())

    @property
    def ground_pick_command(self):
        elapsed = (0. if self._ground_pick_started is None else
                   max(0., min(GROUND_PICK_SECONDS,
                               self.clock() - self._ground_pick_started)))
        angle = 2 * math.pi * elapsed / GROUND_PICK_SECONDS
        return (math.cos(angle), math.sin(angle), 0.)

    def _baseline_rb(self, sample):
        self._last_rb_count = sample.button_press_counts.get(GROUND_PICK_BUTTON, 0)
        self._rb_was_down = GROUND_PICK_BUTTON in sample.buttons

    def _fresh_rb_presses(self, sample):
        if self._rb_baseline_pending:
            self._baseline_rb(sample)
            self._rb_baseline_pending = False
            return 0
        down = GROUND_PICK_BUTTON in sample.buttons
        count = sample.button_press_counts.get(GROUND_PICK_BUTTON)
        if count is None:
            presses = int(down and not self._rb_was_down)
            self._last_rb_count += presses
        else:
            presses = max(0, count - self._last_rb_count)
            self._last_rb_count = count
        self._rb_was_down = down
        return presses

    def _advance_ground_pick(self, now):
        if self._ground_pick_pending and now >= self._transition_until:
            self._ground_pick_pending = False
            self._policy_mode = 'ground_pick'
            self._transition_until = None
            self._ground_pick_started = now
        elif (self._policy_mode == 'ground_pick' and
              now - self._ground_pick_started >= GROUND_PICK_SECONDS):
            self._policy_mode = 'stand'
            self._ground_pick_started = None
            self._transition_until = now + self.rise_seconds

    def __call__(self):
        now = self.clock()
        sample = self.reader.latest()
        if sample.stop_requested or GamepadReader.B_BUTTON in sample.buttons:
            raise GamepadStop('Xbox B键：结束策略并卸力')
        ground_was_active = self._ground_pick_pending or self._policy_mode == 'ground_pick'
        # Advance an accepted action even without fresh input, so link loss
        # cannot leave the phase frozen at a bent-over pose.
        self._advance_ground_pick(now)
        if not sample.connected or not 0 <= now - sample.timestamp <= .3:
            if not self.disconnected_reported:
                print('手柄输入断开或过期：速度命令归零；已开始的俯身动作继续完成并站起，'
                      '断连期间按键不排队。', flush=True)
                self.disconnected_reported = True
            self._baseline_pending = self._rb_baseline_pending = True
            return self._zero_velocity(now)
        self.disconnected_reported = False
        y_presses = self._fresh_y_presses(sample)
        rb_presses = self._fresh_rb_presses(sample)
        if ground_was_active:
            # Include the completion tick: presses made during the action must
            # not unexpectedly start another motion as it finishes.
            return self._zero_velocity(now)
        if self._policy_mode == 'stopping' and now >= self._transition_until:
            self._policy_mode = 'sit'
            self._transition_until = None
        if y_presses:
            # If Y and RB arrive together, posture selection wins. RB is
            # consumed, never queued behind a sitting/rising transition.
            for _ in range(y_presses):
                self._toggle_posture(now)
        elif self.ground_pick_enabled and rb_presses and (self._policy_mode == 'walking' or
                             (self._policy_mode == 'stand' and
                              now >= self._transition_until)):
            self._ground_pick_pending = True
            self._policy_mode = 'stopping'
            self._transition_until = now + self.stop_seconds
            self._advance_ground_pick(now)
            return self._zero_velocity(now)
        if self._policy_mode == 'stopping' and now >= self._transition_until:
            self._policy_mode = 'sit'
            self._transition_until = None
        if self._policy_mode in {'stopping', 'sit'}:
            return self._zero_velocity(now)
        if self._policy_mode == 'stand':
            desired = axes_to_twist(sample.axes, max_vx=self.limits[0],
                                    max_vy=self.limits[1], max_wz=self.limits[2])
            if now < self._transition_until or not any(desired):
                return self._zero_velocity(now)
            self._policy_mode = 'walking'
            self._transition_until = None
            self._zero_velocity(now)
        return self._slew_velocity(sample, now)


class RouladeGamepadCommand(ActionGamepadCommand):
    """Optional LB one-shot; the runtime decides when measured recovery is safe.

    No phase is sent to this episodic policy. It receives zero twist until
    ``finish_roulade`` is called by the runtime's IMU progress monitor. Unlike
    the established ground-pick path, stale/disconnected input aborts an
    accepted roll immediately, including its initial gait-stop interval.
    """

    def __init__(self, reader, *, roulade_enabled=False, roulade_timeout_s=5.,
                 ground_pick_enabled=False, **kwargs):
        super().__init__(reader, ground_pick_enabled=ground_pick_enabled, **kwargs)
        if not isinstance(roulade_enabled, bool):
            raise GamepadError('滚翻启用开关必须为布尔值')
        if not math.isfinite(roulade_timeout_s) or not 0 < roulade_timeout_s <= 10:
            raise GamepadError('滚翻超时必须在(0,10]秒内')
        self.roulade_enabled = roulade_enabled
        self.roulade_timeout_s = roulade_timeout_s
        self._roulade_pending = False
        self._roulade_started = None
        self._lb_baseline_pending = True
        self._last_lb_count = 0
        self._lb_was_down = False
        self._baseline_lb(reader.latest())

    def _baseline_lb(self, sample):
        self._last_lb_count = sample.button_press_counts.get(ROULADE_BUTTON, 0)
        self._lb_was_down = ROULADE_BUTTON in sample.buttons

    def _fresh_lb_presses(self, sample):
        if self._lb_baseline_pending:
            self._baseline_lb(sample)
            self._lb_baseline_pending = False
            return 0
        down = ROULADE_BUTTON in sample.buttons
        count = sample.button_press_counts.get(ROULADE_BUTTON)
        presses = (int(down and not self._lb_was_down) if count is None
                   else max(0, count - self._last_lb_count))
        self._last_lb_count = self._last_lb_count + presses if count is None else count
        self._lb_was_down = down
        return presses

    def finish_roulade(self):
        if self._policy_mode != 'roulade':
            raise GamepadError('只能结束正在执行的滚翻')
        now = self.clock()
        self._roulade_started = None
        self._policy_mode = 'stand'
        self._transition_until = now + self.rise_seconds
        self._zero_velocity(now)

    def __call__(self):
        now = self.clock()
        sample = self.reader.latest()
        if sample.stop_requested or GamepadReader.B_BUTTON in sample.buttons:
            raise GamepadStop('Xbox B键：结束策略并卸力')
        active = self._roulade_pending or self._policy_mode == 'roulade'
        valid = sample.connected and 0 <= now - sample.timestamp <= .3
        if not valid:
            self._lb_baseline_pending = True
            if active:
                raise GamepadStop('滚翻期间手柄断开或过期：结束策略并卸力')
            return super().__call__()
        lb_presses = self._fresh_lb_presses(sample)
        if active:
            # Consume every other action edge, including held/repeated LB.
            self._fresh_y_presses(sample)
            self._fresh_rb_presses(sample)
            if self._roulade_pending and now >= self._transition_until:
                self._roulade_pending = False
                self._roulade_started = now
                self._policy_mode = 'roulade'
                self._transition_until = None
            if (self._roulade_started is not None and
                    now - self._roulade_started >= self.roulade_timeout_s):
                raise GamepadStop('滚翻超时且未确认稳定站立：结束策略并卸力')
            return self._zero_velocity(now)
        # Y and RB retain priority if several buttons arrive together.
        velocity = super().__call__()
        eligible = (self._policy_mode == 'walking' or
                    (self._policy_mode == 'stand' and now >= self._transition_until))
        if self.roulade_enabled and lb_presses and eligible:
            self._roulade_pending = True
            self._policy_mode = 'stopping'
            self._transition_until = now + self.stop_seconds
            return self._zero_velocity(now)
        return velocity


class SuiteGamepadCommand(RouladeGamepadCommand):
    """Opt-in D-pad model selection over the unchanged Xbox action controls.

    Up alternates original/updated walking after stopping, Down requests the
    same one-shot roll as LB, Left selects friction sitstand, Right selects
    CoM sitstand. While seated, either side first stands with the current
    policy; a later press after the rise guard can select a different one.

    Y/RB/LB edges win over simultaneous D-pad edges; multiple directions are
    ignored. Edges during an action, its gait-stop interval, or a rise guard
    are consumed, never queued. Only the runtime uses these variant labels;
    this class never clears action history or runs a neural network.
    """

    def __init__(self, reader, **kwargs):
        super().__init__(reader, **kwargs)
        self.walking_variant = 'original'
        self.sitstand_variant = 'default'
        self._walking_switch_pending = False
        self._suite_baseline_pending = True
        self._suite_counts = {}
        self._suite_down = set()
        self._baseline_suite(reader.latest())

    def _baseline_suite(self, sample):
        watched = DPAD_BUTTONS | {SITSTAND_BUTTON, GROUND_PICK_BUTTON, ROULADE_BUTTON}
        self._suite_counts = {key: sample.button_press_counts.get(key, 0) for key in watched}
        self._suite_down = set(sample.buttons) & watched

    def _fresh_suite_edges(self, sample):
        if self._suite_baseline_pending:
            self._baseline_suite(sample)
            self._suite_baseline_pending = False
            return set()
        pressed = set()
        for key, previous in self._suite_counts.items():
            count = sample.button_press_counts.get(key)
            down = key in sample.buttons
            if (count is not None and count > previous) or (count is None and down and key not in self._suite_down):
                pressed.add(key)
            self._suite_counts[key] = count if count is not None else previous + int(key in pressed)
        self._suite_down = set(sample.buttons) & set(self._suite_counts)
        return pressed

    def _suite_busy(self, now):
        return (self._walking_switch_pending or self._roulade_pending or self._ground_pick_pending
                or self._policy_mode in {'stopping', 'roulade', 'ground_pick'}
                or (self._policy_mode == 'stand' and now < self._transition_until))

    def __call__(self):
        now = self.clock()
        sample = self.reader.latest()
        if sample.stop_requested or GamepadReader.B_BUTTON in sample.buttons:
            raise GamepadStop('Xbox B键：结束策略并卸力')
        valid = sample.connected and 0 <= now - sample.timestamp <= .3
        if not valid:
            self._suite_baseline_pending = True
            if self._walking_switch_pending:
                raise GamepadStop('切换行走模型期间手柄断开或过期：结束策略并卸力')
            return super().__call__()
        edges = self._fresh_suite_edges(sample)
        was_busy = self._suite_busy(now)
        if self._walking_switch_pending:
            # This is our gait-stop interval, not the parent's sit/roll stop.
            # Consume legacy edges too so no action is injected at completion.
            self._fresh_y_presses(sample)
            self._fresh_rb_presses(sample)
            self._fresh_lb_presses(sample)
            if now >= self._transition_until:
                self.walking_variant = 'updated' if self.walking_variant == 'original' else 'original'
                self._walking_switch_pending = False
                self._policy_mode = 'stand'
                self._transition_until = now + self.rise_seconds
            return self._zero_velocity(now)

        # Let the existing B/roll timeout, action completion and Y/RB/LB paths
        # run first. Independent edge bookkeeping does not alter their counts.
        velocity = super().__call__()
        directions = edges & DPAD_BUTTONS
        if (was_busy or self._suite_busy(now) or edges - DPAD_BUTTONS
                or len(directions) != 1 or len(set(sample.buttons) & DPAD_BUTTONS) > 1):
            return velocity
        direction = next(iter(directions))
        eligible = (self._policy_mode == 'walking' or
                    (self._policy_mode == 'stand' and now >= self._transition_until))
        if direction == DPAD_UP_BUTTON and eligible:
            self._walking_switch_pending = True
            self._policy_mode = 'stopping'
            self._transition_until = now + self.stop_seconds
            return self._zero_velocity(now)
        if direction == DPAD_DOWN_BUTTON and eligible and self.roulade_enabled:
            # Set the same state as a fresh LB request, without synthesizing
            # a button or modifying LB counts and held-button baselines.
            self._roulade_pending = True
            self._policy_mode = 'stopping'
            self._transition_until = now + self.stop_seconds
            return self._zero_velocity(now)
        if direction in (DPAD_LEFT_BUTTON, DPAD_RIGHT_BUTTON):
            if self._policy_mode == 'sit':
                self._toggle_posture(now)
                return self._zero_velocity(now)
            if eligible:
                self.sitstand_variant = 'friction' if direction == DPAD_LEFT_BUTTON else 'com'
                self._policy_mode = 'stopping'
                self._transition_until = now + self.stop_seconds
                return self._zero_velocity(now)
        return velocity
