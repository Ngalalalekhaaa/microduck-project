"""Line or single-key velocity commands for an interactive SSH terminal.

This object is the sole stdin reader while attached to the controller. Optional
single-key input starts only when policy control starts and preserves signals.
"""
from __future__ import annotations

import math
import os
import select
import sys
import termios
import time

from .gamepad import COMMAND_LIMITS, MAX_BACKWARD_VX


class TerminalCommand:
    owns_stdin = True
    _ZERO = (0., 0., 0.)
    _ACCELERATION = (.3, .3, 1.)
    _MAX_LINE_BYTES = 64
    _MAX_POLL_BYTES = 4096
    _MAX_POLL_READS = 32
    _MAX_POLL_LINES = 32
    _INVALID_INPUT_HINT = '仅接受0..6、j/k/l或q/stop/exit，每条输入后回车。'

    def __init__(self, stream=None, max_vx=.3, max_vy=.3, max_wz=1.,
                 clock=time.monotonic, *, single_key=False):
        if not isinstance(single_key, bool):
            raise ValueError('single_key 必须为布尔值')
        limits = tuple(float(value) for value in (max_vx, max_vy, max_wz))
        if any(not math.isfinite(value) or not 0 < value <= bound
               for value, bound in zip(limits, COMMAND_LIMITS)):
            raise ValueError('终端速度上限须满足0<vx<=0.4、0<vy<=0.3、0<wz<=1.0')
        self.stream = sys.stdin if stream is None else stream
        try:
            if not self.stream.isatty():
                raise ValueError('数字控制需要交互终端；请使用交互SSH或ssh -t')
            self.fd = self.stream.fileno()
            if not isinstance(self.fd, int) or self.fd < 0:
                raise ValueError('终端文件描述符无效')
        except (AttributeError, OSError, TypeError) as exc:
            raise ValueError('无法读取交互终端文件描述符') from exc
        vx, vy, wz = limits
        self._commands = {
            '0': self._ZERO,
            '1': (min(vx, .30), 0., 0.), '2': (-min(vx, MAX_BACKWARD_VX), 0., 0.),
            'j': (min(vx, .15), 0., 0.),
            'k': (min(vx, .20), 0., 0.),
            'l': (min(vx, .25), 0., 0.),
            '3': (0., vy, 0.), '4': (0., -vy, 0.),
            '5': (0., 0., wz), '6': (0., 0., -wz),
        }
        self.clock = clock
        self._last_time = clock()
        self._desired = self._previous = self._ZERO
        self._buffer = b''
        self._active = False
        self._first_sample = True
        self._stop_reason = None
        self.single_key = single_key
        self._saved_termios = None
        self._restore_error_reported = False
        self._escape_state = None
        self._escape_bytes = bytearray()
        self._paste = False
        self._paste_tail = bytearray()

    @property
    def ready_message(self):
        mode = '直接按键，无需回车：' if self.single_key else '每条命令输入后回车：'
        return ('数字控制已就绪：模型从零速度开始。' + mode +
                '前进1/j/k/l=' + '/'.join(f'{self._commands[key][0]:.2f}' for key in ('1', 'j', 'k', 'l')) +
                'm/s；2后退/3左移/4右移/5左转/6右转；0停步保持模型；q退出卸力。')

    def _zero(self):
        self._desired = self._previous = self._ZERO
        self._last_time = self.clock()

    def _stop(self, reason):
        if self._stop_reason is None:
            self._stop_reason = reason
        self._active = False
        self._zero()
        error = self._restore_terminal()
        if error and not self._restore_error_reported:
            self._stop_reason += f'；终端设置恢复失败：{error}'
            self._restore_error_reported = True
        return self._stop_reason

    def _restore_terminal(self):
        if self._saved_termios is None:
            return None
        try:
            # Keys typed behind q/error belong to this controller, not the
            # shell that regains canonical input after this process exits.
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except (OSError, termios.error, ValueError):
            # A failed flush must not prevent restoring the terminal mode.
            pass
        try:
            termios.tcsetattr(self.fd, termios.TCSANOW, self._saved_termios)
        except (OSError, termios.error, ValueError) as exc:
            # Keep the saved settings so a later close can retry restoration.
            return str(exc)
        self._saved_termios = None
        return None

    def _single_keys(self, chunk, command_count):
        """Consume bounded bytes without treating terminal escape digits as keys."""
        invalid = False
        for value in chunk:
            if value == 4:  # VEOF is ordinary input after ICANON is disabled.
                self._stop('终端输入已关闭（Ctrl+D）；停止策略并卸力')
                break
            if self._paste:
                self._paste_tail.append(value)
                del self._paste_tail[:-6]
                if self._paste_tail.endswith((b'\x1b[201~', b'\x9b201~')):
                    self._paste = False
                    self._paste_tail.clear()
                continue
            state = self._escape_state
            if state is not None:
                if state in ('osc', 'string'):
                    if value == 7 or (self._escape_bytes and self._escape_bytes[-1] == 27 and value == 92):
                        self._escape_state = None
                        self._escape_bytes.clear()
                    else:
                        self._escape_bytes[:] = bytes((value,))
                    continue
                if value == 27:
                    self._escape_state = 'escape'
                    self._escape_bytes.clear()
                    continue
                self._escape_bytes.append(value)
                if len(self._escape_bytes) > self._MAX_LINE_BYTES:
                    self._stop('终端转义序列过长；停止策略并卸力')
                    break
                if state == 'escape':
                    if value == ord('['):
                        self._escape_state = 'csi'
                        self._escape_bytes.clear()
                    elif value == ord('O'):
                        self._escape_state = 'ss3'
                        self._escape_bytes.clear()
                    elif value in (ord(']'), ord('P'), ord('X'), ord('^'), ord('_')):
                        self._escape_state = 'osc' if value == ord(']') else 'string'
                        self._escape_bytes.clear()
                    elif 0x20 <= value <= 0x2f:
                        self._escape_state = 'intermediate'
                    else:
                        self._escape_state = None  # Includes Alt+digit/letter.
                        self._escape_bytes.clear()
                elif state in ('csi', 'ss3') and 0x40 <= value <= 0x7e:
                    if state == 'csi' and self._escape_bytes == b'200~':
                        self._paste = True
                        self._paste_tail.clear()
                    self._escape_state = None
                    self._escape_bytes.clear()
                elif state == 'intermediate' and 0x30 <= value <= 0x7e:
                    self._escape_state = None
                    self._escape_bytes.clear()
                continue
            if value in (27, 0x9b, 0x8f):
                self._escape_state = {27: 'escape', 0x9b: 'csi', 0x8f: 'ss3'}[value]
                self._escape_bytes.clear()
                continue
            if chr(value).isspace():
                continue
            command_count += 1
            if command_count > self._MAX_POLL_LINES:
                self._stop('终端输入过多；停止策略并卸力')
                break
            key = chr(value).lower()
            if key == 'q':
                self._stop('终端请求退出；停止策略并卸力')
                break
            if not self._handle_command(key):
                self._zero()
                invalid = True
        return command_count, invalid

    def _handle_command(self, line):
        """Handle one active command; subclasses can add action selections."""
        if line not in self._commands:
            return False
        if line == '0':
            self._zero()
        else:
            self._desired = self._commands[line]
        return True

    def check_stop(self):
        """Poll bounded input without blocking; return a sticky stop reason."""
        if self._stop_reason is not None:
            return self._stop_reason
        total = 0
        lines = 0
        invalid = False
        try:
            for _ in range(self._MAX_POLL_READS):
                if not select.select([self.fd], [], [], 0)[0]:
                    break
                chunk = os.read(self.fd, self._MAX_POLL_BYTES - total)
                if not chunk:
                    return self._stop('终端输入已关闭（EOF）；停止策略并卸力')
                total += len(chunk)
                if self.single_key and self._active:
                    lines, bad = self._single_keys(chunk, lines)
                    invalid |= bad
                    if self._stop_reason is not None:
                        return self._stop_reason
                else:
                    self._buffer += chunk
                    while b'\n' in self._buffer:
                        lines += 1
                        if lines > self._MAX_POLL_LINES:
                            return self._stop('终端输入过多；停止策略并卸力')
                        raw, self._buffer = self._buffer.split(b'\n', 1)
                        if len(raw) > self._MAX_LINE_BYTES:
                            return self._stop('终端输入行过长；停止策略并卸力')
                        line = raw.decode('utf-8', errors='replace').strip().lower()
                        if line in ('q', 'stop', 'exit'):
                            return self._stop('终端请求退出；停止策略并卸力')
                        # Input during HOME must not queue motion for startup.
                        if not line or not self._active:
                            continue
                        if not self._handle_command(line):
                            self._zero()
                            invalid = True
                    if len(self._buffer) > self._MAX_LINE_BYTES:
                        return self._stop('终端输入行过长；停止策略并卸力')
                if total >= self._MAX_POLL_BYTES:
                    if select.select([self.fd], [], [], 0)[0]:
                        return self._stop('终端输入过多；停止策略并卸力')
                    break
            else:
                if select.select([self.fd], [], [], 0)[0]:
                    return self._stop('终端输入过多；停止策略并卸力')
        except BlockingIOError:
            pass
        except (OSError, ValueError) as exc:
            return self._stop(f'终端读取失败；停止策略并卸力：{exc}')
        except BaseException:
            self._stop('终端输入处理已中断；停止策略并卸力')
            raise
        if invalid:
            hint = ('仅接受当前显示的单键命令，无需回车。' if self.single_key
                    else self._INVALID_INPUT_HINT)
            try:
                print('无效输入已按0处理；' + hint, flush=True)
            except BaseException:
                self._stop('终端输出已中断；停止策略并卸力')
                raise
        return None

    def start_policy(self):
        """Discard old complete/partial input and enable a fresh zero start."""
        reason = self.check_stop()
        if reason is not None:
            return reason
        try:
            # check_stop first preserves any pending complete quit/EOF request.
            # tcflush also clears characters not yet terminated with Enter.
            termios.tcflush(self.fd, termios.TCIFLUSH)
            if self.single_key and self._saved_termios is None:
                original = termios.tcgetattr(self.fd)
                self._saved_termios = original
                changed = list(original)
                changed[6] = list(original[6])
                changed[3] = (changed[3] & ~(termios.ICANON | termios.ECHO | termios.ECHONL)) | termios.ISIG
                changed[6][termios.VMIN] = 0
                changed[6][termios.VTIME] = 0
                termios.tcsetattr(self.fd, termios.TCSANOW, changed)
        except (OSError, termios.error, ValueError) as exc:
            return self._stop(f'终端初始化失败；停止策略并卸力：{exc}')
        except BaseException:
            self._stop('终端初始化已中断；停止策略并卸力')
            raise
        self._buffer = b''
        self._escape_state = None
        self._escape_bytes.clear()
        self._paste = False
        self._paste_tail.clear()
        self._zero()
        self._first_sample = True
        self._active = True
        return None

    def __call__(self):
        self.check_stop()
        now = self.clock()
        if self._stop_reason is not None or not self._active:
            self._previous = self._ZERO
            self._last_time = now
            return self._ZERO
        if self._first_sample:
            self._first_sample = False
            self._previous = self._ZERO
            self._last_time = now
            return self._ZERO
        elapsed = min(.1, max(0., now - self._last_time))
        self._previous = tuple(
            old + max(-rate * elapsed, min(rate * elapsed, desired - old))
            for old, desired, rate in zip(self._previous, self._desired, self._ACCELERATION))
        self._last_time = now
        return self._previous

    def close(self):
        """Stop this source and restore terminal settings without closing stdin."""
        self._stop('终端控制已关闭')
