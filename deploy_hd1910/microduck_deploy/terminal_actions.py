"""Terminal action keys using the same posture state machine as Xbox controls."""
from __future__ import annotations

import time

from .gamepad import (
    GROUND_PICK_BUTTON, MAX_BACKWARD_VX, PadSample, ROULADE_BUTTON,
    RouladeGamepadCommand, SITSTAND_BUTTON,
)
from .terminal_command import TerminalCommand


class TerminalActionCommand(TerminalCommand):
    """7 toggles sit/stand, 8 stands, 9 picks, r performs one forward roll.

    Commands use line input or optional immediate single-key input. Requested
    actions discard the previous walking direction. Directions and additional
    actions received during a ground/roll action or a rise are not queued.
    """

    _INVALID_INPUT_HINT = '接受0..9、a–i、j/k/l、m/n、r或q/stop/exit，每条输入后回车。'

    def __init__(self, stream=None, max_vx=.3, max_vy=.3, max_wz=1.,
                 clock=time.monotonic, *, sitstand_enabled=True,
                 ground_pick_enabled=True, roulade_enabled=True,
                 roulade_timeout_s=5., single_key=False, walking_models=None,
                 sitstand_models=None, default_sitstand_key=None):
        for value in (sitstand_enabled, ground_pick_enabled, roulade_enabled):
            if not isinstance(value, bool):
                raise ValueError('动作启用开关必须为布尔值')
        if (ground_pick_enabled or roulade_enabled) and not sitstand_enabled:
            raise ValueError('俯身和滚翻需要坐站模型用于返回站立')
        super().__init__(stream, max_vx, max_vy, max_wz, clock, single_key=single_key)
        self._limits = (max_vx, max_vy, max_wz)
        self.sitstand_enabled = sitstand_enabled
        self.ground_pick_enabled = ground_pick_enabled
        self.roulade_enabled = roulade_enabled
        self.roulade_timeout_s = roulade_timeout_s
        self.walking_models = dict(walking_models or {})
        if self.walking_models and set(self.walking_models) != set('abcdef'):
            raise ValueError('行走模型按键必须为 a–f')
        self.sitstand_models = dict(sitstand_models or {})
        if self.sitstand_models:
            if (not sitstand_enabled or set(self.sitstand_models) != set('ghimn')
                    or default_sitstand_key not in self.sitstand_models):
                raise ValueError('坐站模型需要 g/h/i/m/n 和已启用的默认坐站模型')
        elif default_sitstand_key is not None:
            raise ValueError('默认坐站字母没有对应的模型')
        self._default_sitstand_key = default_sitstand_key
        self._reset_actions()

    def _reset_actions(self):
        self._press_counts = {}
        self._request_pending = False
        self._last_feedback_key = None
        self._last_feedback_time = float('-inf')
        self.walking_model_key = 'a' if self.walking_models else None
        self.sitstand_model_key = self._default_sitstand_key
        self._walking_switch_pending = None
        self._ground_walk_recovery_until = None
        self._capture_sample()
        self._actions = RouladeGamepadCommand(
            self, max_vx=self._limits[0], max_vy=self._limits[1],
            max_wz=self._limits[2], clock=self.clock,
            ground_pick_enabled=self.ground_pick_enabled,
            roulade_enabled=self.roulade_enabled,
            roulade_timeout_s=self.roulade_timeout_s)
        # Establish the neutral baseline before fresh terminal commands arrive.
        self._capture_sample()
        self._actions()

    def latest(self):
        """Return a snapshot captured before the state machine samples its clock."""
        return self._pad_sample

    def _capture_sample(self):
        """The shared gamepad reader must never report a future timestamp.

        RouladeGamepadCommand reads its own clock before calling latest().
        Sampling a new timestamp in latest() would make valid keyboard input
        appear to come from the future whenever the real clock advances.
        """
        vx, vy, wz = self._desired
        forward_limit = min(self._limits[0], MAX_BACKWARD_VX) if vx < 0 else self._limits[0]
        self._pad_sample = PadSample(
            timestamp=self.clock(), connected=True,
            axes={'ABS_X': -vy / self._limits[1],
                  'ABS_Y': -vx / forward_limit,
                  'ABS_RX': -wz / self._limits[2]},
            raw_axes={}, buttons=(), button_press_counts=dict(self._press_counts))

    @property
    def policy_mode(self):
        return self._actions.policy_mode

    @property
    def ground_pick_command(self):
        return self._actions.ground_pick_command

    def _zero(self):
        super()._zero()
        if hasattr(self, '_actions'):
            self._actions._zero_velocity(self.clock())

    def _busy(self):
        return self._busy_reason() is not None

    def _busy_reason(self):
        state = self._actions
        if self._ground_walk_recovery_until is not None and self.clock() < self._ground_walk_recovery_until:
            return f'低头抬头结束后正在恢复行走模型，还需{self._ground_walk_recovery_until - self.clock():.1f}秒'
        if self._walking_switch_pending is not None:
            return '切换行走模型前正在停步'
        if self._request_pending:
            return '上一条动作指令正在切换'
        if state._ground_pick_pending:
            return '低头抬头前正在停步'
        if state._roulade_pending:
            return '前滚翻前正在停步'
        if state.policy_mode == 'stopping':
            return '坐下前正在停步'
        if state.policy_mode == 'ground_pick':
            return '正在低头后抬头'
        if state.policy_mode == 'roulade':
            return '正在前滚翻'
        if state.policy_mode == 'stand' and self.clock() < state._transition_until:
            return f'正在站起，还需{state._transition_until - self.clock():.1f}秒'
        return None

    def _feedback(self, key, message):
        """Coalesce repeated notices only; never debounce or discard a fresh key."""
        now = self.clock()
        if key != self._last_feedback_key or now - self._last_feedback_time >= .5:
            print(message, flush=True)
            self._last_feedback_key, self._last_feedback_time = key, now

    def _busy_feedback(self, line, reason):
        self._feedback(('busy', line, self.policy_mode),
                       f'按键{line}未执行：{reason}；本次不排队，请完成后轻按。')

    def _press(self, button, count=1):
        self._zero()
        self._press_counts[button] = self._press_counts.get(button, 0) + count
        self._request_pending = True

    def _handle_command(self, line):
        if line in self.sitstand_models:
            reason = self._busy_reason()
            if reason:
                self._busy_feedback(line, reason)
            elif self.policy_mode == 'sit' and line != self.sitstand_model_key:
                self._feedback(('seated-selection', line),
                               f'按键{line}未执行：请先按7用当前模型站起，再选择坐站模型。')
            else:
                changed = line != self.sitstand_model_key
                self._zero()
                self.sitstand_model_key = line
                settling = changed and self.policy_mode == 'stand'
                if settling:
                    self._actions._transition_until = self.clock() + self._actions.rise_seconds
                detail = '站立适应等待3秒后，' if settling else ''
                self._feedback(('sitstand-selected', line),
                               f'已选择 {line}：{self.sitstand_models[line]}；{detail}'
                               '按7坐下，再按7站起。')
            return True
        if line in self.walking_models:
            reason = self._busy_reason()
            if reason:
                self._busy_feedback(line, reason)
            elif self.policy_mode == 'sit':
                self._feedback(('sitting', line),
                               f'按键{line}未执行：当前为坐姿，请先轻按7站起。')
            elif line == self.walking_model_key and self.policy_mode == 'walking':
                self._zero()
                self._feedback(('selected', line),
                               f'当前已是 {line}：{self.walking_models[line]}；速度已归零。')
            else:
                self._zero()
                deadline = self.clock() + self._actions.stop_seconds
                self._walking_switch_pending = (line, deadline)
                self._actions._policy_mode = 'stopping'
                self._actions._transition_until = deadline
                self._feedback(('switching', line),
                               f'已接收{line}：先停步0.6秒，再切换 {self.walking_models[line]}；'
                               '切换后速度为零，请重新按方向键。')
            return True
        if line == '0':
            self._zero()
            return True
        if line in self._commands:
            reason = self._busy_reason()
            if reason:
                self._busy_feedback(line, reason)
            elif self.policy_mode in ('walking', 'stand'):
                self._desired = self._commands[line]
            else:
                self._feedback(('sitting', line), f'按键{line}未执行：当前为坐姿，请先轻按7站起。')
            return True
        if line not in ('7', '8', '9', 'r'):
            return False
        enabled = (self.sitstand_enabled if line in ('7', '8') else
                   self.ground_pick_enabled if line == '9' else self.roulade_enabled)
        if not enabled:
            self._zero()
            self._feedback(('disabled', line), f'按键{line}对应的模型未启用，未执行动作。')
            return True
        reason = self._busy_reason()
        if reason:
            self._busy_feedback(line, reason)
            return True
        mode = self.policy_mode
        if line == '7' and mode in ('walking', 'stand'):
            self._press(SITSTAND_BUTTON)
            detail = '先停步0.6秒，再切换坐下模型' if mode == 'walking' else '切换坐下模型'
            self._feedback(('accepted', line, 'sit'), f'已接收7：{detail}；再轻按7站起。')
        elif line == '7' and mode == 'sit':
            self._press(SITSTAND_BUTTON)
            self._feedback(('accepted', line, 'stand'), '已接收7：切换站立模型；起立等待3秒。')
        elif line == '8' and mode in ('walking', 'sit'):
            # Both edges are consumed together: walking -> stopping -> stand.
            # No sitting-policy frame is executed for an explicit stand request.
            self._press(SITSTAND_BUTTON, 2 if mode == 'walking' else 1)
            self._feedback(('accepted', line, 'stand'), '已接收8：切换站立模型；起立等待3秒。')
        elif line == '8' and mode == 'stand':
            self._feedback(('standing', line), '当前已处于站立模式；轻按7切换坐下。')
        elif line in ('9', 'r') and mode in ('walking', 'stand'):
            self._press(GROUND_PICK_BUTTON if line == '9' else ROULADE_BUTTON)
            action = '低头后抬头（4秒）' if line == '9' else '单次前滚翻'
            self._feedback(('accepted', line, action), f'已接收{line}：先停步0.6秒，再执行{action}。')
        elif mode == 'sit':
            self._feedback(('sitting', line), f'按键{line}未执行：当前为坐姿，请先轻按7站起。')
        return True

    def start_policy(self):
        reason = super().start_policy()
        if reason is not None:
            return reason
        try:
            self._reset_actions()
            if self.walking_models:
                print('行走模型（直接按字母，无需回车）：' + '；'.join(
                    f'{key}={label}' for key, label in self.walking_models.items()) +
                    '。默认a；切换先停步0.6秒，数字动作键保持原用法。', flush=True)
            if self.sitstand_models:
                print('坐站模型（直接按字母选择，无需回车）：' + '；'.join(
                    f'{key}={label}' for key, label in self.sitstand_models.items()) +
                    f'。当前{self.sitstand_model_key}；坐下和站起共用7，字母本身不触发坐下/站起。',
                    flush=True)
            keys = []
            if self.sitstand_enabled:
                keys.append('轻按7切换坐下／站起（8仍可直接站起）')
            if self.ground_pick_enabled:
                keys.append('9低头后抬头（4秒）')
            if self.roulade_enabled:
                keys.append('r单次前滚翻')
            if keys:
                mode = '直接按键，无需回车' if self.single_key else '输入后回车'
                print(f'动作按键（{mode}）：' + '；'.join(keys) + '。'
                      '低头抬头结束回当前行走模型，零速度恢复3秒后重新按方向；'
                      '坐站/滚翻结束保持站立，起立等待3秒；'
                      '0仅停步，q退出卸力。', flush=True)
        except BaseException:
            self._stop('终端动作初始化已中断；停止策略并卸力')
            raise
        return None

    def __call__(self):
        self.check_stop()
        now = self.clock()
        if self._stop_reason is not None or not self._active:
            self._zero()
            return self._ZERO
        if self._ground_walk_recovery_until is not None:
            if now < self._ground_walk_recovery_until:
                self._zero()
                return self._ZERO
            self._ground_walk_recovery_until = None
            self._feedback('ground-walking-ready',
                           '低头抬头结束：行走模型已零速度恢复3秒，请重新按方向键；坐站仍用7。')
        if self._walking_switch_pending is not None:
            key, deadline = self._walking_switch_pending
            if now >= deadline:
                self.walking_model_key = key
                self._walking_switch_pending = None
                self._actions._policy_mode = 'walking'
                self._actions._transition_until = None
                self._feedback(('switched', key),
                               f'已切换 {key}：{self.walking_models[key]}；零速度保持，重新按1–6行走。')
            self._zero()
            return self._ZERO
        if self._first_sample:
            self._first_sample = False
            self._previous = self._ZERO
            self._last_time = now
            return self._ZERO
        previous_mode = self.policy_mode
        try:
            self._capture_sample()
            self._actions()
        except BaseException:
            self._stop('动作控制已中断；停止策略并卸力')
            raise
        self._request_pending = False
        if previous_mode == 'ground_pick' and self.policy_mode == 'stand':
            # The shared Xbox state machine returns to its sitstand session.
            # Keyboard 9 is a walking detour: return to the selected walking
            # session at zero, without invoking a possibly experimental sitstand
            # model. The runtime retains the preceding raw ground action.
            self._actions._policy_mode = 'walking'
            self._actions._transition_until = None
            self._ground_walk_recovery_until = now + self._actions.rise_seconds
            self._zero()
            selected = self.walking_model_key or '当前'
            self._feedback('ground-walking-recovery',
                           f'低头抬头完成：已回行走模型{selected}，零速度恢复3秒；期间方向键不排队。')
            return self._ZERO
        if self.policy_mode != 'walking':
            self._previous = self._ZERO
            self._last_time = now
            return self._ZERO
        if previous_mode != 'walking':
            self._previous = self._ZERO
            self._last_time = now
            return self._ZERO
        elapsed = min(.1, max(0., now - self._last_time))
        self._previous = tuple(
            old + max(-rate * elapsed, min(rate * elapsed, desired - old))
            for old, desired, rate in zip(self._previous, self._desired, self._ACCELERATION))
        self._last_time = now
        return self._previous

    def finish_roulade(self):
        self._actions.finish_roulade()
        self._zero()
