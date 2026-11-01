"""Supported HOME recovery, then an explicitly started real-robot policy trial."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WEB = Path.home() / 'servo_web_hd1910'
sys.path.insert(0, str(ROOT))
from microduck_deploy.gamepad import GamepadError, GamepadStop, HOME_BUTTON, WALK_BUTTON


class RehomeRequested(RuntimeError):
    pass


def checked_gamepad_sample(reader):
    sample = reader.latest()
    if sample.stop_requested:
        raise GamepadStop('Xbox B键：停止流程并卸力')
    if not sample.connected or not 0 <= time.monotonic()-sample.timestamp <= .3:
        raise GamepadError('手柄连接已断开或输入过期，停止按键控制流程')
    return sample


def wait_for_gamepad_action(reader, buttons, *, timeout=None, tick=None):
    """Require a new press in this stage; earlier/held presses do not advance it."""
    first = checked_gamepad_sample(reader)
    seen = {button: first.button_press_counts.get(button, 0) for button in buttons}
    deadline = None if timeout is None else time.monotonic()+timeout
    while deadline is None or time.monotonic() < deadline:
        sample = checked_gamepad_sample(reader)
        if tick is not None:
            tick()
        for button in buttons:
            count = sample.button_press_counts.get(button, 0)
            if count > seen[button]:
                seen[button] = count
                if all(abs(value) <= .05 for value in sample.axes.values()):
                    return button
                print('请先松开摇杆，再重新按一次启动键。', flush=True)
        time.sleep(.02)
    return None


def wait_for_gamepad_button(reader, button, *, timeout=None, tick=None):
    return wait_for_gamepad_action(reader, (button,), timeout=timeout, tick=tick) == button


def wait_for_gamepad_walk(reader, *, tick=None):
    print('\nHOME回位完成并保持上力。请双脚落地、稳住机身、松开摇杆。\n'
          '短按菜单键（≡）：静止校偏约3秒，再对齐HOME约1秒，随后开始策略。\n'
          'A重新回HOME；B卸力并复位到等待A。', flush=True)
    action = wait_for_gamepad_action(reader, (HOME_BUTTON, WALK_BUTTON), tick=tick)
    if action == HOME_BUTTON:
        raise RehomeRequested()
    return action == WALK_BUTTON


def wait_for_b_release(reader):
    while True:
        sample = reader.latest()
        if not sample.connected or not 0 <= time.monotonic()-sample.timestamp <= .3:
            raise GamepadError('手柄连接已断开，无法继续等待新的A键')
        if 305 not in sample.buttons:
            return
        time.sleep(.02)


def verify_home_hold(cfg, bus_factory):
    """Enable the next stage only after measured HOME/torque feedback is ready."""
    import numpy as np
    from microduck_deploy.config import calibration_arrays
    from microduck_deploy.policy import HOME, ticks_to_q
    ids = [joint['id'] for joint in cfg['joints']]
    options = dict(cfg['serial'])
    options['timeout'] = max(.05, options['timeout'])
    with bus_factory(**options) as bus:
        configurations = [bus.read_configuration(sid) for sid in ids]
        feedback = bus.read_feedback_many(ids, sync=True)
    if any(row['torque'] != 1 for row in configurations):
        return False
    if any(value.status or not 0 <= time.monotonic()-value.received_at <= .1 for value in feedback.values()):
        return False
    q = ticks_to_q([feedback[sid].position_ticks for sid in ids], *calibration_arrays(cfg))
    error = float(np.max(np.abs(q-HOME)))
    tolerance = cfg['control']['home_tolerance_rad']
    print(f'HOME反馈最大偏差{math.degrees(error):.2f}°，到位容差{math.degrees(tolerance):.2f}°。', flush=True)
    return error <= tolerance


def wait_for_child(child, reader=None, *, tick=None):
    """Monitor B while HOME, IMU calibration or the policy child is running."""
    if reader is None:
        return child.wait()
    while child.poll() is None:
        checked_gamepad_sample(reader)
        if tick is not None:
            tick()
        time.sleep(.02)
    return child.returncode


def run_sequence(run, ready, python, *, seconds, vx, soft_gains=True, ramp_seconds=0,
                 max_target_jump_deg=None, log_only_target_angles=False, gamepad_args=(),
                 zero_after_seconds=None, keyboard_args=(), config_path=None):
    config_args = ['--config', str(config_path)] if config_path is not None else []
    home = [python, str(ROOT/'scripts/supported_home.py'), '--target', 'home',
            '--log-only-feedback', '--move', *config_args]
    if soft_gains:
        home.append('--soft-gains')
    code = run(home)
    if code:
        return code
    # HOME remains held during the physical transition from suspended to grounded.
    if not ready():
        return 130
    ramp = ['--ramp-seconds', str(ramp_seconds)] if ramp_seconds else []
    jump = ['--max-target-jump-deg', str(max_target_jump_deg)] if max_target_jump_deg is not None else []
    angle_mode = ['--log-only-target-angles'] if log_only_target_angles else []
    zero_after = ['--zero-after-seconds', str(zero_after_seconds)] if zero_after_seconds is not None else []
    return run([python, str(ROOT/'scripts/policy_trial.py'), '--calibrate-imu',
                '--seconds', str(seconds), '--vx', str(vx), '--move', *ramp, *jump, *angle_mode,
                *gamepad_args, *zero_after, *keyboard_args, *config_args])


def wait_for_feet():
    print('\nHOME回位完成，保持上力；精确到位情况见上面的 errors_ticks。\n'
          '请将双脚放到地面，扶稳躯干并保持静止。\n'
          '准备好后输入 walk 或 WALK 并回车：静止校偏约3秒，再对齐HOME约1秒，随后开始策略。\n'
          '60秒未确认、输入其他内容或Ctrl+C：停止并卸力。', flush=True)
    readable, _, _ = select.select([sys.stdin], [], [], 60)
    return bool(readable) and sys.stdin.readline().strip().upper() == 'WALK'


def wait_for_support(timeout=60, reader=None):
    """Give the operator time to support the existing HOME hold before OFF."""
    action = '按B' if reader is not None else '输入 OFF 回车'
    print('\aIMU启动失败，策略未运行。暂时保留原有HOME保持。\n'
          f'请立即托住躯干和头部，{action}后卸力；最长等待{timeout}秒。\n'
          'Ctrl+C立即卸力。不要离开机器人。', flush=True)
    deadline = time.monotonic() + timeout
    stdin_open = True
    while time.monotonic() < deadline:
        if reader is not None:
            checked_gamepad_sample(reader)
            time.sleep(.02)
            continue
        remaining = deadline-time.monotonic()
        print(f'请托住机器人：最多 {remaining:.0f} 秒后卸力。', flush=True)
        readable, _, _ = select.select([sys.stdin] if stdin_open else [], [], [], min(5, remaining))
        if readable:
            answer = sys.stdin.readline()
            if answer.strip().upper() == 'OFF':
                return
            if answer == '':
                stdin_open = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'robot.json',
                   help='Use the same explicit calibration for HOME and walking')
    p.add_argument('--move', action='store_true')
    p.add_argument('--keyboard', action='store_true', help='数字按键即时控制，无需回车；0停步保持模型，q退出卸力')
    p.add_argument('--seconds', type=float, default=30, help='Runtime in seconds; 0 runs until stopped')
    p.add_argument('--zero-after-seconds', type=float, default=None,
                   help='Set velocity to zero after N policy seconds and keep running; requires --seconds 0, no gamepad')
    p.add_argument('--vx', type=float, default=.3)
    p.add_argument('--ramp-seconds', type=float, default=0)
    p.add_argument('--max-target-jump-deg', type=float, default=None)
    p.add_argument('--log-only-target-angles', action='store_true',
                   help='Record target bounds and jumps without blocking or clipping policy outputs')
    from microduck_deploy.gamepad import add_gamepad_args, validate_gamepad_args, gamepad_options
    add_gamepad_args(p)
    p.add_argument('--gamepad-buttons', action='store_true',
                   help='Wait for A to start HOME and Menu to start calibration/policy; B stops all stages')
    args = p.parse_args()
    validate_gamepad_args(p, args)
    if args.gamepad_buttons and not args.gamepad:
        p.error('--gamepad-buttons requires --gamepad')
    if args.keyboard and (args.gamepad or args.seconds != 0 or args.vx != 0
                          or args.zero_after_seconds is not None or args.ramp_seconds != 0):
        p.error('--keyboard requires --seconds 0 --vx 0, without gamepad, timed-zero or ramp options')
    if not math.isfinite(args.seconds) or args.seconds < 0:
        p.error('--seconds must be finite and nonnegative; 0 runs until stopped')
    if not math.isfinite(args.vx) or not 0 <= args.vx <= .3:
        p.error('--vx must be in [0,0.3]')
    if (not math.isfinite(args.ramp_seconds) or args.ramp_seconds < 0
            or (args.seconds > 0 and args.ramp_seconds > args.seconds)):
        p.error('--ramp-seconds must be in [0,seconds]')
    if args.zero_after_seconds is not None:
        if not math.isfinite(args.zero_after_seconds) or args.zero_after_seconds <= 0:
            p.error('--zero-after-seconds must be finite and positive')
        if args.seconds != 0 or args.gamepad:
            p.error('--zero-after-seconds requires --seconds 0 and no --gamepad')
        if args.ramp_seconds > args.zero_after_seconds:
            p.error('--ramp-seconds must not exceed --zero-after-seconds')
    if args.max_target_jump_deg is not None and (
            not math.isfinite(args.max_target_jump_deg) or not 0 < args.max_target_jump_deg <= 45):
        p.error('--max-target-jump-deg must be in (0,45]')
    if not args.move:
        start = '等待A → ' if args.gamepad_buttons else ''
        ready = '菜单键（≡）' if args.gamepad_buttons else '输入WALK'
        duration = '策略持续运行直到停止' if args.seconds == 0 else f'{args.seconds:g}秒策略'
        if args.zero_after_seconds is not None:
            duration = f'策略运行{args.zero_after_seconds:g}秒后速度指令归零，继续运行模型直到手动停止或故障停机'
        if args.keyboard:
            duration = '零速度启动模型 → 数字1–6+回车移动，0停步保持模型，q退出卸力'
        print(f'预览：{start}HOME最少6秒 → {ready} → IMU静止校偏约3秒 → 对齐HOME约1秒 → {duration}，vx={args.vx:g}m/s。'
              '添加 --move 才执行。未访问硬件。')
        return 0
    if args.keyboard:
        from microduck_deploy.terminal_command import TerminalCommand
        # Check the actual SSH terminal before any HOME movement. The policy
        # child becomes the sole reader only after the WALK stage.
        terminal = TerminalCommand(max_vx=args.max_vx,max_vy=args.max_vy,max_wz=args.max_wz)
        terminal.close()
    if args.gamepad:
        from microduck_deploy.gamepad import GamepadReader, GamepadCommand
        pad = GamepadReader(args.gamepad_device)
        try:
            command = GamepadCommand(pad, max_vx=args.max_vx, max_vy=args.max_vy, max_wz=args.max_wz)
            if not args.gamepad_buttons:
                command()  # Check B/disconnect before moving any joint.
                if any(abs(value) > .05 for value in pad.latest().axes.values()):
                    raise ValueError('请松开两只摇杆后重新运行；当前摇杆不在中心')
            print('Xbox已就绪：'+str(pad.metadata), flush=True)
            print('校偏期间松开摇杆；左摇杆控制行走，右摇杆控制转向；B卸力。', flush=True)
        finally:
            pad.close()
    from microduck_deploy.config import load_config
    from microduck_deploy.jaw import jaw_config, jaw_ids, JawButtonToggle, JawController
    from microduck_deploy.servo import ServoBus
    from supported_home import PositionOnlyBus
    cfg = load_config(args.config)
    ids = [j['id'] for j in cfg['joints']]
    stop_ids = ids + jaw_ids(cfg)
    jaw_settings = jaw_config(cfg) if args.gamepad_buttons else None
    (ROOT/'logs').mkdir(exist_ok=True)
    lock = open(ROOT/'logs/supported_sequence.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    options = dict(cfg['serial'])
    options['timeout'] = max(.05, options['timeout'])
    stopped_web = False
    exclusive = False
    child = None
    pad = None
    result = 1
    original_signals = {}
    jaw_toggle = None
    jaw_bus = None
    jaw = None
    jaw_pipe = None

    def close_idle_jaw():
        nonlocal jaw_bus, jaw
        if jaw_bus is not None:
            # The next stage takes over its existing target and torque.
            jaw_bus.close()
        jaw_bus, jaw = None, None

    def update_jaw():
        nonlocal jaw_bus, jaw
        if jaw_toggle is None:
            return
        requested = jaw_toggle.poll()
        if child is not None and child.poll() is None:
            if requested is not None and jaw_pipe is not None:
                os.write(jaw_pipe, b'1\n' if requested else b'0\n')
            return
        if jaw_toggle.desired_open is None:
            return
        if jaw_bus is None:
            jaw_bus = ServoBus(**options)
            jaw = JawController(cfg, jaw_bus)
            jaw.preflight()
            requested = jaw_toggle.desired_open
        targets = jaw.targets(command=requested)
        if targets:
            jaw_bus.sync_positions(targets)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    def run(command):
        nonlocal child, jaw_pipe
        if pad is not None:
            checked_gamepad_sample(pad)
        close_idle_jaw()
        if jaw_toggle is None:
            child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            return wait_for_child(child, pad)
        read_fd, jaw_pipe = os.pipe()
        os.set_blocking(jaw_pipe, False)
        try:
            if jaw_toggle.desired_open is not None:
                os.write(jaw_pipe, b'1\n' if jaw_toggle.desired_open else b'0\n')
            child = subprocess.Popen([*command, '--jaw-command-fd', str(read_fd)],
                                     cwd=ROOT, start_new_session=True, pass_fds=(read_fd,))
        except BaseException:
            os.close(jaw_pipe)
            jaw_pipe = None
            raise
        finally:
            os.close(read_fd)
        try:
            return wait_for_child(child, pad, tick=update_jaw)
        finally:
            os.close(jaw_pipe)
            jaw_pipe = None

    def unload():
        nonlocal child
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        child = None
        close_idle_jaw()
        with PositionOnlyBus(**options) as bus:
            bus.set_torque(stop_ids, False, verify=True)
        if jaw_toggle is not None:
            jaw_toggle.reset()
        print('全部舵机卸力已回读确认，请继续扶住机器人。', flush=True)

    def read_torque_state():
        close_idle_jaw()
        with PositionOnlyBus(**options) as bus:
            configs = [bus.read_configuration(sid) for sid in ids]
        torques = {c['torque'] for c in configs}
        if torques not in ({0}, {1}):
            raise ValueError('扭矩状态混合，先托稳并统一卸力后再试')
        if torques == {1} and any(c['ram_gains'] != {'p':5,'d':0,'i':0} for c in configs):
            raise ValueError('当前上力增益不匹配，先托稳并卸力后再试')
        return torques

    def ready_from_pad():
        close_idle_jaw()
        if not verify_home_hold(cfg, PositionOnlyBus):
            print('HOME反馈尚未到位，A和菜单键暂不响应。请托住机器人，按B复位后再按A重试。', flush=True)
            while True:
                checked_gamepad_sample(pad)
                update_jaw()
                time.sleep(.02)
        if jaw_toggle is None:
            return wait_for_gamepad_walk(pad)
        return wait_for_gamepad_walk(pad, tick=update_jaw)

    def sequence(ready, torques):
        print('开始回HOME：通常约6秒，较远姿态自动延长。请托住躯干和头部、双脚悬空；全程Ctrl+C可停止。', flush=True)
        if args.keyboard:
            print('进入策略后从零速度开始；看到数字控制就绪提示后，直接按数字控制，无需回车。', flush=True)
        if args.zero_after_seconds is not None:
            print(f'模型开始运行{args.zero_after_seconds:g}秒后速度指令归零，继续运行模型，不因计时结束卸力。'
                  'Ctrl+C/q回车或故障停机会卸力，请先托住机器人。', flush=True)
        return run_sequence(run, ready, sys.executable,
                            seconds=args.seconds, vx=args.vx, soft_gains=torques == {0},
                            ramp_seconds=args.ramp_seconds,
                            max_target_jump_deg=args.max_target_jump_deg,
                            log_only_target_angles=args.log_only_target_angles,
                            gamepad_args=gamepad_options(args),
                            zero_after_seconds=args.zero_after_seconds,
                            keyboard_args=(['--keyboard', '--max-vx', str(args.max_vx),
                                            '--max-vy', str(args.max_vy), '--max-wz', str(args.max_wz)]
                                           if args.keyboard else []),
                            config_path=args.config.resolve())

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        original_signals[sig] = signal.signal(sig, interrupt)
    try:
        pidfile = WEB/'server.pid'
        if pidfile.exists():
            pid = int(pidfile.read_text().strip())
            proc = Path('/proc')/str(pid)
            if proc.exists():
                if b'server.py' not in (proc/'cmdline').read_bytes() or (proc/'cwd').resolve() != WEB:
                    raise RuntimeError('server.pid指向其他程序，未停止它')
                os.kill(pid, signal.SIGTERM)
                stopped_web = True
                for _ in range(50):
                    if not proc.exists():
                        break
                    time.sleep(.1)
        usage = subprocess.run(['fuser', cfg['serial']['port']], capture_output=True, text=True)
        if usage.returncode != 1:
            raise RuntimeError('串口仍被占用，未启动：'+usage.stdout+usage.stderr)
        exclusive = True
        if args.gamepad_buttons:
            pad = GamepadReader(args.gamepad_device)
            if cfg.get('sitstand_model'):
                print('策略启动后，Y键切换坐下／站起；坐姿时摇杆禁用，站起3秒后恢复行走。', flush=True)
            if cfg.get('ground_pick_model'):
                print('策略启动后，RB右肩键执行一次4秒俯身再起身；请先站起、松开摇杆再按RB。', flush=True)
            if jaw_settings is not None:
                jaw_toggle = JawButtonToggle(pad, button_code=jaw_settings.get('button_code', 307))
                print('X键：嘴巴张开／闭合切换；B同时卸力身体和嘴部。', flush=True)
            rehome = False
            while True:
                try:
                    if not rehome:
                        print('等待A键。请托住躯干和头部、双脚悬空，短按A开始回HOME；B卸力复位。', flush=True)
                        wait_for_gamepad_button(pad, HOME_BUTTON, tick=update_jaw)
                    rehome = False
                    result = sequence(ready_from_pad, read_torque_state())
                    if result == 78:
                        wait_for_support(reader=pad)
                except RehomeRequested:
                    rehome = True
                    continue
                except GamepadStop as exc:
                    print(str(exc), flush=True)
                    unload()
                    print('B已复位。松开B后可重新短按A，程序继续运行。', flush=True)
                    wait_for_b_release(pad)
                    pad.close()
                    pad = GamepadReader(args.gamepad_device)
                    if jaw_settings is not None:
                        jaw_toggle = JawButtonToggle(pad, button_code=jaw_settings.get('button_code', 307))
                    rehome = False
                    continue
                except GamepadError:
                    raise
                except Exception as exc:
                    print(f'本次过程失败：{exc}；卸力后重新等待A。', file=sys.stderr, flush=True)
                unload()
                print('本次过程结束，重新等待A；Ctrl+C退出整个程序。', flush=True)
        else:
            result = sequence(wait_for_feet, read_torque_state())
            if result == 78:
                wait_for_support()
    except GamepadStop as exc:
        print(str(exc), flush=True)
        result = 130
    except KeyboardInterrupt:
        print('收到停止请求。', flush=True)
        result = 130
    except Exception as exc:
        print(f'流程停止：{exc}', file=sys.stderr, flush=True)
        result = 1
    finally:
        # Finish the child before acquiring the serial port for final torque-OFF.
        for sig in original_signals:
            signal.signal(sig, signal.SIG_IGN)
        if exclusive:
            try:
                unload()
            except Exception as exc:
                result = 1
                print(f'无法确认卸力，请切断舵机电源：{exc}', file=sys.stderr, flush=True)
        if pad is not None:
            try:
                pad.close()
            except OSError as exc:
                print(f'关闭手柄读取器：{exc}', file=sys.stderr, flush=True)
        if stopped_web:
            with open(WEB/'logs/console.log', 'ab', buffering=0) as out:
                server = subprocess.Popen(['bash', 'run.sh'], cwd=WEB, stdin=subprocess.DEVNULL,
                    stdout=out, stderr=out, start_new_session=True)
            (WEB/'server.pid').write_text(str(server.pid)+'\n')
            print('网页调试台已恢复。', flush=True)
        for sig, handler in original_signals.items():
            signal.signal(sig, handler)
        lock.close()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
