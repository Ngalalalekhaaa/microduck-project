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


def run_sequence(run, ready, python, *, seconds, vx, soft_gains=True, ramp_seconds=0,
                 max_target_jump_deg=None):
    home = [python, str(ROOT/'scripts/supported_home.py'), '--target', 'home',
            '--log-only-feedback', '--move']
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
    return run([python, str(ROOT/'scripts/policy_trial.py'), '--calibrate-imu',
                '--seconds', str(seconds), '--vx', str(vx), '--move', *ramp, *jump])


def wait_for_feet():
    print('\nHOME慢移完成，保持上力；精确到位情况见上面的 errors_ticks。\n'
          '请将双脚放到地面，扶稳躯干并保持静止。\n'
          '准备好后输入 walk 或 WALK 并回车：开始校偏和强化学习。\n'
          '60秒未确认、输入其他内容或Ctrl+C：停止并卸力。', flush=True)
    readable, _, _ = select.select([sys.stdin], [], [], 60)
    return bool(readable) and sys.stdin.readline().strip().upper() == 'WALK'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--move', action='store_true')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--vx', type=float, default=.3)
    p.add_argument('--ramp-seconds', type=float, default=0)
    p.add_argument('--max-target-jump-deg', type=float, default=None)
    args = p.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 60:
        p.error('--seconds must be in (0,60]')
    if not math.isfinite(args.vx) or not 0 <= args.vx <= .3:
        p.error('--vx must be in [0,0.3]')
    if not math.isfinite(args.ramp_seconds) or not 0 <= args.ramp_seconds <= args.seconds:
        p.error('--ramp-seconds must be in [0,seconds]')
    if args.max_target_jump_deg is not None and (
            not math.isfinite(args.max_target_jump_deg) or not 0 < args.max_target_jump_deg <= 45):
        p.error('--max-target-jump-deg must be in (0,45]')
    if not args.move:
        print(f'预览：慢速HOME → 输入WALK → {args.seconds:g}秒，vx={args.vx:g}m/s。'
              '添加 --move 才执行。未访问硬件。')
        return 0
    from microduck_deploy.config import load_config
    from supported_home import PositionOnlyBus
    cfg = load_config(ROOT/'robot.json')
    ids = [j['id'] for j in cfg['joints']]
    (ROOT/'logs').mkdir(exist_ok=True)
    lock = open(ROOT/'logs/supported_sequence.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    options = dict(cfg['serial'])
    options['timeout'] = max(.05, options['timeout'])
    stopped_web = False
    exclusive = False
    child = None
    result = 1
    original_signals = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    def run(command):
        nonlocal child
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        return child.wait()

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
        with PositionOnlyBus(**options) as bus:
            configs = [bus.read_configuration(sid) for sid in ids]
        torques = {c['torque'] for c in configs}
        if torques not in ({0}, {1}):
            raise ValueError('扭矩状态混合，先托稳并统一卸力后再试')
        if torques == {1} and any(c['ram_gains'] != {'p':5,'d':0,'i':0} for c in configs):
            raise ValueError('当前上力增益不匹配，先托稳并卸力后再试')
        print('开始慢速HOME。请托住躯干和头部、双脚悬空；全程Ctrl+C可停止。', flush=True)
        result = run_sequence(run, wait_for_feet, sys.executable,
                              seconds=args.seconds, vx=args.vx, soft_gains=torques == {0},
                              ramp_seconds=args.ramp_seconds,
                              max_target_jump_deg=args.max_target_jump_deg)
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
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        if exclusive:
            try:
                with PositionOnlyBus(**options) as bus:
                    bus.set_torque(ids, False, verify=True)
                print('全部舵机卸力已回读确认，请继续扶住机器人。', flush=True)
            except Exception as exc:
                result = 1
                print(f'无法确认卸力，请切断舵机电源：{exc}', file=sys.stderr, flush=True)
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
