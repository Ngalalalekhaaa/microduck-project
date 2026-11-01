"""Boot listener: wait for Xbox, then run the existing A/Menu/B workflow."""
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from microduck_deploy.gamepad import GamepadError, GamepadReader


def settings(environ):
    variant = environ.get('MICRODUCK_VARIANT', 'backlash')
    raw_model = environ.get('MICRODUCK_MODEL', '')
    if any(char in raw_model for char in '\0\n\r'):
        raise ValueError('MICRODUCK_MODEL cannot contain newlines or null characters')
    model = raw_model.strip()
    seconds = float(environ.get('MICRODUCK_SECONDS', '0'))
    speed = float(environ.get('MICRODUCK_MAX_VX', '0.3'))
    if variant not in ('flat', 'backlash'):
        raise ValueError('MICRODUCK_VARIANT must be flat or backlash')
    if not math.isfinite(seconds) or seconds < 0 or 0 < seconds < 3:
        raise ValueError('MICRODUCK_SECONDS must be 0 (no timer) or at least 3')
    if not math.isfinite(speed) or not 0 < speed <= .4:
        raise ValueError('MICRODUCK_MAX_VX must be in (0,0.4]')
    if model and Path(model).suffix.lower() != '.onnx':
        raise ValueError('MICRODUCK_MODEL must be a relative or absolute .onnx path')
    return [model or variant, str(seconds), str(speed)]


def controller_command(arguments, *, model_override=False):
    script = 'xbox_model_hd1910.sh' if model_override else 'xbox_friction_5000_hd1910.sh'
    return ['bash', str(ROOT/script), *arguments]


def session_busy():
    (ROOT/'logs').mkdir(exist_ok=True)
    with open(ROOT/'logs/supported_sequence.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def xbox_ready():
    # This probe opens only the input device. It sends no servo instructions.
    pad = GamepadReader()
    try:
        sample = pad.latest()
        return sample.connected
    finally:
        pad.close()


def supervise(arguments, stop, *, model_override=False, ready=xbox_ready, busy=session_busy,
              launch=subprocess.Popen, emit=print):
    last_state = None

    def status(message):
        nonlocal last_state
        if message != last_state:
            emit(message, flush=True)
            last_state = message

    while not stop.is_set():
        if busy():
            status('已有手动控制流程运行，等待其退出。')
            stop.wait(2)
            continue
        try:
            connected = ready()
        except (GamepadError, OSError, subprocess.TimeoutExpired):
            connected = False
        if not connected:
            status('开机服务已就绪，等待已配对的Xbox手柄开机连接。')
            stop.wait(2)
            continue
        if stop.is_set():
            break
        status('Xbox已连接，启动按键监听；等待新按下A，不自动运动。')
        child = launch(controller_command(arguments, model_override=model_override),
                       cwd=ROOT, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            while child.poll() is None and not stop.wait(.1):
                pass
        finally:
            if child.poll() is None:
                # Parent workflow stops its motion child, then confirms torque OFF.
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    status('控制程序未在15秒内退出，已终止；请检查舵机供电。')
        if not stop.is_set():
            status(f'按键流程退出（{child.returncode}），重新等待手柄和新的A键。')
            stop.wait(2)


def main():
    arguments = settings(os.environ)
    model_override = bool(os.environ.get('MICRODUCK_MODEL', '').strip())
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    duration = '持续运行直到B键或故障停止' if float(arguments[1]) == 0 else f'每次最长{arguments[1]}秒'
    description = arguments[0] if model_override else f'HD-1910摩擦随机化5000轮 {arguments[0]}'
    print(f'默认模型：{description}；{duration}。', flush=True)
    supervise(arguments, stop, model_override=model_override)


if __name__ == '__main__':
    main()
