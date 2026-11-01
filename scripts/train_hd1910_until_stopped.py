"""Alternate bounded HD-1910 resume chunks until STOP is requested."""
from __future__ import annotations

from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

from train_pair import ROOT, VENV, HD1910_TASKS, stop_child

PROJECT = ROOT / "microduck_rl_hd1910"
OUT = ROOT / "runs/hd1910_continuous"
PREDECESSOR = ROOT / "runs/20261001_021804_hd1910_continued/status.json"
SOURCE_EXPERIMENTS = {
    "flat": "local_hd1910_flat_train",
    "backlash": "local_hd1910_backlash_train",
}
STOP = OUT / "STOP"
MIN_START_GIB = 3.0
MIN_RUNNING_GIB = 2.5
CHUNK_ITERATIONS = 250
stopping = False


def on_signal(*_args):
    global stopping
    stopping = True


def free_gib() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def save_status(state: dict) -> None:
    temp = OUT / "status.tmp"
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    temp.replace(OUT / "status.json")


def checkpoints(variant: str) -> list[Path]:
    root = PROJECT / "logs/rsl_rl" / SOURCE_EXPERIMENTS[variant]
    return list(root.glob("*/model_*.pt"))


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def latest_checkpoint(variant: str) -> Path:
    found = checkpoints(variant)
    if not found:
        raise FileNotFoundError(f"No {variant} checkpoints")
    return max(found, key=lambda p: (checkpoint_iteration(p), p.stat().st_mtime))


def requested_stop() -> bool:
    return stopping or STOP.exists()


def stop_predecessor() -> None:
    """Forward STOP to the finite supervisor, which then stops its child."""
    if not PREDECESSOR.exists():
        return
    prev = json.loads(PREDECESSOR.read_text())
    if prev.get("status") != "running":
        return
    parent_pid = None
    child_pid = prev["runs"][-1].get("active_pid") if prev.get("runs") else None
    try:
        if isinstance(child_pid, int):
            child_status = Path(f"/proc/{child_pid}/status")
            ppid_line = next(line for line in child_status.read_text().splitlines()
                             if line.startswith("PPid:"))
            parent_pid = int(ppid_line.split()[1])
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        pass
    # Between training/export/evaluation stages there may be no active child.
    # Locate only the exact finite supervisor command in that short interval.
    candidates = [Path(f"/proc/{parent_pid}")] if parent_pid else Path("/proc").glob("[0-9]*")
    for proc in candidates:
        try:
            args = proc.joinpath("cmdline").read_bytes().split(b"\0")
            if any(Path(os.fsdecode(arg)).name == "resume_hd1910_pair.py" for arg in args if arg):
                os.kill(int(proc.name), signal.SIGINT)
                return
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass


def wait_for_predecessor(state: dict) -> bool:
    while True:
        if requested_stop():
            stop_predecessor()
            return False
        if PREDECESSOR.exists():
            prev = json.loads(PREDECESSOR.read_text())
            if prev["status"] != "running":
                state["predecessor_status"] = prev["status"]
                save_status(state)
                return True
        state["phase"] = "waiting_for_initial_continuation"
        state["disk_free_gib"] = free_gib()
        save_status(state)
        time.sleep(5)


def run_chunk(state: dict, variant: str) -> None:
    source = latest_checkpoint(variant)
    source_iteration = checkpoint_iteration(source)
    name = f"continuous_{variant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment = SOURCE_EXPERIMENTS[variant]
    log_path = OUT / f"{name}.log"
    command = [str(VENV / "bin/train"), HD1910_TASKS[variant],
               "--env.scene.num-envs", "2048",
               "--agent.max-iterations", str(CHUNK_ITERATIONS),
               "--agent.logger", "tensorboard",
               "--agent.experiment-name", experiment,
               "--agent.run-name", name,
               "--agent.resume", "True",
               "--agent.load-run", source.parent.name,
               "--agent.load-checkpoint", source.name,
               "--agent.save-interval", "250",
               "--agent.seed", "42",
               "--agent.upload-model", "False"]
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"),
               PATH=str(VENV / "bin") + os.pathsep + os.environ.get("PATH", ""),
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
               WANDB_MODE="disabled", MUJOCO_GL="egl",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env.pop("MICRODUCK_WARM_START", None)
    item = {"variant": variant, "source": str(source),
            "source_iteration": source_iteration, "command": command,
            "log": str(log_path), "status": "running",
            "started": datetime.now().astimezone().isoformat()}
    state["chunks"].append(item)
    state["phase"] = f"training_{variant}"
    save_status(state)
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=PROJECT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        item["pid"] = child.pid
        save_status(state)
        try:
            while child.poll() is None:
                state["disk_free_gib"] = free_gib()
                save_status(state)
                if requested_stop():
                    item["status"] = "stopped_by_user"
                    stop_child(child)
                    return
                if state["disk_free_gib"] < MIN_RUNNING_GIB:
                    item["status"] = "stopped_low_disk"
                    stop_child(child)
                    raise RuntimeError("Disk below 2.5 GiB reserve; training stopped")
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            item["pid"] = None
            item["exit_code"] = child.returncode
            save_status(state)
    if child.returncode:
        item["status"] = "failed"
        raise RuntimeError(f"{variant} train exited {child.returncode}; inspect {log_path}")
    candidates = list((PROJECT / "logs/rsl_rl" / experiment).glob(f"*_{name}"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one run for {name}: {candidates}")
    run_dir = candidates[0]
    new_checkpoints = list(run_dir.glob("model_*.pt"))
    final = max(new_checkpoints, key=checkpoint_iteration)
    if checkpoint_iteration(final) <= source_iteration:
        raise RuntimeError(f"No iteration advance for {variant}: {final}")
    # Keep the final checkpoint and all metadata/logs. The source checkpoint
    # remains intact; an interrupted chunk can always restart from it.
    for old in new_checkpoints:
        if old != final:
            old.unlink()
    item.update(status="completed", checkpoint=str(final),
                final_iteration=checkpoint_iteration(final),
                finished=datetime.now().astimezone().isoformat())
    # Preserve the original and finite continuation runs. For the continuous
    # runs keep the latest two recoverable checkpoints per variant.
    continuous = sorted(
        (p for p in checkpoints(variant) if "continuous_" in p.parent.name),
        key=lambda p: (checkpoint_iteration(p), p.stat().st_mtime), reverse=True)
    for old in continuous[2:]:
        old.unlink()
    save_status(state)


def main() -> None:
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Continuous HD-1910 trainer is already running") from exc
    if STOP.exists():
        raise RuntimeError(f"Remove {STOP} before starting a new run")
    state = {"status": "running", "phase": "initializing",
             "started": datetime.now().astimezone().isoformat(),
             "pid": os.getpid(), "chunks": [],
             "stop_file": str(STOP), "min_start_gib": MIN_START_GIB,
             "min_running_gib": MIN_RUNNING_GIB}
    save_status(state)
    print(f"Continuous HD-1910 training; stop with: touch {STOP}", flush=True)
    try:
        if not wait_for_predecessor(state):
            state["status"] = "stopped"
            return
        turn = 0
        while not requested_stop():
            available = free_gib()
            state["disk_free_gib"] = available
            if available < MIN_START_GIB:
                state["phase"] = "paused_low_disk"
                save_status(state)
                time.sleep(10)
                continue
            variant = ("flat", "backlash")[turn % 2]
            print(f"Continuing {variant} from {latest_checkpoint(variant)}", flush=True)
            run_chunk(state, variant)
            turn += 1
        state["status"] = "stopped"
    except BaseException as exc:
        state["status"] = "stopped" if requested_stop() else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        state["finished"] = datetime.now().astimezone().isoformat()
        state["phase"] = "idle"
        state["pid"] = None
        save_status(state)


if __name__ == "__main__":
    main()
