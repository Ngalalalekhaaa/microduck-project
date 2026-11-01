"""Run the validated 1699-actor continuation without resuming paused HOME jobs."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
from zoneinfo import ZoneInfo

from train_pair import ROOT, VENV, run_monitored

PROJECT = ROOT / "microduck_rl_hd1910"
TASK = "Mjlab-SitStand-Flat-MicroDuck-HD1910-Robust"
CONFIG_FILES = [
    PROJECT / "src/mjlab_microduck/tasks/mdp.py",
    PROJECT / "src/mjlab_microduck/tasks/microduck_hd1910_sitstand_robust_env_cfg.py",
    PROJECT / "src/mjlab_microduck/tasks/sitstand_warm_start.py",
    PROJECT / "src/mjlab_microduck/tasks/hd1910_observations.py",
    PROJECT / "src/mjlab_microduck/tasks/hd1910_robust_mdp.py",
    PROJECT / "src/mjlab_microduck/tasks/microduck_sitstand_env_cfg.py",
    PROJECT / "src/mjlab_microduck/robot/microduck_constants.py",
    PROJECT / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml",
    PROJECT / "src/mjlab_microduck/actuator/feetech_bam.py",
    PROJECT / "src/mjlab_microduck/robot/hd1910/1910_m6.json",
]


def fingerprints():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in CONFIG_FILES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/20261101_sitstand_robust")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = json.loads((out / "smoke_gate.json").read_text())
    if not gate["passed"] or gate["config_sha256"] != fingerprints():
        raise RuntimeError("Smoke verification missing or stale")
    source = Path(gate["source_onnx"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != gate["source_sha256"]:
        raise RuntimeError("Warm-start ONNX changed after smoke verification")
    stop = out / "STOP"
    if stop.exists():
        raise RuntimeError(f"STOP is present: {stop}")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    name = "sitstand_1699_wide_dr_35deg_" + stamp
    command = [str(VENV / "bin/python"), "-m", "mjlab_microduck.train_cli", TASK,
               "--env.scene.num-envs", str(args.num_envs),
               "--agent.max-iterations", str(args.iterations),
               "--agent.logger", "tensorboard", "--agent.seed", "42",
               "--agent.run-name", name, "--agent.upload-model", "False"]
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
               MUJOCO_GL="egl", WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="0",
               HD1910_SITSTAND_INIT_ONNX=str(source))
    env.pop("MICRODUCK_WARM_START", None)
    item = {"status": "running", "supervisor_pid": os.getpid(), "task": TASK,
            "run_name": name, "num_envs": args.num_envs, "iterations": args.iterations,
            "started": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "source_onnx": str(source), "source_sha256": gate["source_sha256"],
            "config_sha256": fingerprints(), "log": str(out / "train.log"),
            "tilt_penalty_threshold_deg": 15, "tilt_failure_deg": 35,
            "paused_home_jobs_resumed": False}

    def save():
        tmp = out / "train_status.tmp"
        tmp.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(out / "train_status.json")
        if stop.exists() and item["status"] == "running":
            raise KeyboardInterrupt("STOP requested")

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run_monitored(command, project=PROJECT, env=env, log_path=out / "train.log",
                      item=item, save=save)
        item["status"] = "completed"
    except BaseException as exc:
        item["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "failed"
        item["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        item["finished"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        save()


if __name__ == "__main__":
    main()
