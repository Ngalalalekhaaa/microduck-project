"""Repeatable fixed-speed evaluation of the two latest HD-1910 checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "microduck_rl_hd1910"
OUT = ROOT / "runs/20261001_hd1910_velocity_eval"
SPEEDS = ["0.0", "0.05", "0.10", "0.15", "0.20", "0.30"]
MODELS = {
    "flat9750": PROJECT / "logs/rsl_rl/local_hd1910_flat_train/2026-10-01_09-32-10_continuous_flat_20261001_093200/model_9750.pt",
    "backlash9729": PROJECT / "logs/rsl_rl/local_hd1910_backlash_train/2026-10-01_09-21-13_continuous_backlash_20261001_092103/model_9729.pt",
}
TASKS = {
    "flat": "Mjlab-Velocity-Flat-MicroDuck-HD1910",
    "backlash": "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910",
}


def complete(directory: Path) -> bool:
    file = directory / "evaluation.json"
    if not file.exists():
        return False
    report = json.loads(file.read_text())
    return len(report.get("results", {})) == len(SPEEDS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), MUJOCO_GL="egl",
               WANDB_MODE="disabled", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for model_name, checkpoint in MODELS.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for physics, task in TASKS.items():
            name = f"{model_name}_on_{physics}"
            directory = OUT / name
            if complete(directory):
                print("Already complete:", name, flush=True)
                continue
            free_gib = shutil.disk_usage(ROOT).free / 2**30
            if free_gib < 1.5:
                raise RuntimeError(f"Disk reserve below 1.5 GiB: {free_gib:.2f} GiB")
            command = [str(ROOT / "microduck_rl/.venv/bin/python"),
                       str(ROOT / "scripts/evaluate.py"),
                       "--task", task, "--checkpoint", str(checkpoint),
                       "--output", str(directory), "--num-envs", "64",
                       "--seconds", "10", "--speeds", *SPEEDS, "--trace"]
            print("Evaluating:", name, f"free={free_gib:.2f} GiB", flush=True)
            with (OUT / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            if not complete(directory):
                raise RuntimeError(f"Incomplete evaluation: {name}")
    print("All four model/physics combinations complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Evaluation stopped: {error}", file=sys.stderr)
        raise
