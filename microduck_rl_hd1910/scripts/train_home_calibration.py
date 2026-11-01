"""Validate HOME physics, run 64-world/5-iteration PPO smoke, then train.

Uses the interpreter that launches this script; never installs dependencies.
The smoke checkpoint is only a plumbing check and is not used to initialize
the full run. Long runs always start from scratch unless resumed separately
through the standard training CLI.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


TASK = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
NO_CURRENT_TASK = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
FORCE_RATIO_TASK = "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), default="train")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--no-current", action="store_true",
                        help="Train the separate 272D encoder/IMU-only HOME policy from scratch")
    parser.add_argument("--objective", choices=("pose", "force-ratio"), default="pose",
                        help="Force-ratio uses a 384D actor with a fixed reference target and no measured current")
    args = parser.parse_args()
    force_ratio = args.objective == "force-ratio"
    args.no_current = args.no_current or force_ratio
    task = FORCE_RATIO_TASK if force_ratio else (NO_CURRENT_TASK if args.no_current else TASK)
    variant_args = ["--no-current"] if args.no_current else []
    if force_ratio:
        variant_args += ["--objective", "force-ratio"]
    experiment = "microduck_hd1910_home_calibration" + ("_no_current" if args.no_current else "")
    if force_ratio:
        experiment = "microduck_hd1910_home_force_ratio_no_current"
    if args.num_envs < 1 or args.iterations < 1:
        parser.error("--num-envs and --iterations must be positive")
    if args.device == "cpu" and args.phase == "train":
        parser.error("CPU is supported for smoke validation only; use CUDA for the long run")
    os.chdir(ROOT)
    os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYTHONUNBUFFERED"] = "1"
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    checks = ROOT / "logs" / "home_calibration_checks" / tag
    checks.mkdir(parents=True)

    def run(*arguments):
        subprocess.run([sys.executable, *map(str, arguments)], check=True)

    if args.device.startswith("cuda"):
        # Test a real CUDA operation: NVML/device_count alone can still report
        # a GPU whose driver recovery action is Reboot after a device fault.
        run("-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable: inspect nvidia-smi -q (GPU Recovery Action)'; x=torch.ones(4,device='cuda:0'); assert x.sum().item()==4; print(torch.cuda.get_device_name(0))")
    run("scripts/check_home_calibration.py", "--num-envs", 4, "--seconds", 3,
        "--device", args.device, "--output", checks / "physics_preflight.json", *variant_args)

    def train(experiment, name, num_envs, iterations):
        # mjlab defaults to [0]. Its list[int] | str | None CLI does not
        # accept a bare "0" as a GPU list; only override it for CPU smoke.
        gpu_arguments = [] if args.device.startswith("cuda") else ["--gpu-ids", "None"]
        run("-m", "mjlab_microduck.train_cli", task,
            *gpu_arguments,
            "--env.scene.num-envs", num_envs, "--agent.max-iterations", iterations,
            "--agent.seed", args.seed, "--agent.logger", "tensorboard",
            "--agent.experiment-name", experiment, "--agent.run-name", name)
        matches = list((ROOT / "logs" / "rsl_rl" / experiment).glob(f"*_{name}"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one output directory for {name}, found {matches}")
        directory = matches[0]
        checkpoints = list(directory.glob("model_*.pt"))
        checkpoint = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        return directory, checkpoint

    smoke_dir, smoke_checkpoint = train(
        experiment + "_smoke", f"smoke_{tag}", 64, 5,
    )
    run("scripts/verify_home_calibration_export.py", "--checkpoint", smoke_checkpoint,
        "--onnx", smoke_dir / "home_calibration.onnx", "--output", checks / "smoke_export.json", *variant_args)
    summary = {"task": task, "device": args.device, "phase": args.phase,
               "current_input": not args.no_current, "objective": args.objective,
               "smoke_directory": str(smoke_dir), "physics_and_smoke_passed": True}
    (checks / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.phase == "smoke":
        print(f"HOME smoke and export passed. Reports: {checks}", flush=True)
        return

    directory, checkpoint = train(
        experiment, f"home_{'no_current_' if args.no_current else ''}bias3deg_history8_{tag}",
        args.num_envs, args.iterations,
    )
    run("scripts/verify_home_calibration_export.py", "--checkpoint", checkpoint,
        "--onnx", directory / "home_calibration.onnx", "--output", checks / "final_export.json", *variant_args)
    for profile in ("nominal", "train", "stress"):
        run("scripts/check_home_calibration.py", "--checkpoint", checkpoint,
            "--num-envs", 32, "--seconds", 8, "--seed", 1701,
            "--profile", profile, "--device", args.device,
            "--output", directory / f"evaluation_{profile}.json", *variant_args)
    summary.update({"training_directory": str(directory), "completed": True})
    (checks / "status.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Training and evaluation finished: {directory}", flush=True)


if __name__ == "__main__":
    main()
