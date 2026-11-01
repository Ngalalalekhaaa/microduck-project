"""Continue both HD-1910 checkpoints and finalize each continued run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess

from train_pair import ROOT, VENV, HD1910_TASKS, run_monitored

PROJECT = ROOT / "microduck_rl_hd1910"
SOURCES = {
    "flat": (
        "local_hd1910_flat_train",
        "2026-09-28_01-36-32_flat_20260928_013621",
    ),
    "backlash": (
        "local_hd1910_backlash_train",
        "2026-09-28_03-42-13_backlash_20260928_013621",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional-iterations", type=int, default=1000)
    args = parser.parse_args()
    if not 1 <= args.additional_iterations <= 4000:
        parser.error("additional iterations must be 1..4000")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = ROOT / "runs" / f"{stamp}_hd1910_continued"
    output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    status = {
        "started": datetime.now().astimezone().isoformat(),
        "status": "running",
        "additional_iterations": args.additional_iterations,
        "runs": [],
    }

    def save() -> None:
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        temp.replace(status_path)

    save()
    env = dict(
        os.environ,
        PYTHONPATH=str(PROJECT / "src"),
        PATH=str(VENV / "bin") + os.pathsep + os.environ.get("PATH", ""),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="0",
        WANDB_MODE="disabled",
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
    )
    env.pop("MICRODUCK_WARM_START", None)
    try:
        for variant, task in HD1910_TASKS.items():
            free = shutil.disk_usage(ROOT).free / 2**30
            if free < 3:
                raise RuntimeError(f"Only {free:.2f} GiB free; require 3 GiB before {variant}")
            experiment, source_run = SOURCES[variant]
            checkpoint = PROJECT / "logs/rsl_rl" / experiment / source_run / "model_3999.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            name = f"continued_{variant}_{stamp}"
            item = {
                "variant": variant,
                "task": task,
                "source_checkpoint": str(checkpoint),
                "status": "running",
            }
            status["runs"].append(item)
            save()
            cmd = [str(VENV / "bin/train"), task,
                   "--env.scene.num-envs", "2048",
                   "--agent.max-iterations", str(args.additional_iterations),
                   "--agent.logger", "tensorboard",
                   "--agent.experiment-name", experiment,
                   "--agent.run-name", name,
                   "--agent.resume", "True",
                   "--agent.load-run", source_run,
                   "--agent.load-checkpoint", "model_3999.pt",
                   "--agent.save-interval", "250",
                   "--agent.seed", "42",
                   "--agent.upload-model", "False"]
            item["command"] = cmd
            save()
            print(f"Continuing {variant} from {checkpoint}", flush=True)
            run_monitored(cmd, project=PROJECT, env=env,
                          log_path=output / f"{variant}.log", item=item, save=save)
            found = list((PROJECT / "logs/rsl_rl" / experiment).glob(f"*_{name}"))
            if len(found) != 1:
                raise RuntimeError(f"Expected one continued run for {variant}: {found}")
            continued = found[0]
            model = max(continued.glob("model_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))
            item.update(continued_run=str(continued), checkpoint=str(model), status="exporting")
            save()
            artifacts = output / variant
            artifacts.mkdir()
            onnx = artifacts / "policy.onnx"
            cmd = [str(VENV / "bin/python"), str(PROJECT / "scripts/export.py"), task,
                   "--checkpoint-file", str(model), "--num-envs", "1",
                   "--onnx-file", str(onnx), "--device", "cuda:0"]
            run_monitored(cmd, project=PROJECT, env=env,
                          log_path=artifacts / "export.log", item=item, save=save)
            item.update(onnx=str(onnx), status="evaluating")
            save()
            for eval_variant, eval_task in HD1910_TASKS.items():
                cmd = [str(VENV / "bin/python"), str(ROOT / "scripts/evaluate.py"),
                       "--task", eval_task, "--checkpoint", str(model),
                       "--output", str(artifacts / f"eval_{eval_variant}"),
                       "--onnx", str(onnx), "--commands", "stand", "forward",
                       "forward_fast", "backward", "left", "turn"]
                if eval_variant == variant:
                    cmd.append("--video")
                run_monitored(cmd, project=PROJECT, env=env,
                              log_path=artifacts / f"eval_{eval_variant}.log",
                              item=item, save=save)
            item["status"] = "completed"
            save()
        status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status["finished"] = datetime.now().astimezone().isoformat()
        save()
        print(f"Status: {status_path}", flush=True)


if __name__ == "__main__":
    main()
