"""Continue the supplied HD1910 checkpoints with bounded storage and evals.

Only this supervisor's children are stopped. Original weights are preserved.
The smoke gate is a real 64-world, five-iteration resumed PPO run per variant,
followed by the official ONNX export. Long training requires that gate to pass
for the same source/config fingerprints. No robot or network access is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

from train_pair import ROOT, VENV, stop_child

PROJECT = ROOT / "microduck_rl_hd1910"
DEFAULT_OUT = ROOT / "runs/20261002_robust_hd1910"
SOURCE_DIR = Path("/home/luckysir/文档/xwechat_files/wxid_s317mgfyvs8k22_37de/msg/file/2026-10")
TASKS = {
    "backlash": "Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910-Robust",
    "flat": "Mjlab-Velocity-Flat-MicroDuck-HD1910-Robust",
}
CONFIG_FILES = (
    "src/mjlab_microduck/tasks/microduck_hd1910_robust_env_cfg.py",
    "src/mjlab_microduck/tasks/hd1910_robust_mdp.py",
    "src/mjlab_microduck/tasks/hd1910_observations.py",
    "src/mjlab_microduck/tasks/microduck_hd1910_env_cfg.py",
    "src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py",
    "src/mjlab_microduck/tasks/mdp.py",
    "src/mjlab_microduck/actuator/feetech_bam.py",
    "src/mjlab_microduck/tasks/__init__.py",
)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_fingerprints() -> dict[str, str]:
    return {name: fingerprint(PROJECT / name) for name in CONFIG_FILES}


def iteration(checkpoint: Path) -> int:
    match = re.fullmatch(r"model_(\d+)(?:_(?:flat|backlash))?\.pt", checkpoint.name)
    if not match:
        raise ValueError(f"Invalid checkpoint name: {checkpoint}")
    return int(match[1])


def updates_to_label(source_iteration: int, target_iteration: int) -> int:
    # RSL-RL resumes at the saved label (rather than label+1) and saves its
    # last loop index. Include that repeated label so the final name is exact.
    if target_iteration <= source_iteration:
        raise ValueError("Target checkpoint must advance the source label")
    return target_iteration - source_iteration + 1


def train_command(variant: str, *, phase: str, run_name: str,
                  source: Path, num_envs: int, updates: int) -> list[str]:
    return [str(VENV / "bin/train"), TASKS[variant],
            "--env.scene.num-envs", str(num_envs),
            "--agent.max-iterations", str(updates),
            "--agent.experiment-name", f"microduck_hd1910_robust_{variant}_{phase}",
            "--agent.run-name", run_name, "--agent.logger", "tensorboard",
            "--agent.resume", "True", "--agent.load-run", f"^{re.escape(source.parent.name)}$",
            "--agent.load-checkpoint", f"^{re.escape(source.name)}$",
            "--agent.save-interval", "500", "--agent.seed", "42",
            "--agent.upload-model", "False"]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def latest_own_checkpoint(variant: str, original: Path) -> Path:
    root = PROJECT / f"logs/rsl_rl/microduck_hd1910_robust_{variant}_train"
    found = [p for p in root.glob("*/model_*.pt") if not p.parent.name.startswith("source_")]
    return max([original, *found], key=lambda p: (iteration(p), p.stat().st_mtime))


def record_ranking(out: Path, variant: str, checkpoint: Path, profiles: dict) -> None:
    """Report candidate quality without calling stationary survival good walking."""
    nominal, stress = (profiles[key]["aggregate"] for key in ("nominal", "stress"))
    means = [float(p["tracking_mae_normalized"]) for p in (nominal, stress)
             if p.get("tracking_mae_normalized") is not None]
    all_tracking = all(p["tracking_runs"] == p["runs"] for p in (nominal, stress))
    speed_checks = {"forward10": (0, .075), "forward20": (0, .12),
                    "lateral20": (1, .12), "lateral_minus20": (1, .12),
                    "turn100": (2, .6), "turn_minus100": (2, .6)}
    follows = all_tracking
    for profile in (nominal, stress):
        for name, (axis, threshold) in speed_checks.items():
            error = profile["results"][name]["mean_abs_error_vx_vy_wz"]
            follows &= error is not None and error[axis] < threshold
    survival = min(p["survival_fraction"] for p in (nominal, stress))
    invalid_encoder = max(p["target_encoder_proxy_violation_fraction"] or 0.
                          for p in (nominal, stress))
    entry = {"variant": variant, "checkpoint": str(checkpoint),
             "iteration": iteration(checkpoint), "worst_profile_survival_fraction": survival,
             "tracking_mae_normalized_mean": sum(means) / len(means) if means else None,
             "target_encoder_proxy_violation_fraction": invalid_encoder,
             "all_runs_have_tracking_samples": all_tracking,
             "passes_command_tracking_screen": bool(follows),
             "passes_simulation_screen": bool(follows and survival >= .9 and invalid_encoder == 0.),
             "hardware_tested": False}
    path = out / "candidate_ranking.json"
    rankings = json.loads(path.read_text()) if path.exists() else {"candidates": []}
    rankings["candidates"] = [r for r in rankings["candidates"]
                              if (r["variant"], r["iteration"]) != (variant, iteration(checkpoint))]
    rankings["candidates"].append(entry)
    same = [r for r in rankings["candidates"] if r["variant"] == variant]
    same.sort(key=lambda r: (not r["passes_simulation_screen"],
                            4. * (1. - r["worst_profile_survival_fraction"])
                            + 4. * r["target_encoder_proxy_violation_fraction"]
                            + (r["tracking_mae_normalized_mean"] if r["tracking_mae_normalized_mean"] is not None else 100.)))
    rankings.setdefault("best_evaluated", {})[variant] = same[0]
    rankings["screen_scope"] = "Heuristic simulation candidate screen; real-world walking has not been verified"
    atomic_json(path, rankings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "train"), required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-iteration", type=int, default=10000)
    parser.add_argument("--chunk-iterations", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--variants", nargs="+", choices=list(TASKS), default=list(TASKS))
    parser.add_argument("--min-start-gib", type=float, default=3.5)
    parser.add_argument("--min-running-gib", type=float, default=2.5)
    args = parser.parse_args()
    if args.chunk_iterations <= 0 or args.num_envs <= 0:
        parser.error("Environment count and chunk iterations must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop_file = out / "STOP"
    if stop_file.exists():
        raise RuntimeError(f"STOP exists; inspect previous status before removing {stop_file}")
    sources = {v: SOURCE_DIR / f"model_5000_{v}.pt" for v in args.variants}
    source_hashes = {v: fingerprint(p) for v, p in sources.items()}
    config_hashes = config_fingerprints()
    gate = out / "smoke_gate.json"
    if args.phase == "train":
        checked = json.loads(gate.read_text())
        if (not checked.get("passed") or checked["config_sha256"] != config_hashes
                or checked["source_sha256"] != source_hashes):
            raise RuntimeError("Training smoke gate missing, incomplete or stale")
    state = {"status": "running", "phase": args.phase, "pid": os.getpid(),
             "started": datetime.now().astimezone().isoformat(), "stages": [],
             "source_sha256": source_hashes, "config_sha256": config_hashes,
             "original_training_config_verified": False,
             "target_iteration": args.target_iteration, "stop_file": str(stop_file),
             "min_running_gib": args.min_running_gib,
             "checkpoint_every_iterations": 500, "original_weights_preserved": True}
    state_file = out / f"{args.phase}_status.json"
    if state_file.exists():
        previous = json.loads(state_file.read_text())
        if previous["config_sha256"] != config_hashes or previous["source_sha256"] != source_hashes:
            raise RuntimeError("This output contains a different configuration; use a new output directory")
        state["previous_status"] = previous.get("status")
        state["stages"] = previous.get("stages", [])
        state["evaluations"] = previous.get("evaluations", [])

    def save() -> None:
        state["disk_free_gib"] = shutil.disk_usage(ROOT).free / 2**30
        atomic_json(state_file, state)

    def handle_signal(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"),
               PATH=str(VENV / "bin") + os.pathsep + os.environ.get("PATH", ""),
               MUJOCO_GL="egl", WANDB_MODE="disabled", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env.pop("MICRODUCK_WARM_START", None)

    def monitored(command: list[str], name: str) -> None:
        save()
        if state["disk_free_gib"] < args.min_start_gib:
            raise RuntimeError("Disk below the start reserve; did not launch next stage")
        entry = {"name": name, "command": command, "log": str(out / f"{name}.log"),
                 "status": "running", "started": datetime.now().astimezone().isoformat()}
        state["stages"].append(entry)
        with Path(entry["log"]).open("w") as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            entry["pid"] = child.pid
            try:
                while child.poll() is None:
                    save()
                    if stop_file.exists():
                        raise KeyboardInterrupt("STOP file requested")
                    if state["disk_free_gib"] < args.min_running_gib:
                        raise RuntimeError("Disk below running reserve; stopped this stage")
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                stop_child(child)
                entry["pid"] = None
                entry["exit_code"] = child.returncode
                entry["status"] = "completed" if child.returncode == 0 else "stopped_or_failed"
                save()
        if child.returncode:
            raise RuntimeError(f"Stage failed ({child.returncode}); inspect {entry['log']}")

    def continue_train(v: str, source: Path, target: int | None) -> Path:
        experiment = PROJECT / f"logs/rsl_rl/microduck_hd1910_robust_{v}_{args.phase}"
        staged = experiment / f"source_{iteration(source)}"
        staged.mkdir(parents=True, exist_ok=True)
        staged_source = staged / source.name
        if not staged_source.exists():
            shutil.copy2(source, staged_source)
        elif fingerprint(staged_source) != fingerprint(source):
            raise RuntimeError(f"Staged checkpoint differs from source: {staged_source}")
        run_name = f"{v}_{args.phase}_{target or 'five'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        updates = 5 if args.phase == "smoke" else updates_to_label(iteration(source), target)
        monitored(train_command(v, phase=args.phase, run_name=run_name,
                                source=staged_source, num_envs=64 if args.phase == "smoke" else args.num_envs,
                                updates=updates), run_name)
        directories = list(experiment.glob(f"*_{run_name}"))
        if len(directories) != 1:
            raise RuntimeError(f"Expected one training output: {directories}")
        checkpoint = max(directories[0].glob("model_*.pt"), key=iteration)
        if target is not None and iteration(checkpoint) != target:
            raise RuntimeError(f"Training did not produce target {target}: {checkpoint}")
        return checkpoint

    def export(v: str, checkpoint: Path) -> Path:
        destination = out / "exports" / f"{v}_{iteration(checkpoint)}.onnx"
        destination.parent.mkdir(exist_ok=True)
        monitored([str(VENV / "bin/python"), str(PROJECT / "scripts/export.py"), TASKS[v],
                   "--checkpoint-file", str(checkpoint), "--num-envs", "1",
                   "--onnx-file", str(destination), "--device", "cuda:0"],
                  f"export_{v}_{iteration(checkpoint)}")
        monitored([str(VENV / "bin/python"), str(ROOT / "scripts/verify_hd1910_export.py"),
                   "--task", TASKS[v], "--checkpoint", str(checkpoint),
                   "--onnx", str(destination), "--output", str(destination.with_suffix(".parity.json"))],
                  f"parity_{v}_{iteration(checkpoint)}")
        return destination

    def evaluate(v: str, checkpoint: Path, label: str) -> dict:
        profiles = {}
        for profile in ("nominal", "stress"):
            directory = out / "eval" / f"{v}_{label}_{profile}"
            report_file = directory / "evaluation.json"
            if report_file.exists():
                existing = json.loads(report_file.read_text())
                complete = len(existing.get("results", [])) == 14 * 3
                if complete and existing.get("checkpoint_sha256") == fingerprint(checkpoint):
                    profiles[profile] = existing
                    continue
            monitored([str(VENV / "bin/python"), str(ROOT / "scripts/evaluate_hd1910_robust.py"),
                       "--task", TASKS[v], "--checkpoint", str(checkpoint),
                       "--output", str(directory), "--profile", profile,
                       "--num-envs", "32", "--seconds", "30"],
                      f"eval_{v}_{label}_{profile}")
            state.setdefault("evaluations", []).append(str(report_file))
            profiles[profile] = json.loads(report_file.read_text())
        return profiles

    save()
    try:
        if args.phase == "smoke":
            for v in args.variants:
                checkpoint = continue_train(v, sources[v], None)
                onnx = export(v, checkpoint)
                state.setdefault("smoke_outputs", {})[v] = {"checkpoint": str(checkpoint), "onnx": str(onnx)}
            # Real export verification is delegated to official exporter;
            # contract/parity checks are run by the separate evaluation gate.
            atomic_json(gate, {"passed": True, "source_sha256": source_hashes,
                               "config_sha256": config_hashes, "outputs": state["smoke_outputs"]})
        else:
            latest = {v: latest_own_checkpoint(v, p) for v, p in sources.items()}
            targets = range(6000, args.target_iteration + 1, args.chunk_iterations)
            if args.target_iteration < 6000 or (args.target_iteration - 6000) % args.chunk_iterations:
                raise ValueError("Target must align to chunks beginning at 6000")
            for target in targets:
                for v in args.variants:
                    if iteration(latest[v]) >= target:
                        continue
                    latest[v] = continue_train(v, latest[v], target)
                    state.setdefault("latest_checkpoints", {})[v] = str(latest[v])
                    export(v, latest[v])
                    # Re-evaluate the source under exactly the same profiles,
                    # commands and seeds as new candidates. The preliminary
                    # legacy-task baseline is deliberately not used to rank.
                    baseline = evaluate(v, sources[v], "baseline_5000")
                    state.setdefault("baseline_summary", {})[v] = {
                        p: baseline[p]["aggregate"] for p in baseline}
                    profiles = evaluate(v, latest[v], str(target))
                    record_ranking(out, v, latest[v], profiles)
                    save()
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        state["finished"] = datetime.now().astimezone().isoformat()
        save()
    print(f"Completed {args.phase}: {state_file}", flush=True)


if __name__ == "__main__":
    main()
