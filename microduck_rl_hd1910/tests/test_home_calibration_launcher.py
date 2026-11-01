"""Exercise launcher commands and the installed CLI parser without training.

Every subprocess is intercepted, including the CUDA health check. Parsing uses
the real mjlab TrainConfig and Tyro flags that rejected a bare ``--gpu-ids 0``.
"""

import importlib.util
import json
import os
from pathlib import Path
import sys

import mjlab
from mjlab.scripts.train import TrainConfig
import pytest
import tyro


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_home_calibration.py"


def _option(command, option):
    return command[command.index(option) + 1]


def _run_launcher(monkeypatch, tmp_path, *, device, phase="smoke", no_current=False, objective="pose"):
    spec = importlib.util.spec_from_file_location("home_calibration_launcher", SCRIPT)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    # main() assigns these variables directly; register their previous values
    # with monkeypatch so the test does not alter subsequent tests' environment.
    for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MUJOCO_GL", "PYTHONUNBUFFERED"):
        monkeypatch.setenv(key, os.environ.get(key, ""))
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--phase", phase, "--device", device,
        "--num-envs", "512", "--iterations", "7", "--seed", "91",
        *(["--no-current"] if no_current else []),
        "--objective", objective,
    ])
    calls = []

    def intercept(command, *, check):
        assert check is True
        assert command[0] == sys.executable
        calls.append(command)
        if command[1:3] == ["-m", "mjlab_microduck.train_cli"]:
            # Only fabricate the artifacts the launcher discovers after a
            # successful process. No checkpoint, ONNX, or physics is executed.
            experiment = _option(command, "--agent.experiment-name")
            name = _option(command, "--agent.run-name")
            directory = tmp_path / "logs" / "rsl_rl" / experiment / f"test_{name}"
            directory.mkdir(parents=True)
            (directory / "model_0.pt").touch()
            (directory / "model_4.pt").touch()
            (directory / "home_calibration.onnx").touch()

    monkeypatch.setattr(launcher.subprocess, "run", intercept)
    launcher.main()
    return calls, (launcher.FORCE_RATIO_TASK if objective == "force-ratio" else
                   (launcher.NO_CURRENT_TASK if no_current else launcher.TASK))


@pytest.mark.parametrize("device,expected_gpu_ids", [("cpu", None), ("cuda:0", [0])])
def test_smoke_commands_parse_with_installed_mjlab_and_select_requested_device(
    monkeypatch, tmp_path, device, expected_gpu_ids,
):
    calls, task = _run_launcher(monkeypatch, tmp_path, device=device)
    training = [call for call in calls if call[1:3] == ["-m", "mjlab_microduck.train_cli"]]
    assert len(training) == 1
    command = training[0]
    assert command[3] == task
    if device == "cpu":
        assert _option(command, "--gpu-ids") == "None"
        assert not any(call[1] == "-c" for call in calls)
    else:
        assert "--gpu-ids" not in command  # keep mjlab's typed [0] default
        assert calls[0][1] == "-c"  # health check precedes physics/training

    # This is the actual installed parser, not a copied annotation or an argv
    # string check: a future invalid union/list representation fails here.
    parsed = tyro.cli(
        TrainConfig,
        args=command[4:],
        default=TrainConfig.from_task(task),
        config=mjlab.TYRO_FLAGS,
    )
    assert parsed.gpu_ids == expected_gpu_ids
    assert parsed.env.scene.num_envs == 64
    assert parsed.agent.max_iterations == 5
    assert parsed.agent.seed == 91
    assert parsed.agent.logger == "tensorboard"
    physics = next(call for call in calls if call[1] == "scripts/check_home_calibration.py")
    assert _option(physics, "--device") == device
    assert calls.index(physics) < calls.index(command)
    verification = calls[-1]
    assert verification[1] == "scripts/verify_home_calibration_export.py"
    assert Path(_option(verification, "--checkpoint")).name == "model_4.pt"
    status_file, = tmp_path.glob("logs/home_calibration_checks/*/status.json")
    status = json.loads(status_file.read_text())
    assert status["device"] == device and status["physics_and_smoke_passed"]
    assert "completed" not in status  # smoke does not claim a full training run


def test_full_cuda_run_keeps_gpu_default_after_smoke_and_evaluates_all_profiles(
    monkeypatch, tmp_path,
):
    calls, _ = _run_launcher(monkeypatch, tmp_path, device="cuda:0", phase="train")
    training = [call for call in calls if call[1:3] == ["-m", "mjlab_microduck.train_cli"]]
    assert len(training) == 2
    assert all("--gpu-ids" not in call for call in training)
    assert [_option(call, "--env.scene.num-envs") for call in training] == ["64", "512"]
    assert [_option(call, "--agent.max-iterations") for call in training] == ["5", "7"]
    evaluations = [call for call in calls
                   if call[1] == "scripts/check_home_calibration.py" and "--checkpoint" in call]
    assert [_option(call, "--profile") for call in evaluations] == ["nominal", "train", "stress"]
    assert all(_option(call, "--device") == "cuda:0" for call in evaluations)
    status_file, = tmp_path.glob("logs/home_calibration_checks/*/status.json")
    assert json.loads(status_file.read_text())["completed"] is True


def test_no_current_variant_is_used_for_physics_smoke_export_training_and_evaluation(monkeypatch, tmp_path):
    calls, task = _run_launcher(monkeypatch, tmp_path, device="cuda:0", phase="train", no_current=True)
    training = [call for call in calls if call[1:3] == ["-m", "mjlab_microduck.train_cli"]]
    assert task == "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
    assert len(training) == 2 and all(call[3] == task for call in training)
    assert [_option(call, "--env.scene.num-envs") for call in training] == ["64", "512"]
    assert all("home_calibration_no_current" in _option(call, "--agent.experiment-name") for call in training)
    assert all("--agent.resume" not in call and "--agent.load-checkpoint" not in call for call in training)
    checked = [call for call in calls if call[1] in (
        "scripts/check_home_calibration.py", "scripts/verify_home_calibration_export.py")]
    assert len(checked) == 6 and all("--no-current" in call for call in checked)
    parsed = tyro.cli(TrainConfig, args=training[0][4:], default=TrainConfig.from_task(task), config=mjlab.TYRO_FLAGS)
    assert parsed.env.scene.num_envs == 64 and parsed.agent.max_iterations == 5
    status_file, = tmp_path.glob("logs/home_calibration_checks/*/status.json")
    status = json.loads(status_file.read_text())
    assert status["task"] == task and status["current_input"] is False


def test_force_ratio_launcher_keeps_target_task_and_no_current_through_every_stage(monkeypatch, tmp_path):
    calls, task = _run_launcher(monkeypatch, tmp_path, device="cuda:0", phase="train", objective="force-ratio")
    training = [call for call in calls if call[1:3] == ["-m", "mjlab_microduck.train_cli"]]
    assert task == "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"
    assert len(training) == 2 and all(call[3] == task for call in training)
    assert all("home_force_ratio_no_current" in _option(call, "--agent.experiment-name") for call in training)
    checked = [call for call in calls if call[1] in (
        "scripts/check_home_calibration.py", "scripts/verify_home_calibration_export.py")]
    assert all("--no-current" in call and _option(call, "--objective") == "force-ratio" for call in checked)
    parsed = tyro.cli(TrainConfig, args=training[0][4:], default=TrainConfig.from_task(task), config=mjlab.TYRO_FLAGS)
    assert parsed.env.scene.num_envs == 64 and parsed.agent.max_iterations == 5
