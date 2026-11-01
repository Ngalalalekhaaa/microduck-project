"""CLI safety boundaries using temporary files and fakes; no robot access."""

import fcntl
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from microduck_deploy.config import default_config
from microduck_deploy.home_calibration_hardware import current_mapping_template


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "home_calibration.py"
_SPEC = importlib.util.spec_from_file_location("home_calibration_cli_test", _SCRIPT)
flow = importlib.util.module_from_spec(_SPEC)
_PATH = sys.path[:]
try:
    _SPEC.loader.exec_module(flow)
finally:
    sys.path[:] = _PATH


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = default_config()
    cfg["calibration"] = {"pose": "home"}
    cfg["imu_mount_verified"] = True
    cfg["imu"]["i2c_bus"] = 2
    for index, joint in enumerate(cfg["joints"]):
        joint.update(direction=(-1 if index % 2 else 1), mapping_verified=True,
                     zero_tick=2048.25)
    config_path = tmp_path / "robot.json"
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    original = config_path.read_bytes()
    model = tmp_path / "home.onnx"
    model.write_bytes(b"inference is mocked by the CLI tests")
    policy = SimpleNamespace(contract=SimpleNamespace(requires_current=True), inspect=Mock(return_value={
        "providers": ["CPUExecutionProvider"], "zero_input_action": [0.] * 14,
    }))
    loader = Mock(return_value=policy)
    monkeypatch.setattr(flow, "HomeCalibrationPolicy", loader)
    guards = {}
    for name in ("CalibrationServoBus", "ReadOnlyServoBus", "ProcessImuReader", "run_session"):
        guards[name] = Mock(side_effect=AssertionError(f"Unexpected hardware access: {name}"))
        monkeypatch.setattr(flow, name, guards[name])
    fuser = Mock(side_effect=AssertionError("Unexpected device ownership probe"))
    monkeypatch.setattr(flow.subprocess, "run", fuser)
    killers = {name: Mock(side_effect=AssertionError("Must not signal other processes"))
               for name in ("kill", "killpg")}
    for name, mock in killers.items():
        monkeypatch.setattr(os, name, mock)
    state = SimpleNamespace(
        cfg=cfg, config=config_path, original=original, model=model,
        report=tmp_path / "inspection.json", policy=policy, loader=loader,
        guards=guards, fuser=fuser, killers=killers,
    )
    state.args = ["--model", str(model), "--config", str(config_path),
                  "--report", str(state.report)]
    yield state
    assert config_path.read_bytes() == original
    for guard in guards.values():
        guard.assert_not_called()
    for killer in killers.values():
        killer.assert_not_called()


def test_default_inspect_is_offline_and_preserves_robot_config(setup):
    assert flow.main(setup.args) == 0
    setup.loader.assert_called_once_with(setup.model, contract_path=None)
    setup.policy.inspect.assert_called_once_with()
    setup.fuser.assert_not_called()
    report = json.loads(setup.report.read_text())
    assert report["mode"] == "inspect"
    assert report["hardware_opened"] is False
    assert report["current_mapping_verified"] is False
    assert report["current_check"] == "stop"
    assert report["current_stop_ma"] == 800.
    assert report['expected_ram_gains'] == {'p': 5, 'd': 0, 'i': 0}
    assert report['training_nominal_ram_p'] == 5
    assert report['gain_override_from_training'] is False
    assert report["config_sha256"] == hashlib.sha256(setup.original).hexdigest()
    assert report["joint_ids"] == [joint["id"] for joint in setup.cfg["joints"]]
    assert not (setup.config.parent / "logs" / "supported_sequence.lock").exists()


def test_inspect_log_only_records_effective_current_policy_without_hardware(setup):
    assert flow.main(setup.args + ['--current-check', 'log-only']) == 0
    report = json.loads(setup.report.read_text())
    assert report['current_check'] == 'log-only'
    assert report['current_stop_ma'] is None
    assert report['hardware_opened'] is False
    setup.fuser.assert_not_called()


def test_inspect_p8_reports_requested_gain_without_hardware_or_config_changes(setup):
    assert flow.main(setup.args + ['--ram-p', '8']) == 0
    report = json.loads(setup.report.read_text())
    assert report['expected_ram_gains'] == {'p': 8, 'd': 0, 'i': 0}
    assert report['training_nominal_ram_p'] == 5
    assert report['gain_override_from_training'] is True
    setup.fuser.assert_not_called()


@pytest.mark.parametrize('gain', ['6', '16', '32', '5.0'])
def test_unapproved_gain_rejects_before_model_and_hardware(setup, gain):
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args + ['--ram-p', gain])
    assert error.value.code == 2
    setup.loader.assert_not_called()
    setup.fuser.assert_not_called()


def test_invalid_current_policy_rejects_before_model_and_hardware(setup):
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args + ['--current-check', 'ignore-everything'])
    assert error.value.code == 2
    setup.loader.assert_not_called()
    setup.fuser.assert_not_called()


def test_inspect_writes_only_separate_unverified_template_and_refuses_overwrite(setup):
    template = setup.config.parent / "current-template.json"
    assert flow.main(setup.args + ["--write-current-template", str(template)]) == 0
    spec = json.loads(template.read_text())
    assert all(row["verified"] is False for row in spec["joints"])
    before = template.read_bytes()
    with pytest.raises(FileExistsError):
        flow.main(setup.args + ["--write-current-template", str(template)])
    assert template.read_bytes() == before
    setup.fuser.assert_not_called()


def test_inspect_accepts_unverified_mapping_for_readonly_diagnostics(setup):
    mapping = setup.config.parent / "current.json"
    mapping.write_text(json.dumps(current_mapping_template(setup.cfg)))
    assert flow.main(setup.args + ["--current-map", str(mapping)]) == 0
    assert json.loads(setup.report.read_text())["current_mapping_verified"] is False
    setup.fuser.assert_not_called()


def test_run_without_standing_confirmation_rejects_before_model_or_hardware(setup):
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args + ["--mode", "run"])
    assert error.value.code == 2
    setup.loader.assert_not_called()
    setup.fuser.assert_not_called()
    assert not setup.report.exists()


def test_run_without_current_map_rejects_before_hardware(setup):
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args + ["--mode", "run", "--standing-confirmed"])
    assert error.value.code == 2
    setup.fuser.assert_not_called()
    assert not setup.report.exists()


@pytest.mark.parametrize('mode', ['run', 'monitor'])
@pytest.mark.parametrize('current_check', ['stop', 'log-only'])
@pytest.mark.parametrize('expected_p', [5, 8])
def test_no_current_family_can_enter_runtime_without_mapping(setup, monkeypatch, mode, current_check, expected_p):
    setup.policy.contract.requires_current = False
    bus, imu = object(), SimpleNamespace(close=Mock())
    bus_name = 'CalibrationServoBus' if mode == 'run' else 'ReadOnlyServoBus'
    bus_factory = Mock(return_value=nullcontext(bus))
    imu_factory = Mock(return_value=imu)
    runner = Mock(return_value={'samples': 4, 'exit_state': 'fake_no_hardware'})
    monkeypatch.setattr(flow, bus_name, bus_factory)
    monkeypatch.setattr(flow, 'ProcessImuReader', imu_factory)
    monkeypatch.setattr(flow, 'run_session', runner)
    setup.fuser.side_effect = None
    setup.fuser.return_value = subprocess.CompletedProcess([], 1, stdout='', stderr='')
    args = setup.args + ['--mode', mode, '--current-check', current_check]
    if expected_p != 5:
        args += ['--ram-p', str(expected_p)]
    if mode == 'run':
        args += ['--standing-confirmed']
    assert flow.main(args) == 0
    runner.assert_called_once()
    assert runner.call_args.args[1:] == (setup.policy, bus, imu, None)
    assert runner.call_args.kwargs['mode'] == mode
    assert runner.call_args.kwargs['current_check'] == current_check
    assert runner.call_args.kwargs['expected_p'] == expected_p
    imu.close.assert_called_once_with()


@pytest.mark.parametrize("encoding,verified", [("sign_magnitude15", False), ("unsigned16", True)])
def test_run_unverified_or_unsigned_current_map_rejects_before_hardware(setup, encoding, verified):
    spec = current_mapping_template(setup.cfg)
    spec["encoding"] = encoding
    for joint in spec["joints"]:
        joint["verified"] = verified
    mapping = setup.config.parent / "current.json"
    mapping.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="verified signed"):
        flow.main(setup.args + ["--mode", "run", "--standing-confirmed",
                                "--current-map", str(mapping)])
    setup.fuser.assert_not_called()
    assert not setup.report.exists()


@pytest.mark.parametrize("seconds", ["nan", "inf", "0", "-1", "10.1"])
def test_invalid_duration_rejects_before_loading_or_hardware(setup, seconds):
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args + ["--mode", "monitor", "--seconds", seconds])
    assert error.value.code == 2
    setup.loader.assert_not_called()
    setup.fuser.assert_not_called()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink", "temporary", "samples",
                                  "positions", "positions_tmp"])
def test_report_cannot_overwrite_configuration_even_through_aliases(setup, alias):
    report = setup.config
    if alias in ("symlink", "hardlink"):
        report = setup.config.parent / "alias.json"
        if alias == "symlink":
            report.symlink_to(setup.config)
        else:
            os.link(setup.config, report)
    elif alias in ("temporary", "samples", "positions", "positions_tmp"):
        report = setup.config.parent / "result.json"
        suffix = {'temporary': '.json.tmp', 'samples': '.samples.jsonl',
                  'positions': '.positions.csv', 'positions_tmp': '.positions.csv.tmp'}[alias]
        output = report.with_suffix(suffix)
        os.link(setup.config, output)
    with pytest.raises(SystemExit) as error:
        flow.main(setup.args[:-2] + ["--report", str(report)])
    assert error.value.code == 2
    setup.policy.inspect.assert_not_called()
    setup.fuser.assert_not_called()


@pytest.mark.parametrize("returncode", [0, 2])
def test_serial_busy_or_ownership_check_failure_exits_without_killing_or_hardware(setup, returncode):
    setup.fuser.side_effect = None
    setup.fuser.return_value = subprocess.CompletedProcess(
        ["fuser", setup.cfg["serial"]["port"]], returncode, stdout="12345", stderr="busy",
    )
    assert flow.main(setup.args + ["--mode", "monitor"]) == 1
    setup.fuser.assert_called_once_with(
        ["fuser", setup.cfg["serial"]["port"]], capture_output=True, text=True,
    )


def test_imu_busy_exits_without_opening_or_killing_existing_reader(setup):
    setup.fuser.side_effect = [
        subprocess.CompletedProcess([], 1, stdout="", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="12345", stderr=""),
    ]
    assert flow.main(setup.args + ["--mode", "monitor"]) == 1
    commands = [call.args[0] for call in setup.fuser.call_args_list]
    assert commands == [["fuser", setup.cfg["serial"]["port"]], ["fuser", "/dev/i2c-2"]]
    assert all("-k" not in command for command in commands)


def test_existing_sequence_lock_exits_before_device_probe_without_killing(setup):
    logdir = setup.config.parent / "logs"
    logdir.mkdir()
    with (logdir / "supported_sequence.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert flow.main(setup.args + ["--mode", "monitor"]) == 1
        setup.fuser.assert_not_called()
        # The failed contender did not release the existing owner's lock.
        with (logdir / "supported_sequence.lock").open("a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("raise_inside", [False, True])
def test_stop_signals_only_set_stop_event_and_restore_original_handlers(monkeypatch, raise_inside):
    previous = {sig: object() for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    active = dict(previous)

    def install(sig, handler):
        old = active[sig]
        active[sig] = handler
        return old

    monkeypatch.setattr(flow.signal, "signal", install)
    event = threading.Event()

    def exercise():
        with flow.stop_signals(event):
            for sig in previous:
                event.clear()
                active[sig](sig, None)
                assert event.is_set()
            if raise_inside:
                raise RuntimeError("test cleanup")

    if raise_inside:
        with pytest.raises(RuntimeError, match="test cleanup"):
            exercise()
    else:
        exercise()
    assert active == previous


@pytest.mark.parametrize('wrapper', ['hold_calibrated_home.sh', 'run_home_calibration.sh'])
@pytest.mark.parametrize('override', [False, True])
def test_no_current_wrappers_emit_p8_default_with_explicit_p5_override(wrapper, override):
    script = _SCRIPT.parent / 'home_no_current' / wrapper
    # Override only the shell exec builtin, capturing argv without launching
    # Python or opening any robot device. Execute the actual wrapper source.
    harness = 'exec() { printf "%s\\0" "$@"; }\nsource "$1" "${@:2}"\n'
    command = ['bash', '-c', harness, 'wrapper-test', str(script)]
    if override:
        command += ['--ram-p', '5']
    result = subprocess.run(command, check=True, capture_output=True)
    args = result.stdout.decode().rstrip('\0').split('\0')
    selected = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '--ram-p']
    assert selected == (['8', '5'] if override else ['8'])
