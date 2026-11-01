"""Offline suite verification and explicit configuration/launch boundaries."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_runtime import calibrated_config

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/model_suite.py"
spec = importlib.util.spec_from_file_location("model_suite_tested", SCRIPT)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class FakePolicy:
    def __init__(self, path, *, policy_kind="walking"):
        assert Path(path).is_file()
        self.session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])

    def infer(self, obs):
        return np.asarray(obs[:14] * .1, dtype=np.float32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "deploy"
    (root / "models").mkdir(parents=True)
    (root / "samples").mkdir()
    models = {}
    for name, kind in suite.KINDS.items():
        model, samples = root / f"models/{name}.onnx", root / f"samples/{name}.npz"
        model.write_bytes(("fake:"+name).encode())
        obs = np.arange(4*61, dtype=np.float32).reshape(4, 61) * .001
        np.savez(samples, obs=obs, action=(obs[:, :14]*.1).astype(np.float32))
        models[name] = {"kind": kind, "path": str(model.relative_to(root)),
                        "samples": str(samples.relative_to(root)),
                        "sha256": suite.digest(model), "samples_sha256": suite.digest(samples),
                        "observed_target_min": suite.HOME.tolist(),
                        "observed_target_max": suite.HOME.tolist()}
    catalog = {"schema_version": 1, "models": models}
    (root / "models/suite_20261028.json").write_text(json.dumps(catalog))
    (root / "models/old.onnx").write_bytes(b"old policy")
    (root / "xbox_model_hd1910.sh").write_text("#!/bin/sh\nexit 0\n")
    cfg = calibrated_config()
    cfg["model"] = "models/old.onnx"
    cfg["extra_robot_settings"] = {"jaw": [12, 34], "preserve": True}
    cfg["control"]["log_only_target_angles"] = True
    config = root / "robot.json"
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(suite, "OnnxPolicy", FakePolicy)
    return SimpleNamespace(root=root, cfg=cfg, config=config, catalog=catalog)


def test_check_cpu_hash_contract_parity_and_calibration_have_no_side_effects(setup, monkeypatch):
    before = {p: p.read_bytes() for p in setup.root.rglob("*") if p.is_file()}
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: pytest.fail("check must not inspect ports"))
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: pytest.fail("check must not run external commands"))
    result = suite.check_suite(root=setup.root, cfg=setup.cfg)
    assert result["status"] == "ok" and result["hardware_access"] is False
    assert result["calibration"]["status"] == "ok"
    assert all(v["samples_count"] == 4 and v["max_abs_error"] == 0 for v in result["models"].values())
    assert before == {p: p.read_bytes() for p in setup.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("field", ["path", "samples"])
def test_hash_mismatch_rejects_corrupted_artifact(setup, field):
    entry = setup.catalog["models"]["roulade"]
    (setup.root / entry[field]).write_bytes(b"corrupt")
    result = suite.check_suite(root=setup.root)
    assert result["status"] == "failed"
    assert "SHA256" in result["models"]["roulade"]["error"]


def test_action_parity_is_checked_after_matching_hash(setup):
    entry = setup.catalog["models"]["walk"]
    samples = setup.root / entry["samples"]
    with np.load(samples) as saved:
        obs, actions = saved["obs"], saved["action"]
    actions[0, 0] += .1
    np.savez(samples, obs=obs, action=actions)
    entry["samples_sha256"] = suite.digest(samples)
    (setup.root / "models/suite_20261028.json").write_text(json.dumps(setup.catalog))
    assert "输出不一致" in suite.check_suite(root=setup.root)["models"]["walk"]["error"]


def test_catalog_refuses_outside_paths_and_wrong_policy_kind(setup):
    for patch in ({"path": "../outside.onnx"}, {"kind": "ground_pick"}):
        catalog = copy.deepcopy(setup.catalog)
        catalog["models"]["roulade"].update(patch)
        (setup.root / "models/suite_20261028.json").write_text(json.dumps(catalog))
        with pytest.raises(ValueError):
            suite.load_catalog(setup.root)


def test_unselected_hardware_feasibility_is_warning_not_contract_failure(setup):
    setup.catalog["models"]["roulade"]["observed_target_max"][0] = 100.
    (setup.root / "models/suite_20261028.json").write_text(json.dumps(setup.catalog))
    result = suite.check_suite(root=setup.root, cfg=setup.cfg)
    assert result["status"] == "ok"
    assert result["models"]["roulade"]["warnings"]


def test_check_does_not_require_configured_old_model_to_be_present(setup):
    setup.cfg["model"] = "missing-old-model.onnx"
    assert suite.check_suite(root=setup.root, cfg=setup.cfg)["status"] == "ok"


def test_select_defaults_preview_keep_walk_com_and_roll_off(setup, monkeypatch, capsys):
    before = setup.config.read_bytes()
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: pytest.fail("preview should not inspect hardware state"))
    assert suite.main(["select"], root=setup.root) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "preview" and result["walking_choice"] == "keep"
    assert "model" not in result["changes"]
    assert result["changes"]["sitstand_model"]["after"] == "models/sitstand_com.onnx"
    assert result["changes"]["control.roulade_enabled"]["after"] is False
    assert "仅记录" in result["target_check_mode"]
    assert setup.config.read_bytes() == before
    assert not list(setup.root.glob("robot.json.bak.*"))


def test_apply_changes_only_selected_model_keys_preserves_calibration_and_backs_up(setup, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: calls.append(cfg))
    assert suite.main(["select", "--apply", "--walking", "new", "--sitstand", "friction", "--roulade", "on"], root=setup.root) == 0
    after = json.loads(setup.config.read_text())
    assert after["model"] == "models/walk.onnx"
    assert after["sitstand_model"] == "models/sitstand_friction.onnx"
    assert after["ground_pick_model"] == "models/ground_pick.onnx"
    assert after["roulade_model"] == "models/roulade.onnx"
    assert after["control"]["roulade_enabled"] is True
    expected = copy.deepcopy(setup.cfg)
    for key in ("model", "sitstand_model", "ground_pick_model", "roulade_model"):
        after[key] = expected[key] if key in expected else None
        expected.setdefault(key, None)
    after["control"]["roulade_enabled"] = expected["control"]["roulade_enabled"]
    assert after == expected
    assert len(calls) == 1
    backups = list(setup.root.glob("robot.json.bak.*"))
    assert len(backups) == 1 and json.loads(backups[0].read_text()) == setup.cfg


@pytest.mark.parametrize("command", ["select", "run"])
def test_active_control_refuses_apply_or_run_before_write(setup, monkeypatch, command):
    before = setup.config.read_bytes()
    def active(_):
        raise RuntimeError("活动控制进程")
    monkeypatch.setattr(suite, "assert_idle", active)
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    argv = [command] + (["--apply"] if command == "select" else [])
    assert suite.main(argv, root=setup.root) == 1
    assert setup.config.read_bytes() == before
    assert not list(setup.root.glob("robot.json.bak.*"))


def test_run_uses_existing_wrapper_explicit_defaults_and_keeps_walk(setup, monkeypatch):
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: {})
    calls = []
    monkeypatch.setattr(suite.subprocess, "run", lambda args, **kw: (calls.append((args, kw)) or SimpleNamespace(returncode=0)))
    assert suite.main(["run"], root=setup.root) == 0
    assert calls == [(["bash", str(setup.root / "xbox_model_hd1910.sh"), "models/old.onnx", "10.0", "0.2"], {"cwd": setup.root})]
    after = json.loads(setup.config.read_text())
    assert after["model"] == setup.cfg["model"]
    assert after["control"]["roulade_enabled"] is False
    assert after["control"]["log_only_target_angles"] is True


def test_run_rejects_custom_config_wrapper_would_ignore(setup):
    custom = setup.root / "another.json"
    custom.write_bytes(setup.config.read_bytes())
    assert suite.main(["run", "--config", str(custom)], root=setup.root) == 1
    assert not list(setup.root.glob("*.bak.*"))


@pytest.mark.parametrize("arguments", [["--seconds", "-1"], ["--seconds", "2"], ["--seconds", "nan"], ["--max-vx", "0.5"]])
def test_run_rejects_invalid_cli_limits_before_write(setup, arguments):
    before = setup.config.read_bytes()
    assert suite.main(["run", *arguments], root=setup.root) == 1
    assert setup.config.read_bytes() == before


def test_motion_process_detection_uses_argv_tokens_and_ignores_self(tmp_path):
    for pid, argv in ((os.getpid(), ["python", "model_suite.py", "run"]),
                      (910011, ["python", "/x/policy_trial.py", "--move"]),
                      (910012, ["bash", "-c", "echo policy_trial.py"]),
                      (910013, ["python", "/x/model_suite.py", "check"]),
                      (910014, ["python", "/x/model_suite.py", "run"])):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(("\0".join(argv)+"\0").encode())
    assert {x["pid"] for x in suite.motion_processes(tmp_path)} == {910011, 910014}


def test_idle_refuses_active_systemd_before_port_read(setup, monkeypatch):
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self: True if str(self) == "/run/systemd/system" else original(self))
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="LoadState=loaded\nActiveState=active\n", stderr=""))
    monkeypatch.setattr(suite, "motion_processes", lambda: pytest.fail("active service should stop early"))
    with pytest.raises(RuntimeError, match="服务仍活动"):
        suite.assert_idle(setup.cfg)


def test_idle_refuses_occupied_serial_without_opening_it(setup, monkeypatch):
    original_exists, original_dir = Path.exists, Path.is_dir
    port = setup.cfg["serial"]["port"]
    monkeypatch.setattr(Path, "is_dir", lambda self: False if str(self) == "/run/systemd/system" else original_dir(self))
    monkeypatch.setattr(Path, "exists", lambda self: True if str(self) == port else original_exists(self))
    monkeypatch.setattr(suite, "motion_processes", lambda: [])
    monkeypatch.setattr(suite.shutil, "which", lambda name: "/usr/bin/fuser")
    monkeypatch.setattr(suite.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="1234", stderr=""))
    with pytest.raises(RuntimeError, match="串口被占用"):
        suite.assert_idle(setup.cfg)


def test_actual_six_packaged_onnx_models_match_saved_samples():
    # This is a real CPU contract/replay check; it never touches a robot config
    # or imports a hardware constructor. FakePolicy is only patched per fixture.
    result = suite.check_suite(root=SCRIPT.parents[1])
    assert result["status"] == "ok", result
    assert len(result["models"]) == 6
    assert sum(x["samples_count"] for x in result["models"].values()) >= 6
