"""The D-pad launcher enables all four models without losing the base model."""
import copy
import json
from types import SimpleNamespace

import pytest

from test_model_suite import setup, suite


def four_config(setup):
    cfg = copy.deepcopy(setup.cfg)
    cfg["sitstand_model"] = setup.catalog["models"]["sitstand_original"]["path"]
    cfg["ground_pick_model"] = setup.catalog["models"]["ground_pick"]["path"]
    setup.config.write_text(json.dumps(cfg))
    return cfg


def test_four_action_preview_preserves_base_models_and_has_no_writes(setup, monkeypatch, capsys):
    cfg = four_config(setup)
    before = setup.config.read_bytes()
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: pytest.fail("preview must not inspect devices"))
    assert suite.main(["select", "--four-actions"], root=setup.root) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["mode"] == "preview" and out["four_actions"] is True
    assert "model" not in out["changes"] and "sitstand_model" not in out["changes"]
    assert out["changes"]["control.roulade_enabled"]["after"] is True
    assert out["changes"]["control.model_suite_buttons_enabled"]["after"] is True
    assert before == setup.config.read_bytes()
    selected = suite.selection(cfg, setup.catalog, four_actions=True)
    assert selected["model"] == cfg["model"]
    assert selected["sitstand_model"] == cfg["sitstand_model"]
    assert selected["ground_pick_model"] == cfg["ground_pick_model"]


def test_four_action_apply_keeps_calibration_and_stores_all_slots(setup, monkeypatch):
    cfg = four_config(setup)
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: {})
    assert suite.main(["select", "--four-actions", "--apply"], root=setup.root) == 0
    after = json.loads(setup.config.read_text())
    for key in ("joints", "calibration", "imu", "imu_mount_verified", "serial", "extra_robot_settings",
                "model", "sitstand_model", "ground_pick_model"):
        assert after[key] == cfg[key]
    for key, name in (("updated_walking_model", "walk"), ("sitstand_friction_model", "sitstand_friction"),
                      ("sitstand_com_model", "sitstand_com"), ("roulade_model", "roulade")):
        assert after[key] == setup.catalog["models"][name]["path"]
    assert after["control"]["model_suite_buttons_enabled"] is True
    assert suite.load_config(setup.config)["control"]["roulade_enabled"] is True
    assert len(list(setup.root.glob("robot.json.bak.*"))) == 1


def test_four_action_run_uses_original_walk_and_existing_hardware_gate(setup, monkeypatch):
    cfg = four_config(setup)
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: {})
    calls = []
    monkeypatch.setattr(suite.subprocess, "run", lambda args, **kw:
                        (calls.append((args, kw)) or SimpleNamespace(returncode=0)))
    assert suite.main(["run", "--four-actions", "--seconds", "60"], root=setup.root) == 0
    assert calls[0][0][2:] == [cfg["model"], "60.0", "0.2"]
    after = suite.load_config(setup.config)
    assert after["control"]["model_suite_buttons_enabled"] is True


@pytest.mark.parametrize("extra", [["--walking", "new"], ["--sitstand", "friction"], ["--roulade", "off"]])
def test_conflicting_four_action_options_fail_before_write(setup, extra):
    four_config(setup)
    before = setup.config.read_bytes()
    assert suite.main(["select", "--four-actions", "--apply", *extra], root=setup.root) == 1
    assert before == setup.config.read_bytes()


def test_regular_selection_disables_suite_before_turning_roll_off(setup):
    cfg = four_config(setup)
    enabled = suite.selection(cfg, setup.catalog, four_actions=True)
    normal = suite.selection(enabled, setup.catalog)
    assert normal["control"]["model_suite_buttons_enabled"] is False
    assert normal["control"]["roulade_enabled"] is False
    setup.config.write_text(json.dumps(normal))
    assert suite.load_config(setup.config)["control"]["model_suite_buttons_enabled"] is False


def test_missing_existing_sitstand_model_fails_before_apply(setup, monkeypatch):
    cfg = four_config(setup)
    cfg["sitstand_model"] = "models/absent.onnx"
    setup.config.write_text(json.dumps(cfg))
    before = setup.config.read_bytes()
    monkeypatch.setattr(suite, "assert_idle", lambda cfg: pytest.fail("invalid model must fail first"))
    assert suite.main(["select", "--four-actions", "--apply"], root=setup.root) == 1
    assert before == setup.config.read_bytes()


def test_native_padd_and_robotd_processes_are_mutually_exclusive(tmp_path):
    for pid, argv in ((901111, ["/usr/bin/padd", "--hz", "50"]),
                      (901112, ["/opt/microduck/bin/robotd"]),
                      (901113, ["bash", "-c", "echo padd"]),
                      (901114, ["python", "-m", "pytest", "-k", "robotd"])):
        folder = tmp_path / str(pid)
        folder.mkdir()
        (folder / "cmdline").write_bytes(("\0".join(argv)+"\0").encode())
    assert {p['pid'] for p in suite.motion_processes(tmp_path)} == {901111, 901112}


def test_missing_native_services_are_allowed_but_active_native_service_refuses(setup, monkeypatch):
    from pathlib import Path
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self:
                        True if str(self) == "/run/systemd/system" else original(self))
    monkeypatch.setattr(suite, "motion_processes", lambda: [])
    monkeypatch.setattr(suite.subprocess, "run", lambda argv, **kw: SimpleNamespace(
        returncode=1, stdout="" if argv[0] == "fuser" else "LoadState=not-found\nActiveState=inactive\n", stderr=""))
    assert len(suite.assert_idle(setup.cfg)["services"]) == 4

    def systemctl(argv, **kw):
        if argv[0] == "fuser":
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if "padd.service" in argv:
            return SimpleNamespace(returncode=0, stdout="LoadState=loaded\nActiveState=active\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="LoadState=not-found\nActiveState=inactive\n", stderr="")
    monkeypatch.setattr(suite.subprocess, "run", systemctl)
    with pytest.raises(RuntimeError, match="padd.service"):
        suite.assert_idle(setup.cfg)


def test_not_found_but_active_service_is_still_rejected(setup, monkeypatch):
    from pathlib import Path
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self:
                        True if str(self) == "/run/systemd/system" else original(self))
    monkeypatch.setattr(suite.subprocess, "run", lambda *args, **kw: SimpleNamespace(
        returncode=1, stdout="LoadState=not-found\nActiveState=active\n", stderr=""))
    with pytest.raises(RuntimeError, match="服务仍活动"):
        suite.assert_idle(setup.cfg)


def test_fuser_uses_port_argument_supported_by_psmisc(setup, monkeypatch):
    from pathlib import Path
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda self:
                        False if str(self) == "/run/systemd/system" else original(self))
    monkeypatch.setattr(suite, "motion_processes", lambda: [])
    monkeypatch.setattr(suite.shutil, "which", lambda name: "/usr/bin/fuser")
    calls = []
    def fuser(argv, **kw):
        calls.append(argv)
        assert argv == ["fuser", setup.cfg["serial"]["port"]]
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    monkeypatch.setattr(suite.subprocess, "run", fuser)
    suite.assert_idle(setup.cfg)
    assert len(calls) == 1
