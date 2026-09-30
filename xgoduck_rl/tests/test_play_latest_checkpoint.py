"""`play TASK` with no checkpoint flag must pick the latest local model_*.pt."""

from pathlib import Path

import pytest

import mjlab_microduck.tasks  # noqa: F401  — register task ids
from mjlab_microduck.play_hook import find_latest_checkpoint, maybe_inject_latest_checkpoint


def test_xgoduck_uses_tensorboard_not_wandb():
    from mjlab.tasks.registry import load_rl_cfg
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

    cfg = load_rl_cfg("Mjlab-Velocity-Flat-XgoDuck")
    assert cfg.logger == "tensorboard"
    assert cfg.experiment_name == "xgoduck_velocity"
    assert MicroduckRlCfg.logger == "wandb"
    assert load_rl_cfg("Mjlab-Velocity-Flat-MicroDuck").logger == "wandb"

    stand = load_rl_cfg("Mjlab-StandUp-Flat-XgoDuck")
    assert stand.logger == "tensorboard"
    assert stand.experiment_name == "xgoduck_standup"
    pick = load_rl_cfg("Mjlab-GroundPick-Flat-XgoDuck")
    assert pick.logger == "tensorboard"
    assert pick.experiment_name == "xgoduck_ground_pick"
    sit = load_rl_cfg("Mjlab-SitStand-Flat-XgoDuck")
    assert sit.logger == "tensorboard"
    assert sit.experiment_name == "xgoduck_sitstand"
    assert load_rl_cfg("Mjlab-SitStand-Flat-MicroDuck").experiment_name == "microduck_sitstand"


def test_xgoduck_sitstand_uses_xgoduck_robot_and_heights():
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from mjlab_microduck.robot.xgoduck_constants import XGODUCK_STAND_Z
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import SIT_Z, STAND_Z

    # Kinematic trunk z at the sitstand SIT keyframe, soles 1 mm above the floor.
    xgoduck_sit_z = 0.085868

    ids = list_tasks()
    assert "Mjlab-SitStand-Flat-XgoDuck" in ids
    assert "Mjlab-SitStand-Rough-XgoDuck" in ids

    cfg = load_env_cfg("Mjlab-SitStand-Flat-XgoDuck")
    robot = cfg.scene.entities["robot"]
    assert robot.init_state.pos[2] == pytest.approx(XGODUCK_STAND_Z)
    assert robot.collisions[0].conaffinity == {
        r"^(trunk_collision|head_collision)$": 2,
        ".*_collision": 0,
    }
    assert robot.collisions[0].contype == {
        r"^(trunk_collision|head_collision)$": 3,
        ".*_collision": 1,
    }

    twist = cfg.commands["twist"]
    assert twist.stand_z == pytest.approx(XGODUCK_STAND_Z)
    assert twist.sit_z == pytest.approx(xgoduck_sit_z)
    assert xgoduck_sit_z != SIT_Z
    assert XGODUCK_STAND_Z != STAND_Z

    r = cfg.rewards
    for name in (
        "posture_height",
        "posture_height_sharp",
        "posture_height_l1",
        "posture_stillness",
        "posture_composite",
    ):
        assert r[name].params["sit_z"] == pytest.approx(xgoduck_sit_z)
        assert r[name].params["stand_z"] == pytest.approx(XGODUCK_STAND_Z)
    assert r["rise_bootstrap"].params["max_height"] == pytest.approx(XGODUCK_STAND_Z + 0.010)
    assert r["upright_while_tall"].params["height_low"] == pytest.approx(xgoduck_sit_z + 0.015)
    assert r["upright_while_tall"].params["height_high"] == pytest.approx(XGODUCK_STAND_Z - 0.015)

    gs = cfg.events["set_ground_state"].params
    assert gs["sitting_z_min"] == pytest.approx(xgoduck_sit_z)
    assert gs["sitting_z_max"] == pytest.approx(xgoduck_sit_z + 0.015)
    assert gs["standing_z_min"] == pytest.approx(XGODUCK_STAND_Z - 0.005)
    assert gs["standing_z_max"] == pytest.approx(XGODUCK_STAND_Z + 0.005)
    z = cfg.events["reset_base"].params["pose_range"]["z"]
    assert z == pytest.approx((XGODUCK_STAND_Z - 0.005, XGODUCK_STAND_Z + 0.005))


def test_find_latest_checkpoint_picks_highest_iter_of_newest_run(tmp_path: Path):
    older = tmp_path / "2026-01-01_00-00-00_xgoduck"
    newer = tmp_path / "2026-09-03_12-00-00_xgoduck"
    older.mkdir()
    newer.mkdir()
    (older / "model_9999.pt").write_text("old")
    (newer / "model_250.pt").write_text("mid")
    (newer / "model_500.pt").write_text("latest")
    (tmp_path / "wandb_checkpoints").mkdir()
    (tmp_path / "wandb_checkpoints" / "model_99999.pt").write_text("cache")

    assert find_latest_checkpoint(tmp_path).name == "model_500.pt"


def test_inject_skips_when_checkpoint_or_wandb_given(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    argv = ["play", "Mjlab-Velocity-Flat-XgoDuck", "--wandb-run-path", "e/p/r"]
    assert maybe_inject_latest_checkpoint(argv) == argv

    argv = ["play", "Mjlab-Velocity-Flat-XgoDuck", "--checkpoint-file", "x.pt"]
    assert maybe_inject_latest_checkpoint(argv) == argv

    argv = ["play", "Mjlab-Velocity-Flat-XgoDuck", "--agent", "zero"]
    assert maybe_inject_latest_checkpoint(argv) == argv

    argv = ["play", "Mjlab-Velocity-Flat-XgoDuck", "--help"]
    assert maybe_inject_latest_checkpoint(argv) == argv


def test_inject_latest_pt(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "logs" / "rsl_rl" / "xgoduck_velocity" / "2026-09-03_12-00-00_xgoduck"
    run.mkdir(parents=True)
    (run / "model_0.pt").write_text("a")
    (run / "model_250.pt").write_text("b")

    out = maybe_inject_latest_checkpoint(["play", "Mjlab-Velocity-Flat-XgoDuck"])
    assert out[1] == "Mjlab-Velocity-Flat-XgoDuck"
    assert out[2] == "--checkpoint-file"
    assert Path(out[3]).name == "model_250.pt"


def test_inject_exits_when_no_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        maybe_inject_latest_checkpoint(["play", "Mjlab-Velocity-Flat-XgoDuck"])
