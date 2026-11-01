"""Exercise terminal-state accounting and the deployment encoder stop boundary."""
import importlib.util
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluate_hd1910_robust.py"
SPEC = importlib.util.spec_from_file_location("hd1910_robust_evaluation", MODULE_PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def calibration():
    return {"zero": np.full(14, 2048.), "direction": np.ones(14),
            "bounds": np.tile([-1., 1.], (14, 1)), "max_jump": .6}


def capture(stats, time, velocity, *, tilt=0., terminated=False, target=0.):
    count = stats.count
    velocity = np.broadcast_to(np.asarray(velocity), (count, 3)).copy()
    gravity = np.broadcast_to([np.sin(np.deg2rad(tilt)), 0., -np.cos(np.deg2rad(tilt))], (count, 3)).copy()
    q = np.zeros((count, 14))
    targets = np.full((count, 14), target)
    return stats.capture(time, velocity, gravity, q, q, targets, targets,
        np.full(count, terminated), np.zeros(count, dtype=bool),
        np.ones(count, dtype=bool), {"fell_over": np.full(count, terminated)})


def test_failure_frame_excluded_from_tracking_and_reset_world_never_reenters():
    stats = evaluation.RolloutStats(1, (.2, 0., 0.), calibration(),
                                    np.tile([-1., 1.], (14, 1)), 30., 1., 45.)
    capture(stats, .5, [.0, .0, .0])  # warmup contributes no tracking
    capture(stats, 1.5, [.15, .0, .0])
    capture(stats, 2., [99., 99., 99.], tilt=60., terminated=True, target=4.)
    capture(stats, 3., [.2, .0, .0])  # would look healthy after a simulator reset
    result = stats.result()
    assert result["valid_tracking_samples"] == 1
    np.testing.assert_allclose(result["mean_abs_error_vx_vy_wz"], [.05, 0., 0.])
    assert result["survival_fraction"] == 0.
    assert result["first_failure_kind"] == ["fall_forward"]
    assert result["first_failure_time_s"] == [2.]
    assert result["max_tilt_deg"] == pytest.approx(60.)
    # The terminal target is the unsafe one; a post-reset HOME must not hide it.
    assert result["max_abs_target_rad"] == 4.
    assert result["target_encoder_proxy_violation_frames"] == 1
    assert result["encoder_proxy_first_violation_s"] == [2.]


def test_encoder_boundary_matches_nearest_integer_single_turn_counts():
    # Direction -1 must move toward the lower boundary for positive targets.
    target = np.array([[0., 2. * np.pi / 4096.]])
    outside = evaluation.encoder_outside(target, np.array([.49, .49]), np.array([-1., -1.]))
    np.testing.assert_array_equal(outside, [[False, True]])
    # A fractional zero above 4095 still rounds to a legal count until 4095.5.
    outside = evaluation.encoder_outside(np.zeros((1, 2)), np.array([4095.49, 4095.51]), np.ones(2))
    np.testing.assert_array_equal(outside, [[False, True]])


def test_all_failed_before_warmup_keeps_missing_tracking_visible():
    stats = evaluation.RolloutStats(1, (.2, 0., 0.), calibration(),
                                    np.tile([-1., 1.], (14, 1)), 30., 1., 45.)
    stats.mark_failure(np.array([True]), 0., "nonfinite_action")
    result = stats.result()
    assert result["mean_abs_error_vx_vy_wz"] is None
    assert result["mean_first_episode_seconds"] == 0.
    aggregate = evaluation.aggregate_results([evaluation.json_value(result)])
    assert aggregate["tracking_runs"] == 0
    assert aggregate["tracking_mae_normalized"] is None
    assert aggregate["survival_fraction"] == 0.


def test_two_seeds_are_averaged_without_rewarding_longer_lived_sample_counts():
    results = []
    for actual_vx, lifetime in ((.15, 1), (.05, 3)):
        stats = evaluation.RolloutStats(1, (.2, 0., 0.), calibration(),
                                        np.tile([-1., 1.], (14, 1)), 30., 0., 45.)
        for time in range(1, lifetime + 1):
            capture(stats, float(time), [actual_vx, 0., 0.])
        results.append(evaluation.json_value(stats.result()))
    aggregate = evaluation.aggregate_results(results)
    assert aggregate["mean_abs_error_vx_vy_wz"] == pytest.approx([.10, 0., 0.])
    assert aggregate["tracking_runs"] == 2


def test_csv_preserves_missing_metrics_and_joint_names(tmp_path):
    stats = evaluation.RolloutStats(1, (.2, 0., 0.), calibration(),
                                    np.tile([-1., 1.], (14, 1)), 30., 1., 45.)
    stats.mark_failure(np.array([True]), 0., "nonfinite_action")
    result = evaluation.json_value(stats.result())
    result.update(profile="stress", seed=12345, command_name="forward20")
    names = [f"joint_{i}" for i in range(14)]
    evaluation.write_csv_reports(tmp_path, {"results": [result]}, {"names": names})
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert rows[0]["survival_fraction"] == "0.0"
    assert rows[0]["mae_vx"] == ""  # zero would imply perfect tracking
    joints = list(csv.DictReader((tmp_path / "joint_metrics.csv").open()))
    assert [row["joint_name"] for row in joints] == names
    assert all(row["measured_min_rad"] == "" for row in joints)


@pytest.mark.parametrize("profile", ["nominal", "stress"])
def test_robust_fixed_commands_survive_reset_sampler(monkeypatch, profile):
    """Run the real command reset path on CPU, including zero-range axes."""
    source = MODULE_PATH.parents[1] / "microduck_rl_hd1910/src"
    monkeypatch.syspath_prepend(str(source))
    import torch
    from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnlyCfg
    cfg = evaluation.make_eval_cfg("Mjlab-Velocity-Flat-Backlash-MicroDuck-HD1910-Robust",
                                   profile, 12345, 8, 2.)
    command_cfg = cfg.commands["twist"]
    assert type(command_cfg) is VelocityCommandCommandOnlyCfg
    data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(8, 3),
                           root_link_ang_vel_b=torch.zeros(8, 3))
    env = SimpleNamespace(num_envs=8, device="cpu", step_dt=.02,
                          scene={"robot": SimpleNamespace(data=data)})
    term = command_cfg.build(env)
    for command in evaluation.COMMANDS.values():
        for field, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command):
            setattr(term.cfg.ranges, field, (value, value))
        term.reset(torch.arange(8))
        term.compute(.02)
        torch.testing.assert_close(term.command, torch.tensor(command).expand(8, 3))
