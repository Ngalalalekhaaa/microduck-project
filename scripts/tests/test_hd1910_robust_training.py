"""Check checkpoint continuation labels and rejection of stationary 'best' policies."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_hd1910_robust as train


def profiles(error=0., survival=1.):
    commands = ("forward10", "forward20", "lateral20", "lateral_minus20", "turn100", "turn_minus100")
    aggregate = {"tracking_runs": 6, "runs": 6, "survival_fraction": survival,
                 "tracking_mae_normalized": error,
                 "target_encoder_proxy_violation_fraction": 0.,
                 "results": {name: {"mean_abs_error_vx_vy_wz": [error] * 3} for name in commands}}
    return {"nominal": deepcopy({"aggregate": aggregate}), "stress": deepcopy({"aggregate": aggregate})}


def test_resume_range_finishes_at_requested_checkpoint_label():
    # RSL-RL learn() runs range(saved_iter, saved_iter + N), saves last index.
    updates = train.updates_to_label(5000, 6000)
    assert list(range(5000, 5000 + updates))[-1] == 6000
    with pytest.raises(ValueError):
        train.updates_to_label(6000, 6000)


def test_imported_names_and_generated_names_have_same_saved_counter():
    assert train.iteration(Path("model_5000_backlash.pt")) == 5000
    assert train.iteration(Path("model_6000.pt")) == 6000
    with pytest.raises(ValueError):
        train.iteration(Path("model_6000_wrong.pt"))


def test_stationary_survival_does_not_pass_walking_screen(tmp_path):
    stationary = profiles()
    for profile in stationary.values():
        profile["aggregate"]["results"]["forward20"]["mean_abs_error_vx_vy_wz"][0] = .2
    train.record_ranking(tmp_path, "backlash", Path("model_6000.pt"), stationary)
    report = json.loads((tmp_path / "candidate_ranking.json").read_text())
    assert report["best_evaluated"]["backlash"]["worst_profile_survival_fraction"] == 1.
    assert not report["best_evaluated"]["backlash"]["passes_simulation_screen"]
    train.record_ranking(tmp_path, "backlash", Path("model_7000.pt"), profiles(.01, .99))
    report = json.loads((tmp_path / "candidate_ranking.json").read_text())
    assert report["best_evaluated"]["backlash"]["iteration"] == 7000
    assert report["best_evaluated"]["backlash"]["passes_simulation_screen"]
    assert not report["best_evaluated"]["backlash"]["hardware_tested"]


def test_early_failed_runs_with_missing_tracking_cannot_pass(tmp_path):
    missing = profiles()
    missing["stress"]["aggregate"]["tracking_runs"] = 5
    train.record_ranking(tmp_path, "flat", Path("model_6000.pt"), missing)
    report = json.loads((tmp_path / "candidate_ranking.json").read_text())
    assert not report["best_evaluated"]["flat"]["passes_simulation_screen"]
