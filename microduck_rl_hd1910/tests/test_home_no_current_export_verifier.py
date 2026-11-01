"""An old current-based export cannot pass the new model's export check."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "home_export_verifier_test", Path(__file__).parents[1] / "scripts/verify_home_calibration_export.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def metadata():
    return dict(task_id=verifier.NO_CURRENT_TASK, policy_family="hd1910_home_calibration_no_current_v2",
                actor_frame_dim="34", actor_dim="272", history_length="8", current_input="none",
                load_signal="none", observation_normalization="baked_into_onnx",
                action_semantics=verifier.ACTION_SEMANTICS, delta_limit_rad="0.004363323129985824",
                offset_limit_rad="0.13962634015954636", control_dt="0.05")


def test_no_current_metadata_requires_new_family_and_no_current_fields():
    assert verifier.validate_metadata(metadata(), no_current=True)["actor_dim"] == "272"
    with pytest.raises(ValueError):
        verifier.validate_metadata(metadata())
    for key, value in (("task_id", verifier.TASK), ("actor_dim", "392"),
                       ("current_input", "signed_current_fraction"),
                       ("current_normalization_epsilon_a", "0.02")):
        with pytest.raises(ValueError):
            verifier.validate_metadata({**metadata(), key: value}, no_current=True)


def test_force_ratio_export_cannot_be_mistaken_for_pose_export():
    force = {**metadata(), "task_id": verifier.FORCE_RATIO_TASK,
             "policy_family": "hd1910_home_force_ratio_no_current_v1",
             "actor_frame_dim": "48", "actor_dim": "384"}
    assert verifier.validate_metadata(force, objective="force-ratio")["actor_dim"] == "384"
    with pytest.raises(ValueError):
        verifier.validate_metadata(force, no_current=True)
    with pytest.raises(ValueError):
        verifier.validate_metadata(metadata(), objective="force-ratio")


def test_fixed_target_zero_variance_is_valid_but_invalid_normalizers_are_rejected():
    import torch

    mean, std = torch.zeros(1, 384), torch.ones(1, 384)
    for start in range(34, 384, 48):
        std[:, start:start + 14] = 0
    verifier.validate_normalizer_statistics(mean, std, .01)
    for invalid, epsilon in ((-torch.ones_like(std), .01), (std, 0),
                             (torch.full_like(std, float("nan")), .01)):
        with pytest.raises(ValueError):
            verifier.validate_normalizer_statistics(mean, invalid, epsilon)
