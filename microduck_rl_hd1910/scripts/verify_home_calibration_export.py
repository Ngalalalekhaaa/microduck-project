"""CPU-only contract and checkpoint/ONNX parity check for official HOME exports.

This validates an existing export; it neither exports a model nor constructs a
physics environment. Run after the registered runner's normalized ONNX export.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import inspect
import json
import math
from pathlib import Path


TASK = "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910"
NO_CURRENT_TASK = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"
FORCE_RATIO_TASK = "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910"
ACTOR_DIM = 392
ACTION_DIM = 14
SAMPLES = 128
TOLERANCE = 1e-4
ACTION_SEMANTICS = (
    "command_offset=clip(command_offset+clip(output,-1,1)*delta_limit_rad,"
    "-offset_limit_rad,offset_limit_rad)"
)


def validate_metadata(metadata, *, no_current=False, objective="pose"):
    force_ratio = objective == "force-ratio"
    no_current = no_current or force_ratio
    expected = {
        "task_id": FORCE_RATIO_TASK if force_ratio else (NO_CURRENT_TASK if no_current else TASK),
        "policy_family": ("hd1910_home_force_ratio_no_current_v1" if force_ratio else
                          ("hd1910_home_calibration_no_current_v2" if no_current else "hd1910_home_calibration_v1")),
        "actor_frame_dim": "48" if force_ratio else ("34" if no_current else "49"),
        "history_length": "8",
        "observation_normalization": "baked_into_onnx",
        "action_semantics": ACTION_SEMANTICS,
    }
    if no_current:
        expected.update({"current_input": "none", "load_signal": "none", "actor_dim": "384" if force_ratio else "272"})
        if any(key in metadata for key in ("current_normalization", "current_sign", "current_normalization_epsilon_a")):
            raise ValueError("No-current export must not contain current observation fields")
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Metadata {key!r}: expected {value!r}, got {metadata.get(key)!r}")
    for key in ("delta_limit_rad", "offset_limit_rad", "control_dt"):
        value = float(metadata[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Metadata {key!r} must be finite and positive")
    return {key: metadata[key] for key in (*expected, "delta_limit_rad", "offset_limit_rad", "control_dt")}


def validate_normalizer_statistics(mean, std, epsilon):
    import torch

    # A fixed desired ratio has exactly zero variance. rsl_rl divides by
    # std + eps, so zero std is valid; a negative/nonfinite std never is.
    if (not math.isfinite(epsilon) or epsilon <= 0
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all()
            or not (std >= 0).all() or not ((std + epsilon) > 0).all()):
        raise ValueError("Checkpoint normalizer statistics or denominator are invalid")


def verify(checkpoint_path, onnx_path, *, no_current=False, objective="pose"):
    import numpy as np
    import onnxruntime as ort
    import torch
    from mjlab.tasks.registry import load_rl_cfg
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    import mjlab_microduck.tasks  # noqa: F401 -- register this repository's task

    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    force_ratio = objective == "force-ratio"
    no_current = no_current or force_ratio
    task = FORCE_RATIO_TASK if force_ratio else (NO_CURRENT_TASK if no_current else TASK)
    actor_dim = 384 if force_ratio else (272 if no_current else ACTOR_DIM)
    cfg = load_rl_cfg(task)
    kwargs = asdict(cfg.actor)
    kwargs.pop("class_name")
    # Match the runner's MLP constructor; discard generic CNN/RNN defaults.
    parameters = inspect.signature(MLPModel.__init__).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    if not kwargs.get("obs_normalization"):
        raise ValueError("The registered HOME actor must use observation normalization")
    groups = {key: list(value) for key, value in cfg.obs_groups.items()}
    if groups["actor"] != ["actor"]:
        raise ValueError(f"Unexpected actor observation groups: {groups['actor']}")
    obs = TensorDict({"actor": torch.zeros((1, actor_dim))}, batch_size=[1])
    actor = MLPModel(obs, groups, "actor", ACTION_DIM, **kwargs).eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Expected exactly one ONNX input and one output")
    if inputs[0].shape != [1, actor_dim] or outputs[0].shape != [1, ACTION_DIM]:
        raise ValueError(f"Unexpected ONNX shapes: {inputs[0].shape} -> {outputs[0].shape}")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise ValueError("HOME export input/output must be float32")
    metadata = validate_metadata(session.get_modelmeta().custom_metadata_map,
                                 no_current=no_current, objective=objective)

    mean = actor.obs_normalizer._mean.detach()
    std = actor.obs_normalizer._std.detach()
    if mean.shape != (1, actor_dim) or std.shape != mean.shape:
        raise ValueError(f"Checkpoint normalizer does not match the {actor_dim}D actor")
    validate_normalizer_statistics(mean, std, float(actor.obs_normalizer.eps))
    generator = torch.Generator(device="cpu").manual_seed(90371)
    raw_samples = [torch.zeros_like(mean), mean.clone()]
    raw_samples.extend(
        mean + std * torch.randn(mean.shape, generator=generator)
        for _ in range(SAMPLES - 2)
    )
    errors = []
    with torch.inference_mode():
        for raw in raw_samples:
            expected = actor(TensorDict({"actor": raw}, batch_size=[1])).numpy()
            actual = session.run([outputs[0].name], {inputs[0].name: raw.numpy()})[0]
            if not (np.isfinite(expected).all() and np.isfinite(actual).all()):
                raise ValueError("Checkpoint or ONNX produced nonfinite output")
            errors.append(float(np.abs(expected - actual).max()))
    return {
        "passed": max(errors) <= TOLERANCE,
        "task": task,
        "execution_provider": "CPUExecutionProvider",
        "physics_environment_created": False,
        "samples": SAMPLES,
        "sample_distribution": "one zero, one saved mean, 126 saved mean + saved std * standard normal",
        "input_shape": [1, actor_dim],
        "output_shape": [1, ACTION_DIM],
        "max_abs_error": max(errors),
        "tolerance": TOLERANCE,
        "normalizer_statistics_nontrivial": bool((mean != 0).any() or (std != 1).any()),
        "observation_normalization": "Raw observations pass through both checkpoint and exported normalizers",
        "validated_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-current", action="store_true", help="Require the 272D encoder/IMU-only model contract")
    parser.add_argument("--objective", choices=("pose", "force-ratio"), default="pose")
    args = parser.parse_args()
    for path in (args.checkpoint, args.onnx):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    report = {"checkpoint": str(args.checkpoint.resolve()), "onnx": str(args.onnx.resolve())}
    try:
        report.update(verify(args.checkpoint, args.onnx, no_current=args.no_current, objective=args.objective))
    except Exception as exc:
        report.update({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("HOME export validation failed; inspect the saved report")


if __name__ == "__main__":
    main()
