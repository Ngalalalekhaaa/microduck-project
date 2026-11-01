"""CPU-only contract and normalized PPO/ONNX parity check after official export."""
import argparse
from dataclasses import asdict
import json
import inspect
from pathlib import Path

import mjlab.tasks
from mjlab.tasks.registry import load_rl_cfg
import numpy as np
import onnxruntime as ort
from rsl_rl.models import MLPModel
from tensordict import TensorDict
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    kwargs = asdict(load_rl_cfg(args.task).actor)
    kwargs.pop("class_name")
    valid_args = inspect.signature(MLPModel.__init__).parameters
    # RslRlModelCfg also carries CNN/RNN defaults even for an MLP; they are
    # intentionally unused by the same MLP class instantiated by the runner.
    kwargs = {k: v for k, v in kwargs.items() if k in valid_args}
    groups = {"actor": ["actor"]}
    obs = TensorDict({"actor": torch.zeros((1, 61))}, batch_size=[1])
    actor = MLPModel(obs, groups, "actor", 14, **kwargs).eval()
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 61]
    assert session.get_outputs()[0].shape == [1, 14]
    # Test raw inputs around the saved training distribution. This catches
    # missing normalization as well as network, dtype and output mismatches.
    generator = torch.Generator().manual_seed(90371)
    mean = actor.obs_normalizer._mean
    std = actor.obs_normalizer._std
    inputs = [torch.zeros((1, 61)), mean.clone()]
    inputs.extend(mean + std * torch.randn((1, 61), generator=generator) for _ in range(126))
    errors = []
    with torch.inference_mode():
        for raw in inputs:
            expected = actor(TensorDict({"actor": raw}, batch_size=[1])).numpy()
            actual = session.run(None, {session.get_inputs()[0].name: raw.numpy()})[0]
            assert np.isfinite(expected).all() and np.isfinite(actual).all()
            errors.append(float(np.abs(expected - actual).max()))
    report = {"passed": max(errors) < 1e-4, "samples": len(errors),
              "input_shape": [1, 61], "output_shape": [1, 14],
              "max_abs_error": max(errors), "observation_normalizer_included": True,
              "checkpoint": str(args.checkpoint), "onnx": str(args.onnx)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit("ONNX parity failed")


if __name__ == "__main__":
    main()
