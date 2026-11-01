"""Recover the supplied 1699 ONNX actor without inventing a source critic.

The ONNX folded ``std + eps`` into the Div initializer. Restore that exact
denominator and verify deterministic parity before any PPO update. ONNX has
no optimizer, critic, exploration std or normalizer sample count; those are
new training state. The synthetic count is an explicit normalization prior.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from tensordict import TensorDict
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

INIT_ONNX_ENV = "HD1910_SITSTAND_INIT_ONNX"
NORMALIZER_PRIOR_COUNT = 10_000_000


def restore_onnx_actor(actor, source):
    source = Path(source)
    graph = onnx.load(str(source)).graph
    expected_ops = ["Sub", "Div", "Gemm", "Elu", "Gemm", "Elu", "Gemm", "Elu", "Gemm"]
    if [n.op_type for n in graph.node] != expected_ops:
        raise ValueError("Unexpected source graph; refusing to guess its actor contract")
    arrays = {v.name: numpy_helper.to_array(v).copy() for v in graph.initializer}
    denominator = arrays[graph.node[1].input[1]]
    mean = arrays[graph.node[0].input[1]]
    normalizer = actor.obs_normalizer
    if mean.shape != (1, 61) or denominator.shape != mean.shape:
        raise ValueError("Source must contain the normalized 61D deployment actor")
    std = denominator - normalizer.eps
    if not np.isfinite(std).all() or std.min() < -1.e-7:
        raise ValueError("Invalid source normalization denominator")
    state = actor.state_dict()
    for key in state:
        if key.startswith("mlp."):
            value = arrays[key]
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"Source actor shape mismatch: {key}")
            state[key] = torch.as_tensor(value, device=state[key].device)
    for key, value in (("_mean", mean), ("_std", std.clip(0)), ("_var", std.clip(0) ** 2)):
        state[f"obs_normalizer.{key}"] = torch.as_tensor(value, device=state[f"obs_normalizer.{key}"].device)
    state["obs_normalizer.count"].fill_(NORMALIZER_PRIOR_COUNT)
    actor.load_state_dict(state, strict=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape != [1, 61] or session.get_outputs()[0].shape != [1, 14]:
        raise ValueError("Source ONNX I/O contract differs")
    rng = np.random.default_rng(1699)
    inputs = [np.zeros((1, 61), dtype=np.float32), mean]
    inputs += [(mean + denominator * rng.normal(size=(1, 61))).astype(np.float32)
               for _ in range(126)]
    was_training = actor.training
    actor.eval()
    errors = []
    with torch.inference_mode():
        for raw in inputs:
            obs = torch.as_tensor(raw, device=next(actor.parameters()).device)
            actual = actor(TensorDict({"actor": obs}, batch_size=[1])).cpu().numpy()
            expected = session.run(None, {session.get_inputs()[0].name: raw})[0]
            errors.append(float(np.abs(actual - expected).max()))
    actor.train(was_training)
    if not np.isfinite(errors).all() or max(errors) >= 1.e-4:
        raise RuntimeError(f"Source actor restoration failed parity: {max(errors)}")
    return {"passed": True, "source": str(source.resolve()),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "samples": len(inputs), "max_abs_error": max(errors),
            "actor_weights_restored": True, "normalizer_restored": True,
            "normalizer_synthetic_prior_count": NORMALIZER_PRIOR_COUNT,
            "critic_restored": False, "optimizer_restored": False,
            "iteration_and_curriculum_restart": True}


class HD1910SitStandRobustRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        source = os.environ.get(INIT_ONNX_ENV)
        if source:
            if train_cfg.get("resume"):
                raise ValueError("Choose either ONNX warm start or checkpoint resume")
            report = restore_onnx_actor(self.alg.actor, source)
            self.env.unwrapped.common_step_counter = 0
            self.current_learning_iteration = 0
            if log_dir:
                path = Path(log_dir)
                path.mkdir(parents=True, exist_ok=True)
                (path / "actor_warm_start.json").write_text(json.dumps(report, indent=2) + "\n")
                self.save(str(path / "warm_start.pt"), infos={"actor_warm_start": report})
            print("[SitStand ONNX warm start] " + json.dumps(report), flush=True)
