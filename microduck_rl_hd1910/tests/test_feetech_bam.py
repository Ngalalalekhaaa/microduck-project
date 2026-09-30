"""CPU checks of the real Feetech limiter and the combined backlash adapter.

Only the Warp allocation/friction bridge is replaced by CPU stubs. The JSON,
BAM Feetech voltage law, command extraction and all adapter methods are real.
GPU physics integration is covered separately by test_hd1910_rollout.py.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from bam.actuator import TorchBackend
from bam.mjlab import BamActuator

from mjlab_microduck.actuator.feetech_bam import (
    FeetechBacklashBamActuatorCfg,
    FeetechBamActuatorCfg,
)
from mjlab_microduck.tasks.microduck_hd1910_env_cfg import HD1910_M6_JSON


@pytest.fixture(params=[False, True], ids=["plain", "backlash"])
def actuator(request, monkeypatch):
    def initialize_cpu(self, mj_model, model, data, device):
        self._target_ids = torch.tensor(self._target_ids_list)
        self._prev_motor_torque = torch.zeros(data.nworld, len(self._target_names))
        self.kp_scale = torch.ones(data.nworld, 1)
        self._bam_model.actuator.backend = TorchBackend()

    def compute_voltage(self, cmd):
        return self._bam_model.actuator.compute_control(
            cmd.position_target, cmd.pos, cmd.vel, 0.005
        )

    monkeypatch.setattr(BamActuator, "initialize", initialize_cpu)
    monkeypatch.setattr(BamActuator, "compute", compute_voltage)
    cls = FeetechBacklashBamActuatorCfg if request.param else FeetechBamActuatorCfg
    cfg = cls(json_path=str(HD1910_M6_JSON), target_names_expr=("a", "b"), kp_fw=5.0)
    entity = SimpleNamespace(joint_names=["a", "passive_a_backlash", "b", "passive_b_backlash"])
    adapter = cfg.build(entity, [0, 2], ["a", "b"])
    adapter.initialize(None, None, SimpleNamespace(nworld=3), "cpu")
    data = SimpleNamespace(
        joint_pos=torch.tensor([[1.2, 0.01, -1.4, -0.02]]).repeat(3, 1),
        joint_vel=torch.tensor([[0.3, 4.0, -0.2, -5.0]]).repeat(3, 1),
        joint_pos_target=torch.zeros(3, 4),
        joint_vel_target=torch.zeros(3, 4),
        joint_effort_target=torch.zeros(3, 4),
    )
    return adapter, data, request.param


def test_feedback_uses_encoder_position_and_motor_velocity(actuator):
    adapter, data, backlash = actuator
    cmd = adapter.get_command(data)
    expected = data.joint_pos[:, [0, 2]].clone()
    if backlash:
        expected += data.joint_pos[:, [1, 3]]
        assert adapter._backlash_mask.tolist() == [1.0, 1.0]
    torch.testing.assert_close(cmd.pos, expected)
    torch.testing.assert_close(cmd.vel, data.joint_vel[:, [0, 2]])


def test_first_compute_seeds_nonzero_pose_and_keeps_limiter_history(actuator):
    adapter, data, _ = actuator
    cmd = adapter.get_command(data)
    holding = replace(cmd, position_target=cmd.pos.clone())
    torch.testing.assert_close(adapter.compute(holding), torch.zeros_like(cmd.pos))
    torch.testing.assert_close(adapter._bam_model.actuator.q_target_smooth, cmd.pos)
    assert not adapter._goal_reset_pending.any()

    # The published max_velocity=100 produces a 0.5 rad step limit at 5 ms.
    target = replace(cmd, position_target=cmd.pos + 2.0)
    assert torch.isfinite(adapter.compute(target)).all()
    torch.testing.assert_close(adapter._bam_model.actuator.q_target_smooth, cmd.pos + 0.5)
    adapter.compute(target)
    torch.testing.assert_close(adapter._bam_model.actuator.q_target_smooth, cmd.pos + 1.0)


@pytest.mark.parametrize("selection", [None, slice(1, 2), torch.tensor([1])], ids=["all", "slice", "tensor"])
def test_reset_reads_new_pose_lazily_and_preserves_other_worlds(actuator, selection):
    adapter, data, _ = actuator
    cmd = adapter.get_command(data)
    adapter.compute(replace(cmd, position_target=cmd.pos.clone()))
    previous = cmd.pos + 2.0
    adapter._bam_model.actuator.q_target_smooth = previous.clone()
    adapter._prev_motor_torque.fill_(1.0)
    adapter.reset(selection)
    selected = slice(None) if selection is None else selection
    # Reset events change the pose AFTER actuator.reset(), including backlash.
    data.joint_pos[selected] += 0.2
    fresh = adapter.get_command(data)
    voltage = adapter.compute(replace(fresh, position_target=fresh.pos.clone()))
    expected = previous - 0.5
    expected[selected] = fresh.pos[selected]
    torch.testing.assert_close(adapter._bam_model.actuator.q_target_smooth, expected)
    torch.testing.assert_close(voltage[selected], torch.zeros_like(voltage[selected]))
    assert torch.all(adapter._prev_motor_torque[selected] == 0)
    if selection is not None:
        assert torch.all(adapter._prev_motor_torque[[0, 2]] == 1)
    assert torch.isfinite(voltage).all()
