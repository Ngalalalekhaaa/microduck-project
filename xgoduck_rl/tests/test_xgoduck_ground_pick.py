"""XGODUCK ground-pick: descent gravity_x / |gy|, no joint-pose reward."""

import math

import torch
from mjlab.tasks.registry import load_env_cfg

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    DESCENT_END,
    HOLD_END,
    RISE_END,
    make_microduck_ground_pick_env_cfg,
)

# Mid-hold: down-gate = 1 with the segmented GP profile.
_HOLD_PHASE = 0.5 * (DESCENT_END + HOLD_END)
_REST_PHASE = 0.9
# send_conf crouch, statically balanced: trunk pitch ≈ 33° → gx ≈ 0.54
_MAX_FORWARD = 0.65


class _CmdMgr:
    def __init__(self, phase: float):
        ang = 2.0 * math.pi * phase
        self._cmd = torch.tensor([[math.cos(ang), math.sin(ang), 0.0]])

    def get_command(self, _name):
        return self._cmd


class _GravityEnv:
    def __init__(self, gx: float, phase: float, gy: float = 0.0):
        data = type("D", (), {})()
        data.projected_gravity_b = torch.tensor([[gx, gy, -1.0]])
        asset = type("A", (), {})()
        asset.data = data
        self.scene = {"robot": asset}
        self.command_manager = _CmdMgr(phase)


def _gx(gx: float, phase: float, gy: float = 0.0) -> torch.Tensor:
    return microduck_mdp.ground_pick_gravity_x_phased(
        _GravityEnv(gx, phase, gy),
        command_name="twist",
        max_forward=_MAX_FORWARD,
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
    )


def _gy_abs(gy: float, phase: float) -> torch.Tensor:
    return microduck_mdp.ground_pick_gravity_y_abs_phased(
        _GravityEnv(0.0, phase, gy),
        command_name="twist",
        descent_end=DESCENT_END,
        hold_end=HOLD_END,
        rise_end=RISE_END,
    )


def test_gravity_x_rewards_measured_forward_lean():
    r = _gx(0.54, _HOLD_PHASE)
    assert torch.allclose(r, torch.tensor([0.54]), atol=1e-6), r


def test_gravity_x_includes_new_gate_boundary():
    r = _gx(0.65, _HOLD_PHASE)
    assert torch.allclose(r, torch.tensor([0.65]), atol=1e-6), r


def test_gravity_x_gates_off_past_max_forward():
    r = _gx(0.66, _HOLD_PHASE)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6), r


def test_gravity_x_penalizes_backward_lean():
    r = _gx(-0.2, _HOLD_PHASE)
    assert torch.allclose(r, torch.tensor([-0.2]), atol=1e-6), r


def test_gravity_x_zero_when_not_descending():
    r = _gx(0.54, _REST_PHASE)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6), r


def test_gravity_y_abs_is_abs_during_descent():
    r = _gy_abs(-0.3, _HOLD_PHASE)
    assert torch.allclose(r, torch.tensor([0.3]), atol=1e-6), r


def test_gravity_y_abs_zero_when_not_descending():
    r = _gy_abs(0.3, _REST_PHASE)
    assert torch.allclose(r, torch.tensor([0.0]), atol=1e-6), r


def test_microduck_ground_pick_has_no_xgoduck_gravity_terms():
    cfg = make_microduck_ground_pick_env_cfg()
    assert "hip_yaw_roll_hold" not in cfg.rewards
    assert "descent_gravity_x" not in cfg.rewards
    assert "descent_gravity_y_abs" not in cfg.rewards
    assert "descent_pick_pose" not in cfg.rewards
    assert cfg.rewards["mouth_ground_proximity"].params["std"] == 0.10
    assert abs(cfg.rewards["neck_vel_descent"].weight - (-0.1)) < 1e-9


def test_xgoduck_ground_pick_wires_gravity_xy_and_pick_pose():
    for task_id in (
        "Mjlab-GroundPick-Flat-XgoDuck",
        "Mjlab-GroundPick-Rough-XgoDuck",
    ):
        cfg = load_env_cfg(task_id)
        assert "hip_yaw_roll_home" not in cfg.rewards

        hold = cfg.rewards["hip_yaw_roll_hold"]
        assert hold.func is microduck_mdp.joint_deviation_l1
        assert abs(hold.weight - (-7.0)) < 1e-9

        gx = cfg.rewards["descent_gravity_x"]
        assert gx.func is microduck_mdp.ground_pick_gravity_x_phased
        assert abs(gx.weight - 1.0) < 1e-9
        assert abs(gx.params["max_forward"] - 0.65) < 1e-9

        gy = cfg.rewards["descent_gravity_y_abs"]
        assert gy.func is microduck_mdp.ground_pick_gravity_y_abs_phased
        assert abs(gy.weight - (-1.0)) < 1e-9

        assert abs(cfg.rewards["mouth_ground_proximity"].params["std"] - 0.04) < 1e-9
        assert abs(cfg.rewards["neck_vel_descent"].weight - (-0.01)) < 1e-9

        pose = cfg.rewards["descent_pick_pose"]
        assert pose.func is microduck_mdp.phase_pose_track
        assert abs(pose.weight - 1.0) < 1e-9
        assert abs(pose.params["std"] - 0.4) < 1e-9
        tp = pose.params["target_pose"]
        assert abs(tp["neck_pitch"] - math.radians(-120)) < 1e-9
        assert abs(tp["head_pitch"] - math.radians(-90)) < 1e-9
        assert abs(tp["left_hip_pitch"] - math.radians(-80)) < 1e-9
        assert abs(tp["right_hip_pitch"] - math.radians(80)) < 1e-9


def test_neck_pitch_soft_limit_includes_send_conf_minus_120():
    """Hard min -135° so 0.9 soft still contains the real-robot -120° nod."""
    import mujoco
    from pathlib import Path

    xml = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/xgoduck/robot_walk.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "neck_pitch")
    lo, hi = model.jnt_range[jid]
    target = math.radians(-120)
    soft_lo = 0.95 * lo + 0.05 * hi
    assert lo <= math.radians(-135) + 1e-9
    assert soft_lo <= target


def test_head_pitch_soft_limit_includes_pick_minus_90():
    import mujoco
    from pathlib import Path

    xml = Path(__file__).resolve().parents[1] / "src/mjlab_microduck/robot/xgoduck/robot_walk.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "head_pitch")
    lo, hi = model.jnt_range[jid]
    target = math.radians(-90)
    soft_lo = 0.95 * lo + 0.05 * hi
    assert soft_lo <= target
