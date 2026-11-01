"""Cross-repository HOME sensor parity, with CPU tensors and fake hardware.

These tests exercise the actual training observation/history manager and the
deployment frame builder. They do not construct physics or access a serial port.
The training dependency is optional on the deployed computer; run this suite
with the existing training venv in the development workspace.
"""

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Cross-repository parity requires the training venv")
TRAINING_SOURCE = Path(__file__).resolve().parents[2] / "microduck_rl_hd1910" / "src"
if not TRAINING_SOURCE.is_dir():
    pytest.skip("The sibling HD1910 training source is required", allow_module_level=True)
_path = sys.path[:]
sys.path.insert(0, str(TRAINING_SOURCE))
try:
    pytest.importorskip("mjlab", reason="Cross-repository parity requires mjlab")
    from mjlab.managers.observation_manager import ObservationManager
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.home_calibration_runner import home_calibration_metadata
finally:
    sys.path[:] = _path

from microduck_deploy.config import calibration_arrays, default_config
from microduck_deploy import home_calibration_hardware as hardware
from microduck_deploy.home_calibration_hardware import RawCurrentFeedback
from microduck_deploy.home_calibration_policy import (
    HomeObservationHistory, build_home_frame34, build_home_frame48, validate_home_contract,
)
from microduck_deploy.home_calibration_runtime import run_session
from microduck_deploy.imu import ImuState
from microduck_deploy.policy import HOME, JOINT_NAMES, q_to_ticks, ticks_to_q


TASK = "Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910"


class RobotState:
    """Only the encoder and IMU state consumed by the real observation code."""

    def __init__(self, count):
        self.joint_names = list(JOINT_NAMES)
        self.data = SimpleNamespace(
            joint_pos=torch.tensor(HOME, dtype=torch.float32).repeat(count, 1),
            default_joint_pos=torch.tensor(HOME, dtype=torch.float32).repeat(count, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(count, 1),
            root_link_ang_vel_b=torch.zeros(count, 3),
        )

    def find_joints(self, patterns, preserve_order=False):
        patterns = [patterns] if isinstance(patterns, str) else patterns
        ids = [i for i, name in enumerate(self.joint_names)
               if any(re.fullmatch(pattern, name) for pattern in patterns)]
        return ids, [self.joint_names[index] for index in ids]

    @property
    def actuators(self):
        raise AssertionError("No-current observations must not access motor torque/current")


def training_rig(count=4, cfg=None):
    # The workspace's venv also contains an editable install of the original
    # robot; silently importing that source would make this a false parity test.
    assert Path(mdp.__file__).resolve().is_relative_to(TRAINING_SOURCE.resolve())
    cfg = load_env_cfg(TASK) if cfg is None else cfg
    robot = RobotState(count)
    env = SimpleNamespace(
        scene={"robot": robot}, num_envs=count, device="cpu", step_dt=.05,
        common_step_counter=0, episode_length_buf=torch.zeros(count, dtype=torch.long),
        extras={}, _imu_misalign_quat=torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1),
    )
    action = cfg.actions["home_calibration"].build(env)
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    env.observation_manager = ObservationManager(
        {"actor": deepcopy(cfg.observations["actor"])}, env)
    metadata = home_calibration_metadata(env, "parity/model_0.pt")
    contract = validate_home_contract(metadata)
    assert contract.history_length == 8
    expected = {TASK: (34, 272), "Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910": (48, 384)}
    assert (contract.actor_frame_dim, contract.actor_dim) == expected[contract.task_id]
    return env, robot, action, contract


@pytest.fixture
def forbid_current(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No-current training invoked its current proxy")

    monkeypatch.setattr(mdp, "home_calibration_motor_current_proxy", forbidden)


def deploy_frames(robot, action, contract):
    reported = (robot.data.joint_pos + action.bias).numpy()
    return [build_home_frame34(
        reported[index], robot.data.projected_gravity_b[index].numpy(),
        robot.data.root_link_ang_vel_b[index].numpy(), action.command_offset[index].numpy(),
        contract=contract,
    ) for index in range(len(reported))]


def test_random_frames_and_history_are_bit_exact_without_current(forbid_current):
    env, robot, action, contract = training_rig(count=16)
    rng = np.random.default_rng(1031)
    histories = [HomeObservationHistory(contract) for _ in range(env.num_envs)]
    for step in range(19):
        env.common_step_counter += 1
        action.bias.copy_(torch.from_numpy(rng.uniform(-.05, .05, (16, 14)).astype(np.float32)))
        robot.data.joint_pos.copy_(action.home + torch.from_numpy(
            rng.uniform(-.2, .2, (16, 14)).astype(np.float32)))
        gravity = rng.normal(size=(16, 3)).astype(np.float32)
        gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
        robot.data.projected_gravity_b.copy_(torch.from_numpy(gravity))
        robot.data.root_link_ang_vel_b.copy_(torch.from_numpy(
            rng.uniform(-2., 2., (16, 3)).astype(np.float32)))
        action.command_offset.copy_(torch.from_numpy(
            rng.uniform(-.13, .13, (16, 14)).astype(np.float32)))
        training = env.observation_manager.compute(update_history=True)["actor"].numpy()
        frames = deploy_frames(robot, action, contract)
        deployed = np.stack([history.push(frame) for history, frame in zip(histories, frames)])
        assert training.shape == deployed.shape == (16, 272)
        np.testing.assert_array_equal(training, deployed)
        np.testing.assert_array_equal(deployed.reshape(16, 8, 34)[:, -1, 14:17], gravity)
        np.testing.assert_array_equal(deployed.reshape(16, 8, 34)[:, -1, 17:20],
                                      robot.data.root_link_ang_vel_b.numpy())
        np.testing.assert_array_equal(deployed.reshape(16, 8, 34)[:, -1, 20:34],
                                      action.command_offset.numpy())

        # Calling an observation twice in one step must not advance or replace
        # the cached sensor frame used by the actor and privileged critic.
        repeated = env.observation_manager.compute(update_history=False)["actor"].numpy()
        np.testing.assert_array_equal(repeated, training)


def test_partial_reset_repeats_only_reset_world_and_preserves_others(forbid_current):
    env, robot, action, contract = training_rig(count=3)
    histories = [HomeObservationHistory(contract) for _ in range(3)]
    for step in range(11):
        env.common_step_counter += 1
        robot.data.root_link_ang_vel_b[:, 0] = step / 10
        actual = env.observation_manager.compute(update_history=True)["actor"].numpy()
        expected = np.stack([h.push(f) for h, f in zip(histories, deploy_frames(robot, action, contract))])
        np.testing.assert_array_equal(actual, expected)
    selected = torch.tensor([1])
    action.reset(selected)
    env.observation_manager.reset(selected)
    histories[1].reset()
    robot.data.joint_pos[1] += .03
    # No global step advance: resetting this world must invalidate its cache.
    actual = env.observation_manager.compute(update_history=True)["actor"].numpy()
    expected = np.stack([h.push(f) for h, f in zip(histories, deploy_frames(robot, action, contract))])
    np.testing.assert_array_equal(actual, expected)
    stacked = actual.reshape(3, 8, 34)
    np.testing.assert_array_equal(stacked[1], np.repeat(stacked[1, :1], 8, axis=0))
    assert not np.array_equal(stacked[0, 0], stacked[0, -1])


def test_force_target_is_constant_and_full_384_history_matches_deployment(tmp_path, forbid_current):
    from mjlab_microduck.tasks.microduck_home_force_ratio_env_cfg import make_microduck_home_force_ratio_env_cfg

    # A clearly synthetic fixture exercises the contract without blessing a
    # physical reference or requiring reference collection during a unit test.
    ratio = np.arange(1, 15, dtype=float)
    ratio /= ratio.sum()
    reference = tmp_path / "synthetic_unit_test_reference.json"
    reference.write_text(json.dumps({
        "schema_version": 1, "reference_kind": "absolute_joint_actuator_torque_fraction",
        "gate_passed": True, "joint_names": list(JOINT_NAMES),
        "home_angles_rad": HOME.astype(float).tolist(),
        "target_joint_torque_fraction": ratio.tolist(),
        "provenance": "Synthetic unit-test data; not a measured reference",
    }))
    cfg = make_microduck_home_force_ratio_env_cfg(reference_path=reference)
    env, robot, action, contract = training_rig(count=4, cfg=cfg)
    histories = [HomeObservationHistory(contract) for _ in range(4)]
    rng = np.random.default_rng(271)
    for step in range(11):
        env.common_step_counter += 1
        action.bias.copy_(torch.from_numpy(rng.uniform(-.04, .04, (4, 14)).astype(np.float32)))
        robot.data.joint_pos.copy_(action.home + torch.from_numpy(
            rng.uniform(-.1, .1, (4, 14)).astype(np.float32)))
        robot.data.root_link_ang_vel_b.copy_(torch.from_numpy(
            rng.uniform(-1., 1., (4, 3)).astype(np.float32)))
        action.command_offset.copy_(torch.from_numpy(
            rng.uniform(-.12, .12, (4, 14)).astype(np.float32)))
        actual = env.observation_manager.compute(update_history=True)["actor"].numpy()
        reported = (robot.data.joint_pos + action.bias).numpy()
        frames = [build_home_frame48(
            reported[index], robot.data.projected_gravity_b[index].numpy(),
            robot.data.root_link_ang_vel_b[index].numpy(), action.command_offset[index].numpy(),
            contract=contract,
        ) for index in range(4)]
        expected = np.stack([history.push(frame) for history, frame in zip(histories, frames)])
        assert actual.shape == (4, 384)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.reshape(4, 8, 48)[:, :, 34:],
                                      np.broadcast_to(ratio.astype(np.float32), (4, 8, 14)))


def test_monitor_uses_measured_goal_offset_and_never_integrates_suggested_actions(
    tmp_path, monkeypatch, forbid_current,
):
    _, _, _, contract = training_rig(count=1)
    cfg = default_config()
    cfg["calibration"] = {"pose": "home"}
    cfg["imu_mount_verified"] = True
    for index, joint in enumerate(cfg["joints"]):
        joint.update(id=index + 1, direction=(-1 if index % 2 else 1),
                     zero_tick=2048.25 + index * .125, mapping_verified=True)
    zero, direction = calibration_arrays(cfg)
    goals = q_to_ticks(HOME + np.linspace(-.04, .04, 14), zero, direction)
    positions = goals + np.array([2, -3] * 7)
    measured_offset = ticks_to_q(goals, zero, direction) - np.asarray(contract.home_angles_rad)
    clock = SimpleNamespace(now=100.)
    now = lambda: clock.now
    monkeypatch.setattr(hardware, "time", SimpleNamespace(monotonic=now))
    reads, observations, writes = [], [], []

    def feedback(ids, sync):
        assert ids == list(range(1, 15)) and sync
        reads.append(now())
        # Current polarity/magnitude change independently; neither is an actor
        # input. All sensor timestamps and hardware status remain valid.
        return {sid: RawCurrentFeedback(
            servo_id=sid, position_ticks=int(positions[index]), velocity_raw=0,
            load_raw=0, voltage_v=7.4, temperature_c=30, status=0, moving=0,
            goal_ticks=int(goals[index]), current_ma=(-1) ** len(reads) * 13.,
            received_at=now(), current_register_raw=0x8002 if len(reads) % 2 else 2,
        ) for index, sid in enumerate(ids)}

    def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("A monitor must not write any servo register")

    bus = SimpleNamespace(
        read_feedback_many=feedback, read_configuration=lambda sid: {"torque": 1},
        sync_positions=forbidden_write, set_torque=forbidden_write,
    )
    imu = SimpleNamespace(latest=lambda max_age: ImuState(
        now(), np.array([.1, -.2, .3]), np.array([0., 0., -1.]), np.array([0., 0., 9.80665])))

    def infer(observation):
        observations.append(observation.copy())
        return np.ones(14, dtype=np.float32)

    policy = SimpleNamespace(contract=contract, infer=infer,
                             inspect=lambda: {"input_shape": [1, 272], "output_shape": [1, 14]})
    report = run_session(
        cfg, policy, bus, imu, None, mode="monitor", seconds=.2,
        report_path=tmp_path / "monitor.json", clock=now,
        sleep=lambda seconds: setattr(clock, "now", now() + seconds),
    )
    assert report["completed"] and report["goal_write_attempts"] == 0
    assert len(observations) == 4 and writes == []
    expected = build_home_frame34(
        ticks_to_q(positions, zero, direction), [0., 0., -1.], [.1, -.2, .3],
        measured_offset, contract=contract,
    )
    for observation in observations:
        np.testing.assert_array_equal(observation.reshape(8, 34), np.repeat(expected[None], 8, axis=0))
        np.testing.assert_allclose(observation.reshape(8, 34)[-1, 20:], measured_offset, atol=1e-8)
    # Sensor position and goal are deliberately different: the known command
    # must come from the goal register, not the measured encoder or suggestions.
    assert not np.allclose(expected[:14], expected[20:])
