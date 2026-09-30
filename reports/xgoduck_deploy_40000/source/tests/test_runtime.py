"""Hardware-free safety and timing tests for configuration and foreground control."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from microduck_deploy import config, runtime
from microduck_deploy.imu import ImuState
from microduck_deploy.policy import HOME, JOINT_LIMITS, q_to_ticks, ticks_to_q
from microduck_deploy.servo import ServoFeedback


def calibrated_config():
    cfg = config.default_config()
    for i, joint in enumerate(cfg["joints"]):
        joint.update(direction=1 if i % 2 == 0 else -1, mapping_verified=True)
    config.record_reference(cfg, np.arange(14) * 7 + 1900, "home", HOME)
    cfg["imu_mount_verified"] = True
    cfg["control"]["home_duration_s"] = 0.06
    return cfg


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += max(0.0, seconds)


class TimedEvent:
    def __init__(self, clock):
        self.clock = clock
        self.flag = False
        self.waits = []

    def set(self):
        self.flag = True

    def is_set(self):
        return self.flag

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if not self.flag:
            self.clock.advance(timeout or 0)
        return self.flag


class Bus:
    def __init__(self, cfg, clock):
        self.clock = clock
        self.cfg = cfg
        self.ids = [j["id"] for j in cfg["joints"]]
        self.zero, self.direction = config.calibration_arrays(cfg)
        self.positions = dict(zip(self.ids, q_to_ticks(HOME, self.zero, self.direction).tolist()))
        self.events = []
        self.read_count = 0
        self.fail_read_at = None
        self.fail_read_after_enable = False
        self.read_s = 0.001
        self.fail_enable = False
        self.disable_failures = 0
        self.initial_torque = 0
        self.mode = 4
        self.angle_resolution = 1
        self.bad_goal_readback = False
        self.closed = False

    def read_configuration(self, sid):
        self.events.append(("configuration", self.clock.now, sid))
        return {"torque": self.initial_torque, "mode": self.mode,
                "angle_resolution": self.angle_resolution,
                "ram_gains": {"p": 6, "d": 20, "i": 0}}

    def read_feedback_many(self, ids, sync):
        self.assert_ids(ids)
        self.read_count += 1
        self.clock.advance(self.read_s)
        self.events.append(("read", self.clock.now))
        if self.read_count == self.fail_read_at or (
            self.fail_read_after_enable and any(e[0] == "torque" and e[2] for e in self.events)
        ):
            raise OSError("mock serial cable failure")
        return {sid: ServoFeedback(sid, self.positions[sid], 0, 0, 7.4, 30, 0, 0,
                                   self.positions[sid], 0, self.clock.now) for sid in ids}

    def assert_ids(self, ids):
        if list(ids) != self.ids:
            raise AssertionError("servo order changed")

    def sync_positions(self, positions):
        self.clock.advance(0.001)
        self.events.append(("positions", self.clock.now, dict(positions)))
        self.positions.update(positions)

    def read(self, sid, address, length):
        if (address, length) != (42, 2):
            raise AssertionError("unexpected raw register read")
        self.events.append(("goal_readback", self.clock.now, sid))
        return int(self.positions[sid] + int(self.bad_goal_readback)).to_bytes(2, "little")

    def set_torque(self, ids, enable, verify):
        self.assert_ids(ids)
        self.events.append(("torque", self.clock.now, enable, verify))
        if enable and self.fail_enable:
            raise OSError("mock partial torque-enable failure")
        if not enable and self.disable_failures:
            self.disable_failures -= 1
            raise OSError("mock torque-disable timeout")

    def set_pid_ram(self, ids, p, d, i):
        self.events.append(("gains", self.clock.now, list(ids), (p, d, i)))

    def close(self):
        self.closed = True
        self.events.append(("close", self.clock.now))


class Imu:
    def __init__(self, clock):
        self.clock = clock
        self.calls = 0
        self.fail_at = None
        self.fail_when = lambda: False
        self.gravity = np.array([0.0, 0.0, -1.0])
        self.closed = False
        self.close_error = None

    def latest(self, max_age):
        self.calls += 1
        if self.calls == self.fail_at or self.fail_when():
            raise runtime.StopRequested("mock IMU stale")
        return ImuState(self.clock.now, np.zeros(3), self.gravity.copy(), np.array([0, 0, 9.81]))

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class Policy:
    def __init__(self, clock):
        self.clock = clock
        self.observations = []
        self.times = []
        self.actions = []
        self.inference_s = 0.001
        self.output = np.zeros(14, dtype=np.float32)
        self.fail = None

    def infer(self, obs):
        self.times.append(self.clock.now)
        self.observations.append(obs.copy())
        self.clock.advance(self.inference_s)
        if self.fail:
            raise self.fail
        result = self.output.copy()
        self.actions.append(result)
        return result


@contextmanager
def rig(*, owned=False, cfg=None):
    cfg = calibrated_config() if cfg is None else cfg
    clock = Clock()
    bus, imu, policy = Bus(cfg, clock), Imu(clock), Policy(clock)
    ctl = runtime.Controller(cfg, bus=None if owned else bus,
                             imu_reader=None if owned else imu, policy=policy)
    ctl.stop = TimedEvent(clock)
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        stack.enter_context(patch.object(runtime.time, "monotonic", clock.monotonic))
        stack.enter_context(patch.object(runtime, "log_path", return_value=Path(directory) / "run.jsonl"))
        stack.enter_context(patch.object(ctl, "_install_signals"))
        stack.enter_context(patch.object(runtime.sys, "stdin", SimpleNamespace(isatty=lambda: False)))
        stack.enter_context(patch.object(runtime.threading, "Thread"))  # no background/hardware threads
        factory = stack.enter_context(patch.object(runtime, "ServoBus", return_value=bus))
        stack.enter_context(patch.object(runtime, "ImuReader", return_value=imu))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        yield SimpleNamespace(cfg=cfg, clock=clock, bus=bus, imu=imu, policy=policy,
                              ctl=ctl, factory=factory, path=Path(directory) / "run.jsonl")


class CalibrationConfigTests(unittest.TestCase):
    def test_template_does_not_assume_servo_zero_or_mapping(self):
        cfg = config.default_config()
        self.assertIsNone(cfg["calibration"])
        self.assertTrue(all(j["zero_tick"] is None and j["direction"] is None
                            and not j["mapping_verified"] for j in cfg["joints"]))
        with self.assertRaises(ValueError):
            config.require_calibration(cfg)

    def test_home_capture_formula_preserves_reference_with_mixed_directions(self):
        cfg = calibrated_config()
        ticks = np.arange(14) * 7 + 1900
        zero, direction = config.calibration_arrays(cfg)
        np.testing.assert_allclose(zero, ticks - direction * HOME * 4096 / (2 * np.pi))
        np.testing.assert_allclose(ticks_to_q(ticks, zero, direction), HOME, atol=1e-15)
        np.testing.assert_array_equal(q_to_ticks(HOME, zero, direction), ticks)
        self.assertFalse(np.allclose(zero, 2048))

    def test_bad_capture_is_rejected_before_mutating_calibration(self):
        for ticks, reference in (
            (np.full(14, 1900.5), HOME), (np.full(14, -1), HOME),
            (np.full(14, 4096), HOME), (np.full(14, np.nan), HOME),
            (np.full(14, 1900), np.full(14, np.nan)),
            (np.full(14, 1900), np.full(14, np.inf)),
        ):
            cfg = calibrated_config()
            original = copy.deepcopy(cfg)
            with self.subTest(ticks=ticks[0], reference=reference[0]), self.assertRaises(ValueError):
                config.record_reference(cfg, ticks, "home", reference)
            self.assertEqual(cfg, original)

    def test_calibration_cannot_put_home_across_single_turn_boundary(self):
        cfg = calibrated_config()
        original = copy.deepcopy(cfg)
        with self.assertRaises(ValueError):
            config.record_reference(cfg, np.zeros(14), "zero", np.zeros(14))
        self.assertEqual(cfg, original)

    def test_motion_needs_mapping_calibration_and_verified_imu(self):
        for mutate in (
            lambda c: c.update(calibration=None),
            lambda c: c.update(imu_mount_verified=False),
            lambda c: c["joints"][0].update(mapping_verified=False),
            lambda c: c["joints"][0].update(direction=0),
            lambda c: c["joints"][0].update(zero_tick=float("nan")),
        ):
            cfg = calibrated_config()
            mutate(cfg)
            with self.assertRaises(ValueError):
                config.require_calibration(cfg)
        cfg = calibrated_config()
        cfg["imu_mount_verified"] = False
        config.require_calibration(cfg, imu=False)

    def test_axes_reject_mirror_and_duplicate_axes(self):
        np.testing.assert_array_equal(config.axes_rotation(["+y", "-x", "+z"]),
                                      [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        for axes in (["+x", "+y", "-z"], ["+x", "+x", "+z"], ["x", "y", "z"]):
            with self.assertRaises(ValueError):
                config.axes_rotation(axes)

    def test_load_rejects_changed_joint_order_rate_or_nonfinite_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for mutate in (
                lambda c: c["joints"].reverse(),
                lambda c: c["joints"][0].update(id=c["joints"][1]["id"]),
                lambda c: c["control"].update(frequency_hz=100),
                lambda c: c["control"].update(max_cycle_s=float("nan")),
            ):
                cfg = calibrated_config()
                mutate(cfg)
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    config.load_config(path)


class RuntimeGuardTests(unittest.TestCase):
    def test_feedback_faults_missing_stale_status_voltage_temperature_and_multiturn(self):
        with rig() as r:
            feedback = r.bus.read_feedback_many(r.bus.ids, sync=True)
            runtime.check_feedback(feedback, r.cfg)
            sid = r.bus.ids[0]
            for change in ({"received_at": r.clock.now - 1}, {"status": 4},
                           {"voltage_v": 6.0}, {"temperature_c": 60}, {"position_ticks": 4096}):
                bad = dict(feedback)
                bad[sid] = dataclasses.replace(bad[sid], **change)
                with self.subTest(change=change), self.assertRaises(runtime.StopRequested):
                    runtime.check_feedback(bad, r.cfg)
            del feedback[sid]
            with self.assertRaises(runtime.StopRequested):
                runtime.check_feedback(feedback, r.cfg)

    def test_target_rejection_never_clips_and_enforces_jump_and_encoder_range(self):
        with rig() as r:
            target = HOME.copy()
            np.testing.assert_array_equal(runtime.validate_targets(target, HOME, r.cfg),
                                          list(r.bus.positions.values()))
            target[0] = JOINT_LIMITS[0, 1] + 0.1
            before = target.copy()
            with self.assertRaises(runtime.StopRequested):
                runtime.validate_targets(target, HOME, r.cfg)
            np.testing.assert_array_equal(target, before)
            with self.assertRaises(runtime.StopRequested):
                runtime.validate_targets(HOME, HOME + 1.0, r.cfg)
            with self.assertRaises(runtime.StopRequested):
                runtime.validate_targets(np.full(14, np.nan), HOME, r.cfg)
            r.cfg["joints"][0]["zero_tick"] = 5000
            with self.assertRaises(ValueError):
                runtime.validate_targets(HOME, HOME, r.cfg)

    def test_posture_rejects_nonfinite_nonunit_and_tilted_input(self):
        cfg = calibrated_config()
        for gravity in ([0, 0, 0], [0, 0, -9.81], [0, 0, np.nan], [1, 0, 0]):
            state = SimpleNamespace(projected_gravity=np.array(gravity), base_ang_vel=np.zeros(3))
            with self.subTest(gravity=gravity), self.assertRaises((runtime.StopRequested, ValueError)):
                runtime.check_posture(HOME, state, cfg)
        with self.assertRaises((runtime.StopRequested, ValueError)):
            runtime.check_posture(np.full(14, np.nan),
                                  SimpleNamespace(projected_gravity=[0, 0, -1], base_ang_vel=np.zeros(3)), cfg)

    def test_imu_reader_rejects_stale_failed_and_nonfinite_state(self):
        reader = runtime.ImuReader.__new__(runtime.ImuReader)
        reader.lock = threading.Lock()
        reader.error = None
        reader.state = ImuState(10, np.zeros(3), np.array([0, 0, -1]), np.zeros(3))
        with patch.object(runtime.time, "monotonic", return_value=10.01):
            self.assertIs(reader.latest(0.06), reader.state)
            reader.state = dataclasses.replace(reader.state, timestamp=9.0)
            with self.assertRaises(runtime.StopRequested):
                reader.latest(0.06)
            reader.state = dataclasses.replace(reader.state, timestamp=10, base_ang_vel=np.full(3, np.nan))
            with self.assertRaises(runtime.StopRequested):
                reader.latest(0.06)
            reader.error = OSError("mock disconnected IMU")
            with self.assertRaises(runtime.StopRequested):
                reader.latest(0.06)


class ControllerTests(unittest.TestCase):
    def test_construction_and_read_only_never_write_motor_registers(self):
        with rig() as r:
            self.assertEqual(r.bus.events, [])
            r.cfg["imu_mount_verified"] = False
            result = r.ctl.execute(seconds=0.11, motion=False)
            self.assertGreater(result["samples"], 2)
            self.assertTrue(all(event[0] == "read" for event in r.bus.events))
            self.assertFalse(r.ctl.enabled)
            self.assertFalse(r.bus.closed)  # injected resources remain caller-owned

    def test_uncalibrated_or_unverified_motion_fails_before_opening_hardware(self):
        for field in ("calibration", "imu_mount_verified"):
            with rig(owned=True) as r:
                r.cfg[field] = None if field == "calibration" else False
                with self.assertRaises(ValueError):
                    r.ctl.execute(seconds=0.1, motion=True)
                r.factory.assert_not_called()
                self.assertEqual(r.bus.events, [])

    def test_home_sequence_loads_current_goal_before_torque_then_smoothly_reaches_home(self):
        with rig() as r:
            start = HOME.copy()
            start[0] += 0.12
            r.bus.positions = dict(zip(r.bus.ids, q_to_ticks(start, r.bus.zero, r.bus.direction).tolist()))
            initial_ticks = dict(r.bus.positions)
            r.ctl.execute(seconds=0.08, motion=True, home_only=True)
            events = r.bus.events
            enabled = next(i for i, event in enumerate(events) if event[0] == "torque" and event[2])
            writes_before = [event for event in events[:enabled] if event[0] == "positions"]
            self.assertEqual(len(writes_before), 1)
            self.assertEqual(writes_before[0][2], initial_ticks)
            self.assertTrue(any(event[0] == "gains" and event[3] == (5, 0, 0) for event in events[:enabled]))
            home_writes = [event for event in events[enabled + 1:] if event[0] == "positions"][:3]
            q0 = [ticks_to_q(event[2][r.bus.ids[0]], r.bus.zero[0], r.bus.direction[0]) for event in home_writes]
            self.assertTrue(all(a >= b for a, b in zip(q0, q0[1:])))
            self.assertLess(abs(q0[-1] - HOME[0]), 0.002)
            self.assertEqual(r.policy.times, [])  # HOME is independent of policy inference.
            hold_write = [e for e in events[enabled + 1:] if e[0] == "positions"][3]
            self.assertGreaterEqual(hold_write[1] - events[enabled][1], r.cfg["control"]["home_duration_s"])
            self.assertTrue(all(np.all(obs[34:48] == 0) for obs in r.policy.observations))
            self.assertFalse([event for event in events if event[0] == "torque"][-1][2])
            self.assertEqual(sum(event[0] == "gains" and event[3] == (6, 20, 0) for event in events), 14)

    def test_observation_uses_previous_raw_policy_output_and_full_period_velocity(self):
        with rig() as r:
            r.policy.output = np.linspace(-0.02, 0.02, 14, dtype=np.float32)
            r.ctl.execute(seconds=0.11, motion=True)
            self.assertGreater(len(r.policy.observations), 2)
            np.testing.assert_array_equal(r.policy.observations[0][34:48], np.zeros(14))
            for obs in r.policy.observations[1:]:
                np.testing.assert_array_equal(obs[34:48], r.policy.output)
            # A zero wait after the initial 20ms priming period used to run the
            # second cycle immediately, distorting qd and the policy frequency.
            np.testing.assert_allclose(np.diff(r.policy.times), 0.02, atol=1e-9)
            samples = [json.loads(line) for line in r.path.read_text().splitlines()][1:]
            self.assertGreaterEqual(samples[0]["t"], 0.02)
            np.testing.assert_allclose(samples[0]["qd"], np.zeros(14), atol=1e-10)

    def test_bad_start_torque_or_mode_prevents_enabling(self):
        for issue in ("existing_torque", "wrong_mode", "far_pose", "tilt"):
            with self.subTest(issue=issue), rig() as r:
                if issue == "existing_torque":
                    r.bus.initial_torque = 1
                elif issue == "wrong_mode":
                    r.bus.mode = 0
                elif issue == "far_pose":
                    start = HOME.copy()
                    start[2] -= 0.7
                    r.bus.positions = dict(zip(r.bus.ids, q_to_ticks(start, r.bus.zero, r.bus.direction).tolist()))
                else:
                    r.imu.gravity = np.array([1.0, 0, 0])
                with self.assertRaises(runtime.StopRequested):
                    r.ctl.execute(seconds=0.1, motion=True)
                self.assertFalse(any(event[0] == "torque" and event[2] for event in r.bus.events))

    def test_goal_readback_failure_prevents_torque_enable(self):
        with rig() as r:
            r.bus.bad_goal_readback = True
            with self.assertRaises(runtime.StopRequested):
                r.ctl.execute(seconds=0.1, motion=True)
            self.assertFalse(any(event[0] == "torque" and event[2] for event in r.bus.events))

    def test_first_goal_write_failure_is_already_inside_unload_protection(self):
        with rig() as r:
            def failed_preload(positions):
                # HD firmware may energize on a goal write even while torque was
                # reported off. Cleanup protection must precede this first write.
                self.assertTrue(r.ctl.enabled)
                raise OSError("mock partial first-goal write")

            r.bus.sync_positions = failed_preload
            with self.assertRaises(OSError):
                r.ctl.execute(seconds=0.1, motion=True)
            self.assertEqual([e[2] for e in r.bus.events if e[0] == "torque"], [False])
            self.assertFalse(r.ctl.enabled)

    def test_stop_during_preload_verification_prevents_later_explicit_enable(self):
        with rig() as r:
            original_read = r.bus.read

            def request_stop_on_readback(sid, address, length):
                r.ctl.stop.set()  # e.g. watchdog or termination signal
                return original_read(sid, address, length)

            r.bus.read = request_stop_on_readback
            with self.assertRaises(runtime.StopRequested):
                r.ctl.execute(seconds=0.1, motion=True)
            self.assertEqual([e[2] for e in r.bus.events if e[0] == "torque"], [False])

    def test_fresh_feedback_failure_before_enable_restores_gains_without_torque(self):
        with rig() as r:
            r.bus.fail_read_at = 2  # after PID writes, before holding/enabling
            with self.assertRaises(OSError):
                r.ctl.execute(seconds=0.1, motion=True)
            self.assertFalse(any(event[0] in ("torque", "positions") for event in r.bus.events))
            self.assertEqual(sum(event[0] == "gains" and event[3] == (6, 20, 0)
                                 for event in r.bus.events), 14)

    def test_home_feedback_deadline_failure_does_not_send_late_home_target(self):
        with rig() as r:
            original_read = r.bus.read_feedback_many

            def slow_after_enable(ids, sync):
                if any(e[0] == "torque" and e[2] for e in r.bus.events):
                    r.bus.read_s = 0.06
                return original_read(ids, sync)

            r.bus.read_feedback_many = slow_after_enable
            with self.assertRaises(runtime.StopRequested):
                r.ctl.execute(seconds=0.1, motion=True)
            enabled = next(i for i, event in enumerate(r.bus.events) if event[0] == "torque" and event[2])
            self.assertFalse(any(event[0] == "positions" for event in r.bus.events[enabled + 1:]))
            self.assertFalse([event for event in r.bus.events if event[0] == "torque"][-1][2])

    def test_stop_request_prevents_further_goal_writes(self):
        with rig() as r:
            r.ctl.stop.set()
            with self.assertRaises(runtime.StopRequested):
                r.ctl._send(HOME, HOME)
            self.assertEqual(r.bus.events, [])

    def test_serial_or_imu_failure_after_enable_always_unloads_and_restores_gains(self):
        for source in ("serial", "imu"):
            with self.subTest(source=source), rig() as r:
                if source == "serial":
                    r.bus.fail_read_after_enable = True
                else:
                    r.imu.fail_when = lambda: any(e[0] == "torque" and e[2] for e in r.bus.events)
                with self.assertRaises((OSError, runtime.StopRequested)):
                    r.ctl.execute(seconds=0.1, motion=True)
                torque = [event[2] for event in r.bus.events if event[0] == "torque"]
                self.assertEqual(torque, [True, False])
                self.assertFalse(r.ctl.enabled)
                self.assertTrue(r.ctl.stop.is_set())
                self.assertEqual(sum(event[0] == "gains" and event[3] == (6, 20, 0)
                                     for event in r.bus.events), 14)

    def test_partial_enable_failure_also_unloads_with_retries(self):
        with rig() as r:
            r.bus.fail_enable = True
            r.bus.disable_failures = 2
            with self.assertRaises(OSError):
                r.ctl.execute(seconds=0.1, motion=True)
            torque = [(event[2], event[3]) for event in r.bus.events if event[0] == "torque"]
            self.assertEqual(torque, [(True, True), (False, True), (False, False), (False, False)])
            self.assertFalse(r.ctl.enabled)

    def test_slow_policy_is_rejected_before_its_target_is_written(self):
        with rig() as r:
            r.policy.inference_s = 0.06
            r.policy.output[0] = 0.1
            with self.assertRaises(runtime.StopRequested):
                r.ctl.execute(seconds=0.2, motion=True)
            first_policy_time = r.policy.times[0]
            self.assertFalse(any(event[0] == "positions" and event[1] > first_policy_time
                                 for event in r.bus.events))
            self.assertFalse([event for event in r.bus.events if event[0] == "torque"][-1][2])

    def test_consecutive_overruns_stop_and_unload(self):
        with rig() as r:
            r.cfg["control"]["max_overruns"] = 2
            r.policy.inference_s = 0.025
            with self.assertRaisesRegex(runtime.StopRequested, "20ms"):
                r.ctl.execute(seconds=0.5, motion=True)
            self.assertEqual(len(r.policy.times), 2)
            self.assertFalse([event for event in r.bus.events if event[0] == "torque"][-1][2])

    def test_policy_failure_closes_owned_resources_after_unloading(self):
        with rig(owned=True) as r:
            r.policy.fail = ValueError("mock nonfinite inference")
            with self.assertRaises(ValueError):
                r.ctl.execute(seconds=0.1, motion=True)
            self.assertTrue(r.bus.closed)
            self.assertTrue(r.imu.closed)
            unload = next(i for i, event in enumerate(r.bus.events) if event[0] == "torque" and not event[2])
            close = next(i for i, event in enumerate(r.bus.events) if event[0] == "close")
            self.assertLess(unload, close)

    def test_imu_close_failure_does_not_skip_bus_cleanup(self):
        with rig(owned=True) as r:
            r.imu.close_error = OSError("mock imu close failure")
            try:
                r.ctl.execute(seconds=0.08, motion=True)
            except OSError:
                pass  # Either propagation or reporting is acceptable; bus must close.
            self.assertTrue(r.bus.closed)
            self.assertFalse([event for event in r.bus.events if event[0] == "torque"][-1][2])

    def test_watchdog_attempts_unload_after_heartbeat_timeout(self):
        with rig() as r:
            r.ctl.enabled = True
            r.ctl.heartbeat = r.clock.now - 0.16
            r.ctl._watchdog()
            self.assertTrue(r.ctl.stop.is_set())
            self.assertIn("150ms", r.ctl.reason)
            self.assertEqual([event[2:] for event in r.bus.events if event[0] == "torque"], [(False, False)])


if __name__ == "__main__":
    unittest.main()
