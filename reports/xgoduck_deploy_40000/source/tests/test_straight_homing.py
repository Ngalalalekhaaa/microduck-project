"""Straight-reference homing checks with independent goals and measured feedback.

Uses the existing fake clock and controller rig; no hardware, ONNX session, or
GPU is opened. Encoder feedback is deliberately separate from written goals.
"""

from contextlib import contextmanager
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from microduck_deploy import config, homing, runtime
from microduck_deploy.policy import (
    HOME, STRAIGHT_REFERENCE, RADIANS_PER_TICK, q_to_ticks, ticks_to_q,
)
from test_runtime import rig


CAPTURED = np.arange(14, dtype=int) * 11 + 1700


def straight_config():
    cfg = config.default_config()
    for index, joint in enumerate(cfg["joints"]):
        joint.update(direction=1 if index % 2 == 0 else -1, mapping_verified=True)
    config.record_reference(cfg, CAPTURED, "straight", STRAIGHT_REFERENCE)
    cfg["imu_mount_verified"] = True
    return cfg


@contextmanager
def straight_rig(*, cfg=None, response="follow"):
    """Plant advances only on reads; a target write never changes measured q."""
    cfg = straight_config() if cfg is None else cfg
    with rig(cfg=cfg) as r:
        bus = r.bus
        bus.positions = dict(zip(bus.ids, CAPTURED.tolist()))
        bus.goals = dict(bus.positions)
        bus.start_q = ticks_to_q(CAPTURED, bus.zero, bus.direction)
        bus.enabled_reads = 0
        original_config = bus.read_configuration
        original_feedback = bus.read_feedback_many

        def read_configuration(sid):
            return {**original_config(sid), "position_min": 0, "position_max": 4095}

        def sync_positions(targets):
            r.clock.advance(0.001)
            bus.events.append(("positions", r.clock.now, dict(targets)))
            bus.goals.update(targets)

        def read(sid, address, length):
            if (address, length) != (42, 2):
                raise AssertionError("unexpected raw register read")
            bus.events.append(("goal_readback", r.clock.now, sid))
            return int(bus.goals[sid] + int(bus.bad_goal_readback)).to_bytes(2, "little")

        def read_feedback_many(ids, sync):
            armed = any(event[0] == "torque" and event[2] for event in bus.events)
            if armed:
                bus.enabled_reads += 1
                goal_q = ticks_to_q([bus.goals[sid] for sid in ids], bus.zero, bus.direction)
                if response == "follow":
                    bus.positions.update(bus.goals)
                elif response == "blocked":
                    pass
                elif response == "reverse":
                    q = goal_q.copy()
                    q[3] = bus.start_q[3] - (goal_q[3] - bus.start_q[3])
                    bus.positions.update(zip(ids, q_to_ticks(q, bus.zero, bus.direction).tolist()))
                elif response == "jump":
                    q = goal_q.copy()
                    q[3] += 0.20
                    bus.positions.update(zip(ids, q_to_ticks(q, bus.zero, bus.direction).tolist()))
                else:
                    raise AssertionError(response)
            measured = original_feedback(ids, sync)
            return {sid: replace(value, goal_ticks=bus.goals[sid])
                    for sid, value in measured.items()}

        bus.read_configuration = read_configuration
        bus.sync_positions = sync_positions
        bus.read = read
        bus.read_feedback_many = read_feedback_many
        yield r


def execute_straight(r, **kwargs):
    return r.ctl.execute(seconds=0.08, motion=True, home_only=True,
                         from_pose="straight", **kwargs)


def motor_writes(r):
    return [e for e in r.bus.events if e[0] in ("positions", "torque")]


class StraightCalibrationTests(unittest.TestCase):
    def test_capture_recovers_nonzero_reference_with_mixed_directions(self):
        cfg = straight_config()
        zero, direction = config.calibration_arrays(cfg)
        np.testing.assert_allclose(ticks_to_q(CAPTURED, zero, direction),
                                   STRAIGHT_REFERENCE, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(q_to_ticks(STRAIGHT_REFERENCE, zero, direction), CAPTURED)
        self.assertGreater(np.max(np.abs(STRAIGHT_REFERENCE - HOME)), 0.6)
        self.assertFalse(np.allclose(zero, CAPTURED))
        self.assertEqual(cfg["calibration"]["joint_ids"], [j["id"] for j in cfg["joints"]])
        self.assertEqual(cfg["calibration"]["directions"], direction.tolist())
        homing.validate_reference_calibration(cfg, "straight")

    def test_capture_rejects_both_single_turn_boundaries_without_mutating_config(self):
        cfg = straight_config()
        _, direction = config.calibration_arrays(cfg)
        delta_ticks = direction * (HOME - STRAIGHT_REFERENCE) / RADIANS_PER_TICK
        for index, tick in ((int(np.flatnonzero(delta_ticks > 0)[0]), 4095),
                            (int(np.flatnonzero(delta_ticks < 0)[0]), 0)):
            with self.subTest(index=index, tick=tick):
                original = copy.deepcopy(cfg)
                readings = CAPTURED.copy()
                readings[index] = tick
                with self.assertRaisesRegex(ValueError, "跨过编码器单圈边界"):
                    config.record_reference(cfg, readings, "straight", STRAIGHT_REFERENCE)
                self.assertEqual(cfg, original)

    def test_capture_cannot_label_model_zero_as_straight(self):
        cfg = straight_config()
        before = copy.deepcopy(cfg)
        with self.assertRaisesRegex(ValueError, "参考角与所选姿态不一致"):
            config.record_reference(cfg, CAPTURED, "straight", np.zeros(14))
        self.assertEqual(cfg, before)

    def test_reference_snapshot_corruption_fails_before_any_motor_write(self):
        def alter_reference(c):
            c["calibration"]["reference_q_rad"][3] += 0.02
        def alter_capture(c):
            c["calibration"]["captured_ticks"][3] += 1
        def alter_ids(c):
            c["calibration"]["joint_ids"][3] = 99
        def alter_direction(c):
            c["calibration"]["directions"][3] *= -1
        def alter_zero(c):
            c["joints"][3]["zero_tick"] += 1
        def alter_pose(c):
            c["calibration"]["pose"] = "zero"
        for mutation in (alter_reference, alter_capture, alter_ids, alter_direction, alter_zero, alter_pose):
            with self.subTest(mutation=mutation.__name__):
                cfg = straight_config()
                mutation(cfg)
                with straight_rig(cfg=cfg) as r:
                    with self.assertRaises(ValueError):
                        execute_straight(r)
                    self.assertEqual(r.bus.events, [])


class StraightHomingTests(unittest.TestCase):
    def test_default_home_and_run_keep_the_original_start_limit(self):
        for home_only in (True, False):
            with self.subTest(home_only=home_only), straight_rig() as r:
                with self.assertRaisesRegex(runtime.StopRequested, "离HOME太远"):
                    r.ctl.execute(seconds=0.08, motion=True, home_only=home_only)
                self.assertEqual(motor_writes(r), [])

    def test_reference_exception_is_rejected_for_policy_or_read_only_execution(self):
        for motion, home_only in ((True, False), (False, False), (False, True)):
            with self.subTest(motion=motion, home_only=home_only), straight_rig() as r:
                with self.assertRaisesRegex(ValueError, "只用于home"):
                    r.ctl.execute(seconds=0.08, motion=motion, home_only=home_only,
                                  from_pose="straight")
                self.assertEqual(r.bus.events, [])

    def test_current_pose_must_still_be_near_the_captured_reference(self):
        with straight_rig() as r:
            q = STRAIGHT_REFERENCE.copy()
            q[3] += np.deg2rad(6)
            r.bus.positions.update(zip(r.bus.ids, q_to_ticks(q, r.bus.zero, r.bus.direction).tolist()))
            with self.assertRaisesRegex(runtime.StopRequested, "偏离直腿标定参考"):
                execute_straight(r)
            self.assertEqual(motor_writes(r), [])

    def test_motion_during_pid_setup_is_rechecked_before_first_goal(self):
        with straight_rig() as r:
            original_gains = r.bus.set_pid_ram
            def gains_then_move(ids, p, d, i):
                original_gains(ids, p, d, i)
                if list(ids) == r.bus.ids and (p, d, i) == (5, 0, 0):
                    q = STRAIGHT_REFERENCE.copy()
                    q[3] += np.deg2rad(6)
                    r.bus.positions.update(zip(r.bus.ids, q_to_ticks(q, r.bus.zero, r.bus.direction).tolist()))
            r.bus.set_pid_ram = gains_then_move
            with self.assertRaisesRegex(runtime.StopRequested, "偏离直腿标定参考"):
                execute_straight(r)
            self.assertEqual(motor_writes(r), [])
            self.assertEqual(sum(e[0] == "gains" and e[3] == (6, 20, 0) for e in r.bus.events), 14)

    def test_eight_second_move_is_monotonic_speed_bounded_and_never_loads_onnx(self):
        with straight_rig() as r, patch.object(runtime, "OnnxPolicy", side_effect=AssertionError("ONNX opened")) as factory:
            r.ctl.policy = None
            result = execute_straight(r)
            factory.assert_not_called()
            self.assertTrue(result["motion"])
            samples = r.ctl.session["home_samples"]
            self.assertEqual(len(samples), 400)
            targets = np.vstack([STRAIGHT_REFERENCE] + [sample["target"] for sample in samples])
            signed_steps = np.diff(targets, axis=0) * np.sign(HOME - STRAIGHT_REFERENCE)
            self.assertTrue(np.all(signed_steps >= -1e-12))
            np.testing.assert_allclose(targets[-1], HOME, rtol=0, atol=1e-14)
            speed = np.max(np.abs(np.diff(targets, axis=0))) / 0.02
            self.assertLessEqual(speed, homing.REFERENCE_MAX_SPEED_RAD_S)
            self.assertLessEqual(r.ctl.session["home_plan"]["peak_target_speed_deg_s"], 12.0)
            # Encoder rounding can add one count to an individual step. Check
            # emitted commands against that quantization bound, not a float-only limit.
            writes = [e for e in r.bus.events if e[0] == "positions"]
            ticks = np.array([[e[2][sid] for sid in r.bus.ids] for e in writes[:401]])
            max_tick_step = np.max(np.abs(np.diff(ticks, axis=0)))
            self.assertLessEqual(max_tick_step * RADIANS_PER_TICK / 0.02,
                                 homing.REFERENCE_MAX_SPEED_RAD_S + RADIANS_PER_TICK / 0.02)
            torque_on = next(e for e in r.bus.events if e[0] == "torque" and e[2])
            self.assertGreaterEqual(writes[400][1] - torque_on[1], 7.97)
            np.testing.assert_allclose(ticks_to_q([r.bus.positions[sid] for sid in r.bus.ids],
                                                r.bus.zero, r.bus.direction),
                                       HOME, atol=RADIANS_PER_TICK / 2)
            self.assertEqual([e[2] for e in r.bus.events if e[0] == "torque"], [True, False])

    def test_impossible_move_durations_are_rejected_without_motor_writes(self):
        for duration in (0, 7.99, 30.01, float("nan"), float("inf")):
            with self.subTest(duration=duration), straight_rig() as r:
                with self.assertRaises(ValueError):
                    execute_straight(r, move_seconds=duration)
                self.assertEqual(motor_writes(r), [])

    def test_one_slow_read_extends_homing_without_catchup_goal_bursts(self):
        with straight_rig() as r:
            original_feedback = r.bus.read_feedback_many
            delayed_reads = []

            def delayed_once(ids, sync):
                # Mid-trajectory read is slower than the control period but
                # still within max_cycle_s. Later reads return to normal.
                delay = r.bus.enabled_reads == 199
                r.bus.read_s = 0.035 if delay else 0.001
                if delay:
                    delayed_reads.append(r.clock.now)
                return original_feedback(ids, sync)

            r.bus.read_feedback_many = delayed_once
            execute_straight(r)
            self.assertEqual(len(delayed_reads), 1)
            self.assertEqual(len(r.ctl.session["home_samples"]), 400)
            writes = [e for e in r.bus.events if e[0] == "positions"][:401]
            self.assertEqual(len(writes), 401)  # current-angle preload + 400 targets
            finished = np.array([e[1] for e in writes])
            # This mock spends exactly 1 ms writing and records completion.
            # Assert both start and completion intervals so a late cycle cannot
            # squeeze the next 20 ms trajectory increment into a 2 ms burst.
            begun = finished - 0.001
            self.assertTrue(np.all(np.diff(begun) >= 0.02 - 1e-9))
            self.assertTrue(np.all(np.diff(finished) >= 0.02 - 1e-9))
            self.assertGreater(np.max(np.diff(begun)), 0.03)
            self.assertGreater(begun[-1] - begun[0], 8.01)
            np.testing.assert_allclose(r.ctl.session["home_samples"][-1]["target"], HOME,
                                       rtol=0, atol=1e-14)
            np.testing.assert_allclose(ticks_to_q([r.bus.positions[sid] for sid in r.bus.ids],
                                                r.bus.zero, r.bus.direction),
                                       HOME, atol=RADIANS_PER_TICK / 2)
            self.assertEqual([e[2] for e in r.bus.events if e[0] == "torque"], [True, False])

    def test_blocked_reverse_or_jumping_measurements_stop_before_the_next_goal(self):
        for response, reason in (("blocked", "跟踪误差"), ("reverse", "跟踪误差"), ("jump", "角度跳变")):
            with self.subTest(response=response), straight_rig(response=response) as r:
                with self.assertRaisesRegex(runtime.StopRequested, reason):
                    execute_straight(r)
                events = r.bus.events
                self.assertEqual([e[2] for e in events if e[0] == "torque"], [True, False])
                self.assertLess(len(r.ctl.session["home_samples"]), 400)
                # The failing sensor sample must be followed by unloading, with
                # no further target write in between (or after unloading).
                last_read = max(i for i, e in enumerate(events) if e[0] == "read")
                self.assertFalse(any(e[0] == "positions" for e in events[last_read + 1:]))
                self.assertTrue(any(e[0] == "torque" and not e[2] for e in events[last_read + 1:]))
                self.assertNotEqual(r.bus.goals, r.bus.positions)
                if response == "jump":
                    self.assertEqual(sum(e[0] == "positions" for e in events), 1)  # preload only
                else:
                    self.assertLess(r.clock.now - next(e[1] for e in events if e[0] == "torque" and e[2]), 4.0)

    def test_firmware_limits_reject_unknown_ranges_and_either_path_endpoint(self):
        cfg = straight_config()
        plan = homing.plan_home(cfg, STRAIGHT_REFERENCE, from_pose="straight")
        row = plan["joints"][3]
        low, high = sorted((row["current_tick"], row["home_tick"]))
        for lo, hi in ((None, None), (0, 0), (0, 4096), (True, 4095),
                       (low + 1, 4095), (0, high - 1)):
            with self.subTest(lo=lo, hi=hi), straight_rig() as r:
                original_configuration = r.bus.read_configuration
                def restricted(sid):
                    value = original_configuration(sid)
                    if sid == row["id"]:
                        value.update(position_min=lo, position_max=hi)
                    return value
                r.bus.read_configuration = restricted
                with self.assertRaisesRegex(ValueError, "限位"):
                    execute_straight(r)
                self.assertEqual(motor_writes(r), [])
                self.assertFalse(any(e[0] == "gains" for e in r.bus.events))


if __name__ == "__main__":
    unittest.main()
