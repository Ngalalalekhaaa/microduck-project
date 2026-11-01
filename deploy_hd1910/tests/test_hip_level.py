"""Stationary four-hip trimming: deterministic plant and isolated targets."""
import copy
import json
import math
import unittest

import numpy as np

from microduck_deploy.config import calibration_arrays, default_config, record_reference
from microduck_deploy.hip_level import hip_targets, level_hips, selected_indices
from microduck_deploy.policy import HOME, STRAIGHT_REFERENCE, q_to_ticks, ticks_to_q


def calibrated():
    cfg = default_config()
    for sid, joint in zip(list(range(1, 10)) + list(range(11, 16)), cfg['joints']):
        joint.update(id=sid, mapping_verified=True, direction=1 if sid == 15 else -1)
    record_reference(cfg, np.full(14, 2048), 'straight', STRAIGHT_REFERENCE)
    return cfg


class Plant:
    def __init__(self, attitude=(2., -1.), jacobian=None):
        self.attitude = np.array(attitude, dtype=float)
        self.jacobian = np.eye(2) * .087890625 if jacobian is None else np.array(jacobian, dtype=float)
        self.offset = np.zeros(2, dtype=int)
        self.moves = []
        self.samples = 0
        self.std = .01

    def move(self, offset):
        self.offset = np.array(offset, dtype=int)
        self.moves.append(list(offset))

    def sample(self):
        self.samples += 1
        pitch, roll = self.attitude + self.jacobian @ self.offset
        p, r = math.radians(pitch), math.radians(roll)
        return dict(pitch_deg=float(pitch), roll_deg=float(roll),
                    pitch_std_deg=self.std, roll_std_deg=self.std,
                    gravity=[math.sin(p), -math.sin(r) * math.cos(p),
                             -math.cos(r) * math.cos(p)],
                    joint_ticks={'2': 2048 - float(self.offset[1]),
                                 '3': 2048 - float(self.offset[0]),
                                 '12': 2048 - float(self.offset[1]),
                                 '13': 2048 + float(self.offset[0])},
                    joint_directions={str(sid): -1 for sid in (2, 3, 12, 13)})


class DeadbandPlant(Plant):
    """Small setpoint changes are absorbed by stiction; larger ones move both hips."""
    def __init__(self, *args, deadband=12, negative_deadband=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadband = deadband
        self.negative_deadband = deadband if negative_deadband is None else negative_deadband

    def sample(self):
        commanded = self.offset.copy()
        deadband = np.where(commanded >= 0, self.deadband, self.negative_deadband)
        self.offset = np.sign(commanded) * np.maximum(np.abs(commanded) - deadband, 0)
        try:
            return super().sample()
        finally:
            self.offset = commanded


class HipTargetsTests(unittest.TestCase):
    def test_only_four_targets_follow_model_axes_and_do_not_mutate_config(self):
        cfg = calibrated()
        before = copy.deepcopy(cfg)
        zero, direction = calibration_arrays(cfg)
        home = q_to_ticks(HOME, zero, direction)
        indices = {2: 1, 3: 2, 12: 10, 13: 11}
        self.assertEqual(selected_indices(cfg), indices)
        self.assertEqual(hip_targets(cfg, (9, 4)), {
            2: int(home[1] - 4), 3: int(home[2] - 9),
            12: int(home[10] - 4), 13: int(home[11] + 9)})
        actual = home.copy()
        for sid, tick in hip_targets(cfg, (9, 4)).items():
            actual[indices[sid]] = tick
        change = ticks_to_q(actual, zero, direction) - ticks_to_q(home, zero, direction)
        self.assertAlmostEqual(change[2], -change[11])
        self.assertAlmostEqual(change[1], change[10])
        self.assertGreater(change[2], 0)
        self.assertGreater(change[1], 0)
        np.testing.assert_array_equal(np.delete(change, list(indices.values())), np.zeros(10))
        self.assertEqual(cfg, before)

    def test_directions_are_configuration_driven(self):
        cfg = calibrated()
        cfg['joints'][2]['direction'] = 1
        home = q_to_ticks(HOME, *calibration_arrays(cfg))
        self.assertEqual(hip_targets(cfg, (-9, 0)), {
            2: int(home[1]), 3: int(home[2] - 9),
            12: int(home[10]), 13: int(home[11] - 9)})

    def test_mapping_bounds_and_single_turn_rejected(self):
        cfg = calibrated()
        for value in ((-45, 0), (0, 45), (1.5, 0), (True, 1), (1,), 4):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hip_targets(cfg, value)
        wrong = copy.deepcopy(cfg)
        wrong['joints'][11]['id'] = 12
        with self.assertRaises(ValueError):
            hip_targets(wrong, (1, 0))
        cfg['joints'][2]['zero_tick'] = -cfg['joints'][2]['direction'] * HOME[2] / (2 * np.pi / 4096) + 2
        with self.assertRaises(ValueError):
            hip_targets(cfg, (9, 0))


class LevelHipsTests(unittest.TestCase):
    def test_converges_with_cross_coupling_and_either_response_sign(self):
        for jacobian in ([[.08789, .02], [-.015, .10]], [[-.09, .01], [.01, .08]]):
            with self.subTest(jacobian=jacobian):
                plant = Plant(jacobian=jacobian)
                report = {}
                self.assertIs(level_hips(plant, report), report)
                self.assertEqual(report['status'], 'converged')
                self.assertTrue(report['vertical_within_tolerance'])
                self.assertEqual(plant.moves[:8], [[9, 0], [0, 0], [-9, 0], [0, 0],
                                                  [0, 9], [0, 0], [0, -9], [0, 0]])
                self.assertTrue(np.all(np.abs(plant.moves) <= 44))
                self.assertTrue(np.all(np.abs(np.diff(plant.moves[7:], axis=0)) <= 5))
                self.assertEqual(report['offset_ticks'], plant.offset.tolist())
                self.assertFalse(report['calibration_saved'])
                json.dumps(report)

    def test_initially_level_never_writes_and_requires_two_windows(self):
        plant = Plant(attitude=(.2, -.3))
        report = level_hips(plant, {})
        self.assertEqual(report['status'], 'converged')
        self.assertEqual(plant.moves, [])
        self.assertEqual(plant.samples, 2)

    def test_only_one_axis_within_tolerance_does_not_claim_success(self):
        plant = Plant(attitude=(.2, -2))
        report = level_hips(plant, {})
        self.assertEqual(report['status'], 'converged')
        self.assertTrue(plant.moves)
        self.assertLessEqual(abs(report['final']['roll_deg']), .5)

    def test_weak_body_response_uses_bounded_probes_then_freezes_without_rollback(self):
        plant = Plant(jacobian=np.zeros((2, 2)))
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'weak_body_probe_response')
        self.assertEqual(report['status'], 'not_converged')
        self.assertEqual(plant.moves, [[9, 0], [18, 0], [27, 0], [36, 0], [44, 0]])
        self.assertEqual(report['offset_ticks'], [44, 0])
        self.assertTrue(report['probes'][-1]['both_joints_followed'])

    def test_adaptive_probe_and_trim_cross_mechanical_deadband(self):
        plant = DeadbandPlant()
        report = level_hips(plant, {})
        self.assertEqual(report['status'], 'converged', report['reason'])
        self.assertEqual(plant.moves[:6], [[9, 0], [18, 0], [0, 0],
                                          [-9, 0], [-18, 0], [0, 0]])
        self.assertFalse(report['probes'][0]['both_joints_followed'])
        self.assertTrue(report['probes'][1]['both_joints_followed'])
        trims = [row for row in report['samples'] if row['stage'].startswith('trim_')]
        self.assertEqual(trims[0]['pitch_deg'], 2.)
        self.assertEqual(trims[1]['pitch_deg'], 2.)
        self.assertGreater(len(trims), 2)
        self.assertTrue(np.all(np.abs(plant.moves) <= 44))
        previous = np.zeros(2, dtype=int)
        for row in trims:
            self.assertTrue(np.all(np.abs(np.array(row['offset_ticks']) - previous) <= 5))
            previous = np.array(row['offset_ticks'])

    def test_asymmetric_probe_span_uses_sum_of_accepted_amplitudes(self):
        plant = DeadbandPlant(deadband=9, negative_deadband=0, attitude=(1., -.5))
        report = level_hips(plant, {}, iterations=1)
        self.assertEqual(report['reason'], 'iteration_limit')
        self.assertEqual([p['amplitude_ticks'] for p in report['probes']], [9, 18, 9, 9, 18, 9])
        self.assertAlmostEqual(report['response_deg_per_tick'][0][0], .087890625 * 18 / 27)

    def test_probe_adapts_to_27_then_rejects_only_if_still_not_effective(self):
        plant = DeadbandPlant(deadband=20, attitude=(1., -.8))
        report = level_hips(plant, {})
        self.assertEqual(plant.moves[:3], [[9, 0], [18, 0], [27, 0]])
        self.assertTrue(report['probes'][2]['both_joints_followed'])
        self.assertEqual(report['status'], 'converged', report['reason'])

    def test_no_joint_travel_stops_at_probe_cap(self):
        plant = DeadbandPlant(deadband=80)
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'insufficient_joint_probe_response')
        self.assertEqual(plant.moves, [[9, 0], [18, 0], [27, 0], [36, 0], [44, 0]])
        self.assertFalse(report['calibration_saved'])

    def test_load_offset_and_stiction_require_36_tick_probe(self):
        class LoadedPitchPlant(Plant):
            def sample(self):
                commanded = self.offset.copy()
                # At baseline ID13 is already 20 ticks ahead of its target.
                # Another 9 ticks must overcome stiction before it moves.
                self.offset = np.sign(commanded) * np.maximum(np.abs(commanded) - 29, 0)
                try:
                    result = super().sample()
                    result['joint_ticks']['13'] += 20
                    return result
                finally:
                    self.offset = commanded
        plant = LoadedPitchPlant(attitude=(.9, -.8))
        report = level_hips(plant, {})
        self.assertEqual(plant.moves[:4], [[9, 0], [18, 0], [27, 0], [36, 0]])
        self.assertEqual([row['both_joints_followed'] for row in report['probes'][:4]],
                         [False, False, False, True])
        self.assertEqual(report['status'], 'converged', report['reason'])
        self.assertEqual(report['baseline']['joint_ticks']['13'], 2068)
        self.assertTrue(np.all(np.abs(plant.moves) <= 44))

    def test_smaller_configured_bound_caps_adaptive_probe(self):
        plant = DeadbandPlant(deadband=20)
        report = level_hips(plant, {}, max_ticks=15)
        self.assertEqual(report['reason'], 'insufficient_joint_probe_response')
        self.assertEqual(plant.moves, [[9, 0], [15, 0]])

    def test_one_stuck_joint_cannot_be_hidden_by_pair_mean_or_body_motion(self):
        class OneStuckJoint(Plant):
            def sample(self):
                result = super().sample()
                result['joint_ticks']['13'] = 2048
                return result
        plant = OneStuckJoint(attitude=(-2., 0), jacobian=np.eye(2) * .04)
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'insufficient_joint_probe_response')
        self.assertFalse(report['probes'][-1]['both_joints_followed'])
        self.assertTrue(report['probes'][-1]['body_response_measurable'])

    def test_encoder_reversed_direction_stops_instead_of_increasing_probe(self):
        class ReversedJoint(Plant):
            def sample(self):
                result = super().sample()
                result['joint_ticks']['13'] = 2048 - float(self.offset[0])
                return result
        plant = ReversedJoint()
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'reversed_joint_probe_response')
        self.assertEqual(plant.moves, [[9, 0]])

    def test_invalid_joint_feedback_is_rejected_before_motion(self):
        for mutate in (lambda result: result.pop('joint_ticks'),
                       lambda result: result['joint_ticks'].update({'12': float('nan')}),
                       lambda result: result['joint_ticks'].update({'12': 4096}),
                       lambda result: result['joint_directions'].update({'13': 0})):
            class InvalidFeedback(Plant):
                def sample(self):
                    result = super().sample()
                    mutate(result)
                    return result
            plant = InvalidFeedback()
            report = level_hips(plant, {})
            self.assertEqual(report['reason'], 'invalid_joint_feedback')
            self.assertEqual(plant.moves, [])

    def test_deadband_does_not_expand_total_offset_limit(self):
        plant = DeadbandPlant(attitude=(4.3, -3.1))
        report = level_hips(plant, {})
        self.assertEqual(report['status'], 'not_converged')
        self.assertEqual(report['reason'], 'offset_limit')
        self.assertTrue(np.all(np.abs(plant.moves) <= 44))
        self.assertFalse(report['vertical_within_tolerance'])

    def test_trim_stall_reports_joint_or_body_cause_with_bounded_small_steps(self):
        for stuck_joints in (True, False):
            class TrimStallPlant(Plant):
                def sample(self):
                    result = super().sample()
                    if len(self.moves) > 8:
                        commanded = self.offset.copy()
                        self.offset = np.zeros(2, dtype=int)
                        baseline_result = super().sample()
                        self.offset = commanded
                        if not stuck_joints:
                            baseline_result['joint_ticks'] = result['joint_ticks']
                        return baseline_result
                    return result
            plant = TrimStallPlant()
            report = level_hips(plant, {})
            self.assertEqual(report['reason'], 'joint_response_stalled' if stuck_joints
                             else 'body_response_stalled')
            self.assertEqual(report['offset_ticks'][0], -20)
            self.assertEqual(len([s for s in report['samples'] if s['stage'].startswith('trim_')]), 4)

    def test_dependent_axes_rejected(self):
        plant = Plant(jacobian=[[.08, .08], [0, 0]])
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'weak_or_dependent_response')
        self.assertEqual(plant.moves[-1], [0, 0])

    def test_unreachable_attitude_stops_without_claiming_success(self):
        plant = Plant(attitude=(7., 0.))
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'offset_limit')
        self.assertEqual(report['status'], 'not_converged')
        self.assertEqual(report['offset_ticks'], [-44, 0])
        self.assertGreater(abs(report['final']['pitch_deg']), .5)

    def test_unstable_or_tilted_baseline_never_moves(self):
        for attitude, std, reason in (((13, 0), .01, 'body_tilt_limit'),
                                      ((2, 0), .25, 'unstable_pitch_window')):
            with self.subTest(reason=reason):
                plant = Plant(attitude=attitude)
                plant.std = std
                report = level_hips(plant, {})
                self.assertEqual(report['reason'], reason)
                self.assertEqual(plant.moves, [])

    def test_tilt_growth_stops_at_current_target(self):
        plant = Plant(attitude=(2., 0.), jacobian=[[.24, 0], [0, .08]])
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'tilt_growth_limit')
        self.assertEqual(plant.moves, [[9, 0]])

    def test_inconsistent_probe_sign_rejected(self):
        class NonlinearPlant(Plant):
            def sample(self):
                self.jacobian[0, 0] = -.087890625 if self.offset[0] < 0 else .087890625
                return super().sample()
        plant = NonlinearPlant()
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'inconsistent_probe_response')
        self.assertEqual(plant.moves[-1], [-9, 0])

    def test_baseline_drift_rejected(self):
        class DriftPlant(Plant):
            def sample(self):
                if self.samples == 3:
                    self.attitude[0] += .4
                return super().sample()
        report = level_hips(DriftPlant(), {})
        self.assertEqual(report['reason'], 'baseline_did_not_return')

    def test_baseline_return_gets_one_extra_settling_window_without_more_moves(self):
        class SlowSettlingPlant(Plant):
            def sample(self):
                if self.samples == 3:
                    self.attitude[0] += .4
                elif self.samples == 4:
                    self.attitude[0] -= .4
                return super().sample()
        plant = SlowSettlingPlant()
        report = level_hips(plant, {})
        self.assertEqual(report['status'], 'converged')
        self.assertTrue(any(row['stage'].endswith('_settled') for row in report['samples']))
        self.assertEqual(plant.moves[:4], [[9, 0], [0, 0], [-9, 0], [0, 0]])

    def test_feedback_direction_is_configured_instead_of_hardcoded(self):
        class PositiveDirectionPlant(Plant):
            def sample(self):
                result = super().sample()
                result['joint_ticks']['3'] = 2048 + float(self.offset[0])
                result['joint_directions']['3'] = 1
                return result
        self.assertEqual(level_hips(PositiveDirectionPlant(), {})['status'], 'converged')

    def test_reversed_joint_during_trim_is_rejected_even_if_imu_improves(self):
        class WrongTrimPlant(Plant):
            def sample(self):
                result = super().sample()
                if len(self.moves) > 8:
                    result['joint_ticks']['3'] = 2048 + float(self.offset[0])
                return result
        report = level_hips(WrongTrimPlant(), {})
        self.assertEqual(report['reason'], 'reversed_joint_trim_response')
        self.assertEqual(len(report['corrections']), 1)

    def test_harmful_trim_response_stops_without_rollback(self):
        class HarmfulPlant(Plant):
            def sample(self):
                if len(self.moves) > 8:
                    self.jacobian = -np.eye(2) * .087890625
                return super().sample()
        plant = HarmfulPlant()
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'response_worsened')
        self.assertEqual(len(plant.moves), 9)

    def test_confirmation_must_remain_within_tolerance(self):
        class BadConfirmationPlant(Plant):
            def sample(self):
                if self.samples == 1:
                    self.attitude[0] = .8
                return super().sample()
        plant = BadConfirmationPlant(attitude=(.2, .2))
        report = level_hips(plant, {})
        self.assertEqual(report['reason'], 'unstable_baseline')
        self.assertEqual(plant.moves, [])

    def test_iteration_limit_and_parameter_validation(self):
        report = level_hips(Plant(), {}, iterations=1)
        self.assertEqual(report['reason'], 'iteration_limit')
        for kwargs in ({'max_ticks': 45}, {'probe_ticks': 10},
                       {'max_ticks': 3}, {'tolerance_deg': float('nan')},
                       {'iterations': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                level_hips(Plant(), {}, **kwargs)

    def test_hardware_exception_keeps_pending_target_for_caller(self):
        class FailingPlant(Plant):
            def move(self, offset):
                raise OSError('serial failed')
        report = {}
        with self.assertRaisesRegex(OSError, 'serial failed'):
            level_hips(FailingPlant(), report)
        self.assertEqual(report['offset_ticks'], [0, 0])
        self.assertEqual(report['pending_offset_ticks'], [9, 0])


if __name__ == '__main__':
    unittest.main()
