"""Run with python -m unittest discover -s tests (CPU, no robot connection)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from microduck_deploy.policy import (
    ACTION_SIZE, COMMAND_NAMES, DEFAULT_MODEL_PATH, DEFAULT_SERVO_IDS, HOME,
    JOINT_LIMITS, JOINT_NAMES, OBSERVATION_NAMES, OnnxPolicy, PolicyContractError,
    RADIANS_PER_TICK, action_to_targets, build_obs61, policy_calibration_reference,
    q_to_ticks, ticks_to_q, validate_metadata,
)


def valid_metadata():
    return {
        "joint_names": ",".join(JOINT_NAMES),
        "observation_names": ",".join(OBSERVATION_NAMES),
        "command_names": ",".join(COMMAND_NAMES),
        "default_joint_pos": ",".join(f"{q:.3f}" for q in HOME),
        "action_scale": "1.0",
    }


def standing_observation(**kwargs):
    return build_obs61([0, 0, 0], [0, 0, -1], HOME, np.zeros(14), np.zeros(14), **kwargs)


class CalibrationTests(unittest.TestCase):
    def test_known_scalar_quarter_turn_and_reversed_direction(self):
        self.assertAlmostEqual(float(ticks_to_q(2024, 1000, 1)), np.pi / 2)
        self.assertAlmostEqual(float(ticks_to_q(2024, 1000, -1)), -np.pi / 2)
        self.assertEqual(int(q_to_ticks(-np.pi / 2, 1000, -1)), 2024)

    def test_home_capture_is_not_encoder_center_or_joint_zero(self):
        captured = np.arange(14) * 31 + 1700
        direction = np.array([1, -1] * 7)
        # Caller captured a physically known HOME; derive q=0 encoder intercept.
        zero = captured - direction * HOME / RADIANS_PER_TICK
        np.testing.assert_allclose(ticks_to_q(captured, zero, direction), HOME, atol=1e-15)
        np.testing.assert_array_equal(q_to_ticks(HOME, zero, direction), captured)
        self.assertFalse(np.allclose(zero, 2048))
        self.assertFalse(np.allclose(zero, captured))

    def test_arbitrary_raw_counts_roundtrip_without_wrapping(self):
        ticks = np.array([0, 1, 2047, 2048, 4094, 4095])
        signs = np.array([1, -1, 1, -1, 1, -1])
        zero = np.array([1300.2, 1600.3, 1900.5, 2400.7, 2700.1, 3000.8])
        np.testing.assert_array_equal(q_to_ticks(ticks_to_q(ticks, zero, signs), zero, signs), ticks)
        for ticks in (-1, 4096, 1000.5, np.nan):
            with self.subTest(ticks=ticks), self.assertRaises(PolicyContractError):
                ticks_to_q(ticks, 1000, 1)
        for q in (-1.0, 10.0, np.inf):
            with self.subTest(q=q), self.assertRaises(PolicyContractError):
                q_to_ticks(q, 0, 1)

    def test_invalid_direction_and_accidental_broadcast_are_rejected(self):
        for direction in (0, 0.5, 2, np.nan):
            with self.subTest(direction=direction), self.assertRaises(PolicyContractError):
                ticks_to_q(1800, 1900, direction)
        with self.assertRaises(PolicyContractError):
            ticks_to_q(np.ones((14, 1)) * 1800, np.ones(14) * 1900, 1)

    def test_quantization_error_is_at_most_half_a_tick(self):
        q = np.linspace(-0.9, 0.9, 14)
        zero = np.linspace(1800.1, 2200.8, 14)
        sign = np.array([1, -1] * 7)
        recovered = ticks_to_q(q_to_ticks(q, zero, sign), zero, sign)
        self.assertLessEqual(float(np.max(np.abs(recovered - q))), RADIANS_PER_TICK / 2)

    def test_reference_is_a_copy_and_servo_ids_omit_mouth(self):
        np.testing.assert_array_equal(policy_calibration_reference("zero"), np.zeros(14))
        reference = policy_calibration_reference("home")
        reference[0] = 10
        self.assertEqual(HOME[0], 0)
        self.assertEqual(DEFAULT_SERVO_IDS, (20, 21, 22, 23, 24, 30, 31, 32, 33, 10, 11, 12, 13, 14))
        with self.assertRaises(PolicyContractError):
            policy_calibration_reference("encoder_center")


class ObservationAndActionTests(unittest.TestCase):
    def test_every_observation_slot_and_previous_raw_action(self):
        gyro = [1, 2, 3]
        gravity = [0.6, 0, -0.8]
        offsets = np.arange(14) / 100
        velocity = np.arange(14) + 20
        previous = np.arange(14) + 40  # deliberately outside [-1, 1]
        twist = [0.15, -0.1, 0.25]
        head = [0.1, 0.2, 0.3, 0.4]
        body = [0, 0, 0.005, 0.02, -0.03, 0]
        obs = build_obs61(gyro, gravity, HOME + offsets, velocity, previous, twist, head, body)
        self.assertEqual(obs.shape, (61,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs[0:3], gyro)
        np.testing.assert_allclose(obs[3:6], gravity)
        np.testing.assert_allclose(obs[6:20], offsets, atol=1e-8)
        np.testing.assert_array_equal(obs[20:34], velocity)
        np.testing.assert_array_equal(obs[34:48], previous)
        np.testing.assert_allclose(obs[48:51], twist)
        np.testing.assert_allclose(obs[51:55], head)
        np.testing.assert_allclose(obs[55:61], body)

    def test_home_observation_has_only_gravity_nonzero(self):
        expected = np.zeros(61, dtype=np.float32)
        expected[5] = -1
        np.testing.assert_array_equal(standing_observation(), expected)

    def test_actions_are_home_offsets_without_clip_or_command_double_add(self):
        action = np.linspace(-2, 2, 14)
        before = action.copy()
        np.testing.assert_allclose(action_to_targets(action), HOME + action)
        np.testing.assert_array_equal(action, before)
        np.testing.assert_array_equal(action_to_targets(np.zeros(14)), HOME)
        with self.assertRaisesRegex(PolicyContractError, "joint limits"):
            action_to_targets(action, limits=JOINT_LIMITS)
        with self.assertRaises(PolicyContractError):
            action_to_targets(np.zeros(14), limits=np.zeros((14, 2)))

    def test_invalid_sensor_values_are_not_silently_sanitized(self):
        for gravity in ([0, 0, 0], [0, 0, -9.81], [np.nan, 0, -1]):
            with self.subTest(gravity=gravity), self.assertRaises(PolicyContractError):
                build_obs61([0, 0, 0], gravity, HOME, np.zeros(14), np.zeros(14))
        for q in (np.zeros(15), np.full(14, np.inf), np.zeros((1, 14))):
            with self.assertRaises(PolicyContractError):
                build_obs61([0, 0, 0], [0, 0, -1], q, np.zeros(14), np.zeros(14))
        with self.assertRaises(PolicyContractError):
            build_obs61([1e100, 0, 0], [0, 0, -1], HOME, np.zeros(14), np.zeros(14))
        with self.assertRaises(PolicyContractError):
            action_to_targets(np.full(14, np.nan))


class ModelContractTests(unittest.TestCase):
    def test_loader_rejects_wrong_model_tensor_signature(self):
        session = Mock()
        session.get_modelmeta.return_value = SimpleNamespace(custom_metadata_map=valid_metadata())
        good_input = SimpleNamespace(name="obs", type="tensor(float)", shape=[1, 61])
        good_output = SimpleNamespace(name="actions", type="tensor(float)", shape=[1, 14])
        cases = (
            ([SimpleNamespace(name="obs", type="tensor(float)", shape=[1, 60])], [good_output]),
            ([SimpleNamespace(name="obs", type="tensor(double)", shape=[1, 61])], [good_output]),
            ([good_input, good_input], [good_output]),
            ([good_input], [SimpleNamespace(name="actions", type="tensor(float)", shape=[1, 15])]),
            ([good_input], [SimpleNamespace(name="output", type="tensor(float)", shape=[1, 14])]),
            ([good_input], [good_output, good_output]),
        )
        with patch("onnxruntime.InferenceSession", return_value=session):
            for inputs, outputs in cases:
                session.get_inputs.return_value = inputs
                session.get_outputs.return_value = outputs
                with self.subTest(inputs=inputs, outputs=outputs), self.assertRaises(PolicyContractError):
                    OnnxPolicy("unused.onnx")

    def test_rounded_metadata_accepted_but_not_used_as_exact_home(self):
        metadata = valid_metadata()
        validate_metadata(metadata)
        self.assertNotEqual(float(metadata["default_joint_pos"].split(",")[1]), HOME[1])
        self.assertEqual(action_to_targets(np.zeros(14))[1], -0.0873)

    def test_metadata_rejects_reordering_missing_fields_scale_and_wrong_home(self):
        changes = {
            "joint_names": ",".join(reversed(JOINT_NAMES)),
            "observation_names": ",".join(reversed(OBSERVATION_NAMES)),
            "command_names": "body_pose,head_pose,twist",
            "action_scale": "0.5",
            "default_joint_pos": ",".join(["0"] * 14),
        }
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(PolicyContractError):
                validate_metadata({**valid_metadata(), key: value})
        for key in valid_metadata():
            data = valid_metadata()
            del data[key]
            with self.subTest(missing=key), self.assertRaises(PolicyContractError):
                validate_metadata(data)

    def test_inference_rejects_bad_input_before_onnx_and_bad_output_after_onnx(self):
        policy = OnnxPolicy.__new__(OnnxPolicy)
        policy.session = Mock()
        for obs in (np.zeros(60), np.zeros((1, 61)), np.full(61, np.nan), np.full(61, 1e100)):
            with self.assertRaises(PolicyContractError):
                policy.infer(obs)
        policy.session.run.assert_not_called()
        for output in (np.zeros(14, dtype=np.float32), np.zeros((1, 15), dtype=np.float32),
                       np.full((1, 14), np.nan, dtype=np.float32), np.zeros((1, 14), dtype=np.float64)):
            policy.session.run.return_value = [output]
            with self.assertRaises(PolicyContractError):
                policy.infer(standing_observation())

    def test_both_packaged_models_accept_real_standing_inputs_and_return_finite_actions(self):
        for variant in ("flat", "backlash"):
            path = DEFAULT_MODEL_PATH.parent / f"hd1910_{variant}.onnx"
            with self.subTest(variant=variant):
                self.assertTrue(path.is_file(), f"packaged model missing: {path}")
                policy = OnnxPolicy(path)
                self.assertEqual(policy.session.get_providers(), ["CPUExecutionProvider"])
                for vx in (0.0, 0.15):
                    obs = standing_observation(twist=[vx, 0, 0])
                    action = policy.infer(obs)
                    self.assertEqual(action.shape, (ACTION_SIZE,))
                    self.assertEqual(action.dtype, np.float32)
                    self.assertTrue(np.all(np.isfinite(action)))
                    # Regression check for these four initial inputs, not a proof
                    # all dynamically encountered targets fit anatomical limits.
                    action_to_targets(action, limits=JOINT_LIMITS)
                    next_obs = build_obs61([0, 0, 0], [0, 0, -1], HOME,
                                          np.zeros(14), action, twist=[vx, 0, 0])
                    np.testing.assert_array_equal(next_obs[34:48], action)
                    self.assertTrue(np.all(np.isfinite(policy.infer(next_obs))))


if __name__ == "__main__":
    unittest.main()
