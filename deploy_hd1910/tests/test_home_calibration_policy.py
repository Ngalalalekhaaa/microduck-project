"""HOME deployment contract tests: CPU only, no serial/SSH or robot access."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microduck_deploy.home_calibration_policy import (
    HomeCalibrationContractError, HomeCalibrationPolicy, HomeCommandAccumulator,
    HomeObservationHistory, build_home_frame34, build_home_frame48, build_home_frame49,
    validate_home_contract,
)
from microduck_deploy.policy import HOME, JOINT_NAMES


def valid_contract():
    """Independent fixture for the current training export's public contract."""
    home = HOME.astype(np.float32).astype(float).tolist()
    schema = [
        {"name": "reported_joint_position_minus_home", "start": 0, "size": 14, "unit": "rad"},
        {"name": "signed_current_fraction", "start": 14, "size": 14, "unit": "dimensionless"},
        {"name": "sum_absolute_current", "start": 28, "size": 1, "unit": "A"},
        {"name": "projected_gravity", "start": 29, "size": 3, "unit": "unit_vector"},
        {"name": "body_gyro", "start": 32, "size": 3, "unit": "rad/s"},
        {"name": "command_offset_from_home", "start": 35, "size": 14, "unit": "rad"},
    ]
    return {
        "task_id": "Mjlab-HomeCalibration-Flat-MicroDuck-HD1910",
        "policy_family": "hd1910_home_calibration_v1", "run_path": "test/model_100.pt",
        "joint_names": list(JOINT_NAMES), "default_joint_pos": home,
        "home_angles_rad_json": json.dumps(home), "control_dt": 0.05,
        "actor_frame_dim": 49, "history_length": 8, "actor_dim": 392,
        "encoder_ticks": 4096, "current_normalization_epsilon_a": 0.02,
        "current_normalization": "I/max(sum(abs(I)),current_normalization_epsilon_a)",
        "current_sign": "positive_motor_torque_along_positive_model_joint_axis",
        "history_order": "oldest_to_newest_frame_major_first_sample_repeated_on_reset",
        "observation_schema_json": json.dumps(schema),
        "observation_normalization": "baked_into_onnx",
        "action_semantics": "command_offset=clip(command_offset+clip(output,-1,1)*delta_limit_rad,-offset_limit_rad,offset_limit_rad)",
        "delta_limit_rad": math.radians(.25), "offset_limit_rad": math.radians(8),
        "settle_seconds": .5,
        "deployment_target": "existing_calibrated_HOME+command_offset; do_not_subtract_unknown_training_bias",
        "load_signal": "synthetic_signed_phase_current_A=BAM_motor_torque/kt; hardware_mapping_not_validated",
        "role": "simulation_candidate; current walking runtime has a different observation/action contract",
    }


def metadata_for(contract):
    return {
        k: (",".join(f"{x:.3f}" if isinstance(x, (float, int)) else str(x) for x in v)
            if isinstance(v, list) else str(v))
        for k, v in contract.items()
    }


def valid_no_current_contract():
    result = valid_contract()
    for key in ('current_normalization', 'current_sign', 'current_normalization_epsilon_a'):
        result.pop(key)
    result.update(
        task_id='Mjlab-HomeCalibration-NoCurrent-Flat-MicroDuck-HD1910',
        policy_family='hd1910_home_calibration_no_current_v2',
        actor_frame_dim=34, actor_dim=272, load_signal='none', current_input='none',
        role='simulation_candidate; encoder_and_imu_only_home_correction',
        observation_schema_json=json.dumps([
            {'name': 'reported_joint_position_minus_home', 'start': 0, 'size': 14, 'unit': 'rad'},
            {'name': 'projected_gravity', 'start': 14, 'size': 3, 'unit': 'unit_vector'},
            {'name': 'body_gyro', 'start': 17, 'size': 3, 'unit': 'rad/s'},
            {'name': 'command_offset_from_home', 'start': 20, 'size': 14, 'unit': 'rad'},
        ]),
    )
    return result


def valid_force_ratio_contract():
    result = valid_no_current_contract()
    schema = json.loads(result['observation_schema_json'])
    schema.append({'name': 'target_joint_torque_fraction', 'start': 34,
                   'size': 14, 'unit': 'dimensionless'})
    target = np.arange(1, 15, dtype=float)
    target /= target.sum()
    result.update(
        task_id='Mjlab-HomeForceRatio-NoCurrent-Flat-MicroDuck-HD1910',
        policy_family='hd1910_home_force_ratio_no_current_v1',
        actor_frame_dim=48, actor_dim=384,
        role='simulation_candidate; encoder_and_imu_only_home_force_ratio_correction',
        objective='joint_torque_magnitude_ratio',
        force_ratio_definition='abs(qfrc_actuator_servo)/sum(abs(qfrc_actuator_servo))',
        force_ratio_input='fixed_reference_target_only; no_measured_force_or_current',
        target_force_ratio_json=json.dumps(target.tolist()),
        observation_schema_json=json.dumps(schema),
    )
    return result


class ForceRatioContractTests(unittest.TestCase):
    def test_exact_contract_and_fixed_reference_fill_actor_tail(self):
        metadata = valid_force_ratio_contract()
        contract = validate_home_contract(metadata)
        self.assertEqual(contract, validate_home_contract(metadata_for(metadata)))
        self.assertFalse(contract.requires_current)
        self.assertEqual((contract.actor_frame_dim, contract.actor_dim), (48, 384))
        self.assertEqual(contract.target_force_ratio, tuple(json.loads(metadata['target_force_ratio_json'])))
        args = (HOME + .02, [0, 0, -1], [.1, -.2, .3], np.zeros(14))
        frame48 = build_home_frame48(*args, contract=contract)
        frame34 = build_home_frame34(*args, contract=validate_home_contract(valid_no_current_contract()))
        np.testing.assert_array_equal(frame48[:34], frame34)
        np.testing.assert_array_equal(frame48[34:], np.asarray(contract.target_force_ratio, dtype=np.float32))
        history = HomeObservationHistory(contract).push(frame48)
        np.testing.assert_array_equal(history.reshape(8, 48), np.repeat(frame48[None], 8, axis=0))
        with self.assertRaises(HomeCalibrationContractError):
            build_home_frame34(*args, contract=contract)
        with self.assertRaises(HomeCalibrationContractError):
            build_home_frame48(*args, contract=validate_home_contract(valid_no_current_contract()))

    def test_missing_contract_fields_and_invalid_targets_are_rejected(self):
        for key in valid_force_ratio_contract():
            broken = valid_force_ratio_contract()
            broken.pop(key)
            with self.subTest(missing=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)
        for target in ([0.] * 14, [1.] * 14, [1.] * 13,
                       [-.1, 1.1] + [0.] * 12, [float('nan')] + [0.] * 13,
                       [float('inf')] + [0.] * 13):
            broken = valid_force_ratio_contract()
            broken['target_force_ratio_json'] = json.dumps(target)
            with self.subTest(target=target), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)
        for key, value in [('actor_dim', 272), ('actor_frame_dim', 34),
                           ('force_ratio_input', 'measured_current'), ('objective', 'pose')]:
            broken = valid_force_ratio_contract()
            broken[key] = value
            with self.subTest(mixed=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)


class NoCurrentContractTests(unittest.TestCase):
    def test_complete_v2_contract_has_no_current_and_matches_embedded_metadata(self):
        typed = validate_home_contract(valid_no_current_contract())
        self.assertEqual(typed, validate_home_contract(metadata_for(valid_no_current_contract())))
        self.assertFalse(typed.requires_current)
        self.assertIsNone(typed.current_normalization_epsilon_a)
        self.assertEqual((typed.actor_frame_dim, typed.actor_dim), (34, 272))
        self.assertTrue(validate_home_contract(valid_contract()).requires_current)

    def test_missing_or_mixed_v1_v2_metadata_is_rejected(self):
        for key in valid_no_current_contract():
            broken = valid_no_current_contract()
            broken.pop(key)
            with self.subTest(missing=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)
        for key in ('current_normalization', 'current_sign', 'current_normalization_epsilon_a'):
            broken = valid_no_current_contract()
            broken[key] = valid_contract()[key]
            with self.subTest(mixed=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)
        for key, value in [('actor_dim', 392), ('actor_frame_dim', 49),
                           ('current_input', 'optional'), ('task_id', valid_contract()['task_id'])]:
            broken = valid_no_current_contract()
            broken[key] = value
            with self.subTest(wrong=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)

    def test_no_current_builder_matches_common_v1_slots_but_rejects_cross_family(self):
        current = validate_home_contract(valid_contract())
        no_current = validate_home_contract(valid_no_current_contract())
        q, g, gyro, offset = HOME + .013, [0, 0, -1], [.1, -.2, .3], np.linspace(-.05, .05, 14)
        old = build_home_frame49(q, np.linspace(-.3, .3, 14), g, gyro, offset, contract=current)
        new = build_home_frame34(q, g, gyro, offset, contract=no_current)
        np.testing.assert_array_equal(new, np.concatenate((old[:14], old[29:])))
        self.assertEqual(new.shape, (34,))
        with self.assertRaises(HomeCalibrationContractError):
            build_home_frame34(q, g, gyro, offset, contract=current)
        with self.assertRaises(HomeCalibrationContractError):
            build_home_frame49(q, np.zeros(14), g, gyro, offset, contract=no_current)


def frame(contract, **overrides):
    args = dict(q_reported_rad=HOME, current_joint_a=np.zeros(14), gravity=[0, 0, -1],
                gyro_rad_s=np.zeros(3), command_offset_rad=np.zeros(14))
    args.update(overrides)
    return build_home_frame49(**args, contract=contract)


class ContractTests(unittest.TestCase):
    def test_embedded_rounded_csv_and_exact_json_have_same_contract(self):
        typed = validate_home_contract(valid_contract())
        embedded = validate_home_contract(metadata_for(valid_contract()))
        self.assertEqual(typed, embedded)
        self.assertEqual(typed.actor_dim, 392)
        np.testing.assert_array_equal(typed.home_angles_rad, HOME.astype(np.float32))

    def test_every_missing_contract_field_is_rejected(self):
        for key in valid_contract():
            broken = valid_contract()
            del broken[key]
            with self.subTest(key=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)

    def test_wrong_joint_order_units_history_and_action_meaning_rejected(self):
        cases = [
            ("joint_names", list(reversed(JOINT_NAMES))), ("actor_dim", 61),
            ("history_length", 4), ("control_dt", .02), ("encoder_ticks", 1024),
            ("history_order", "newest_first"), ("observation_normalization", "external"),
            ("current_sign", "unsigned"), ("delta_limit_rad", .25),
            ("offset_limit_rad", 8), ("settle_seconds", 0),
            ("action_semantics", "absolute_position"), ("run_path", ""),
            ("current_normalization_epsilon_a", float("nan")),
        ]
        for key, value in cases:
            broken = valid_contract()
            broken[key] = value
            with self.subTest(key=key), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)
        broken = valid_contract()
        schema = json.loads(broken["observation_schema_json"])
        schema[2]["unit"] = "mA"
        broken["observation_schema_json"] = json.dumps(schema)
        with self.assertRaisesRegex(HomeCalibrationContractError, "schema"):
            validate_home_contract(broken)
        schema = json.loads(valid_contract()["observation_schema_json"])
        schema[0]["start"] = False
        broken["observation_schema_json"] = json.dumps(schema)
        with self.assertRaisesRegex(HomeCalibrationContractError, "schema"):
            validate_home_contract(broken)

    def test_rounded_home_cannot_replace_exact_home_and_bad_json_is_rejected(self):
        for value in (json.dumps(np.round(HOME, 3).tolist()), "not JSON", "[NaN]", "null"):
            broken = valid_contract()
            broken["home_angles_rad_json"] = value
            with self.subTest(value=value), self.assertRaises(HomeCalibrationContractError):
                validate_home_contract(broken)


class FrameAndHistoryTests(unittest.TestCase):
    def setUp(self):
        self.contract = validate_home_contract(valid_contract())

    def test_exact_slots_signed_current_total_and_absolute_angle_quantization(self):
        q = HOME + np.linspace(-.01, .01, 14)
        current = np.linspace(-.7, .6, 14, dtype=np.float32)
        offset = np.linspace(-.04, .04, 14, dtype=np.float32)
        value = frame(self.contract, q_reported_rad=q, current_joint_a=current,
                      gyro_rad_s=[.1, -.2, .3], command_offset_rad=offset)
        resolution = np.float32(2 * math.pi / 4096)
        expected_position = np.rint(q.astype(np.float32) / resolution) * resolution - HOME.astype(np.float32)
        np.testing.assert_array_equal(value[:14], expected_position)
        np.testing.assert_allclose(value[14:28], current / np.abs(current).sum(), atol=1e-7)
        self.assertAlmostEqual(float(value[28]), float(np.abs(current).sum()), places=6)
        np.testing.assert_array_equal(value[29:32], [0, 0, -1])
        np.testing.assert_array_equal(value[32:35], np.array([.1, -.2, .3], dtype=np.float32))
        np.testing.assert_array_equal(value[35:], offset)
        self.assertEqual(value.dtype, np.float32)
        # Quantizing HOME itself need not produce zero position error.
        self.assertGreater(np.max(np.abs(frame(self.contract)[:14])), 0)

    def test_zero_and_small_currents_use_epsilon_without_losing_sign(self):
        value = frame(self.contract)
        np.testing.assert_array_equal(value[14:29], np.zeros(15))
        current = np.zeros(14, dtype=np.float32)
        current[0], current[1] = .003, -.002
        value = frame(self.contract, current_joint_a=current)
        np.testing.assert_allclose(value[14:16], [.15, -.1], atol=1e-7)
        self.assertAlmostEqual(float(value[28]), .005, places=7)

    def test_sensor_units_shapes_nonfinite_overflow_and_out_of_range_offsets_rejected(self):
        cases = [
            {"gravity": [0, 0, -9.81]}, {"gravity": [0, 0, 0]},
            {"q_reported_rad": np.zeros((1, 14))}, {"q_reported_rad": np.full(14, np.nan)},
            {"gyro_rad_s": [0, np.inf, 0]}, {"gyro_rad_s": [1e100, 0, 0]},
            {"current_joint_a": np.full(14, 1e38)},
            {"command_offset_rad": np.full(14, math.radians(9))},
        ]
        for values in cases:
            with self.subTest(values=values), self.assertRaises(HomeCalibrationContractError):
                frame(self.contract, **values)
        with self.assertRaises(TypeError):
            frame(self.contract, bias=np.ones(14))

    def test_history_oldest_first_reset_padding_and_no_mutable_aliases(self):
        history = HomeObservationHistory(self.contract)
        first = frame(self.contract)
        stacked = history.push(first)
        np.testing.assert_array_equal(stacked.reshape(8, 49), np.tile(first, (8, 1)))
        first[:] = 99
        stacked[:] = 88
        frames = [frame(self.contract, gyro_rad_s=[i, 0, 0]) for i in range(1, 10)]
        for item in frames:
            stacked = history.push(item)
        np.testing.assert_array_equal(stacked.reshape(8, 49), np.stack(frames[-8:]))
        history.reset()
        np.testing.assert_array_equal(history.push(frames[0]).reshape(8, 49), np.tile(frames[0], (8, 1)))

    def test_invalid_frame_does_not_advance_history(self):
        history = HomeObservationHistory(self.contract)
        first = frame(self.contract)
        history.push(first)
        with self.assertRaises(HomeCalibrationContractError):
            history.push(np.full(49, np.nan))
        last = frame(self.contract, gyro_rad_s=[1, 0, 0])
        result = history.push(last).reshape(8, 49)
        np.testing.assert_array_equal(result[:-1], np.tile(first, (7, 1)))
        np.testing.assert_array_equal(result[-1], last)


class AccumulatorTests(unittest.TestCase):
    def setUp(self):
        self.contract = validate_home_contract(valid_contract())
        self.accumulator = HomeCommandAccumulator(self.contract)

    def test_first_ten_calls_hold_and_eleventh_applies_bounded_increment(self):
        action = np.array([4, -4] * 7, dtype=np.float32)
        for i in range(10):
            self.assertTrue(self.accumulator.is_settling)
            np.testing.assert_array_equal(self.accumulator.advance(action), HOME.astype(np.float32))
            self.assertEqual(self.accumulator.control_steps, i + 1)
        self.assertFalse(self.accumulator.is_settling)
        target = self.accumulator.advance(action)
        expected = np.sign(action) * np.float32(math.radians(.25))
        np.testing.assert_array_equal(self.accumulator.command_offset, expected)
        np.testing.assert_array_equal(target, HOME.astype(np.float32) + expected)

    def test_saturation_has_no_windup_and_reset_returns_to_home(self):
        for _ in range(150):
            self.accumulator.advance(np.ones(14))
        limit = np.float32(math.radians(8))
        np.testing.assert_array_equal(self.accumulator.command_offset, np.full(14, limit))
        np.testing.assert_array_equal(self.accumulator.last_delta, np.zeros(14))
        self.accumulator.advance(-np.ones(14))
        np.testing.assert_array_equal(self.accumulator.command_offset,
                                      np.full(14, limit - np.float32(math.radians(.25))))
        exposed = self.accumulator.command_offset
        exposed[:] = 10
        self.assertLess(np.max(self.accumulator.command_offset), 1)
        self.accumulator.reset()
        self.assertEqual(self.accumulator.control_steps, 0)
        np.testing.assert_array_equal(self.accumulator.command_offset, np.zeros(14))

    def test_invalid_action_even_during_settle_leaves_state_unchanged(self):
        for action in (np.full(14, np.nan), np.full(14, np.inf), np.zeros(13), np.full(14, 1e100)):
            with self.assertRaises(HomeCalibrationContractError):
                self.accumulator.advance(action)
            self.assertEqual(self.accumulator.control_steps, 0)
            np.testing.assert_array_equal(self.accumulator.command_offset, np.zeros(14))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional training-parity check requires torch")
    def test_float32_sequence_matches_training_torch_order_and_clipping(self):
        import torch
        offset = torch.zeros(14, device="cpu")
        rng = np.random.default_rng(930)
        for step in range(240):
            action = rng.normal(size=14).astype(np.float32)
            delta = torch.from_numpy(action).clamp(-1, 1) * self.contract.delta_limit_rad
            if step * self.contract.control_dt < self.contract.settle_seconds - 1e-8:
                delta.zero_()
            offset.add_(delta).clamp_(-self.contract.offset_limit_rad, self.contract.offset_limit_rad)
            self.accumulator.advance(action)
            np.testing.assert_array_equal(self.accumulator.command_offset, offset.numpy())


@unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                     "ONNX inference integration requires onnx and onnxruntime")
class OnnxIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "home.onnx"
        self.contract_path = Path(self.directory.name) / "contract.json"

    def make_model(self, contract=None, *, width=392, output_nan=False):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        values = valid_contract() if contract is None else contract
        weights = np.zeros((width, 14), dtype=np.float32)
        weights[:14, :] = np.eye(14)
        if output_nan:
            weights[0, 0] = np.nan
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["obs", "weight"], ["actions"])], "home_cpu_test",
            [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 14])],
            [numpy_helper.from_array(weights, "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
        for key, value in metadata_for(values).items():
            prop = model.metadata_props.add()
            prop.key, prop.value = key, value
        onnx.save(model, self.model_path)
        self.contract_path.write_text(json.dumps(values))

    def test_real_cpu_inference_sidecar_and_inspection_match_bytes(self):
        self.make_model()
        policy = HomeCalibrationPolicy(self.model_path, contract_path=self.contract_path)
        values = np.linspace(-2, 2, 392, dtype=np.float32)
        # Raw output is not clipped; the separate accumulator owns clipping.
        np.testing.assert_array_equal(policy.infer(values), values[:14])
        report = policy.inspect()
        self.assertEqual(report["providers"], ["CPUExecutionProvider"])
        self.assertEqual(report["model_sha256"], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        self.assertEqual(report["zero_input_action"], [0.] * 14)
        json.dumps(report)

    def test_v2_real_cpu_shape_inference_and_cross_family_width_rejected(self):
        self.make_model(valid_no_current_contract(), width=272)
        policy = HomeCalibrationPolicy(self.model_path, contract_path=self.contract_path)
        values = np.linspace(-2, 2, 272, dtype=np.float32)
        np.testing.assert_array_equal(policy.infer(values), values[:14])
        self.assertEqual(policy.inspect()['input_shape'], [1, 272])
        self.assertFalse(policy.inspect()['requires_current_input'])
        with self.assertRaises(HomeCalibrationContractError):
            policy.infer(np.zeros(392))
        self.make_model(valid_no_current_contract(), width=392)
        with self.assertRaises(HomeCalibrationContractError):
            HomeCalibrationPolicy(self.model_path)

    def test_force_ratio_cpu_inference_uses_384_and_rejects_wrong_target_sidecar(self):
        metadata = valid_force_ratio_contract()
        self.make_model(metadata, width=384)
        policy = HomeCalibrationPolicy(self.model_path, contract_path=self.contract_path)
        values = np.linspace(-2, 2, 384, dtype=np.float32)
        np.testing.assert_array_equal(policy.infer(values), values[:14])
        self.assertEqual(policy.inspect()['input_shape'], [1, 384])
        changed = dict(metadata)
        changed['target_force_ratio_json'] = json.dumps([1.] + [0.] * 13)
        self.contract_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(HomeCalibrationContractError, 'sidecar'):
            HomeCalibrationPolicy(self.model_path, contract_path=self.contract_path)
        self.make_model(metadata, width=272)
        with self.assertRaises(HomeCalibrationContractError):
            HomeCalibrationPolicy(self.model_path)
        self.make_model(valid_contract(), width=272)
        with self.assertRaises(HomeCalibrationContractError):
            HomeCalibrationPolicy(self.model_path)

    def test_walking_shape_missing_contract_and_mismatched_sidecar_rejected(self):
        self.make_model(width=61)
        with self.assertRaises(HomeCalibrationContractError):
            HomeCalibrationPolicy(self.model_path)
        self.make_model(contract={})
        with self.assertRaises(HomeCalibrationContractError):
            HomeCalibrationPolicy(self.model_path)
        self.make_model()
        changed = valid_contract()
        changed["run_path"] = "another/model_200.pt"
        self.contract_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(HomeCalibrationContractError, "sidecar"):
            HomeCalibrationPolicy(self.model_path, contract_path=self.contract_path)

    def test_nonfinite_input_output_and_wrong_observation_shape_rejected(self):
        self.make_model()
        policy = HomeCalibrationPolicy(self.model_path)
        for observation in (np.zeros(61), np.zeros((1, 392)), np.full(392, np.inf)):
            with self.assertRaises(HomeCalibrationContractError):
                policy.infer(observation)
        self.make_model(output_nan=True)
        policy = HomeCalibrationPolicy(self.model_path)
        with self.assertRaises(HomeCalibrationContractError):
            policy.infer(np.zeros(392))


if __name__ == "__main__":
    unittest.main()
