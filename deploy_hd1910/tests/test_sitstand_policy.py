"""Explicit sitstand loading preserves the strict walking policy contract."""

import unittest
from unittest.mock import patch

import numpy as np

from microduck_deploy.policy import (
    ACTION_SIZE, COMMAND_NAMES, DEFAULT_MODEL_PATH, HOME, JOINT_NAMES,
    OBSERVATION_NAMES, SITSTAND_COMMAND_NAMES, OnnxPolicy, PolicyContractError,
    build_obs61, validate_metadata,
)


SITSTAND_MODEL = DEFAULT_MODEL_PATH.parent / "hd1910_sitstand_20261015.onnx"


def sitstand_metadata():
    return {
        "joint_names": ",".join(JOINT_NAMES),
        "observation_names": ",".join(OBSERVATION_NAMES),
        "command_names": ",".join(SITSTAND_COMMAND_NAMES),
        "default_joint_pos": ",".join(f"{q:.3f}" for q in HOME),
        "action_scale": "1.0",
    }


def posture_observation(flag, previous_action=None):
    previous = np.zeros(ACTION_SIZE) if previous_action is None else previous_action
    return build_obs61([0, 0, 0], [0, 0, -1], HOME, np.zeros(ACTION_SIZE),
                       previous, twist=[flag, 0, 0])


class SitstandMetadataTests(unittest.TestCase):
    def test_sitstand_requires_explicit_policy_kind(self):
        metadata = sitstand_metadata()
        validate_metadata(metadata, policy_kind="sitstand")
        for kwargs in ({}, {"policy_kind": "walking"}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(PolicyContractError, "command_names"):
                validate_metadata(metadata, **kwargs)

    def test_sitstand_rejects_missing_extra_and_reordered_commands(self):
        metadata = sitstand_metadata()
        for commands in ("", "twist", "head_pose,twist", "twist,head_pose,body_pose",
                         "twist,head_pose,unknown", "twist,head_pose,head_pose"):
            with self.subTest(commands=commands), self.assertRaisesRegex(PolicyContractError, "command_names"):
                validate_metadata({**metadata, "command_names": commands}, policy_kind="sitstand")
        del metadata["command_names"]
        with self.assertRaisesRegex(PolicyContractError, "command_names"):
            validate_metadata(metadata, policy_kind="sitstand")

    def test_sitstand_retains_joint_observation_home_and_scale_checks(self):
        changes = {
            "joint_names": ",".join(reversed(JOINT_NAMES)),
            "observation_names": ",".join(reversed(OBSERVATION_NAMES)),
            "default_joint_pos": ",".join(["0"] * ACTION_SIZE),
            "action_scale": "0.5",
        }
        for key, value in changes.items():
            with self.subTest(key=key), self.assertRaises(PolicyContractError):
                validate_metadata({**sitstand_metadata(), key: value}, policy_kind="sitstand")
        for key in sitstand_metadata():
            metadata = sitstand_metadata()
            del metadata[key]
            with self.subTest(missing=key), self.assertRaises(PolicyContractError):
                validate_metadata(metadata, policy_kind="sitstand")

    def test_walking_commands_remain_unchanged(self):
        metadata = {**sitstand_metadata(), "command_names": ",".join(COMMAND_NAMES)}
        validate_metadata(metadata)
        validate_metadata(metadata, policy_kind="walking")
        with self.assertRaisesRegex(PolicyContractError, "command_names"):
            validate_metadata(metadata, policy_kind="sitstand")

    def test_unknown_policy_kind_is_rejected_before_loading_onnx(self):
        with patch("onnxruntime.InferenceSession") as load:
            for kind in ("", "stand", "Walking", "sit", None):
                with self.subTest(kind=kind):
                    with self.assertRaisesRegex(PolicyContractError, "policy_kind"):
                        validate_metadata(sitstand_metadata(), policy_kind=kind)
                    with self.assertRaisesRegex(PolicyContractError, "policy_kind"):
                        OnnxPolicy("does-not-exist.onnx", policy_kind=kind)
            load.assert_not_called()


class SitstandModelTests(unittest.TestCase):
    def test_supplied_model_runs_stand_and_sit_on_cpu(self):
        self.assertTrue(SITSTAND_MODEL.is_file(), f"packaged model missing: {SITSTAND_MODEL}")
        policy = OnnxPolicy(SITSTAND_MODEL, policy_kind="sitstand")
        self.assertEqual(policy.policy_kind, "sitstand")
        self.assertEqual(policy.session.get_providers(), ["CPUExecutionProvider"])
        actions = []
        for flag in (0, 1):
            with self.subTest(flag=flag):
                obs = posture_observation(flag)
                np.testing.assert_array_equal(obs[48:51], [flag, 0, 0])
                np.testing.assert_array_equal(obs[51:], np.zeros(10))
                action = policy.infer(obs)
                self.assertEqual(action.shape, (ACTION_SIZE,))
                self.assertEqual(action.dtype, np.float32)
                self.assertTrue(np.isfinite(action).all())
                actions.append(action)
                following = posture_observation(flag, action)
                np.testing.assert_array_equal(following[34:48], action)
                self.assertTrue(np.isfinite(policy.infer(following)).all())
        self.assertFalse(np.allclose(actions[0], actions[1]), "posture flag must reach the policy")

    def test_supplied_model_is_rejected_by_default_walking_loader(self):
        with self.assertRaisesRegex(PolicyContractError, "command_names"):
            OnnxPolicy(SITSTAND_MODEL)


if __name__ == "__main__":
    unittest.main()
