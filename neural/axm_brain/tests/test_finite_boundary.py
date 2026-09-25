import copy
import unittest

from neural.axm_brain import (
    AXMBrain,
    BrainConfig,
    BrainIOContract,
    BrainSnapshotError,
    Channel,
    Experience,
)


class FiniteBoundaryTests(unittest.TestCase):
    def brain(self):
        return AXMBrain(
            BrainConfig(
                input_size=2,
                hidden_size=4,
                output_size=2,
                seed=17,
            )
        )

    def contract(self):
        return BrainIOContract(
            name="finite-boundary-fixture",
            inputs=(
                Channel("load", 0.0, 100.0, 0.0),
                Channel("success", 0.0, 1.0, 0.0),
            ),
            outputs=("reuse", "explore"),
        )

    def test_config_rejects_nonfinite_training_values(self):
        for field in (
            "learning_rate",
            "reward_learning_rate",
            "sleep_learning_scale",
            "sleep_prune_threshold",
            "weight_limit",
        ):
            kwargs = {
                "input_size": 1,
                "hidden_size": 2,
                "output_size": 1,
                field: float("inf"),
            }
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    BrainConfig(**kwargs).validate()

    def test_predict_rejects_nonfinite_observation(self):
        brain = self.brain()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    brain.predict([value, 0.0])

    def test_experience_rejects_nonfinite_target_and_reward(self):
        brain = self.brain()
        with self.assertRaises(ValueError):
            brain.experience(
                Experience([0.0, 0.0], target=[float("nan"), 0.0])
            )
        with self.assertRaises(ValueError):
            brain.experience(
                Experience([0.0, 0.0], reward=float("inf"))
            )

    def test_snapshot_creation_rejects_nonfinite_internal_state(self):
        brain = self.brain()
        brain.w_in[0][0] = float("nan")
        with self.assertRaises(BrainSnapshotError):
            brain.to_snapshot()

    def test_restore_rejects_forged_nonfinite_snapshot(self):
        brain = self.brain()
        snapshot = copy.deepcopy(brain.to_snapshot())
        snapshot["body"]["state"]["hidden"][0] = float("nan")
        snapshot["sha256"] = "0" * 64
        with self.assertRaises(BrainSnapshotError):
            AXMBrain.from_snapshot(snapshot)

    def test_channel_rejects_nonfinite_and_out_of_range_input(self):
        channel = Channel("load", 0.0, 100.0, 0.0)
        for value in (float("nan"), float("inf"), float("-inf"), -0.1, 100.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    channel.encode(value)

    def test_channel_definition_rejects_nonfinite_bounds(self):
        for channel in (
            Channel("x", float("nan"), 1.0, 0.0),
            Channel("x", 0.0, float("inf"), 0.0),
            Channel("x", 0.0, 1.0, float("nan")),
        ):
            with self.subTest(channel=channel):
                with self.assertRaises(ValueError):
                    channel.validate()

    def test_contract_decode_rejects_nonfinite_output(self):
        contract = self.contract()
        with self.assertRaises(ValueError):
            contract.decode([float("nan"), 0.0])
        with self.assertRaises(ValueError):
            contract.decode([0.0, float("inf")])


if __name__ == "__main__":
    unittest.main()
