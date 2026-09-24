import unittest

from neural.axm_brain import BoundBrain, BrainIOContract, Channel


class ContractTests(unittest.TestCase):
    def contract(self):
        return BrainIOContract(
            name="fixture-software",
            inputs=(
                Channel("success", 0.0, 1.0, 0.0),
                Channel("load", 0.0, 100.0, 0.0),
            ),
            outputs=("reuse", "explore"),
        )

    def test_named_state_encodes_in_stable_order(self):
        contract = self.contract()
        self.assertEqual(
            contract.encode({"load": 50.0, "success": 1.0}),
            [1.0, 0.0],
        )

    def test_unknown_state_is_rejected(self):
        with self.assertRaises(ValueError):
            self.contract().encode({"mystery": 1.0})

    def test_contract_is_persisted_with_brain(self):
        contract = self.contract()
        bound = contract.new_brain(hidden_size=4, seed=9)
        restored = BoundBrain.from_snapshot(bound.to_snapshot())
        self.assertEqual(
            restored.contract.fingerprint,
            contract.fingerprint,
        )
        self.assertEqual(
            restored.brain.to_snapshot(),
            bound.brain.to_snapshot(),
        )

    def test_tampered_contract_is_rejected(self):
        bound = self.contract().new_brain(hidden_size=4, seed=9)
        snapshot = bound.to_snapshot()
        snapshot["contract"]["name"] = "different-meaning"
        with self.assertRaises(ValueError):
            BoundBrain.from_snapshot(snapshot)


if __name__ == "__main__":
    unittest.main()
