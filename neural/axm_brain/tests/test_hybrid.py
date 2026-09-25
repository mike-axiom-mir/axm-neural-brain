import copy
import unittest

from neural.axm_brain import BoundBrain, BrainIOContract
from neural.axm_brain.hybrid import (
    HybridBrain,
    HybridOrgan,
    Predicate,
    load_organ_pack,
    make_organ_pack,
    sha256_value,
)


def build_hybrid():
    contract = BrainIOContract.from_dict(
        {
            "schema": "axm-brain-io/v0.1",
            "name": "hybrid-proof/v0.1",
            "inputs": [
                {"name": "signal", "minimum": 0.0, "maximum": 1.0, "default": 0.0}
            ],
            "outputs": ["reuse", "explore"],
        }
    )
    bound = contract.new_brain(
        hidden_size=6,
        seed=41,
        replay_capacity=32,
        sleep_replay_passes=1,
    )
    bare = HybridBrain(bound)
    state = {"signal": 1.0}
    initial = bare.deliberate(state)["neural_proposal"]["option"]
    target = "explore" if initial == "reuse" else "reuse"
    organ = HybridOrgan(
        organ_id="axm.hybrid.proof",
        version="0.1.0",
        purpose="Provide an explicit peer proposal for the bounded hybrid proof.",
        conditions=(Predicate("signal", "gte", 0.5),),
        proposal_option=target,
        proposal_score=0.9,
        provenance={"kind": "local-test-fixture", "source": "test_hybrid"},
        limitations=("test fixture only",),
    )
    return HybridBrain(bound, (organ,)), state, initial, target


class HybridBrainTests(unittest.TestCase):
    def test_disagreement_requires_evidence_and_grants_no_execution(self):
        hybrid, state, initial, target = build_hybrid()
        decision = hybrid.deliberate(state)
        self.assertEqual(decision["classification"], "TEST_REQUIRED")
        self.assertEqual(decision["neural_proposal"]["option"], initial)
        self.assertEqual(decision["organ_proposal"]["option"], target)
        self.assertFalse(decision["execution_authorized"])
        self.assertFalse(decision["automatic_adoption"])
        self.assertTrue(decision["host_decision_required"])

    def test_host_teaching_changes_and_retains_neural_proposal(self):
        hybrid, state, initial, target = build_hybrid()
        for _ in range(180):
            hybrid.learn_from_host_outcome(state, chosen_option=target, reward=1.0)
        hybrid.bound_brain.brain.sleep()
        hybrid.bound_brain.brain.wake()

        after = hybrid.deliberate(state)["neural_proposal"]["option"]
        self.assertNotEqual(initial, target)
        self.assertEqual(after, target)

        restored = BoundBrain.from_snapshot(hybrid.bound_brain.to_snapshot())
        restored_hybrid = HybridBrain(restored, hybrid.organs)
        retained = restored_hybrid.deliberate(state)["neural_proposal"]["option"]
        self.assertEqual(retained, target)

    def test_organ_pack_tamper_fails_closed(self):
        hybrid, _, _, _ = build_hybrid()
        pack = make_organ_pack(hybrid.organs)
        restored = load_organ_pack(copy.deepcopy(pack))
        self.assertEqual(restored[0].fingerprint, hybrid.organs[0].fingerprint)

        tampered = copy.deepcopy(pack)
        tampered["organs"][0]["proposal"]["option"] = "silent-rewrite"
        with self.assertRaises(ValueError):
            load_organ_pack(tampered)

    def test_measured_winner_becomes_review_only_candidate(self):
        hybrid, state, _, target = build_hybrid()
        decision = hybrid.deliberate(state)
        other = "explore" if target == "reuse" else "reuse"
        experiment = hybrid.record_experiment(
            decision,
            {target: 1.0, other: 0.2},
        )
        candidate = hybrid.organ_candidate_from_experiment(
            experiment,
            organ_id="axm.hybrid.learned-candidate",
            version="0.1.0",
            purpose="Candidate derived from measured hybrid disagreement.",
            conditions=(Predicate("signal", "gte", 0.5),),
            proposal_score=0.8,
        )
        self.assertFalse(candidate["automatic_adoption"])
        self.assertTrue(candidate["host_admission_required"])
        self.assertEqual(candidate["candidate"]["status"], "candidate")
        inner = dict(candidate["candidate"])
        inner_digest = inner.pop("sha256")
        self.assertEqual(inner_digest, sha256_value(inner))


if __name__ == "__main__":
    unittest.main()
