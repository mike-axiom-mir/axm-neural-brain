import unittest

from neural.axm_brain import (
    AXMBrain,
    BrainConfig,
    BrainIOContract,
    Channel,
    Experience,
    StateAnalyzer,
)


class AnalyzerTests(unittest.TestCase):
    def brain(self):
        return AXMBrain(
            BrainConfig(
                input_size=2,
                hidden_size=4,
                output_size=1,
                seed=11,
                replay_capacity=8,
            )
        )

    def test_capture_reports_mechanical_growth_state(self):
        brain = self.brain()
        brain.experience(
            Experience([1.0, 0.0], target=[0.8], directions=("CREATE",))
        )
        report = StateAnalyzer.capture(brain)
        self.assertEqual(report["host_experiences"], 1)
        self.assertEqual(report["experience_directions"]["CREATE"], 1)
        self.assertGreater(report["parameter_summary"]["count"], 0)
        self.assertEqual(
            report["truth_boundary"],
            "mechanical-state-report-not-semantic-understanding",
        )

    def test_compare_sees_learning_delta(self):
        brain = self.brain()
        before = brain.to_snapshot()
        brain.experience(
            Experience([1.0, -0.5], target=[0.9], directions=("LEARN",))
        )
        after = brain.to_snapshot()
        delta = StateAnalyzer.compare_snapshots(before, after)
        self.assertGreater(delta["changed_parameters"], 0)
        self.assertEqual(delta["host_experiences_delta"], 1)
        self.assertEqual(delta["direction_experience_delta"]["LEARN"], 1)

    def test_sleep_delta_can_be_separated(self):
        brain = self.brain()
        brain.experience(
            Experience([0.5, 0.25], target=[0.4], directions=("USE",))
        )
        before_sleep = brain.to_snapshot()
        brain.sleep()
        after_sleep = brain.to_snapshot()
        delta = StateAnalyzer.compare_snapshots(before_sleep, after_sleep)
        self.assertEqual(delta["host_experiences_delta"], 0)
        self.assertEqual(delta["sleep_count_delta"], 1)
        self.assertGreater(delta["changed_parameters"], 0)

    def test_invalid_root_direction_is_rejected(self):
        brain = self.brain()
        with self.assertRaises(ValueError):
            brain.experience(
                Experience([0.0, 0.0], directions=("ENTERTAIN",))
            )

    def test_bound_brain_carries_direction_metadata(self):
        contract = BrainIOContract(
            name="fixture",
            inputs=(Channel("load", 0.0, 100.0),),
            outputs=("reuse",),
        )
        bound = contract.new_brain(hidden_size=3, seed=4)
        bound.experience(
            {"load": 20.0},
            directions=("USE", "LEARN"),
        )
        self.assertEqual(bound.brain.direction_experience_counts["USE"], 1)
        self.assertEqual(bound.brain.direction_experience_counts["LEARN"], 1)


if __name__ == "__main__":
    unittest.main()
