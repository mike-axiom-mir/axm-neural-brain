import copy
import unittest

from neural.axm_brain import AXMBrain, BrainConfig, BrainSnapshotError, Experience


class AXMBrainTests(unittest.TestCase):
    def config(self):
        return BrainConfig(
            input_size=3,
            hidden_size=6,
            output_size=2,
            seed=42,
            replay_capacity=16,
        )

    def test_same_seed_same_birth(self):
        a = AXMBrain(self.config())
        b = AXMBrain(self.config())
        self.assertEqual(a.to_snapshot(), b.to_snapshot())

    def test_direct_teaching_changes_response(self):
        brain = AXMBrain(self.config())
        obs = [1.0, -0.5, 0.25]
        before = AXMBrain.from_snapshot(brain.to_snapshot()).predict(obs, update_state=False)
        for _ in range(80):
            brain.experience(Experience(obs, target=[0.9, -0.9]), remember=False)
            brain.hidden = [0.0] * brain.config.hidden_size
        after = brain.predict(obs, update_state=False)
        self.assertGreater(after[0], before[0])
        self.assertLess(after[1], before[1])

    def test_reward_learning_changes_weights(self):
        brain = AXMBrain(self.config())
        before = copy.deepcopy(brain.w_out)
        brain.experience(
            Experience([0.2, 0.4, -0.1], reward=1.0),
            remember=False,
        )
        self.assertNotEqual(before, brain.w_out)

    def test_snapshot_roundtrip(self):
        brain = AXMBrain(self.config())
        brain.experience(
            Experience([0.1, 0.2, 0.3], target=[0.4, -0.2], reward=0.5)
        )
        restored = AXMBrain.from_snapshot(brain.to_snapshot())
        self.assertEqual(brain.to_snapshot(), restored.to_snapshot())

    def test_tamper_rejected(self):
        brain = AXMBrain(self.config())
        snap = brain.to_snapshot()
        snap["body"]["state"]["steps"] = 999
        with self.assertRaises(BrainSnapshotError):
            AXMBrain.from_snapshot(snap)

    def test_sleep_replays_memory(self):
        brain = AXMBrain(self.config())
        brain.experience(
            Experience([0.8, 0.1, -0.3], target=[0.7, -0.5])
        )
        before = copy.deepcopy(brain.w_out)
        result = brain.sleep()
        self.assertEqual(brain.mode, "sleep")
        self.assertGreater(result["replayed"], 0)
        self.assertNotEqual(before, brain.w_out)

    def test_wake_resets_transient_recurrent_state(self):
        brain = AXMBrain(self.config())
        brain.experience(Experience([1.0, 0.0, 0.0]))
        self.assertNotEqual(brain.hidden, [0.0] * brain.config.hidden_size)
        brain.sleep()
        brain.wake()
        self.assertEqual(brain.mode, "wake")
        self.assertEqual(brain.hidden, [0.0] * brain.config.hidden_size)
        self.assertEqual(brain.cycle, 1)

    def test_replay_is_bounded(self):
        cfg = BrainConfig(
            input_size=1,
            hidden_size=3,
            output_size=1,
            seed=7,
            replay_capacity=3,
        )
        brain = AXMBrain(cfg)
        for i in range(10):
            brain.experience(Experience([float(i)], target=[0.0]))
        self.assertEqual(len(brain.replay), 3)
        self.assertEqual(brain.replay[0].observation, [7.0])


if __name__ == "__main__":
    unittest.main()
