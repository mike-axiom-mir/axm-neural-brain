import importlib
import math
import unittest


def _load_core():
    last_error = None
    for name in ("axm_brain", "neural.axm_brain"):
        try:
            module = importlib.import_module(name)
            required = ("AXMBrain", "BrainConfig", "Experience")
            if all(hasattr(module, item) for item in required):
                return module, None
        except Exception as exc:
            last_error = exc
    return None, last_error


CORE, CORE_IMPORT_ERROR = _load_core()


@unittest.skipUnless(
    CORE is not None,
    "NOT TESTED: dedicated AXM neural core is not importable on this revision",
)
class LearningEvidenceTests(unittest.TestCase):
    """Behavioral evidence probes; parameter drift alone is not a pass condition."""

    def config(self, seed=42, replay_capacity=16, sleep_replay_passes=2):
        return CORE.BrainConfig(
            input_size=2,
            hidden_size=6,
            output_size=1,
            seed=seed,
            replay_capacity=replay_capacity,
            sleep_replay_passes=sleep_replay_passes,
        )

    def brain(self, **kwargs):
        return CORE.AXMBrain(self.config(**kwargs))

    @staticmethod
    def _zero_transient(brain):
        brain.hidden = [0.0] * brain.config.hidden_size
        brain.last_output = [0.0] * brain.config.output_size

    def _measure(self, brain, observation):
        self._zero_transient(brain)
        return brain.predict(observation, update_state=False)[0]

    def _train_target(self, brain, observation, target, steps, remember=False):
        for _ in range(steps):
            self._zero_transient(brain)
            brain.experience(
                CORE.Experience(list(observation), target=[target]),
                remember=remember,
            )

    def test_supervised_behavior_change_is_retained_after_restore(self):
        observation = (0.8, -0.35)
        target = 0.85
        brain = self.brain()
        before = self._measure(brain, observation)

        self._train_target(brain, observation, target, steps=100, remember=False)
        after = self._measure(brain, observation)
        self.assertLess(abs(target - after), abs(target - before))

        restored = CORE.AXMBrain.from_snapshot(brain.to_snapshot())
        restored_after = self._measure(restored, observation)
        self.assertAlmostEqual(after, restored_after, places=12)

    def test_replay_has_behavioral_value_against_no_replay_control(self):
        observation = (0.65, 0.15)
        target = -0.75
        with_replay = self.brain(seed=9)
        no_replay = self.brain(seed=9)

        self._zero_transient(with_replay)
        with_replay.experience(
            CORE.Experience(list(observation), target=[target]),
            remember=True,
        )
        self._zero_transient(no_replay)
        no_replay.experience(
            CORE.Experience(list(observation), target=[target]),
            remember=False,
        )

        pre_a = self._measure(with_replay, observation)
        pre_b = self._measure(no_replay, observation)
        self.assertAlmostEqual(pre_a, pre_b, places=12)

        with_replay.sleep()
        no_replay.sleep()
        with_replay.wake()
        no_replay.wake()

        replay_error = abs(target - self._measure(with_replay, observation))
        control_error = abs(target - self._measure(no_replay, observation))
        self.assertLess(
            replay_error,
            control_error,
            "replay changed state but did not beat the no-replay behavioral control",
        )

    def test_identical_replay_is_deterministic(self):
        events = (
            CORE.Experience([0.2, 0.7], target=[0.5]),
            CORE.Experience([-0.4, 0.1], target=[-0.6], reward=0.25),
            CORE.Experience([0.9, -0.2], reward=-0.5),
        )
        first = self.brain(seed=17)
        second = self.brain(seed=17)

        for event in events:
            first.experience(CORE.Experience.from_dict(event.to_dict()))
            second.experience(CORE.Experience.from_dict(event.to_dict()))

        first.sleep()
        second.sleep()
        self.assertEqual(first.to_snapshot(), second.to_snapshot())

    def test_reward_causes_retained_behavior_change_not_only_weight_drift(self):
        observation = (0.45, -0.25)
        brain = self.brain(seed=23)
        before = self._measure(brain, observation)

        for _ in range(30):
            self._zero_transient(brain)
            brain.experience(
                CORE.Experience(list(observation), reward=1.0),
                remember=False,
            )

        after = self._measure(brain, observation)
        self.assertGreater(abs(after - before), 1e-10)

        restored = CORE.AXMBrain.from_snapshot(brain.to_snapshot())
        self.assertAlmostEqual(after, self._measure(restored, observation), places=12)

    def test_checkpoint_restore_preserves_future_trajectory(self):
        observation = (0.3, 0.55)
        brain = self.brain(seed=31)
        self._train_target(brain, observation, 0.4, steps=12, remember=True)
        restored = CORE.AXMBrain.from_snapshot(brain.to_snapshot())

        future = (
            CORE.Experience([0.1, -0.2], target=[-0.3], reward=0.1),
            CORE.Experience([0.6, 0.4], target=[0.7]),
            CORE.Experience([-0.8, 0.2], reward=-0.25),
        )
        for event in future:
            brain.experience(CORE.Experience.from_dict(event.to_dict()))
            restored.experience(CORE.Experience.from_dict(event.to_dict()))

        brain.sleep()
        restored.sleep()
        self.assertEqual(brain.to_snapshot(), restored.to_snapshot())

    def test_interference_probe_measures_retest_delta(self):
        brain = self.brain(seed=55)
        a_obs = (0.9, -0.9)
        b_obs = (-0.9, 0.9)
        a_target = 0.8
        b_target = -0.8

        self._train_target(brain, a_obs, a_target, steps=80, remember=False)
        a_error_before_b = abs(a_target - self._measure(brain, a_obs))

        self._train_target(brain, b_obs, b_target, steps=80, remember=False)
        a_error_after_b = abs(a_target - self._measure(brain, a_obs))

        forgetting_delta = a_error_after_b - a_error_before_b
        self.assertTrue(math.isfinite(forgetting_delta))


if __name__ == "__main__":
    unittest.main()
