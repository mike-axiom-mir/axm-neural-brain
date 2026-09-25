import copy
import unittest

from neural.axm_brain import AXMBrain, BrainConfig, Experience
from neural.axm_brain.state import snapshot_payload


class StateIntegrityTests(unittest.TestCase):
    def brain(self):
        return AXMBrain(BrainConfig(2, 3, 2, seed=17, replay_capacity=2))

    def test_rejected_dimensions_leave_complete_state_unchanged(self):
        for event in (Experience([1.0]), Experience([1.0, 0.0], target=[1.0])):
            with self.subTest(event=event):
                brain = self.brain()
                before = brain.to_snapshot()
                with self.assertRaises(ValueError):
                    brain.experience(event)
                self.assertEqual(brain.to_snapshot(), before)

    def test_restore_rejects_rehashed_wrong_tensor_dimensions(self):
        brain = self.brain()
        for name in ('w_in', 'w_rec', 'w_out', 'trace_in', 'trace_rec', 'trace_out',
                     'hidden', 'last_output', 'b_hidden', 'b_out'):
            with self.subTest(name=name):
                body = brain.to_snapshot()['body']
                body['state'][name].pop()
                with self.assertRaises(ValueError):
                    AXMBrain.from_snapshot(snapshot_payload(body))
        body = brain.to_snapshot()['body']
        body['state']['w_in'][0].pop()
        with self.assertRaises(ValueError):
            AXMBrain.from_snapshot(snapshot_payload(body))

    def test_restore_rejects_invalid_modes_counters_rng_and_replay(self):
        brain = self.brain()
        brain.experience(Experience([0.2, 0.3], target=[0.1, 0.4]))
        mutations = [
            ('mode', 'dream'), ('steps', -1), ('steps', 1.5), ('cycle', True),
            ('rng_state', 0), ('rng_state', 1 << 64),
            ('direction_experience_counts', {'CREATE': -1}),
            ('replay', [Experience([0.0]).to_dict()]),
            ('replay', [Experience([0.0, 0.0], target=[1.0]).to_dict()]),
            ('replay', [Experience([0.0, 0.0]).to_dict()] * 3),
        ]
        for name, value in mutations:
            with self.subTest(name=name, value=value):
                body = brain.to_snapshot()['body']
                body['state'][name] = value
                with self.assertRaises(ValueError):
                    AXMBrain.from_snapshot(snapshot_payload(body))

    def test_integer_config_fields_are_not_coerced(self):
        for name in ('input_size', 'hidden_size', 'output_size', 'seed',
                     'replay_capacity', 'sleep_replay_passes'):
            for value in (True, 1.5, '2'):
                with self.subTest(name=name, value=value):
                    kwargs = dict(input_size=2, hidden_size=3, output_size=2)
                    kwargs[name] = value
                    with self.assertRaises(ValueError):
                        BrainConfig(**kwargs).validate()

    def test_restored_learning_sleep_and_wake_continue_exactly(self):
        brain = self.brain()
        brain.experience(Experience([0.2, 0.3], target=[0.1, 0.4], reward=0.5))
        saved = brain.to_snapshot()
        original_saved = copy.deepcopy(saved)
        restored = AXMBrain.from_snapshot(saved)
        for candidate in (brain, restored):
            candidate.experience(Experience([0.7, -0.4], reward=-0.2))
            candidate.sleep()
            candidate.wake()
            candidate.experience(Experience([0.1, 0.8], target=[-0.2, 0.9]))
        self.assertEqual(brain.to_snapshot(), restored.to_snapshot())
        self.assertEqual(saved, original_saved)
