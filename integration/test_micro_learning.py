import unittest

from neural.axm_brain import AXMBrain, BrainConfig
from neural.axm_brain.simulation import learn_in_simulation
from axm_neural_network.microsim import MicroDynamics, DynamicsRules


class DirectSimulationLearningTests(unittest.TestCase):
    def parent(self):
        return AXMBrain(BrainConfig(3,8,2,seed=41,learning_rate=.04,replay_capacity=32))

    def test_direct_experience_improves_unseen_seeds_and_retains_state(self):
        parent = self.parent()
        before = parent.to_snapshot()
        result = learn_in_simulation(parent,MicroDynamics(),range(12),range(100,112),epochs=6)['body']
        self.assertLess(result['held_out_after']['mse'],result['held_out_before']['mse']*.5)
        self.assertEqual(result['held_out_after'],result['held_out_after_restore'])
        self.assertEqual(parent.to_snapshot(),before)
        self.assertEqual(result['training_transitions'],864)
        self.assertEqual(result['candidate']['body']['state']['host_experience_count'],864)
        self.assertTrue(result['restore_exact'])
        self.assertFalse(result['automatic_adoption'])
        self.assertTrue(all(e['source']=='deterministic_simulation' for e in result['candidate']['body']['state']['replay']))

    def test_repeat_preserves_trace_and_learned_state(self):
        sim = MicroDynamics(DynamicsRules(horizon=3))
        a = learn_in_simulation(self.parent(),sim,range(3),range(10,13),epochs=2)['body']
        b = learn_in_simulation(self.parent(),sim,range(3),range(10,13),epochs=2)['body']
        self.assertEqual(a['experience_trace_sha256'],b['experience_trace_sha256'])
        self.assertEqual(a['candidate'],b['candidate'])
        self.assertEqual(a['batches'],b['batches'])

    def test_overlap_and_budget_fail_without_parent_mutation(self):
        parent = self.parent()
        before = parent.to_snapshot()
        with self.assertRaises(ValueError): learn_in_simulation(parent,MicroDynamics(),[1,2],[2,3])
        with self.assertRaises(ValueError): learn_in_simulation(parent,MicroDynamics(),[1,2],[3,4],max_transitions=1)
        self.assertEqual(before,parent.to_snapshot())


if __name__=='__main__': unittest.main()
