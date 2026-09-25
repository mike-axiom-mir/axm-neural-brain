import unittest
from evaluation.verify_innovation import verify


class InnovationEvidenceTests(unittest.TestCase):
    def test_learned_selection_against_controls_on_disjoint_instances(self):
        evidence = verify()
        self.assertEqual(evidence['held_out_task_instances'], 36)
        for key in ('all_archives_restore', 'all_experiments_reproduce', 'all_parent_states_preserved', 'all_choices_retained'):
            self.assertTrue(evidence[key], key)
        metrics = evidence['mean_metrics']
        for control in ('fixed_mse', 'uniform_random_expected_mse', 'genesis_selector_mse'):
            self.assertLess(metrics['learned_mse'], metrics[control], control)
        # An exhaustive oracle is a ceiling, not a beatable fair-use baseline.
        self.assertGreaterEqual(metrics['learned_mse']+1e-14, metrics['exhaustive_oracle_mse'])


if __name__ == '__main__': unittest.main()
