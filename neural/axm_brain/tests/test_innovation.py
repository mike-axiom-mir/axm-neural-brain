from copy import deepcopy
from dataclasses import replace
import unittest

from neural.axm_brain import AXMBrain, BrainConfig, Experience
from neural.axm_brain.innovation import (
    DiscoveryArchive, ExperimentSelector, ExperimentTask, Mutation, Sample,
    descendant, reproduce_experiment, run_experiment, verify_experiment,
)
from neural.axm_brain.state import snapshot_payload


def parent():
    result = AXMBrain(BrainConfig(1, 4, 1, seed=21, replay_capacity=8))
    for _ in range(12): result.experience(Experience([.3], target=[.6]))
    return result


def task(name='alpha'):
    def sample(label, x, y): return Sample(name+'/'+label, (x,), (y,))
    return ExperimentTask(name, 'group/'+name, (.4, -.2),
                          (sample('t1', -.8, -.6), sample('t2', .5, .4)),
                          (sample('v', -.2, -.1),), (sample('h', .7, .6),),
                          (sample('r', .3, .6),),
                          raw_prompt='original human words', interpreted_intent='separate interpretation')


class InnovationTests(unittest.TestCase):
    def setUp(self):
        self.operators = (Mutation('slow', (('learning_rate', .002),)), Mutation('fast', (('learning_rate', .2),)))

    def records(self, tasks=None):
        tasks = tasks or (task(), task('beta'))
        return [run_experiment(parent(), t, m, epochs=2) for t in tasks for m in self.operators]

    def test_descendants_preserve_parent_and_full_history(self):
        brain = parent()
        before = brain.to_snapshot()
        record = run_experiment(brain, task(), self.operators[1], epochs=3)
        self.assertEqual(before, brain.to_snapshot())
        self.assertEqual(record['body']['parent'], before)
        self.assertEqual(record['body']['candidate_before']['body']['state'], before['body']['state'])
        self.assertEqual(record['body']['baseline']['host_experiences'], 6)
        self.assertEqual(record['body']['candidate']['host_experiences'], 6)
        self.assertNotEqual(record['body']['baseline_after'], record['body']['candidate_after'])
        self.assertTrue(record['body']['candidate']['restore_exact'])
        self.assertFalse(record['body']['automatic_adoption'])
        self.assertEqual(record['body']['task']['raw_prompt'], 'original human words')
        self.assertEqual(record['body']['task']['interpreted_intent'], 'separate interpretation')
        self.assertTrue(reproduce_experiment(record))

    def test_noop_control_is_behaviorally_identical(self):
        record = run_experiment(parent(), task(), Mutation('same', (('learning_rate', .03),)), epochs=3)
        self.assertEqual(record['body']['candidate_after'], record['body']['baseline_after'])
        self.assertEqual(record['body']['candidate'], record['body']['baseline'])

    def test_memory_reduction_and_composition_are_explicit(self):
        brain = parent()
        memory = Mutation('memory', (('replay_capacity', 2),))
        combined = Mutation.compose('combined', memory, self.operators[1])
        child = descendant(brain.to_snapshot(), combined)
        self.assertEqual(len(child.replay), 2)
        self.assertEqual(len(brain.replay), 8)
        self.assertEqual(child.config.learning_rate, .2)
        alias = Mutation('different label', combined.changes)
        self.assertEqual(combined.mechanism_id, alias.mechanism_id)
        with self.assertRaises(ValueError): Mutation.compose('conflict', *self.operators)
        with self.assertRaises(ValueError): Mutation('permission', (('execution_authorized', True),))

    def test_split_leakage_rejected_by_ids_and_observations(self):
        t = task()
        with self.assertRaises(ValueError): replace(t, held_out=t.train)
        with self.assertRaises(ValueError): replace(t, held_out=(Sample('new-id', t.train[0].observation, (.1,)),))
        with self.assertRaises(ValueError): Sample('bad', (float('nan'),), (.1,))

    def test_budget_or_bad_dimensions_reject_without_parent_change(self):
        brain = parent()
        before = brain.to_snapshot()
        with self.assertRaises(ValueError): run_experiment(brain, task(), self.operators[0], max_updates=1)
        wrong = replace(task(), held_out=(Sample('wrong', (.1,.2), (.3,)),))
        with self.assertRaises(ValueError): run_experiment(brain, wrong, self.operators[0])
        self.assertEqual(before, brain.to_snapshot())

    def test_archive_retains_regressions_and_rejects_rehashed_lies(self):
        record = self.records()[0]
        archive = DiscoveryArchive()
        self.assertTrue(archive.append(record))
        self.assertFalse(archive.append(record))
        before = archive.snapshot()
        exposed = archive.records
        exposed[0]['body']['adoption_state'] = 'adopted'
        self.assertEqual(before, archive.snapshot())
        self.assertEqual(DiscoveryArchive.restore(before).snapshot(), before)
        for field, bad in [('automatic_adoption', True), ('adoption_state', 'adopted')]:
            forged = deepcopy(record['body'])
            forged[field] = bad
            with self.assertRaises(ValueError): archive.append(snapshot_payload(forged))
        forged = deepcopy(record['body'])
        forged['candidate']['validation_mse'] = 0.0
        with self.assertRaises(ValueError): verify_experiment(snapshot_payload(forged))
        forged = deepcopy(record['body'])
        forged['candidate_before'] = forged['candidate_after']
        with self.assertRaises(ValueError): verify_experiment(snapshot_payload(forged))
        forged = deepcopy(record['body'])
        forged['epochs'] += 1
        with self.assertRaises(ValueError): verify_experiment(snapshot_payload(forged))

    def test_archive_reordering_fails_even_with_rehashed_outer_wrapper(self):
        archive = DiscoveryArchive()
        for r in self.records(): archive.append(r)
        body = archive.snapshot()['body']
        body['entries'].reverse()
        with self.assertRaises(ValueError): DiscoveryArchive.restore(snapshot_payload(body))

    def test_mechanism_index_ignores_redundant_fields_but_keeps_evidence(self):
        archive = DiscoveryArchive()
        equivalent = Mutation('cosmetic rename', (('learning_rate', .2), ('replay_capacity', 8)))
        for m in (self.operators[1], equivalent):
            archive.append(run_experiment(parent(), task(), m))
        self.assertEqual(len(archive.records), 2)
        self.assertEqual(len(archive.mechanism_index()), 1)

    def test_selector_rejects_incomplete_comparisons_atomically(self):
        selector = ExperimentSelector(self.operators, ('a','b'))
        before = selector.snapshot()
        with self.assertRaises(ValueError): selector.fit(self.records()[:-1])
        self.assertEqual(before, selector.snapshot())
        with self.assertRaises(ValueError): selector.fit(self.records(), max_updates=1)
        self.assertEqual(before, selector.snapshot())

    def test_selector_retains_choices_and_blocks_seen_groups(self):
        selector = ExperimentSelector(self.operators, ('a','b'))
        selector.fit(self.records(), epochs=15)
        restored = ExperimentSelector.restore(selector.snapshot())
        self.assertEqual(restored.snapshot(), selector.snapshot())
        before = selector.snapshot()
        self.assertEqual(selector.choose((.6,-.1), group_id='new'), restored.choose((.6,-.1), group_id='new'))
        self.assertEqual(before, selector.snapshot())
        with self.assertRaises(ValueError): selector.choose((.4,-.2), group_id='group/alpha')

    def test_held_out_labels_do_not_change_selector_training(self):
        tasks = (task(), task('beta'))
        changed = tuple(replace(t, held_out=tuple(replace(s, target=(-.9,)) for s in t.held_out)) for t in tasks)
        first = ExperimentSelector(self.operators, ('a','b'))
        second = ExperimentSelector(self.operators, ('a','b'))
        first.fit(self.records(tasks), epochs=15)
        second.fit(self.records(changed), epochs=15)
        self.assertEqual(first.brain.to_snapshot(), second.brain.to_snapshot())
        self.assertEqual(first.choose((.2,-.7),group_id='new'), second.choose((.2,-.7),group_id='new'))
        self.assertNotEqual(first.training_receipts, second.training_receipts)


if __name__ == '__main__': unittest.main()
