"""Reproducible bounded benchmark, not an open-ended intelligence score.

Selection is made before held-out candidate outcomes are read. Training and
evaluation task IDs, parent seeds, data points and task parameters are disjoint.
The family split tests new instances of two declared laboratory conditions, not
transfer to unseen domains. Evidence may reveal failure; metrics are never clamped.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random

from neural.axm_brain import AXMBrain, BrainConfig, Experience
from neural.axm_brain.innovation import (
    DiscoveryArchive, ExperimentSelector, ExperimentTask, Mutation, Sample,
    reproduce_experiment, run_experiment,
)

OPERATORS = (
    Mutation('conservative', (('learning_rate', .003),)),
    Mutation('ordinary', (('learning_rate', .03),)),
    Mutation('rapid-adaptation', (('learning_rate', .3),)),
)
FEATURES = ('observed_target_roughness', 'parent_training_error')


def task_instance(seed, condition, *, split):
    rng = random.Random(seed)
    slope = rng.uniform(.65, 1.4)
    offset = rng.uniform(-.2, .2)
    parent = AXMBrain(BrainConfig(1, 5, 1, seed=seed, learning_rate=.03, replay_capacity=0))
    truth = lambda x: math.tanh(slope*x+offset)
    # All parents share the same pretraining procedure. Neither held-out split
    # nor its outcomes enter this pretraining stream.
    for _ in range(70):
        for x in (-.95, -.55, -.15, .25, .65, .95):
            parent.hidden = [0.0]*5
            parent.experience(Experience([x], target=[truth(x)]), remember=False)
    if condition == 'shift':
        target_fn = lambda x: -truth(x)
        noise = .0
    elif condition == 'noisy-source':
        target_fn = truth
        noise = 1.0
    else:
        raise ValueError('unknown benchmark condition')
    xs = [rng.uniform(-.9, .9) for _ in range(14)]
    xs.sort()
    train_targets = [max(-.99, min(.99, target_fn(x) + rng.uniform(-noise, noise))) for x in xs]
    # Features come only from the observed training stream and parent predictions.
    roughness = sum(abs(train_targets[i]-train_targets[i-1]) for i in range(1,len(xs)))/(len(xs)-1)
    errors = []
    for x, y in zip(xs, train_targets):
        probe = AXMBrain.from_snapshot(parent.to_snapshot())
        probe.hidden = [0.0]*5
        errors.append((probe.predict([x], update_state=False)[0]-y)**2)
    features = (min(1.0, roughness)*2-1, sum(errors)/len(errors)-.5)
    task_id = f'{split}/{condition}/{seed}'
    train = tuple(Sample(f'{task_id}/train/{i}', (x,), (y,)) for i,(x,y) in enumerate(zip(xs, train_targets)))
    def samples(kind, count):
        return tuple(Sample(f'{task_id}/{kind}/{i}', (x,), (target_fn(x),))
                     for i,x in enumerate(rng.uniform(-.98,.98) for _ in range(count)))
    regression = tuple(Sample(f'{task_id}/regression/{i}', (x,), (truth(x),))
                       for i,x in enumerate((-.82, -.38, .18, .78)))
    task = ExperimentTask(task_id, task_id, features, train, samples('validation', 9),
                          samples('test', 13), regression,
                          source_event='synthetic_function_experiment', raw_prompt=None,
                          interpreted_intent='Compare adaptation with noisy teaching under controlled numeric functions.')
    return parent, task


def benchmark(*, selector_seed=19, training_start=100, evaluation_start=1000,
              training_per_condition=6, evaluation_per_condition=6):
    if set(range(training_start,training_start+training_per_condition)) & set(range(evaluation_start,evaluation_start+evaluation_per_condition)):
        raise ValueError('training and evaluation seed overlap')
    archive = DiscoveryArchive()
    train_records = []
    for seed in range(training_start, training_start+training_per_condition):
        for condition in ('shift', 'noisy-source'):
            parent, task = task_instance(seed, condition, split='train')
            for mutation in OPERATORS:
                record = run_experiment(parent, task, mutation, epochs=3)
                archive.append(record)
                train_records.append(record)
    selector = ExperimentSelector(OPERATORS, FEATURES, seed=selector_seed)
    genesis = selector.snapshot()
    selector.fit(train_records, epochs=220)
    restored = ExperimentSelector.restore(selector.snapshot())
    fixed_id = min((m.mechanism_id for m in OPERATORS), key=lambda mid: sum(
        r['body']['candidate']['validation_mse'] for r in train_records if r['body']['mutation']['mechanism_id']==mid))
    rng = random.Random(selector_seed)
    rows = []
    for seed in range(evaluation_start, evaluation_start+evaluation_per_condition):
        for condition in ('shift', 'noisy-source'):
            parent, task = task_instance(seed, condition, split='evaluation')
            choice = selector.choose(task.features, group_id=task.group_id)
            retained = restored.choose(task.features, group_id=task.group_id)
            before = ExperimentSelector.restore(genesis).choose(task.features, group_id=task.group_id)
            if choice != retained: raise AssertionError('selector restore changed behavior')
            random_id = rng.choice(OPERATORS).mechanism_id
            outcomes = {}
            for mutation in OPERATORS:
                record = run_experiment(parent, task, mutation, epochs=3)
                archive.append(record)
                outcomes[mutation.mechanism_id] = record
            selected = outcomes[choice['mutation']['mechanism_id']]['body']
            errors = {mid:r['body']['candidate']['held_out_mse'] for mid,r in outcomes.items()}
            rows.append({
                'task': task.task_id, 'condition': condition, 'features': list(task.features),
                'choice': choice['mutation']['name'], 'record_sha256': outcomes[choice['mutation']['mechanism_id']]['sha256'],
                'learned_mse': selected['candidate']['held_out_mse'],
                'genesis_selector_mse': errors[before['mutation']['mechanism_id']],
                'fixed_mse': errors[fixed_id], 'random_mse': errors[random_id],
                'uniform_random_expected_mse': sum(errors.values())/len(errors),
                'exhaustive_oracle_mse': min(errors.values()),
                'regression_delta': selected['candidate']['forgetting_delta'],
                'supervised_updates': selected['candidate']['supervised_updates'],
                'parent_unchanged': parent.to_snapshot()==selected['parent'], 'restart_choice_equal': choice==retained,
            })
    names = ('learned_mse','genesis_selector_mse','fixed_mse','random_mse','uniform_random_expected_mse','exhaustive_oracle_mse')
    metrics = {name:sum(row[name] for row in rows)/len(rows) for name in names}
    metrics['beats_best_training_fixed'] = metrics['learned_mse'] < metrics['fixed_mse']
    metrics['beats_random_expectation'] = metrics['learned_mse'] < metrics['uniform_random_expected_mse']
    metrics['beats_untrained_selector'] = metrics['learned_mse'] < metrics['genesis_selector_mse']
    # One exact complete experiment is reproduced, and all archive entries have
    # their measurements re-derived from retained checkpoints during restore.
    replay_exact = reproduce_experiment(train_records[0])
    archive_snapshot = archive.snapshot()
    archive_restored = DiscoveryArchive.restore(archive_snapshot).snapshot() == archive_snapshot
    report = {
        'schema':'axm.innovation-benchmark/v1', 'selector_seed':selector_seed,
        'environment':{'python':platform.python_version(),'implementation':platform.python_implementation(), 'platform':platform.system()},
        'training_tasks':2*training_per_condition, 'evaluation_tasks':len(rows),
        'feature_names':list(FEATURES),'operators':[m.to_dict() for m in OPERATORS],
        'selector_training_objective':'validation MSE only', 'evaluation_metric':'held-out MSE',
        'training_seed_range':[training_start,training_start+training_per_condition-1],
        'evaluation_seed_range':[evaluation_start,evaluation_start+evaluation_per_condition-1],
        'metrics':metrics,'rows':rows, 'experiment_reproduction_exact':replay_exact,
        'archive_restore_exact':archive_restored,'archive_sha256':archive_snapshot['sha256'],
        'selector_genesis':genesis,'selector_after':selector.snapshot(),
        'adoption_state':'archive-only','automatic_adoption':False,
        'limits':['Synthetic supervised scalar functions and three learning-rate operators only.',
                  'Held-out task instances, not unseen task families or UC/Windows validation.',
                  'Not open-ended invention, general intelligence, a latency benchmark, or automatic parent replacement.',
                  'Regressions remain separate; lower task error does not erase forgetting costs.'],
    }
    return report, archive_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--selector-seed', type=int, default=19)
    args = parser.parse_args()
    report, archive = benchmark(selector_seed=args.selector_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        args.archive.write_text(json.dumps(archive, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report['metrics'], indent=2))
    if not report['experiment_reproduction_exact'] or not report['archive_restore_exact']:
        raise SystemExit('reproduction or restoration failed')


if __name__ == '__main__': main()
