"""Three predetermined disjoint train/evaluation cohorts; preserve every result."""
import argparse
import json
from pathlib import Path

from .innovation_benchmark import benchmark


CASES = ((19, 100, 1000), (29, 200, 2000), (43, 300, 3000))


def verify():
    reports = []
    for seed, train, evaluation in CASES:
        report, _ = benchmark(selector_seed=seed, training_start=train, evaluation_start=evaluation)
        report['selector_genesis_sha256'] = report.pop('selector_genesis')['sha256']
        report['selector_after_sha256'] = report.pop('selector_after')['sha256']
        reports.append(report)
    keys = ('learned_mse', 'genesis_selector_mse', 'fixed_mse', 'random_mse',
            'uniform_random_expected_mse', 'exhaustive_oracle_mse')
    means = {k:sum(r['metrics'][k] for r in reports)/len(reports) for k in keys}
    return {'schema':'axm.innovation-evidence/v1', 'reports':reports, 'mean_metrics':means,
            'held_out_task_instances':sum(r['evaluation_tasks'] for r in reports),
            'all_archives_restore':all(r['archive_restore_exact'] for r in reports),
            'all_experiments_reproduce':all(r['experiment_reproduction_exact'] for r in reports),
            'all_parent_states_preserved':all(row['parent_unchanged'] for r in reports for row in r['rows']),
            'all_choices_retained':all(row['restart_choice_equal'] for r in reports for row in r['rows']),
            'scope':'Three synthetic numeric cohorts, two known laboratory conditions, three learning-rate operators; no UC or unseen-domain claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='reports'}, indent=2))
    if not all(receipt[k] for k in ('all_archives_restore','all_experiments_reproduce','all_parent_states_preserved','all_choices_retained')):
        raise SystemExit('continuity evidence failed')


if __name__ == '__main__': main()
