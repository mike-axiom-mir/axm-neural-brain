# Neural innovation laboratory

This is the first executable slice of [the innovation-engine research direction](research/NEURAL_INNOVATION_ENGINE.md).
It integrates the existing recurrent core, finite-state fixes, hybrid organ workspace,
learning probes and persistence/lineage implementations. It adds a bounded laboratory
and a neural experiment selector. Existing core equations and snapshot formats remain
unchanged; all named experimental histories are retained in the integration commit.

## What runs

`neural.axm_brain.innovation` provides:

- Isolated descendants from an exact, validated parent checkpoint.
- Explicit changes to learning rate, replay capacity, replay passes and replay scale.
- Composition of nonconflicting changes; conflicting settings are rejected.
- Equal-data baseline and candidate training with an explicit host update budget.
- Separate training, validation, held-out and earlier-task regression measurements.
- Checkpoints, restoration checks, experience counts and storage/parameter costs.
- Reproducible experiment records and a hash-linked archive retaining failed or
  regressive outcomes. Cosmetic names and redundant settings do not multiply the
  effective-configuration index. This is configuration deduplication, not proof of
  semantic novelty or invention.
- A neural chooser trained on prior validation outcomes. Unseen evaluation groups
  are required by default. Its learned state, feature meanings, operator order and
  training receipt references survive restart.

The task record retains the original prompt separately from any interpretation.
Autonomous numeric experiments use a null prompt. The caller supplies all data;
the library has no shell, network or automatic persistence behavior.

## Run the evidence

From the repository root, using Python 3.11 or later:

```sh
python -m unittest discover -s neural/axm_brain/tests -v
python -m unittest discover -s tests -v
python -m unittest discover -s evaluation -v
python -m evaluation.verify_innovation --output innovation-evidence.json
```

For a complete restorable archive from the default cohort:

```sh
python -m evaluation.innovation_benchmark --output experiment-report.json --archive experiment-archive.json
```

These commands write only the output paths explicitly requested by the caller.
The archive contains actual parent and descendant states; the compact committed
evidence receipt includes hashes and an exact reconstruction procedure instead
of duplicating every checkpoint in Git.

## Measured result and limits

The committed [receipt](verification/2026-09-25-neural-growth/innovation-evidence.json)
contains all three predetermined cohorts and all 36 held-out task instances.
The laboratory uses two known synthetic conditions: adapting to a changed numeric
function and handling noisy teaching. The chooser receives only training-stream
roughness and parent training error as task features. It learns from validation
errors; held-out labels never train it or determine its training order.

Across those cohorts, mean held-out squared error was:

| Choice method | Mean squared error |
| --- | ---: |
| Learned chooser | 0.00798653 |
| Best fixed operator chosen using training-task validation | 0.03569009 |
| Seeded random choice | 0.12767746 |
| Exact uniform-random expectation | 0.23459776 |
| Untrained chooser | 0.10652175 |
| Exhaustive held-out oracle, for comparison only | 0.00342073 |

The learned chooser beat the fixed operator and random expectation in each cohort.
One untrained initialization happened to beat its trained descendant; this result
is retained in the receipt. Learning is therefore not guaranteed to improve every
initial state. Earlier-task forgetting is also recorded separately: an improvement
on the new task does not erase a regression on the old one.

This earns a narrow result: prior experiment evidence can improve selection among
three known learning-rate settings on new instances of these two laboratory
conditions. It does not establish unseen-domain transfer, new topology invention,
UC integration, Windows/WALDO training, or open-ended self-improvement. Other
supported replay mutations and their composition have mechanical tests, not this
same learned-selection performance claim.

## Roots in the implementation

- Truth: split datasets, no-op controls, explicit failures, recomputable metrics,
  exact replay checks and bounded claims.
- Agency: every result is archive-only; a selected candidate cannot replace the
  continuing parent or authorize host execution.
- Continuity: parent state is preserved, descendants retain state and provenance,
  and the chooser and archive restore independently.
- Wisdom: small declared mutations, measured comparisons and a host resource
  budget precede broader architecture search.

The next research step is transferring this mechanism to an independently defined
task family and testing richer mutations. Topology changes require an explicit
state-migration contract; this version does not guess one.

## Direct simulation learning added from the latest research

The newly expanded research treats simulation as a source of experience.
`neural.axm_brain.simulation` now consumes verified micro-dynamics transitions from
the sibling network repository and teaches a cloned brain immediately after each
batch. Packets retain their simulation source. The continuing parent stays intact.

Using the sibling network checkout:

```sh
PYTHONPATH=.:../axm-neural-network/src python -m unittest discover -s integration -v
PYTHONPATH=.:../axm-neural-network/src python -m integration.run_micro_evidence --output micro-learning.json
```

Add `--full` to the last command to retain complete brain checkpoints and each
batch's before/after checkpoint references. The compact receipt retains their
fingerprints and reconstruction parameters.

Three brain seeds each learned from 864 direct simulated transitions. On 12 unseen
simulator seeds, prediction MSE fell from 0.24713 to 0.00432, from 0.14307 to
0.00106, and from 0.04945 to 0.00219. Each learned state restored exactly and kept
the same held-out behavior. Full learning-loop throughput was about 1,400–1,800
transitions/second locally, including validation, learning and checkpoint evidence.
This is a different unit of work from simulator-only throughput.

This first path uses a fixed host-defined action curriculum. It does not yet learn
which simulation family to choose or establish transfer to external observations.
Measured timings are machine/load dependent and are retained without speed claims
for other hardware. Training transition budgets and disjoint evaluation seeds are
checked before any candidate learning starts.
