# Neural Innovation Engine

**Status: research architecture / not yet an earned capability**

This note defines a direction for teaching an AXM neural system to improve **how it searches for useful neural changes**, rather than only training one fixed network harder.

It does not claim that AXM Neural Brain can currently innovate, self-redesign, discover superior neural architectures, or safely adopt its own proposals. Those claims require behavioral evidence from implemented experiments.

## Why this belongs here

This is a brain-level mechanism, not a Universal Creation-specific feature.

- `axm-neural-brain` owns the general neural brain, learning state, checkpoints, rollback, and future brain-internal learning mechanisms.
- `axm-uc-neural` is a consumer/bridge experiment that can later expose UC experiences to the neural brain.
- `axm-neural-network` is the research surface for lower-level neural mechanisms, frameworks, and possible AXM-owned substrate work.
- Universal Creation remains free to use deterministic, neural, hybrid, recurrent, or other cognition. It should not become the canonical owner of a general neural innovation mechanism.

Therefore this design lives canonically in `axm-neural-brain`. Other repositories may reference it later without copying the architecture into several competing versions.

## Core idea

Ordinary learning asks:

> Given this brain, how should its learned state change from experience?

The innovation loop adds another question:

> Given the brain's observed limits and the history of earlier experiments, what neural change is worth testing next?

The long-term target is not uncontrolled self-modification. It is an evidence-producing experimental loop in which candidate neural changes are generated, tested in isolated descendants, compared, retained as reusable discoveries when justified, and only adopted through an explicit choice.

## Four nested loops

### Loop 1 — learning

```text
experience
-> update learned state
-> later behavior changes
```

This remains the minimum evidence boundary already used by AXM Neural Brain: a learning claim requires retained experience-caused change that alters later behavior under controlled comparison.

### Loop 2 — neural experimentation

```text
observed limitation
-> propose candidate neural change
-> instantiate isolated descendant
-> run controlled experience
-> compare with parent/baselines
-> preserve evidence
```

A candidate may change weights, routing, memory organization, topology, learning rules, curriculum, module composition, or another explicitly represented neural mechanism.

### Loop 3 — learning how to search

```text
experiment history
+ problem characteristics
+ measured outcomes
-> learn which classes of experiment are promising
-> choose better future experiments
```

The innovation mechanism should eventually learn from its own experiment history instead of sampling the neural design space blindly.

This is a meta-learning objective, but the useful unit is not a vague "creativity score." It is grounded evidence about which kinds of changes helped which kinds of problems under which conditions.

### Loop 4 — invention composition

```text
useful mechanism A
+ useful mechanism B
+ new problem context
-> candidate composition C
-> isolated test
-> reusable neural primitive if evidence survives
```

Many useful innovations may be compositions or adaptations of earlier discoveries rather than entirely new mechanisms.

## Proposed roles

These are architectural functions, not mandatory permanent agents or governance roles.

### Worker brain

The neural system that performs tasks and accumulates experience.

### Limitation observer

Produces explicit, inspectable observations such as:

- repeated failure modes;
- stalled learning;
- unusually expensive pathways;
- instability or forgetting;
- contexts where different strategies diverge;
- surprising successes worth reproducing.

An observer report is evidence input, not proof that a redesign is needed.

### Candidate generator

Produces bounded candidate changes from the current brain state, the observed limitation, available neural primitives, and prior experiment evidence.

Early implementations may use hand-authored mutation operators or simple search. A learned proposal model is a later stage, not a prerequisite.

### Isolated descendant

A candidate runs as a clone/descendant of a known checkpoint.

The live parent is never overwritten merely because a proposal exists or a benchmark improves.

### Experiment evaluator

Runs parent, candidate, and appropriate baselines against declared tests.

Evaluation dimensions should remain separately inspectable. A possible initial set is:

- task capability;
- generalization to held-out situations;
- adaptation speed;
- compute/memory cost;
- stability across restart/checkpoint restore;
- catastrophic forgetting/regression;
- novelty relative to archived mechanisms.

Do not collapse these into a single permanent reward number unless an experiment explicitly needs a temporary scalar objective. A scalar can hide trade-offs.

### Discovery archive

Stores reusable findings with lineage and evidence.

A retained discovery should describe at minimum:

- originating parent checkpoint;
- problem/limitation that motivated it;
- exact candidate change;
- training/experience conditions;
- comparison baselines;
- measured outcomes;
- regressions/failures;
- reproducibility state;
- whether the mechanism was adopted, rejected, or only archived.

Failed experiments can remain useful evidence, but failure does not become a capability.

### Meta-learner

Consumes experiment records and learns correlations between problem state, proposed change type, and measured result.

Its first useful milestone is not "invent a new brain." It is something smaller and testable, for example:

> Given several allowed mutation operators and a class of observed learning failures, select an operator that outperforms random selection across held-out experimental tasks.

That result would demonstrate learned improvement in the **search process** itself.

## Simulation as a native learning organ

The innovation loop becomes substantially more interesting if simulation is not treated only as an external benchmark after a candidate has been built.

A machine can use many small simulations as a **direct source of experience**:

```text
current learned state
-> select or construct a bounded micro-simulation
-> predict / choose / act
-> run state transition
-> observe consequence
-> encode the result as experience
-> update learned state
-> choose the next simulation differently
-> repeat
```

The important research hypothesis is not that AXM already runs millions of useful simulations. The hypothesis is that **machine-scale simulation throughput can become a native learning channel**, with the achieved rate measured rather than assumed.

A micro-simulation does not need graphics, a full game engine, or a realistic world. It can be as small as:

- one state transition;
- one routing decision;
- one memory-retrieval conflict;
- one action/consequence pair;
- one candidate neural mechanism under one controlled condition;
- one deterministic software workflow with varied inputs;
- one small procedural environment.

This makes simulation closer to a synthetic nervous system than a post-hoc test harness.

### Keep experience sources distinct

Simulation can teach useful structure while also teaching the wrong structure if its world is incomplete.

The brain therefore needs to retain where an experience came from:

```text
experience_source =
  external_observation
  deterministic_simulation
  learned_world_model
  replay
  imported_dataset
```

A simulated result must never silently become evidence that the same thing happened externally.

This source boundary also makes transfer measurable:

```text
learn in simulation
-> test on unseen simulations
-> test on external / higher-fidelity conditions
-> measure what transferred and what failed
```

### Three simulation layers

The research direction can begin with the cheapest and most inspectable layer.

#### 1. Deterministic micro-simulations

Explicit state-transition rules with reproducible seeds.

These are useful for proving the complete learning loop because a failed result can be reconstructed exactly.

#### 2. Procedural simulation families

One simulator can generate many related situations by varying initial state, constraints, topology, noise, opponents, resources, or other declared factors.

This lets the learner experience a distribution instead of memorizing one fixed scenario.

#### 3. Learned world-model rollouts

A later neural model may learn to predict possible state transitions and generate internal rollouts.

These are potentially powerful but must remain labeled as model-produced predictions rather than external facts.

The learned simulator itself can later become an object of experimentation: compare its predictions against grounded outcomes, find where it is weak, and improve it.

### Simulation-selection learning

The deeper loop is not merely "run more simulations."

It is:

> learn which simulation is worth running next.

A meta-learner can observe:

- current uncertainty;
- recurring failures;
- areas with little experience;
- disagreement between candidate mechanisms;
- earlier simulations that produced large learning gains;
- simulations whose lessons transferred well to held-out conditions.

It can then choose a next experiment expected to be informative.

The claim to earn is narrower than "machine curiosity" or "autonomous science":

> Given a fixed simulation budget, learned simulation selection improves later held-out behavior compared with random or fixed simulation selection.

### Neural innovation through simulation

Candidate neural changes can use the same fabric:

```text
parent checkpoint
-> candidate neural change
-> isolated descendant
-> many bounded micro-simulations
-> direct learning inside descendant
-> held-out comparison
-> archive causal evidence
-> explicit adoption decision
```

This changes the role of simulation from evaluator to **experience generator**.

A useful candidate may emerge not because a single architecture scored higher immediately, but because it learned more effectively across a family of simulated experiences.

### Resource truth

Simulation is not free.

The system should measure:

- simulations or state transitions per second;
- CPU/GPU/RAM cost;
- learner-update cost;
- storage written;
- wall-clock time;
- duplicate/uninformative simulation rate;
- transfer to held-out conditions.

A claim such as "millions of simulations" is meaningful only when the simulation unit and measured hardware throughput are stated.

The goal is not maximal simulation count. It is useful experience per available compute.

### Simulation can become a method of thinking

The long-term possibility is a brain that does not wait for all useful experience to arrive from the external world.

It can learn a bounded pattern like:

```text
notice uncertainty
-> construct possible situation
-> simulate consequence
-> compare possibilities
-> choose external action
-> observe reality
-> update both task knowledge and simulation quality
```

If this is eventually demonstrated, simulation is no longer merely a training utility around the brain. It has become part of the brain's learned cognitive process.

## Candidate experiment contract

Every innovation attempt should be reconstructible from an explicit experiment record.

A minimal conceptual record:

```json
{
  "parent_checkpoint": "...",
  "problem_observation": "...",
  "hypothesis": "...",
  "candidate_change": {
    "kind": "...",
    "parameters": {}
  },
  "experience_set": "...",
  "held_out_set": "...",
  "baselines": ["parent", "random-or-control"],
  "evaluation_dimensions": [
    "capability",
    "generalization",
    "adaptation_speed",
    "cost",
    "stability",
    "regression"
  ],
  "result": "...",
  "lineage": "...",
  "adoption_state": "archive-only"
}
```

The exact schema can change after implementation work begins. The important property is causal traceability: the system must be able to distinguish what was changed, what experience occurred, and what behavior followed.

## Adoption boundary

Discovery and adoption are separate operations.

```text
discover
-> reproduce
-> cross-test
-> inspect regressions
-> archive evidence
-> explicit adoption decision
```

A locally better candidate must not silently replace the continuing parent.

This preserves the existing AXM separation between learned growth and canonical software truth, provenance, verification, checkpoints, and rollback.

## Relationship to Genesis and checkpoints

Innovation experiments need a stable reference state.

- A Genesis state can define the reproducible pre-learning starting point for a class of experiments.
- Ordinary checkpoints capture learned descendants.
- Candidate innovations branch from a named checkpoint.
- Parent and descendant must remain independently restorable.
- Comparison must record which starting state each run used.

This allows AXM to distinguish improvement caused by a candidate architecture from improvement caused merely by different accumulated experience.

## Relationship to UC

UC can later provide a rich experience environment for this engine:

```text
UC creation/use
-> explicit experience bridge
-> neural observation/learning
-> limitation becomes measurable
-> neural candidate experiment
-> descendant tested
-> evidence returned
```

But UC should not automatically accept a neural discovery as canonical creation machinery.

The bridge may expose experience and consume learned behavior while the neural brain retains its own experimental lineage.

## First implementation sequence

### Stage 0 — preserve evidence contracts

Before innovation search, ensure the current neural brain can:

1. start from a reproducible state;
2. learn from direct experience;
3. persist learned state;
4. restore checkpoints;
5. compare behavior before and after learning.

Without this, innovation experiments cannot distinguish actual improvement from noise.

### Stage 1 — deterministic mutation laboratory

Implement a very small allowed set of neural mutations.

Examples:

- hidden-width change;
- memory-capacity change;
- routing choice;
- optimizer/learning-rate policy;
- optional recurrent or episodic-memory module.

Generate candidates deterministically or by seeded random search. This validates experiment plumbing before adding a learned innovator.

### Stage 2 — novelty and lineage archive

Retain structured experiment evidence and detect when a candidate is materially different from already tested mechanisms.

Do not archive color-variant-equivalent neural configurations as separate "inventions." Novelty must correspond to a meaningful mechanism or behavior difference.

### Stage 3 — learned experiment selector

Train a small meta-model to select among known candidate operators using prior experiment outcomes.

Compare it against:

- random operator choice;
- fixed heuristic choice;
- exhaustive small search where feasible.

The claim to earn is narrow: **past experiment evidence improves future experiment selection**.

### Stage 4 — mechanism composition

Allow the selector/generator to compose previously useful neural primitives into new bounded candidates.

Composition still goes through isolated descendant testing and the same evidence contract.

### Stage 5 — open-ended architecture proposals

Only after earlier stages are reproducible should the system attempt broader topology or learning-rule generation.

At that point the research question becomes whether the system can generate mechanisms outside its initial hand-authored combinations while remaining inspectable and reproducible.

## What would count as progress

Useful evidence includes:

- a candidate consistently outperforming its parent on a declared limitation without unacceptable regression;
- the result reproducing from the same parent checkpoint and experiment definition;
- the mechanism transferring to held-out tasks;
- the meta-learner selecting useful experiments better than non-learned baselines;
- composed archived mechanisms solving a problem that either mechanism alone did not solve.

## What would not count

The following are insufficient by themselves:

- weights changed;
- a network got larger;
- a mutation loop ran;
- training loss decreased;
- one lucky benchmark increased;
- the system emitted a novel-looking architecture;
- the candidate passed only the environment it was generated against;
- a meta-model learned to imitate the experiment history without improving future search.

## Main failure modes to watch

### Benchmark capture

The innovator may learn to exploit the evaluator instead of improving broadly.

Use held-out situations, multiple dimensions, and regression checks.

### Complexity inflation

If larger candidates receive more opportunity, the system may equate complexity with innovation.

Track compute, memory, and mechanism simplicity separately from task score.

### Experiment-history overfitting

A meta-learner may memorize which mutations worked in old tasks.

Evaluate experiment selection on new problem instances and, later, new task families.

### False novelty

Different parameterizations may be behaviorally equivalent.

Novelty should distinguish meaningful mechanism/behavior differences from superficial configuration changes.

### Parent destruction

Direct mutation of the live brain would make failures difficult to diagnose and could erase useful learned state.

All architectural candidates begin as isolated descendants.

## AXM root fit

**Truth:** innovation claims must be tied to reproducible behavioral evidence.

**Agency / non-domination:** the innovator proposes and tests; it does not gain automatic authority over the continuing machine.

**Continuity:** parent state, descendant lineage, checkpoints, and regressions remain inspectable and restorable.

**Wisdom before speed:** broad open-ended redesign comes after smaller experiment-selection claims are earned.

## Long-term research question

The deepest target is not merely a network that learns.

It is a neural system that can accumulate an inspectable **science of its own learning mechanisms**:

```text
experience
-> discover limitation
-> design experiment
-> test descendant
-> retain causal evidence
-> learn which experiments are worth trying
-> compose proven mechanisms
-> generate better experiments
```

If implemented successfully, past neural invention becomes useful input to future neural invention without requiring automatic self-replacement or abandoning AXM's evidence and recovery boundaries.
