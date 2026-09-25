# Neural Substrate + Hybrid Architecture Findings

**Date:** 2026-09-25  
**Status:** research note / direction input, not proof of learning  
**Repository context:** general AXM neural-brain architecture  
**AXM roots:** Truth before story · Agency / non-domination · Continuity · Wisdom before speed

## Why this note exists

The current AXM direction separates three things that are often collapsed into one word ("AI"):

1. **Deterministic software capability** — software can already contain tools, rules, workflows, search spaces, verification, composition, and creative/exploratory modes.
2. **Intent translation** — a human request has to be converted from ambiguous language into machine-usable goals, constraints, parameters, and operations.
3. **Learned adaptation** — a neural layer can change internal parameters from experience and later influence decisions.

The key experiment is therefore not "put an AI inside software." It is closer to:

```text
working deterministic software
        ↓
real software execution / creation experience
        ↓
structured trace of what caused the run and what happened
        ↓
initially untrained neural layer
        ↓
retained learned state
        ↓
later controlled comparison of behavior
```

The deterministic software remains independently usable. The neural layer is an optional adaptive substrate, not the source of the original software capability.

## Genesis means zero learned experience, not necessarily zero numeric weights

For a normal neural network, setting every weight to literal zero can prevent useful symmetry breaking. A cleaner AXM definition of **Genesis** is:

- zero accumulated AXM experience;
- zero learned UC/task-specific behavior;
- a fixed, reproducible initialization state;
- a saved checkpoint from before learning begins.

That gives us a real comparison point without pretending every parameter must numerically equal zero.

## PyTorch, JAX, and MLX are software substrates, not learned intelligence

A useful mental model:

```text
AXM architecture + learning rule + experience design
                    ↓
        PyTorch / JAX / MLX
                    ↓
      tensor math / autodiff / compilation
                    ↓
             CPU / GPU / TPU
```

These frameworks provide machinery such as arrays/tensors, matrix operations, automatic differentiation, compilation, accelerator support, and checkpoint-friendly primitives. They do **not** supply the learned AXM behavior by themselves.

### PyTorch

PyTorch provides tensors, neural-network building blocks, and an automatic-differentiation engine (autograd). Autograd records computation needed to calculate gradients so parameters can be updated during training.

AXM implication: PyTorch is a valid experimental calculator/runtime while we test the architecture. Using it does not mean the framework supplied the learned behavior.

### JAX

JAX is a numerical-computing library centered on array operations and program transformations such as:

- `grad` for automatic differentiation;
- `jit` for compilation;
- `vmap` for vectorization.

It can target CPU, GPU, and TPU through its compilation stack.

AXM implication: JAX is interesting as a comparatively low-level mathematical substrate because it keeps the boundary between "our model/learning logic" and "execution machinery" unusually visible.

### MLX

Apple's MLX is an open-source array framework for Apple Silicon aimed at machine-learning research, training, and fine-tuning.

AXM implication: the same high-level AXM learning contract could potentially sit over multiple backends rather than being fused to one framework forever.

## Big systems also use framework layers

Using a framework is not unusual or a shortcut unique to small projects.

- Meta describes PyTorch as foundational to its AI workloads and infrastructure.
- Anthropic publicly lists PyTorch, JAX, and Triton among development frameworks used for current Claude systems.
- Apple exposes MLX as its open-source array framework for Apple Silicon.
- OpenWALDO currently trains through MLX on Apple Silicon and PyTorch on supported Linux CPU/NVIDIA/AMD systems.

The useful AXM question is therefore not "can we avoid all frameworks immediately?" It is:

> Can we make the framework boundary explicit, swappable, inspectable, and eventually replaceable where owning that layer creates real value?

## Existing research already proves important pieces

### AlphaDev

Google DeepMind's AlphaDev uses reinforcement learning to search over low-level assembly instructions. Candidate programs are executed, checked for correctness, and rewarded for correctness and latency.

Relevant lesson: a learner can improve behavior by acting through a deterministic software execution environment and receiving grounded outcome signals.

### OpenWALDO provenance

OpenWALDO records model compose, training inputs, run history, checkpoints, outputs, hashes, and BOM-style lineage.

Relevant lesson: if we care about what a learning system became, keeping its experience and provenance attached is as important as keeping the final weights.

These are precedents for pieces of the AXM experiment. They are **not** evidence that the exact AXM "neural brain underneath independently capable deterministic software" arrangement has already been proven.

## The AXM hybrid opportunity

AXM can deliberately keep multiple mechanisms instead of forcing every problem through one layer:

```text
Human
  │
  ├─ structured controls / pictograms
  ├─ deterministic intent grammar
  └─ optional language-model interpreter
          │
          ▼
Deterministic UC capability fabric
          │
          ├─ normal execution
          ├─ creative/exploratory execution
          └─ verification
          │
          ▼
Experience + provenance recorder
          │
          ▼
Neural adaptive layer
          │
          └─ later influences choices through explicit gates
```

The interpreter, creator, verifier, recorder, and learner should remain separable. That keeps failures diagnosable.

## The learning trace must preserve cause, not only actions

If the neural layer sees only tool calls, it may learn patterns of *usage* without learning what intent caused that usage.

A useful event contract should preserve at least:

```text
source_event
raw_human_prompt          # nullable for autonomous creative runs
interpreted_intent        # nullable if no interpreter was involved
context_refs
selected_capabilities
parameters_and_actions
artifact_or_result_refs
verification_results
learning_signal
checkpoint_before
checkpoint_after
runtime_environment
```

For no-prompt creative experiments:

```text
source_event = "creative_autonomous"
raw_human_prompt = null
```

Do not silently overwrite the source prompt with an AI interpretation. They are different evidence and may disagree.

## Why the no-prompt creative test is useful

The current UC creative mode can test capability without mixing in human-language understanding.

That helps separate two questions:

1. Can the system create/use capabilities?
2. Can the system correctly understand a human request?

If both are tested at once and the result is poor, the failure source becomes ambiguous.

A later **Neural Creative Mode**, default OFF, can provide a controlled "show what the learned state does" test without requiring a human prompt.

## Reproducibility warning

"Deterministic software" and "neural framework" do not automatically imply bit-for-bit reproducible training.

GPU kernels, parallel reductions, random initialization, library versions, and hardware can introduce nondeterminism. Controlled AXM comparisons should therefore record:

- random seeds;
- framework and dependency versions;
- CPU/GPU model and driver/runtime;
- deterministic-algorithm settings where supported;
- exact Genesis checkpoint;
- exact experience stream;
- checkpoint hashes;
- verification outputs.

Claims should be behavioral and evidence-based, not inferred merely because weight files changed.

## Possible ownership ladder

Do **not** block the current experiment on rebuilding framework infrastructure.

A reasonable long path is:

### Stage A — framework-backed experiment

Use PyTorch/JAX/MLX where useful while AXM owns:

- experience schema;
- learning objective;
- architecture choices;
- checkpoints;
- provenance;
- behavioral tests;
- software bridge.

### Stage B — minimal AXM neural kernel

Reimplement only the small subset actually required:

- tensor/array representation;
- forward operations;
- activations;
- loss calculation;
- gradients/backprop or another chosen learning rule;
- optimizer/update rule;
- serialization/checkpoints.

This is much smaller than recreating all of PyTorch or JAX.

### Stage C — multiple interchangeable backends

Run the same AXM experiment contract against:

- PyTorch backend;
- JAX backend;
- future AXM-owned backend.

Compare behavior, speed, reproducibility, and correctness.

### Stage D — deeper compute ownership where justified

Only after profiling proves a reason, move selected kernels or compute paths lower into C/C++/GPU/custom compute.

## Full-puzzle goal

A useful long-term AXM research goal is **full-stack comprehension**:

```text
human intent
→ translation
→ deterministic software
→ experience record
→ neural representation
→ learning rule
→ tensor operations
→ compiler/runtime
→ hardware execution
→ resulting behavior
```

The goal is not to claim that AXM already understands more than existing framework or model teams. The goal is to keep enough of the complete chain together that we can progressively inspect, replace, hybridize, and understand each layer without losing the relationships between them.

## Questions for later experiments

- Does a neural layer trained on source → action → outcome outperform one trained on action → outcome alone?
- What should count as a learning signal when UC is in autonomous creative mode?
- Which decisions should the neural layer be allowed to influence first?
- Can the same Genesis checkpoint produce measurably different descendants under different experience histories?
- Can learned behavior survive restart while deterministic canonical truth remains unchanged?
- How much of the framework stack does AXM actually need before an owned minimal runtime becomes worthwhile?
- Can deterministic communication mappings grow from repeated grounded intent without requiring an LLM for familiar cases?
- What evidence is sufficient to say the system learned rather than merely replayed or memorized?

## Sources

Primary / official references checked 2026-09-25:

1. PyTorch — Autograd mechanics  
   https://docs.pytorch.org/docs/stable/notes/autograd

2. PyTorch — Neural-network parameter initialization  
   https://docs.pytorch.org/docs/stable/nn.init.html

3. JAX — How to think in JAX  
   https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html

4. JAX — JIT compilation  
   https://docs.jax.dev/en/latest/201/jit.html

5. JAX — Automatic differentiation  
   https://docs.jax.dev/en/latest/automatic-differentiation.html

6. Apple — AI & Machine Learning / MLX  
   https://developer.apple.com/machine-learning/

7. Meta Engineering — Building Meta's GenAI Infrastructure  
   https://engineering.fb.com/2024/03/12/data-center-engineering/building-metas-genai-infrastructure/

8. Anthropic — Transparency Hub  
   https://www.anthropic.com/transparency

9. Google DeepMind — AlphaDev discovers faster sorting algorithms  
   https://deepmind.google/blog/alphadev-discovers-faster-sorting-algorithms/

10. OpenWALDO — Training  
    https://openwaldo.org/training/

11. OpenWALDO — FAQ  
    https://openwaldo.org/faq/

## Evidence boundary

This note is a research map, not proof that AXM has already achieved continual learning, creativity growth, self-teaching, or a superior architecture.

Those claims are earned only by controlled runs showing retained behavioral change attributable to experience.
