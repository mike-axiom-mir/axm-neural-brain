# AXM Neural Persistence, Lineage and Recovery

Status: specialist branch / heavy experimental.

This subsystem owns durable state semantics for the dedicated AXM Neural Brain. It deliberately does **not** own neural learning math.

## Donor provenance

The first reference implementation is the live donor work in:

- repository: mike-axiom-mir/axm-uc-neural
- PR: #3, AXM direct-learning neural brain v0.1
- donor branch: codex/axm-direct-brain-v0.1
- donor head inspected for this extraction: 163410ce05a42879dca478d8ea0de7f6d17c4820
- donor files inspected: neural/axm_brain/state.py, genesis.py, continuity.py, roots.py, core.py and their tests

The dedicated repo must retain this provenance instead of presenting extracted semantics as newly invented.

## Boundary

Neural Core owns learning behavior, recurrent state and learning rules. This module owns how admitted states are identified, verified, chained, restored, rolled back and preserved in history.

The integration contract is deliberately narrow: Neural Core supplies a complete state body and a core-specific birth validator. Persistence wraps that state in a versioned hashed envelope and records canonical lineage events without modifying neural internals.

## Durable invariants

1. **Complete snapshot identity** — snapshot schema, state schema and complete body are hashed together.
2. **Immutable Genesis** — G0 can only be admitted after explicit birth validation. Changing an admitted G0 requires a new lineage/version.
3. **Append-only canonical history** — admitted events are never silently rewritten or deleted.
4. **Explicit continuity decisions** — every canonical transition carries PRESERVE, SUPERSEDE or FORGET records. PRESERVE must pass. SUPERSEDE names a replacement. FORGET remains visible in provenance.
5. **Four-root transition evidence** — Truth, Agency, Continuity and Wisdom Before Speed must all have explicit PASS evidence before a transition can become canonical.
6. **Recovery is a new event** — rollback changes the active-state pointer but does not erase the state that was rolled back from.
7. **Branch provenance** — a new lineage may point to a parent lineage/state and must record why it forked.
8. **Restart equivalence** — serializing and restoring a journal must reconstruct the same active state and the same hashed event history.
9. **Nested integrity** — re-hashing an outer journal cannot hide a changed previously admitted event; each event verifies its own digest.
10. **No authority expansion** — this layer has no network, shell or autonomous filesystem authority. The host chooses where bytes are stored and when an admitted state is activated.

## Schemas

- axm.state-snapshot/v1
- axm.genesis-admission/v1
- axm.genesis-validation/v1
- axm.lineage-transition/v1
- axm.rollback-receipt/v1
- axm.lineage-journal/v1
- axm-roots/v0.1

Schema identity participates in hashes, so changing interpretation without versioning changes identity rather than silently reinterpreting old state.

## Recovery model

A lineage is event-sourced:

G0 -> S1 -> S2 -> rollback receipt -> S1 active -> S3 ...

The rollback receipt is appended after S2. S2 remains in history. A later S3 may use S1 as its parent state while still extending the event chain after the rollback receipt. This separates state ancestry from audit-event ancestry.

## Truth boundary

Hash equality proves byte-level canonical equivalence under the declared schema. It does not prove that the neural behavior is useful, intelligent or semantically equivalent outside the tested restore/continuity probes.

Required Notice: Copyright 2026 Mike - Axiom/Mir.
