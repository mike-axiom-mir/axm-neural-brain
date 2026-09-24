# AXM Neural Brain

**Status: PRODUCTION / HEAVY EXPERIMENTAL**

This repository is the dedicated AXM neural-brain substrate. The first core extraction is deliberately conservative: it lifts the already-tested direct-learning kernel from `mike-axiom-mir/axm-uc-neural` PR #3 into a software-body-neutral home instead of inventing a second architecture.

## Core boundary

The neural core provides deterministic from-zero initialization, recurrent hidden state, online supervised learning, reward-modulated eligibility learning, bounded replay memory, explicit wake/sleep consolidation, complete SHA-256 checked snapshots, named I/O contracts with stable fingerprints, read-only mechanical state analysis, CREATE / USE / PLAY / LEARN / DISCOVER experience tags, and the canonical AXM root contract embedded in snapshots.

The neural core has no network, shell, tool, or autonomous filesystem authority. Host software owns permissions, persistence locations, observations, actions, and execution. Network coordination belongs in `axm-neural-network`, not here.

## Provenance

The extracted kernel is pinned to:

- donor repo: `mike-axiom-mir/axm-uc-neural`
- donor PR: #3, `codex/axm-direct-brain-v0.1`
- donor commit: `163410ce05a42879dca478d8ea0de7f6d17c4820`

Exact donor blob identities are recorded in `provenance/direct-brain-donor-v0.1.json` and checked by executable tests.

The package initializer is intentionally narrower than the donor initializer: Genesis/Continuity implementations are not duplicated into this Neural Core lane because persistence/lineage is a separate convergence lane. The core keeps the canonical root contract required by snapshots.

## Evidence boundary

Changing weights is not proof of useful learning. Current tests establish deterministic mechanics, serializable state, explicit contract binding, and bounded learning/replay behavior. Claims about useful learning, generalization, interference resistance, or long-horizon intelligence require independent behavioral evaluation.

WALDO remains an independent comparison/fallback and is not absorbed into this repository.

## AXM roots

Truth before story. Agency / non-domination. Continuity. Wisdom before speed.

Required Notice: Copyright 2026 Mike - Axiom/Mir.
