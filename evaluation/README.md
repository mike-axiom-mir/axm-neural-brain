# AXM Neural Learning Evaluation

This directory is the independent proof/falsification surface for the dedicated AXM Neural Brain.

It does not own the learning kernel. The Neural Core specialist owns the implementation. This surface asks whether experience produces retained, observable behavior change under controlled comparison and whether that evidence survives persistence and consolidation boundaries.

## Seed provenance

Current donor inspected for the first evaluation surface:

- repository: mike-axiom-mir/axm-uc-neural
- PR: #3, AXM direct-learning neural brain v0.1
- donor head: 163410ce05a42879dca478d8ea0de7f6d17c4820
- donor branch: codex/axm-direct-brain-v0.1
- GitHub Actions run: 35953978949 (AXM direct brain), completed successfully
- test job: 107488257957, completed successfully

The donor is evidence and provenance, not a second implementation to fork indefinitely.

## Evidence language

Every claim uses one of these labels:

- TESTED — an executable test ran and passed against the named revision.
- MEASURED — a numeric or behavioral result was produced from a controlled probe.
- INFERRED — supported by code/tests but not directly exercised by the claimed probe.
- NOT TESTED — no adequate direct evidence yet.

Parameter change alone is never accepted as evidence of useful learning.

## Required proof surfaces

The integration suite in test_learning_evidence.py is intentionally implementation-light and tries both axm_brain and neural.axm_brain package layouts so the evaluation surface can follow the dedicated repo without dictating its final package layout.

It covers or prepares direct checks for controlled supervised before/after behavior, retained behavior after snapshot restore, replay usefulness versus an otherwise identical no-replay control, deterministic replay, reward-caused retained behavior change, wake/sleep behavioral effect rather than weight drift alone, interference/catastrophic-forgetting measurement, and continuation equivalence after checkpoint restore.

If the dedicated core has not landed yet, integration probes are reported as skipped / NOT TESTED, not silently treated as success.

## Truth boundary

A passing probe establishes only the behavior named by that probe on the tested revision and fixture. It does not establish intelligence, semantic understanding, generalization, safe autonomy, or useful long-horizon continual learning.
