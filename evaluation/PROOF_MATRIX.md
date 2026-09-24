# Evidence matrix — donor PR #3 inspection

Revision inspected: mike-axiom-mir/axm-uc-neural@163410ce05a42879dca478d8ea0de7f6d17c4820.

The donor's AXM direct brain GitHub Actions workflow completed successfully on that revision.

| Surface | Status | Evidence / boundary |
|---|---|---|
| deterministic same-seed birth | TESTED | donor test test_same_seed_same_birth is in the successful workflow |
| supervised experience changes later response | TESTED | donor test test_direct_teaching_changes_response checks output direction, not only weights |
| snapshot integrity / exact round-trip state | TESTED | donor test_snapshot_roundtrip and tamper rejection ran successfully |
| restart persistence of later behavior | INFERRED | exact state round-trip is tested, but an explicit restart-then-behavior comparison is not present |
| wake resets transient recurrent state | TESTED | donor test_wake_resets_transient_recurrent_state |
| sleep causes parameter change | TESTED | donor test_sleep_replays_memory |
| sleep improves or retains useful behavior | NOT TESTED | current donor sleep test accepts weight change; that is insufficient for a usefulness claim |
| replay is useful versus no-replay control | NOT TESTED | no controlled counterfactual in donor tests |
| reward update changes parameters | TESTED | donor test_reward_learning_changes_weights |
| reward experience causes retained behavior change | NOT TESTED | donor does not compare behavior before and after reward learning |
| deterministic replay / consolidation | NOT TESTED | no twin-run snapshot or behavior equality probe is present |
| catastrophic forgetting / interference | NOT TESTED | no A-to-B-to-retest-A probe is present |
| checkpoint restore preserves future trajectory | NOT TESTED | round-trip state is tested, but continue-from-checkpoint equivalence is not |
| retained behavior change caused by experience | TESTED (supervised only) | supervised fixture demonstrates directionally changed response; reward, sleep and replay remain open |

## Current target-repo state

At creation of this evaluation branch, mike-axiom-mir/axm-neural-brain@173150be94c0a3f89a80ad8786c2b7c660f7f1fb contains the project README but not yet the extracted neural core. Therefore dedicated-repo integration results remain NOT TESTED until the Neural Core specialist lands the implementation.
