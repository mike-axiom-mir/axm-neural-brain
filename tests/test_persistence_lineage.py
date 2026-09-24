import copy
import json
import unittest

from axm_persistence.core import (
    AXM_ROOTS,
    BirthValidation,
    ContinuityDecision,
    GenesisRecord,
    IntegrityError,
    LineageError,
    LineageJournal,
    RollbackReceipt,
    TransitionRecord,
    make_snapshot,
    verify_restore_equivalence,
    verify_snapshot,
)


def roots_pass():
    return {
        root: {
            "status": "PASS",
            "reason": f"fixture evidence for {root}",
            "evidence_refs": [f"fixture:{root}"],
        }
        for root in AXM_ROOTS
    }


def birth_validation():
    return BirthValidation(
        (
            ("brain_snapshot_integrity", True, "verified"),
            ("four_roots_exact", True, "verified"),
            ("pre_experience_neural_birth", True, "verified"),
            ("configuration_bound", True, "verified"),
            ("provenance_present", True, "verified"),
        )
    )


def snap(step=0, value=1):
    return make_snapshot(
        "axm-brain/v0.2",
        {"config": {"seed": 7}, "state": {"step": step, "value": value}},
    )


def genesis():
    return GenesisRecord.admit(
        lineage="fixture-lineage",
        snapshot=snap(),
        birth_validation=birth_validation(),
        provenance={
            "source_repo": "mike-axiom-mir/axm-uc-neural",
            "source_pr": 3,
            "source_head": "163410ce05a42879dca478d8ea0de7f6d17c4820",
        },
        identities_roles={"system": "AXM Neural Brain"},
        platform_assumptions={"runtime": "python"},
        admitting_authority="fixture-admission",
        commit_evidence="fixture:durable-commit",
    )


class PersistenceLineageTests(unittest.TestCase):
    def test_snapshot_roundtrip_restore_equivalence(self):
        original = snap(step=4, value=99)

        class Holder:
            def __init__(self, snapshot):
                self.snapshot = snapshot

        verified = verify_restore_equivalence(
            original,
            restore=lambda snapshot: Holder(snapshot),
            capture=lambda holder: holder.snapshot,
        )
        self.assertEqual(verified, original)

    def test_snapshot_integrity_detects_changed_body(self):
        damaged = copy.deepcopy(snap())
        damaged["body"]["state"]["step"] = 999
        with self.assertRaises(IntegrityError):
            verify_snapshot(damaged)

    def test_genesis_is_deterministic_and_integrity_checked(self):
        first = genesis()
        second = genesis()
        self.assertEqual(first.record_id, second.record_id)
        changed = first.to_dict()
        changed["provenance"]["source_pr"] = 999
        with self.assertRaises(IntegrityError):
            GenesisRecord.from_dict(changed)

    def test_failed_birth_validation_cannot_be_admitted(self):
        validation = BirthValidation(
            (
                ("snapshot_integrity", True, "verified"),
                ("pre_experience_neural_birth", False, "experience exists"),
            )
        )
        with self.assertRaises(LineageError):
            GenesisRecord.admit(
                lineage="bad",
                snapshot=snap(),
                birth_validation=validation,
                provenance={"source": "fixture"},
                admitting_authority="fixture",
                commit_evidence="fixture:commit",
            )

    def test_preserve_requires_passing_result(self):
        with self.assertRaises(LineageError):
            ContinuityDecision("anchor", "PRESERVE", "REGRESSION", "regressed")

    def test_supersede_and_forget_remain_explicit_records(self):
        g0 = genesis()
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=g0.record_id,
            parent_state_id=g0.record_id,
            parent_snapshot=g0.snapshot,
            next_snapshot=snap(step=1, value=2),
            continuity=(
                ContinuityDecision(
                    "old-probe",
                    "SUPERSEDE",
                    "REGRESSION",
                    "new probe explicitly replaces this behavior",
                    ("evidence:old",),
                    replacement_probe_id="new-probe",
                ),
                ContinuityDecision(
                    "obsolete-probe",
                    "FORGET",
                    "NOT_TESTED",
                    "behavior explicitly retired",
                    ("decision:retire",),
                ),
                ContinuityDecision(
                    "new-probe",
                    "PRESERVE",
                    "PASS",
                    "replacement behavior verified",
                    ("evidence:new",),
                ),
            ),
            root_review=roots_pass(),
            reason="explicit continuity transition",
            evidence_refs=("run:1",),
        )
        self.assertEqual(transition.to_dict()["continuity"][0]["disposition"], "SUPERSEDE")
        self.assertEqual(transition.to_dict()["continuity"][1]["disposition"], "FORGET")

    def test_restart_roundtrip_preserves_journal_and_active_state(self):
        g0 = genesis()
        journal = LineageJournal(g0)
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            parent_state_id=journal.active_state_id,
            parent_snapshot=journal.active_snapshot,
            next_snapshot=snap(step=1, value=2),
            continuity=(ContinuityDecision("anchor", "PRESERVE", "PASS", "behavior retained"),),
            root_review=roots_pass(),
            reason="verified first transition",
        )
        journal.append_transition(transition)
        serialized = json.loads(json.dumps(journal.to_snapshot()))
        restored = LineageJournal.from_snapshot(serialized)
        self.assertEqual(restored.to_snapshot(), journal.to_snapshot())
        self.assertEqual(restored.active_state_id, transition.record_id)

    def test_inner_event_integrity_is_checked_after_outer_rehash(self):
        g0 = genesis()
        journal = LineageJournal(g0)
        serialized = journal.to_snapshot()
        serialized["body"]["events"][0]["commit_evidence"] = "changed"
        outer_rehashed = make_snapshot("axm.lineage-journal/v1", serialized["body"])
        with self.assertRaises(IntegrityError):
            LineageJournal.from_snapshot(outer_rehashed)

    def test_rollback_changes_pointer_without_deleting_history(self):
        g0 = genesis()
        journal = LineageJournal(g0)
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            parent_state_id=journal.active_state_id,
            parent_snapshot=journal.active_snapshot,
            next_snapshot=snap(step=1, value=2),
            continuity=(ContinuityDecision("anchor", "PRESERVE", "PASS", "retained"),),
            root_review=roots_pass(),
            reason="first state",
        )
        journal.append_transition(transition)
        receipt = RollbackReceipt.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            from_state_id=journal.active_state_id,
            target_state_id=g0.record_id,
            target_snapshot=g0.snapshot,
            reason="later state failed recovery validation",
            restore_verification_sha256=g0.snapshot["sha256"],
            evidence_refs=("recovery:test",),
        )
        journal.append_rollback(receipt)
        self.assertEqual(journal.active_state_id, g0.record_id)
        self.assertEqual(len(journal.events), 3)
        self.assertEqual(journal.events[1]["record_id"], transition.record_id)
        self.assertEqual(journal.events[2]["record_id"], receipt.record_id)

    def test_wrong_restore_receipt_is_rejected(self):
        g0 = genesis()
        with self.assertRaises(LineageError):
            RollbackReceipt.create(
                lineage=g0.lineage,
                previous_event_id=g0.record_id,
                from_state_id=g0.record_id,
                target_state_id=g0.record_id,
                target_snapshot=g0.snapshot,
                reason="bad verification",
                restore_verification_sha256="0" * 64,
            )

    def test_branch_genesis_records_parent_lineage_provenance(self):
        parent = genesis()
        child = GenesisRecord.admit(
            lineage="child-lineage",
            snapshot=snap(),
            birth_validation=birth_validation(),
            provenance={"source": "fixture"},
            admitting_authority="fixture",
            commit_evidence="fixture:child",
            parent_lineage=parent.lineage,
            parent_state_id=parent.record_id,
            fork_reason="bounded experiment",
        )
        self.assertEqual(child.to_dict()["branch_from"]["lineage"], parent.lineage)
        self.assertEqual(child.to_dict()["branch_from"]["state_id"], parent.record_id)


if __name__ == "__main__":
    unittest.main()
