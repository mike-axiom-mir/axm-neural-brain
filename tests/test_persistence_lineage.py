import copy
import json
import unittest

from axm_persistence.core import (
    AXM_ROOTS,
    BirthValidation,
    ContinuityDecision,
    CheckpointReceipt,
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

    def test_checkpoint_receipt_is_append_only_and_keeps_active_state(self):
        g0 = genesis()
        journal = LineageJournal(g0)
        receipt = CheckpointReceipt.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            state_id=journal.active_state_id,
            state_snapshot=journal.active_snapshot,
            durable_ref="host-checkpoint:fixture-001",
            restore_verification_sha256=journal.active_snapshot["sha256"],
            evidence_refs=("restore:fixture-001",),
        )
        before_state = journal.active_state_id
        journal.append_checkpoint(receipt)
        self.assertEqual(journal.active_state_id, before_state)
        self.assertEqual(journal.last_event_id, receipt.record_id)
        restored = LineageJournal.from_snapshot(
            json.loads(json.dumps(journal.to_snapshot()))
        )
        self.assertEqual(restored.to_snapshot(), journal.to_snapshot())
        self.assertEqual(restored.active_state_id, before_state)

    def test_checkpoint_receipt_rejects_false_restore_hash(self):
        g0 = genesis()
        with self.assertRaises(LineageError):
            CheckpointReceipt.create(
                lineage=g0.lineage,
                previous_event_id=g0.record_id,
                state_id=g0.record_id,
                state_snapshot=g0.snapshot,
                durable_ref="host-checkpoint:bad",
                restore_verification_sha256="0" * 64,
            )

    def test_checkpoint_for_non_active_state_is_rejected(self):
        g0 = genesis()
        journal = LineageJournal(g0)
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            parent_state_id=journal.active_state_id,
            parent_snapshot=journal.active_snapshot,
            next_snapshot=snap(step=1, value=2),
            continuity=(
                ContinuityDecision("anchor", "PRESERVE", "PASS", "retained"),
            ),
            root_review=roots_pass(),
            reason="advance active state",
        )
        journal.append_transition(transition)
        stale = CheckpointReceipt.create(
            lineage=g0.lineage,
            previous_event_id=journal.last_event_id,
            state_id=g0.record_id,
            state_snapshot=g0.snapshot,
            durable_ref="host-checkpoint:stale",
            restore_verification_sha256=g0.snapshot["sha256"],
        )
        with self.assertRaises(LineageError):
            journal.append_checkpoint(stale)

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


from axm_persistence.core import AXM_ROOT_CONTRACT_SHA256, sha256_value
from axm_persistence.neural_adapter import (
    admit_brain_genesis,
    birth_validation_from_snapshot,
    capture_brain,
    restore_brain,
    verify_brain_restore_equivalence,
)
from neural.axm_brain import AXMBrain, AXM_ROOT_CONTRACT, BrainConfig, Experience


def _rehash_record(record):
    body = copy.deepcopy(record)
    body.pop("record_sha256", None)
    body.pop("record_id", None)
    digest = sha256_value(body)
    prefix = {
        "axm.genesis-admission/v1": "g0",
        "axm.lineage-transition/v1": "s",
        "axm.rollback-receipt/v1": "rb",
        "axm.checkpoint-receipt/v1": "cp",
    }[body["schema"]]
    return {
        **body,
        "record_sha256": digest,
        "record_id": f"{prefix}:{body['lineage']}:{digest}",
    }


class NeuralCorePersistenceIntegrationTests(unittest.TestCase):
    def brain(self, seed=71):
        return AXMBrain(
            BrainConfig(
                input_size=2,
                hidden_size=4,
                output_size=1,
                seed=seed,
                replay_capacity=8,
            )
        )

    def brain_genesis(self):
        return admit_brain_genesis(
            self.brain(),
            lineage="adapter-fixture",
            provenance={
                "source_repo": "mike-axiom-mir/axm-neural-brain",
                "core_head": "a0f5de4b19bf515e145caf50b12130ab740869b5",
                "donor_repo": "mike-axiom-mir/axm-uc-neural",
                "donor_pr": 3,
                "donor_head": "163410ce05a42879dca478d8ea0de7f6d17c4820",
            },
            admitting_authority="fixture-admission",
            commit_evidence="fixture:durable-commit",
            identities_roles={"system": "AXM Neural Brain"},
            platform_assumptions={"runtime": "python-3.11"},
        )

    def test_core_and_persistence_share_exact_root_fingerprint(self):
        self.assertEqual(
            AXM_ROOT_CONTRACT_SHA256,
            AXM_ROOT_CONTRACT.fingerprint,
        )

    def test_real_neural_birth_can_be_admitted_and_restored(self):
        record = self.brain_genesis()
        record.verify()
        self.assertTrue(record.record_id.startswith("g0:adapter-fixture:"))
        restored = restore_brain(record.snapshot)
        self.assertEqual(restored.to_snapshot(), self.brain().to_snapshot())

    def test_experienced_brain_is_not_g0(self):
        brain = self.brain()
        brain.experience(
            Experience([0.2, -0.3], target=[0.4], directions=("LEARN",))
        )
        self.assertFalse(
            birth_validation_from_snapshot(brain.to_snapshot()).passed
        )
        with self.assertRaises(LineageError):
            admit_brain_genesis(
                brain,
                lineage="experienced",
                provenance={"source": "fixture"},
                admitting_authority="fixture",
                commit_evidence="fixture:commit",
            )

    def test_slept_brain_is_not_g0(self):
        brain = self.brain()
        brain.sleep()
        self.assertFalse(
            birth_validation_from_snapshot(brain.to_snapshot()).passed
        )

    def test_learned_state_survives_json_restart_exactly(self):
        brain = self.brain()
        for _ in range(4):
            brain.experience(
                Experience(
                    [0.8, -0.4],
                    target=[0.65],
                    reward=0.5,
                    directions=("USE", "LEARN"),
                )
            )
        brain.sleep()
        wrapped = capture_brain(brain)
        serialized = json.loads(json.dumps(wrapped))
        self.assertEqual(
            verify_brain_restore_equivalence(serialized),
            wrapped,
        )
        self.assertEqual(
            restore_brain(serialized).to_snapshot(),
            brain.to_snapshot(),
        )

    def test_nested_native_tamper_is_rejected_after_outer_rehash(self):
        wrapped = capture_brain(self.brain())
        body = copy.deepcopy(wrapped["body"])
        body["native_snapshot"]["body"]["state"]["steps"] = 999
        forged_outer = make_snapshot(wrapped["state_schema"], body)
        with self.assertRaises(Exception):
            restore_brain(forged_outer)

    def test_real_brain_snapshots_form_transition_record(self):
        brain = self.brain()
        g0 = self.brain_genesis()
        parent = g0.snapshot
        brain.experience(
            Experience([0.3, 0.1], target=[0.2], directions=("LEARN",))
        )
        candidate = capture_brain(brain)
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=g0.record_id,
            parent_state_id=g0.record_id,
            parent_snapshot=parent,
            next_snapshot=candidate,
            continuity=(
                ContinuityDecision(
                    "fixture-behavior",
                    "PRESERVE",
                    "PASS",
                    "behavior probe passed in fixture",
                    ("fixture:behavior",),
                ),
            ),
            root_review=roots_pass(),
            reason="adapter integration fixture",
            evidence_refs=("fixture:transition",),
        )
        transition.verify()
        self.assertEqual(
            transition.to_dict()["parent_snapshot_sha256"],
            parent["sha256"],
        )


class SemanticRehashTamperTests(unittest.TestCase):
    def test_rehashed_genesis_cannot_remove_commit_evidence(self):
        forged = genesis().to_dict()
        forged["commit_evidence"] = ""
        forged = _rehash_record(forged)
        with self.assertRaises(IntegrityError):
            GenesisRecord.from_dict(forged)

    def test_rehashed_genesis_cannot_change_immutability_rule(self):
        forged = genesis().to_dict()
        forged["immutability_rule"] = "rewritable"
        forged = _rehash_record(forged)
        with self.assertRaises(IntegrityError):
            GenesisRecord.from_dict(forged)

    def test_rehashed_genesis_cannot_forge_birth_validation_schema(self):
        forged = genesis().to_dict()
        forged["birth_validation"]["schema"] = "fake/v9"
        forged["birth_validation_sha256"] = sha256_value(
            forged["birth_validation"]
        )
        forged = _rehash_record(forged)
        with self.assertRaises(IntegrityError):
            GenesisRecord.from_dict(forged)

    def test_rehashed_transition_cannot_duplicate_probe_decisions(self):
        g0 = genesis()
        transition = TransitionRecord.create(
            lineage=g0.lineage,
            previous_event_id=g0.record_id,
            parent_state_id=g0.record_id,
            parent_snapshot=g0.snapshot,
            next_snapshot=snap(step=1, value=2),
            continuity=(
                ContinuityDecision(
                    "a",
                    "PRESERVE",
                    "PASS",
                    "ok",
                ),
            ),
            root_review=roots_pass(),
            reason="fixture",
        ).to_dict()
        transition["continuity"].append(
            copy.deepcopy(transition["continuity"][0])
        )
        transition = _rehash_record(transition)
        with self.assertRaises(IntegrityError):
            TransitionRecord.from_dict(transition)

    def test_rehashed_checkpoint_cannot_change_host_authority_rule(self):
        g0 = genesis()
        receipt = CheckpointReceipt.create(
            lineage=g0.lineage,
            previous_event_id=g0.record_id,
            state_id=g0.record_id,
            state_snapshot=g0.snapshot,
            durable_ref="host-checkpoint:fixture",
            restore_verification_sha256=g0.snapshot["sha256"],
        ).to_dict()
        receipt["authority_rule"] = "lineage owns storage"
        receipt = _rehash_record(receipt)
        with self.assertRaises(IntegrityError):
            CheckpointReceipt.from_dict(receipt)

    def test_rehashed_rollback_cannot_change_append_only_rule(self):
        g0 = genesis()
        receipt = RollbackReceipt.create(
            lineage=g0.lineage,
            previous_event_id=g0.record_id,
            from_state_id=g0.record_id,
            target_state_id=g0.record_id,
            target_snapshot=g0.snapshot,
            reason="fixture",
            restore_verification_sha256=g0.snapshot["sha256"],
        ).to_dict()
        receipt["history_rule"] = "rewritable"
        receipt = _rehash_record(receipt)
        with self.assertRaises(IntegrityError):
            RollbackReceipt.from_dict(receipt)


if __name__ == "__main__":
    unittest.main()
