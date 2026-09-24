from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

SNAPSHOT_SCHEMA = "axm.state-snapshot/v1"
GENESIS_SCHEMA = "axm.genesis-admission/v1"
TRANSITION_SCHEMA = "axm.lineage-transition/v1"
ROLLBACK_SCHEMA = "axm.rollback-receipt/v1"
JOURNAL_SCHEMA = "axm.lineage-journal/v1"

AXM_ROOTS = ("TRUTH", "AGENCY", "CONTINUITY", "WISDOM_BEFORE_SPEED")
ROOT_DESCRIPTIONS = {
    "TRUTH": "Keep claims and transition decisions grounded in inspectable evidence; keep uncertainty explicit instead of turning it into certainty.",
    "AGENCY": "Do not silently turn capability into domination or obedience. Keep user and system choices explicit inside granted authority.",
    "CONTINUITY": "Preserve demonstrated capability, provenance, and recoverable lineage unless an explicit supersede or forget decision says otherwise.",
    "WISDOM_BEFORE_SPEED": "Prefer reversible and evidenced change over faster irreversible change when uncertainty is material.",
}
AXM_ROOT_CONTRACT = {
    "schema": "axm-roots/v0.1",
    "roots": [{"name": root, "description": ROOT_DESCRIPTIONS[root]} for root in AXM_ROOTS],
    "semantics": {
        "reward_objective": False,
        "obedience_rule": False,
        "capability_lock": False,
        "transition_review_contract": True,
        "hidden_control": False,
    },
}
CONTINUITY_DISPOSITIONS = ("PRESERVE", "SUPERSEDE", "FORGET")
ROOT_STATUSES = ("PASS", "HOLD", "UNKNOWN")


class IntegrityError(ValueError):
    pass


class LineageError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


AXM_ROOT_CONTRACT_SHA256 = sha256_value(AXM_ROOT_CONTRACT)


def _nonempty(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise LineageError(f"{field} must be non-empty")
    return text


def make_snapshot(state_schema: str, body: Mapping[str, Any]) -> dict:
    payload = {
        "schema": SNAPSHOT_SCHEMA,
        "state_schema": _nonempty(state_schema, "state_schema"),
        "body": deepcopy(dict(body)),
    }
    return {**payload, "sha256": sha256_value(payload)}


def verify_snapshot(snapshot: Mapping[str, Any], *, expected_state_schema: str | None = None) -> dict:
    if not isinstance(snapshot, Mapping):
        raise IntegrityError("snapshot must be a mapping")
    required = {"schema", "state_schema", "body", "sha256"}
    missing = required - set(snapshot)
    if missing:
        raise IntegrityError(f"snapshot missing fields: {sorted(missing)}")
    payload = {
        "schema": snapshot["schema"],
        "state_schema": snapshot["state_schema"],
        "body": snapshot["body"],
    }
    if payload["schema"] != SNAPSHOT_SCHEMA:
        raise IntegrityError(f"unsupported snapshot schema: {payload['schema']!r}")
    if expected_state_schema is not None and payload["state_schema"] != expected_state_schema:
        raise IntegrityError("state schema mismatch")
    if snapshot["sha256"] != sha256_value(payload):
        raise IntegrityError("snapshot integrity check failed")
    return deepcopy(dict(snapshot))


def snapshot_identity(snapshot: Mapping[str, Any]) -> str:
    verified = verify_snapshot(snapshot)
    return f"state:{verified['state_schema']}:{verified['sha256']}"


def verify_restore_equivalence(
    snapshot: Mapping[str, Any],
    *,
    restore: Callable[[dict], Any],
    capture: Callable[[Any], Mapping[str, Any]],
) -> dict:
    before = verify_snapshot(snapshot)
    after = verify_snapshot(capture(restore(deepcopy(before))))
    if before != after:
        raise IntegrityError("restore round-trip is not snapshot-equivalent")
    return after


@dataclass(frozen=True)
class BirthValidation:
    checks: tuple[tuple[str, bool, str], ...]

    def __post_init__(self) -> None:
        if not self.checks:
            raise LineageError("birth validation requires at least one check")
        seen = set()
        normalized = []
        for name, passed, detail in self.checks:
            name = _nonempty(name, "birth validation check name")
            if name in seen:
                raise LineageError(f"duplicate birth validation check: {name}")
            seen.add(name)
            normalized.append((name, bool(passed), _nonempty(detail, "birth validation detail")))
        object.__setattr__(self, "checks", tuple(normalized))

    @property
    def passed(self) -> bool:
        return all(passed for _, passed, _ in self.checks)

    def to_dict(self) -> dict:
        return {
            "schema": "axm.genesis-validation/v1",
            "passed": self.passed,
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in self.checks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BirthValidation":
        return cls(tuple((str(x["name"]), bool(x["passed"]), str(x["detail"])) for x in data["checks"]))


@dataclass(frozen=True)
class ContinuityDecision:
    probe_id: str
    disposition: str
    result: str
    reason: str
    evidence_refs: tuple[str, ...] = ()
    replacement_probe_id: str | None = None

    def __post_init__(self) -> None:
        probe_id = _nonempty(self.probe_id, "probe_id")
        disposition = str(self.disposition).strip().upper()
        result = str(self.result).strip().upper()
        if disposition not in CONTINUITY_DISPOSITIONS:
            raise LineageError(f"unsupported continuity disposition: {disposition}")
        if result not in ("PASS", "REGRESSION", "NOT_TESTED"):
            raise LineageError(f"unsupported continuity result: {result}")
        reason = _nonempty(self.reason, "continuity decision reason")
        replacement = self.replacement_probe_id
        if disposition == "PRESERVE" and result != "PASS":
            raise LineageError("PRESERVE requires a passing continuity result")
        if disposition == "SUPERSEDE":
            replacement = _nonempty(replacement, "replacement_probe_id")
            if replacement == probe_id:
                raise LineageError("SUPERSEDE replacement must name a different probe")
        elif replacement is not None:
            raise LineageError("replacement_probe_id is only valid for SUPERSEDE")
        object.__setattr__(self, "probe_id", probe_id)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "result", result)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "replacement_probe_id", replacement)
        object.__setattr__(self, "evidence_refs", tuple(_nonempty(x, "evidence_ref") for x in self.evidence_refs))

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "disposition": self.disposition,
            "result": self.result,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "replacement_probe_id": self.replacement_probe_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContinuityDecision":
        return cls(
            str(data["probe_id"]),
            str(data["disposition"]),
            str(data["result"]),
            str(data["reason"]),
            tuple(str(x) for x in data.get("evidence_refs", [])),
            data.get("replacement_probe_id"),
        )


def normalize_root_review(review: Mapping[str, Mapping[str, Any]]) -> dict:
    if set(review) != set(AXM_ROOTS):
        raise LineageError("root review must cover exactly the four AXM roots")
    normalized = {}
    for root in AXM_ROOTS:
        item = review[root]
        status = str(item["status"]).strip().upper()
        if status not in ROOT_STATUSES:
            raise LineageError(f"unsupported root status for {root}: {status}")
        normalized[root] = {
            "status": status,
            "reason": _nonempty(item["reason"], f"{root} reason"),
            "evidence_refs": [_nonempty(x, f"{root} evidence ref") for x in item.get("evidence_refs", ())],
        }
    return normalized


def _record_body(record: Mapping[str, Any]) -> dict:
    body = dict(record)
    body.pop("record_sha256", None)
    body.pop("record_id", None)
    return body


@dataclass(frozen=True)
class GenesisRecord:
    record: dict

    @classmethod
    def admit(
        cls,
        *,
        lineage: str,
        snapshot: Mapping[str, Any],
        birth_validation: BirthValidation,
        provenance: Mapping[str, Any],
        admitting_authority: str,
        commit_evidence: str,
        identities_roles: Mapping[str, Any] | None = None,
        platform_assumptions: Mapping[str, Any] | None = None,
        interface_contract_sha256: str | None = None,
        parent_lineage: str | None = None,
        parent_state_id: str | None = None,
        fork_reason: str | None = None,
    ) -> "GenesisRecord":
        lineage = _nonempty(lineage, "lineage")
        snap = verify_snapshot(snapshot)
        if not birth_validation.passed:
            failed = [n for n, passed, _ in birth_validation.checks if not passed]
            raise LineageError("genesis birth validation failed: " + ", ".join(failed))
        if not dict(provenance):
            raise LineageError("genesis provenance must be non-empty")
        if (parent_lineage is None) != (parent_state_id is None):
            raise LineageError("parent_lineage and parent_state_id must be supplied together")
        branch_from = None
        if parent_lineage is not None:
            branch_from = {
                "lineage": _nonempty(parent_lineage, "parent_lineage"),
                "state_id": _nonempty(parent_state_id, "parent_state_id"),
                "reason": _nonempty(fork_reason, "fork_reason"),
            }
        elif fork_reason is not None:
            raise LineageError("fork_reason requires parent lineage provenance")

        body = {
            "schema": GENESIS_SCHEMA,
            "lineage": lineage,
            "previous_event_id": None,
            "snapshot": snap,
            "snapshot_id": snapshot_identity(snap),
            "roots": {"contract": deepcopy(AXM_ROOT_CONTRACT), "sha256": AXM_ROOT_CONTRACT_SHA256},
            "birth_validation": birth_validation.to_dict(),
            "birth_validation_sha256": sha256_value(birth_validation.to_dict()),
            "provenance": deepcopy(dict(provenance)),
            "identities_roles": deepcopy(dict(identities_roles or {})),
            "platform_assumptions": deepcopy(dict(platform_assumptions or {})),
            "interface_contract_sha256": interface_contract_sha256,
            "admitting_authority": _nonempty(admitting_authority, "admitting_authority"),
            "commit_evidence": _nonempty(commit_evidence, "commit_evidence"),
            "branch_from": branch_from,
            "immutability_rule": "G0 is immutable; correction requires a new lineage/version rather than silent replacement",
        }
        digest = sha256_value(body)
        record = {**body, "record_sha256": digest, "record_id": f"g0:{lineage}:{digest}"}
        obj = cls(record)
        obj.verify()
        return obj

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GenesisRecord":
        obj = cls(deepcopy(dict(data)))
        obj.verify()
        return obj

    def verify(self) -> None:
        r = self.record
        if r.get("schema") != GENESIS_SCHEMA or r.get("previous_event_id") is not None:
            raise IntegrityError("invalid genesis schema/event ancestry")
        verify_snapshot(r["snapshot"])
        if r.get("snapshot_id") != snapshot_identity(r["snapshot"]):
            raise IntegrityError("genesis snapshot identity mismatch")
        roots = r.get("roots", {})
        if roots.get("contract") != AXM_ROOT_CONTRACT or roots.get("sha256") != AXM_ROOT_CONTRACT_SHA256:
            raise IntegrityError("genesis AXM root contract mismatch")
        birth = BirthValidation.from_dict(r["birth_validation"])
        if not birth.passed or r.get("birth_validation_sha256") != sha256_value(birth.to_dict()):
            raise IntegrityError("genesis birth validation mismatch")
        expected = sha256_value(_record_body(r))
        if r.get("record_sha256") != expected:
            raise IntegrityError("genesis record integrity check failed")
        lineage = _nonempty(r.get("lineage"), "lineage")
        if r.get("record_id") != f"g0:{lineage}:{expected}":
            raise IntegrityError("genesis identity mismatch")

    @property
    def record_id(self) -> str:
        return self.record["record_id"]

    @property
    def lineage(self) -> str:
        return self.record["lineage"]

    @property
    def snapshot(self) -> dict:
        return deepcopy(self.record["snapshot"])

    def to_dict(self) -> dict:
        return deepcopy(self.record)


@dataclass(frozen=True)
class TransitionRecord:
    record: dict

    @classmethod
    def create(
        cls,
        *,
        lineage: str,
        previous_event_id: str,
        parent_state_id: str,
        parent_snapshot: Mapping[str, Any],
        next_snapshot: Mapping[str, Any],
        continuity: Iterable[ContinuityDecision],
        root_review: Mapping[str, Mapping[str, Any]],
        reason: str,
        evidence_refs: Iterable[str] = (),
    ) -> "TransitionRecord":
        lineage = _nonempty(lineage, "lineage")
        parent = verify_snapshot(parent_snapshot)
        nxt = verify_snapshot(next_snapshot)
        decisions = tuple(continuity)
        if not decisions or len({d.probe_id for d in decisions}) != len(decisions):
            raise LineageError("transition requires unique explicit continuity decisions")
        review = normalize_root_review(root_review)
        held = [root for root in AXM_ROOTS if review[root]["status"] != "PASS"]
        if held:
            raise LineageError("canonical transition cannot be admitted with held roots: " + ", ".join(held))
        body = {
            "schema": TRANSITION_SCHEMA,
            "lineage": lineage,
            "previous_event_id": _nonempty(previous_event_id, "previous_event_id"),
            "parent_state_id": _nonempty(parent_state_id, "parent_state_id"),
            "parent_snapshot_sha256": parent["sha256"],
            "next_snapshot": nxt,
            "next_snapshot_id": snapshot_identity(nxt),
            "continuity": [d.to_dict() for d in decisions],
            "root_contract_sha256": AXM_ROOT_CONTRACT_SHA256,
            "root_review": review,
            "reason": _nonempty(reason, "transition reason"),
            "evidence_refs": [_nonempty(x, "transition evidence ref") for x in evidence_refs],
        }
        digest = sha256_value(body)
        record = {**body, "record_sha256": digest, "record_id": f"s:{lineage}:{digest}"}
        obj = cls(record)
        obj.verify()
        return obj

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransitionRecord":
        obj = cls(deepcopy(dict(data)))
        obj.verify()
        return obj

    def verify(self) -> None:
        r = self.record
        if r.get("schema") != TRANSITION_SCHEMA:
            raise IntegrityError("unsupported transition schema")
        verify_snapshot(r["next_snapshot"])
        if r.get("next_snapshot_id") != snapshot_identity(r["next_snapshot"]):
            raise IntegrityError("transition next snapshot identity mismatch")
        decisions = [ContinuityDecision.from_dict(x) for x in r.get("continuity", [])]
        if not decisions:
            raise IntegrityError("transition has no continuity decisions")
        review = normalize_root_review(r["root_review"])
        if any(review[root]["status"] != "PASS" for root in AXM_ROOTS):
            raise IntegrityError("canonical transition contains a held root")
        if r.get("root_contract_sha256") != AXM_ROOT_CONTRACT_SHA256:
            raise IntegrityError("transition root contract mismatch")
        expected = sha256_value(_record_body(r))
        if r.get("record_sha256") != expected:
            raise IntegrityError("transition record integrity check failed")
        lineage = _nonempty(r.get("lineage"), "lineage")
        if r.get("record_id") != f"s:{lineage}:{expected}":
            raise IntegrityError("transition identity mismatch")

    @property
    def record_id(self) -> str:
        return self.record["record_id"]

    @property
    def snapshot(self) -> dict:
        return deepcopy(self.record["next_snapshot"])

    def to_dict(self) -> dict:
        return deepcopy(self.record)


@dataclass(frozen=True)
class RollbackReceipt:
    record: dict

    @classmethod
    def create(
        cls,
        *,
        lineage: str,
        previous_event_id: str,
        from_state_id: str,
        target_state_id: str,
        target_snapshot: Mapping[str, Any],
        reason: str,
        restore_verification_sha256: str,
        evidence_refs: Iterable[str] = (),
    ) -> "RollbackReceipt":
        lineage = _nonempty(lineage, "lineage")
        snap = verify_snapshot(target_snapshot)
        restore_hash = _nonempty(restore_verification_sha256, "restore_verification_sha256")
        if restore_hash != snap["sha256"]:
            raise LineageError("rollback restore verification does not match target snapshot")
        body = {
            "schema": ROLLBACK_SCHEMA,
            "lineage": lineage,
            "previous_event_id": _nonempty(previous_event_id, "previous_event_id"),
            "from_state_id": _nonempty(from_state_id, "from_state_id"),
            "target_state_id": _nonempty(target_state_id, "target_state_id"),
            "target_snapshot_sha256": snap["sha256"],
            "reason": _nonempty(reason, "rollback reason"),
            "restore_verification_sha256": restore_hash,
            "evidence_refs": [_nonempty(x, "rollback evidence ref") for x in evidence_refs],
            "history_rule": "rollback changes active state pointer; admitted history remains append-only",
        }
        digest = sha256_value(body)
        record = {**body, "record_sha256": digest, "record_id": f"rb:{lineage}:{digest}"}
        obj = cls(record)
        obj.verify()
        return obj

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RollbackReceipt":
        obj = cls(deepcopy(dict(data)))
        obj.verify()
        return obj

    def verify(self) -> None:
        r = self.record
        if r.get("schema") != ROLLBACK_SCHEMA:
            raise IntegrityError("unsupported rollback schema")
        expected = sha256_value(_record_body(r))
        if r.get("record_sha256") != expected:
            raise IntegrityError("rollback record integrity check failed")
        lineage = _nonempty(r.get("lineage"), "lineage")
        if r.get("record_id") != f"rb:{lineage}:{expected}":
            raise IntegrityError("rollback identity mismatch")

    @property
    def record_id(self) -> str:
        return self.record["record_id"]

    def to_dict(self) -> dict:
        return deepcopy(self.record)


class LineageJournal:
    def __init__(self, genesis: GenesisRecord) -> None:
        genesis.verify()
        self._lineage = genesis.lineage
        self._events = [genesis.to_dict()]
        self._states = {genesis.record_id: genesis.snapshot}
        self._active_state_id = genesis.record_id

    @property
    def lineage(self) -> str:
        return self._lineage

    @property
    def events(self) -> tuple[dict, ...]:
        return tuple(deepcopy(self._events))

    @property
    def active_state_id(self) -> str:
        return self._active_state_id

    @property
    def active_snapshot(self) -> dict:
        return deepcopy(self._states[self._active_state_id])

    @property
    def last_event_id(self) -> str:
        return self._events[-1]["record_id"]

    def append_transition(self, transition: TransitionRecord) -> None:
        transition.verify()
        r = transition.record
        if r["lineage"] != self._lineage:
            raise LineageError("transition lineage mismatch")
        if r["previous_event_id"] != self.last_event_id:
            raise LineageError("transition does not extend current event chain")
        if r["parent_state_id"] != self._active_state_id:
            raise LineageError("transition parent is not current active state")
        if r["parent_snapshot_sha256"] != self.active_snapshot["sha256"]:
            raise LineageError("transition parent snapshot hash mismatch")
        self._events.append(transition.to_dict())
        self._states[transition.record_id] = transition.snapshot
        self._active_state_id = transition.record_id

    def append_rollback(self, receipt: RollbackReceipt) -> None:
        receipt.verify()
        r = receipt.record
        if r["lineage"] != self._lineage:
            raise LineageError("rollback lineage mismatch")
        if r["previous_event_id"] != self.last_event_id:
            raise LineageError("rollback does not extend current event chain")
        if r["from_state_id"] != self._active_state_id:
            raise LineageError("rollback source is not current active state")
        if r["target_state_id"] not in self._states:
            raise LineageError("rollback target is not an admitted state")
        target = self._states[r["target_state_id"]]
        if target["sha256"] != r["target_snapshot_sha256"] or r["restore_verification_sha256"] != target["sha256"]:
            raise LineageError("rollback target/restore verification mismatch")
        self._events.append(receipt.to_dict())
        self._active_state_id = r["target_state_id"]

    def to_snapshot(self) -> dict:
        return make_snapshot(
            JOURNAL_SCHEMA,
            {
                "schema": JOURNAL_SCHEMA,
                "lineage": self._lineage,
                "events": deepcopy(self._events),
                "active_state_id": self._active_state_id,
            },
        )

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "LineageJournal":
        wrapped = verify_snapshot(snapshot, expected_state_schema=JOURNAL_SCHEMA)
        body = wrapped["body"]
        if body.get("schema") != JOURNAL_SCHEMA:
            raise IntegrityError("journal body schema mismatch")
        events = body.get("events")
        if not isinstance(events, list) or not events:
            raise IntegrityError("journal requires a genesis event")
        journal = cls(GenesisRecord.from_dict(events[0]))
        for raw in events[1:]:
            schema = raw.get("schema")
            if schema == TRANSITION_SCHEMA:
                journal.append_transition(TransitionRecord.from_dict(raw))
            elif schema == ROLLBACK_SCHEMA:
                journal.append_rollback(RollbackReceipt.from_dict(raw))
            else:
                raise IntegrityError(f"unsupported journal event schema: {schema!r}")
        if journal.active_state_id != body.get("active_state_id"):
            raise IntegrityError("journal active state pointer mismatch")
        if journal.lineage != body.get("lineage"):
            raise IntegrityError("journal lineage mismatch")
        return journal
