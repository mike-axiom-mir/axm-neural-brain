from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from neural.axm_brain import AXMBrain, AXM_ROOT_CONTRACT
from neural.axm_brain.state import verify_snapshot as verify_brain_snapshot

from .core import (
    AXM_ROOT_CONTRACT_SHA256,
    BirthValidation,
    GenesisRecord,
    IntegrityError,
    make_snapshot,
    verify_snapshot,
)

NEURAL_WRAPPER_SCHEMA = "axm-direct-brain-persistence/v1"


def _all_zero(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_zero(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_zero(item) for item in value)
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    return False


def verify_native_brain_snapshot(snapshot: Mapping[str, Any]) -> dict:
    native = deepcopy(dict(snapshot))
    body = verify_brain_snapshot(native)
    if body.get("schema") != AXMBrain.SCHEMA:
        raise IntegrityError("unsupported native brain schema")
    roots = body.get("roots")
    if not isinstance(roots, Mapping):
        raise IntegrityError("native brain snapshot is missing AXM roots")
    if roots.get("contract") != AXM_ROOT_CONTRACT.to_dict():
        raise IntegrityError("native brain AXM root contract mismatch")
    if roots.get("sha256") != AXM_ROOT_CONTRACT.fingerprint:
        raise IntegrityError("native brain AXM root fingerprint mismatch")
    if AXM_ROOT_CONTRACT.fingerprint != AXM_ROOT_CONTRACT_SHA256:
        raise IntegrityError("persistence/core AXM root fingerprint drift")
    return native


def wrap_brain_snapshot(snapshot: Mapping[str, Any]) -> dict:
    native = verify_native_brain_snapshot(snapshot)
    return make_snapshot(
        NEURAL_WRAPPER_SCHEMA,
        {
            "schema": NEURAL_WRAPPER_SCHEMA,
            "brain_schema": AXMBrain.SCHEMA,
            "native_snapshot": native,
            "native_snapshot_sha256": native["sha256"],
            "root_contract_sha256": AXM_ROOT_CONTRACT.fingerprint,
        },
    )


def capture_brain(brain: AXMBrain) -> dict:
    if not isinstance(brain, AXMBrain):
        raise TypeError("capture_brain requires AXMBrain")
    return wrap_brain_snapshot(brain.to_snapshot())


def restore_brain(snapshot: Mapping[str, Any]) -> AXMBrain:
    wrapped = verify_snapshot(snapshot, expected_state_schema=NEURAL_WRAPPER_SCHEMA)
    body = wrapped["body"]
    if body.get("schema") != NEURAL_WRAPPER_SCHEMA:
        raise IntegrityError("neural wrapper body schema mismatch")
    if body.get("brain_schema") != AXMBrain.SCHEMA:
        raise IntegrityError("neural wrapper brain schema mismatch")
    if body.get("root_contract_sha256") != AXM_ROOT_CONTRACT.fingerprint:
        raise IntegrityError("neural wrapper root fingerprint mismatch")
    native = verify_native_brain_snapshot(body["native_snapshot"])
    if body.get("native_snapshot_sha256") != native["sha256"]:
        raise IntegrityError("neural wrapper native snapshot fingerprint mismatch")
    return AXMBrain.from_snapshot(native)


def verify_brain_restore_equivalence(snapshot: Mapping[str, Any]) -> dict:
    before = verify_snapshot(snapshot, expected_state_schema=NEURAL_WRAPPER_SCHEMA)
    after = capture_brain(restore_brain(before))
    if before != after:
        raise IntegrityError("AXM brain restore is not snapshot-equivalent")
    return after


def birth_validation_from_snapshot(snapshot: Mapping[str, Any]) -> BirthValidation:
    native = verify_native_brain_snapshot(snapshot)
    body = native["body"]
    state = body.get("state")
    if not isinstance(state, Mapping):
        raise IntegrityError("native brain state must be a mapping")
    counters_zero = all(
        int(state.get(name, 0)) == 0
        for name in (
            "steps",
            "sleep_count",
            "host_experience_count",
            "supervised_update_count",
            "reward_update_count",
            "sleep_replay_event_count",
            "total_pruned_weights",
            "untagged_experience_count",
        )
    )
    directions_zero = _all_zero(state.get("direction_experience_counts", {}))
    replay_empty = state.get("replay") == []
    phase_is_birth = state.get("mode") == "wake" and int(state.get("cycle", 0)) == 0
    transient_zero = all(
        _all_zero(state.get(name))
        for name in ("hidden", "last_output", "trace_in", "trace_rec", "trace_out")
    )
    return BirthValidation(
        (
            ("brain_snapshot_integrity", True, "native SHA-256 verified"),
            (
                "four_roots_exact",
                body.get("roots", {}).get("sha256") == AXM_ROOT_CONTRACT.fingerprint,
                "native snapshot carries the canonical AXM root contract",
            ),
            (
                "pre_experience_counters_zero",
                counters_zero and directions_zero,
                "no lived, learning, sleep, or direction counters are non-zero",
            ),
            ("pre_experience_replay_empty", replay_empty, "replay history is empty"),
            ("birth_phase", phase_is_birth, "brain is in initial wake cycle zero"),
            (
                "transient_state_zero",
                transient_zero,
                "hidden, output, and eligibility state is zero",
            ),
            (
                "configuration_bound",
                isinstance(body.get("config"), Mapping),
                "brain configuration is inside the hashed native snapshot",
            ),
        )
    )


def admit_brain_genesis(
    brain: AXMBrain,
    *,
    lineage: str,
    provenance: Mapping[str, Any],
    admitting_authority: str,
    commit_evidence: str,
    identities_roles: Mapping[str, Any] | None = None,
    platform_assumptions: Mapping[str, Any] | None = None,
    interface_contract_sha256: str | None = None,
    parent_lineage: str | None = None,
    parent_state_id: str | None = None,
    fork_reason: str | None = None,
) -> GenesisRecord:
    native = brain.to_snapshot()
    wrapped = wrap_brain_snapshot(native)
    return GenesisRecord.admit(
        lineage=lineage,
        snapshot=wrapped,
        birth_validation=birth_validation_from_snapshot(native),
        provenance=provenance,
        admitting_authority=admitting_authority,
        commit_evidence=commit_evidence,
        identities_roles=identities_roles,
        platform_assumptions=platform_assumptions,
        interface_contract_sha256=interface_contract_sha256,
        parent_lineage=parent_lineage,
        parent_state_id=parent_state_id,
        fork_reason=fork_reason,
    )
