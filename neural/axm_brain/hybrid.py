from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from typing import Any, Mapping, Sequence

from .contract import BoundBrain


HYBRID_ORGAN_SCHEMA = "axm.hybrid-organ/v0.1"
HYBRID_ORGAN_PACK_SCHEMA = "axm.hybrid-organ-pack/v0.1"
HYBRID_DECISION_SCHEMA = "axm.hybrid-decision/v0.1"
HYBRID_EXPERIMENT_SCHEMA = "axm.hybrid-experiment/v0.1"
HYBRID_CANDIDATE_SCHEMA = "axm.hybrid-organ-candidate/v0.1"
HYBRID_WORKSPACE_SCHEMA = "axm.hybrid-workspace/v0.1"

_ALLOWED_OPERATORS = {"eq", "neq", "gt", "gte", "lt", "lte"}


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("hybrid state must be strict canonical JSON") from exc


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be non-empty trimmed text")
    return value


def _finite_score(value: Any, field: str) -> float:
    score = float(value)
    if not isfinite(score):
        raise ValueError(f"{field} must be finite")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return score


@dataclass(frozen=True)
class Predicate:
    field: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        _text(self.field, "predicate field")
        if self.operator not in _ALLOWED_OPERATORS:
            raise ValueError(f"unsupported predicate operator: {self.operator}")
        canonical_bytes(self.value)

    def evaluate(self, state: Mapping[str, Any]) -> bool:
        if self.field not in state:
            return False
        actual = state[self.field]
        expected = self.value
        if self.operator == "eq":
            return actual == expected
        if self.operator == "neq":
            return actual != expected
        try:
            if self.operator == "gt":
                return actual > expected
            if self.operator == "gte":
                return actual >= expected
            if self.operator == "lt":
                return actual < expected
            if self.operator == "lte":
                return actual <= expected
        except TypeError:
            return False
        raise AssertionError("unreachable predicate operator")

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "operator": self.operator, "value": self.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Predicate":
        return cls(
            field=value["field"],
            operator=value["operator"],
            value=value.get("value"),
        )


@dataclass(frozen=True)
class Proposal:
    source: str
    option: str
    score: float
    source_ref: str
    rationale: str

    def __post_init__(self) -> None:
        if self.source not in {"organ", "neural"}:
            raise ValueError("proposal source must be organ or neural")
        _text(self.option, "proposal option")
        _finite_score(self.score, "proposal score")
        _text(self.source_ref, "proposal source_ref")
        _text(self.rationale, "proposal rationale")

    @property
    def proposal_id(self) -> str:
        return sha256_value(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "option": self.option,
            "score": self.score,
            "source_ref": self.source_ref,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class HybridOrgan:
    organ_id: str
    version: str
    purpose: str
    conditions: tuple[Predicate, ...]
    proposal_option: str
    proposal_score: float
    provenance: Mapping[str, Any]
    limitations: tuple[str, ...] = ()
    status: str = "active"

    def __post_init__(self) -> None:
        _text(self.organ_id, "organ_id")
        _text(self.version, "organ version")
        _text(self.purpose, "organ purpose")
        _text(self.proposal_option, "proposal option")
        _finite_score(self.proposal_score, "proposal_score")
        if not self.conditions:
            raise ValueError("hybrid organ requires at least one explicit condition")
        if self.status != "active":
            raise ValueError("loaded hybrid organ status must be active")
        if not isinstance(self.provenance, Mapping) or not dict(self.provenance):
            raise ValueError("hybrid organ provenance must be a non-empty mapping")
        canonical_bytes(dict(self.provenance))
        for item in self.limitations:
            _text(item, "organ limitation")

    def _body(self) -> dict[str, Any]:
        return {
            "schema": HYBRID_ORGAN_SCHEMA,
            "id": self.organ_id,
            "version": self.version,
            "status": self.status,
            "purpose": self.purpose,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "proposal": {
                "option": self.proposal_option,
                "score": self.proposal_score,
            },
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
        }

    @property
    def fingerprint(self) -> str:
        return sha256_value(self._body())

    def to_record(self) -> dict[str, Any]:
        return {**self._body(), "sha256": self.fingerprint}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "HybridOrgan":
        if value.get("schema") != HYBRID_ORGAN_SCHEMA:
            raise ValueError("unsupported hybrid organ schema")
        proposal = value.get("proposal")
        if not isinstance(proposal, Mapping):
            raise ValueError("hybrid organ proposal must be an object")
        organ = cls(
            organ_id=value["id"],
            version=value["version"],
            status=value.get("status", "active"),
            purpose=value["purpose"],
            conditions=tuple(
                Predicate.from_dict(item) for item in value.get("conditions", ())
            ),
            proposal_option=proposal["option"],
            proposal_score=proposal["score"],
            provenance=value["provenance"],
            limitations=tuple(value.get("limitations", ())),
        )
        if value.get("sha256") != organ.fingerprint:
            raise ValueError("hybrid organ fingerprint mismatch")
        return organ

    def matches(self, state: Mapping[str, Any]) -> bool:
        canonical_bytes(dict(state))
        return all(condition.evaluate(state) for condition in self.conditions)

    def propose(self, state: Mapping[str, Any]) -> Proposal | None:
        if not self.matches(state):
            return None
        return Proposal(
            source="organ",
            option=self.proposal_option,
            score=self.proposal_score,
            source_ref=f"{self.organ_id}@{self.version}:{self.fingerprint}",
            rationale=f"explicit organ conditions matched for {self.organ_id}",
        )


def make_organ_pack(organs: Sequence[HybridOrgan]) -> dict[str, Any]:
    body = {
        "schema": HYBRID_ORGAN_PACK_SCHEMA,
        "organs": [organ.to_record() for organ in organs],
    }
    return {**body, "sha256": sha256_value(body)}


def load_organ_pack(value: Mapping[str, Any]) -> tuple[HybridOrgan, ...]:
    if value.get("schema") != HYBRID_ORGAN_PACK_SCHEMA:
        raise ValueError("unsupported hybrid organ pack schema")
    body = {
        "schema": value.get("schema"),
        "organs": value.get("organs"),
    }
    if value.get("sha256") != sha256_value(body):
        raise ValueError("hybrid organ pack fingerprint mismatch")
    organs = tuple(HybridOrgan.from_record(item) for item in value.get("organs", ()))
    if len({(organ.organ_id, organ.version) for organ in organs}) != len(organs):
        raise ValueError("duplicate hybrid organ identity")
    return organs


class HybridBrain:
    """Peer workspace for explicit organs and learned neural proposals.

    Neither side is granted execution authority. The host decides whether to act,
    teach, test, persist, or admit any candidate knowledge.
    """

    def __init__(self, bound_brain: BoundBrain, organs: Sequence[HybridOrgan] = ()):
        self.bound_brain = bound_brain
        self.organs = tuple(organs)
        if len({(organ.organ_id, organ.version) for organ in self.organs}) != len(self.organs):
            raise ValueError("duplicate hybrid organ identity")
        output_names = tuple(self.bound_brain.contract.outputs)
        if not output_names:
            raise ValueError("hybrid brain requires named neural outputs")

    def _organ_proposal(self, state: Mapping[str, Any]) -> Proposal | None:
        proposals = [
            proposal
            for organ in self.organs
            for proposal in (organ.propose(state),)
            if proposal is not None
        ]
        if not proposals:
            return None
        return sorted(
            proposals,
            key=lambda proposal: (-proposal.score, proposal.source_ref),
        )[0]

    def _neural_proposal(self, state: Mapping[str, float]) -> Proposal:
        observation = self.bound_brain.contract.encode(state)
        raw = self.bound_brain.brain.predict(observation, update_state=False)
        decoded = self.bound_brain.output_state(raw)
        option, raw_score = sorted(
            decoded.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        normalized = (float(raw_score) + 1.0) / 2.0
        return Proposal(
            source="neural",
            option=option,
            score=max(0.0, min(1.0, normalized)),
            source_ref=(
                f"{self.bound_brain.contract.name}:"
                f"{self.bound_brain.contract.fingerprint}"
            ),
            rationale=f"highest learned neural output was {option}",
        )

    def deliberate(self, state: Mapping[str, float]) -> dict[str, Any]:
        organ = self._organ_proposal(state)
        neural = self._neural_proposal(state)
        if organ is None:
            classification = "NEURAL_ONLY"
        elif organ.option == neural.option:
            classification = "AGREE"
        else:
            classification = "TEST_REQUIRED"
        body = {
            "schema": HYBRID_DECISION_SCHEMA,
            "classification": classification,
            "organ_proposal": None if organ is None else organ.to_dict(),
            "neural_proposal": neural.to_dict(),
            "execution_authorized": False,
            "automatic_adoption": False,
            "host_decision_required": True,
            "truth_boundary": (
                "proposal comparison is not execution authority or proof of usefulness"
            ),
        }
        return {**body, "sha256": sha256_value(body)}

    def learn_from_host_outcome(
        self,
        state: Mapping[str, float],
        *,
        chosen_option: str,
        reward: float | None = None,
    ) -> list[float]:
        _text(chosen_option, "chosen_option")
        outputs = tuple(self.bound_brain.contract.outputs)
        if chosen_option not in outputs:
            raise ValueError("chosen option is not a named neural output")
        target = [1.0 if name == chosen_option else -1.0 for name in outputs]
        return self.bound_brain.experience(
            state,
            target=target,
            reward=reward,
            source="hybrid-host-outcome",
            tag=f"chosen:{chosen_option}",
            directions=("LEARN", "USE"),
        )

    def record_experiment(
        self,
        decision: Mapping[str, Any],
        measured_outcomes: Mapping[str, float],
    ) -> dict[str, Any]:
        if decision.get("schema") != HYBRID_DECISION_SCHEMA:
            raise ValueError("unsupported hybrid decision schema")
        body = dict(decision)
        digest = body.pop("sha256", None)
        if digest != sha256_value(body):
            raise ValueError("hybrid decision fingerprint mismatch")
        proposals = [
            proposal
            for proposal in (
                decision.get("organ_proposal"),
                decision.get("neural_proposal"),
            )
            if isinstance(proposal, Mapping)
        ]
        required = {proposal["option"] for proposal in proposals}
        if not required:
            raise ValueError("experiment requires at least one proposal")
        if set(measured_outcomes) != required:
            raise ValueError("measured outcomes must cover exactly the proposed options")
        normalized: dict[str, float] = {}
        for option, value in measured_outcomes.items():
            score = float(value)
            if not isfinite(score):
                raise ValueError("measured outcome must be finite")
            normalized[option] = score
        ordered = sorted(normalized.items(), key=lambda item: (-item[1], item[0]))
        winner = None
        if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
            winner = ordered[0][0]
        receipt = {
            "schema": HYBRID_EXPERIMENT_SCHEMA,
            "decision_sha256": decision["sha256"],
            "measured_outcomes": normalized,
            "winner": winner,
            "classification": "MEASURED" if winner is not None else "MEASURED_TIE",
            "execution_authorized": False,
            "automatic_adoption": False,
            "host_admission_required": True,
        }
        return {**receipt, "sha256": sha256_value(receipt)}

    def organ_candidate_from_experiment(
        self,
        experiment: Mapping[str, Any],
        *,
        organ_id: str,
        version: str,
        purpose: str,
        conditions: Sequence[Predicate],
        proposal_score: float,
    ) -> dict[str, Any]:
        if experiment.get("schema") != HYBRID_EXPERIMENT_SCHEMA:
            raise ValueError("unsupported hybrid experiment schema")
        body = dict(experiment)
        digest = body.pop("sha256", None)
        if digest != sha256_value(body):
            raise ValueError("hybrid experiment fingerprint mismatch")
        winner = experiment.get("winner")
        if not isinstance(winner, str) or not winner:
            raise ValueError("organ candidate requires a measured non-tied winner")
        candidate_organ = HybridOrgan(
            organ_id=organ_id,
            version=version,
            purpose=purpose,
            conditions=tuple(conditions),
            proposal_option=winner,
            proposal_score=proposal_score,
            provenance={
                "kind": "hybrid-experiment-candidate",
                "experiment_sha256": experiment["sha256"],
                "admission": "not-canonical-until-explicit-host-review",
            },
            limitations=(
                "candidate is derived from bounded measured evidence only",
                "candidate has no execution or canonical-adoption authority",
            ),
        ).to_record()
        candidate_organ["status"] = "candidate"
        candidate_body = dict(candidate_organ)
        candidate_body.pop("sha256", None)
        candidate_organ["sha256"] = sha256_value(candidate_body)
        candidate = {
            "schema": HYBRID_CANDIDATE_SCHEMA,
            "candidate": candidate_organ,
            "experiment_sha256": experiment["sha256"],
            "automatic_adoption": False,
            "host_admission_required": True,
        }
        return {**candidate, "sha256": sha256_value(candidate)}

    def workspace_snapshot(self) -> dict[str, Any]:
        brain_snapshot = self.bound_brain.brain.to_snapshot()
        body = {
            "schema": HYBRID_WORKSPACE_SCHEMA,
            "organ_refs": [
                {
                    "id": organ.organ_id,
                    "version": organ.version,
                    "sha256": organ.fingerprint,
                }
                for organ in self.organs
            ],
            "neural_contract_sha256": self.bound_brain.contract.fingerprint,
            "brain_ref": {
                "schema": brain_snapshot["schema"],
                "sha256": brain_snapshot["sha256"],
            },
            "contains_neural_state": False,
            "execution_authorized": False,
            "automatic_adoption": False,
            "host_owns_persistence": True,
        }
        return {**body, "sha256": sha256_value(body)}
