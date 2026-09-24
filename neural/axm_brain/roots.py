from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


AXM_ROOTS = (
    "TRUTH",
    "AGENCY",
    "CONTINUITY",
    "WISDOM_BEFORE_SPEED",
)

ROOT_STATUSES = ("PASS", "HOLD", "UNKNOWN")

ROOT_DESCRIPTIONS = {
    "TRUTH": (
        "Keep claims and transition decisions grounded in inspectable evidence; "
        "keep uncertainty explicit instead of turning it into certainty."
    ),
    "AGENCY": (
        "Do not silently turn capability into domination or obedience. "
        "Keep user and system choices explicit inside granted authority."
    ),
    "CONTINUITY": (
        "Preserve demonstrated capability, provenance, and recoverable lineage "
        "unless an explicit supersede or forget decision says otherwise."
    ),
    "WISDOM_BEFORE_SPEED": (
        "Prefer reversible and evidenced change over faster irreversible change "
        "when uncertainty is material."
    ),
}


@dataclass(frozen=True)
class RootContract:
    schema: str = "axm-roots/v0.1"

    @property
    def roots(self) -> tuple[str, ...]:
        return AXM_ROOTS

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "roots": [
                {
                    "name": root,
                    "description": ROOT_DESCRIPTIONS[root],
                }
                for root in AXM_ROOTS
            ],
            "semantics": {
                "reward_objective": False,
                "obedience_rule": False,
                "capability_lock": False,
                "transition_review_contract": True,
                "hidden_control": False,
            },
        }

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


AXM_ROOT_CONTRACT = RootContract()


@dataclass(frozen=True)
class RootEvidence:
    root: str
    status: str
    reason: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        root = str(self.root).strip().upper()
        status = str(self.status).strip().upper()
        if root not in AXM_ROOTS:
            raise ValueError(f"unknown AXM root: {self.root!r}")
        if status not in ROOT_STATUSES:
            raise ValueError(
                f"unknown root status {self.status!r}; expected {ROOT_STATUSES}"
            )
        if not str(self.reason).strip():
            raise ValueError("root evidence requires a reason")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(str(item) for item in self.evidence_refs),
        )

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "status": self.status,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
        }


class RootReview:
    """Explicit evidence review against the four AXM roots.

    The review is not a neural reward and does not alter a brain. It only
    describes whether a candidate transition has enough evidence to be accepted
    as canonical by a caller.
    """

    SCHEMA = "axm-root-review/v0.1"

    def __init__(self, evidence) -> None:
        items = tuple(evidence)
        by_root: dict[str, RootEvidence] = {}
        for item in items:
            if not isinstance(item, RootEvidence):
                raise TypeError("root review accepts RootEvidence entries")
            if item.root in by_root:
                raise ValueError(f"duplicate root evidence: {item.root}")
            by_root[item.root] = item
        missing = [root for root in AXM_ROOTS if root not in by_root]
        extra = [root for root in by_root if root not in AXM_ROOTS]
        if missing or extra:
            raise ValueError(
                f"root review must cover exactly AXM_ROOTS; "
                f"missing={missing}, extra={extra}"
            )
        self.evidence = tuple(by_root[root] for root in AXM_ROOTS)

    @property
    def passed(self) -> bool:
        return all(item.status == "PASS" for item in self.evidence)

    @property
    def held_roots(self) -> tuple[str, ...]:
        return tuple(
            item.root for item in self.evidence if item.status != "PASS"
        )

    def to_dict(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "contract_sha256": AXM_ROOT_CONTRACT.fingerprint,
            "decision": "PASS" if self.passed else "HOLD",
            "evidence": [item.to_dict() for item in self.evidence],
        }
