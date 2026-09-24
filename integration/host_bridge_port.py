from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Mapping

BRIDGE_SCHEMA = "axm-host-brain-bridge/v1.1"
EXPERIENCE_SCHEMA = "axm-host-experience/v1"
ACTION_SCHEMA = "axm-host-action-proposal/v1"

EXPECTED_ROOT_CONTRACT_REF = {
    "schema": "axm-root-contract-ref/v1",
    "owner_repo": "mike-axiom-mir/axm-neural-brain",
    "owner_commit": "a0f5de4b19bf515e145caf50b12130ab740869b5",
    "contract_schema": "axm-roots/v0.1",
    "contract_sha256": "7d1eaeb05ce9353bccb5783a045ce9be91bf327c17bd93b47fdb68fd6bc46ed2",
}

EXPECTED_AUTHORITY = {
    "permissions": "host",
    "persistence": "host",
    "observations": "host",
    "execution": "host",
}

def _sha256_json(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def _finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number

class BoundBrainHostPort:
    """Brain-side endpoint for a pre-validated host bridge contract."""

    def __init__(
        self,
        bound_brain,
        contract: Mapping[str, object],
        contract_sha256: str,
    ):
        body = deepcopy(dict(contract))
        if body.get("schema") != BRIDGE_SCHEMA:
            raise ValueError("unsupported bridge schema")
        if body.get("authority") != EXPECTED_AUTHORITY:
            raise ValueError("bridge authority boundary mismatch")
        if body.get("root_contract") != EXPECTED_ROOT_CONTRACT_REF:
            raise ValueError("AXM root contract boundary mismatch")
        if _sha256_json(body) != contract_sha256:
            raise ValueError("bridge contract fingerprint mismatch")

        brain_io = body.get("brain_io")
        if not isinstance(brain_io, Mapping):
            raise ValueError("bridge contract is missing brain_io")
        brain_io_sha256 = _sha256_json(brain_io)
        if brain_io_sha256 != body.get("brain_io_sha256"):
            raise ValueError("brain I/O fingerprint mismatch")
        if bound_brain.contract.to_dict() != dict(brain_io):
            raise ValueError("bound brain I/O contract mismatch")
        if bound_brain.contract.fingerprint != brain_io_sha256:
            raise ValueError("bound brain I/O fingerprint mismatch")

        self.bound_brain = bound_brain
        self.contract = body
        self.contract_sha256 = contract_sha256
        self.input_names = tuple(item["name"] for item in brain_io["inputs"])
        self.output_names = tuple(brain_io["outputs"])

    def accept(self, event: Mapping[str, object]) -> dict:
        if event.get("schema") != EXPERIENCE_SCHEMA:
            raise ValueError("unsupported experience schema")
        if event.get("contract_sha256") != self.contract_sha256:
            raise ValueError("experience contract fingerprint mismatch")

        observations = dict(event.get("observations", {}))
        unknown = set(observations) - set(self.input_names)
        if unknown:
            raise ValueError(f"unknown observation channel(s): {sorted(unknown)}")
        for name, value in observations.items():
            _finite(value, f"observation {name}")

        target_map = event.get("target")
        target = None
        if target_map is not None:
            if not self.contract.get("allow_target", False):
                raise ValueError("target teaching signal is disabled")
            target_map = dict(target_map)
            if set(target_map) != set(self.output_names):
                raise ValueError("target must name every output channel exactly")
            target = [_finite(target_map[name], f"target {name}") for name in self.output_names]

        reward = event.get("reward")
        if reward is not None:
            if not self.contract.get("allow_reward", False):
                raise ValueError("reward teaching signal is disabled")
            reward = _finite(reward, "reward")

        raw_output = self.bound_brain.experience(
            observations,
            target=target,
            reward=reward,
            source=str(event.get("source", "host")),
            tag=str(event.get("tag", "")),
            directions=tuple(event.get("directions", ())),
        )
        values = self.bound_brain.output_state(raw_output)
        if set(values) != set(self.output_names):
            raise ValueError("brain returned output channels outside contract")
        return {
            "schema": ACTION_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "event_id": str(event.get("event_id", "")),
            "values": values,
            "advisory_only": True,
        }
