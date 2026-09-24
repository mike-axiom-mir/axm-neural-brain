from __future__ import annotations

import hashlib
import json
from copy import deepcopy


class BrainSnapshotError(ValueError):
    pass


def _canonical_bytes(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def snapshot_payload(body: dict) -> dict:
    payload = deepcopy(body)
    return {
        "body": payload,
        "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
    }


def verify_snapshot(snapshot: dict) -> dict:
    if not isinstance(snapshot, dict) or "body" not in snapshot or "sha256" not in snapshot:
        raise BrainSnapshotError("snapshot must contain body and sha256")
    expected = hashlib.sha256(_canonical_bytes(snapshot["body"])).hexdigest()
    if expected != snapshot["sha256"]:
        raise BrainSnapshotError("brain snapshot integrity check failed")
    return deepcopy(snapshot["body"])
