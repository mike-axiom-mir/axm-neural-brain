import copy
import unittest

from integration.host_bridge_port import (
    BoundBrainHostPort,
    EXPECTED_ROOT_CONTRACT_REF,
    _sha256_json,
)

UC_BRAIN_IO = {
    "schema": "axm-brain-io/v0.1",
    "name": "axm-uc-bounded-outcome/v0.1",
    "inputs": [
        {"name": "route_ready", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
        {"name": "artifact_verified", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
        {"name": "candidate_reused", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
        {"name": "ambiguity_hold", "minimum": 0.0, "maximum": 1.0, "default": 0.0},
    ],
    "outputs": ["reuse_preference", "explore_preference"],
}

BRIDGE = {
    "schema": "axm-host-brain-bridge/v1.1",
    "name": "axm.uc.host-experience-reference/v1",
    "brain_io": UC_BRAIN_IO,
    "brain_io_sha256": "dfe07368b941bb8ac4eaa635f7a50ca652d01a43db90d91c24ef2c0751105fa1",
    "allow_target": True,
    "allow_reward": True,
    "authority": {
        "permissions": "host",
        "persistence": "host",
        "observations": "host",
        "execution": "host",
    },
    "root_contract": EXPECTED_ROOT_CONTRACT_REF,
}
BRIDGE_SHA = "13a554927a73d582056bef04680442e390294a2c788824af70f23ff26be0fd87"


class FakeContract:
    def to_dict(self):
        return copy.deepcopy(UC_BRAIN_IO)

    @property
    def fingerprint(self):
        return _sha256_json(UC_BRAIN_IO)


class FakeBoundBrain:
    def __init__(self):
        self.contract = FakeContract()
        self.accepted = 0

    def experience(self, observations, **kwargs):
        self.accepted += 1
        reuse = float(observations.get("candidate_reused", 0.0))
        ready = float(observations.get("route_ready", 0.0))
        return [reuse, ready - 1.0]

    def output_state(self, values):
        return dict(zip(UC_BRAIN_IO["outputs"], values))


class BrainPortTests(unittest.TestCase):
    def test_pinned_contract_hash_is_exact(self):
        self.assertEqual(_sha256_json(BRIDGE), BRIDGE_SHA)

    def test_canonical_root_reference_is_exact(self):
        self.assertEqual(
            EXPECTED_ROOT_CONTRACT_REF["contract_sha256"],
            "7d1eaeb05ce9353bccb5783a045ce9be91bf327c17bd93b47fdb68fd6bc46ed2",
        )

    def test_accepts_exact_contract_and_returns_advisory_output(self):
        brain = FakeBoundBrain()
        port = BoundBrainHostPort(brain, BRIDGE, BRIDGE_SHA)
        result = port.accept(
            {
                "schema": "axm-host-experience/v1",
                "contract_sha256": BRIDGE_SHA,
                "event_id": "uc-1",
                "observations": {
                    "route_ready": 1.0,
                    "artifact_verified": 1.0,
                    "candidate_reused": 1.0,
                    "ambiguity_hold": 0.0,
                },
                "target": None,
                "reward": None,
                "source": "axm-uc-neural",
                "tag": "uc-outcome",
                "directions": ["CREATE"],
            }
        )
        self.assertEqual(brain.accepted, 1)
        self.assertTrue(result["advisory_only"])
        self.assertEqual(
            set(result["values"]),
            {"reuse_preference", "explore_preference"},
        )

    def test_mismatch_fails_before_bound_brain_is_called(self):
        brain = FakeBoundBrain()
        port = BoundBrainHostPort(brain, BRIDGE, BRIDGE_SHA)
        with self.assertRaises(ValueError):
            port.accept(
                {
                    "schema": "axm-host-experience/v1",
                    "contract_sha256": "0" * 64,
                    "event_id": "bad",
                    "observations": {"route_ready": 1.0},
                }
            )
        self.assertEqual(brain.accepted, 0)

    def test_authority_rewrite_invalidates_contract(self):
        changed = copy.deepcopy(BRIDGE)
        changed["authority"]["execution"] = "neural"
        with self.assertRaises(ValueError):
            BoundBrainHostPort(FakeBoundBrain(), changed, BRIDGE_SHA)

    def test_root_contract_rewrite_invalidates_contract(self):
        changed = copy.deepcopy(BRIDGE)
        changed["root_contract"]["contract_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            BoundBrainHostPort(FakeBoundBrain(), changed, BRIDGE_SHA)


if __name__ == "__main__":
    unittest.main()
