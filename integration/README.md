# Host bridge port

This directory is the neural-brain side of the explicit host↔brain boundary.

The canonical bridge contract is owned by
`mike-axiom-mir/axm-neural-network`. This branch pins the exact owner commit
`c16f5d815d74efd3c3819c8aa8ac8eceff6e0c9f` and the first UC reference
fingerprints in `NETWORK_CONTRACT_PIN.json`.

`BoundBrainHostPort` accepts only a contract whose complete canonical SHA-256
matches the supplied fingerprint, whose embedded brain I/O contract matches the
bound brain exactly, whose host-authority fields remain host-owned, and whose
root-contract reference exactly matches the canonical `axm-roots/v0.1`
fingerprint from neural-core extraction
`a0f5de4b19bf515e145caf50b12130ab740869b5`.

The port returns named advisory outputs only. It does not choose whether an
output is executed, where state is persisted, or which observations the host
provides.

The first real integration proof lives in the paired neural-network branch and
runs the same UC bridge suite against both the pinned
`mike-axiom-mir/axm-uc-neural` PR #3 donor and the dedicated neural-core
extraction. No learning kernel is copied into this adapter surface.

Evidence boundary:
- TESTED here: exact contract hash, exact brain-I/O hash, exact canonical root
  contract reference, fail-closed mismatch, and advisory named output shape
  through a bound-brain-compatible fixture.
- TESTED in the paired neural-network branch: real event→experience→retained
  response→restart path.
- NOT TESTED: general usefulness for UC decisions or other AXM hosts.
