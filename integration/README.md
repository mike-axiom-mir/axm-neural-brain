# Host bridge port

This directory is the neural-brain side of the explicit host↔brain boundary.

The canonical bridge contract is owned by
`mike-axiom-mir/axm-neural-network`. This branch pins the exact owner commit
and the first UC reference fingerprints in `NETWORK_CONTRACT_PIN.json`.

`BoundBrainHostPort` accepts only a contract whose complete canonical
SHA-256 matches the supplied fingerprint, whose embedded brain I/O contract
matches the bound brain exactly, and whose host-authority fields and four AXM
roots remain unchanged.

The port returns named advisory outputs only. It does not choose whether an
output is executed, where state is persisted, or which observations the host
provides.

The first real donor integration proof lives in the neural-network branch and
imports `mike-axiom-mir/axm-uc-neural` PR #3 at
`163410ce05a42879dca478d8ea0de7f6d17c4820`. This repository does not copy or
reinterpret the learning kernel while the Neural Core specialist is extracting
it.

Evidence boundary:
- TESTED here: exact contract hash, exact brain-I/O hash, fail-closed mismatch,
  and advisory named output shape through a bound-brain-compatible fixture.
- TESTED in the paired neural-network branch: real donor event→experience→
  retained response→restart path.
- NOT TESTED: general usefulness for UC decisions or other AXM hosts.
