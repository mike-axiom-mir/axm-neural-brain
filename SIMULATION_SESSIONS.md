# Persistent simulation learning sessions

`neural.axm_brain.simulation_session.SimulationSession` learns directly from
host-provided deterministic simulation families. It uses the shared contract
from `axm-neural-network`; ordinary Brain imports retain their existing
dependency-free behavior.

The session owns an experimental descendant of the supplied parent checkpoint.
It persists the complete learner, ordered provider identities and parameters,
training/held-out seed split, scheduler RNG, per-family visit counts and moving
prediction errors, and a linked episode history. JSON save/restore continues
the same sequence as an uninterrupted run. `reproduce()` reruns the learning
history to check causality beyond stored integrity hashes.

Three explicit host-selected policies are available:

| Policy | Next simulation family |
| --- | --- |
| `fixed` | Round robin in the declared family order |
| `random` | Seeded uniform sampling |
| `error_guided` | One initial visit per family, then sample proportional to moving pre-update error plus 0.01 |

The last policy is an adaptive heuristic driven by prediction errors. It is
not a trained neural meta-selector. Which policy is useful must be measured on
held-out examples with an equal experience budget; none is assumed superior.
Evaluation never changes scheduling or learned state. It is reported separately
from training transitions. The transient recurrent state resets per numeric
prediction; this path does not test temporal credit assignment.

`advance(episodes)` reserves a conservative worst-case transition budget before
running providers. It updates an isolated copy and commits the whole call only
after every packet verifies. Invalid later transitions leave the session
unchanged. The contract caps providers, history, seed lists and transitions;
it is not a wall-time/memory sandbox for arbitrary Python provider code.

Checkpoints remain experimental descendants. No canonical parent replacement,
external action, filesystem write or training occurs automatically. Hosts own
storage; the UC host adds an exclusive writer lock and atomic checkpoint file.

Example (matching checkouts on `PYTHONPATH`):

```python
from neural.axm_brain import AXMBrain, BrainConfig
from neural.axm_brain.simulation_session import SimulationSession
from axm_neural_network.microsim import MicroDynamics

parent = AXMBrain(BrainConfig(3, 8, 2, seed=41))
providers = {"dynamics": MicroDynamics()}
session = SimulationSession(parent, providers, range(12), range(100, 112))
session.advance(24)
saved = session.to_snapshot()
continued = SimulationSession.from_snapshot(saved, providers)
print(continued.evaluate())
```

Fresh integration tests cover exact JSON continuation under every policy,
held-out isolation, actual replay, atomic failure, changed-provider rejection,
forged accounting and budget failures. UC separately exercises five inputs and
four targets against its existing canvas-fit implementation.
