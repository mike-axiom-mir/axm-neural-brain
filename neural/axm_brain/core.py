from __future__ import annotations

from dataclasses import asdict, dataclass
from math import tanh
from typing import Iterable, Optional

from .roots import AXM_ROOT_CONTRACT
from .state import snapshot_payload, verify_snapshot
from .taxonomy import ROOT_DIRECTIONS, normalize_directions


class XorShift64:
    """Tiny deterministic PRNG with explicitly serializable state."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        value = seed & self.MASK
        self.state = value if value else 0x9E3779B97F4A7C15

    def next_u64(self) -> int:
        x = self.state
        x ^= (x << 13) & self.MASK
        x ^= x >> 7
        x ^= (x << 17) & self.MASK
        self.state = x & self.MASK
        return self.state

    def uniform(self, low: float, high: float) -> float:
        unit = self.next_u64() / float(self.MASK)
        return low + (high - low) * unit


@dataclass(frozen=True)
class BrainConfig:
    input_size: int
    hidden_size: int
    output_size: int
    seed: int = 1
    learning_rate: float = 0.03
    reward_learning_rate: float = 0.006
    eligibility_decay: float = 0.92
    reward_baseline_decay: float = 0.98
    replay_capacity: int = 256
    sleep_replay_passes: int = 2
    sleep_learning_scale: float = 0.35
    sleep_prune_threshold: float = 1e-4
    weight_limit: float = 3.0

    def validate(self) -> None:
        for name in ("input_size", "hidden_size", "output_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 < self.eligibility_decay <= 1.0:
            raise ValueError("eligibility_decay must be in (0, 1]")
        if not 0.0 < self.reward_baseline_decay < 1.0:
            raise ValueError("reward_baseline_decay must be in (0, 1)")
        if self.replay_capacity < 0:
            raise ValueError("replay_capacity must be >= 0")
        if self.sleep_replay_passes < 0:
            raise ValueError("sleep_replay_passes must be >= 0")


@dataclass
class Experience:
    observation: list[float]
    target: Optional[list[float]] = None
    reward: Optional[float] = None
    source: str = "host"
    tag: str = ""
    directions: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "observation": list(self.observation),
            "target": (
                None
                if self.target is None
                else list(self.target)
            ),
            "reward": self.reward,
            "source": self.source,
            "tag": self.tag,
            "directions": list(
                normalize_directions(self.directions)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        return cls(
            observation=[
                float(x)
                for x in data["observation"]
            ],
            target=(
                None
                if data.get("target") is None
                else [
                    float(x)
                    for x in data["target"]
                ]
            ),
            reward=(
                None
                if data.get("reward") is None
                else float(data["reward"])
            ),
            source=str(data.get("source", "host")),
            tag=str(data.get("tag", "")),
            directions=normalize_directions(
                data.get("directions", ())
            ),
        )


def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [
        [0.0 for _ in range(cols)]
        for _ in range(rows)
    ]


def _clip(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


class AXMBrain:
    """State-native recurrent neural substrate with direct online learning.

    The brain has no tool, network, shell, or autonomous filesystem authority.
    The host supplies numeric observations and optional teaching/reward signals.

    The four AXM roots are embedded as an inspectable contract in every brain
    snapshot. They are not a neural reward, obedience rule, or capability lock.
    """

    SCHEMA = "axm-direct-brain/v0.2"
    ROOT_CONTRACT = AXM_ROOT_CONTRACT

    def __init__(self, config: BrainConfig):
        config.validate()
        self.config = config
        self.rng = XorShift64(config.seed)
        scale_in = (
            1.0
            / max(1, config.input_size) ** 0.5
        )
        scale_h = (
            1.0
            / max(1, config.hidden_size) ** 0.5
        )
        self.w_in = self._random_matrix(
            config.hidden_size,
            config.input_size,
            scale_in,
        )
        self.w_rec = self._random_matrix(
            config.hidden_size,
            config.hidden_size,
            scale_h,
        )
        self.w_out = self._random_matrix(
            config.output_size,
            config.hidden_size,
            scale_h,
        )
        self.b_hidden = [
            0.0
        ] * config.hidden_size
        self.b_out = [
            0.0
        ] * config.output_size
        self.hidden = [
            0.0
        ] * config.hidden_size
        self.last_output = [
            0.0
        ] * config.output_size
        self.trace_in = _zeros(
            config.hidden_size,
            config.input_size,
        )
        self.trace_rec = _zeros(
            config.hidden_size,
            config.hidden_size,
        )
        self.trace_out = _zeros(
            config.output_size,
            config.hidden_size,
        )
        self.reward_baseline = 0.0
        self.replay: list[Experience] = []
        self.mode = "wake"
        self.cycle = 0
        self.steps = 0
        self.sleep_count = 0
        self.host_experience_count = 0
        self.supervised_update_count = 0
        self.reward_update_count = 0
        self.sleep_replay_event_count = 0
        self.total_pruned_weights = 0
        self.direction_experience_counts = {
            direction: 0
            for direction in ROOT_DIRECTIONS
        }
        self.untagged_experience_count = 0

    def _random_matrix(
        self,
        rows: int,
        cols: int,
        scale: float,
    ) -> list[list[float]]:
        return [
            [
                self.rng.uniform(
                    -scale,
                    scale,
                )
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    @staticmethod
    def _validate_vector(
        values: Iterable[float],
        size: int,
        name: str,
    ) -> list[float]:
        vec = [
            float(v)
            for v in values
        ]
        if len(vec) != size:
            raise ValueError(
                f"{name} must have length "
                f"{size}, got {len(vec)}"
            )
        return vec

    def wake(self) -> None:
        if self.mode == "wake":
            return
        self.mode = "wake"
        self.cycle += 1
        self.hidden = [
            0.0
        ] * self.config.hidden_size
        self.last_output = [
            0.0
        ] * self.config.output_size

    def predict(
        self,
        observation: Iterable[float],
        *,
        update_state: bool = True,
    ) -> list[float]:
        obs = self._validate_vector(
            observation,
            self.config.input_size,
            "observation",
        )
        previous_hidden = list(self.hidden)
        hidden = []
        for i in range(
            self.config.hidden_size
        ):
            total = self.b_hidden[i]
            total += sum(
                w * x
                for w, x in zip(
                    self.w_in[i],
                    obs,
                )
            )
            total += sum(
                w * h
                for w, h in zip(
                    self.w_rec[i],
                    previous_hidden,
                )
            )
            hidden.append(tanh(total))
        output = []
        for o in range(
            self.config.output_size
        ):
            total = (
                self.b_out[o]
                + sum(
                    w * h
                    for w, h in zip(
                        self.w_out[o],
                        hidden,
                    )
                )
            )
            output.append(tanh(total))
        if update_state:
            self.hidden = hidden
            self.last_output = output
            self._update_eligibility(
                obs,
                previous_hidden,
                hidden,
            )
            self.steps += 1
        return output

    def experience(
        self,
        event: Experience,
        *,
        remember: bool = True,
    ) -> list[float]:
        if self.mode != "wake":
            raise RuntimeError(
                "direct experience is accepted "
                "only in wake mode"
            )
        normalized = Experience.from_dict(
            event.to_dict()
        )
        self.host_experience_count += 1
        if normalized.directions:
            for direction in (
                normalized.directions
            ):
                self.direction_experience_counts[
                    direction
                ] += 1
        else:
            self.untagged_experience_count += 1
        output = self.predict(
            normalized.observation,
            update_state=True,
        )
        if normalized.target is not None:
            self._supervised_update(
                normalized.observation,
                normalized.target,
                scale=1.0,
            )
        if normalized.reward is not None:
            self._reward_update(
                float(normalized.reward)
            )
        if (
            remember
            and self.config.replay_capacity > 0
        ):
            self.replay.append(normalized)
            if (
                len(self.replay)
                > self.config.replay_capacity
            ):
                del self.replay[
                    :
                    len(self.replay)
                    - self.config.replay_capacity
                ]
        return output

    def _update_eligibility(
        self,
        obs: list[float],
        previous_hidden: list[float],
        hidden: list[float],
    ) -> None:
        d = self.config.eligibility_decay
        for i in range(
            self.config.hidden_size
        ):
            for j in range(
                self.config.input_size
            ):
                self.trace_in[i][j] = (
                    d * self.trace_in[i][j]
                    + hidden[i] * obs[j]
                )
            for j in range(
                self.config.hidden_size
            ):
                self.trace_rec[i][j] = (
                    d * self.trace_rec[i][j]
                    + hidden[i]
                    * previous_hidden[j]
                )
        for o in range(
            self.config.output_size
        ):
            for i in range(
                self.config.hidden_size
            ):
                self.trace_out[o][i] = (
                    d * self.trace_out[o][i]
                    + self.last_output[o]
                    * hidden[i]
                )

    def _supervised_update(
        self,
        observation: list[float],
        target: Iterable[float],
        *,
        scale: float,
    ) -> None:
        target_vec = self._validate_vector(
            target,
            self.config.output_size,
            "target",
        )
        lr = (
            self.config.learning_rate
            * scale
        )
        out_errors = [
            target_vec[o]
            - self.last_output[o]
            for o in range(
                self.config.output_size
            )
        ]
        hidden_errors = [
            0.0
        ] * self.config.hidden_size
        for i in range(
            self.config.hidden_size
        ):
            hidden_errors[i] = sum(
                out_errors[o]
                * self.w_out[o][i]
                for o in range(
                    self.config.output_size
                )
            )

        for o in range(
            self.config.output_size
        ):
            grad = (
                out_errors[o]
                * (
                    1.0
                    - self.last_output[o] ** 2
                )
            )
            for i in range(
                self.config.hidden_size
            ):
                self.w_out[o][i] = _clip(
                    self.w_out[o][i]
                    + lr
                    * grad
                    * self.hidden[i],
                    self.config.weight_limit,
                )
            self.b_out[o] = _clip(
                self.b_out[o]
                + lr * grad,
                self.config.weight_limit,
            )

        for i in range(
            self.config.hidden_size
        ):
            grad = (
                hidden_errors[i]
                * (
                    1.0
                    - self.hidden[i] ** 2
                )
            )
            for j in range(
                self.config.input_size
            ):
                self.w_in[i][j] = _clip(
                    self.w_in[i][j]
                    + lr
                    * grad
                    * observation[j],
                    self.config.weight_limit,
                )
            self.b_hidden[i] = _clip(
                self.b_hidden[i]
                + lr * grad,
                self.config.weight_limit,
            )
        self.supervised_update_count += 1

    def _reward_update(
        self,
        reward: float,
    ) -> None:
        cfg = self.config
        advantage = (
            reward
            - self.reward_baseline
        )
        self.reward_baseline = (
            cfg.reward_baseline_decay
            * self.reward_baseline
            + (
                1.0
                - cfg.reward_baseline_decay
            )
            * reward
        )
        rate = (
            cfg.reward_learning_rate
            * advantage
        )
        for i in range(
            cfg.hidden_size
        ):
            for j in range(
                cfg.input_size
            ):
                self.w_in[i][j] = _clip(
                    self.w_in[i][j]
                    + rate
                    * self.trace_in[i][j],
                    cfg.weight_limit,
                )
            for j in range(
                cfg.hidden_size
            ):
                self.w_rec[i][j] = _clip(
                    self.w_rec[i][j]
                    + rate
                    * self.trace_rec[i][j],
                    cfg.weight_limit,
                )
        for o in range(
            cfg.output_size
        ):
            for i in range(
                cfg.hidden_size
            ):
                self.w_out[o][i] = _clip(
                    self.w_out[o][i]
                    + rate
                    * self.trace_out[o][i],
                    cfg.weight_limit,
                )
        self.reward_update_count += 1

    def sleep(self) -> dict:
        if self.mode == "sleep":
            return {
                "replayed": 0,
                "pruned": 0,
                "sleep_count": (
                    self.sleep_count
                ),
            }
        self.mode = "sleep"
        before_hidden = list(
            self.hidden
        )
        replayed = 0
        for _ in range(
            self.config.sleep_replay_passes
        ):
            for event in list(
                self.replay
            ):
                self.hidden = [
                    0.0
                ] * self.config.hidden_size
                self.predict(
                    event.observation,
                    update_state=True,
                )
                if event.target is not None:
                    self._supervised_update(
                        event.observation,
                        event.target,
                        scale=(
                            self.config
                            .sleep_learning_scale
                        ),
                    )
                if event.reward is not None:
                    self._reward_update(
                        float(event.reward)
                        * self.config
                        .sleep_learning_scale
                    )
                replayed += 1
        pruned = (
            self._prune_small_weights()
        )
        self.hidden = before_hidden
        self.sleep_count += 1
        self.sleep_replay_event_count += (
            replayed
        )
        self.total_pruned_weights += (
            pruned
        )
        return {
            "replayed": replayed,
            "pruned": pruned,
            "sleep_count": self.sleep_count,
        }

    def _prune_small_weights(
        self,
    ) -> int:
        threshold = (
            self.config
            .sleep_prune_threshold
        )
        pruned = 0
        for matrix in (
            self.w_in,
            self.w_rec,
            self.w_out,
        ):
            for row in matrix:
                for i, value in enumerate(
                    row
                ):
                    if (
                        0.0
                        < abs(value)
                        < threshold
                    ):
                        row[i] = 0.0
                        pruned += 1
        return pruned

    def to_snapshot(self) -> dict:
        body = {
            "schema": self.SCHEMA,
            "roots": {
                "contract": (
                    AXM_ROOT_CONTRACT
                    .to_dict()
                ),
                "sha256": (
                    AXM_ROOT_CONTRACT
                    .fingerprint
                ),
            },
            "config": asdict(
                self.config
            ),
            "state": {
                "rng_state": (
                    self.rng.state
                ),
                "w_in": self.w_in,
                "w_rec": self.w_rec,
                "w_out": self.w_out,
                "b_hidden": self.b_hidden,
                "b_out": self.b_out,
                "hidden": self.hidden,
                "last_output": (
                    self.last_output
                ),
                "trace_in": (
                    self.trace_in
                ),
                "trace_rec": (
                    self.trace_rec
                ),
                "trace_out": (
                    self.trace_out
                ),
                "reward_baseline": (
                    self.reward_baseline
                ),
                "replay": [
                    event.to_dict()
                    for event
                    in self.replay
                ],
                "mode": self.mode,
                "cycle": self.cycle,
                "steps": self.steps,
                "sleep_count": (
                    self.sleep_count
                ),
                "host_experience_count": (
                    self.host_experience_count
                ),
                "supervised_update_count": (
                    self.supervised_update_count
                ),
                "reward_update_count": (
                    self.reward_update_count
                ),
                "sleep_replay_event_count": (
                    self.sleep_replay_event_count
                ),
                "total_pruned_weights": (
                    self.total_pruned_weights
                ),
                "direction_experience_counts": (
                    self.direction_experience_counts
                ),
                "untagged_experience_count": (
                    self.untagged_experience_count
                ),
            },
        }
        return snapshot_payload(body)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: dict,
    ) -> "AXMBrain":
        body = verify_snapshot(snapshot)
        if (
            body.get("schema")
            != cls.SCHEMA
        ):
            raise ValueError(
                "unsupported brain schema: "
                f"{body.get('schema')}"
            )
        roots = body.get("roots")
        if not isinstance(roots, dict):
            raise ValueError(
                "brain snapshot is missing "
                "the AXM root contract"
            )
        if (
            roots.get("sha256")
            != AXM_ROOT_CONTRACT.fingerprint
            or roots.get("contract")
            != AXM_ROOT_CONTRACT.to_dict()
        ):
            raise ValueError(
                "brain snapshot AXM root "
                "contract mismatch"
            )

        brain = cls(
            BrainConfig(
                **body["config"]
            )
        )
        state = body["state"]
        brain.rng.state = int(
            state["rng_state"]
        )
        for name in (
            "w_in",
            "w_rec",
            "w_out",
            "trace_in",
            "trace_rec",
            "trace_out",
        ):
            setattr(
                brain,
                name,
                [
                    [
                        float(v)
                        for v in row
                    ]
                    for row
                    in state[name]
                ],
            )
        for name in (
            "b_hidden",
            "b_out",
            "hidden",
            "last_output",
        ):
            setattr(
                brain,
                name,
                [
                    float(v)
                    for v
                    in state[name]
                ],
            )
        brain.reward_baseline = float(
            state["reward_baseline"]
        )
        brain.replay = [
            Experience.from_dict(x)
            for x in state["replay"]
        ]
        brain.mode = str(
            state["mode"]
        )
        brain.cycle = int(
            state["cycle"]
        )
        brain.steps = int(
            state["steps"]
        )
        brain.sleep_count = int(
            state["sleep_count"]
        )
        brain.host_experience_count = int(
            state.get(
                "host_experience_count",
                0,
            )
        )
        brain.supervised_update_count = int(
            state.get(
                "supervised_update_count",
                0,
            )
        )
        brain.reward_update_count = int(
            state.get(
                "reward_update_count",
                0,
            )
        )
        brain.sleep_replay_event_count = int(
            state.get(
                "sleep_replay_event_count",
                0,
            )
        )
        brain.total_pruned_weights = int(
            state.get(
                "total_pruned_weights",
                0,
            )
        )
        persisted_counts = state.get(
            "direction_experience_counts",
            {},
        )
        brain.direction_experience_counts = {
            direction: int(
                persisted_counts.get(
                    direction,
                    0,
                )
            )
            for direction
            in ROOT_DIRECTIONS
        }
        brain.untagged_experience_count = int(
            state.get(
                "untagged_experience_count",
                0,
            )
        )
        return brain
