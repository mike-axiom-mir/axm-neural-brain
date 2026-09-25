from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from math import isfinite
from typing import Mapping, Sequence

from .core import AXMBrain, BrainConfig


@dataclass(frozen=True)
class Channel:
    name: str
    minimum: float = -1.0
    maximum: float = 1.0
    default: float = 0.0

    def validate(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("channel name must be non-empty and trimmed")
        for field_name, raw in (
            ("minimum", self.minimum),
            ("maximum", self.maximum),
            ("default", self.default),
        ):
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"channel {self.name}: {field_name} must be a finite number"
                ) from exc
            if not isfinite(number):
                raise ValueError(
                    f"channel {self.name}: {field_name} must be finite"
                )
        if self.maximum <= self.minimum:
            raise ValueError(f"channel {self.name}: maximum must be greater than minimum")
        if not self.minimum <= self.default <= self.maximum:
            raise ValueError(f"channel {self.name}: default outside channel range")

    def encode(self, value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"channel {self.name}: value must be a finite number"
            ) from exc
        if not isfinite(value):
            raise ValueError(f"channel {self.name}: value must be finite")
        if value < self.minimum or value > self.maximum:
            raise ValueError(
                f"channel {self.name}: value {value} outside "
                f"[{self.minimum}, {self.maximum}]"
            )
        unit = (value - self.minimum) / (self.maximum - self.minimum)
        return unit * 2.0 - 1.0


@dataclass(frozen=True)
class BrainIOContract:
    name: str
    inputs: tuple[Channel, ...]
    outputs: tuple[str, ...]
    schema: str = "axm-brain-io/v0.1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("contract name is required")
        if not self.inputs or not self.outputs:
            raise ValueError("contract requires at least one input and one output")
        for channel in self.inputs:
            channel.validate()
        input_names = [channel.name for channel in self.inputs]
        if len(set(input_names)) != len(input_names):
            raise ValueError("duplicate input channel")
        if any(not name or name.strip() != name for name in self.outputs):
            raise ValueError("output names must be non-empty and trimmed")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("duplicate output channel")

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "name": self.name,
            "inputs": [asdict(channel) for channel in self.inputs],
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BrainIOContract":
        return cls(
            schema=str(data.get("schema", "axm-brain-io/v0.1")),
            name=str(data["name"]),
            inputs=tuple(Channel(**item) for item in data["inputs"]),
            outputs=tuple(str(item) for item in data["outputs"]),
        )

    def encode(self, state: Mapping[str, float]) -> list[float]:
        allowed = {channel.name for channel in self.inputs}
        unknown = set(state) - allowed
        if unknown:
            raise ValueError(f"unknown input channel(s): {sorted(unknown)}")
        return [
            channel.encode(state.get(channel.name, channel.default))
            for channel in self.inputs
        ]

    def decode(self, values: Sequence[float]) -> dict[str, float]:
        if len(values) != len(self.outputs):
            raise ValueError(
                f"expected {len(self.outputs)} outputs, got {len(values)}"
            )
        decoded = {}
        for name, value in zip(self.outputs, values):
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"output {name}: value must be a finite number"
                ) from exc
            if not isfinite(number):
                raise ValueError(f"output {name}: value must be finite")
            decoded[name] = number
        return decoded

    def new_brain(
        self,
        *,
        hidden_size: int,
        seed: int = 1,
        **kwargs,
    ) -> "BoundBrain":
        brain = AXMBrain(
            BrainConfig(
                input_size=len(self.inputs),
                hidden_size=hidden_size,
                output_size=len(self.outputs),
                seed=seed,
                **kwargs,
            )
        )
        return BoundBrain(self, brain)


class BoundBrain:
    """Binds learned parameters to the exact software signal meaning they learned."""

    SCHEMA = "axm-bound-brain/v0.1"

    def __init__(self, contract: BrainIOContract, brain: AXMBrain):
        if brain.config.input_size != len(contract.inputs):
            raise ValueError("brain input size does not match contract")
        if brain.config.output_size != len(contract.outputs):
            raise ValueError("brain output size does not match contract")
        self.contract = contract
        self.brain = brain

    def experience(
        self,
        state: Mapping[str, float],
        *,
        target=None,
        reward=None,
        source="host",
        tag="",
        directions=(),
    ):
        from .core import Experience

        observation = self.contract.encode(state)
        return self.brain.experience(
            Experience(
                observation=observation,
                target=target,
                reward=reward,
                source=source,
                tag=tag,
                directions=directions,
            )
        )

    def output_state(self, raw_output: Sequence[float]) -> dict[str, float]:
        return self.contract.decode(raw_output)

    def to_snapshot(self) -> dict:
        return {
            "schema": self.SCHEMA,
            "contract": self.contract.to_dict(),
            "contract_sha256": self.contract.fingerprint,
            "brain": self.brain.to_snapshot(),
        }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "BoundBrain":
        if snapshot.get("schema") != cls.SCHEMA:
            raise ValueError("unsupported bound brain schema")
        contract = BrainIOContract.from_dict(snapshot["contract"])
        if contract.fingerprint != snapshot.get("contract_sha256"):
            raise ValueError("brain interface contract integrity check failed")
        brain = AXMBrain.from_snapshot(snapshot["brain"])
        return cls(contract, brain)
