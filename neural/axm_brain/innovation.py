"""Bounded descendant experiments and a learned experiment selector.

The continuing brain is never mutated. Results remain archive-only. Dataset
partitions, exact checkpoints, changes, costs and regressions are retained as
separate evidence. The selector learns validation outcomes, never test scores.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
import math

from .core import AXMBrain, BrainConfig, Experience
from .state import snapshot_payload, verify_snapshot

MUTABLE_FIELDS = frozenset({"learning_rate", "replay_capacity", "sleep_replay_passes", "sleep_learning_scale"})


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _text(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty trimmed text")
    return value


@dataclass(frozen=True)
class Mutation:
    name: str
    changes: tuple[tuple[str, float | int], ...]

    def __post_init__(self):
        _text(self.name, "mutation name")
        values = dict(self.changes)
        if len(values) != len(self.changes) or not values or set(values) - MUTABLE_FIELDS:
            raise ValueError("mutation must contain unique supported configuration fields")
        for key, value in values.items():
            _number(value, key)
            if value < 0 or (key in ("replay_capacity", "sleep_replay_passes") and type(value) is not int):
                raise ValueError("mutation values must be nonnegative and preserve field types")
        object.__setattr__(self, "changes", tuple(sorted(values.items())))

    @property
    def mechanism_id(self):
        # Names do not create a new mechanism. Integer/float rates normalize.
        values = {k: (v if k in ("replay_capacity", "sleep_replay_passes") else float(v)) for k, v in self.changes}
        return snapshot_payload(values)["sha256"]

    def to_dict(self):
        return {"name": self.name, "changes": dict(self.changes), "mechanism_id": self.mechanism_id}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"name", "changes", "mechanism_id"}:
            raise ValueError("invalid mutation record")
        result = cls(value["name"], tuple(value["changes"].items()))
        if result.mechanism_id != value["mechanism_id"]:
            raise ValueError("mutation mechanism fingerprint mismatch")
        return result

    @classmethod
    def compose(cls, name, *parts):
        changes = {}
        for part in parts:
            for key, value in part.changes:
                if key in changes and changes[key] != value:
                    raise ValueError("conflicting mutation composition")
                changes[key] = value
        return cls(name, tuple(changes.items()))


@dataclass(frozen=True)
class Sample:
    sample_id: str
    observation: tuple[float, ...]
    target: tuple[float, ...]

    def __post_init__(self):
        _text(self.sample_id, "sample id")
        if not self.observation or not self.target:
            raise ValueError("sample requires observation and target")
        object.__setattr__(self, "observation", tuple(_number(x, "observation") for x in self.observation))
        object.__setattr__(self, "target", tuple(_number(x, "target") for x in self.target))


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    group_id: str
    features: tuple[float, ...]
    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    held_out: tuple[Sample, ...]
    regression: tuple[Sample, ...] = ()
    source_event: str = "declared_experiment"
    raw_prompt: str | None = None
    interpreted_intent: str | None = None

    def __post_init__(self):
        for name in ("task_id", "group_id", "source_event"):
            _text(getattr(self, name), name)
        for name in ("raw_prompt", "interpreted_intent"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text or null")
        object.__setattr__(self, "features", tuple(_number(x, "feature") for x in self.features))
        if not self.features:
            raise ValueError("explicit task features required")
        seen_ids, seen_inputs = set(), set()
        for name in ("train", "validation", "held_out", "regression"):
            samples = tuple(getattr(self, name))
            object.__setattr__(self, name, samples)
            if name != "regression" and not samples:
                raise ValueError("training, validation and held-out splits must be nonempty")
            current_inputs = set()
            for sample in samples:
                if not isinstance(sample, Sample) or sample.sample_id in seen_ids:
                    raise ValueError("samples require globally distinct evidence ids")
                seen_ids.add(sample.sample_id)
                # Regression can retest an earlier task at the same input.
                if name != "regression" and sample.observation in seen_inputs:
                    raise ValueError("train/validation/test observation overlap")
                current_inputs.add(sample.observation)
            if name != "regression": seen_inputs.update(current_inputs)

    def validate_dimensions(self, config):
        for name in ("train", "validation", "held_out", "regression"):
            for sample in getattr(self, name):
                if len(sample.observation) != config.input_size or len(sample.target) != config.output_size:
                    raise ValueError("sample dimensions do not match the parent brain")

    def to_dict(self):
        # Round-trip tuple containers into a stable JSON wire representation.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_dict(cls, value):
        data = deepcopy(value)
        for name in ("train", "validation", "held_out", "regression"):
            data[name] = tuple(Sample(**sample) for sample in data[name])
        return cls(**data)


def _reset_transient(brain):
    brain.hidden = [0.0]*brain.config.hidden_size
    brain.last_output = [0.0]*brain.config.output_size


def _error(brain, samples):
    if not samples:
        return None
    probe = AXMBrain.from_snapshot(brain.to_snapshot())
    errors = []
    for sample in samples:
        _reset_transient(probe)
        output = probe.predict(sample.observation, update_state=False)
        errors.append(sum((a-b)**2 for a, b in zip(output, sample.target))/len(output))
    return _number(sum(errors)/len(errors), "evaluation error")


def descendant(parent_snapshot, mutation):
    body = verify_snapshot(parent_snapshot)
    AXMBrain.from_snapshot(parent_snapshot)  # Validate the original state first.
    body["config"].update(dict(mutation.changes))
    cfg = BrainConfig(**body["config"])
    cfg.validate()
    capacity = cfg.replay_capacity
    body["state"]["replay"] = body["state"]["replay"][-capacity:] if capacity else []
    return AXMBrain.from_snapshot(snapshot_payload(body))


def _train(brain, task, epochs, consolidate):
    brain.wake()
    for _ in range(epochs):
        for sample in task.train:
            _reset_transient(brain)
            brain.experience(Experience(list(sample.observation), target=list(sample.target),
                                        source=task.source_event, tag=sample.sample_id))
    if consolidate:
        brain.sleep()
        brain.wake()


def _measure(brain, task, initial, before_regression):
    snapshot = brain.to_snapshot()
    restored = AXMBrain.from_snapshot(snapshot)
    cfg = brain.config
    regression = _error(brain, task.regression)
    return {
        "train_mse": _error(brain, task.train),
        "validation_mse": _error(brain, task.validation),
        "held_out_mse": _error(brain, task.held_out),
        "regression_mse": regression,
        "forgetting_delta": None if regression is None else regression-before_regression,
        "restore_exact": restored.to_snapshot() == snapshot,
        "restored_held_out_mse": _error(restored, task.held_out),
        "host_experiences": brain.host_experience_count-initial.host_experience_count,
        "supervised_updates": brain.supervised_update_count-initial.supervised_update_count,
        "sleep_replays": brain.sleep_replay_event_count-initial.sleep_replay_event_count,
        "parameter_count": cfg.hidden_size*cfg.input_size + cfg.hidden_size**2 + cfg.output_size*cfg.hidden_size + cfg.hidden_size + cfg.output_size,
        "snapshot_bytes": len(json.dumps(snapshot, sort_keys=True).encode()),
    }


def run_experiment(parent, task, mutation, *, epochs=1, consolidate=False, max_updates=100_000):
    """Run equal-data parent and candidate controls without changing the parent."""
    if type(epochs) is not int or epochs < 1 or type(consolidate) is not bool:
        raise ValueError("epochs must be positive and consolidation explicitly boolean")
    if type(max_updates) is not int or max_updates < 1:
        raise ValueError("host update budget must be positive")
    task.validate_dimensions(parent.config)
    original = parent.to_snapshot()
    baseline = AXMBrain.from_snapshot(original)
    candidate = descendant(original, mutation)
    before = candidate.to_snapshot()
    total = 0
    for brain in (baseline, candidate):
        replays = min(brain.config.replay_capacity, len(brain.replay) + epochs*len(task.train))
        total += epochs*len(task.train) + (replays*brain.config.sleep_replay_passes if consolidate else 0)
    if total > max_updates:
        raise ValueError("experiment exceeds the host update budget")
    before_regression = _error(parent, task.regression)
    initial_error = _error(parent, task.held_out)
    _train(baseline, task, epochs, consolidate)
    _train(candidate, task, epochs, consolidate)
    record = {
        "schema": "axm.neural-experiment/v1", "adoption_state": "archive-only",
        "task": task.to_dict(), "task_sha256": snapshot_payload(task.to_dict())["sha256"],
        "mutation": mutation.to_dict(), "epochs": epochs, "consolidate": consolidate,
        "parent": original, "candidate_before": before,
        "baseline_after": baseline.to_snapshot(), "candidate_after": candidate.to_snapshot(),
        "initial_held_out_mse": initial_error, "initial_regression_mse": before_regression,
        "baseline": _measure(baseline, task, parent, before_regression),
        "candidate": _measure(candidate, task, AXMBrain.from_snapshot(before), before_regression),
        "execution_authorized": False, "automatic_adoption": False,
    }
    if parent.to_snapshot() != original:
        raise RuntimeError("parent changed during isolated experiment")
    return snapshot_payload(record)


def verify_experiment(record):
    body = verify_snapshot(record)
    if body.get("schema") != "axm.neural-experiment/v1" or body.get("adoption_state") != "archive-only":
        raise ValueError("unsupported experiment or adoption state")
    if body.get("execution_authorized") is not False or body.get("automatic_adoption") is not False:
        raise ValueError("experiment evidence cannot grant authority")
    task = ExperimentTask.from_dict(body["task"])
    if snapshot_payload(task.to_dict())["sha256"] != body["task_sha256"]:
        raise ValueError("task evidence fingerprint mismatch")
    mutation = Mutation.from_dict(body["mutation"])
    parent = AXMBrain.from_snapshot(body["parent"])
    task.validate_dimensions(parent.config)
    expected_before = descendant(body["parent"], mutation).to_snapshot()
    if expected_before != body["candidate_before"]:
        raise ValueError("candidate is not the declared descendant")
    if type(body["epochs"]) is not int or body["epochs"] < 1 or type(body["consolidate"]) is not bool:
        raise ValueError("invalid experiment schedule")
    # Recompute measured outcomes from the retained models and datasets.
    before_regression = _error(parent, task.regression)
    if body["initial_held_out_mse"] != _error(parent, task.held_out) or body["initial_regression_mse"] != before_regression:
        raise ValueError("parent measurement mismatch")
    for label, initial in (("baseline", parent), ("candidate", AXMBrain.from_snapshot(expected_before))):
        brain = AXMBrain.from_snapshot(body[label+"_after"])
        if brain.config != initial.config:
            raise ValueError("experiment changed undeclared configuration")
        expected = _measure(brain, task, initial, before_regression)
        updates = body["epochs"]*len(task.train)
        if expected["host_experiences"] != updates or brain.mode != "wake":
            raise ValueError("experiment state does not match the declared schedule")
        if body[label] != expected:
            raise ValueError("experiment measurements do not match retained state")
    return body


def reproduce_experiment(record, *, max_updates=100_000):
    body = verify_experiment(record)
    repeated = run_experiment(AXMBrain.from_snapshot(body["parent"]), ExperimentTask.from_dict(body["task"]),
                              Mutation.from_dict(body["mutation"]), epochs=body["epochs"],
                              consolidate=body["consolidate"], max_updates=max_updates)
    return repeated == record


class DiscoveryArchive:
    """Exact retries deduplicate; failed and regressive measurements remain evidence."""
    def __init__(self):
        self._records = []

    def append(self, record):
        verify_experiment(record)
        if any(r["sha256"] == record["sha256"] for r in self._records):
            return False
        self._records.append(deepcopy(record))
        return True

    @property
    def records(self):
        return deepcopy(self._records)

    def snapshot(self):
        previous, entries = None, []
        for record in self._records:
            entry = snapshot_payload({"previous": previous, "experiment": record})
            entries.append(entry)
            previous = entry["sha256"]
        return snapshot_payload({"schema": "axm.discovery-archive/v1", "entries": entries})

    @classmethod
    def restore(cls, snapshot):
        body = verify_snapshot(snapshot)
        if body.get("schema") != "axm.discovery-archive/v1" or not isinstance(body.get("entries"), list):
            raise ValueError("invalid discovery archive")
        archive, previous = cls(), None
        for entry in body["entries"]:
            data = verify_snapshot(entry)
            if data["previous"] != previous or not archive.append(data["experiment"]):
                raise ValueError("archive chain order or duplicate mismatch")
            previous = entry["sha256"]
        return archive

    def mechanism_index(self):
        index = {}
        for record in self._records:
            body = record["body"]
            # Index effective configuration, not cosmetic names, redundant no-op
            # fields or initialization seeds. Full lineage remains in each record.
            configuration = dict(body["candidate_before"]["body"]["config"])
            configuration.pop("seed")
            mechanism = snapshot_payload(configuration)["sha256"]
            index.setdefault(mechanism, []).append(record["sha256"])
        return index


class ExperimentSelector:
    """Small neural regressor over declared task features and validation evidence."""
    def __init__(self, mutations, feature_names, *, seed=19, hidden_size=8):
        self.mutations = tuple(mutations)
        self.feature_names = tuple(_text(s, "feature name") for s in feature_names)
        if not self.mutations or len({m.mechanism_id for m in self.mutations}) != len(self.mutations):
            raise ValueError("selector requires distinct mechanisms")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("selector requires distinct feature names")
        self.brain = AXMBrain(BrainConfig(len(self.feature_names), hidden_size, len(self.mutations),
                                         seed=seed, learning_rate=.03, replay_capacity=0))
        self.training_groups = set()
        self.training_receipts = []

    def fit(self, records, *, epochs=150, max_updates=100_000):
        if type(epochs) is not int or epochs < 1 or type(max_updates) is not int or max_updates < 1:
            raise ValueError("invalid selector training budget")
        table, groups, receipts = {}, set(), []
        mechanism_order = [m.mechanism_id for m in self.mutations]
        for record in records:
            body = verify_experiment(record)
            task = body["task"]
            mechanism = body["mutation"]["mechanism_id"]
            if mechanism not in mechanism_order or len(task["features"]) != len(self.feature_names):
                raise ValueError("incompatible selector evidence")
            # Ordering cannot depend on a hash containing held-out labels.
            key = (task["task_id"], body["parent"]["sha256"], body["epochs"], body["consolidate"])
            row = table.setdefault(key, {"features": task["features"], "outcomes": {},
                                         "task_sha256": body["task_sha256"]})
            if row["task_sha256"] != body["task_sha256"]:
                raise ValueError("conflicting task identity across operator comparisons")
            if mechanism in row["outcomes"]:
                raise ValueError("duplicate operator evidence for the same task")
            # Bounded monotone target; only validation MSE is used for learning.
            error = body["candidate"]["validation_mse"]
            row["outcomes"][mechanism] = 1.0 - 2.0*error/(1.0+error)
            groups.add(task["group_id"])
            receipts.append(record["sha256"])
        if not table or any(set(row["outcomes"]) != set(mechanism_order) for row in table.values()):
            raise ValueError("each training task needs a complete operator comparison")
        if len(table)*epochs > max_updates:
            raise ValueError("selector training exceeds the host update budget")
        # Fit on a clone; rejected evidence cannot partially train the selector.
        candidate = AXMBrain.from_snapshot(self.brain.to_snapshot())
        for _ in range(epochs):
            for key in sorted(table):
                row = table[key]
                _reset_transient(candidate)
                candidate.experience(Experience(row["features"], target=[row["outcomes"][m] for m in mechanism_order]), remember=False)
        self.brain = candidate
        self.training_groups.update(groups)
        self.training_receipts.extend(receipts)

    def choose(self, features, *, group_id, require_unseen=True):
        _text(group_id, "evaluation group")
        if type(require_unseen) is not bool:
            raise ValueError("require_unseen must be boolean")
        if require_unseen and group_id in self.training_groups:
            raise ValueError("evaluation group overlaps selector training")
        if len(features) != len(self.feature_names):
            raise ValueError("feature dimensions mismatch")
        values = [_number(v, "feature") for v in features]
        probe = AXMBrain.from_snapshot(self.brain.to_snapshot())
        _reset_transient(probe)
        scores = probe.predict(values, update_state=False)
        index = max(range(len(scores)), key=lambda j: (scores[j], -j))
        return {"mutation": self.mutations[index].to_dict(), "predicted_scores": scores,
                "group_id": group_id, "execution_authorized": False, "automatic_adoption": False}

    def snapshot(self):
        return snapshot_payload({"schema": "axm.experiment-selector/v1", "feature_names": list(self.feature_names),
                                 "mutations": [m.to_dict() for m in self.mutations], "brain": self.brain.to_snapshot(),
                                 "training_groups": sorted(self.training_groups), "training_receipts": list(self.training_receipts),
                                 "objective": "validation-mse-only/v1", "automatic_adoption": False})

    @classmethod
    def restore(cls, snapshot):
        body = verify_snapshot(snapshot)
        if body.get("schema") != "axm.experiment-selector/v1" or body.get("objective") != "validation-mse-only/v1" or body.get("automatic_adoption") is not False:
            raise ValueError("selector schema, objective or adoption boundary mismatch")
        brain = AXMBrain.from_snapshot(body["brain"])
        result = cls([Mutation.from_dict(m) for m in body["mutations"]], body["feature_names"],
                     seed=brain.config.seed, hidden_size=brain.config.hidden_size)
        if brain.config != result.brain.config:
            raise ValueError("selector brain does not match declared interface")
        groups, receipts = body["training_groups"], body["training_receipts"]
        if not isinstance(groups, list) or groups != sorted(set(groups)) or not isinstance(receipts, list):
            raise ValueError("invalid selector provenance")
        for group in groups: _text(group, "training group")
        if any(not isinstance(r, str) or len(r) != 64 or any(c not in '0123456789abcdef' for c in r) for r in receipts):
            raise ValueError("invalid training receipt fingerprint")
        result.brain, result.training_groups, result.training_receipts = brain, set(groups), list(receipts)
        return result
