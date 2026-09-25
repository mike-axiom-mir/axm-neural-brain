from __future__ import annotations

from math import sqrt

from .core import AXMBrain
from .state import verify_snapshot
from .taxonomy import ROOT_DIRECTIONS


def _flatten(matrix_or_vector) -> list[float]:
    if not matrix_or_vector:
        return []
    if isinstance(matrix_or_vector[0], list):
        return [float(value) for row in matrix_or_vector for value in row]
    return [float(value) for value in matrix_or_vector]


def _parameter_vector(state: dict) -> list[float]:
    values: list[float] = []
    for name in ("w_in", "w_rec", "w_out", "b_hidden", "b_out"):
        values.extend(_flatten(state[name]))
    return values


def _l2(values: list[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _mean_abs(values: list[float]) -> float:
    return 0.0 if not values else sum(abs(value) for value in values) / len(values)


def _matrix_stats(values) -> dict:
    flat = _flatten(values)
    nonzero = sum(1 for value in flat if value != 0.0)
    count = len(flat)
    return {
        "count": count,
        "nonzero": nonzero,
        "sparsity": 0.0 if count == 0 else 1.0 - (nonzero / count),
        "mean_abs": _mean_abs(flat),
        "max_abs": max((abs(value) for value in flat), default=0.0),
        "l2": _l2(flat),
    }


class StateAnalyzer:
    """Read-only mechanical state microscope for an AXM brain.

    Reports measured state and deltas. It intentionally does not infer semantic
    concepts or claim that parameter movement equals understanding.
    """

    SCHEMA = "axm-neural-state-analysis/v0.1"

    @classmethod
    def capture(cls, brain: AXMBrain) -> dict:
        matrices = {
            "input": _matrix_stats(brain.w_in),
            "recurrent": _matrix_stats(brain.w_rec),
            "output": _matrix_stats(brain.w_out),
            "hidden_bias": _matrix_stats(brain.b_hidden),
            "output_bias": _matrix_stats(brain.b_out),
        }
        all_parameters = []
        for values in (
            brain.w_in,
            brain.w_rec,
            brain.w_out,
            brain.b_hidden,
            brain.b_out,
        ):
            all_parameters.extend(_flatten(values))
        nonzero = sum(1 for value in all_parameters if value != 0.0)
        return {
            "schema": cls.SCHEMA,
            "phase": brain.mode,
            "wake_cycle": brain.cycle,
            "forward_steps": brain.steps,
            "host_experiences": brain.host_experience_count,
            "sleep_count": brain.sleep_count,
            "learning_updates": {
                "supervised": brain.supervised_update_count,
                "reward": brain.reward_update_count,
                "sleep_replay_events": brain.sleep_replay_event_count,
                "pruned_weights_total": brain.total_pruned_weights,
            },
            "experience_directions": dict(brain.direction_experience_counts),
            "untagged_experiences": brain.untagged_experience_count,
            "replay_buffered": len(brain.replay),
            "reward_baseline": brain.reward_baseline,
            "parameter_summary": {
                "count": len(all_parameters),
                "nonzero": nonzero,
                "sparsity": 0.0 if not all_parameters else 1.0 - (nonzero / len(all_parameters)),
                "mean_abs": _mean_abs(all_parameters),
                "max_abs": max((abs(value) for value in all_parameters), default=0.0),
                "l2": _l2(all_parameters),
            },
            "matrices": matrices,
            "state_norms": {
                "hidden_l2": _l2([float(v) for v in brain.hidden]),
                "output_l2": _l2([float(v) for v in brain.last_output]),
            },
            "truth_boundary": "mechanical-state-report-not-semantic-understanding",
        }

    @classmethod
    def compare_snapshots(cls, before_snapshot: dict, after_snapshot: dict) -> dict:
        before = verify_snapshot(before_snapshot)
        after = verify_snapshot(after_snapshot)
        if before.get("schema") != after.get("schema"):
            raise ValueError("cannot compare different brain schemas")
        if before.get("config") != after.get("config"):
            raise ValueError("cannot compare snapshots with different brain configs")
        before_state = before["state"]
        after_state = after["state"]
        a = _parameter_vector(before_state)
        b = _parameter_vector(after_state)
        if len(a) != len(b):
            raise ValueError("parameter shape changed")
        delta = [new - old for old, new in zip(a, b)]
        changed = sum(1 for value in delta if abs(value) > 1e-12)
        sign_flips = sum(
            1
            for old, new in zip(a, b)
            if old != 0.0 and new != 0.0 and ((old < 0) != (new < 0))
        )
        norm_a = _l2(a)
        norm_b = _l2(b)
        if norm_a == 0.0 and norm_b == 0.0:
            cosine = 1.0
        elif norm_a == 0.0 or norm_b == 0.0:
            cosine = 0.0
        else:
            cosine = sum(old * new for old, new in zip(a, b)) / (norm_a * norm_b)
        before_directions = before_state.get("direction_experience_counts", {})
        after_directions = after_state.get("direction_experience_counts", {})
        return {
            "schema": "axm-neural-state-delta/v0.1",
            "parameters": len(delta),
            "changed_parameters": changed,
            "changed_fraction": 0.0 if not delta else changed / len(delta),
            "mean_abs_delta": _mean_abs(delta),
            "max_abs_delta": max((abs(value) for value in delta), default=0.0),
            "l2_delta": _l2(delta),
            "cosine_similarity": cosine,
            "sign_flips": sign_flips,
            "host_experiences_delta": int(after_state.get("host_experience_count", 0))
            - int(before_state.get("host_experience_count", 0)),
            "sleep_count_delta": int(after_state.get("sleep_count", 0))
            - int(before_state.get("sleep_count", 0)),
            "direction_experience_delta": {
                direction: int(after_directions.get(direction, 0))
                - int(before_directions.get(direction, 0))
                for direction in ROOT_DIRECTIONS
            },
            "truth_boundary": "parameter-drift-is-not-by-itself-semantic-understanding",
        }
