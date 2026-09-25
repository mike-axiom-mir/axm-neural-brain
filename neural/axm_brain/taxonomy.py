from __future__ import annotations

ROOT_DIRECTIONS = ("CREATE", "USE", "PLAY", "LEARN", "DISCOVER")


def normalize_directions(values) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip().upper()
        if value not in ROOT_DIRECTIONS:
            raise ValueError(
                f"unknown AXM direction {raw!r}; expected one of {ROOT_DIRECTIONS}"
            )
        seen.add(value)
    return tuple(direction for direction in ROOT_DIRECTIONS if direction in seen)
