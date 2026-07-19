"""Shared validation for bounded compilation and batching."""

from collections.abc import Sequence as _Sequence

import jax as _jax


def validate_trace_budget(max_traces: int) -> int:
    if not isinstance(max_traces, int) or isinstance(max_traces, bool):
        raise TypeError("max_traces must be an integer")
    if max_traces < 0:
        raise ValueError("max_traces must be non-negative")
    return max_traces


def _validate_argname_sequence(names: _Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(names, _Sequence) or isinstance(names, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings")
    values = tuple(names)
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{name} must contain only strings")
    if any(not value for value in values):
        raise ValueError(f"{name} entries must be nonempty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def validate_argname_sets(
    static_argnames: _Sequence[str], donate_argnames: _Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    static = _validate_argname_sequence(static_argnames, "static_argnames")
    donate = _validate_argname_sequence(donate_argnames, "donate_argnames")
    overlap = set(static) & set(donate)
    if overlap:
        raise ValueError(f"static_argnames and donate_argnames must not overlap: {sorted(overlap)}")
    return static, donate


def validate_mesh(mesh: _jax.sharding.Mesh | None) -> None:
    if mesh is None:
        return
    if not isinstance(mesh, _jax.sharding.Mesh):
        raise TypeError("mesh must be a jax.sharding.Mesh or None")
    if mesh.empty:
        raise ValueError("mesh must not be empty")


def validate_axis_size(axis_size: int | None) -> None:
    if axis_size is None:
        return
    if not isinstance(axis_size, int) or isinstance(axis_size, bool):
        raise TypeError("axis_size must be an integer or None")
    if axis_size < 1:
        raise ValueError("axis_size must be positive")
