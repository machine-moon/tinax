"""Chex trace-count instrumentation for tests."""

from collections.abc import Callable as _Callable
from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager

import chex as _chex
import jax as _jax


@_contextmanager
def isolated_trace_counter() -> _Iterator[None]:
    """Clear Chex's process-global trace counter before and after test instrumentation."""
    _chex.clear_trace_counter()
    try:
        yield
    finally:
        _chex.clear_trace_counter()


def trace_limited_jit[**P, R](
    function: _Callable[P, R],
    *,
    max_traces: int,
) -> _Callable[P, R]:
    """JIT a callable with an exact Chex trace budget for test instrumentation."""
    if not isinstance(max_traces, int) or isinstance(max_traces, bool):
        raise TypeError("max_traces must be an integer")
    if max_traces < 0:
        raise ValueError("max_traces must be nonnegative")
    traced = _chex.assert_max_traces(function, n=max_traces)
    return _jax.jit(traced)
