"""Bounded nonfinite observation through injected host callbacks."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import core as _core


def observe_nonfinite(
    values: jax.Array,
    *,
    callback: Callable[[Any, Any], None],
    max_indices: int,
) -> jax.Array:
    """Report a count and bounded row-major flat indices while returning values unchanged.

    When non-finite values are present, schedules ``callback`` with the total count
    and up to ``max_indices`` row-major flat indices. The callback is an
    observational, asynchronous effect; ``values`` is always returned unchanged.

    Args:
        values: JAX array to scan for non-finite elements.
        callback: Host callable invoked as ``callback(count, indices)`` only when
            non-finite values exist. Runs asynchronously via ``jax.debug.callback``.
        max_indices: Maximum number of flat indices to report. Must be non-negative;
            unused slots are filled with ``-1``.

    Returns:
        The input ``values``, unchanged.

    Raises:
        TypeError: If ``callback`` is not callable, ``max_indices`` is not an integer
            (booleans are rejected), or ``values`` is not a JAX array.
        ValueError: If ``max_indices`` is negative.
    """
    if not callable(callback):
        raise TypeError("callback must be callable")
    if not isinstance(max_indices, int) or isinstance(max_indices, bool):
        raise TypeError("max_indices must be an integer and not a boolean")
    if max_indices < 0:
        raise ValueError("max_indices must be non-negative")
    if not isinstance(values, (jax.Array, _core.Tracer)):
        raise TypeError("values must be a JAX array")

    nonfinite = ~jnp.isfinite(values)
    count = jnp.count_nonzero(nonfinite)
    indices = jnp.nonzero(nonfinite.reshape(-1), size=max_indices, fill_value=-1)[0]

    def emit(_: None) -> None:
        jax.debug.callback(callback, count, indices)

    jax.lax.cond(count > 0, emit, lambda _: None, operand=None)
    return values
