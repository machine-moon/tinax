"""jax.vmap fused with a trace-budgeted jax.jit and an optional mesh context."""

from collections.abc import Callable as _Callable
from collections.abc import Sequence as _Sequence
from typing import Any as _Any

import chex as _chex
import jax as _jax

import tinax.jit._validation as _validation


def batched_jit[R](
    function: _Callable[..., R],
    *,
    in_axes: int | None | _Sequence[_Any] = 0,
    out_axes: _Any = 0,
    axis_size: int | None = None,
    max_traces: int,
    mesh: _jax.sharding.Mesh | None = None,
) -> _Callable[..., R]:
    """Batch a callable with jax.vmap, then JIT it with an exact Chex trace budget.

    ``in_axes``/``out_axes`` are passed through unvalidated: raw ``jax.vmap`` already gives
    clear, specific errors for a malformed axis specification. ``axis_size`` is validated
    because raw ``jax.vmap`` silently accepts a boolean there, producing an unrelated,
    confusing error instead of a clear rejection. This is the auto-parallel, JAX-native
    replacement for legacy ``pmap``-style batching; combine with a Manual-axis
    ``tinax.parallel.shard_mapped`` for the manual-SPMD alternative.

    Args:
        function: Callable to batch and JIT-compile.
        in_axes: Input batch axis specification, forwarded to ``jax.vmap``. Defaults to 0.
        out_axes: Output batch axis specification, forwarded to ``jax.vmap``. Defaults to 0.
        axis_size: Explicit batch size, or ``None`` to infer it from the arguments. Must be
            positive if given.
        max_traces: Maximum number of distinct traces to permit. Must be non-negative.
        mesh: Mesh to enter around every call, or ``None`` to use the ambient mesh.

    Returns:
        The batched, trace-budgeted, JIT-compiled callable.

    Raises:
        TypeError: If ``function`` is not callable, ``axis_size`` is not an integer or
            ``None`` (booleans rejected), ``max_traces`` is not an integer (booleans
            rejected), or ``mesh`` is not a ``jax.sharding.Mesh`` or ``None``.
        ValueError: If ``axis_size`` is not positive, ``max_traces`` is negative, or ``mesh``
            is empty.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    _validation.validate_axis_size(axis_size)
    budget = _validation.validate_trace_budget(max_traces)
    _validation.validate_mesh(mesh)

    traced = _chex.assert_max_traces(function, n=budget)
    vmapped = _jax.vmap(traced, in_axes=in_axes, out_axes=out_axes, axis_size=axis_size)
    jitted = _jax.jit(vmapped)
    if mesh is None:
        return jitted

    def with_mesh(*args: object, **kwargs: object) -> R:
        with _jax.set_mesh(mesh):
            return jitted(*args, **kwargs)

    return with_mesh
