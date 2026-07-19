"""Chex trace-budgeted jax.jit with validated static/donate names and an optional mesh context."""

from collections.abc import Callable as _Callable
from collections.abc import Sequence as _Sequence

import chex as _chex
import jax as _jax

import tinax.jit._validation as _validation


def bounded_jit[**P, R](
    function: _Callable[P, R],
    *,
    max_traces: int,
    static_argnames: _Sequence[str] = (),
    donate_argnames: _Sequence[str] = (),
    mesh: _jax.sharding.Mesh | None = None,
) -> _Callable[P, R]:
    """JIT a callable with an exact Chex trace budget, validated at wrap time, not first call.

    ``static_argnames``/``donate_argnames`` overlap or duplication is caught here; raw
    ``jax.jit`` only raises on first invocation, which can be arbitrarily deep in a loop. An
    optional ``mesh`` is entered around every call, removing the need for every call site to
    remember ``with jax.set_mesh(mesh):``.

    Args:
        function: Callable to JIT-compile.
        max_traces: Maximum number of distinct traces to permit. Must be non-negative.
        static_argnames: Keyword argument names to treat as static. Defaults to none.
        donate_argnames: Keyword argument names whose buffers may be donated. Defaults to
            none.
        mesh: Mesh to enter around every call, or ``None`` to use the ambient mesh.

    Returns:
        The trace-budgeted, JIT-compiled callable.

    Raises:
        TypeError: If ``function`` is not callable, ``max_traces`` is not an integer
            (booleans rejected), ``static_argnames``/``donate_argnames`` is not a sequence of
            strings, or ``mesh`` is not a ``jax.sharding.Mesh`` or ``None``.
        ValueError: If ``max_traces`` is negative, an argname entry is empty or duplicated,
            ``static_argnames`` and ``donate_argnames`` overlap, or ``mesh`` is empty.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    budget = _validation.validate_trace_budget(max_traces)
    static, donate = _validation.validate_argname_sets(static_argnames, donate_argnames)
    _validation.validate_mesh(mesh)

    traced = _chex.assert_max_traces(function, n=budget)
    jitted = _jax.jit(traced, static_argnames=static, donate_argnames=donate)
    if mesh is None:
        return jitted

    def with_mesh(*args: P.args, **kwargs: P.kwargs) -> R:
        with _jax.set_mesh(mesh):
            return jitted(*args, **kwargs)

    return with_mesh
