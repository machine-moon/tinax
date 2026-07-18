"""Profiler traces with explicit asynchronous completion."""

from collections.abc import Callable
from os import PathLike

import jax


def profile_call[**P, R](
    log_dir: str | PathLike[str],
    function: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Profile one call and keep the trace open until result arrays and staged effects complete.

    Args:
        log_dir: Directory the profiler trace is written to.
        function: Callable to profile. Invoked once as ``function(*args, **kwargs)``.
        *args: Positional arguments forwarded to ``function``.
        **kwargs: Keyword arguments forwarded to ``function``.

    Returns:
        The value returned by ``function``, after its result arrays are ready and
        staged effects have completed inside the trace.

    Raises:
        TypeError: If ``function`` is not callable.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    with jax.profiler.trace(log_dir):
        result = function(*args, **kwargs)
        jax.block_until_ready(result)
        jax.effects_barrier()
    return result
