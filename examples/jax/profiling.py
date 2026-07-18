"""Profiler traces need explicit array and effect completion before closing."""

from collections.abc import Callable
from os import PathLike

import jax

from tinax.diagnostics import profile_call


def profile_completed_array_call(
    log_dir: str | PathLike[str], function: Callable[[jax.Array], jax.Array], argument: jax.Array
) -> jax.Array:
    """Profile one array call and explicitly wait for its arrays and staged effects."""
    return profile_call(log_dir, function, argument)
