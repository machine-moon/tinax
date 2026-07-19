"""Validated sharding visualization."""

import jax as _jax


def visualize_array_sharding(array: _jax.Array) -> None:
    """Print an array's device sharding, rejecting a non-array with a clear TypeError.

    ``array`` must be concrete, not a value traced under ``jax.jit``: this prints real
    device placement and cannot run inside a trace.

    Args:
        array: Concrete JAX array whose sharding is printed.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array``.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    _jax.debug.visualize_array_sharding(array)
