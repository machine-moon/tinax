"""Static-argument JIT caches are created explicitly rather than at module import."""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp


def make_static_reduction() -> Callable[..., jax.Array]:
    """Create a reusable reduction that specializes on its static operation."""
    def reduce_values(values: jax.Array, *, operation: Literal["sum", "mean"]) -> jax.Array:
        if operation == "sum":
            return jnp.sum(values)
        if operation == "mean":
            return jnp.mean(values)
        raise ValueError(f"unknown reduction: {operation}")

    return jax.jit(reduce_values, static_argnames="operation")
