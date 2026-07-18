"""Lax control flow keeps data-dependent branches and loops transformable."""

import jax
import jax.numpy as jnp


def normalize_or_zero(values: jax.Array) -> jax.Array:
    """Normalize nonzero vectors with a staged conditional."""
    norm = jnp.linalg.norm(values)
    return jax.lax.cond(norm > 0, lambda operand: operand / norm, jnp.zeros_like, values)


def running_decay(values: jax.Array, decay: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute recurrent totals with a staged scan."""
    def step(total: jax.Array, value: jax.Array) -> tuple[jax.Array, jax.Array]:
        next_total = decay * total + value
        return next_total, next_total

    initial = jnp.zeros((), dtype=values.dtype)
    return jax.lax.scan(step, initial, values)
