"""Checkify keeps device error data separate from an explicit host throw."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify


def positive_log(values: jax.Array) -> jax.Array:
    """Check the logarithm domain inside a checkify transformation."""
    checkify.check(jnp.all(values > 0), "values must be positive")
    return jnp.log(values)


def checked_positive_log(values: jax.Array) -> tuple[checkify.Error, jax.Array]:
    """Return error data and logarithms without throwing on the host."""
    return checkify.checkify(positive_log)(values)
