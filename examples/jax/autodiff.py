"""Autodiff transforms pure scalar-loss functions."""

import jax
import jax.numpy as jnp


def mean_squared_error(weight: jax.Array, features: jax.Array, targets: jax.Array) -> jax.Array:
    """Return a scalar loss suitable for differentiation."""
    return jnp.mean((features * weight - targets) ** 2)


def loss_gradient(weight: jax.Array, features: jax.Array, targets: jax.Array) -> jax.Array:
    """Differentiate the loss with respect to its first argument."""
    return jax.grad(mean_squared_error)(weight, features, targets)


def loss_and_gradient(
    weight: jax.Array, features: jax.Array, targets: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Compute a loss and its gradient in one transformed evaluation."""
    return jax.value_and_grad(mean_squared_error)(weight, features, targets)
