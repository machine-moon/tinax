"""Vmap adds a batch axis while sharing model parameters."""

import jax


def affine(weight: jax.Array, bias: jax.Array, features: jax.Array) -> jax.Array:
    """Apply one affine prediction to one feature row."""
    return features @ weight + bias


def batched_affine(weight: jax.Array, bias: jax.Array, features: jax.Array) -> jax.Array:
    """Map affine prediction over rows without mapping the parameters."""
    return jax.vmap(affine, in_axes=(None, None, 0))(weight, bias, features)
