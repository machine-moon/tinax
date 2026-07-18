"""Eager Chex value-assertion examples."""

import chex as _chex
import jax as _jax
import jax.numpy as _jnp


def log_with_value_assertions(values: _jax.Array) -> _jax.Array:
    """Check concrete values eagerly before computing their logarithms."""
    _chex.assert_tree_all_finite(values)
    _chex.assert_trees_all_equal(values > 0, _jnp.ones_like(values, dtype=_jnp.bool_), strict=True)
    return _jnp.log(values)
