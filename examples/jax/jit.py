"""Reusable JIT caches are created explicitly rather than at module import."""

from collections.abc import Callable

import jax


def make_cached_polynomial() -> Callable[[jax.Array], jax.Array]:
    """Create one jitted polynomial callable with a caller-owned cache."""
    def polynomial(values: jax.Array) -> jax.Array:
        return values * values + 2.0 * values + 1.0

    return jax.jit(polynomial)
