"""Typed PRNG keys make process- and step-local randomness explicit."""

import jax
import jax.numpy as jnp

from tinax.randomness import derive_process_step_key, split_key


def process_step_keys(seed: int, process_index: int, step: int) -> jax.Array:
    """Create two typed keys derived from an explicit seed, process, and step."""
    root_key = jax.random.key(seed)
    step_key = derive_process_step_key(
        root_key,
        process_index=jnp.asarray(process_index, dtype=jnp.uint32),
        step=jnp.asarray(step, dtype=jnp.uint32),
    )
    _, operation_keys = split_key(step_key, count=2)
    return operation_keys
