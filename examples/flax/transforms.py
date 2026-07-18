"""Explicit tree-mode and graph-mode NNX transform recipes."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx


class TreeModeDropout(nnx.Module):
    """Project with tree-shaped state and a call-time dropout key."""

    def __init__(self, features: int, params_key: jax.Array) -> None:
        self.projection = nnx.Linear(
            features,
            features,
            use_bias=False,
            rngs=nnx.Rngs(params=params_key),
        )
        self.dropout = nnx.Dropout(0.5)
        self.calls = nnx.BatchStat(jnp.zeros((), dtype=jnp.int32))

    def __call__(self, inputs: jax.Array, *, dropout_key: jax.Array) -> jax.Array:
        """Apply the model and mutate only its call counter."""
        self.calls[...] = self.calls[...] + 1
        return self.dropout(self.projection(inputs), rngs=dropout_key)


class SharedGraphDropout(nnx.Module):
    """Reuse one projection through two graph references with a call-time key."""

    def __init__(self, features: int, params_key: jax.Array) -> None:
        shared = nnx.Linear(
            features,
            features,
            use_bias=False,
            rngs=nnx.Rngs(params=params_key),
        )
        self.left = shared
        self.right = shared
        self.dropout = nnx.Dropout(0.5)
        self.calls = nnx.BatchStat(jnp.zeros((), dtype=jnp.int32))

    def __call__(self, inputs: jax.Array, *, dropout_key: jax.Array) -> jax.Array:
        """Apply both aliases and mutate only the shared graph's call counter."""
        self.calls[...] = self.calls[...] + 1
        hidden = self.left(inputs) + self.right(inputs)
        return self.dropout(hidden, rngs=dropout_key)


@nnx.jit(graph=False, graph_updates=False)
def tree_mode_call(model: TreeModeDropout, inputs: jax.Array, dropout_key: jax.Array) -> jax.Array:
    """Use the lower-overhead tree transform for an alias-free model."""
    return model(inputs, dropout_key=dropout_key)


@nnx.jit(graph=True, graph_updates=True)
def graph_mode_call(model: SharedGraphDropout, inputs: jax.Array, dropout_key: jax.Array) -> jax.Array:
    """Use graph mode when references and mutations must be preserved."""
    return model(inputs, dropout_key=dropout_key)


def bind_graph_mode_call(
    model: SharedGraphDropout,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Cache repeated graph traversal while leaving data and RNG keys dynamic."""
    with nnx.set_graph_mode(True), nnx.set_graph_updates(True):
        return nnx.cached_partial(graph_mode_call, model)
