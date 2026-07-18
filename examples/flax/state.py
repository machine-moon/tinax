"""GraphState recipes for explicit functional state boundaries."""

import jax
import jax.numpy as jnp
from flax import nnx

from tinax.nnx import clone_graph


class RunningTotal(nnx.Module):
    """Accumulate scalar values in mutable NNX state."""

    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.zeros((), dtype=jnp.float32))

    def __call__(self, increment: jax.Array) -> jax.Array:
        """Add one scalar to the running total."""
        self.total[...] = self.total[...] + increment
        return self.total[...]


@jax.jit
def advance_graph_state(
    state: nnx.GraphState[RunningTotal], increment: jax.Array
) -> tuple[nnx.GraphState[RunningTotal], jax.Array]:
    """Run one pure step by merging and unpacking a GraphState."""
    model = nnx.merge(state)
    total = model(increment)
    return nnx.unpack(model, graph=True), total


class SharedTotals(nnx.Module):
    """Hold two names for one stateful submodule."""

    def __init__(self) -> None:
        shared = RunningTotal()
        self.primary = shared
        self.alias = shared


def clone_shared_totals(model: SharedTotals) -> SharedTotals:
    """Copy variables across split and merge while preserving graph aliases."""
    return clone_graph(model)
