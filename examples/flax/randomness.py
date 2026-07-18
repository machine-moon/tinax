"""Named NNX RNG streams can be snapshotted and replayed exactly."""

import jax
from flax import nnx

from tinax.nnx import restore_graph, snapshot_graph


class NamedNormalSampler(nnx.Module):
    """Draw values from one explicitly named mutable RNG stream."""

    def __init__(self, sample_key: jax.Array) -> None:
        self.rngs = nnx.Rngs(sample=sample_key)

    def __call__(self, shape: tuple[int, ...]) -> jax.Array:
        """Consume one key from the sample stream and draw a normal array."""
        return jax.random.normal(self.rngs.sample(), shape)


def replay_next_sample(module: NamedNormalSampler, shape: tuple[int, ...]) -> tuple[jax.Array, jax.Array]:
    """Snapshot a named stream and prove its next sample is exactly replayable."""
    snapshot = snapshot_graph(module)
    sample = module(shape)
    restored = restore_graph(snapshot, copy=True)
    return sample, restored(shape)
