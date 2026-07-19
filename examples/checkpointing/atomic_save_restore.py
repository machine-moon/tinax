"""Atomically save and restore NNX training state with tinax.checkpointing."""

import tempfile
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
import optax
from flax import nnx

from tinax.checkpointing import (
    abstract_restore_target,
    load_training_checkpoint,
    save_training_checkpoint,
)


def main() -> None:
    """Save model, optimizer, rng, auxiliary, and iterator state together, then restore them."""
    model = nnx.Linear(1, 1, rngs=nnx.Rngs(params=0))
    optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
    rng = jax.random.key(7)
    auxiliary = {"loss": jnp.asarray(0.5)}
    iterator = iter(grain.MapDataset.range(16).batch(4).to_iter_dataset())
    next(iterator)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint"
        save_training_checkpoint(path, 1, nnx.state(model), nnx.state(optimizer), rng, auxiliary, iterator).result()

        fresh_model = nnx.Linear(1, 1, rngs=nnx.Rngs(params=99))
        fresh_optimizer = nnx.Optimizer(fresh_model, optax.adam(0.01), wrt=nnx.Param)
        restore_iterator = iter(grain.MapDataset.range(16).batch(4).to_iter_dataset())
        restored = load_training_checkpoint(
            path,
            abstract_restore_target(nnx.state(fresh_model)),
            abstract_restore_target(nnx.state(fresh_optimizer)),
            abstract_restore_target(rng),
            abstract_restore_target(auxiliary),
            restore_iterator,
        )
        print(f"restored_step={restored.step} next_batch={jax.device_get(next(restored.iterator)).tolist()}")


if __name__ == "__main__":
    main()
