"""Resume NNX training exactly from one atomic Orbax V1 checkpoint."""

from dataclasses import dataclass
from os import PathLike
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast

import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from tinax.checkpointing import (
    abstract_restore_target,
    load_training_checkpoint,
    save_training_checkpoint,
)


class AuxiliaryState(TypedDict):
    """Track non-model state needed to resume the example."""

    examples_seen: int
    last_loss: jax.Array


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """Capture the resume-critical state after a training run."""

    model: nnx.State
    optimizer: nnx.State
    rng: jax.Array
    auxiliary: AuxiliaryState
    next_batch: np.ndarray[Any, Any]


@dataclass(frozen=True, slots=True)
class AtomicResumeResult:
    """Hold uninterrupted and resumed outcomes for exact comparison."""

    checkpoint_step: int
    completed_step: int
    uninterrupted: TrainingSnapshot
    resumed: TrainingSnapshot


def _new_training_state(
    *, model_seed: int, transformation: optax.GradientTransformation
) -> tuple[nnx.Linear, nnx.Optimizer]:
    model = nnx.Linear(1, 1, rngs=nnx.Rngs(params=model_seed))
    optimizer = nnx.Optimizer(model, transformation, wrt=nnx.Param)
    return model, optimizer


def _new_iterator(
    dataset: grain.MapDataset[int],
    *,
    data_seed: int,
    shuffle: bool,
    batch_size: int,
    drop_remainder: bool,
) -> grain.DatasetIterator[Any]:
    ordered = dataset.seed(data_seed)
    if shuffle:
        ordered = ordered.shuffle()
    return iter(ordered.batch(batch_size, drop_remainder=drop_remainder))


def _train_step(
    model: nnx.Linear,
    optimizer: nnx.Optimizer,
    rng: jax.Array,
    batch: np.ndarray[Any, Any],
    *,
    input_scale: float,
    target_slope: float,
    target_intercept: float,
    noise_scale: float,
) -> tuple[jax.Array, jax.Array]:
    rng, noise_key = jax.random.split(rng)
    inputs = jnp.asarray(batch, dtype=jnp.float32).reshape((-1, 1)) / input_scale
    targets = target_slope * inputs + target_intercept
    noisy_inputs = inputs + noise_scale * jax.random.normal(noise_key, inputs.shape)

    def loss_fn(current_model: nnx.Linear) -> jax.Array:
        return jnp.mean(jnp.square(current_model(noisy_inputs) - targets))

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return rng, loss


def _run_steps(
    model: nnx.Linear,
    optimizer: nnx.Optimizer,
    rng: jax.Array,
    auxiliary: AuxiliaryState,
    iterator: grain.DatasetIterator[Any],
    steps: int,
    *,
    input_scale: float,
    target_slope: float,
    target_intercept: float,
    noise_scale: float,
) -> tuple[jax.Array, AuxiliaryState]:
    for _ in range(steps):
        batch = np.asarray(next(iterator))
        rng, loss = _train_step(
            model,
            optimizer,
            rng,
            batch,
            input_scale=input_scale,
            target_slope=target_slope,
            target_intercept=target_intercept,
            noise_scale=noise_scale,
        )
        auxiliary = {
            "examples_seen": auxiliary["examples_seen"] + int(batch.shape[0]),
            "last_loss": loss,
        }
    return rng, auxiliary


def _snapshot(
    model: nnx.Linear,
    optimizer: nnx.Optimizer,
    rng: jax.Array,
    auxiliary: AuxiliaryState,
    iterator: grain.DatasetIterator[Any],
) -> TrainingSnapshot:
    return TrainingSnapshot(
        model=nnx.state(model),
        optimizer=nnx.state(optimizer),
        rng=rng,
        auxiliary=auxiliary,
        next_batch=np.asarray(next(iterator)),
    )


def run_atomic_resume(
    path: str | PathLike[str],
    *,
    dataset: grain.MapDataset[int],
    batch_size: int,
    data_seed: int,
    shuffle: bool,
    drop_remainder: bool,
    checkpoint_step: int,
    completed_step: int,
    model_seed: int,
    transformation: optax.GradientTransformation,
    initial_rng: jax.Array,
    initial_auxiliary: AuxiliaryState,
    input_scale: float,
    target_slope: float,
    target_intercept: float,
    noise_scale: float,
) -> AtomicResumeResult:
    """Compare uninterrupted NNX training with a complete V1 save and restore."""
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise TypeError("checkpoint_step must be an integer and not a boolean")
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise TypeError("completed_step must be an integer and not a boolean")
    if checkpoint_step < 0 or completed_step < 0:
        raise ValueError("checkpoint_step and completed_step must be non-negative")
    if checkpoint_step > completed_step:
        raise ValueError("checkpoint_step must not exceed completed_step")
    iterator_arguments = {
        "data_seed": data_seed,
        "shuffle": shuffle,
        "batch_size": batch_size,
        "drop_remainder": drop_remainder,
    }
    uninterrupted_iterator = _new_iterator(dataset, **iterator_arguments)
    checkpoint_iterator = _new_iterator(dataset, **iterator_arguments)
    restore_iterator = _new_iterator(dataset, **iterator_arguments)
    try:
        uninterrupted_model, uninterrupted_optimizer = _new_training_state(
            model_seed=model_seed, transformation=transformation
        )
        uninterrupted_rng, uninterrupted_auxiliary = _run_steps(
            uninterrupted_model,
            uninterrupted_optimizer,
            initial_rng,
            initial_auxiliary,
            uninterrupted_iterator,
            completed_step,
            input_scale=input_scale,
            target_slope=target_slope,
            target_intercept=target_intercept,
            noise_scale=noise_scale,
        )
        uninterrupted = _snapshot(
            uninterrupted_model,
            uninterrupted_optimizer,
            uninterrupted_rng,
            uninterrupted_auxiliary,
            uninterrupted_iterator,
        )

        checkpoint_model, checkpoint_optimizer = _new_training_state(
            model_seed=model_seed, transformation=transformation
        )
        checkpoint_rng, checkpoint_auxiliary = _run_steps(
            checkpoint_model,
            checkpoint_optimizer,
            initial_rng,
            initial_auxiliary,
            checkpoint_iterator,
            checkpoint_step,
            input_scale=input_scale,
            target_slope=target_slope,
            target_intercept=target_intercept,
            noise_scale=noise_scale,
        )
        response = save_training_checkpoint(
            path,
            checkpoint_step,
            nnx.state(checkpoint_model),
            nnx.state(checkpoint_optimizer),
            checkpoint_rng,
            checkpoint_auxiliary,
            checkpoint_iterator,
        )
        response.result()

        resumed_model, resumed_optimizer = _new_training_state(
            model_seed=model_seed, transformation=transformation
        )
        restored = load_training_checkpoint(
            path,
            abstract_restore_target(nnx.state(resumed_model)),
            abstract_restore_target(nnx.state(resumed_optimizer)),
            abstract_restore_target(initial_rng),
            abstract_restore_target(initial_auxiliary),
            restore_iterator,
        )
        nnx.update(resumed_model, restored.model)
        nnx.update(resumed_optimizer, restored.optimizer)
        resumed_rng, resumed_auxiliary = _run_steps(
            resumed_model,
            resumed_optimizer,
            restored.rng,
            cast(AuxiliaryState, restored.auxiliary),
            restored.iterator,
            completed_step - restored.step,
            input_scale=input_scale,
            target_slope=target_slope,
            target_intercept=target_intercept,
            noise_scale=noise_scale,
        )
        resumed = _snapshot(
            resumed_model,
            resumed_optimizer,
            resumed_rng,
            resumed_auxiliary,
            restored.iterator,
        )
        return AtomicResumeResult(restored.step, completed_step, uninterrupted, resumed)
    finally:
        uninterrupted_iterator.close()
        checkpoint_iterator.close()
        restore_iterator.close()


def main() -> None:
    """Run the atomic resume recipe in a temporary directory."""
    with TemporaryDirectory() as directory:
        result = run_atomic_resume(
            f"{directory}/resume",
            dataset=grain.MapDataset.range(64),
            batch_size=4,
            data_seed=2027,
            shuffle=True,
            drop_remainder=True,
            checkpoint_step=3,
            completed_step=8,
            model_seed=17,
            transformation=optax.adam(learning_rate=0.01),
            initial_rng=jax.random.key(29),
            initial_auxiliary={
                "examples_seen": 0,
                "last_loss": jnp.asarray(0.0, dtype=jnp.float32),
            },
            input_scale=64.0,
            target_slope=2.0,
            target_intercept=1.0,
            noise_scale=0.01,
        )
    print(f"resumed step {result.checkpoint_step} through step {result.completed_step}")


if __name__ == "__main__":
    main()
