import sys
from pathlib import Path
from typing import Any, cast

import grain
import jax
import numpy as np
import optax
import pytest

from tinax.checkpointing import (
    abstract_restore_target,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _new_iterator() -> grain.DatasetIterator[Any]:
    dataset = grain.MapDataset.range(128).seed(314159).shuffle().batch(4, drop_remainder=True)
    return iter(dataset)


def _initial_state(
    transformation: optax.GradientTransformation,
) -> tuple[dict[str, jax.Array], Any, jax.Array, dict[str, Any]]:
    model = {"weight": jax.numpy.array([0.5, -0.25], dtype=jax.numpy.float32)}
    optimizer = transformation.init(model)
    rng = jax.random.key(2718)
    auxiliary = {"schedule": np.array([3, 5, 8], dtype=np.int16), "seen": 0}
    return model, optimizer, rng, auxiliary


def _run_steps(
    transformation: optax.GradientTransformation,
    model: dict[str, jax.Array],
    optimizer: Any,
    rng: jax.Array,
    auxiliary: dict[str, Any],
    iterator: grain.DatasetIterator[Any],
    count: int,
) -> tuple[dict[str, jax.Array], Any, jax.Array, dict[str, Any]]:
    for _ in range(count):
        batch = jax.numpy.asarray(next(iterator), dtype=jax.numpy.float32)
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.uniform(noise_key, model["weight"].shape, minval=-0.5, maxval=0.5)
        mean = jax.numpy.mean(batch)
        gradients = {"weight": jax.numpy.array([mean, -mean]) + noise}
        updates, optimizer = transformation.update(gradients, optimizer, model)
        model = cast(dict[str, jax.Array], optax.apply_updates(model, updates))
        auxiliary = {"schedule": auxiliary["schedule"], "seen": auxiliary["seen"] + batch.size}
    return model, optimizer, rng, auxiliary


def _assert_tree_exact(actual: Any, expected: Any) -> None:
    actual_leaves, actual_structure = jax.tree.flatten(actual)
    expected_leaves, expected_structure = jax.tree.flatten(expected)
    assert actual_structure == expected_structure
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))


@pytest.mark.xfail(
    sys.platform == "win32",
    reason=(
        "orbax-checkpoint has no uvloop build on Windows and falls back to nest_asyncio, "
        "which does not support CPython 3.14's asyncio.Runner.close() shutdown path; "
        "upstream gap, not a tinax defect"
    ),
    raises=RuntimeError,
    strict=False,
)
def test_training_checkpoint_resumes_exactly(tmp_path: Path) -> None:
    total_steps = 7
    checkpoint_step = 3
    transformation = optax.adam(0.005)

    uninterrupted_iterator = _new_iterator()
    resumed_source_iterator = _new_iterator()
    resumed_iterator = _new_iterator()
    try:
        uninterrupted = _run_steps(
            transformation,
            *_initial_state(transformation),
            uninterrupted_iterator,
            total_steps,
        )
        uninterrupted_next_batch = np.asarray(next(uninterrupted_iterator))

        partial = _run_steps(
            transformation,
            *_initial_state(transformation),
            resumed_source_iterator,
            checkpoint_step,
        )
        response = save_training_checkpoint(
            tmp_path / "step-3",
            checkpoint_step,
            *partial,
            resumed_source_iterator,
        )
        assert response.result(timeout=30) is None

        restored = load_training_checkpoint(
            tmp_path / "step-3",
            abstract_restore_target(partial[0]),
            abstract_restore_target(partial[1]),
            abstract_restore_target(partial[2]),
            abstract_restore_target(partial[3]),
            resumed_iterator,
        )
        assert restored.step == checkpoint_step
        assert restored.iterator is resumed_iterator

        resumed = _run_steps(
            transformation,
            restored.model,
            restored.optimizer,
            restored.rng,
            restored.auxiliary,
            restored.iterator,
            total_steps - checkpoint_step,
        )
        resumed_next_batch = np.asarray(next(restored.iterator))

        _assert_tree_exact(resumed[0], uninterrupted[0])
        _assert_tree_exact(resumed[1], uninterrupted[1])
        np.testing.assert_array_equal(jax.random.key_data(resumed[2]), jax.random.key_data(uninterrupted[2]))
        _assert_tree_exact(resumed[3], uninterrupted[3])
        np.testing.assert_array_equal(resumed_next_batch, uninterrupted_next_batch)
    finally:
        uninterrupted_iterator.close()
        resumed_source_iterator.close()
        resumed_iterator.close()
