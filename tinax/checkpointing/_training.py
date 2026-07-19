"""Atomic training checkpoint composition for Orbax V1."""

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any

import grain
from orbax.checkpoint import v1 as ocp

from ._names import TrainingCheckpointNames
from ._operations import load_checkpointables, save_checkpointables


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """A complete restored training state from one atomic checkpoint.

    Attributes:
        step: Non-negative training step the checkpoint was written at.
        model: Restored model state.
        optimizer: Restored optimizer state.
        rng: Restored RNG state.
        auxiliary: Restored user auxiliary state.
        iterator: Restored Grain input iterator.

    Raises:
        TypeError: If ``step`` is not an integer (booleans rejected) or ``iterator`` is
            not a ``grain.DatasetIterator``.
        ValueError: If ``step`` is negative.
    """

    step: int
    model: Any
    optimizer: Any
    rng: Any
    auxiliary: Any
    iterator: grain.DatasetIterator[Any]

    def __post_init__(self) -> None:
        _validate_step(self.step)
        _validate_iterator(self.iterator)


def save_training_checkpoint(
    path: str | PathLike[str],
    step: int,
    model: Any,
    optimizer: Any,
    rng: Any,
    auxiliary: Any,
    iterator: grain.DatasetIterator[Any],
    *,
    names: TrainingCheckpointNames | None = None,
) -> ocp.AsyncResponse[None]:
    """Start one atomic training save on every process's main thread and return its completion response.

    Args:
        path: Destination checkpoint directory.
        step: Non-negative training step.
        model: Model state to save.
        optimizer: Optimizer state to save.
        rng: RNG state to save.
        auxiliary: User auxiliary state saved alongside ``step``.
        iterator: Grain input iterator whose state is saved.
        names: Optional override for the five checkpointable names. Defaults to
            ``TrainingCheckpointNames()``.

    Returns:
        An Orbax ``AsyncResponse``; call ``.result()`` to await completion.

    Raises:
        TypeError: If ``step`` is not an integer (booleans rejected), ``iterator`` is
            not a ``grain.DatasetIterator``, or ``names`` is not a
            ``TrainingCheckpointNames``.
        ValueError: If ``step`` is negative.
    """
    _validate_step(step)
    _validate_iterator(iterator)
    resolved_names = _resolve_training_names(names)
    return save_checkpointables(
        path,
        {
            resolved_names.model: model,
            resolved_names.optimizer: optimizer,
            resolved_names.rng: rng,
            resolved_names.auxiliary: {"step": step, "state": auxiliary},
            resolved_names.iterator: iterator,
        },
    )


def load_training_checkpoint(
    path: str | PathLike[str],
    model_target: Any,
    optimizer_target: Any,
    rng_target: Any,
    auxiliary_target: Any,
    iterator: grain.DatasetIterator[Any],
    *,
    names: TrainingCheckpointNames | None = None,
) -> TrainingCheckpoint:
    """Preflight value targets before restoring the stateful iterator from the same atomic checkpoint.

    Args:
        path: Source checkpoint directory.
        model_target: Restore target for model state.
        optimizer_target: Restore target for optimizer state.
        rng_target: Restore target for RNG state.
        auxiliary_target: Restore target for user auxiliary state.
        iterator: Grain iterator to restore in place.
        names: Optional override for the five checkpointable names. Defaults to
            ``TrainingCheckpointNames()``.

    Returns:
        A ``TrainingCheckpoint`` with the restored step, model, optimizer, rng,
        auxiliary state, and iterator.

    Raises:
        TypeError: If ``iterator`` is not a ``grain.DatasetIterator``, ``names`` is not a
            ``TrainingCheckpointNames``, or the auxiliary checkpointable does not restore
            as a mapping.
        ValueError: If the auxiliary checkpointable is missing ``step`` or ``state``, or
            the restored step is negative.
    """
    _validate_iterator(iterator)
    resolved_names = _resolve_training_names(names)
    restored = load_checkpointables(
        path,
        {
            resolved_names.model: model_target,
            resolved_names.optimizer: optimizer_target,
            resolved_names.rng: rng_target,
            resolved_names.auxiliary: {"step": 0, "state": auxiliary_target},
        },
    )
    auxiliary = restored[resolved_names.auxiliary]
    if not isinstance(auxiliary, Mapping):
        raise TypeError("the auxiliary checkpointable must restore as a mapping")
    if "step" not in auxiliary or "state" not in auxiliary:
        raise ValueError("the auxiliary checkpointable is missing step or state")
    _validate_step(auxiliary["step"])
    restored_iterator = load_checkpointables(path, {resolved_names.iterator: iterator})[resolved_names.iterator]
    return TrainingCheckpoint(
        step=auxiliary["step"],
        model=restored[resolved_names.model],
        optimizer=restored[resolved_names.optimizer],
        rng=restored[resolved_names.rng],
        auxiliary=auxiliary["state"],
        iterator=restored_iterator,
    )


def _resolve_training_names(names: object | None) -> TrainingCheckpointNames:
    if names is None:
        return TrainingCheckpointNames()
    if not isinstance(names, TrainingCheckpointNames):
        raise TypeError("names must be a TrainingCheckpointNames instance")
    return names


def _validate_step(step: object) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer and not a boolean")
    if step < 0:
        raise ValueError("step must be non-negative")


def _validate_iterator(iterator: object) -> None:
    if not isinstance(iterator, grain.DatasetIterator):
        raise TypeError("iterator must be a Grain DatasetIterator")
