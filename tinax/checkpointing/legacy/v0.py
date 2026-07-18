"""Explicit legacy Orbax V0 checkpoint readers and writers."""

from os import PathLike
from typing import Any, cast

import grain
import orbax.checkpoint as ocp

from .._names import _validated_checkpoint_path


def save_legacy_v0_pytree(path: str | PathLike[str], state: Any) -> None:
    """Write one immutable legacy V0 PyTree checkpoint synchronously on every process's main thread.

    Args:
        path: Destination checkpoint directory. Must not use Orbax's reserved temporary
            suffix.
        state: PyTree of values to checkpoint.

    Raises:
        ValueError: If ``path`` uses Orbax's reserved temporary suffix.
    """
    checkpoint_path = _validated_checkpoint_path(path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(checkpoint_path, args=ocp.args.PyTreeSave(state), force=False)


def load_legacy_v0_pytree(path: str | PathLike[str], target: Any | None = None) -> Any:
    """Read one legacy V0 PyTree checkpoint synchronously on every process's main thread.

    Args:
        path: Source checkpoint directory.
        target: Optional PyTree restore target guiding structure and sharding. ``None``
            restores the stored structure.

    Returns:
        The restored PyTree.

    Raises:
        ValueError: If ``path`` uses Orbax's reserved temporary suffix.
    """
    checkpoint_path = _validated_checkpoint_path(path)
    restore_args = None if target is None else ocp.checkpoint_utils.construct_restore_args(target)
    with ocp.PyTreeCheckpointer() as checkpointer:
        return checkpointer.restore(
            checkpoint_path,
            args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args),
        )


def save_legacy_v0_grain_iterator(
    path: str | PathLike[str],
    iterator: grain.DatasetIterator[Any],
) -> None:
    """Write one immutable Grain iterator with its legacy V0 handler on every process's main thread.

    Args:
        path: Destination checkpoint directory.
        iterator: Grain ``DatasetIterator`` whose state is saved.

    Raises:
        TypeError: If ``iterator`` is not a ``grain.DatasetIterator``.
        ValueError: If ``path`` uses Orbax's reserved temporary suffix.
    """
    _validate_iterator(iterator)
    checkpoint_path = _validated_checkpoint_path(path)
    with ocp.Checkpointer(_grain_handler()) as checkpointer:
        checkpointer.save(
            checkpoint_path,
            args=grain.checkpoint.CheckpointSave(iterator),
            force=False,
        )


def load_legacy_v0_grain_iterator(
    path: str | PathLike[str], iterator: grain.DatasetIterator[Any]
) -> grain.DatasetIterator[Any]:
    """Read one Grain iterator with its legacy V0 handler on every process's main thread.

    Args:
        path: Source checkpoint directory.
        iterator: Grain ``DatasetIterator`` to restore in place.

    Returns:
        The restored ``grain.DatasetIterator``.

    Raises:
        TypeError: If ``iterator`` is not a ``grain.DatasetIterator``, or the checkpoint did
            not restore a ``DatasetIterator``.
        ValueError: If ``path`` uses Orbax's reserved temporary suffix.
    """
    _validate_iterator(iterator)
    checkpoint_path = _validated_checkpoint_path(path)
    with ocp.Checkpointer(_grain_handler()) as checkpointer:
        restored = checkpointer.restore(
            checkpoint_path,
            args=grain.checkpoint.CheckpointRestore(iterator),
        )
    if not isinstance(restored, grain.DatasetIterator):
        raise TypeError("the legacy V0 Grain checkpoint did not restore a DatasetIterator")
    return restored


def _grain_handler() -> ocp.CheckpointHandler:
    return cast(ocp.CheckpointHandler, grain.checkpoint.CheckpointHandler())


def _validate_iterator(iterator: object) -> None:
    if not isinstance(iterator, grain.DatasetIterator):
        raise TypeError("iterator must be a Grain DatasetIterator")
