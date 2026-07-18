"""Validated atomic Orbax V1 checkpointable operations."""

from collections.abc import Mapping
from os import PathLike
from typing import Any

from orbax.checkpoint import v1 as ocp

from ._names import _portable_name_key, _validated_checkpoint_path, validate_checkpointable_name


def save_checkpointables(
    path: str | PathLike[str],
    checkpointables: Mapping[str, Any],
) -> ocp.AsyncResponse[None]:
    """Start one immutable atomic V1 save on every process's main thread and return its completion response.

    Args:
        path: Destination checkpoint directory. Must not use Orbax's reserved temporary
            suffix.
        checkpointables: Non-empty mapping of checkpointable name to value. Names must
            be portable and distinct.

    Returns:
        An Orbax ``AsyncResponse``; call ``.result()`` to await completion before
        loading from or deleting the destination.

    Raises:
        TypeError: If ``checkpointables`` is not a mapping or a name is not a string.
        ValueError: If ``checkpointables`` is empty, a name is invalid or not distinct,
            or ``path`` uses the reserved temporary suffix.
    """
    values = _validated_checkpointable_mapping(checkpointables, argument="checkpointables", reject_none=False)
    return ocp.save_checkpointables_async(_validated_checkpoint_path(path), values, overwrite=False)


def load_checkpointables(
    path: str | PathLike[str],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Load all named V1 targets together on every process's main thread after save completion.

    Args:
        path: Source checkpoint directory.
        targets: Non-empty mapping of checkpointable name to an explicit non-``None``
            restore target. Names must be portable and distinct.

    Returns:
        A dict mapping each requested name to its restored value.

    Raises:
        TypeError: If ``targets`` is not a mapping or a name is not a string.
        ValueError: If ``targets`` is empty, a name is invalid or not distinct, a target
            is ``None``, or ``path`` uses the reserved temporary suffix.
    """
    abstract_values = _validated_checkpointable_mapping(targets, argument="targets", reject_none=True)
    return ocp.load_checkpointables(_validated_checkpoint_path(path), abstract_values)


def _validated_checkpointable_mapping(
    checkpointables: object, *, argument: str, reject_none: bool
) -> dict[str, Any]:
    if not isinstance(checkpointables, Mapping):
        raise TypeError(f"{argument} must be a mapping")
    if not checkpointables:
        raise ValueError(f"{argument} must not be empty")
    values: dict[str, Any] = {}
    portable_names: set[str] = set()
    for name, value in checkpointables.items():
        validated_name = validate_checkpointable_name(name)
        portable_name = _portable_name_key(validated_name)
        if portable_name in portable_names:
            raise ValueError(f"{argument} names must be portable and distinct")
        if reject_none and value is None:
            raise ValueError(f"target {validated_name!r} must be an explicit non-None restore target")
        portable_names.add(portable_name)
        values[validated_name] = value
    return values
