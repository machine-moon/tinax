"""Host-side Flax msgpack pytree interchange."""

import os
from collections.abc import Mapping
from typing import Any

from flax import serialization as _serialization

from ._files import atomic_write_path
from ._validation import coerce_path, validate_byte_limit


def save_msgpack(
    path: str | os.PathLike[str],
    state: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    """Atomically serialize a host pytree to a Flax msgpack file.

    Args:
        path: Destination file path. Written atomically through a temporary file.
        state: Mapping pytree of host arrays to serialize, such as a pure parameter dict.
        overwrite: Whether to replace an existing destination.

    Raises:
        TypeError: If ``overwrite`` is not a bool or ``state`` is not a mapping.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping pytree of host arrays")
    payload = _serialization.to_bytes(state)
    destination = coerce_path(path)
    with atomic_write_path(destination, overwrite=overwrite) as temporary:
        temporary.write_bytes(payload)


def load_msgpack(
    path: str | os.PathLike[str],
    target: Mapping[str, Any],
    *,
    max_bytes: int,
) -> Any:
    """Deserialize a Flax msgpack file into the structure of ``target`` within a byte budget.

    Args:
        path: Path to the Flax msgpack file.
        target: Mapping pytree whose structure and leaf types the payload is restored into.
        max_bytes: Maximum encoded file bytes to read.

    Returns:
        A pytree matching ``target`` populated from the file's contents.

    Raises:
        TypeError: If ``target`` is not a mapping or ``max_bytes`` is not an integer.
        ValueError: If ``max_bytes`` is negative or the file exceeds the byte budget.
        FileNotFoundError: If no regular file exists at ``path``.
    """
    if not isinstance(target, Mapping):
        raise TypeError("target must be a mapping pytree matching the saved state")
    byte_budget = validate_byte_limit(max_bytes, name="max_bytes")
    source = coerce_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no msgpack file found at {source}")
    encoded_bytes = source.stat().st_size
    if encoded_bytes > byte_budget:
        raise ValueError(f"msgpack file is {encoded_bytes} bytes, exceeding max_bytes={byte_budget}")
    payload = source.read_bytes()
    return _serialization.from_bytes(target, payload)
