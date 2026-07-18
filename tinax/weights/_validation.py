"""Validation for bounded host-side weight interchange."""

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ._formats import supports_serialization


def validate_host_tensors(
    tensors: Mapping[str, npt.NDArray[np.generic]],
) -> dict[str, npt.NDArray[np.generic]]:
    """Validate explicit C-contiguous NumPy host tensors."""
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping of names to numpy.ndarray values")
    validated: dict[str, npt.NDArray[np.generic]] = {}
    for name, tensor in tensors.items():
        validate_tensor_name(name)
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"tensor {name!r} must be a numpy.ndarray on the host; convert or gather it explicitly before saving"
            )
        if not tensor.flags.c_contiguous:
            raise ValueError(f"tensor {name!r} must be C-contiguous")
        if not supports_serialization(tensor.dtype):
            raise TypeError(f"tensor {name!r} has unsupported NumPy dtype {tensor.dtype}")
        validated[name] = tensor
    return validated


def validated_metadata(metadata: object, *, max_bytes: int | None) -> dict[str, str]:
    """Copy string metadata after enforcing its encoded byte budget."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping of strings to strings or None")
    validated: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be strings")
        if not isinstance(value, str):
            raise TypeError(f"metadata value for key {key!r} must be a string")
        validated[key] = value
    encoded_size = 0
    if validated:
        encoded = json.dumps(validated, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        encoded_size = len(encoded)
    if max_bytes is not None and encoded_size > max_bytes:
        raise ValueError(f"metadata requires {encoded_size} bytes, exceeding max_metadata_bytes={max_bytes}")
    return dict(sorted(validated.items()))


def normalize_names(names: Iterable[str] | None) -> tuple[str, ...] | None:
    """Copy and validate an optional tensor-name selection."""
    if names is None:
        return None
    if isinstance(names, (str, bytes)):
        raise TypeError("names must be an iterable of tensor names, not a string")
    try:
        normalized = tuple(names)
    except TypeError as error:
        raise TypeError("names must be an iterable of tensor names or None") from error
    seen: set[str] = set()
    for name in normalized:
        validate_tensor_name(name)
        if name in seen:
            raise ValueError(f"duplicate requested tensor name: {name!r}")
        seen.add(name)
    return normalized


def coerce_path(path: str | os.PathLike[str]) -> Path:
    """Return a path after validating its public boundary type."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or os.PathLike[str]")
    return Path(path)


def validate_tensor_name(name: object) -> None:
    """Validate a non-reserved Safetensors tensor name."""
    if not isinstance(name, str):
        raise TypeError("tensor names must be strings")
    if not name:
        raise ValueError("tensor names must be non-empty")
    if name == "__metadata__":
        raise ValueError("tensor name '__metadata__' is reserved by the Safetensors format")


def validate_byte_limit(value: object, *, name: str) -> int:
    """Validate a non-negative byte count without accepting booleans."""
    if type(value) is not int:
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
