"""Bounded host-side Safetensors interchange."""

import os
from collections.abc import Iterable, Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt
from safetensors import safe_open as _safe_open
from safetensors.numpy import save_file as _save_file

from ._files import atomic_write_path
from ._formats import numpy_dtype
from ._models import LoadedSafetensors, SafetensorsInfo, TensorInfo
from ._validation import (
    coerce_path,
    normalize_names,
    validate_byte_limit,
    validate_host_tensors,
    validate_tensor_name,
    validated_metadata,
)


class _SafeSlice(Protocol):
    def get_shape(self) -> list[int]: ...

    def get_dtype(self) -> str: ...


class _SafeOpen(Protocol):
    def keys(self) -> list[str]: ...

    def metadata(self) -> dict[str, str] | None: ...

    def get_slice(self, name: str) -> _SafeSlice: ...

    def get_tensor(self, name: str) -> object: ...


def inspect_safetensors(path: str | os.PathLike[str], *, max_metadata_bytes: int = 0) -> SafetensorsInfo:
    """Inspect tensor keys, shapes, dtypes, sizes, and bounded metadata without loading payloads.

    Args:
        path: Path to the Safetensors file.
        max_metadata_bytes: Maximum header metadata bytes to accept. ``0`` allows none.

    Returns:
        A ``SafetensorsInfo`` describing the file's tensors and validated metadata.

    Raises:
        TypeError: If ``max_metadata_bytes`` is not an integer.
        ValueError: If ``max_metadata_bytes`` is negative, the metadata exceeds the budget,
            or a tensor name is invalid or duplicated.
    """
    metadata_budget = validate_byte_limit(max_metadata_bytes, name="max_metadata_bytes")
    with _safe_open(coerce_path(path), framework="np") as handle:
        return _inspect_handle(handle, max_metadata_bytes=metadata_budget)


def load_safetensors(
    path: str | os.PathLike[str],
    *,
    max_bytes: int,
    names: Iterable[str] | None = None,
    max_metadata_bytes: int = 0,
) -> LoadedSafetensors:
    """Inspect and selectively load NumPy tensors within an explicit nominal byte budget.

    Args:
        path: Path to the Safetensors file.
        max_bytes: Maximum total nominal payload bytes to load.
        names: Tensor names to load, or ``None`` to load all present tensors.
        max_metadata_bytes: Maximum header metadata bytes to accept. ``0`` allows none.

    Returns:
        A ``LoadedSafetensors`` holding the selected host tensors and file information.

    Raises:
        TypeError: If a byte limit is not an integer, or a loaded tensor has an
            unmaterializable or unexpected dtype.
        ValueError: If a byte limit is negative, requested names are absent, the selection
            exceeds ``max_bytes``, or a loaded tensor has an unexpected shape.
    """
    tensor_budget = validate_byte_limit(max_bytes, name="max_bytes")
    metadata_budget = validate_byte_limit(max_metadata_bytes, name="max_metadata_bytes")
    requested_names = normalize_names(names)
    with _safe_open(coerce_path(path), framework="np") as handle:
        info = _inspect_handle(handle, max_metadata_bytes=metadata_budget)
        selected_names = tuple(info.tensors) if requested_names is None else requested_names
        missing = sorted(set(selected_names) - set(info.tensors))
        if missing:
            raise ValueError(f"requested tensors are not present in the file: {missing}")
        required_bytes = sum(info.tensors[name].nbytes for name in selected_names)
        if required_bytes > tensor_budget:
            raise ValueError(f"selected tensors require {required_bytes} bytes, exceeding max_bytes={tensor_budget}")

        expected_dtypes: dict[str, np.dtype[np.generic]] = {}
        for name in selected_names:
            format_dtype = info.tensors[name].dtype
            expected_dtype = numpy_dtype(format_dtype)
            if expected_dtype is None:
                raise TypeError(
                    f"tensor {name!r} has Safetensors dtype {format_dtype!r}, "
                    "which the active NumPy installation cannot materialize"
                )
            expected_dtypes[name] = expected_dtype

        loaded: dict[str, npt.NDArray[np.generic]] = {}
        for name in selected_names:
            tensor = handle.get_tensor(name)
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Safetensors returned a non-NumPy value for tensor {name!r}")
            expected = info.tensors[name]
            if tuple(tensor.shape) != expected.shape:
                raise ValueError(f"loaded tensor {name!r} has shape {tuple(tensor.shape)}, expected {expected.shape}")
            if tensor.dtype != expected_dtypes[name]:
                raise TypeError(f"loaded tensor {name!r} has dtype {tensor.dtype}, expected {expected_dtypes[name]}")
            loaded[name] = tensor
    return LoadedSafetensors(loaded, info)


def save_safetensors(
    path: str | os.PathLike[str],
    tensors: Mapping[str, npt.NDArray[np.generic]],
    *,
    overwrite: bool,
    metadata: Mapping[str, str] | None = None,
    max_metadata_bytes: int = 0,
) -> None:
    """Atomically save explicit C-contiguous host tensors in the destination directory.

    Args:
        path: Destination file path. Written atomically through a temporary file.
        tensors: Mapping of tensor name to host NumPy array to serialize.
        overwrite: Whether to replace an existing destination.
        metadata: Optional string-to-string header metadata.
        max_metadata_bytes: Maximum metadata bytes to accept. ``0`` allows none.

    Raises:
        TypeError: If ``overwrite`` is not a bool, ``max_metadata_bytes`` is not an integer,
            or ``tensors`` or ``metadata`` have invalid types.
        ValueError: If ``max_metadata_bytes`` is negative, the metadata exceeds the budget, or
            a tensor is invalid.
        FileExistsError: If the destination exists and ``overwrite`` is ``False``.
    """
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    metadata_budget = validate_byte_limit(max_metadata_bytes, name="max_metadata_bytes")
    host_tensors = validate_host_tensors(tensors)
    metadata_copy = validated_metadata(metadata, max_bytes=metadata_budget)
    destination = coerce_path(path)
    with atomic_write_path(destination, overwrite=overwrite) as temporary:
        _save_file(host_tensors, temporary, metadata=metadata_copy or None)


def _inspect_handle(handle: _SafeOpen, *, max_metadata_bytes: int) -> SafetensorsInfo:
    metadata = validated_metadata(handle.metadata(), max_bytes=max_metadata_bytes)
    tensor_info: dict[str, TensorInfo] = {}
    for name in handle.keys():
        validate_tensor_name(name)
        if name in tensor_info:
            raise ValueError(f"duplicate tensor name in Safetensors header: {name!r}")
        tensor_slice = handle.get_slice(name)
        tensor_info[name] = TensorInfo(tuple(tensor_slice.get_shape()), tensor_slice.get_dtype())
    return SafetensorsInfo(tensor_info, metadata)
