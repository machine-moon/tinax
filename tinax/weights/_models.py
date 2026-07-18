"""Immutable Safetensors inspection and loading models."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

from ._formats import numpy_dtype, serialized_nbytes
from ._validation import validate_tensor_name, validated_metadata


@dataclass(frozen=True, slots=True)
class TensorInfo:
    """Describe one tensor without materializing its payload.

    Attributes:
        shape: Tuple of non-negative tensor dimensions.
        dtype: Safetensors dtype string (for example, ``"F32"``).

    Raises:
        TypeError: If ``shape`` is not a tuple of integers or ``dtype`` is not a string.
        ValueError: If a ``shape`` dimension is negative or the dtype and shape are invalid.
    """

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            raise TypeError("tensor shape must be a tuple of non-negative integers")
        for dimension in self.shape:
            if type(dimension) is not int:
                raise TypeError("tensor shape must be a tuple of non-negative integers")
            if dimension < 0:
                raise ValueError("tensor shape dimensions must be non-negative")
        if not isinstance(self.dtype, str):
            raise TypeError("tensor dtype must be a Safetensors dtype string")
        serialized_nbytes(self.shape, self.dtype)

    @property
    def nbytes(self) -> int:
        """Return the nominal serialized payload size."""
        return serialized_nbytes(self.shape, self.dtype)


@dataclass(frozen=True, slots=True)
class SafetensorsInfo:
    """Describe a validated Safetensors header and its bounded metadata.

    Attributes:
        tensors: Read-only mapping of tensor name to ``TensorInfo``.
        metadata: Read-only mapping of validated header metadata.

    Raises:
        TypeError: If ``tensors`` is not a mapping or a value is not a ``TensorInfo``.
        ValueError: If a tensor name is invalid.
    """

    tensors: Mapping[str, TensorInfo]
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.tensors, Mapping):
            raise TypeError("tensors must be a mapping of names to TensorInfo values")
        tensor_copy: dict[str, TensorInfo] = {}
        for name, info in self.tensors.items():
            validate_tensor_name(name)
            if not isinstance(info, TensorInfo):
                raise TypeError(f"tensor info for {name!r} must be a TensorInfo")
            tensor_copy[name] = info
        metadata_copy = validated_metadata(self.metadata, max_bytes=None)
        object.__setattr__(self, "tensors", MappingProxyType(tensor_copy))
        object.__setattr__(self, "metadata", MappingProxyType(metadata_copy))

    @property
    def total_bytes(self) -> int:
        """Return the nominal payload size of all tensors."""
        return sum(info.nbytes for info in self.tensors.values())


@dataclass(frozen=True, slots=True)
class LoadedSafetensors:
    """Hold selected host tensors and the inspected file information.

    Attributes:
        tensors: Read-only mapping of tensor name to host NumPy array.
        info: The ``SafetensorsInfo`` the tensors were validated against.

    Raises:
        TypeError: If ``info`` is not a ``SafetensorsInfo``, ``tensors`` is not a mapping, a
            tensor is not a NumPy array, or a dtype disagrees with ``info``.
        ValueError: If a tensor name is invalid, absent from ``info``, or has an unexpected
            shape.
    """

    tensors: Mapping[str, npt.NDArray[np.generic]]
    info: SafetensorsInfo

    def __post_init__(self) -> None:
        if not isinstance(self.info, SafetensorsInfo):
            raise TypeError("info must be a SafetensorsInfo")
        if not isinstance(self.tensors, Mapping):
            raise TypeError("tensors must be a mapping of names to numpy.ndarray values")
        tensor_copy: dict[str, npt.NDArray[np.generic]] = {}
        for name, tensor in self.tensors.items():
            validate_tensor_name(name)
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"loaded tensor {name!r} must be a numpy.ndarray")
            if name not in self.info.tensors:
                raise ValueError(f"loaded tensor {name!r} is absent from the inspected file information")
            expected = self.info.tensors[name]
            if tuple(tensor.shape) != expected.shape:
                raise ValueError(f"loaded tensor {name!r} has shape {tuple(tensor.shape)}, expected {expected.shape}")
            expected_dtype = numpy_dtype(expected.dtype)
            if expected_dtype is None or tensor.dtype != expected_dtype:
                raise TypeError(f"loaded tensor {name!r} has dtype {tensor.dtype}, expected {expected.dtype}")
            tensor_copy[name] = tensor
        object.__setattr__(self, "tensors", MappingProxyType(tensor_copy))

    @property
    def metadata(self) -> Mapping[str, str]:
        """Return the validated metadata from the file header."""
        return self.info.metadata
