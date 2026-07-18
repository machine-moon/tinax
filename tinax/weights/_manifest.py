"""Exact tensor naming, transformation, and output contracts."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TensorRule:
    """Describe one source-to-destination tensor transformation.

    Attributes:
        source_name: Non-empty source tensor name.
        destination_name: Non-empty destination tensor name.
        transform: Callable mapping the source NumPy array to the destination array.
        expected_shape: Tuple of non-negative dimensions the transform must produce.
        expected_dtype: NumPy dtype the transform must produce.

    Raises:
        TypeError: If a name is not a string, ``transform`` is not callable,
            ``expected_shape`` is not a tuple of integers, or ``expected_dtype`` is not a
            valid NumPy dtype.
        ValueError: If a name is empty or an ``expected_shape`` dimension is negative.
    """

    source_name: str
    destination_name: str
    transform: Callable[[npt.NDArray[np.generic]], npt.NDArray[np.generic]]
    expected_shape: tuple[int, ...]
    expected_dtype: np.dtype[np.generic]

    def __post_init__(self) -> None:
        _validate_name(self.source_name, role="source")
        _validate_name(self.destination_name, role="destination")
        if not callable(self.transform):
            raise TypeError("transform must be callable")
        if not isinstance(self.expected_shape, tuple):
            raise TypeError("expected_shape must be a tuple of non-negative integers")
        for dimension in self.expected_shape:
            if type(dimension) is not int:
                raise TypeError("expected_shape must be a tuple of non-negative integers")
            if dimension < 0:
                raise ValueError("expected_shape dimensions must be non-negative")
        try:
            dtype = np.dtype(self.expected_dtype)
        except TypeError as error:
            raise TypeError(f"expected_dtype is not a valid NumPy dtype: {self.expected_dtype!r}") from error
        object.__setattr__(self, "expected_dtype", dtype)


@dataclass(frozen=True, slots=True)
class TensorManifest:
    """Apply immutable rules with exact source coverage.

    Attributes:
        rules: Tuple of ``TensorRule`` with distinct source and destination names.

    Raises:
        TypeError: If ``rules`` is not a tuple of ``TensorRule`` instances.
        ValueError: If two rules share a source name or collide on a destination name.
    """

    rules: tuple[TensorRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.rules, tuple):
            raise TypeError("rules must be a tuple of TensorRule instances")
        sources: set[str] = set()
        destinations: dict[str, str] = {}
        for rule in self.rules:
            if not isinstance(rule, TensorRule):
                raise TypeError("rules must contain only TensorRule instances")
            if rule.source_name in sources:
                raise ValueError(f"duplicate source tensor name in manifest: {rule.source_name!r}")
            previous_source = destinations.get(rule.destination_name)
            if previous_source is not None:
                raise ValueError(
                    f"destination tensor collision for {rule.destination_name!r}: "
                    f"sources {previous_source!r} and {rule.source_name!r}"
                )
            sources.add(rule.source_name)
            destinations[rule.destination_name] = rule.source_name

    def apply(
        self, tensors: Mapping[str, npt.NDArray[np.generic]]
    ) -> dict[str, npt.NDArray[np.generic]]:
        """Transform tensors after requiring exact source-name coverage.

        Args:
            tensors: Mapping of source name to host NumPy array. Its key set must exactly
                match the manifest's source names.

        Returns:
            A dict mapping each destination name to its transformed NumPy array.

        Raises:
            TypeError: If ``tensors`` is not a mapping, a source is not a NumPy array, or a
                transform returns the wrong type or dtype.
            ValueError: If the source names do not exactly match, or a transform produces an
                unexpected shape.
        """
        if not isinstance(tensors, Mapping):
            raise TypeError("tensors must be a mapping of names to numpy.ndarray values")
        for name in tensors:
            _validate_name(name, role="source")
        expected_names = {rule.source_name for rule in self.rules}
        actual_names = set(tensors)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            raise ValueError(f"source tensor coverage differs: missing={missing}, unexpected={unexpected}")

        transformed_tensors: dict[str, npt.NDArray[np.generic]] = {}
        for rule in self.rules:
            source = tensors[rule.source_name]
            if not isinstance(source, np.ndarray):
                raise TypeError(f"source tensor {rule.source_name!r} must be a numpy.ndarray")
            transformed = rule.transform(source)
            if not isinstance(transformed, np.ndarray):
                raise TypeError(f"transform for source tensor {rule.source_name!r} must return a numpy.ndarray")
            actual_shape = tuple(transformed.shape)
            if actual_shape != rule.expected_shape:
                raise ValueError(
                    f"transform for source tensor {rule.source_name!r} produced shape {actual_shape}, "
                    f"expected {rule.expected_shape}"
                )
            if transformed.dtype != rule.expected_dtype:
                raise TypeError(
                    f"transform for source tensor {rule.source_name!r} produced dtype {transformed.dtype}, "
                    f"expected {rule.expected_dtype}"
                )
            transformed_tensors[rule.destination_name] = transformed
        return transformed_tensors


def _validate_name(name: object, *, role: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"{role} tensor names must be strings")
    if not name:
        raise ValueError(f"{role} tensor names must be non-empty")
