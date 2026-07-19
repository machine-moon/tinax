"""Backend-agnostic pre-JAX environment policy honored by the JAX runtime."""

import os as _os
from collections.abc import Iterable

from tinax.device._validation import require_jax_unimported, validate_device_index


def set_visible_cuda(indices: Iterable[int]) -> None:
    """Set ``CUDA_VISIBLE_DEVICES`` from explicit device indices before JAX initializes.

    Args:
        indices: Non-negative NVIDIA device indices to expose, in order.

    Raises:
        RuntimeError: If JAX has already been imported.
        TypeError: If an index is not an integer, or is a boolean.
        ValueError: If an index is negative.
    """
    require_jax_unimported("set_visible_cuda()")
    selected = tuple(validate_device_index(index, f"indices[{position}]") for position, index in enumerate(indices))
    _os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in selected)


def configure_single_chip(chip: int) -> None:
    """Expose exactly one local TPU chip to a forthcoming JAX import.

    Args:
        chip: Non-negative local TPU chip index to expose.

    Raises:
        RuntimeError: If JAX has already been imported.
        TypeError: If ``chip`` is not an integer, or is a boolean.
        ValueError: If ``chip`` is negative.
    """
    require_jax_unimported("configure_single_chip()")
    validated = validate_device_index(chip, "chip")
    _os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    _os.environ["TPU_VISIBLE_CHIPS"] = str(validated)


def configure_jax(*, preallocate: bool = False, matmul_precision: str | None = "high") -> None:
    """Set JAX/XLA runtime options before JAX initializes.

    Args:
        preallocate: Whether XLA may preallocate the device memory pool; must be set
            before JAX initializes.
        matmul_precision: Default matmul precision applied via ``jax.config`` once JAX
            is imported, or ``None`` to leave it unchanged.

    Raises:
        RuntimeError: If JAX has already been imported.
        TypeError: If ``preallocate`` is not a boolean, or ``matmul_precision`` is
            neither a string nor ``None``.
    """
    if not isinstance(preallocate, bool):
        raise TypeError("preallocate must be a boolean")
    if matmul_precision is not None and not isinstance(matmul_precision, str):
        raise TypeError("matmul_precision must be a string or None")
    require_jax_unimported("configure_jax()")
    _os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" if preallocate else "false"
    if matmul_precision is not None:
        import jax as _jax

        _jax.config.update("jax_default_matmul_precision", matmul_precision)
