"""Shared validation for device environment configuration."""

import sys as _sys


def require_jax_unimported(operation: str) -> None:
    if any(name == "jax" or name.startswith("jax.") for name in _sys.modules):
        raise RuntimeError(f"{operation} must run before importing JAX")


def validate_device_index(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer and not a boolean")
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value
