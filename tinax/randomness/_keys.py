"""Validated typed-key derivation and splitting."""

from typing import Any

import jax


def _typed_scalar_key(key: Any) -> Any:
    dtype = getattr(key, "dtype", None)
    shape = getattr(key, "shape", None)
    if dtype is None or shape is None:
        raise TypeError("key must be a typed scalar JAX PRNG key")
    if tuple(shape) != ():
        raise ValueError("key must be a scalar JAX PRNG key")
    if not jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        raise TypeError("key must use a typed JAX PRNG dtype")
    return key


def _integer_scalar(name: str, value: Any) -> Any:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer scalar and not a boolean")
    if isinstance(value, int):
        if value < 0 or value > 2**32 - 1:
            raise ValueError(f"{name} must be between 0 and 2**32 - 1")
        return value
    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if dtype is None or shape is None or tuple(shape) != ():
        raise TypeError(f"{name} must be an integer scalar")
    if not jax.dtypes.issubdtype(dtype, jax.numpy.unsignedinteger) or dtype.itemsize > 4:
        raise TypeError(f"{name} JAX scalars must use an unsigned dtype of at most 32 bits")
    return value


def derive_key(key: jax.Array, *coordinates: int | jax.Array) -> jax.Array:
    """Fold ordered integer coordinates into one typed scalar key without consuming global state.

    Args:
        key: Typed scalar JAX PRNG key to derive from.
        *coordinates: Integer coordinates folded in order via ``jax.random.fold_in``.
            Concrete Python values must lie in ``0..2**32 - 1``; JIT-time values must
            be unsigned scalar JAX arrays no wider than 32 bits.

    Returns:
        A new typed scalar JAX key derived from ``key`` and the coordinates. The
        input key is not consumed.

    Raises:
        TypeError: If ``key`` is not a typed scalar PRNG key, or a coordinate is a
            boolean, non-scalar, or a JAX scalar with a signed or oversized dtype.
        ValueError: If ``key`` is not scalar, or a Python coordinate falls outside
            ``0..2**32 - 1``.
    """
    derived = _typed_scalar_key(key)
    for index, coordinate in enumerate(coordinates):
        derived = jax.random.fold_in(derived, _integer_scalar(f"coordinates[{index}]", coordinate))
    return derived


def derive_process_step_key(
    key: jax.Array,
    *,
    process_index: int | jax.Array,
    step: int | jax.Array,
    stream: int | jax.Array = 0,
) -> jax.Array:
    """Derive one key using the stable process, step, then stream coordinate order.

    Args:
        key: Typed scalar JAX PRNG key to derive from.
        process_index: Coordinate identifying the process; folded in first.
        step: Coordinate identifying the step; folded in second.
        stream: Coordinate identifying the substream; folded in last. Defaults to 0.

    Returns:
        A new typed scalar JAX key derived in process, step, then stream order.

    Raises:
        TypeError: If ``key`` is not a typed scalar PRNG key, or a coordinate is a
            boolean, non-scalar, or a JAX scalar with a signed or oversized dtype.
        ValueError: If ``key`` is not scalar, or a Python coordinate falls outside
            ``0..2**32 - 1``.
    """
    return derive_key(key, process_index, step, stream)


def split_key(key: jax.Array, *, count: int) -> tuple[jax.Array, jax.Array]:
    """Return a continuation key and a leading-axis array of explicitly consumed operation keys.

    Args:
        key: Typed scalar JAX PRNG key to split.
        count: Number of operation keys to produce. Must be a non-negative integer.

    Returns:
        A ``(continuation_key, operation_keys)`` tuple. ``continuation_key`` is a
        scalar key for further derivation; ``operation_keys`` has leading dimension
        ``count``.

    Raises:
        TypeError: If ``key`` is not a typed scalar PRNG key, or ``count`` is not an
            integer (booleans are rejected).
        ValueError: If ``key`` is not scalar, or ``count`` is negative.
    """
    _typed_scalar_key(key)
    if not isinstance(count, int) or isinstance(count, bool):
        raise TypeError("count must be an integer and not a boolean")
    if count < 0:
        raise ValueError("count must be non-negative")
    keys = jax.random.split(key, count + 1)
    return keys[0], keys[1:]
