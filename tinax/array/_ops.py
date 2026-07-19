"""Validated array operations that close silent-failure gaps in jax.numpy and jax.nn."""

from collections.abc import Sequence as _Sequence

import jax as _jax
import jax.numpy as _jnp
import numpy.typing as _npt
from jax import core as _core


def stack_batch(arrays: _Sequence[_jax.Array], *, axis: int = 0) -> _jax.Array:
    """Stack arrays after requiring one shared dtype, where jnp.stack silently promotes mismatches.

    Args:
        arrays: Non-empty sequence of JAX arrays to stack. All elements must share one dtype.
        axis: Position of the new stacked axis. Defaults to 0.

    Returns:
        The arrays stacked along ``axis`` via ``jax.numpy.stack``.

    Raises:
        TypeError: If ``arrays`` is not a sequence, an element is not a ``jax.Array``, or
            ``axis`` is not an integer (booleans rejected).
        ValueError: If ``arrays`` is empty or its elements do not share one dtype.
    """
    if not isinstance(arrays, _Sequence) or isinstance(arrays, (str, bytes)):
        raise TypeError("arrays must be a sequence of jax.Array instances")
    array_tuple = tuple(arrays)
    if not array_tuple:
        raise ValueError("arrays must not be empty")
    if any(not isinstance(array, (_jax.Array, _core.Tracer)) for array in array_tuple):
        raise TypeError("arrays must contain only jax.Array instances")
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError("axis must be an integer")
    dtypes = {array.dtype for array in array_tuple}
    if len(dtypes) > 1:
        raise ValueError(f"arrays must share one dtype, got {sorted(str(dtype) for dtype in dtypes)}")
    return _jnp.stack(array_tuple, axis=axis)


def safe_astype(array: _jax.Array, dtype: _npt.DTypeLike, *, allow_lossy: bool = False) -> _jax.Array:
    """Cast to dtype, requiring allow_lossy=True for a narrowing cast that jnp.astype performs silently.

    Args:
        array: JAX array to cast.
        dtype: Target dtype.
        allow_lossy: Whether to permit a cast that can lose precision or overflow. Defaults
            to ``False``.

    Returns:
        ``array`` cast to ``dtype`` via ``jax.numpy.astype``.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array`` or ``allow_lossy`` is not a bool.
        ValueError: If the cast from ``array``'s dtype to ``dtype`` is not value-preserving
            and ``allow_lossy`` is ``False``.
    """
    if not isinstance(array, (_jax.Array, _core.Tracer)):
        raise TypeError("array must be a jax.Array")
    if not isinstance(allow_lossy, bool):
        raise TypeError("allow_lossy must be a bool")
    target_dtype = _jnp.dtype(dtype)
    if not allow_lossy and not _jnp.can_cast(array.dtype, target_dtype, casting="safe"):
        raise ValueError(
            f"casting {array.dtype} to {target_dtype} can lose precision or overflow; "
            "pass allow_lossy=True to permit it"
        )
    return _jnp.astype(array, target_dtype)


def one_hot(indices: _jax.Array, *, num_classes: int, dtype: _npt.DTypeLike | None = None) -> _jax.Array:
    """Encode integer indices as one-hot rows, rejecting inputs jax.nn.one_hot would silently zero out.

    Does not validate that concrete index values lie in ``0..num_classes - 1``: that requires
    concrete values and would break tracing under ``jax.jit``. Out-of-range indices still
    produce an all-zero row, matching ``jax.nn.one_hot``'s own documented behavior.

    Args:
        indices: JAX array of integer indices to encode.
        num_classes: Number of one-hot classes. Must be a positive integer.
        dtype: Output dtype. ``None`` uses ``jax.nn.one_hot``'s default.

    Returns:
        A one-hot encoding of ``indices`` with a trailing ``num_classes`` axis.

    Raises:
        TypeError: If ``indices`` is not a ``jax.Array``, its dtype is not an integer dtype,
            or ``num_classes`` is not an integer (booleans rejected).
        ValueError: If ``num_classes`` is not positive.
    """
    if not isinstance(indices, (_jax.Array, _core.Tracer)):
        raise TypeError("indices must be a jax.Array")
    if not _jax.dtypes.issubdtype(indices.dtype, _jnp.integer):
        raise TypeError("indices must have an integer dtype")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be an integer")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    return _jax.nn.one_hot(indices, num_classes, dtype=dtype)
