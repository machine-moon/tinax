"""Logical JAX array inspection without host materialization."""

from dataclasses import dataclass

import jax as _jax


@dataclass(frozen=True, slots=True)
class ArrayInfo:
    """Immutable logical metadata whose byte count is nominal rather than allocator memory.

    Attributes:
        shape: Logical shape of the array.
        dtype: Array dtype. Extended dtypes report ``logical_nbytes`` as ``None``.
        logical_nbytes: Logical size in bytes, or ``None`` for extended dtypes whose
            element size is not a fixed byte count.
        committed: Whether the array is committed to a specific device.
        fully_addressable: Whether every shard is addressable by the current process.
    """

    shape: tuple[int, ...]
    dtype: object
    logical_nbytes: int | None
    committed: bool
    fully_addressable: bool


def inspect_array(array: _jax.Array) -> ArrayInfo:
    """Inspect logical metadata without synchronizing or materializing array data.

    Args:
        array: JAX array to inspect. Its data is never copied to the host.

    Returns:
        An ``ArrayInfo`` describing the array's shape, dtype, logical byte count,
        commitment, and addressability.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array``.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    dtype = array.dtype
    logical_nbytes = None if _jax.dtypes.issubdtype(dtype, _jax.dtypes.extended) else array.nbytes
    return ArrayInfo(tuple(array.shape), dtype, logical_nbytes, array.committed, array.is_fully_addressable)
