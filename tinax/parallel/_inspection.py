"""Logical and addressable payload inspection without host materialization."""

import jax as _jax


def _standard_itemsize(array: _jax.Array) -> int | None:
    if _jax.dtypes.issubdtype(array.dtype, _jax.dtypes.extended):
        return None
    return int(array.dtype.itemsize)


def logical_payload_nbytes(array: _jax.Array) -> int | None:
    """Return nominal logical bytes, or None when the dtype has no standard byte representation.

    Args:
        array: JAX array to size. Its data is not materialized.

    Returns:
        The logical payload size in bytes, or ``None`` for extended dtypes.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array``.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    itemsize = _standard_itemsize(array)
    return None if itemsize is None else int(array.size) * itemsize


def addressable_payload_nbytes(array: _jax.Array) -> int | None:
    """Return nominal addressable bytes including replicas, or None for an extended dtype.

    Args:
        array: JAX array to size. Its data is not materialized.

    Returns:
        The total bytes across this process's addressable shards (replicas counted), or
        ``None`` for extended dtypes.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array``.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    itemsize = _standard_itemsize(array)
    if itemsize is None:
        return None
    return sum(int(shard.data.size) * itemsize for shard in array.addressable_shards)


def addressable_indices(array: _jax.Array) -> dict[_jax.Device, tuple[slice | int, ...] | None]:
    """Return each addressable device's global logical index.

    Args:
        array: JAX array whose sharding is inspected.

    Returns:
        A mapping from each addressable ``jax.Device`` to the global index tuple of the
        shard it holds, or ``None`` where unindexed.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array``.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    return dict(array.sharding.addressable_devices_indices_map(array.shape))
