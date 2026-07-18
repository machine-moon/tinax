"""Explicit host placement and process-local array construction."""

from typing import Literal as _Literal

import jax as _jax
import numpy as _np

import tinax.sharding._validation as _validation


def place_host_array(
    array: _np.ndarray,
    sharding: _jax.sharding.NamedSharding,
    *,
    copy: bool,
) -> _jax.Array:
    """Place host data asynchronously; copy snapshots now, while false permits a lifetime source alias.

    Args:
        array: Host NumPy array to place.
        sharding: Fully addressable ``jax.sharding.NamedSharding`` describing placement.
        copy: If ``True``, snapshot ``array`` before the asynchronous transfer. If
            ``False``, permit aliasing the source for the transfer's lifetime.

    Returns:
        A JAX array placed according to ``sharding``.

    Raises:
        TypeError: If ``array`` is not a host NumPy array, ``sharding`` is not a
            ``NamedSharding``, or ``copy`` is not a bool.
        ValueError: If ``sharding`` is not fully addressable or cannot partition
            ``array``'s shape.
    """
    _validation.validate_host_array(array, "array")
    _validation.validate_sharding(sharding)
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")
    if not sharding.is_fully_addressable:
        raise ValueError("host placement requires a fully addressable sharding")
    _validation.validate_partitionable(sharding, array.shape)
    source = array.copy(order="K") if copy else array
    return _jax.device_put(source, sharding, may_alias=True)


def from_process_local_data(
    local_data: _np.ndarray,
    sharding: _jax.sharding.NamedSharding,
    global_shape: tuple[int, ...],
    *,
    copy: bool,
    replica_policy: _Literal["disallow", "assume_consistent"],
    process_index: int | None = None,
    process_count: int | None = None,
) -> _jax.Array:
    """Create without gathering; copy controls source lifetime and assume_consistent attests replica equality.

    Constructs a global array from each process's local shard. This is not a gather:
    callers are responsible for replica consistency.

    Args:
        local_data: This process's local host shard.
        sharding: Target ``jax.sharding.NamedSharding``.
        global_shape: Logical shape of the assembled global array.
        copy: If ``True``, snapshot ``local_data``; if ``False``, permit aliasing it.
        replica_policy: ``"disallow"`` rejects cross-process replicas;
            ``"assume_consistent"`` attests that replicated data is identical.
        process_index: Optional declared process index; must match the JAX runtime.
        process_count: Optional declared process count; must match the JAX runtime.

    Returns:
        The assembled global JAX array.

    Raises:
        TypeError: If an argument has the wrong type (booleans rejected for integers).
        ValueError: If ranks disagree, ``replica_policy`` is invalid, an index or count is
            out of range, the sharding cannot partition ``global_shape``, local extents are
            inconsistent, or cross-process replicas exist under ``"disallow"``.
        RuntimeError: If a declared ``process_index`` or ``process_count`` does not match
            the JAX runtime.
    """
    _validation.validate_host_array(local_data, "local_data")
    _validation.validate_sharding(sharding)
    _validation.validate_global_shape(global_shape)
    if local_data.ndim != len(global_shape):
        raise ValueError("local_data and global_shape must have the same rank")
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")
    if not isinstance(replica_policy, str):
        raise TypeError("replica_policy must be a string")
    if replica_policy not in {"disallow", "assume_consistent"}:
        raise ValueError("replica_policy must be 'disallow' or 'assume_consistent'")

    if process_index is not None:
        if not isinstance(process_index, int) or isinstance(process_index, bool):
            raise TypeError("process_index must be an integer or None")
        if process_index < 0:
            raise ValueError("process_index must be nonnegative")
        if process_index != _jax.process_index():
            raise RuntimeError("declared process_index does not match the JAX runtime")
    if process_count is not None:
        if not isinstance(process_count, int) or isinstance(process_count, bool):
            raise TypeError("process_count must be an integer or None")
        if process_count < 1:
            raise ValueError("process_count must be positive")
        if process_count != _jax.process_count():
            raise RuntimeError("declared process_count does not match the JAX runtime")
    if process_index is not None and process_count is not None and process_index >= process_count:
        raise ValueError("process_index must be less than process_count")

    _validation.validate_partitionable(sharding, global_shape)
    _validation.validate_process_local_extents(local_data.shape, sharding, global_shape)
    if replica_policy == "disallow" and _validation.has_cross_process_replicas(sharding, global_shape):
        raise ValueError("cross-process replicas require replica_policy='assume_consistent'")

    source = local_data.copy(order="K") if copy else local_data
    return _jax.make_array_from_process_local_data(sharding, source, global_shape=global_shape)
