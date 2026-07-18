"""Shared validation for concrete layouts and process-local array construction."""

from math import prod as _prod

import jax as _jax
import numpy as _np


def require_explicit_axes(mesh: _jax.sharding.Mesh) -> None:
    manual = tuple(
        name
        for name, axis_type in zip(mesh.axis_names, mesh.axis_types, strict=True)
        if axis_type is _jax.sharding.AxisType.Manual
    )
    if manual:
        raise ValueError(f"Manual mesh axes require jax.shard_map, not a concrete layout: {manual}")
    automatic = tuple(
        name
        for name, axis_type in zip(mesh.axis_names, mesh.axis_types, strict=True)
        if axis_type is _jax.sharding.AxisType.Auto
    )
    if automatic:
        raise ValueError(f"Auto mesh axes are compiler-selected, not concrete layout axes: {automatic}")


def validate_host_array(array: _np.ndarray, name: str) -> None:
    if not isinstance(array, _np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    is_numeric = _jax.dtypes.issubdtype(array.dtype, _np.number)
    is_boolean = _jax.dtypes.issubdtype(array.dtype, _np.bool_)
    if not is_numeric and not is_boolean:
        raise TypeError(f"{name} must have a JAX-compatible numeric or boolean dtype")


def validate_sharding(sharding: _jax.sharding.NamedSharding) -> None:
    if not isinstance(sharding, _jax.sharding.NamedSharding):
        raise TypeError("sharding must be a jax.sharding.NamedSharding")
    if not isinstance(sharding.mesh, _jax.sharding.Mesh) or sharding.mesh.empty:
        raise ValueError("sharding must use a nonempty concrete mesh")
    require_explicit_axes(sharding.mesh)


def validate_global_shape(shape: tuple[int, ...]) -> None:
    if not isinstance(shape, tuple):
        raise TypeError("global_shape must be a tuple")
    if any(not isinstance(size, int) or isinstance(size, bool) for size in shape):
        raise TypeError("global_shape entries must be integers")
    if any(size < 0 for size in shape):
        raise ValueError("global_shape entries must be nonnegative")


def validate_partitionable(sharding: _jax.sharding.NamedSharding, shape: tuple[int, ...]) -> None:
    try:
        sharding.shard_shape(shape)
    except ValueError as error:
        raise ValueError("shape is not compatible with the requested layout") from error


def _normalize_index(index: object, shape: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    if not isinstance(index, tuple) or len(index) != len(shape):
        raise ValueError("sharding returned an invalid array index")
    normalized: list[tuple[int, int]] = []
    for coordinate, size in zip(index, shape, strict=True):
        if isinstance(coordinate, slice):
            start, stop, step = coordinate.indices(size)
            if step != 1:
                raise ValueError("strided sharding indices are not supported")
        elif isinstance(coordinate, int) and not isinstance(coordinate, bool):
            position = coordinate + size if coordinate < 0 else coordinate
            if position < 0 or position >= size:
                raise ValueError("sharding returned an out-of-range array index")
            start, stop = position, position + 1
        else:
            raise ValueError("sharding returned an unsupported array index")
        normalized.append((start, stop))
    return tuple(normalized)


def validate_process_local_extents(
    local_shape: tuple[int, ...],
    sharding: _jax.sharding.NamedSharding,
    global_shape: tuple[int, ...],
) -> None:
    raw_indices = sharding.addressable_devices_indices_map(global_shape)
    if not raw_indices:
        raise ValueError("sharding has no devices addressable by this process")
    indices = tuple(_normalize_index(index, global_shape) for index in raw_indices.values())
    for dimension, (local_extent, global_extent) in enumerate(zip(local_shape, global_shape, strict=True)):
        unique_spans = {index[dimension] for index in indices}
        addressable_extent = sum(stop - start for start, stop in unique_spans)
        if local_extent not in {global_extent, addressable_extent}:
            raise ValueError(
                f"local_data dimension {dimension} must equal global extent {global_extent} "
                f"or addressable extent {addressable_extent}, got {local_extent}"
            )


def has_cross_process_replicas(
    sharding: _jax.sharding.NamedSharding, shape: tuple[int, ...]
) -> bool:
    owners: dict[tuple[tuple[int, int], ...], set[int]] = {}
    for device, raw_index in sharding.devices_indices_map(shape).items():
        index = _normalize_index(raw_index, shape)
        if _prod(stop - start for start, stop in index) == 0:
            continue
        owners.setdefault(index, set()).add(device.process_index)
    return any(len(processes) > 1 for processes in owners.values())
