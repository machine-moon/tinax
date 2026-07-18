"""Validated device meshes and concrete array layouts."""

from collections.abc import Sequence as _Sequence
from math import prod as _prod

import jax as _jax

import tinax.sharding._validation as _validation


def make_mesh(
    devices: _Sequence[_jax.Device],
    shape: _Sequence[int],
    axis_names: _Sequence[str],
    *,
    axis_types: _Sequence[_jax.sharding.AxisType],
) -> _jax.sharding.Mesh:
    """Build a mesh from exactly the caller's devices and explicit axis policy.

    Args:
        devices: Non-empty sequence of distinct ``jax.Device`` instances, used exactly
            once each.
        shape: Positive size per mesh axis. Its product must equal the device count.
        axis_names: Unique, non-empty name per mesh axis.
        axis_types: ``jax.sharding.AxisType`` per mesh axis.

    Returns:
        A ``jax.sharding.Mesh`` over the provided devices.

    Raises:
        TypeError: If an argument is not the expected sequence or element type (booleans
            rejected for sizes).
        ValueError: If devices are empty or duplicated, a size is not positive, names are
            empty or not unique, the lengths disagree, or the shape does not use every
            device exactly once.
    """
    if not isinstance(devices, _Sequence) or isinstance(devices, (str, bytes)):
        raise TypeError("devices must be a sequence of jax.Device instances")
    device_tuple = tuple(devices)
    if not device_tuple:
        raise ValueError("devices must not be empty")
    if any(not isinstance(device, _jax.Device) for device in device_tuple):
        raise TypeError("devices must contain only jax.Device instances")
    if len(set(device_tuple)) != len(device_tuple):
        raise ValueError("devices must not contain duplicates")

    if not isinstance(shape, _Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError("shape must be a sequence of integers")
    mesh_shape = tuple(shape)
    if not mesh_shape:
        raise ValueError("shape must contain at least one mesh axis")
    if any(not isinstance(size, int) or isinstance(size, bool) for size in mesh_shape):
        raise TypeError("shape entries must be integers")
    if any(size < 1 for size in mesh_shape):
        raise ValueError("shape entries must be positive")

    if not isinstance(axis_names, _Sequence) or isinstance(axis_names, (str, bytes)):
        raise TypeError("axis_names must be a sequence of strings")
    names = tuple(axis_names)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("axis_names must contain only strings")
    if any(not name for name in names):
        raise ValueError("axis_names must be nonempty")
    if len(set(names)) != len(names):
        raise ValueError("axis_names must be unique")

    if not isinstance(axis_types, _Sequence) or isinstance(axis_types, (str, bytes)):
        raise TypeError("axis_types must be a sequence of AxisType values")
    types = tuple(axis_types)
    if any(not isinstance(axis_type, _jax.sharding.AxisType) for axis_type in types):
        raise TypeError("axis_types must contain only jax.sharding.AxisType values")
    if len(mesh_shape) != len(names) or len(names) != len(types):
        raise ValueError("shape, axis_names, and axis_types must have the same length")
    if _prod(mesh_shape) != len(device_tuple):
        raise ValueError("shape must use every provided device exactly once")

    return _jax.make_mesh(mesh_shape, names, axis_types=types, devices=device_tuple)


def layout(
    mesh: _jax.sharding.Mesh,
    partitions: _Sequence[str | tuple[str, ...] | None],
) -> _jax.sharding.NamedSharding:
    """Create an Explicit named layout whose rank is inferred from its partition entries.

    Args:
        mesh: Mesh with Explicit axes to build the layout over.
        partitions: One entry per array dimension: a mesh-axis name, a non-empty tuple of
            names, or ``None`` for replication.

    Returns:
        A ``jax.sharding.NamedSharding`` for the described layout.

    Raises:
        TypeError: If ``mesh`` is not a ``jax.sharding.Mesh`` or an entry has the wrong
            type.
        ValueError: If ``mesh`` does not use Explicit axes, a partition axis is empty or
            unknown, or a mesh axis partitions more than one dimension.
    """
    if not isinstance(mesh, _jax.sharding.Mesh):
        raise TypeError("mesh must be a jax.sharding.Mesh")
    _validation.require_explicit_axes(mesh)
    if not isinstance(partitions, _Sequence) or isinstance(partitions, (str, bytes)):
        raise TypeError("partitions must be a sequence")

    entries = tuple(partitions)
    used_axes: list[str] = []
    for entry in entries:
        if entry is None:
            continue
        if isinstance(entry, str):
            axes = (entry,)
        elif isinstance(entry, tuple) and entry:
            axes = entry
        else:
            raise TypeError("partition entries must be strings, nonempty tuples, or None")
        if any(not isinstance(axis, str) for axis in axes):
            raise TypeError("partition axes must be strings")
        if any(not axis for axis in axes):
            raise ValueError("partition axes must be nonempty")
        used_axes.extend(axes)

    unknown = set(used_axes) - set(mesh.axis_names)
    if unknown:
        raise ValueError(f"unknown mesh axes: {sorted(unknown)}")
    if len(set(used_axes)) != len(used_axes):
        raise ValueError("a mesh axis may partition at most one array dimension")
    return _jax.sharding.NamedSharding(mesh, _jax.sharding.PartitionSpec(*entries))
