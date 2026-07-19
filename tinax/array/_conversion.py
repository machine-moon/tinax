"""Explicit array conversion and host materialization contracts."""

import jax as _jax
import jax.numpy as _jnp
import numpy as _np


def _validate_device(device: _jax.Device | _jax.sharding.Sharding | None) -> None:
    if device is not None and not isinstance(device, (_jax.Device, _jax.sharding.Sharding)):
        raise TypeError("device must be a jax.Device, jax.sharding.Sharding, or None")


def from_numpy(
    array: _np.ndarray,
    *,
    copy: bool,
    device: _jax.Device | _jax.sharding.Sharding | None = None,
) -> _jax.Array:
    """Create a JAX array from a NumPy array with explicit copy and placement policy.

    Args:
        array: Source NumPy array to convert.
        copy: If ``True``, take an independent host snapshot before transfer so that
            later mutation of ``array`` cannot affect the result. If ``False``, allow
            JAX to alias the source buffer when the backend permits it.
        device: Device or sharding to place the result on. ``None`` uses JAX's
            default placement.

    Returns:
        A JAX array holding the converted data, placed according to ``device``.

    Raises:
        TypeError: If ``array`` is not a ``numpy.ndarray``, ``copy`` is not a
            ``bool``, or ``device`` is not a ``jax.Device``, ``jax.sharding.Sharding``,
            or ``None``.
    """
    if not isinstance(array, _np.ndarray):
        raise TypeError("array must be a numpy.ndarray")
    if not isinstance(copy, bool):
        raise TypeError("copy must be a bool")
    _validate_device(device)
    if copy:
        snapshot = _np.array(array, copy=True, order="K", subok=False)
        return _jnp.asarray(snapshot, copy=None, device=device)
    return _jnp.asarray(array, copy=copy, device=device)


def from_dlpack(
    array: object,
    *,
    copy: bool | None = None,
    device: _jax.Device | _jax.sharding.Sharding | None = None,
) -> _jax.Array:
    """Import a DLPack provider, completing ``copy=True`` before returning independent storage.

    Args:
        array: Object exposing the DLPack protocol to import.
        copy: Copy policy. ``True`` forces an independent copy and blocks until the
            transfer completes; ``False`` requests a zero-copy import; ``None`` lets
            the backend choose.
        device: Device or sharding for the imported array. ``None`` uses JAX's
            default placement.

    Returns:
        A JAX array backed by the imported buffer. When ``copy`` is ``True`` the
        storage is guaranteed independent, including for same-device JAX providers.

    Raises:
        TypeError: If ``copy`` is not a ``bool`` or ``None``, or ``device`` is not a
            ``jax.Device``, ``jax.sharding.Sharding``, or ``None``.
    """
    if copy is not None and not isinstance(copy, bool):
        raise TypeError("copy must be a bool or None")
    _validate_device(device)
    imported = _jax.dlpack.from_dlpack(array, device=device, copy=copy)
    if copy:
        copied = _jnp.array(imported, copy=True, device=device)
        copied.block_until_ready()
        return copied
    return imported


def to_numpy(array: _jax.Array, *, writable: bool) -> _np.ndarray:
    """Synchronize a fully addressable JAX array and return a writable copy or read-only host value.

    Blocks until the array is ready, then materializes its data on the host.

    Args:
        array: Fully addressable JAX array to materialize on the host.
        writable: If ``True``, return an independent, writable copy. If ``False``,
            return a read-only NumPy value whose ``writeable`` flag is cleared.

    Returns:
        The array's data as a ``numpy.ndarray``. Writable results are independent
        copies; read-only results are a synchronized host view.

    Raises:
        TypeError: If ``array`` is not a ``jax.Array`` or ``writable`` is not a
            ``bool``.
        ValueError: If ``array`` is not fully addressable on the current process.
    """
    if not isinstance(array, _jax.Array):
        raise TypeError("array must be a jax.Array")
    if not isinstance(writable, bool):
        raise TypeError("writable must be a bool")
    if not array.is_fully_addressable:
        raise ValueError("host materialization requires a fully addressable array")
    array.block_until_ready()
    host = _np.asarray(array)
    if writable:
        return host.copy()
    host.flags.writeable = False
    return host
