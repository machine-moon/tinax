"""JAX runtime device information within what the backend already exposes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable summary of the active JAX runtime backend and its visible devices."""

    backend: str
    device_count: int
    local_device_count: int
    process_index: int
    process_count: int
    device_kinds: tuple[str, ...]


def device_info() -> DeviceInfo:
    """Summarize the active JAX runtime backend and its visible devices.

    Returns:
        A :class:`DeviceInfo` describing the default backend, global and local device
        counts, this process's index and the process count, and the visible device
        kinds. Importing this module stays inert; JAX is imported only when called.
    """
    import jax as _jax

    return DeviceInfo(
        backend=_jax.default_backend(),
        device_count=_jax.device_count(),
        local_device_count=_jax.local_device_count(),
        process_index=_jax.process_index(),
        process_count=_jax.process_count(),
        device_kinds=tuple(device.device_kind for device in _jax.devices()),
    )
