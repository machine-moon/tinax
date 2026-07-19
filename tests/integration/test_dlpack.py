from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from tinax.array import from_dlpack, to_numpy


@pytest.mark.parametrize("copy", [None, False, True])
def test_from_dlpack_preserves_jax_tri_state_copy_values(copy: bool | None) -> None:
    source = jnp.arange(8, dtype=jnp.float32)
    result = from_dlpack(source, copy=copy)
    np.testing.assert_array_equal(to_numpy(result, writable=False), np.arange(8, dtype=np.float32))


def test_from_dlpack_accepts_a_numpy_provider() -> None:
    source = np.arange(6, dtype=np.int32)
    result = from_dlpack(source)
    np.testing.assert_array_equal(to_numpy(result, writable=False), source)


def test_from_dlpack_copy_true_is_independent_of_a_numpy_provider() -> None:
    source = np.zeros(8_000_000, dtype=np.int32)
    result = from_dlpack(source, copy=True)
    source[:] = 1

    assert np.count_nonzero(to_numpy(result, writable=False)) == 0


def test_from_dlpack_copy_true_strengthens_jax_0_11_same_device_behavior() -> None:
    source = jnp.arange(8, dtype=jnp.int32)
    source.block_until_ready()
    result = from_dlpack(source, copy=True)
    result.block_until_ready()

    assert not np.shares_memory(to_numpy(source, writable=False), to_numpy(result, writable=False))


def test_from_dlpack_copy_false_sharing_is_observable_without_private_pointers() -> None:
    source = jnp.arange(8, dtype=jnp.int32)
    source.block_until_ready()
    result = from_dlpack(source, copy=False)
    result.block_until_ready()
    source_host = to_numpy(source, writable=False)
    result_host = to_numpy(result, writable=False)
    assert np.shares_memory(source_host, result_host)


def test_from_dlpack_accepts_device_and_single_device_sharding() -> None:
    source = np.arange(4, dtype=np.float32)
    device = jax.devices("cpu")[0]
    for placement in (device, SingleDeviceSharding(device)):
        result = from_dlpack(source, copy=True, device=placement)
        assert result.committed
        assert result.devices() == {device}


def test_from_dlpack_preserves_copy_requirements_for_device_transfer() -> None:
    devices = jax.devices("cpu")
    source = jax.device_put(jnp.arange(4), devices[0])
    with pytest.raises(ValueError, match="requires a copy"):
        from_dlpack(source, copy=False, device=devices[1])
    transferred = from_dlpack(source, copy=None, device=devices[1])
    assert transferred.devices() == {devices[1]}


def test_from_dlpack_preserves_multi_device_sharding_error() -> None:
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("replica",))
    sharding = NamedSharding(mesh, PartitionSpec())
    with pytest.raises(ValueError, match="singular device"):
        from_dlpack(jnp.arange(4), copy=None, device=sharding)


def test_from_dlpack_preserves_provider_exceptions() -> None:
    class FailingProvider:
        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

        def __dlpack__(self, *, stream: int | None = None) -> object:
            raise RuntimeError("provider export failed")

    with pytest.raises(RuntimeError, match="provider export failed"):
        from_dlpack(FailingProvider(), copy=None)


def test_from_dlpack_rejects_wrong_argument_categories() -> None:
    with pytest.raises(TypeError, match="__dlpack__"):
        from_dlpack(object(), copy=None)
    with pytest.raises(TypeError, match="copy must be a bool or None"):
        from_dlpack(jnp.arange(2), copy=cast(bool | None, 1))
    with pytest.raises(TypeError, match="device must be"):
        from_dlpack(jnp.arange(2), copy=None, device=cast(jax.Device, object()))
