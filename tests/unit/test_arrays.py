from dataclasses import FrozenInstanceError
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

import tinax.arrays as arrays
import tinax.arrays._conversion as conversion_module
from tinax.arrays import ArrayInfo, from_numpy, inspect_array, to_numpy


def test_arrays_package_has_only_intentional_public_exports() -> None:
    public_names = {name for name in vars(arrays) if not name.startswith("_")}
    assert public_names == {"ArrayInfo", "from_dlpack", "from_numpy", "inspect_array", "to_numpy"}


def test_from_numpy_copy_true_snapshots_the_source() -> None:
    source = np.arange(6, dtype=np.float32)
    result = from_numpy(source, copy=True)
    result.block_until_ready()
    source[:] = -1
    np.testing.assert_array_equal(to_numpy(result, writable=False), np.arange(6, dtype=np.float32))


def test_from_numpy_copy_true_detaches_host_storage_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    source = np.arange(8, dtype=np.int32)
    dispatched: list[np.ndarray] = []
    real_asarray = conversion_module._jnp.asarray

    def capture(array: np.ndarray, *, copy: bool | None, device: object) -> jax.Array:
        dispatched.append(array)
        return real_asarray(array, copy=copy, device=device)

    monkeypatch.setattr(conversion_module._jnp, "asarray", capture)
    from_numpy(source, copy=True)

    assert len(dispatched) == 1
    assert not np.shares_memory(dispatched[0], source)


def test_from_numpy_copy_false_preserves_the_value_without_promising_aliasing() -> None:
    source = np.arange(5, dtype=np.int32)
    result = from_numpy(source, copy=False)
    np.testing.assert_array_equal(to_numpy(result, writable=False), source)


def test_from_numpy_accepts_device_and_supported_sharding_placement() -> None:
    source = np.arange(8, dtype=np.int32)
    device = jax.devices("cpu")[0]
    placed = from_numpy(source, copy=True, device=device)
    assert placed.committed
    assert placed.devices() == {device}

    mesh = Mesh(np.asarray(jax.devices("cpu"), dtype=object), ("data",))
    sharding = NamedSharding(mesh, PartitionSpec("data"))
    sharded = from_numpy(source, copy=True, device=sharding)
    assert sharded.sharding == sharding
    assert len(sharded.addressable_shards) == len(jax.devices("cpu"))
    np.testing.assert_array_equal(to_numpy(sharded, writable=False), source)


def test_from_numpy_rejects_wrong_categories_and_unsupported_dtypes() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        from_numpy(cast(np.ndarray, [1, 2]), copy=True)
    with pytest.raises(TypeError, match="copy must be a bool"):
        from_numpy(np.arange(2), copy=cast(bool, 1))
    with pytest.raises(TypeError, match="device must be"):
        from_numpy(np.arange(2), copy=True, device=cast(jax.Device, object()))
    with pytest.raises(TypeError):
        from_numpy(np.array(["PEP"]), copy=True)
    with pytest.raises(TypeError):
        from_numpy(np.array([object()], dtype=object), copy=True)


def test_to_numpy_exposes_read_only_or_writable_output_policy() -> None:
    array = jnp.arange(4, dtype=jnp.int32)
    read_only = to_numpy(array, writable=False)
    assert not read_only.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        read_only[0] = 99

    writable = to_numpy(array, writable=True)
    assert writable.flags.writeable
    writable[:] = -1
    np.testing.assert_array_equal(to_numpy(array, writable=False), np.arange(4, dtype=np.int32))


def test_to_numpy_rejects_wrong_categories() -> None:
    with pytest.raises(TypeError, match="jax.Array"):
        to_numpy(cast(jax.Array, np.arange(2)), writable=False)
    with pytest.raises(TypeError, match="writable must be a bool"):
        to_numpy(jnp.arange(2), writable=cast(bool, 1))


def test_to_numpy_rejects_non_fully_addressable_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    array = jnp.arange(2)
    monkeypatch.setattr(type(array), "is_fully_addressable", property(lambda _: False))
    with pytest.raises(ValueError, match="fully addressable"):
        to_numpy(array, writable=False)


def test_inspect_array_reports_nominal_logical_metadata() -> None:
    array = jax.device_put(jnp.ones((2, 3), dtype=jnp.float32), jax.devices("cpu")[0])
    info = inspect_array(array)
    assert info == ArrayInfo(
        shape=(2, 3),
        dtype=np.dtype(np.float32),
        logical_nbytes=24,
        committed=True,
        fully_addressable=True,
    )


def test_inspect_array_does_not_count_replicas_as_logical_payload() -> None:
    devices = jax.devices("cpu")
    mesh = Mesh(np.asarray(devices, dtype=object), ("replica",))
    replicated = NamedSharding(mesh, PartitionSpec())
    array = jax.device_put(np.arange(8, dtype=np.float32), replicated)
    info = inspect_array(array)
    assert info.logical_nbytes == 32
    assert sum(shard.data.nbytes for shard in array.addressable_shards) == 32 * len(devices)


def test_inspect_array_preserves_typed_key_dtype_without_inventing_payload_bytes() -> None:
    key = jax.random.key(0)
    info = inspect_array(key)
    assert info.shape == ()
    assert info.dtype == key.dtype
    assert not isinstance(info.dtype, np.dtype)
    assert info.logical_nbytes is None


def test_array_info_is_frozen_and_inspection_rejects_non_arrays() -> None:
    info = inspect_array(jnp.arange(2))
    with pytest.raises(FrozenInstanceError):
        setattr(info, "shape", ())
    with pytest.raises(TypeError, match="jax.Array"):
        inspect_array(cast(jax.Array, np.arange(2)))
