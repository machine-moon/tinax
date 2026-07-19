"""Release-gated accelerator and true multihost contracts."""

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

from tinax.array import from_numpy, to_numpy
from tinax.parallel import from_process_local_data, layout, make_mesh


@pytest.mark.accelerator
def test_explicit_accelerator_placement_round_trips() -> None:
    accelerators = [device for device in jax.local_devices() if device.platform in {"gpu", "tpu"}]
    if not accelerators:
        pytest.skip("requires a JAX GPU or TPU backend")
    source = np.arange(8, dtype=np.float32)

    placed = from_numpy(source, copy=True, device=accelerators[0])

    assert placed.devices() == {accelerators[0]}
    np.testing.assert_array_equal(to_numpy(placed, writable=False), source)


@pytest.mark.multihost
def test_process_local_construction_on_a_real_multihost_runtime() -> None:
    if jax.process_count() < 2:
        pytest.skip("requires a distributed JAX runtime with at least two processes")
    devices = jax.devices()
    local_devices = jax.local_devices()
    mesh = make_mesh(
        devices,
        (len(devices),),
        ("data",),
        axis_types=(jax.sharding.AxisType.Explicit,),
    )
    sharding = layout(mesh, ("data",))
    local_data = np.full((len(local_devices) * 2,), jax.process_index(), dtype=np.int32)

    array = from_process_local_data(
        local_data,
        sharding,
        (len(devices) * 2,),
        copy=True,
        replica_policy="disallow",
        process_index=jax.process_index(),
        process_count=jax.process_count(),
    )
    array.block_until_ready()

    assert array.shape == (len(devices) * 2,)
    for shard in array.addressable_shards:
        expected = np.full(shard.data.shape, jax.process_index(), dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(shard.data), expected)
    multihost_utils.sync_global_devices("tinax-process-local-contract")
