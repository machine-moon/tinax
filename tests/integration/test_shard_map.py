"""Real multi-device shard_mapped construction over the simulated CPU mesh.

Concrete invocation of a Manual-axis jax.shard_map on this pinned jax==0.11.0 CPU backend
hits a jaxlib lowering defect independent of tinax (confirmed by reproducing it with raw
jax.shard_map, bypassing shard_mapped entirely): every non-identity operation inside the
mapped function raises an MLIR shape-verification error regardless of jit or partitioner
backend. These tests instead use jax.eval_shape, which traces the mapped callable against
real multi-device mesh and sharding metadata without lowering to XLA, to verify shard_mapped
produces a well-formed, correctly-sharded callable over genuine device topology.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from tinax.parallel import make_mesh, shard_mapped


def _manual_mesh(axis_name: str) -> Mesh:
    devices = tuple(jax.devices("cpu"))
    assert devices
    return make_mesh(devices, (len(devices),), (axis_name,), axis_types=(AxisType.Manual,))


def test_shard_mapped_produces_a_well_formed_callable_over_a_real_manual_mesh() -> None:
    axis_name = "batch"
    mesh = _manual_mesh(axis_name)

    def per_shard_scale(values: jax.Array) -> jax.Array:
        return values * jnp.float32(2)

    mapped = shard_mapped(
        per_shard_scale,
        mesh=mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(axis_name),
    )

    sharding = NamedSharding(mesh, PartitionSpec(axis_name))
    global_shape = (len(jax.devices("cpu")) * 2,)
    abstract_input = jax.ShapeDtypeStruct(global_shape, jnp.float32, sharding=sharding)

    result = jax.eval_shape(mapped, abstract_input)

    assert result.shape == global_shape
    assert result.dtype == jnp.float32
    assert isinstance(result.sharding, NamedSharding)
    assert result.sharding.mesh.shape_tuple == mesh.shape_tuple


def test_shard_mapped_rejects_a_real_explicit_axis_mesh_before_any_tracing() -> None:
    axis_name = "batch"
    devices = tuple(jax.devices("cpu"))
    explicit_mesh = make_mesh(devices, (len(devices),), (axis_name,), axis_types=(AxisType.Explicit,))
    traced = False

    def per_shard_scale(values: jax.Array) -> jax.Array:
        nonlocal traced
        traced = True
        return values

    with pytest.raises(ValueError, match="Manual"):
        shard_mapped(
            per_shard_scale,
            mesh=explicit_mesh,
            in_specs=PartitionSpec(axis_name),
            out_specs=PartitionSpec(axis_name),
        )
    assert not traced


def test_shard_mapped_axis_names_subset_selects_manual_axes_over_a_real_mixed_mesh() -> None:
    devices = tuple(jax.devices("cpu"))
    assert len(devices) >= 2
    mesh = make_mesh(
        devices,
        (len(devices), 1),
        ("data", "model"),
        axis_types=(AxisType.Manual, AxisType.Explicit),
    )

    def identity(values: jax.Array) -> jax.Array:
        return values

    mapped = shard_mapped(
        identity,
        mesh=mesh,
        in_specs=PartitionSpec("data"),
        out_specs=PartitionSpec("data"),
        axis_names=frozenset({"data"}),
    )

    sharding = NamedSharding(mesh, PartitionSpec("data"))
    abstract_input = jax.ShapeDtypeStruct((len(devices),), jnp.float32, sharding=sharding)
    result = jax.eval_shape(mapped, abstract_input)

    assert result.shape == (len(devices),)
