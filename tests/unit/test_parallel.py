from collections.abc import Callable
from typing import Literal, cast
from unittest.mock import Mock

import jax
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

import tinax.parallel as sharding_api
from tinax.parallel import (
    addressable_indices,
    addressable_payload_nbytes,
    from_process_local_data,
    layout,
    logical_payload_nbytes,
    make_mesh,
    place_host_array,
    shard_mapped,
)


@pytest.fixture
def cpu_devices() -> tuple[jax.Device, ...]:
    devices = tuple(jax.devices("cpu"))
    assert devices
    return devices


@pytest.fixture
def axis_name() -> str:
    return "items"


@pytest.fixture
def explicit_mesh(cpu_devices: tuple[jax.Device, ...], axis_name: str) -> Mesh:
    return make_mesh(
        cpu_devices,
        (len(cpu_devices),),
        (axis_name,),
        axis_types=(AxisType.Explicit,),
    )


@pytest.fixture
def manual_mesh(cpu_devices: tuple[jax.Device, ...], axis_name: str) -> Mesh:
    return make_mesh(
        cpu_devices,
        (len(cpu_devices),),
        (axis_name,),
        axis_types=(AxisType.Manual,),
    )


def _simulated_mesh(process_grid: tuple[tuple[int, ...], ...]) -> Mesh:
    reference = jax.devices("cpu")[0]
    devices: list[jax.Device] = []
    for device_id, process_index in enumerate(process for row in process_grid for process in row):
        device = Mock(spec=jax.Device)
        device.configure_mock(
            id=device_id,
            process_index=process_index,
            platform=reference.platform,
            device_kind=reference.device_kind,
            client=reference.client,
            addressable=process_index == jax.process_index(),
        )
        devices.append(cast(jax.Device, device))
    return Mesh(
        np.asarray(devices, dtype=object).reshape(len(process_grid), len(process_grid[0])),
        ("rows", "columns"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def test_regular_package_exports_are_available() -> None:
    assert sharding_api.make_mesh is make_mesh
    assert sharding_api.layout is layout
    assert sharding_api.shard_mapped is shard_mapped
    assert not hasattr(sharding_api, "addressable_buffer_nbytes")


def test_make_mesh_uses_exact_caller_devices_and_axis_policy(
    cpu_devices: tuple[jax.Device, ...], axis_name: str
) -> None:
    mesh = make_mesh(
        cpu_devices,
        (len(cpu_devices), 1),
        (axis_name, "singleton"),
        axis_types=(AxisType.Explicit, AxisType.Auto),
    )

    assert mesh.shape == {axis_name: len(cpu_devices), "singleton": 1}
    assert mesh.axis_types == (AxisType.Explicit, AxisType.Auto)
    assert set(mesh.devices.flat) == set(cpu_devices)


def test_make_mesh_validates_devices_shape_names_and_axis_types(
    cpu_devices: tuple[jax.Device, ...], axis_name: str
) -> None:
    with pytest.raises(ValueError, match="every provided device"):
        make_mesh(
            cpu_devices,
            (len(cpu_devices) + 1,),
            (axis_name,),
            axis_types=(AxisType.Explicit,),
        )
    with pytest.raises(ValueError, match="duplicates"):
        make_mesh(
            (cpu_devices[0], cpu_devices[0]),
            (2,),
            (axis_name,),
            axis_types=(AxisType.Explicit,),
        )
    with pytest.raises(ValueError, match="unique"):
        make_mesh(
            cpu_devices,
            (len(cpu_devices), 1),
            (axis_name, axis_name),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
    with pytest.raises(ValueError, match="same length"):
        make_mesh(
            cpu_devices,
            (len(cpu_devices),),
            (axis_name,),
            axis_types=(),
        )
    with pytest.raises(TypeError, match="integers"):
        make_mesh(
            cpu_devices,
            (True,),
            (axis_name,),
            axis_types=(AxisType.Explicit,),
        )


def test_layout_and_placement_cover_partitioning_and_replication(
    cpu_devices: tuple[jax.Device, ...], explicit_mesh: Mesh, axis_name: str
) -> None:
    rows = len(cpu_devices) * 2
    source = np.arange(rows * 3, dtype=np.float32).reshape(rows, 3)
    partitioned = place_host_array(source, layout(explicit_mesh, (axis_name, None)), copy=False)
    partitioned.block_until_ready()

    assert isinstance(partitioned.sharding, NamedSharding)
    assert partitioned.sharding.spec == PartitionSpec(axis_name, None)
    assert len(partitioned.addressable_shards) == len(cpu_devices)
    assert {shard.data.shape for shard in partitioned.addressable_shards} == {(2, 3)}
    assert len(addressable_indices(partitioned)) == len(cpu_devices)
    assert logical_payload_nbytes(partitioned) == source.nbytes
    assert addressable_payload_nbytes(partitioned) == source.nbytes

    replicated = place_host_array(source, layout(explicit_mesh, (None, None)), copy=False)
    replicated.block_until_ready()
    assert isinstance(replicated.sharding, NamedSharding)
    assert replicated.sharding.spec == PartitionSpec(None, None)
    assert addressable_payload_nbytes(replicated) == source.nbytes * len(cpu_devices)


def test_payload_sizes_are_unknown_for_typed_keys(explicit_mesh: Mesh) -> None:
    key = jax.device_put(jax.random.key(0), layout(explicit_mesh, ()))

    assert jax.dtypes.issubdtype(key.dtype, jax.dtypes.extended)
    assert logical_payload_nbytes(key) is None
    assert addressable_payload_nbytes(key) is None


def test_layout_infers_rank_and_rejects_auto_and_manual_axes(
    cpu_devices: tuple[jax.Device, ...], explicit_mesh: Mesh, axis_name: str
) -> None:
    assert layout(explicit_mesh, ()).spec == PartitionSpec()
    assert layout(explicit_mesh, (axis_name, None)).spec == PartitionSpec(axis_name, None)

    auto_mesh = make_mesh(
        cpu_devices,
        (len(cpu_devices),),
        (axis_name,),
        axis_types=(AxisType.Auto,),
    )
    with pytest.raises(ValueError, match="Auto"):
        layout(auto_mesh, (axis_name,))

    manual_mesh = make_mesh(
        cpu_devices,
        (len(cpu_devices),),
        (axis_name,),
        axis_types=(AxisType.Manual,),
    )
    with pytest.raises(ValueError, match="Manual"):
        layout(manual_mesh, (axis_name,))


def test_layout_rejects_unknown_and_reused_axes(explicit_mesh: Mesh, axis_name: str) -> None:
    with pytest.raises(ValueError, match="unknown"):
        layout(explicit_mesh, ("missing",))
    with pytest.raises(ValueError, match="at most one"):
        layout(explicit_mesh, (axis_name, axis_name))


def test_host_copy_snapshots_before_asynchronous_placement(explicit_mesh: Mesh, axis_name: str) -> None:
    rows = explicit_mesh.shape[axis_name] * 2
    source = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
    expected = source.copy()

    placed = place_host_array(source, layout(explicit_mesh, (axis_name, None)), copy=True)
    source.fill(-1)
    placed.block_until_ready()

    np.testing.assert_array_equal(np.asarray(placed), expected)


def test_process_local_data_validates_addressable_extents_and_policy(explicit_mesh: Mesh, axis_name: str) -> None:
    rows = explicit_mesh.shape[axis_name] * 2
    array_layout = layout(explicit_mesh, (axis_name, None))
    invalid_policy = cast(Literal["disallow", "assume_consistent"], "unchecked")

    with pytest.raises(ValueError, match="addressable extent"):
        from_process_local_data(
            np.ones((rows - 1, 3), dtype=np.float32),
            array_layout,
            (rows, 3),
            copy=True,
            replica_policy="disallow",
        )
    with pytest.raises(ValueError, match="same rank"):
        from_process_local_data(
            np.ones((rows,), dtype=np.float32),
            array_layout,
            (rows, 3),
            copy=True,
            replica_policy="disallow",
        )
    with pytest.raises(ValueError, match="replica_policy"):
        from_process_local_data(
            np.ones((rows, 3), dtype=np.float32),
            array_layout,
            (rows, 3),
            copy=True,
            replica_policy=invalid_policy,
        )


def test_process_identity_is_checked_only_when_supplied(
    monkeypatch: pytest.MonkeyPatch, explicit_mesh: Mesh, axis_name: str
) -> None:
    rows = explicit_mesh.shape[axis_name] * 2
    source = np.arange(rows * 2, dtype=np.int32).reshape(rows, 2)
    array_layout = layout(explicit_mesh, (axis_name, None))

    def unexpected_process_lookup() -> int:
        raise AssertionError("process identity should not be queried")

    with monkeypatch.context() as context:
        context.setattr(jax, "process_index", unexpected_process_lookup)
        result = from_process_local_data(
            source,
            array_layout,
            source.shape,
            copy=True,
            replica_policy="disallow",
        )
    np.testing.assert_array_equal(np.asarray(result), source)

    with pytest.raises(RuntimeError, match="process_index"):
        from_process_local_data(
            source,
            array_layout,
            source.shape,
            copy=True,
            replica_policy="disallow",
            process_index=jax.process_index() + 1,
        )


def test_process_local_extent_validation_handles_sparse_and_replicated_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagonal_processes = _simulated_mesh(((0, 1), (1, 0)))
    tiled = NamedSharding(diagonal_processes, PartitionSpec("rows", "columns"))
    with pytest.raises(ValueError, match="dimension 0.*addressable extent 8"):
        from_process_local_data(
            np.ones((4, 4), dtype=np.float32),
            tiled,
            (8, 8),
            copy=True,
            replica_policy="disallow",
        )

    grouped_processes = _simulated_mesh(((0, 0), (1, 1)))
    row_partitioned = NamedSharding(grouped_processes, PartitionSpec("rows", None))
    with pytest.raises(ValueError, match="dimension 1.*addressable extent 8"):
        from_process_local_data(
            np.ones((4, 4), dtype=np.float32),
            row_partitioned,
            (8, 8),
            copy=True,
            replica_policy="disallow",
        )

    expected = jax.device_put(np.zeros(1, dtype=np.float32))

    def make_process_local_array(
        _: NamedSharding,
        __: np.ndarray,
        global_shape: tuple[int, ...],
    ) -> jax.Array:
        assert global_shape == (8, 8)
        return expected

    monkeypatch.setattr(jax, "make_array_from_process_local_data", make_process_local_array)
    result = from_process_local_data(
        np.ones((4, 8), dtype=np.float32),
        row_partitioned,
        (8, 8),
        copy=True,
        replica_policy="disallow",
    )
    assert result is expected


def test_process_local_replica_policy_attests_consistency_without_gathering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _simulated_mesh(((0, 1), (0, 1)))
    array_layout = NamedSharding(mesh, PartitionSpec("rows", None))
    source = np.arange(64, dtype=np.int32).reshape(8, 8)

    with pytest.raises(ValueError, match="assume_consistent"):
        from_process_local_data(
            source,
            array_layout,
            source.shape,
            copy=True,
            replica_policy="disallow",
        )

    expected = jax.device_put(source)
    calls: list[tuple[NamedSharding, np.ndarray, tuple[int, ...]]] = []

    def make_without_gather(
        sharding: NamedSharding,
        local_data: np.ndarray,
        global_shape: tuple[int, ...],
    ) -> jax.Array:
        calls.append((sharding, local_data, global_shape))
        return expected

    monkeypatch.setattr(jax, "make_array_from_process_local_data", make_without_gather)
    result = from_process_local_data(
        source,
        array_layout,
        source.shape,
        copy=True,
        replica_policy="assume_consistent",
    )
    assert result is expected
    assert len(calls) == 1
    called_sharding, called_data, called_shape = calls[0]
    assert called_sharding is array_layout
    assert called_shape == source.shape
    np.testing.assert_array_equal(called_data, source)
    assert called_data is not source


def test_process_local_copy_allows_immediate_source_mutation(explicit_mesh: Mesh, axis_name: str) -> None:
    rows = explicit_mesh.shape[axis_name] * 2
    source = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
    expected = source.copy()

    result = from_process_local_data(
        source,
        layout(explicit_mesh, (axis_name, None)),
        source.shape,
        copy=True,
        replica_policy="disallow",
    )
    source.fill(-1)
    result.block_until_ready()

    np.testing.assert_array_equal(np.asarray(result), expected)


def test_process_local_data_supports_scalars_and_empty_dimensions(explicit_mesh: Mesh, axis_name: str) -> None:
    scalar = from_process_local_data(
        np.array(7, dtype=np.int32),
        layout(explicit_mesh, ()),
        (),
        copy=True,
        replica_policy="disallow",
    )
    assert scalar.shape == ()
    assert int(scalar) == 7

    source = np.empty((0, 3), dtype=np.float32)
    empty = from_process_local_data(
        source,
        layout(explicit_mesh, (axis_name, None)),
        source.shape,
        copy=True,
        replica_policy="disallow",
    )
    assert empty.shape == (0, 3)
    assert logical_payload_nbytes(empty) == 0
    assert addressable_payload_nbytes(empty) == 0


def test_shard_mapped_delegates_to_jax_shard_map_with_validated_arguments(
    monkeypatch: pytest.MonkeyPatch, manual_mesh: Mesh, axis_name: str
) -> None:
    calls: list[dict[str, object]] = []
    sentinel = object()

    def fake_shard_map(function: object, **kwargs: object) -> object:
        calls.append({"function": function, **kwargs})
        return sentinel

    monkeypatch.setattr(jax, "shard_map", fake_shard_map)

    def body(x: jax.Array) -> jax.Array:
        return x

    in_specs = PartitionSpec(axis_name)
    out_specs = PartitionSpec(axis_name)
    result = shard_mapped(body, mesh=manual_mesh, in_specs=in_specs, out_specs=out_specs)

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0] == {
        "function": body,
        "mesh": manual_mesh,
        "in_specs": in_specs,
        "out_specs": out_specs,
        "axis_names": frozenset(),
        "check_vma": True,
    }


def test_shard_mapped_rejects_wrong_categories(manual_mesh: Mesh) -> None:
    with pytest.raises(TypeError, match="callable"):
        shard_mapped(
            cast(Callable[[jax.Array], jax.Array], None),
            mesh=manual_mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
        )
    with pytest.raises(TypeError, match="jax.sharding.Mesh"):
        shard_mapped(lambda x: x, mesh=cast(Mesh, object()), in_specs=PartitionSpec(), out_specs=PartitionSpec())
    with pytest.raises(TypeError, match="frozenset"):
        shard_mapped(
            lambda x: x,
            mesh=manual_mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
            axis_names=cast(frozenset[str], {1, 2}),
        )
    with pytest.raises(TypeError, match="check_vma must be a bool"):
        shard_mapped(
            lambda x: x,
            mesh=manual_mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
            check_vma=cast(bool, 1),
        )


def test_shard_mapped_rejects_an_empty_mesh(cpu_devices: tuple[jax.Device, ...]) -> None:
    empty_mesh = Mesh(np.empty((0,), dtype=object), ("items",), axis_types=(AxisType.Manual,))
    with pytest.raises(ValueError, match="not be empty"):
        shard_mapped(lambda x: x, mesh=empty_mesh, in_specs=PartitionSpec(), out_specs=PartitionSpec())


def test_shard_mapped_rejects_axis_names_outside_the_mesh(manual_mesh: Mesh) -> None:
    with pytest.raises(ValueError, match="subset"):
        shard_mapped(
            lambda x: x,
            mesh=manual_mesh,
            in_specs=PartitionSpec(),
            out_specs=PartitionSpec(),
            axis_names=frozenset({"ghost"}),
        )


def test_shard_mapped_rejects_non_manual_mesh_axes(explicit_mesh: Mesh, axis_name: str) -> None:
    with pytest.raises(ValueError, match="Manual"):
        shard_mapped(
            lambda x: x,
            mesh=explicit_mesh,
            in_specs=PartitionSpec(axis_name),
            out_specs=PartitionSpec(axis_name),
        )


def test_shard_mapped_rejects_specs_referencing_axes_outside_the_effective_manual_set(
    cpu_devices: tuple[jax.Device, ...], axis_name: str
) -> None:
    mesh = make_mesh(
        cpu_devices,
        (len(cpu_devices), 1),
        (axis_name, "other"),
        axis_types=(AxisType.Manual, AxisType.Explicit),
    )
    with pytest.raises(ValueError, match="outside the effective manual set"):
        shard_mapped(
            lambda x: x,
            mesh=mesh,
            in_specs=PartitionSpec(axis_name),
            out_specs=PartitionSpec("other"),
            axis_names=frozenset({axis_name}),
        )


def test_shard_mapped_rejects_a_non_partition_spec_leaf(manual_mesh: Mesh, axis_name: str) -> None:
    with pytest.raises(TypeError, match="PartitionSpec"):
        shard_mapped(
            lambda x: x,
            mesh=manual_mesh,
            in_specs=cast(PartitionSpec, "not-a-spec"),
            out_specs=PartitionSpec(axis_name),
        )
