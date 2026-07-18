from collections.abc import Callable, Mapping

import jax
import optax
import pytest
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

from tinax.sharding import (
    abstract_sharded_state,
    eager_sharded_init,
    make_mesh,
    state_shardings,
)


def _mesh(axis_name: str, axis_type: AxisType) -> Mesh:
    devices = tuple(jax.devices("cpu"))
    assert devices
    return make_mesh(
        devices,
        (len(devices),),
        (axis_name,),
        axis_types=(axis_type,),
    )


def _linear_initializer(width: int, axis_name: str) -> Callable[[], nnx.Linear]:
    kernel_init = nnx.with_partitioning(
        nnx.initializers.lecun_normal(),
        (None, axis_name),
        optimizer_sharding=(axis_name, None),
    )
    return lambda: nnx.Linear(
        width,
        width,
        use_bias=False,
        kernel_init=kernel_init,
        rngs=nnx.Rngs(0),
    )


def _named_sharding_leaves(state: Mapping[object, object]) -> list[NamedSharding]:
    return [leaf for leaf in jax.tree.leaves(state) if isinstance(leaf, NamedSharding)]


def test_eager_initialization_and_optimizer_state_have_independent_shardings() -> None:
    axis_name = "tensor"
    mesh = _mesh(axis_name, AxisType.Auto)
    width = len(mesh.devices.flat) * 2
    initialize_model = _linear_initializer(width, axis_name)

    def initialize_training_state() -> tuple[nnx.Linear, nnx.Optimizer]:
        model = initialize_model()
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        return model, optimizer

    model, optimizer = eager_sharded_init(initialize_training_state, mesh)
    model_state = nnx.unpack(model)
    optimizer_state = nnx.unpack(optimizer.opt_state)
    model_layouts = state_shardings(model_state, mesh)
    optimizer_layouts = state_shardings(optimizer_state, mesh)

    parameter_spec = PartitionSpec(None, axis_name)
    optimizer_spec = PartitionSpec(axis_name, None)
    assert model.kernel.get_value().sharding.spec == parameter_spec
    optimizer_arrays = [leaf for leaf in jax.tree.leaves(optimizer_state) if isinstance(leaf, jax.Array)]
    optimizer_matrices = [array for array in optimizer_arrays if array.ndim == 2]
    expected_optimizer_layout = NamedSharding(mesh, optimizer_spec)
    assert len(optimizer_matrices) == 2
    assert all(
        array.sharding.is_equivalent_to(expected_optimizer_layout, ndim=2) for array in optimizer_matrices
    )
    assert [sharding.spec for sharding in _named_sharding_leaves(model_layouts)] == [parameter_spec]
    optimizer_specs = [sharding.spec for sharding in _named_sharding_leaves(optimizer_layouts)]
    assert optimizer_specs.count(optimizer_spec) == 2
    assert PartitionSpec() in optimizer_specs
    assert parameter_spec != optimizer_spec

    mapping_layouts = state_shardings(dict(model_state), mesh)
    assert isinstance(mapping_layouts, dict)
    assert [sharding.spec for sharding in _named_sharding_leaves(mapping_layouts)] == [parameter_spec]


def test_abstract_state_uses_current_nnx_sharding_metadata() -> None:
    axis_name = "parameters"
    mesh = _mesh(axis_name, AxisType.Auto)
    width = len(mesh.devices.flat) * 2

    state = abstract_sharded_state(_linear_initializer(width, axis_name), mesh)
    leaves = jax.tree.leaves(state)

    assert isinstance(state, nnx.GraphState)
    assert len(leaves) == 1
    kernel = leaves[0]
    assert isinstance(kernel, jax.ShapeDtypeStruct)
    assert kernel.shape == (width, width)
    assert kernel.sharding.spec == PartitionSpec(None, axis_name)
    assert isinstance(nnx.merge(state), nnx.Linear)


def test_eager_initialization_restores_setting_and_supports_explicit_meshes() -> None:
    axis_name = "weights"
    mesh = _mesh(axis_name, AxisType.Explicit)
    width = len(mesh.devices.flat) * 2

    with nnx.use_eager_sharding(False):
        model = eager_sharded_init(_linear_initializer(width, axis_name), mesh)
        assert not nnx.using_eager_sharding()

    assert model.kernel.get_value().sharding.spec == PartitionSpec(None, axis_name)


def test_eager_initialization_rejects_manual_meshes() -> None:
    axis_name = "manual"
    mesh = _mesh(axis_name, AxisType.Manual)
    with pytest.raises(ValueError, match="Manual"):
        eager_sharded_init(_linear_initializer(len(mesh.devices.flat), axis_name), mesh)
