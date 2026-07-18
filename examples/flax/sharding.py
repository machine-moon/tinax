"""Eager NNX parameter partitioning under an explicit JAX mesh."""

import jax
from flax import nnx
from jax.sharding import Mesh, PartitionSpec

from tinax.sharding import layout as _layout


class ModelParallelLinear(nnx.Module):
    """Partition a linear kernel and output with caller-selected shardings."""

    def __init__(
        self,
        width: int,
        params_key: jax.Array,
        *,
        kernel_layout: tuple[str | tuple[str, ...] | None, ...],
        output_layout: tuple[str | tuple[str, ...] | None, ...],
    ) -> None:
        self.output_layout = output_layout
        self.projection = nnx.Linear(
            width,
            width,
            use_bias=False,
            kernel_metadata={"out_sharding": kernel_layout},
            rngs=nnx.Rngs(params=params_key),
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Apply the projection with an explicit output partition."""
        return self.projection(inputs, out_sharding=PartitionSpec(*self.output_layout))


@jax.jit(static_argnames=("width", "kernel_layout", "output_layout"))
def _initialize_model_parallel_linear(
    params_key: jax.Array,
    *,
    width: int,
    kernel_layout: tuple[str | tuple[str, ...] | None, ...],
    output_layout: tuple[str | tuple[str, ...] | None, ...],
) -> ModelParallelLinear:
    return ModelParallelLinear(
        width,
        params_key,
        kernel_layout=kernel_layout,
        output_layout=output_layout,
    )


def initialize_model_parallel_linear(
    mesh: Mesh,
    params_key: jax.Array,
    *,
    width: int,
    model_axis: str,
    kernel_layout: tuple[str | tuple[str, ...] | None, ...],
    output_layout: tuple[str | tuple[str, ...] | None, ...],
) -> ModelParallelLinear:
    """Validate caller policy and JIT-initialize eagerly partitioned parameters."""
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a jax.sharding.Mesh")
    if mesh.empty:
        raise ValueError("mesh must not be empty")
    if not isinstance(width, int) or isinstance(width, bool):
        raise TypeError("width must be an integer")
    if width < 1:
        raise ValueError("width must be positive")
    if not isinstance(model_axis, str):
        raise TypeError("model_axis must be a string")
    if not model_axis:
        raise ValueError("model_axis must be nonempty")
    if model_axis not in mesh.axis_names:
        raise ValueError("model_axis must name a mesh axis")

    for name, requested_layout in (
        ("kernel_layout", kernel_layout),
        ("output_layout", output_layout),
    ):
        if not isinstance(requested_layout, tuple):
            raise TypeError(f"{name} must be a tuple")
        if len(requested_layout) != 2:
            raise ValueError(f"{name} must contain two entries")

    kernel_sharding = _layout(mesh, kernel_layout)
    output_sharding = _layout(mesh, output_layout)
    for name, requested_layout in (
        ("kernel_layout", kernel_layout),
        ("output_layout", output_layout),
    ):
        final_entry = requested_layout[-1]
        if final_entry != model_axis and not (isinstance(final_entry, tuple) and model_axis in final_entry):
            raise ValueError(f"model_axis must partition the final dimension of {name}")

    try:
        kernel_sharding.shard_shape((width, width))
    except ValueError as error:
        raise ValueError("width is not compatible with kernel_layout") from error
    try:
        output_sharding.shard_shape((mesh.size, width))
    except ValueError as error:
        raise ValueError("width is not compatible with output_layout") from error

    with jax.set_mesh(mesh), nnx.use_eager_sharding(True):
        return _initialize_model_parallel_linear(
            params_key,
            width=width,
            kernel_layout=kernel_layout,
            output_layout=output_layout,
        )
