"""Build a mesh, lay out a sharded array, and construct a manual shard_map with tinax.parallel."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp

from tinax.parallel import layout, make_mesh, shard_mapped


def main() -> None:
    """Run a real sharded reduction over an Explicit mesh, then inspect a Manual shard_map."""
    devices = jax.devices()

    explicit = make_mesh(devices, (4,), ("data",), axis_types=(jax.sharding.AxisType.Explicit,))
    sharding = layout(explicit, ("data",))
    values = jax.device_put(jnp.arange(8, dtype=jnp.float32), sharding)
    print(f"sharded_sum={float(jax.jit(jnp.sum)(values))}")

    manual = make_mesh(devices, (4,), ("data",), axis_types=(jax.sharding.AxisType.Manual,))
    spec = jax.sharding.PartitionSpec("data")
    scaled = shard_mapped(lambda shard: shard * 2.0, mesh=manual, in_specs=spec, out_specs=spec)

    abstract = jax.ShapeDtypeStruct((8,), jnp.float32, sharding=jax.sharding.NamedSharding(manual, spec))
    traced = jax.eval_shape(scaled, abstract)
    print(f"shard_mapped_output shape={traced.shape} dtype={traced.dtype}")


if __name__ == "__main__":
    main()
