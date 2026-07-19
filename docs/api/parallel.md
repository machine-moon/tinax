# Parallel

`tinax.parallel` is the home for JAX's multi-device story: meshes, explicit layouts, host placement, process-local arrays, Manual-axis `shard_map`, and Flax NNX sharding integration.

Process-local construction is not a gather. Callers must ensure replica consistency when their layout requires it. `layout` builds concrete Explicit-axis layouts; `shard_mapped` is the Manual-axis counterpart for callables that issue explicit collectives, validated against the mesh before staging.

See the [Distributed Placement](../design.md#distributed-placement) guide for task-oriented usage.

::: tinax.parallel.make_mesh

::: tinax.parallel.layout

::: tinax.parallel.shard_mapped

::: tinax.parallel.place_host_array

::: tinax.parallel.from_process_local_data

::: tinax.parallel.logical_payload_nbytes

::: tinax.parallel.addressable_payload_nbytes

::: tinax.parallel.addressable_indices

::: tinax.parallel.abstract_sharded_state

::: tinax.parallel.eager_sharded_init

::: tinax.parallel.state_shardings
