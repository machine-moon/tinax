# Sharding

`tinax.sharding` owns meshes, explicit layouts, host placement, process-local arrays, and Flax NNX sharding integration.

Process-local construction is not a gather. Callers must ensure replica consistency when their layout requires it.

See the [Distributed Placement](../design.md#distributed-placement) guide for task-oriented usage.

::: tinax.sharding.make_mesh

::: tinax.sharding.layout

::: tinax.sharding.place_host_array

::: tinax.sharding.from_process_local_data

::: tinax.sharding.logical_payload_nbytes

::: tinax.sharding.addressable_payload_nbytes

::: tinax.sharding.addressable_indices

::: tinax.sharding.abstract_sharded_state

::: tinax.sharding.eager_sharded_init

::: tinax.sharding.state_shardings
