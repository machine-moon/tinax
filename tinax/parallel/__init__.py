"""Meshes, layouts, shard_map, process-local placement, and Flax NNX integration."""

from tinax.parallel._arrays import from_process_local_data as from_process_local_data
from tinax.parallel._arrays import place_host_array as place_host_array
from tinax.parallel._inspection import addressable_indices as addressable_indices
from tinax.parallel._inspection import addressable_payload_nbytes as addressable_payload_nbytes
from tinax.parallel._inspection import logical_payload_nbytes as logical_payload_nbytes
from tinax.parallel._mesh import layout as layout
from tinax.parallel._mesh import make_mesh as make_mesh
from tinax.parallel._nnx import abstract_sharded_state as abstract_sharded_state
from tinax.parallel._nnx import eager_sharded_init as eager_sharded_init
from tinax.parallel._nnx import state_shardings as state_shardings
from tinax.parallel._shard_map import shard_mapped as shard_mapped
