"""Stable sharding primitives and Flax NNX integration."""

from tinax.sharding._arrays import from_process_local_data as from_process_local_data
from tinax.sharding._arrays import place_host_array as place_host_array
from tinax.sharding._inspection import addressable_indices as addressable_indices
from tinax.sharding._inspection import addressable_payload_nbytes as addressable_payload_nbytes
from tinax.sharding._inspection import logical_payload_nbytes as logical_payload_nbytes
from tinax.sharding._mesh import layout as layout
from tinax.sharding._mesh import make_mesh as make_mesh
from tinax.sharding._nnx import abstract_sharded_state as abstract_sharded_state
from tinax.sharding._nnx import eager_sharded_init as eager_sharded_init
from tinax.sharding._nnx import state_shardings as state_shardings
