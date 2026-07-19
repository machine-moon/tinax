# Design Principles

Tinax is deliberately narrow. It does not add a framework around JAX; it makes policies at ecosystem boundaries explicit.

## Explicit ownership

Copying, mutation, host materialization, and placement can alter correctness and cost. Tinax requires the caller to select these policies through arguments such as `copy`, `writable`, `overwrite`, and `device`.

## Explicit lifecycle

Workers, asynchronous checkpoint responses, and profiler traces have lifetimes. Tinax returns the objects that need completion and provides context-managed worker lifetime where that is the policy.

## Inert imports

`import tinax` has no JAX initialization or optional-integration side effects. Import a module only when its behavior is needed:

```python
from tinax.random import derive_process_step_key
```

## Stable versus examples

The Public API is the compatibility surface. `examples/` is tested but may change without a migration path. Use examples to learn upstream patterns, and use the Public API where Tinax adds validation or lifecycle policy.

A `tinax.lax` module for `scan`/`cond`/`while_loop` was evaluated for v0.1.3 and rejected: on the pinned JAX version, every one of those primitives already raises a specific, actionable `TypeError` for a carry dtype, shape, or pytree-structure mismatch. A tinax wrapper would duplicate the same check for the same exception type; revisit this decision if a future JAX release regresses those error messages.

## Error contracts

Public functions raise `TypeError` for the wrong category of object and `ValueError` for a valid category with an invalid value. Integer counts and steps reject booleans.

## Applying the principles

### Array Ownership

Use `tinax.array` when a value crosses a NumPy, JAX, or DLPack boundary and ownership must be clear.

```python
import numpy as np

from tinax.array import from_numpy, to_numpy

host = np.arange(4, dtype=np.float32)
device = from_numpy(host, copy=True)
host[0] = 99
```

`device` is created from a host snapshot. Choose `copy=False` only when aliasing and upstream conversion behavior are acceptable for the call site.

```python
readonly = to_numpy(device, writable=False)
writable = to_numpy(device, writable=True)
```

The read-only result is a synchronized host view. `writable=True` returns an independent host copy. Materialization rejects global arrays that are not fully addressable on the current process.

```python
from tinax.array import from_dlpack

independent = from_dlpack(producer, copy=True)
```

`copy=True` waits for transfer completion before returning and guarantees independent storage, including for same-device JAX producers.

`tinax.array` also validates a small set of operations where `jax.numpy`/`jax.nn` fail silently instead of raising.

```python
import jax.numpy as jnp

from tinax.array import safe_astype

narrowed = safe_astype(jnp.array([300], dtype=jnp.int32), jnp.int8, allow_lossy=True)
```

`jnp.astype` would otherwise wrap `300` to `44` without warning; `safe_astype` requires `allow_lossy=True` before permitting a cast that is not value-preserving.

### Compiled and Batched Execution

`tinax.jit` validates the recompilation and argument-name policy of `jax.jit` at wrap time, not on an arbitrary later call.

```python
from tinax.jit import bounded_jit

train_step = bounded_jit(step, max_traces=1, static_argnames=("mode",))
```

`static_argnames`/`donate_argnames` overlap or duplication raises immediately here; raw `jax.jit` only raises the first time a call actually hits the conflict, which can be deep inside a training loop. An optional `mesh` argument enters `jax.set_mesh` around every call, so call sites never need to remember the context manager themselves.

`batched_jit` fuses `jax.vmap` with the same trace budget and mesh handling, and is the JAX-native, auto-parallel replacement for legacy `pmap`-style batching:

```python
from tinax.jit import batched_jit

batched_step = batched_jit(step, in_axes=(None, 0), max_traces=1)
```

Combine `batched_jit` with a Manual-axis `tinax.parallel.shard_mapped` for the manual-SPMD alternative to automatic partitioning.

### Differentiation

Raw `jax.grad` silently accepts a boolean `argnums`, differentiating with respect to the wrong argument instead of raising. `tinax.grad` shares one hardened `argnums` check across `value_and_grad`, `jacobian`, and `hessian`.

```python
from tinax.grad import jacobian, value_and_grad

loss_and_grad = value_and_grad(loss)
value, gradient = loss_and_grad(params, batch)

jacobian_fn = jacobian(residual, mode="reverse")
```

`jacobian` requires an explicit `mode`: raw JAX splits this into two separately named top-level functions, `jax.jacfwd` and `jax.jacrev`, with a real wide-versus-tall performance tradeoff that is easy to get backwards. Bare `grad` is not exposed; `value_and_grad` is a strict superset, so discard the value when only the gradient is needed.

### Deterministic Data Pipelines

`tinax.data` covers ArrayRecord and Parquet data formats, dataset splitting, and Grain's deterministic process-aware batch pipelines together.

```python
from tinax.data import read_array_record, split_by_column, training_batches

rows = read_array_record("dataset.arrayrecord")
train_rows, test_rows = split_by_column(rows, "split")
```

`read_array_record` raises `FileNotFoundError` for a missing path instead of the backend's internal `RuntimeError`. `split_by_column` requires every row's discriminator value to be one of the two given `values`; a plain `filter`-based split would silently drop any row holding a third value instead of surfacing the mismatch. `split_random` rejects a `train_ratio` of exactly `0` or `1`, since either produces a degenerately empty partition.

`read_array_record`/`write_array_record` lazily import `array-record`, an optional package installed separately, and raise a clear `ImportError` if it is absent instead of failing to import `tinax.data` at all. `read_parquet_record`/`write_parquet_record` cover the same contract for Parquet, backed by `pyarrow`, a core dependency; both pairs return the same random-access `grain.MapDataset`, so callers pick a record format without changing the rest of the pipeline. Unlike `write_array_record`, `write_parquet_record` takes a finite `Sequence`, not an `Iterable`: Parquet's columnar layout requires the full row set up front and cannot stream row by row.

### Deterministic Randomness

JAX keys are values, not hidden global state. Tinax makes coordinate order and consumed-key ownership explicit.

```python
import jax

from tinax.random import derive_process_step_key, split_key

root = jax.random.key(0)
step_key = derive_process_step_key(root, process_index=0, step=42)
next_root, operation_keys = split_key(step_key, count=2)
```

`derive_process_step_key` always folds in coordinates in process, step, stream order. `split_key` returns the continuation key first, followed by the operation keys that the caller consumes.

Python coordinates must be integers from `0` through `2**32 - 1`. Dynamic coordinates inside JIT must be unsigned scalar JAX values no wider than 32 bits. These limits prevent signed and oversized values from silently colliding under `jax.random.fold_in`.

### Training Checkpoints

Checkpoint all resume-critical values together. Tinax's Orbax V1 helpers keep destinations immutable and restoration targets explicit.

```python
from pathlib import Path

from tinax.checkpointing import save_checkpointables

response = save_checkpointables(
    Path("checkpoints/step-100"),
    {"params": params, "optimizer": optimizer_state},
)
response.result()
```

Each active process must invoke distributed checkpoint operations in the same order. Finish the returned asynchronous response before loading from the destination or shutting down.

For a training loop, use `TrainingCheckpoint`, `save_training_checkpoint`, and `load_training_checkpoint` to keep model state, optimizer state, iterator state, and metadata together.

The `checkpointing.legacy.v0` namespace is explicitly legacy-only. Do not use it for new checkpoints.

### Distributed Placement

Tinax makes mesh construction and data placement separate operations.

```python
import numpy as np
import jax
from jax.sharding import AxisType

from tinax.parallel import layout, make_mesh, place_host_array

mesh = make_mesh(jax.devices(), (len(jax.devices()),), ("data",), axis_types=(AxisType.Explicit,))
array_layout = layout(mesh, ("data",))
array = place_host_array(np.arange(8), array_layout)
```

`place_host_array` starts with a host value and places it according to an explicit layout. `from_process_local_data` is different: it constructs a global array from each process's local input and is not a gather. Callers are responsible for replica consistency.

Use `logical_payload_nbytes` for the logical value size and `addressable_payload_nbytes` for the physical data addressable by this process.

`layout` builds concrete layouts over Explicit mesh axes; `shard_mapped` is the Manual-axis counterpart for callables that issue explicit collectives.

```python
from jax.sharding import PartitionSpec

from tinax.parallel import shard_mapped

manual_mesh = make_mesh(jax.devices(), (len(jax.devices()),), ("data",), axis_types=(AxisType.Manual,))
collective_step = shard_mapped(
    train_step,
    mesh=manual_mesh,
    in_specs=PartitionSpec("data"),
    out_specs=PartitionSpec("data"),
)
```

`shard_mapped` requires every axis the call is manual over to already be `AxisType.Manual`, and validates that `in_specs`/`out_specs` reference only that axis set, before the callable is ever staged.

### Safetensors Interchange

Tinax's weight helpers exchange explicit host NumPy tensors. They never silently gather global JAX arrays.

```python
from tinax.weights import load_safetensors

loaded = load_safetensors(
    "model.safetensors",
    max_bytes=2 * 1024 * 1024 * 1024,
    names=("encoder.weight",),
)
weights = loaded.tensors
```

The loader inspects the manifest before materializing payloads, validates selected names, and rejects loads over the byte budget. `save_safetensors` validates host tensor layout and writes atomically; the destination is immutable unless `overwrite=True` is selected explicitly.
