# Design Principles

Tinax is deliberately narrow. It does not add a framework around JAX; it makes policies at ecosystem boundaries explicit.

## Explicit ownership

Copying, mutation, host materialization, and placement can alter correctness and cost. Tinax requires the caller to select these policies through arguments such as `copy`, `writable`, `overwrite`, and `device`.

## Explicit lifecycle

Workers, asynchronous checkpoint responses, and profiler traces have lifetimes. Tinax returns the objects that need completion and provides context-managed worker lifetime where that is the policy.

## Inert imports

`import tinax` has no JAX initialization or optional-integration side effects. Import a domain only when its behavior is needed:

```python
from tinax.randomness import derive_process_step_key
```

## Stable versus examples

The Public API is the compatibility surface. `examples/` is tested but may change without a migration path. Use examples to learn upstream patterns, and use the Public API where Tinax adds validation or lifecycle policy.

## Error contracts

Public functions raise `TypeError` for the wrong category of object and `ValueError` for a valid category with an invalid value. Integer counts and steps reject booleans.

## Applying the principles

### Array Ownership

Use `tinax.arrays` when a value crosses a NumPy, JAX, or DLPack boundary and ownership must be clear.

```python
import numpy as np

from tinax.arrays import from_numpy, to_numpy

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
from tinax.arrays import from_dlpack

independent = from_dlpack(producer, copy=True)
```

`copy=True` waits for transfer completion before returning and guarantees independent storage, including for same-device JAX producers.

### Deterministic Randomness

JAX keys are values, not hidden global state. Tinax makes coordinate order and consumed-key ownership explicit.

```python
import jax

from tinax.randomness import derive_process_step_key, split_key

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

from tinax.sharding import layout, make_mesh, place_host_array

mesh = make_mesh(jax.devices(), axis_names=("data",))
array_layout = layout(mesh, ("data",))
array = place_host_array(np.arange(8), array_layout)
```

`place_host_array` starts with a host value and places it according to an explicit layout. `from_process_local_data` is different: it constructs a global array from each process's local input and is not a gather. Callers are responsible for replica consistency.

Use `logical_payload_nbytes` for the logical value size and `addressable_payload_nbytes` for the physical data addressable by this process.

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
