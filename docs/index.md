# Tinax

## Reliable boundaries for JAX applications

Tinax is a typed library for the JAX ecosystem. It turns high-consequence boundaries into small explicit calls: copying arrays, deriving keys, placing data, managing workers, checkpointing state, and reading weights.

```python
import numpy as np

from tinax.array import from_numpy, inspect_array, to_numpy

device = from_numpy(np.arange(8, dtype=np.float32), copy=True)
info = inspect_array(device)
host = to_numpy(device, writable=False)
```

`copy=True` is visible. Host materialization is visible. The caller decides when those costs and ownership changes occur.

## What Tinax Owns

| module | Use it for |
| --- | --- |
| [`array`](api/array.md) | NumPy, JAX, DLPack, host copies, logical array inspection, and validated array operations |
| [`jit`](api/jit.md) | Trace-budgeted compilation, an optional mesh context, and vmap batching |
| [`grad`](api/grad.md) | Hardened autodiff: value_and_grad, an explicit forward/reverse jacobian, and hessian |
| [`data`](api/data.md) | Read/Write interface, dataset splitting, deterministic input pipelines, and multiprocessing worker lifetime |
| [`random`](api/random.md) | Typed JAX key validation, coordinate derivation, and key ownership |
| [`debug`](api/debug.md) | Bounded host observation, completed profiler-call scopes, and sharding visualization |
| [`nn`](api/nn.md) | Independent Flax NNX graph snapshots, restoration, and copies |
| [`stdlib`](api/stdlib.md) | Explicit argparse conversion and isolated stream loggers |
| [`checkpointing`](api/checkpointing.md) | Atomic Orbax V1 checkpointables and explicit restore targets |
| [`parallel`](api/parallel.md) | Meshes, layouts, shard_map, placement, process-local data, and NNX integration |
| [`weights`](api/weights.md) | Tensor manifests and bounded Safetensors interchange |

`examples/` contains tested recipes, not stable APIs. Importing `tinax` alone does not initialize JAX or optional integrations.

## Start Here

1. Read [Installation](installation.md) to select the supported Python and JAX environment.
2. Read [Design Principles](design.md) for the explicit-policy model and a worked example for every module in the table above.
3. Consult the [Public API reference](api/array.md) for each module's exact signatures and contracts.
