# Tinax

## Reliable boundaries for JAX applications

Tinax is a typed library for the JAX ecosystem. It turns high-consequence boundaries into small explicit calls: copying arrays, deriving keys, placing data, managing workers, checkpointing state, and reading weights.

```python
import numpy as np

from tinax.arrays import from_numpy, inspect_array, to_numpy

device = from_numpy(np.arange(8, dtype=np.float32), copy=True)
info = inspect_array(device)
host = to_numpy(device, writable=False)
```

`copy=True` is visible. Host materialization is visible. The caller decides when those costs and ownership changes occur.

## What Tinax Owns

| Domain | Use it for |
| --- | --- |
| [`arrays`](api/arrays.md) | NumPy, JAX, DLPack, host copies, and logical array inspection |
| [`grain`](api/grain.md) | Deterministic input pipelines and multiprocessing worker lifetime |
| [`randomness`](api/randomness.md) | Typed JAX key validation, coordinate derivation, and key ownership |
| [`diagnostics`](api/diagnostics.md) | Bounded host observation and completed profiler-call scopes |
| [`nnx`](api/nnx.md) | Independent Flax NNX graph snapshots, restoration, and copies |
| [`stdlib`](api/stdlib.md) | Explicit argparse conversion and isolated stream loggers |
| [`checkpointing`](api/checkpointing.md) | Atomic Orbax V1 checkpointables and explicit restore targets |
| [`sharding`](api/sharding.md) | Meshes, layouts, placement, process-local data, and NNX integration |
| [`weights`](api/weights.md) | Tensor manifests and bounded Safetensors interchange |

`examples/` contains tested recipes, not stable APIs. Importing `tinax` alone does not initialize JAX or optional integrations.

## Start Here

1. Read [Installation](installation.md) to select the supported Python and JAX environment.
2. Read [Design Principles](design.md) for the explicit-policy model and a worked example for every domain in the table above.
3. Consult the [Public API reference](api/arrays.md) for each domain's exact signatures and contracts.
