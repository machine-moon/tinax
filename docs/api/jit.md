# Jit

`tinax.jit` wraps `jax.jit` with a Chex trace budget validated at wrap time, an optional mesh context, and a `jax.vmap` fusion for batching. Together `bounded_jit` and `batched_jit` are the JAX-native, auto-parallel replacement for legacy `pmap`-style batching; combine `batched_jit` with a Manual-axis `tinax.parallel.shard_mapped` for the manual-SPMD alternative.

```python
from tinax.jit import bounded_jit

train_step = bounded_jit(step, max_traces=1)
```

`static_argnames`/`donate_argnames` overlap or duplication is caught immediately, not on an arbitrary later call.

::: tinax.jit.bounded_jit

::: tinax.jit.batched_jit
