# Arrays

`tinax.array` owns NumPy, JAX, DLPack, host materialization, logical array inspection, and a small set of validated array operations. Its conversion functions make copying and ownership explicit; its operations close silent-failure gaps in the equivalent `jax.numpy`/`jax.nn` calls.

See the [Array Ownership](../design.md#array-ownership) guide for task-oriented usage.

::: tinax.array.from_numpy

::: tinax.array.from_dlpack

::: tinax.array.to_numpy

::: tinax.array.inspect_array

::: tinax.array.ArrayInfo

::: tinax.array.stack_batch

::: tinax.array.safe_astype

::: tinax.array.one_hot
