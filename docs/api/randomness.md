# Randomness

`tinax.randomness` owns typed scalar key validation, ordered deterministic coordinate derivation, and explicit continuation-key ownership. It never discovers process identity or stores mutable RNG state.

Concrete Python coordinates must be between `0` and `2**32 - 1`. Dynamic coordinates inside JIT-compiled calls must be scalar unsigned JAX values of at most 32 bits, preventing silent `fold_in` collisions from signed or oversized values.

See the [Deterministic Randomness](../design.md#deterministic-randomness) guide for task-oriented usage.

```python
import jax

from tinax.randomness import derive_process_step_key, split_key

key = derive_process_step_key(jax.random.key(0), process_index=0, step=10)
next_key, batch_keys = split_key(key, count=8)
```

::: tinax.randomness.derive_key

::: tinax.randomness.derive_process_step_key

::: tinax.randomness.split_key
