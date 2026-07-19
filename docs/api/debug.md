# Debug

`tinax.debug` sends bounded summaries through injected JAX callbacks, keeps profiler scopes open until result arrays and staged effects complete, and validates sharding visualization. Callback execution remains an observational, asynchronous effect.

```python
from tinax.debug import profile_call

result = profile_call("profiles/step-10", train_step, state, batch)
```

::: tinax.debug.observe_nonfinite

::: tinax.debug.profile_call

::: tinax.debug.visualize_array_sharding
