# Diagnostics

`tinax.diagnostics` sends bounded summaries through injected JAX callbacks and keeps profiler scopes open until result arrays and staged effects complete. Callback execution remains an observational, asynchronous effect.

```python
from tinax.diagnostics import profile_call

result = profile_call("profiles/step-10", train_step, state, batch)
```

::: tinax.diagnostics.observe_nonfinite

::: tinax.diagnostics.profile_call
