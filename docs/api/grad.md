# Grad

`tinax.grad` covers JAX's autodiff surface with hardened `argnums` validation shared across all three transforms: `value_and_grad`, `jacobian` (an explicit `mode="forward"` or `"reverse"` in place of separate `jacfwd`/`jacrev` calls), and `hessian`.

```python
from tinax.grad import value_and_grad

loss_and_grad = value_and_grad(loss)
value, gradient = loss_and_grad(params, batch)
```

Raw `jax.grad` silently accepts a boolean `argnums`, differentiating with respect to the wrong argument with no error. `tinax.grad` rejects it, along with duplicate and empty selections.

::: tinax.grad.value_and_grad

::: tinax.grad.jacobian

::: tinax.grad.hessian
