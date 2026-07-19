"""Differentiate with the shared argnums contract of tinax.grad."""

import jax
import jax.numpy as jnp

from tinax.grad import hessian, jacobian, value_and_grad


def main() -> None:
    """Take a value-and-gradient, a Jacobian, and a Hessian of small functions."""
    point = jnp.asarray([1.0, 2.0, 3.0])

    value, gradient = value_and_grad(lambda x: jnp.sum(x * x))(point)
    print(f"value={value} grad={gradient}")

    jac = jacobian(lambda x: jnp.stack([jnp.sum(x), jnp.prod(x)]), mode="reverse")(point)
    print(f"jacobian=\n{jac}")

    hess = hessian(lambda x: jnp.sum(x**3))(point)
    assert isinstance(hess, jax.Array)
    print(f"hessian_diagonal={jnp.diagonal(hess)}")


if __name__ == "__main__":
    main()
