"""Observe non-finite array elements without changing the value."""

import jax
import jax.numpy as jnp

from tinax.debug import observe_nonfinite


def main() -> None:
    """Report the count and bounded indices of non-finite elements from inside a jitted call."""

    def report(count: jax.Array, indices: jax.Array) -> None:
        print(f"non-finite count={int(count)} flat_indices={jax.device_get(indices).tolist()}")

    @jax.jit
    def normalize(values: jax.Array) -> jax.Array:
        observed = observe_nonfinite(values, callback=report, max_indices=4)
        return observed * 2.0

    values = jnp.asarray([1.0, jnp.inf, 3.0, jnp.nan])
    result = normalize(values)
    jax.block_until_ready(result)
    print(f"result={jax.device_get(result).tolist()}")


if __name__ == "__main__":
    main()
