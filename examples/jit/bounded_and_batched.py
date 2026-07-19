"""Compile with a trace budget and batch with tinax.jit."""

import jax
import jax.numpy as jnp

from tinax.jit import batched_jit, bounded_jit


def main() -> None:
    """Trace-budget a scalar function, then vmap-and-jit it over a batch."""
    scaled = bounded_jit(lambda x: x * 2.0 + 1.0, max_traces=1)
    print(f"scaled(3.0)={scaled(jnp.asarray(3.0))}")
    print(f"scaled(4.0)={scaled(jnp.asarray(4.0))}")

    dot = batched_jit(lambda row: jnp.sum(row * row), max_traces=1)
    rows = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    print(f"row_norms={jax.device_get(dot(rows))}")


if __name__ == "__main__":
    main()
