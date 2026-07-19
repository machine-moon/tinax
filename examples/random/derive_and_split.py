"""Derive and split typed PRNG keys without consuming global state."""

import jax

from tinax.random import derive_process_step_key, split_key


def main() -> None:
    """Derive a per-process/step key and split off a bounded set of operation keys."""
    root = jax.random.key(2027)

    step_key = derive_process_step_key(root, process_index=1, step=42)
    print(f"step_key_data={jax.random.key_data(step_key)}")

    continuation, operation_keys = split_key(step_key, count=3)
    samples = jax.random.normal(continuation, ())
    print(f"operation_keys_leading_dim={operation_keys.shape[0]} sample={samples}")


if __name__ == "__main__":
    main()
