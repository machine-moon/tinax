"""Snapshot, restore, and clone Flax NNX graph state with tinax.nn."""

import jax.numpy as jnp
from flax import nnx

from tinax.nn import clone_graph, restore_graph, snapshot_graph


class Counter(nnx.Module):
    """Hold one mutable scalar in NNX state."""

    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.zeros((), dtype=jnp.float32))

    def add(self, increment: float) -> None:
        """Accumulate one scalar into the running total."""
        self.total[...] = self.total[...] + increment


def main() -> None:
    """Capture an independent snapshot, mutate the live module, then restore and clone it."""
    counter = Counter()
    counter.add(5.0)

    snapshot = snapshot_graph(counter)
    counter.add(100.0)
    print(f"live_total={float(counter.total[...])}")

    restored = restore_graph(snapshot, copy=True)
    print(f"restored_total={float(restored.total[...])}")

    twin = clone_graph(counter)
    twin.add(1.0)
    print(f"clone_total={float(twin.total[...])} original_total={float(counter.total[...])}")


if __name__ == "__main__":
    main()
