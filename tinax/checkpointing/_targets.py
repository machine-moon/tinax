"""Explicit mixed-leaf restore targets for Orbax V1 checkpoints."""

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np


@dataclass(frozen=True, slots=True)
class _NumpyRestoreTarget:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]


def abstract_restore_target(state: Any) -> Any:
    """Replace JAX and NumPy arrays with shape/dtype targets while preserving JAX sharding and static leaves.

    Args:
        state: Arbitrary pytree of restore targets. JAX arrays become
            ``jax.ShapeDtypeStruct`` (retaining sharding); NumPy arrays become an
            internal shape/dtype target; other leaves are returned unchanged.

    Returns:
        A pytree matching ``state`` with concrete arrays replaced by abstract
        shape/dtype restore targets.
    """

    def abstract_leaf(leaf: object) -> object:
        if isinstance(leaf, jax.Array):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding)
        if isinstance(leaf, np.ndarray):
            return _NumpyRestoreTarget(tuple(leaf.shape), leaf.dtype)
        return leaf

    return jax.tree.map(abstract_leaf, state)
