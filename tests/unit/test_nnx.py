from typing import cast

import jax.numpy as jnp
import pytest
from flax import nnx

from tinax.nnx import clone_graph, restore_graph, snapshot_graph


class Accumulator(nnx.Module):
    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.asarray(0, dtype=jnp.int32))


class SharedAccumulator(nnx.Module):
    def __init__(self) -> None:
        shared = Accumulator()
        self.left = shared
        self.right = shared


def test_graph_snapshot_restores_type_state_and_aliases() -> None:
    module = SharedAccumulator()
    module.left.total[...] = 4

    snapshot = snapshot_graph(module)
    module.left.total[...] = 8
    restored = restore_graph(snapshot, copy=True)

    assert isinstance(snapshot, nnx.GraphState)
    assert isinstance(restored, SharedAccumulator)
    assert restored.left is restored.right
    assert restored.left is not module.left
    assert int(restored.left.total[...]) == 4
    assert int(module.left.total[...]) == 8


def test_clone_graph_has_independent_variables_and_preserved_aliases() -> None:
    module = SharedAccumulator()
    cloned = clone_graph(module)

    assert isinstance(cloned, SharedAccumulator)
    assert cloned is not module
    assert cloned.left is cloned.right
    assert cloned.left is not module.left
    assert cloned.left.total is not module.left.total

    cloned.left.total[...] = 9
    assert int(cloned.right.total[...]) == 9
    assert int(module.left.total[...]) == 0


def test_snapshot_does_not_mutate_the_source_module() -> None:
    module = SharedAccumulator()
    snapshot_graph(module)

    module.left.total[...] = 3
    assert module.left is module.right
    assert int(module.right.total[...]) == 3


def test_nnx_lifecycle_validates_object_categories_and_copy_policy() -> None:
    with pytest.raises(TypeError, match="nnx.Module"):
        snapshot_graph(cast(nnx.Module, object()))
    with pytest.raises(TypeError, match="nnx.Module"):
        clone_graph(cast(nnx.Module, object()))
    with pytest.raises(TypeError, match="GraphState"):
        restore_graph(cast(nnx.GraphState[nnx.Module], object()), copy=True)

    snapshot = snapshot_graph(Accumulator())
    with pytest.raises(TypeError, match="boolean"):
        restore_graph(snapshot, copy=cast(bool, 1))
