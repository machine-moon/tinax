"""Validated Flax NNX graph-state lifecycle primitives."""

from typing import Any

from flax import nnx


def _module[T: nnx.Module](module: T) -> T:
    if not isinstance(module, nnx.Module):
        raise TypeError("module must be a flax.nnx.Module")
    return module


def snapshot_graph[T: nnx.Module](module: T) -> nnx.GraphState[T]:
    """Capture independent variable state while preserving the module's internal graph aliases.

    Args:
        module: Flax NNX module to snapshot.

    Returns:
        A ``flax.nnx.GraphState`` holding independent variable state with the
        module's internal aliases preserved.

    Raises:
        TypeError: If ``module`` is not a ``flax.nnx.Module``.
    """
    cloned: Any = nnx.clone(_module(module), variables=True, graph=True)
    return nnx.unpack(cloned, graph=True)


def restore_graph[T: nnx.Module](snapshot: nnx.GraphState[T], *, copy: bool) -> T:
    """Instantiate a graph snapshot with explicit NNX graph-copy policy.

    Args:
        snapshot: Graph state previously produced by ``snapshot_graph``.
        copy: If ``True``, copy variables while merging; if ``False``, reuse them.

    Returns:
        A reconstructed Flax NNX module of the snapshot's concrete type.

    Raises:
        TypeError: If ``snapshot`` is not a ``flax.nnx.GraphState``, ``copy`` is not a
            ``bool``, or the snapshot does not describe a ``flax.nnx.Module``.
    """
    if not isinstance(snapshot, nnx.GraphState):
        raise TypeError("snapshot must be a flax.nnx.GraphState")
    if not isinstance(copy, bool):
        raise TypeError("copy must be a boolean")
    restored = nnx.merge(snapshot, copy=copy)
    if not isinstance(restored, nnx.Module):
        raise TypeError("snapshot must describe a flax.nnx.Module")
    return restored


def clone_graph[T: nnx.Module](module: T) -> T:
    """Clone mutable variables while preserving aliases and the concrete NNX module type.

    Args:
        module: Flax NNX module to clone.

    Returns:
        An independent clone of ``module`` with the same concrete type and preserved
        internal aliases.

    Raises:
        TypeError: If ``module`` is not a ``flax.nnx.Module``, or Flax NNX returns a
            clone of an unexpected concrete type.
    """
    cloned: Any = nnx.clone(_module(module), variables=True, graph=True)
    if not isinstance(cloned, type(module)):
        raise TypeError("Flax NNX returned a clone with an unexpected concrete type")
    return cloned
