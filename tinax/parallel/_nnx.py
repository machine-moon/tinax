"""Flax NNX eager-sharding and abstract-state integration."""

from collections.abc import Callable as _Callable
from collections.abc import Mapping as _Mapping

import jax as _jax
from flax import nnx as _nnx


def _validate_mesh(mesh: _jax.sharding.Mesh) -> None:
    if not isinstance(mesh, _jax.sharding.Mesh):
        raise TypeError("mesh must be a jax.sharding.Mesh")
    if mesh.empty:
        raise ValueError("mesh must not be empty")
    if any(axis_type is _jax.sharding.AxisType.Manual for axis_type in mesh.axis_types):
        raise ValueError("NNX eager sharding does not support Manual mesh axes")
    if not mesh.are_all_axes_auto and not mesh.are_all_axes_explicit:
        raise ValueError("NNX eager sharding requires all mesh axes to be Auto or all to be Explicit")


def eager_sharded_init[T](initializer: _Callable[[], T], mesh: _jax.sharding.Mesh) -> T:
    """Run an NNX initializer once under a homogeneous mesh with eager sharding enabled.

    Args:
        initializer: Zero-argument callable that constructs and returns NNX state.
        mesh: Homogeneous mesh (all Auto or all Explicit axes, no Manual).

    Returns:
        The initializer's result, sharded according to ``mesh``.

    Raises:
        TypeError: If ``initializer`` is not callable or ``mesh`` is not a
            ``jax.sharding.Mesh``.
        ValueError: If ``mesh`` is empty, uses Manual axes, or mixes Auto and Explicit
            axes.
    """
    if not callable(initializer):
        raise TypeError("initializer must be callable")
    _validate_mesh(mesh)
    with _jax.set_mesh(mesh), _nnx.use_eager_sharding(True):
        return _nnx.jit(initializer)()


def abstract_sharded_state[T](initializer: _Callable[[], T], mesh: _jax.sharding.Mesh) -> _nnx.GraphState[T]:
    """Create sharding-aware abstract graph state without allocating parameters.

    Args:
        initializer: Zero-argument callable that constructs NNX state.
        mesh: Homogeneous mesh (all Auto or all Explicit axes, no Manual).

    Returns:
        A ``flax.nnx.GraphState`` describing the abstract, sharding-aware state.

    Raises:
        TypeError: If ``initializer`` is not callable, ``mesh`` is not a
            ``jax.sharding.Mesh``, or the initializer does not return an NNX graph node.
        ValueError: If ``mesh`` is empty, uses Manual axes, or mixes Auto and Explicit
            axes.
    """
    if not callable(initializer):
        raise TypeError("initializer must be callable")
    _validate_mesh(mesh)
    with _nnx.use_eager_sharding(True):
        graphdef, state = _nnx.get_abstract_model(initializer, mesh, graph=True)
    if not isinstance(graphdef, _nnx.GraphDef) or not isinstance(state, _Mapping):
        raise TypeError("initializer must return an NNX graph node")
    return _nnx.GraphState(graphdef, state)


def state_shardings[StateMapping: _Mapping](state: StateMapping, mesh: _jax.sharding.Mesh) -> StateMapping:
    """Derive named shardings from one model or optimizer state independently.

    Args:
        state: Mapping of NNX state (for example, model or optimizer state).
        mesh: Homogeneous mesh (all Auto or all Explicit axes, no Manual).

    Returns:
        A mapping of the same structure whose leaves are the derived named shardings.

    Raises:
        TypeError: If ``state`` is not a mapping or ``mesh`` is not a
            ``jax.sharding.Mesh``.
        ValueError: If ``mesh`` is empty, uses Manual axes, or mixes Auto and Explicit
            axes.
    """
    if not isinstance(state, _Mapping):
        raise TypeError("state must be a mapping")
    _validate_mesh(mesh)
    return _nnx.get_named_sharding(state, mesh)
