"""Validated jax.shard_map over Manual mesh axes, checked before staging."""

from collections.abc import Callable as _Callable

import jax as _jax

import tinax.parallel._validation as _validation


def shard_mapped[**P, R](
    function: _Callable[P, R],
    *,
    mesh: _jax.sharding.Mesh,
    in_specs: object,
    out_specs: object,
    axis_names: frozenset[str] = frozenset(),
    check_vma: bool = True,
) -> _Callable[P, R]:
    """Manually shard-map a callable, validating specs against Manual mesh axes before staging.

    ``layout`` builds concrete Explicit-axis layouts; ``shard_mapped`` is the Manual-axis
    counterpart for callables that issue explicit collectives. Every mesh axis this call is
    manual over (``axis_names``, or every mesh axis if empty, matching ``jax.shard_map``'s own
    semantics) must be ``AxisType.Manual``; every ``in_specs``/``out_specs`` leaf may only
    reference axes within that set. Validation runs at wrap time, before any argument is staged.

    Args:
        function: Callable to shard-map. Each call receives one shard of its mapped arguments.
        mesh: Mesh whose axes this call is manual over.
        in_specs: Pytree of ``jax.sharding.PartitionSpec`` leaves describing argument sharding.
        out_specs: Pytree of ``jax.sharding.PartitionSpec`` leaves describing output sharding.
        axis_names: Mesh axes this call is manual over. Empty means every mesh axis.
        check_vma: Whether to enable ``jax.shard_map``'s replication validity checks. Defaults
            to ``True``.

    Returns:
        The shard-mapped callable.

    Raises:
        TypeError: If ``function`` is not callable, ``mesh`` is not a ``jax.sharding.Mesh``,
            ``axis_names`` is not a ``frozenset`` of strings, ``check_vma`` is not a bool, or an
            ``in_specs``/``out_specs`` leaf is not a ``jax.sharding.PartitionSpec``.
        ValueError: If ``mesh`` is empty, ``axis_names`` is not a subset of the mesh's axes, an
            effective manual axis is not ``AxisType.Manual``, or a spec references an axis
            outside the effective manual set.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(mesh, _jax.sharding.Mesh):
        raise TypeError("mesh must be a jax.sharding.Mesh")
    if mesh.empty:
        raise ValueError("mesh must not be empty")
    if not isinstance(axis_names, frozenset) or any(not isinstance(name, str) for name in axis_names):
        raise TypeError("axis_names must be a frozenset of strings")
    if not isinstance(check_vma, bool):
        raise TypeError("check_vma must be a bool")

    effective_axes = _validation.require_manual_axes(mesh, axis_names)
    _validation.validate_specs_against_axes(in_specs, effective_axes, "in_specs")
    _validation.validate_specs_against_axes(out_specs, effective_axes, "out_specs")

    return _jax.shard_map(
        function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        axis_names=axis_names,
        check_vma=check_vma,
    )
