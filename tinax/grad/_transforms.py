"""Validated autodiff transforms: value_and_grad, jacobian, and hessian."""

from collections.abc import Callable as _Callable
from collections.abc import Sequence as _Sequence
from typing import Literal as _Literal

import jax as _jax

import tinax.grad._validation as _validation


def value_and_grad[**P](
    function: _Callable[P, object],
    *,
    argnums: int | _Sequence[int] = 0,
    has_aux: bool = False,
) -> _Callable[P, tuple[object, object]]:
    """Differentiate function, validating argnums where raw jax.grad silently accepts a bool.

    Bare ``grad`` is not exposed: ``value_and_grad`` is a strict superset, and callers who
    only need the gradient can discard the value.

    Args:
        function: Callable to differentiate.
        argnums: Positional argument index or indices to differentiate with respect to.
            Must be nonnegative integers with no duplicates. Defaults to 0.
        has_aux: Whether ``function`` returns ``(output, auxiliary)``. Defaults to ``False``.

    Returns:
        A callable returning ``(value, gradient)`` for the selected arguments.

    Raises:
        TypeError: If ``function`` is not callable, ``argnums`` is not an integer or sequence
            of integers (booleans rejected), or ``has_aux`` is not a bool.
        ValueError: If ``argnums`` is negative, empty (as a sequence), or contains a
            duplicate index.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    validated_argnums = _validation.validate_argnums(argnums)
    if not isinstance(has_aux, bool):
        raise TypeError("has_aux must be a bool")
    return _jax.value_and_grad(function, argnums=validated_argnums, has_aux=has_aux)


def jacobian[**P](
    function: _Callable[P, object],
    *,
    mode: _Literal["forward", "reverse"],
    argnums: int | _Sequence[int] = 0,
    has_aux: bool = False,
) -> _Callable[P, object]:
    """Compute a Jacobian, requiring an explicit mode instead of a separate jacfwd/jacrev call.

    Raw JAX splits this into two separately named top-level functions with a real
    wide-versus-tall performance tradeoff; ``mode`` turns that implicit choice into one
    explicit, validated parameter.

    Args:
        function: Callable to differentiate.
        mode: ``"forward"`` uses forward-mode autodiff (``jax.jacfwd``), efficient for few
            inputs and many outputs. ``"reverse"`` uses reverse-mode autodiff
            (``jax.jacrev``), efficient for many inputs and few outputs.
        argnums: Positional argument index or indices to differentiate with respect to.
            Must be nonnegative integers with no duplicates. Defaults to 0.
        has_aux: Whether ``function`` returns ``(output, auxiliary)``. Defaults to ``False``.

    Returns:
        A callable returning the Jacobian for the selected arguments.

    Raises:
        TypeError: If ``function`` is not callable, ``mode`` is not a string, ``argnums`` is
            not an integer or sequence of integers (booleans rejected), or ``has_aux`` is not
            a bool.
        ValueError: If ``mode`` is not ``"forward"`` or ``"reverse"``, ``argnums`` is
            negative, empty (as a sequence), or contains a duplicate index.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    if mode not in ("forward", "reverse"):
        raise ValueError("mode must be 'forward' or 'reverse'")
    validated_argnums = _validation.validate_argnums(argnums)
    if not isinstance(has_aux, bool):
        raise TypeError("has_aux must be a bool")
    transform = _jax.jacfwd if mode == "forward" else _jax.jacrev
    return transform(function, argnums=validated_argnums, has_aux=has_aux)


def hessian[**P](
    function: _Callable[P, object],
    *,
    argnums: int | _Sequence[int] = 0,
) -> _Callable[P, object]:
    """Compute a Hessian, sharing value_and_grad's and jacobian's argnums validation.

    Args:
        function: Callable to differentiate twice.
        argnums: Positional argument index or indices to differentiate with respect to.
            Must be nonnegative integers with no duplicates. Defaults to 0.

    Returns:
        A callable returning the Hessian for the selected arguments.

    Raises:
        TypeError: If ``function`` is not callable, or ``argnums`` is not an integer or
            sequence of integers (booleans rejected).
        ValueError: If ``argnums`` is negative, empty (as a sequence), or contains a
            duplicate index.
    """
    if not callable(function):
        raise TypeError("function must be callable")
    validated_argnums = _validation.validate_argnums(argnums)
    return _jax.hessian(function, argnums=validated_argnums)
