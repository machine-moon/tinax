"""Shared argnums validation for autodiff transforms."""

from collections.abc import Sequence as _Sequence


def validate_argnums(argnums: int | _Sequence[int]) -> int | tuple[int, ...]:
    if isinstance(argnums, bool):
        raise TypeError("argnums must be an integer or a sequence of integers, not a bool")
    if isinstance(argnums, int):
        if argnums < 0:
            raise ValueError("argnums must be nonnegative")
        return argnums
    if isinstance(argnums, _Sequence) and not isinstance(argnums, (str, bytes)):
        values = tuple(argnums)
        if not values:
            raise ValueError("argnums must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("argnums must contain only integers, not bools")
        if any(value < 0 for value in values):
            raise ValueError("argnums must be nonnegative")
        if len(set(values)) != len(values):
            raise ValueError("argnums must not contain duplicates")
        return values
    raise TypeError("argnums must be an integer or a sequence of integers")
