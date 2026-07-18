"""Trace-time Chex shape and dtype contracts."""

from collections.abc import Sequence as _Sequence

import chex as _chex
import jax as _jax


def assert_array_contract(
    array: _jax.Array,
    *,
    shape: _Sequence[int | None],
    dtype: _jax.typing.DTypeLike,
) -> None:
    """Assert array properties available during eager execution or JIT tracing."""
    _chex.assert_shape(array, shape)
    _chex.assert_type(array, dtype)
