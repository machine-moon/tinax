"""Debug callbacks receive bounded summaries instead of full device arrays."""

from collections.abc import Callable
from typing import Any

import jax

from tinax.diagnostics import observe_nonfinite


def report_nonfinite(
    values: jax.Array,
    *,
    callback: Callable[[Any, Any], None],
    summary_size: int,
) -> jax.Array:
    """Report a count and a caller-bounded flat-index summary for nonfinite values."""
    if not isinstance(summary_size, int) or isinstance(summary_size, bool):
        raise TypeError("summary_size must be an integer")
    if summary_size < 0:
        raise ValueError("summary_size must be nonnegative")
    return observe_nonfinite(values, callback=callback, max_indices=summary_size)
