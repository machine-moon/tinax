"""Pure path labels and an explicit partitioned-SGD recipe."""

from typing import Any as _Any

import jax as _jax
import optax as _optax
from flax import nnx as _nnx


def _terminal_name(path: tuple[_jax.tree_util.KeyEntry, ...]) -> str | None:
    entry = path[-1] if path else None
    if isinstance(entry, _jax.tree_util.GetAttrKey):
        return entry.name
    if isinstance(entry, (_jax.tree_util.DictKey, _jax.tree_util.FlattenedIndexKey)):
        return entry.key if isinstance(entry.key, str) else None
    return None


def label_kernel_bias_parameters(params: _Any) -> _Any:
    """Return pure labels based only on each parameter's terminal semantic name."""
    pure_params = _nnx.as_pure(params)
    return _jax.tree.map_with_path(
        lambda path, _: _terminal_name(path) if _terminal_name(path) in {"kernel", "bias"} else "other",
        pure_params,
    )


def kernel_bias_sgd(
    *,
    kernel_learning_rate: float,
    bias_learning_rate: float,
    other_learning_rate: float,
    momentum: float | None,
    nesterov: bool,
) -> _optax.GradientTransformationExtraArgs:
    """Build SGD partitions for terminal kernel, bias, and other parameter names."""
    return _optax.partition(
        {
            "kernel": _optax.sgd(kernel_learning_rate, momentum=momentum, nesterov=nesterov),
            "bias": _optax.sgd(bias_learning_rate, momentum=momentum, nesterov=nesterov),
            "other": _optax.sgd(other_learning_rate, momentum=momentum, nesterov=nesterov),
        },
        label_kernel_bias_parameters,
    )
