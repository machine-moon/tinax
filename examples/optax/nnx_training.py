"""Flax NNX training steps backed by an existing Optax optimizer."""

from collections.abc import Callable as _Callable
from typing import Any as _Any

from flax import nnx as _nnx


def make_train_step(
    optimizer: _nnx.Optimizer,
    loss_fn: _Callable[..., _Any],
    *,
    has_aux: bool = False,
) -> _Callable[..., _Any]:
    """Build one jitted update step using the optimizer's differentiation filter."""
    loss_and_grad = _nnx.value_and_grad(
        loss_fn,
        argnums=_nnx.DiffState(0, optimizer.wrt),
        has_aux=has_aux,
    )

    @_nnx.jit
    def train_step(
        model: _nnx.Module,
        current_optimizer: _nnx.Optimizer,
        *args: _Any,
        **kwargs: _Any,
    ) -> _Any:
        value, grads = loss_and_grad(model, *args, **kwargs)
        current_optimizer.update(model, grads)
        return value

    return train_step
