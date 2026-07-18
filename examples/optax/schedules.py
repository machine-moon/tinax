"""Explicit Optax optimizer and learning-rate schedule recipes."""

import optax as _optax


def warmup_cosine_adam(
    *,
    initial_learning_rate: float,
    peak_learning_rate: float,
    end_learning_rate: float,
    warmup_steps: int,
    total_steps: int,
    exponent: float,
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    nesterov: bool,
) -> _optax.GradientTransformationExtraArgs:
    """Build explicitly configured Adam with warmup and cosine decay over total_steps."""
    schedule = _optax.warmup_cosine_decay_schedule(
        init_value=initial_learning_rate,
        peak_value=peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=end_learning_rate,
        exponent=exponent,
    )
    return _optax.adam(
        learning_rate=schedule,
        b1=b1,
        b2=b2,
        eps=eps,
        eps_root=eps_root,
        nesterov=nesterov,
    )
