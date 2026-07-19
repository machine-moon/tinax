from collections.abc import Callable
from typing import Literal, cast

import jax
import jax.numpy as jnp
import pytest

import tinax.grad as grad_api
from tinax.grad import hessian, jacobian, value_and_grad


def test_grad_package_has_only_intentional_public_exports() -> None:
    public_names = {name for name in vars(grad_api) if not name.startswith("_")}
    assert public_names == {"hessian", "jacobian", "value_and_grad"}


def _quadratic_loss(weight: jax.Array, features: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.mean((features * weight - targets) ** 2)


def test_value_and_grad_matches_jax_value_and_grad() -> None:
    weight = jnp.asarray(2.0)
    features = jnp.asarray([1.0, 2.0])
    targets = jnp.asarray([2.0, 3.0])

    tinax_result = value_and_grad(_quadratic_loss)(weight, features, targets)
    jax_result = jax.value_and_grad(_quadratic_loss)(weight, features, targets)

    assert jnp.array_equal(cast(jax.Array, tinax_result[0]), jax_result[0])
    assert jnp.array_equal(cast(jax.Array, tinax_result[1]), jax_result[1])


def test_value_and_grad_rejects_a_non_callable_function() -> None:
    with pytest.raises(TypeError, match="callable"):
        value_and_grad(cast(Callable[..., object], None))


def test_value_and_grad_rejects_a_non_bool_has_aux() -> None:
    with pytest.raises(TypeError, match="has_aux"):
        value_and_grad(_quadratic_loss, has_aux=cast(bool, 1))


@pytest.mark.parametrize(
    ("argnums", "match"),
    [
        (True, "not a bool"),
        ((), "not be empty"),
        ((0, 0), "duplicate"),
        (-1, "nonnegative"),
        ((0, -1), "nonnegative"),
        ((0, True), "not bools"),
        ("0", "integer or a sequence"),
    ],
)
def test_argnums_validation_is_shared_across_all_three_transforms(argnums: object, match: str) -> None:
    for transform in (value_and_grad, hessian):
        with pytest.raises((TypeError, ValueError), match=match):
            transform(_quadratic_loss, argnums=cast(int, argnums))
    with pytest.raises((TypeError, ValueError), match=match):
        jacobian(_quadratic_loss, mode="forward", argnums=cast(int, argnums))


def test_jacobian_requires_an_explicit_mode() -> None:
    with pytest.raises(TypeError, match="callable"):
        jacobian(cast(Callable[..., object], None), mode="forward")
    with pytest.raises(TypeError, match="mode must be a string"):
        jacobian(lambda x: x, mode=cast(Literal["forward", "reverse"], None))
    with pytest.raises(ValueError, match="'forward' or 'reverse'"):
        jacobian(lambda x: x, mode=cast(Literal["forward", "reverse"], "sideways"))


def test_jacobian_rejects_a_non_bool_has_aux() -> None:
    with pytest.raises(TypeError, match="has_aux"):
        jacobian(lambda x: x, mode="forward", has_aux=cast(bool, 1))


def test_jacobian_forward_and_reverse_both_match_jax_and_agree_with_each_other() -> None:
    def elementwise_square(values: jax.Array) -> jax.Array:
        return values**2

    x = jnp.asarray([1.0, 2.0, 3.0])
    forward = cast(jax.Array, jacobian(elementwise_square, mode="forward")(x))
    reverse = cast(jax.Array, jacobian(elementwise_square, mode="reverse")(x))

    assert jnp.array_equal(forward, jax.jacfwd(elementwise_square)(x))
    assert jnp.array_equal(reverse, jax.jacrev(elementwise_square)(x))
    assert jnp.array_equal(forward, reverse)


def test_hessian_matches_jax_hessian() -> None:
    weight = jnp.asarray(2.0)
    features = jnp.asarray([1.0, 2.0])
    targets = jnp.asarray([2.0, 3.0])

    tinax_result = cast(jax.Array, hessian(_quadratic_loss)(weight, features, targets))
    jax_result = jax.hessian(_quadratic_loss)(weight, features, targets)

    assert jnp.array_equal(tinax_result, jax_result)


def test_hessian_rejects_a_non_callable_function() -> None:
    with pytest.raises(TypeError, match="callable"):
        hessian(cast(Callable[..., object], None))
