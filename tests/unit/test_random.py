from typing import cast

import jax
import jax.numpy as jnp
import pytest

from tinax.random import derive_key, derive_process_step_key, split_key


def test_ordered_coordinates_are_deterministic_and_distinct() -> None:
    key = jax.random.key(17)

    derived = derive_key(key, 2, 5)
    same = derive_key(key, 2, 5)
    reordered = derive_key(key, 5, 2)

    assert bool(jnp.array_equal(derived, same))
    assert not bool(jnp.array_equal(derived, reordered))
    assert jax.dtypes.issubdtype(derived.dtype, jax.dtypes.prng_key)


def test_process_step_stream_derivation_is_jittable() -> None:
    key = jax.random.key(23)
    derive = jax.jit(derive_process_step_key)

    process_index = jnp.asarray(1, dtype=jnp.uint32)
    step = jnp.asarray(7, dtype=jnp.uint32)
    stream = jnp.asarray(3, dtype=jnp.uint32)
    actual = derive(key, process_index=process_index, step=step, stream=stream)

    assert bool(jnp.array_equal(actual, derive_process_step_key(key, process_index=1, step=7, stream=3)))
    assert not bool(jnp.array_equal(actual, derive(key, process_index=jnp.uint32(2), step=step, stream=stream)))
    assert not bool(
        jnp.array_equal(actual, derive(key, process_index=process_index, step=jnp.uint32(8), stream=stream))
    )
    assert not bool(jnp.array_equal(actual, derive(key, process_index=process_index, step=step, stream=jnp.uint32(4))))


def test_split_key_returns_continuation_and_operation_keys() -> None:
    key = jax.random.key(29)

    continuation, operation_keys = split_key(key, count=3)
    expected = jax.random.split(key, 4)

    assert bool(jnp.array_equal(continuation, expected[0]))
    assert bool(jnp.array_equal(operation_keys, expected[1:]))
    assert operation_keys.shape == (3,)


def test_split_key_supports_an_empty_operation_batch() -> None:
    continuation, operation_keys = split_key(jax.random.key(31), count=0)

    assert continuation.shape == ()
    assert operation_keys.shape == (0,)


@pytest.mark.parametrize(
    "key",
    [object(), jnp.asarray([1, 2], dtype=jnp.uint32), jax.random.split(jax.random.key(1), 2)],
)
def test_key_operations_reject_non_typed_scalar_keys(key: object) -> None:
    with pytest.raises((TypeError, ValueError), match="typed|scalar"):
        derive_key(cast(jax.Array, key), 0)


@pytest.mark.parametrize("coordinate", [True, 1.5, "1", jnp.asarray([1], dtype=jnp.int32)])
def test_key_derivation_rejects_non_integer_scalar_coordinates(coordinate: object) -> None:
    with pytest.raises(TypeError, match="integer scalar"):
        derive_key(jax.random.key(1), cast(int | jax.Array, coordinate))


@pytest.mark.parametrize("coordinate", [-1, 2**32])
def test_key_derivation_rejects_python_coordinates_outside_uint32(coordinate: int) -> None:
    with pytest.raises(ValueError, match=r"2\*\*32"):
        derive_key(jax.random.key(1), coordinate)


def test_dynamic_coordinates_require_a_uint32_compatible_dtype() -> None:
    with pytest.raises(TypeError, match="unsigned dtype"):
        derive_key(jax.random.key(1), jnp.asarray(1, dtype=jnp.int32))


@pytest.mark.parametrize("count", [True, 1.5, "1", None])
def test_split_key_rejects_non_integer_counts(count: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        split_key(jax.random.key(1), count=cast(int, count))


def test_split_key_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        split_key(jax.random.key(1), count=-1)
