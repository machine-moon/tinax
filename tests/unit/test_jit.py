from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh

import tinax.jit as jit_api
from tinax.jit import batched_jit, bounded_jit


@pytest.fixture
def explicit_mesh() -> Mesh:
    devices = tuple(jax.devices("cpu"))
    assert devices
    return Mesh(devices, ("data",), axis_types=(AxisType.Explicit,))


def test_jit_package_has_only_intentional_public_exports() -> None:
    public_names = {name for name in vars(jit_api) if not name.startswith("_")}
    assert public_names == {"batched_jit", "bounded_jit"}


def test_bounded_jit_matches_the_plain_function_within_its_trace_budget() -> None:
    jitted = bounded_jit(lambda x: x * 2, max_traces=1)
    result = jitted(jnp.arange(4, dtype=jnp.float32))
    assert jnp.array_equal(result, jnp.arange(4, dtype=jnp.float32) * 2)


def test_bounded_jit_raises_once_the_trace_budget_is_exceeded() -> None:
    jitted = bounded_jit(lambda x: x * 2, max_traces=1)
    jitted(jnp.arange(4, dtype=jnp.float32))
    with pytest.raises(AssertionError):
        jitted(jnp.arange(6, dtype=jnp.float32))


def test_bounded_jit_rejects_a_non_callable_function() -> None:
    with pytest.raises(TypeError, match="callable"):
        bounded_jit(cast(Callable[[jax.Array], jax.Array], None), max_traces=1)


def test_bounded_jit_rejects_non_integer_or_negative_max_traces() -> None:
    with pytest.raises(TypeError, match="max_traces"):
        bounded_jit(lambda x: x, max_traces=cast(int, True))
    with pytest.raises(ValueError, match="non-negative"):
        bounded_jit(lambda x: x, max_traces=-1)


def test_bounded_jit_validates_static_and_donate_argnames_at_wrap_time() -> None:
    with pytest.raises(TypeError, match="static_argnames"):
        bounded_jit(lambda x: x, max_traces=1, static_argnames=cast(tuple[str, ...], [1]))
    with pytest.raises(ValueError, match="static_argnames.*nonempty"):
        bounded_jit(lambda x: x, max_traces=1, static_argnames=("",))
    with pytest.raises(ValueError, match="static_argnames.*duplicates"):
        bounded_jit(lambda x, y: x, max_traces=1, static_argnames=("x", "x"))
    with pytest.raises(ValueError, match="must not overlap"):
        bounded_jit(lambda x: x, max_traces=1, static_argnames=("x",), donate_argnames=("x",))


def test_bounded_jit_rejects_wrong_mesh_categories(explicit_mesh: Mesh) -> None:
    with pytest.raises(TypeError, match="jax.sharding.Mesh"):
        bounded_jit(lambda x: x, max_traces=1, mesh=cast(Mesh, object()))
    empty_mesh = Mesh(np.empty((0,), dtype=object), ("data",), axis_types=(AxisType.Explicit,))
    with pytest.raises(ValueError, match="not be empty"):
        bounded_jit(lambda x: x, max_traces=1, mesh=empty_mesh)


def test_bounded_jit_enters_the_given_mesh_around_every_call(
    monkeypatch: pytest.MonkeyPatch, explicit_mesh: Mesh
) -> None:
    entered: list[Mesh] = []

    @contextmanager
    def fake_set_mesh(mesh: Mesh) -> Iterator[None]:
        entered.append(mesh)
        yield

    monkeypatch.setattr(jax, "set_mesh", fake_set_mesh)
    jitted = bounded_jit(lambda x: x, max_traces=1, mesh=explicit_mesh)

    jitted(jnp.arange(2, dtype=jnp.float32))

    assert entered == [explicit_mesh]


def test_batched_jit_matches_manual_vmap_then_jit() -> None:
    def affine(weight: jax.Array, features: jax.Array) -> jax.Array:
        return features @ weight

    weight = jnp.eye(3, dtype=jnp.float32)
    features = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    batched = batched_jit(affine, in_axes=(None, 0), max_traces=1)
    expected = jax.jit(jax.vmap(affine, in_axes=(None, 0)))(weight, features)

    assert jnp.array_equal(batched(weight, features), expected)


def test_batched_jit_rejects_a_non_callable_function() -> None:
    with pytest.raises(TypeError, match="callable"):
        batched_jit(cast(Callable[[jax.Array], jax.Array], None), max_traces=1)


def test_batched_jit_rejects_a_boolean_or_non_positive_axis_size() -> None:
    with pytest.raises(TypeError, match="axis_size"):
        batched_jit(lambda x: x, max_traces=1, axis_size=cast(int, True))
    with pytest.raises(ValueError, match="positive"):
        batched_jit(lambda x: x, max_traces=1, axis_size=0)


def test_batched_jit_rejects_non_integer_or_negative_max_traces() -> None:
    with pytest.raises(TypeError, match="max_traces"):
        batched_jit(lambda x: x, max_traces=cast(int, True))
    with pytest.raises(ValueError, match="non-negative"):
        batched_jit(lambda x: x, max_traces=-1)


def test_batched_jit_raises_once_the_trace_budget_is_exceeded() -> None:
    batched = batched_jit(lambda x: x * 2, max_traces=1)
    batched(jnp.arange(4, dtype=jnp.float32))
    with pytest.raises(AssertionError):
        batched(jnp.arange(6, dtype=jnp.float32))


def test_batched_jit_enters_the_given_mesh_around_every_call(
    monkeypatch: pytest.MonkeyPatch, explicit_mesh: Mesh
) -> None:
    entered: list[Mesh] = []

    @contextmanager
    def fake_set_mesh(mesh: Mesh) -> Iterator[None]:
        entered.append(mesh)
        yield

    monkeypatch.setattr(jax, "set_mesh", fake_set_mesh)
    batched = batched_jit(lambda x: x, max_traces=1, mesh=explicit_mesh)

    batched(jnp.arange(2, dtype=jnp.float32))

    assert entered == [explicit_mesh]
