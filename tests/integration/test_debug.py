from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinax.debug import observe_nonfinite, profile_call


def test_nonfinite_observation_is_bounded_jittable_and_value_preserving() -> None:
    observed: list[tuple[int, np.ndarray]] = []

    def capture(count: object, indices: object) -> None:
        observed.append((int(np.asarray(count)), np.asarray(indices)))

    values = jnp.asarray([[1.0, jnp.nan], [jnp.inf, 4.0]], dtype=jnp.float32)
    result = jax.jit(lambda array: observe_nonfinite(array, callback=capture, max_indices=3))(values)
    result.block_until_ready()
    jax.effects_barrier()

    np.testing.assert_array_equal(result, values)
    assert len(observed) == 1
    assert observed[0][0] == 2
    np.testing.assert_array_equal(observed[0][1], np.asarray([1, 2, -1]))


def test_finite_values_do_not_invoke_the_callback() -> None:
    calls: list[object] = []
    result = observe_nonfinite(
        jnp.arange(4, dtype=jnp.float32),
        callback=lambda *values: calls.append(values),
        max_indices=2,
    )
    result.block_until_ready()
    jax.effects_barrier()

    assert not calls


@pytest.mark.parametrize("max_indices", [True, 1.5, "1", None])
def test_nonfinite_observation_rejects_non_integer_bounds(max_indices: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        observe_nonfinite(
            jnp.ones(1),
            callback=lambda *_: None,
            max_indices=cast(int, max_indices),
        )


def test_nonfinite_observation_validates_callback_values_and_range() -> None:
    with pytest.raises(TypeError, match="callable"):
        observe_nonfinite(jnp.ones(1), callback=cast(Callable[[Any, Any], None], None), max_indices=1)
    with pytest.raises(TypeError, match="JAX array"):
        observe_nonfinite(cast(jax.Array, object()), callback=lambda *_: None, max_indices=1)
    with pytest.raises(TypeError, match="JAX array"):
        observe_nonfinite(cast(jax.Array, np.ones(1)), callback=lambda *_: None, max_indices=1)
    with pytest.raises(ValueError, match="non-negative"):
        observe_nonfinite(jnp.ones(1), callback=lambda *_: None, max_indices=-1)


def test_profile_call_completes_pytrees_and_effects_before_trace_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    @contextmanager
    def trace(path: str | Path) -> Iterator[None]:
        assert Path(path) == tmp_path
        events.append("trace-start")
        try:
            yield
        finally:
            events.append("trace-stop")

    def block_until_ready(result: object) -> object:
        events.append("arrays-ready")
        return result

    def effects_barrier() -> None:
        events.append("effects-ready")

    monkeypatch.setattr(jax.profiler, "trace", trace)
    monkeypatch.setattr(jax, "block_until_ready", block_until_ready)
    monkeypatch.setattr(jax, "effects_barrier", effects_barrier)

    result = profile_call(tmp_path, lambda left, *, right: {"sum": left + right}, 2, right=3)

    assert result == {"sum": 5}
    assert events == ["trace-start", "arrays-ready", "effects-ready", "trace-stop"]


def test_profile_call_closes_trace_when_the_function_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    @contextmanager
    def trace(_: str | Path) -> Iterator[None]:
        events.append("trace-start")
        try:
            yield
        finally:
            events.append("trace-stop")

    def fail() -> None:
        raise RuntimeError("call failed")

    monkeypatch.setattr(jax.profiler, "trace", trace)
    with pytest.raises(RuntimeError, match="call failed"):
        profile_call(tmp_path, fail)
    assert events == ["trace-start", "trace-stop"]


def test_profile_call_rejects_non_callable_values(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="callable"):
        profile_call(tmp_path, cast(Callable[..., object], None))
