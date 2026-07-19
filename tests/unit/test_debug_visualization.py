from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tinax.debug as debug_api
from tinax.debug import visualize_array_sharding


def test_debug_package_has_only_intentional_public_exports() -> None:
    public_names = {name for name in vars(debug_api) if not name.startswith("_")}
    assert public_names == {"observe_nonfinite", "profile_call", "visualize_array_sharding"}


def test_visualize_array_sharding_delegates_to_jax_debug(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []
    monkeypatch.setattr(jax.debug, "visualize_array_sharding", lambda array: seen.append(array))

    array = jnp.arange(4)
    result = visualize_array_sharding(array)

    assert result is None
    assert seen == [array]


def test_visualize_array_sharding_rejects_a_non_jax_array() -> None:
    with pytest.raises(TypeError, match="jax.Array"):
        visualize_array_sharding(cast(jax.Array, np.arange(4)))
    with pytest.raises(TypeError, match="jax.Array"):
        visualize_array_sharding(cast(jax.Array, object()))
