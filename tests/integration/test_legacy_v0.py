from pathlib import Path

import grain
import jax
import numpy as np
import pytest

from tinax.checkpointing.legacy.v0 import (
    load_legacy_v0_grain_iterator,
    load_legacy_v0_pytree,
    save_legacy_v0_grain_iterator,
    save_legacy_v0_pytree,
)


def test_legacy_v0_grain_iterator_round_trip(tmp_path: Path) -> None:
    source = iter(grain.MapDataset.range(6))
    replacement = iter(grain.MapDataset.range(6))
    try:
        assert next(source) == 0
        assert next(source) == 1
        save_legacy_v0_grain_iterator(tmp_path / "iterator", source)

        restored = load_legacy_v0_grain_iterator(tmp_path / "iterator", replacement)

        assert restored is replacement
        assert next(restored) == 2
    finally:
        source.close()
        replacement.close()


def test_legacy_v0_pytree_round_trip(tmp_path: Path) -> None:
    state = {"count": 4, "weight": jax.numpy.arange(3, dtype=jax.numpy.float32)}
    target = {
        "count": 0,
        "weight": jax.ShapeDtypeStruct(
            state["weight"].shape,
            state["weight"].dtype,
            sharding=state["weight"].sharding,
        ),
    }

    save_legacy_v0_pytree(tmp_path / "pytree", state)
    restored = load_legacy_v0_pytree(tmp_path / "pytree", target)

    assert restored["count"] == 4
    np.testing.assert_array_equal(restored["weight"], state["weight"])
    assert restored["weight"].sharding == state["weight"].sharding


def test_legacy_v0_rejects_orbax_temporary_destination_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved temporary suffix"):
        save_legacy_v0_pytree(tmp_path / "state.orbax-checkpoint-tmp-1", {"value": 1})
