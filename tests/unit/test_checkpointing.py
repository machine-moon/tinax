import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import grain
import jax
import numpy as np
import pytest

import tinax.checkpointing._training as training_module
from tinax.checkpointing import (
    TrainingCheckpointNames,
    abstract_restore_target,
    load_checkpointables,
    load_training_checkpoint,
    save_checkpointables,
    save_training_checkpoint,
    validate_checkpointable_name,
)


def test_mixed_restore_target_round_trip_preserves_types_and_array_contracts(tmp_path: Path) -> None:
    jax_array = jax.numpy.arange(4, dtype=jax.numpy.float32)
    numpy_array = np.arange(3, dtype=np.int16)
    state = {
        "enabled": True,
        "jax": jax_array,
        "label": "run-7",
        "none": None,
        "numpy": numpy_array,
        "ratio": 1.25,
        "step": 7,
    }
    target = abstract_restore_target(state)

    assert isinstance(target["jax"], jax.ShapeDtypeStruct)
    assert target["jax"].shape == jax_array.shape
    assert target["jax"].dtype == jax_array.dtype
    assert target["jax"].sharding == jax_array.sharding
    assert target["numpy"].shape == numpy_array.shape
    assert target["numpy"].dtype == numpy_array.dtype
    assert target["enabled"] is True
    assert type(target["label"]) is str
    assert target["none"] is None
    assert type(target["ratio"]) is float
    assert target["ratio"] == 1.25
    assert type(target["step"]) is int
    assert target["step"] == 7
    assert target["label"] == "run-7"

    completed: list[None] = []
    response = save_checkpointables(tmp_path / "mixed", {"training.state": state})
    response.on_complete(completed.append)
    assert response.result(timeout=30) is None
    assert completed == [None]

    restored = load_checkpointables(tmp_path / "mixed", {"training.state": target})["training.state"]

    assert isinstance(restored["jax"], jax.Array)
    assert isinstance(restored["numpy"], np.ndarray)
    assert restored["jax"].sharding == jax_array.sharding
    np.testing.assert_array_equal(restored["jax"], jax_array)
    np.testing.assert_array_equal(restored["numpy"], numpy_array)
    assert restored["enabled"] is True
    assert type(restored["label"]) is str
    assert restored["label"] == "run-7"
    assert restored["none"] is None
    assert type(restored["ratio"]) is float
    assert restored["ratio"] == 1.25
    assert type(restored["step"]) is int
    assert restored["step"] == 7


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "model/state",
        "model\\state",
        "model\nstate",
        "model:state",
        "model.",
        "_CHECKPOINT_METADATA",
        "AUTO",
        "metrics",
        "orbax.checkpoint",
        "CON",
        "nul.bin",
        "LPT9.log",
    ],
)
def test_invalid_checkpointable_names_are_rejected(name: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="checkpointable name|path component|control"):
        save_checkpointables(tmp_path / "unused", {name: {"value": 1}})


def test_checkpointable_name_requires_a_string() -> None:
    with pytest.raises(TypeError, match="strings"):
        validate_checkpointable_name(1)


def test_portable_checkpointable_name_is_returned() -> None:
    assert validate_checkpointable_name("model.v1") == "model.v1"


def test_training_checkpointable_names_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="distinct"):
        TrainingCheckpointNames(model="state", optimizer="state")
    with pytest.raises(ValueError, match="portable and distinct"):
        TrainingCheckpointNames(model="State", optimizer="state")


def test_checkpointable_mapping_names_must_be_portably_distinct(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="portable and distinct"):
        save_checkpointables(tmp_path / "unused", {"State": {}, "state": {}})


@pytest.mark.parametrize("name", ["state.orbax-checkpoint-tmp", "state.orbax-checkpoint-tmp-123"])
def test_reserved_orbax_temporary_paths_are_rejected(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="reserved temporary suffix"):
        save_checkpointables(tmp_path / name, {"state": {}})


def test_reserved_orbax_temporary_paths_cannot_hide_behind_dot_components(tmp_path: Path) -> None:
    path = f"{tmp_path}/state.orbax-checkpoint-tmp-123/."
    with pytest.raises(ValueError, match="reserved temporary suffix"):
        save_checkpointables(path, {"state": {}})


def test_load_requires_explicit_top_level_targets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit non-None"):
        load_checkpointables(tmp_path / "unused", {"state": None})


def test_failed_value_preflight_never_attempts_iterator_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    def fail_preflight(path: object, targets: object) -> dict[str, Any]:
        del path
        calls.append(targets)
        raise KeyError("incompatible target")

    monkeypatch.setattr(training_module, "load_checkpointables", fail_preflight)
    iterator = iter(grain.MapDataset.range(2))
    try:
        with pytest.raises(KeyError, match="incompatible"):
            load_training_checkpoint(tmp_path / "unused", {}, {}, {}, {}, iterator)
        assert len(calls) == 1
        assert next(iterator) == 0
    finally:
        iterator.close()


@pytest.mark.parametrize("step", [True, False, 1.5, "1"])
def test_training_checkpoint_rejects_non_integer_steps(step: object, tmp_path: Path) -> None:
    iterator = iter(grain.MapDataset.range(2))
    try:
        with pytest.raises(TypeError, match="integer"):
            save_training_checkpoint(tmp_path / "unused", cast(int, step), {}, {}, {}, {}, iterator)
    finally:
        iterator.close()


def test_training_checkpoint_rejects_negative_step(tmp_path: Path) -> None:
    iterator = iter(grain.MapDataset.range(2))
    try:
        with pytest.raises(ValueError, match="non-negative"):
            save_training_checkpoint(tmp_path / "unused", -1, {}, {}, {}, {}, iterator)
    finally:
        iterator.close()


def test_missing_and_incompatible_targets_raise_upstream_errors(tmp_path: Path) -> None:
    value = jax.numpy.arange(2, dtype=jax.numpy.float32)
    path = tmp_path / "state"
    save_checkpointables(path, {"present": {"value": value}}).result(timeout=30)

    with pytest.raises(KeyError, match="missing"):
        load_checkpointables(path, {"missing": abstract_restore_target({"value": value})})

    wrong_shape = jax.ShapeDtypeStruct((3,), value.dtype, sharding=value.sharding)
    with pytest.raises(KeyError, match="incompatible"):
        load_checkpointables(path, {"present": {"value": wrong_shape}})


def test_stable_import_does_not_import_or_reexport_legacy_v0() -> None:
    script = """
import sys
import tinax.checkpointing as checkpointing
assert "tinax.checkpointing.legacy.v0" not in sys.modules
assert not hasattr(checkpointing, "save_legacy_v0_pytree")
assert not hasattr(checkpointing, "save_legacy_v0_grain_iterator")
"""
    subprocess.run([sys.executable, "-c", script], check=True)
