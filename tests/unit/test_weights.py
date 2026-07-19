from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import cast

import numpy as np
import pytest

from tinax.weights import (
    LoadedSafetensors,
    SafetensorsInfo,
    TensorInfo,
    TensorManifest,
    TensorRule,
    load_msgpack,
    save_msgpack,
)


def test_manifest_applies_owned_names_transforms_and_contracts() -> None:
    source = {
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "norm.weight": np.ones(3, dtype=np.float32),
    }
    manifest = TensorManifest(
        (
            TensorRule(
                source_name="linear.weight",
                destination_name="dense.kernel",
                transform=lambda value: value.T,
                expected_shape=(3, 2),
                expected_dtype=np.dtype(np.float32),
            ),
            TensorRule(
                source_name="norm.weight",
                destination_name="norm.scale",
                transform=lambda value: value,
                expected_shape=(3,),
                expected_dtype=np.dtype(np.float32),
            ),
        )
    )

    mapped = manifest.apply(source)

    assert set(mapped) == {"dense.kernel", "norm.scale"}
    np.testing.assert_array_equal(mapped["dense.kernel"], source["linear.weight"].T)
    np.testing.assert_array_equal(mapped["norm.scale"], source["norm.weight"])


def test_manifest_requires_exact_source_coverage() -> None:
    manifest = TensorManifest(
        (
            TensorRule("a", "x", lambda value: value, (1,), np.dtype(np.float32)),
            TensorRule("b", "y", lambda value: value, (1,), np.dtype(np.float32)),
        )
    )

    with pytest.raises(ValueError, match=r"missing=\['b'\], unexpected=\['c'\]"):
        manifest.apply({"a": np.ones(1, dtype=np.float32), "c": np.ones(1, dtype=np.float32)})


def test_manifest_validates_each_transform_before_running_the_next() -> None:
    later_transform_ran = False

    def later_transform(value: np.ndarray) -> np.ndarray:
        nonlocal later_transform_ran
        later_transform_ran = True
        return value

    manifest = TensorManifest(
        (
            TensorRule("a", "x", lambda value: value.reshape(2, 1), (2,), np.dtype(np.float32)),
            TensorRule("b", "y", later_transform, (1,), np.dtype(np.float32)),
        )
    )

    with pytest.raises(ValueError, match=r"source tensor 'a'.*shape \(2, 1\).*expected \(2,\)"):
        manifest.apply({"a": np.ones(2, dtype=np.float32), "b": np.ones(1, dtype=np.float32)})
    assert not later_transform_ran


def test_manifest_rejects_transform_dtype_mismatch() -> None:
    manifest = TensorManifest((TensorRule("a", "x", lambda value: value.astype(np.int32), (2,), np.dtype(np.float32)),))

    with pytest.raises(TypeError, match=r"source tensor 'a'.*dtype int32.*expected float32"):
        manifest.apply({"a": np.ones(2, dtype=np.float32)})


def test_manifest_rejects_source_and_destination_collisions() -> None:
    first = TensorRule("a", "x", lambda value: value, (1,), np.dtype(np.float32))

    with pytest.raises(ValueError, match="duplicate source tensor name"):
        TensorManifest((first, TensorRule("a", "y", lambda value: value, (1,), np.dtype(np.float32))))
    with pytest.raises(ValueError, match=r"destination tensor collision for 'x'.*'a'.*'b'"):
        TensorManifest((first, TensorRule("b", "x", lambda value: value, (1,), np.dtype(np.float32))))


def test_manifest_and_rules_are_immutable() -> None:
    rule = TensorRule("a", "x", lambda value: value, (1,), np.dtype(np.float32))
    manifest = TensorManifest((rule,))

    with pytest.raises(FrozenInstanceError):
        setattr(rule, "source_name", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(manifest, "rules", ())


def test_loaded_safetensors_validates_info_membership_shape_and_dtype() -> None:
    info = SafetensorsInfo({"value": TensorInfo((2,), "F32")}, {})

    with pytest.raises(ValueError, match="absent"):
        LoadedSafetensors({"other": np.ones(2, dtype=np.float32)}, info)
    with pytest.raises(ValueError, match="shape"):
        LoadedSafetensors({"value": np.ones(3, dtype=np.float32)}, info)
    with pytest.raises(TypeError, match="dtype"):
        LoadedSafetensors({"value": np.ones(2, dtype=np.int32)}, info)


def test_msgpack_round_trips_a_nested_parameter_pytree(tmp_path) -> None:
    state = {
        "dense": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3), "bias": np.zeros(3, dtype=np.float32)},
        "step": np.asarray(7, dtype=np.int32),
    }
    path = tmp_path / "state.msgpack"

    save_msgpack(path, state, overwrite=False)
    target = {
        "dense": {"kernel": np.empty((2, 3), dtype=np.float32), "bias": np.empty((3,), dtype=np.float32)},
        "step": np.empty((), dtype=np.int32),
    }
    restored = load_msgpack(path, target, max_bytes=1 << 20)

    assert np.array_equal(restored["dense"]["kernel"], state["dense"]["kernel"])
    assert np.array_equal(restored["dense"]["bias"], state["dense"]["bias"])
    assert int(restored["step"]) == 7


def test_msgpack_save_refuses_to_overwrite_without_permission(tmp_path) -> None:
    path = tmp_path / "state.msgpack"
    save_msgpack(path, {"value": np.ones(1, dtype=np.float32)}, overwrite=False)

    with pytest.raises(FileExistsError, match="already exists"):
        save_msgpack(path, {"value": np.ones(1, dtype=np.float32)}, overwrite=False)
    save_msgpack(path, {"value": np.zeros(1, dtype=np.float32)}, overwrite=True)


@pytest.mark.parametrize("overwrite", [1, None, "true"])
def test_msgpack_save_rejects_non_bool_overwrite(tmp_path, overwrite: object) -> None:
    state = {"value": np.ones(1, dtype=np.float32)}
    with pytest.raises(TypeError, match="overwrite"):
        save_msgpack(tmp_path / "state.msgpack", state, overwrite=cast(bool, overwrite))


def test_msgpack_save_rejects_non_mapping_state(tmp_path) -> None:
    with pytest.raises(TypeError, match="mapping"):
        save_msgpack(tmp_path / "state.msgpack", cast(Mapping[str, object], [1, 2, 3]), overwrite=False)


def test_msgpack_load_enforces_the_byte_budget(tmp_path) -> None:
    path = tmp_path / "state.msgpack"
    state = {"value": np.arange(64, dtype=np.float32)}
    save_msgpack(path, state, overwrite=False)

    with pytest.raises(ValueError, match="max_bytes"):
        load_msgpack(path, {"value": np.empty(64, dtype=np.float32)}, max_bytes=8)


def test_msgpack_load_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no msgpack file"):
        load_msgpack(tmp_path / "absent.msgpack", {"value": np.empty(1, dtype=np.float32)}, max_bytes=1 << 20)
