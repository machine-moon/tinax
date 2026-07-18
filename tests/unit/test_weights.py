from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tinax.weights import LoadedSafetensors, SafetensorsInfo, TensorInfo, TensorManifest, TensorRule


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
    manifest = TensorManifest(
        (TensorRule("a", "x", lambda value: value.astype(np.int32), (2,), np.dtype(np.float32)),)
    )

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
