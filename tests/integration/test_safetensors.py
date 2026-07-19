import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import numpy.typing as npt
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from safetensors import SafetensorError

import tinax.weights._safetensors as safetensors_module
from tinax.weights import TensorInfo, inspect_safetensors, load_safetensors, save_safetensors


def test_safetensors_round_trip_inspection_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "weights.data"
    metadata_budget = 64
    tensors = {
        "matrix": np.arange(6, dtype=np.float32).reshape(2, 3),
        "ids": np.array([2, 5], dtype=np.int32),
    }

    save_safetensors(
        path,
        tensors,
        overwrite=False,
        metadata={"format": "host-weights", "revision": "1"},
        max_metadata_bytes=metadata_budget,
    )
    info = inspect_safetensors(path, max_metadata_bytes=metadata_budget)
    loaded = load_safetensors(path, max_bytes=32, max_metadata_bytes=metadata_budget)

    assert info.tensors["matrix"] == TensorInfo((2, 3), "F32")
    assert info.tensors["ids"] == TensorInfo((2,), "I32")
    assert info.total_bytes == 32
    assert dict(info.metadata) == {"format": "host-weights", "revision": "1"}
    assert dict(loaded.metadata) == dict(info.metadata)
    np.testing.assert_array_equal(loaded.tensors["matrix"], tensors["matrix"])
    np.testing.assert_array_equal(loaded.tensors["ids"], tensors["ids"])


def test_safetensors_supports_zero_length_and_scalar_tensors(tmp_path: Path) -> None:
    path = tmp_path / "edge-cases"
    tensors = {
        "empty": np.empty((0, 3), dtype=np.float16),
        "scalar": np.array(7, dtype=np.int64),
    }

    save_safetensors(path, tensors, overwrite=False)
    loaded = load_safetensors(path, max_bytes=8)

    assert loaded.info.tensors["empty"].shape == (0, 3)
    assert loaded.info.tensors["empty"].nbytes == 0
    assert loaded.tensors["scalar"].shape == ()
    assert loaded.tensors["scalar"].item() == 7


def test_safetensors_round_trips_bfloat16_when_numpy_supports_it(tmp_path: Path) -> None:
    try:
        dtype = np.dtype("bfloat16")
    except TypeError:
        pytest.skip("the active NumPy dtype registry does not provide bfloat16")
    path = tmp_path / "bf16"
    source = np.array([1.0, -2.0], dtype=dtype)

    save_safetensors(path, {"values": source}, overwrite=False)
    loaded = load_safetensors(path, max_bytes=source.nbytes)

    assert loaded.info.tensors["values"].dtype == "BF16"
    np.testing.assert_array_equal(loaded.tensors["values"], source)


def test_safetensors_rejects_float8_even_when_registered_with_numpy(tmp_path: Path) -> None:
    dtype = np.dtype("float8_e5m2")
    source = np.array([1.0, -2.0], dtype=dtype)

    with pytest.raises(TypeError, match="unsupported NumPy dtype"):
        save_safetensors(tmp_path / "float8", {"values": source}, overwrite=False)


def test_selective_loading_and_budget_rejection_precede_get_tensor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "selective"
    save_safetensors(
        path,
        {
            "large": np.arange(16, dtype=np.float32),
            "small": np.array([3], dtype=np.float32),
        },
        overwrite=False,
    )
    selected = load_safetensors(path, names=["small"], max_bytes=4)
    assert set(selected.tensors) == {"small"}

    get_tensor_calls: list[str] = []
    real_safe_open = getattr(safetensors_module, "_safe_open")

    class HandleSpy:
        def __init__(self, handle: Any) -> None:
            self.handle = handle

        def __enter__(self) -> "HandleSpy":
            self.handle.__enter__()
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object:
            return self.handle.__exit__(exc_type, exc_value, traceback)

        def keys(self) -> list[str]:
            return self.handle.keys()

        def metadata(self) -> dict[str, str] | None:
            return self.handle.metadata()

        def get_slice(self, name: str) -> Any:
            return self.handle.get_slice(name)

        def get_tensor(self, name: str) -> object:
            get_tensor_calls.append(name)
            return self.handle.get_tensor(name)

    def open_spy(filename: str | os.PathLike[str], framework: str) -> HandleSpy:
        return HandleSpy(real_safe_open(filename, framework=framework))

    monkeypatch.setattr(safetensors_module, "_safe_open", open_spy)
    with pytest.raises(ValueError, match=r"require 68 bytes, exceeding max_bytes=67"):
        load_safetensors(path, max_bytes=67)
    assert get_tensor_calls == []


def test_metadata_is_returned_validated_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "metadata"
    save_safetensors(
        path,
        {"x": np.ones(1, dtype=np.float32)},
        overwrite=False,
        metadata={"note": "long"},
        max_metadata_bytes=32,
    )

    with pytest.raises(ValueError, match="metadata requires .* exceeding max_metadata_bytes=0"):
        inspect_safetensors(path)
    with pytest.raises(ValueError, match="metadata requires .* exceeding max_metadata_bytes=1"):
        inspect_safetensors(path, max_metadata_bytes=1)

    invalid_metadata = cast(Mapping[str, str], {"revision": 3})
    with pytest.raises(TypeError, match="metadata value for key 'revision' must be a string"):
        save_safetensors(tmp_path / "invalid-metadata", {"x": np.ones(1)}, overwrite=False, metadata=invalid_metadata)


def test_corrupt_and_truncated_files_are_rejected(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(b"not a Safetensors file")
    with pytest.raises(SafetensorError):
        inspect_safetensors(corrupt)

    truncated = tmp_path / "truncated"
    save_safetensors(truncated, {"x": np.arange(4, dtype=np.float32)}, overwrite=False)
    contents = truncated.read_bytes()
    truncated.write_bytes(contents[:-1])
    with pytest.raises(SafetensorError):
        load_safetensors(truncated, max_bytes=16)


def test_overwrite_is_explicit_and_preserves_existing_values(tmp_path: Path) -> None:
    path = tmp_path / "weights"
    save_safetensors(path, {"x": np.array([1], dtype=np.int32)}, overwrite=False)

    with pytest.raises(FileExistsError, match="destination already exists"):
        save_safetensors(path, {"x": np.array([2], dtype=np.int32)}, overwrite=False)
    assert load_safetensors(path, max_bytes=4).tensors["x"].item() == 1

    save_safetensors(path, {"x": np.array([2], dtype=np.int32)}, overwrite=True)
    assert load_safetensors(path, max_bytes=4).tensors["x"].item() == 2


def test_save_rejects_symlink_destinations(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    destination = tmp_path / "weights"
    destination.symlink_to(target)

    with pytest.raises(ValueError, match="destination must not be a symbolic link"):
        save_safetensors(destination, {"x": np.ones(1, dtype=np.float32)}, overwrite=True)
    assert target.read_bytes() == b"unchanged"


def test_atomic_save_failure_preserves_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "weights"
    save_safetensors(destination, {"x": np.array([1], dtype=np.int32)}, overwrite=False)
    original = destination.read_bytes()

    def fail_after_partial_write(
        tensors: dict[str, npt.NDArray[np.generic]],
        filename: str | os.PathLike[str],
        metadata: dict[str, str] | None = None,
    ) -> None:
        del tensors, metadata
        Path(filename).write_bytes(b"partial")
        raise RuntimeError("injected serialization failure")

    monkeypatch.setattr(safetensors_module, "_save_file", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected serialization failure"):
        save_safetensors(destination, {"x": np.array([2], dtype=np.int32)}, overwrite=True)

    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def test_save_rejects_non_contiguous_and_unsupported_host_values(tmp_path: Path) -> None:
    non_contiguous = np.arange(8, dtype=np.float32)[::2]
    with pytest.raises(ValueError, match="tensor 'x' must be C-contiguous"):
        save_safetensors(tmp_path / "non-contiguous", {"x": non_contiguous}, overwrite=False)
    with pytest.raises(TypeError, match="tensor 'x' has unsupported NumPy dtype"):
        save_safetensors(tmp_path / "unsupported", {"x": np.array(["peptide"])}, overwrite=False)


def test_global_jax_arrays_require_explicit_host_conversion(tmp_path: Path) -> None:
    devices = np.array(jax.devices())
    sharding = NamedSharding(Mesh(devices, ("data",)), PartitionSpec("data"))
    global_array = jax.device_put(np.arange(len(devices), dtype=np.float32), sharding)

    with pytest.raises(TypeError, match=r"must be a numpy.ndarray on the host.*explicitly"):
        save_safetensors(
            tmp_path / "jax-array",
            {"x": cast(npt.NDArray[np.generic], global_array)},
            overwrite=False,
        )

    host_array = np.asarray(jax.device_get(global_array))
    save_safetensors(tmp_path / "host-array", {"x": host_array}, overwrite=False)
    restored = load_safetensors(tmp_path / "host-array", max_bytes=host_array.nbytes)
    np.testing.assert_array_equal(restored.tensors["x"], host_array)
