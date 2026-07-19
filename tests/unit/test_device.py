import subprocess
import sys
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from tinax.device import (
    DeviceInfo,
    GpuInfo,
    TpuInfo,
    configure_jax,
    configure_single_chip,
    device_info,
    set_visible_cuda,
    tpus,
)
from tinax.device._validation import require_jax_unimported, validate_device_index

_HAPPY_PATH = """
import os
from tinax.device import configure_jax, configure_single_chip, set_visible_cuda

set_visible_cuda([2, 0, 1])
configure_single_chip(3)
configure_jax(preallocate=False, matmul_precision=None)

assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,0,1", os.environ["CUDA_VISIBLE_DEVICES"]
assert os.environ["TPU_PROCESS_BOUNDS"] == "1,1,1", os.environ["TPU_PROCESS_BOUNDS"]
assert os.environ["TPU_VISIBLE_CHIPS"] == "3", os.environ["TPU_VISIBLE_CHIPS"]
assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
assert "jax" not in __import__("sys").modules, "environment policy must stay pre-JAX"
"""


def test_environment_policy_applies_before_jax_in_a_fresh_interpreter() -> None:
    completed = subprocess.run([sys.executable, "-c", _HAPPY_PATH], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mutator", [lambda: set_visible_cuda([0]), lambda: configure_single_chip(0)])
def test_environment_policy_refuses_to_run_after_jax_is_imported(mutator: Callable[[], None]) -> None:
    import jax

    assert jax is not None
    with pytest.raises(RuntimeError, match="before importing JAX"):
        mutator()


@pytest.mark.parametrize("preallocate", [1, "true", None])
def test_configure_jax_rejects_a_non_bool_preallocate(preallocate: object) -> None:
    with pytest.raises(TypeError, match="preallocate"):
        configure_jax(preallocate=cast(bool, preallocate))


def test_configure_jax_rejects_a_non_string_matmul_precision() -> None:
    with pytest.raises(TypeError, match="matmul_precision"):
        configure_jax(matmul_precision=cast(str, 3))


@pytest.mark.parametrize("index", [True, 1.5, "0", None])
def test_validate_device_index_requires_a_non_bool_integer(index: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        validate_device_index(index, "chip")


def test_validate_device_index_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        validate_device_index(-1, "chip")


def test_require_jax_unimported_detects_the_imported_runtime() -> None:
    import jax

    assert jax is not None
    with pytest.raises(RuntimeError, match="before importing JAX"):
        require_jax_unimported("op()")


def test_device_info_summarizes_the_cpu_runtime() -> None:
    info = device_info()

    assert isinstance(info, DeviceInfo)
    assert info.backend == "cpu"
    assert info.device_count == 4
    assert info.local_device_count == 4
    assert info.process_index == 0
    assert info.process_count == 1
    assert len(info.device_kinds) == 4


def test_tpus_returns_empty_on_a_host_without_local_tpu() -> None:
    pytest.importorskip("tpu_info")

    assert tpus() == []


def test_detection_structs_are_immutable() -> None:
    gpu = GpuInfo(index=0, name="A100", total_bytes=1.0, free_bytes=0.5, compute_capability="8.0")
    tpu = TpuInfo(generation="v5e", count=4, devices_per_chip=1, hbm_gib=16)

    assert float(gpu.compute_capability) == 8.0
    with pytest.raises(FrozenInstanceError):
        setattr(gpu, "free_bytes", 0.0)
    with pytest.raises(FrozenInstanceError):
        setattr(tpu, "count", 8)
