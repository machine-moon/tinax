"""Optional accelerator detection gated behind the gpu and tpu extras."""

import importlib as _importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuInfo:
    """A detected NVIDIA GPU; ``compute_capability`` is comparable via ``float()``."""

    index: int
    name: str
    total_bytes: float
    free_bytes: float
    compute_capability: str


@dataclass(frozen=True)
class TpuInfo:
    """A detected local TPU chip generation, its per-chip specs, and how many are present.

    ``generation`` is the lowercase chip name (for example ``"v5e"``); ``devices_per_chip``
    and ``hbm_gib`` are the fixed specs that ``tpu-info`` records for that generation.
    """

    generation: str
    count: int
    devices_per_chip: int
    hbm_gib: int


def gpus() -> list[GpuInfo]:
    """Probe visible NVIDIA GPUs through ``nvidia-ml-py`` (imported as ``pynvml``).

    Returns:
        The visible NVIDIA GPUs in device-index order, or an empty list if none are
        present. Memory fields are byte counts.

    Raises:
        ImportError: If ``nvidia-ml-py`` is unavailable; install ``tinax[gpu]``.
    """
    try:
        pynvml = _importlib.import_module("pynvml")
    except ImportError as error:
        raise ImportError("GPU detection needs the optional extra: install tinax[gpu].") from error
    pynvml.nvmlInit()
    try:
        detected: list[GpuInfo] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            detected.append(GpuInfo(index, name, float(memory.total), float(memory.free), f"{major}.{minor}"))
        return detected
    finally:
        pynvml.nvmlShutdown()


def tpus() -> list[TpuInfo]:
    """Probe local TPU chips through ``tpu-info``.

    Detection is a local PCI scan; it needs neither ``libtpu`` nor a running runtime.

    Returns:
        A single-entry list describing the local TPU chip generation, its per-chip
        specs, and count, or an empty list when no local TPU chips are present.

    Raises:
        ImportError: If ``tpu-info`` is unavailable; install ``tinax[tpu]``.
    """
    try:
        device = _importlib.import_module("tpu_info.device")
    except ImportError as error:
        raise ImportError("TPU detection needs the optional extra: install tinax[tpu].") from error
    chip, count = device.get_local_chips()
    if chip is None:
        return []
    specs = chip.value
    return [
        TpuInfo(
            generation=str(specs.name),
            count=int(count),
            devices_per_chip=int(specs.devices_per_chip),
            hbm_gib=int(specs.hbm_gib),
        )
    ]
