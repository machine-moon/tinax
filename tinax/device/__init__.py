"""Backend-agnostic pre-JAX environment policy, JAX runtime info, and optional detection."""

from tinax.device._detect import GpuInfo as GpuInfo
from tinax.device._detect import TpuInfo as TpuInfo
from tinax.device._detect import gpus as gpus
from tinax.device._detect import tpus as tpus
from tinax.device._env import configure_jax as configure_jax
from tinax.device._env import configure_single_chip as configure_single_chip
from tinax.device._env import set_visible_cuda as set_visible_cuda
from tinax.device._info import DeviceInfo as DeviceInfo
from tinax.device._info import device_info as device_info
