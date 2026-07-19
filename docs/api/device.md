# Device

`tinax.device` owns backend-agnostic pre-JAX environment policy that the JAX runtime honors, a summary of the active JAX runtime, and optional accelerator detection gated behind the `gpu` and `tpu` extras. It does not select hardware or apply out-of-memory heuristics; those policies belong to the application.

```python
from tinax.device import configure_jax, configure_single_chip, device_info, set_visible_cuda

set_visible_cuda([0, 1])          # before importing jax
configure_single_chip(0)          # expose one local TPU chip before importing jax
configure_jax(preallocate=False, matmul_precision="high")

info = device_info()              # imports jax on demand
```

Detection needs the matching extra (`pip install tinax[gpu]` or `tinax[tpu]`):

```python
from tinax.device import gpus, tpus

for gpu in gpus():
    print(gpu.index, gpu.name, gpu.free_bytes)
```

::: tinax.device.set_visible_cuda

::: tinax.device.configure_single_chip

::: tinax.device.configure_jax

::: tinax.device.device_info

::: tinax.device.DeviceInfo

::: tinax.device.gpus

::: tinax.device.tpus

::: tinax.device.GpuInfo

::: tinax.device.TpuInfo
