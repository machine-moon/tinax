# Tinax

Tinax is a small, typed library of explicit productivity primitives for JAX, Flax NNX, Optax, Orbax, Grain, Chex, and Safetensors workflows.

It provides stable policies for array and RNG ownership, bounded diagnostics, NNX graph copies, explicit stdlib application boundaries, deterministic input pipelines, complete checkpoints, sharding, and weight interchange. Tested ecosystem recipes live under `examples/` without stable API guarantees.

## Requirements

- Python 3.12, 3.13, or 3.14

## Install

```bash
pip install tinax
```

Install a JAX accelerator distribution when needed:

```bash
pip install "tinax[gpu]"
pip install "tinax[tpu]"
```

See the [installation guide](https://tinax.org/installation/) for platform and accelerator details.

## Quick Start

```python
import numpy as np

from tinax.arrays import from_numpy, inspect_array, to_numpy

host = np.arange(8, dtype=np.float32)
device = from_numpy(host, copy=True)
info = inspect_array(device)
round_trip = to_numpy(device, writable=False)
```

Importing `tinax` alone is inert. Import the domain that owns the behavior you need.

## Documentation

- [Documentation](https://tinax.org/)
- [Changelog](CHANGES)
- [Contributing](docs/contributing.md)
- [Security policy](docs/security.md)
- [Release guide](docs/releases.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
