"""Save, inspect, and bounded-load host tensors with tinax.weights."""

import tempfile
from pathlib import Path

import numpy as np

from tinax.weights import inspect_safetensors, load_safetensors, save_safetensors


def main() -> None:
    """Write two tensors, inspect the header, then load one within a byte budget."""
    tensors = {
        "encoder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "encoder.bias": np.zeros(3, dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.safetensors"
        save_safetensors(path, tensors, overwrite=True)

        info = inspect_safetensors(path)
        for name, tensor in info.tensors.items():
            print(f"{name}: shape={tensor.shape} dtype={tensor.dtype} nbytes={tensor.nbytes}")

        loaded = load_safetensors(path, max_bytes=1 << 20, names=["encoder.bias"])
        print(f"loaded_keys={sorted(loaded.tensors)}")


if __name__ == "__main__":
    main()
