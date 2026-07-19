"""Convert, inspect, and safely operate on arrays with tinax.array."""

import numpy as np

from tinax.array import from_numpy, inspect_array, one_hot, safe_astype, stack_batch, to_numpy


def main() -> None:
    """Round-trip a host array through JAX and exercise the validated operations."""
    host = np.arange(6, dtype=np.float32).reshape(2, 3)
    device_array = from_numpy(host, copy=True)

    info = inspect_array(device_array)
    print(f"shape={info.shape} dtype={info.dtype} logical_nbytes={info.logical_nbytes}")

    batched = stack_batch([device_array, device_array + 1.0])
    print(f"stacked shape={batched.shape}")

    small_ints = from_numpy(np.array([1, 2, 3], dtype=np.int16), copy=True)
    widened = safe_astype(small_ints, np.int32)
    print(f"widened dtype={widened.dtype}")

    indices = from_numpy(np.array([0, 2, 1], dtype=np.int32), copy=True)
    encoded = one_hot(indices, num_classes=3)
    print(f"one_hot=\n{to_numpy(encoded, writable=False)}")


if __name__ == "__main__":
    main()
