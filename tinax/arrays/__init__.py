"""Explicit NumPy, JAX, DLPack, and host array interoperation."""

from tinax.arrays._conversion import from_dlpack as from_dlpack
from tinax.arrays._conversion import from_numpy as from_numpy
from tinax.arrays._conversion import to_numpy as to_numpy
from tinax.arrays._inspection import ArrayInfo as ArrayInfo
from tinax.arrays._inspection import inspect_array as inspect_array
