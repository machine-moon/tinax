"""Explicit NumPy, JAX, DLPack interoperation and validated array operations."""

from tinax.array._conversion import from_dlpack as from_dlpack
from tinax.array._conversion import from_numpy as from_numpy
from tinax.array._conversion import to_numpy as to_numpy
from tinax.array._inspection import ArrayInfo as ArrayInfo
from tinax.array._inspection import inspect_array as inspect_array
from tinax.array._ops import one_hot as one_hot
from tinax.array._ops import safe_astype as safe_astype
from tinax.array._ops import stack_batch as stack_batch
