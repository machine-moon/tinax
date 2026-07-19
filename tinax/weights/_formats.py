"""Safetensors dtype sizes and NumPy conversion rules."""

import numpy as np


def dtype_bits(dtype: str) -> int:
    """Return the serialized bits per logical element."""
    match dtype:
        case "F4":
            return 4
        case "F6_E2M3" | "F6_E3M2":
            return 6
        case "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ" | "F8_E5M2FNUZ":
            return 8
        case "I16" | "U16" | "F16" | "BF16":
            return 16
        case "I32" | "U32" | "F32":
            return 32
        case "C64" | "F64" | "I64" | "U64":
            return 64
        case _:
            raise ValueError(f"unsupported Safetensors dtype in file: {dtype!r}")


def serialized_nbytes(shape: tuple[int, ...], dtype: str) -> int:
    """Return a tensor's byte-aligned serialized payload size."""
    elements = 1
    for dimension in shape:
        elements *= dimension
    bit_count = elements * dtype_bits(dtype)
    if bit_count % 8:
        raise ValueError(f"tensor shape {shape} is not byte-aligned for dtype {dtype!r}")
    return bit_count // 8


def numpy_dtype(dtype: str) -> np.dtype[np.generic] | None:
    """Return the active NumPy dtype for a materializable format dtype."""
    match dtype:
        case "BOOL":
            name = "bool"
        case "U8":
            name = "uint8"
        case "I8":
            name = "int8"
        case "I16":
            name = "int16"
        case "U16":
            name = "uint16"
        case "I32":
            name = "int32"
        case "U32":
            name = "uint32"
        case "I64":
            name = "int64"
        case "U64":
            name = "uint64"
        case "F16" | "F32" | "F64":
            name = f"float{dtype[1:]}"
        case "BF16":
            name = "bfloat16"
        case "C64":
            name = "complex64"
        case _:
            return None
    try:
        return np.dtype(name)
    except TypeError:
        return None


def supports_serialization(dtype: np.dtype[np.generic]) -> bool:
    """Return whether Safetensors can serialize the NumPy dtype."""
    match dtype.name:
        case (
            "bool"
            | "uint8"
            | "int8"
            | "int16"
            | "uint16"
            | "float16"
            | "bfloat16"
            | "int32"
            | "uint32"
            | "float32"
            | "complex64"
            | "float64"
            | "int64"
            | "uint64"
        ):
            return True
        case _:
            return False
