# Installation

Tinax supports CPython 3.14 with the standard GIL-enabled build. Free-threaded CPython is unsupported because Grain 0.2.18 does not publish a `cp314t` wheel.

Supported platforms are glibc Linux x86-64/AArch64, Windows x86-64, and Apple Silicon macOS. Tinax pins the JAX ecosystem versions it tests.

## CPU

```bash
pip install tinax
```

## Accelerators

Install the appropriate JAX distribution explicitly:

```bash
pip install "tinax[gpu]"
pip install "tinax[tpu]"
```

Confirm that your selected JAX extra supports your platform and accelerator runtime before installation.

## Development

Use the locked development environment from a source checkout:

```bash
uv sync --locked
uv run pytest
```
