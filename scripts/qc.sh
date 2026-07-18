#!/usr/bin/env sh

set +e

uv sync

echo "=== Ruff Linting ==="
uv run ruff check tinax tests examples

echo "=== Ty Type Checking ==="
uv run ty check tinax tests examples

echo "=== Pytest Unit Tests ==="
uv run pytest -v tests

