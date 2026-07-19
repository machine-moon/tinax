# Contributing to Tinax

Tinax values small, explicit changes with focused tests.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check tinax tests examples
uv run ty check tinax tests examples
```

Run `uv build` for packaging changes. Public API must remain independent except at explicit integration boundaries, and `import tinax` must remain inert.

## Changes

- Add focused tests for changed behavior.
- Update `CHANGES.md` for user-visible changes.
- Keep `examples/` clearly outside the stable API contract.
- Do not add lint or type-checker suppressions.

Report security vulnerabilities privately under the [security policy](security.md), not through public issues.
