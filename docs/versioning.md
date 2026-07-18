# Versioning

Tinax follows Semantic Versioning: `MAJOR.MINOR.PATCH`. The `version` in `pyproject.toml` is the sole authoritative version source.

## Pre-1.0 Policy

Tinax is currently pre-1.0. Public imports and documented behavior are treated carefully, but incompatible changes may occur in minor releases when they improve correctness or clarify an unsafe contract. Release notes describe required migrations.

Patch releases fix defects without intentionally changing documented public behavior. Minor releases may add stable APIs, deprecate APIs, or make incompatible corrections. Major releases establish stronger compatibility guarantees.

The compatibility surface includes stable domain imports, documented function signatures, checkpoint formats, and Safetensors manifest semantics. Modules under `examples/` are tested recipes and have no stable API guarantee.

## Release Tags

Published releases use tags in the form `vX.Y.Z`, matched by the [release workflow](https://github.com/machine-moon/tinax/actions/workflows/release.yml) on the ref name alone; annotated (`git tag -a`) and lightweight tags both trigger it. Annotated is still preferred for its permanent author/date/message record, but nothing in the pipeline depends on that metadata. A release candidate may use `vX.Y.Z-rcN`; candidates are not promoted automatically and must be rebuilt after changes.
