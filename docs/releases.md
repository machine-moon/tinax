# Releases

Publishing is automated. Pushing a `vX.Y.Z` tag runs the [release workflow](https://github.com/machine-moon/tinax/actions/workflows/release.yml), which tests, builds, and validates the sdist and wheel, publishes them to PyPI, and creates the matching GitHub release with the built artifacts attached.

## How tagging works

A tag is a separate ref from a branch, not part of any commit's metadata. `git push origin master` never pushes tags, and the release workflow never fires from an ordinary branch push. A tag only exists on GitHub once you push it explicitly:

```bash
git tag -a vX.Y.Z -m "Tinax X.Y.Z"   # create the tag locally, pointing at the current commit
git push origin vX.Y.Z               # push just that tag; this is what triggers the release
```

`-a` makes it an annotated tag (a real object with author, date, and message, the kind you want for releases) instead of a bare pointer. `-m` supplies that message, which only needs to be something recognizable to you later (`git show vX.Y.Z`, `git tag -n`) — the workflow's `gh release create --generate-notes` step builds the actual GitHub release notes from commit history, not from this message.

The workflow trigger matches on the tag's ref name alone (`v*.*.*`), so a lightweight tag (`git tag vX.Y.Z`, no `-a`/`-m`) publishes exactly the same way. Annotated is the convention this project uses since it's a permanent record, but nothing in the pipeline depends on it.

To release the branch you just pushed to `master`, tag its current HEAD after CI is green; the tag doesn't need to be created in any particular order relative to the branch push, only pushed separately afterward.

## Cutting a release

1. Update the `version` in `pyproject.toml` and add an entry to `CHANGES.md`, then commit and `git push origin master` as usual. This does **not** publish anything; it only runs CI.
2. Tag the commit and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "Tinax X.Y.Z"
   git push origin vX.Y.Z
   ```

3. Watch the `Release` workflow run. It fails closed: if the tag doesn't match `pyproject.toml`'s version, or if tests, lint, type checking, `zensical build`, `twine check`, or the inert-import wheel smoke test fail, nothing is published.
4. Once it succeeds, verify from a clean environment:

   ```bash
   uv venv --python 3.14 /tmp/tinax-pypi-check
   uv pip install --python /tmp/tinax-pypi-check/bin/python "tinax==X.Y.Z"
   /tmp/tinax-pypi-check/bin/python -I -c "import tinax"
   ```

Do not re-push a tag once it has been used for a publish; cut a new patch version instead.
