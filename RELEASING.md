# Releasing

How to publish coordpy to PyPI as
[`coordpy-ai`](https://pypi.org/project/coordpy-ai/).

## Pre-flight

1. Bump the version in one place:

   ```
   coordpy/_version.py        # __version__ = "X.Y.Z"
   ```

   `pyproject.toml` reads it dynamically from that module via
   `[tool.setuptools.dynamic]`, so there is nothing else to edit.

2. Add an entry to `CHANGELOG.md` under
   `## [X.Y.Z]`.

3. Make sure main is clean and the wheel passes the gates:

   ```
   ./scripts/release.sh check
   ```

4. Tag the release:

   ```
   git tag -a vX.Y.Z -m "coordpy-ai vX.Y.Z"
   git push --follow-tags
   ```

## Recommended: PyPI Trusted Publisher

`.github/workflows/release.yml` builds and uploads on `v*` tags
using PyPI Trusted Publishers, so no API token is stored in the
repository.

One-time setup on PyPI
(<https://pypi.org/manage/account/publishing/>):

```
PyPI Project Name : coordpy-ai
Owner             : adotdong29
Repository        : context-zero
Workflow filename : release.yml
Environment       : pypi
```

In GitHub, create an environment named `pypi` under
**Settings → Environments**.

Then push a tag:

```
git tag -a v0.5.16 -m "coordpy-ai 0.5.16"
git push --follow-tags
```

The workflow builds, checks, smoke-tests, and uploads. See
<https://docs.pypi.org/trusted-publishers/> for background.

## Manual upload

You will need a PyPI API token from
<https://pypi.org/manage/account/token/>. Store it in `~/.pypirc`
(mode 600) or pass it via `TWINE_PASSWORD`.

### TestPyPI dry run

Recommended for first-time uploads of a new project.

```
python -m pip install --upgrade build twine
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload --repository testpypi dist/*
```

Verify in a clean venv:

```
python -m venv /tmp/cp-test
/tmp/cp-test/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    coordpy-ai
/tmp/cp-test/bin/python -c "import coordpy; print(coordpy.__version__)"
/tmp/cp-test/bin/coordpy --profile local_smoke --out-dir /tmp/cp-out
```

### Real PyPI

```
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Or just:

```
./scripts/release.sh upload
```

After upload, sanity-check from a fresh venv:

```
python -m venv /tmp/cp-prod
/tmp/cp-prod/bin/pip install coordpy-ai
/tmp/cp-prod/bin/python -c "import coordpy; print(coordpy.__version__)"
/tmp/cp-prod/bin/coordpy --profile local_smoke --out-dir /tmp/cp-out
```

## Versioning

- **MAJOR**: incompatible change to the report schema or the
  capsule contract. Don't ship without a deprecation cycle.
- **MINOR**: additive change to the public SDK; existing code
  keeps working.
- **PATCH**: bug fixes and ergonomic improvements with no API
  surface change.

The research-line tag (`coordpy.sdk.v3.4x`) lives separately on
`coordpy.SDK_VERSION` and is independent of the PyPI version.

## Yanking

If a release ships a critical bug, yank rather than delete:

```
python -m twine yank coordpy-ai==X.Y.Z --reason "<reason>"
```

Then publish a fix as `X.Y.(Z+1)`.
