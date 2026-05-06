# Releasing

How to publish coordpy to PyPI as
[`coordpy-ai`](https://pypi.org/project/coordpy-ai/).

> Run release commands from a virtual environment. The build and
> upload tooling needs `pip install --upgrade`, which fails on
> distro-managed Python interpreters (Debian, Ubuntu, RHEL).
> `./scripts/release.sh` handles this by maintaining its own
> private venv at `.release-venv/`. The manual path below assumes
> you are already inside a venv.

## Pre-flight

1. Bump the version in one place:

   ```
   coordpy/_version.py        # __version__ = "X.Y.Z"
   ```

   `pyproject.toml` reads it dynamically via
   `[tool.setuptools.dynamic]`.

2. Add an entry to `CHANGELOG.md` under `## [X.Y.Z]`.

3. Commit the version bump and changelog before tagging:

   ```
   git add coordpy/_version.py CHANGELOG.md
   git commit -m "Release X.Y.Z"
   ```

4. Verify the wheel passes the gates:

   ```
   ./scripts/release.sh check
   ```

   This builds the sdist + wheel, runs `twine check`,
   `check-wheel-contents`, installs the wheel into a fresh venv,
   and runs `tests/test_smoke_full.py` plus
   `examples/build_with_coordpy.py`.

5. Tag the release. The tag must match `coordpy/_version.py`
   exactly:

   ```
   git tag -a vX.Y.Z -m "coordpy-ai vX.Y.Z"
   git push --follow-tags
   ```

   `./scripts/release.sh upload` enforces the tag-matches-version
   invariant; the GitHub workflow does not, so be careful when
   tagging by hand.

## Recommended: PyPI Trusted Publisher

`.github/workflows/release.yml` builds and uploads on `v*` tags
using PyPI Trusted Publishers. No API token is stored in the repo.

One-time setup at <https://pypi.org/manage/account/publishing/>:

```
PyPI Project Name : coordpy-ai
Owner             : adotdong29
Repository        : context-zero
Workflow filename : release.yml
Environment       : pypi
```

In GitHub, create an environment named `pypi` under
**Settings → Environments**.

Push from a clean working tree:

```
git tag -a v0.5.16 -m "coordpy-ai 0.5.16"
git push --follow-tags
```

The workflow builds, checks, smoke-tests, and uploads. See
<https://docs.pypi.org/trusted-publishers/> for background.

## Manual upload

You will need:

- Python 3.10+ inside a virtual environment.
- `build`, `twine`, and `check-wheel-contents` installed in that
  venv. (`./scripts/release.sh` will install them automatically.)
- A PyPI API token from <https://pypi.org/manage/account/token/>,
  in `~/.pypirc` (mode 600) or in the `TWINE_PASSWORD` env var.

### TestPyPI dry run

Recommended for first-time uploads of a new project name.

```
./scripts/release.sh testpypi
```

Or by hand from inside a venv:

```
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m check_wheel_contents dist/*.whl
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
./scripts/release.sh upload
```

Or by hand from inside a venv:

```
rm -rf dist build *.egg-info
python -m build
python -m twine check dist/*
python -m check_wheel_contents dist/*.whl
python -m twine upload dist/*
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

## Yanking a release

If a release ships a critical bug, yank it (don't delete it).
PyPI does not currently expose a CLI for this; use the project
admin UI:

<https://pypi.org/manage/project/coordpy-ai/releases/>

Click the affected version, then **Options → Yank**, and provide
a reason. Yanked releases stay installable when explicitly pinned
(`coordpy-ai==X.Y.Z`) but are skipped by `pip install coordpy-ai`.

Then publish a fix as `X.Y.(Z+1)`.
