# Releasing

How to publish coordpy to PyPI as
[`coordpy-ai`](https://pypi.org/project/coordpy-ai/).

> Run release commands from a virtual environment.
> `./scripts/release.sh` creates and maintains its own at
> `.release-venv/`; pick the interpreter with
> `PYTHON=python3.12 ./scripts/release.sh ...`.

## Cut a release

1. **Bump the version** in `coordpy/_version.py`:

   ```
   __version__ = "X.Y.Z"
   ```

   `pyproject.toml` reads it via `[tool.setuptools.dynamic]`.
   Don't edit it anywhere else.

2. **Add a `## [X.Y.Z]` entry** to `CHANGELOG.md`.

3. **Commit** the bump and changelog:

   ```bash
   git add coordpy/_version.py CHANGELOG.md
   git commit -m "Release X.Y.Z"
   ```

4. **Verify the wheel passes the gates:**

   ```bash
   ./scripts/release.sh check
   ```

   This builds the sdist + wheel, runs `twine check`,
   `check-wheel-contents`, installs the wheel into a fresh venv,
   runs `tests/test_smoke_full.py`, then exercises the installed
   `coordpy` and `coordpy-capsule verify` entry points against the
   bundled `local_smoke` profile.

5. **Tag** the release. The tag must match `_version.py` exactly:

   ```bash
   git tag -a vX.Y.Z -m "coordpy-ai vX.Y.Z"
   git push --follow-tags
   ```

   `./scripts/release.sh upload` exits 3 (without uploading)
   unless **all three** of these hold:
   - HEAD has an exact-match git tag,
   - the tag is `v` + `coordpy._version.__version__`,
   - the working tree is clean.

   The Trusted Publisher workflow does not run that check, so it
   trusts the tag at face value. Always tag from a clean tree.

## Recommended path: PyPI Trusted Publisher

Once the project is registered on PyPI, pushing a `v*` tag
triggers `.github/workflows/release.yml`, which builds the sdist
+ wheel, runs the same gates as `release.sh check`, and uploads
via OIDC — no API token stored in the repo.

### One-time setup

Use this when configuring a new or replacement Trusted Publisher.
PyPI calls the config a **pending publisher** until the project
actually exists on PyPI; after the first successful upload it
converts into a regular Trusted Publisher automatically and there
is nothing more to configure.

At <https://pypi.org/manage/account/publishing/>, click **Add a
pending publisher** and fill in:

```
PyPI Project Name : coordpy-ai
Owner             : adotdong29
Repository        : CoordPy
Workflow filename : release.yml
Environment       : pypi
```

In GitHub, create a matching environment: **Settings →
Environments → New environment**, named `pypi`. Add required
reviewers under that environment if you want the upload step to
require a manual approval click.

If the OIDC exchange fails on the first push, the four field
names above must match the workflow exactly. PyPI returns 404
if any field is wrong.

### Push the tag

```bash
git tag -a vX.Y.Z -m "coordpy-ai vX.Y.Z"
git push --follow-tags
```

Watch the workflow under **Actions → Publish to PyPI**.

## Manual upload

You will need a PyPI API token from
<https://pypi.org/manage/account/token/>, configured one of two
ways.

`~/.pypirc` (run `chmod 600 ~/.pypirc` after writing it):

```
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmc...        # paste your token

[testpypi]
  repository = https://test.pypi.org/legacy/
  username = __token__
  password = pypi-AgENdGVzdC5weXBpLm9yZ...   # separate test token
```

Or env vars:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgE...
```

### TestPyPI dry run

Worth doing once for a brand-new project name. TestPyPI does
not allow re-uploading the same version, so if you need a
second attempt, append a `.devN` suffix (`X.Y.Z.dev1`) and
upload that.

```bash
./scripts/release.sh testpypi
```

Verify in a clean venv (the `--extra-index-url` is required,
because TestPyPI does not host most dependencies):

```bash
python -m venv /tmp/cp-test
/tmp/cp-test/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    coordpy-ai
/tmp/cp-test/bin/coordpy --profile local_smoke --out-dir /tmp/cp-out
```

(`local_smoke` is the bundled mock-mode profile that ships with
the package; it needs no network and finishes in well under a
second.)

### Real PyPI

```bash
./scripts/release.sh upload
```

After upload, sanity-check from a fresh venv. The new version
typically lands within a minute, but PyPI's CDN can take up to
five before `pip install` sees it:

```bash
python -m venv /tmp/cp-prod
/tmp/cp-prod/bin/pip install --upgrade pip
/tmp/cp-prod/bin/pip index versions coordpy-ai     # confirm the new version is listed
/tmp/cp-prod/bin/pip install coordpy-ai
/tmp/cp-prod/bin/python -c "import coordpy; print(coordpy.__version__)"
/tmp/cp-prod/bin/coordpy --profile local_smoke --out-dir /tmp/cp-out
```

If `release.sh upload` fails partway through (for example, a
network drop during `twine upload`), the wheel may already be on
PyPI even though the script returned non-zero. Check with `pip
index versions coordpy-ai` before retrying. Re-uploading the
same filename is rejected by PyPI; if the version landed but
something else broke, bump to the next patch version and
release that.

## Yanking

If a release ships a critical bug, yank it; don't delete it.
PyPI does not currently expose a CLI for yank, so use the
project admin UI:
<https://pypi.org/manage/project/coordpy-ai/> (this URL only
resolves once the project has been published at least once;
before that it returns 404). Click the affected version, then
**Options → Yank**, and provide a reason.

Yanked releases stay installable when explicitly pinned
(`coordpy-ai==X.Y.Z`) but are skipped by plain
`pip install coordpy-ai`. Publish a fix as `X.Y.(Z+1)`.

## Versioning

Standard semantic versioning, with one project-specific rule:
the report schema and the capsule contract are part of the
public surface and bumping either of them requires a major
version bump (and a deprecation cycle).

The research-line tag exposed at `coordpy.SDK_VERSION`
(currently `coordpy.sdk.v3.43`) is independent of the PyPI
version and tracks the research programme rather than the
shipped contract.
