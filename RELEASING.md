# Releasing CoordPy

This document is the **runbook** for publishing CoordPy to PyPI under
the distribution name [``coordpy-ai``](https://pypi.org/project/coordpy-ai/).

## Pre-flight (every release)

1. Bump the version in **one** place:

       coordpy/_version.py        # __version__ = "X.Y.Z"

   ``pyproject.toml`` reads the version dynamically from that
   module via ``[tool.setuptools.dynamic]`` so there is nothing else
   to edit.

2. Add a new entry to ``CHANGELOG.md`` under
   ``## [X.Y.Z] — short title``.

3. Make sure ``main`` is clean and the smoke suite passes against
   a freshly-built wheel:

       ./scripts/release.sh check

4. Tag the release:

       git tag -a vX.Y.Z -m "coordpy-ai vX.Y.Z"
       git push --follow-tags

## Recommended path: PyPI Trusted Publisher (no API token needed)

CoordPy ships a GitHub Actions workflow at
``.github/workflows/release.yml`` that builds the sdist + wheel and
uploads to PyPI via [Trusted Publishers][tp] when you push a
``v*`` tag. Set up the project once in PyPI:

1. Go to <https://pypi.org/manage/account/publishing/> and add a
   pending publisher with these fields:

       PyPI Project Name : coordpy-ai
       Owner             : adotdong29
       Repository        : context-zero
       Workflow filename : release.yml
       Environment       : pypi

2. In your GitHub repository, create an environment named ``pypi``
   under **Settings → Environments**.

3. Push a tag:

       git tag -a v0.5.16 -m "coordpy-ai 0.5.16"
       git push --follow-tags

   The workflow builds, ``twine check``-s, and uploads automatically.

[tp]: https://docs.pypi.org/trusted-publishers/

## Manual path: build + upload from your laptop

You will need a PyPI API token from
<https://pypi.org/manage/account/token/>. Save it in
``~/.pypirc`` (mode ``600``) or pass it via the
``TWINE_PASSWORD`` env var.

### TestPyPI dry run (always do this first for a new project)

    python -m pip install --upgrade build twine
    rm -rf dist build *.egg-info
    python -m build
    python -m twine check dist/*
    python -m twine upload --repository testpypi dist/*

Then verify in a clean venv:

    python -m venv /tmp/coordpy-test
    /tmp/coordpy-test/bin/pip install \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        coordpy-ai
    /tmp/coordpy-test/bin/python -c "import coordpy; print(coordpy.__version__)"
    /tmp/coordpy-test/bin/coordpy --profile local_smoke --out-dir /tmp/cp-test

### Real PyPI upload

    rm -rf dist build *.egg-info
    python -m build
    python -m twine check dist/*
    python -m twine upload dist/*

(Or equivalently: ``./scripts/release.sh upload``.)

After upload, sanity-check by installing from PyPI in a clean venv:

    python -m venv /tmp/coordpy-prod
    /tmp/coordpy-prod/bin/pip install coordpy-ai
    /tmp/coordpy-prod/bin/python -c "import coordpy; print(coordpy.__version__)"
    /tmp/coordpy-prod/bin/coordpy --profile local_smoke --out-dir /tmp/cp-prod

## Version naming convention

CoordPy follows semantic versioning at the wire level:

- **MAJOR** — incompatible change to the report schema or the
  capsule contract. Don't ship without a deprecation cycle.
- **MINOR** — additive change to the public SDK; old code keeps
  working byte-for-byte.
- **PATCH** — bug fixes and ergonomic improvements that don't add
  or remove API.

The SDK research-line tag (``coordpy.sdk.v3.4x``) lives separately
in ``coordpy.SDK_VERSION`` and is independent from the PyPI
version.

## Yanking a release

If a release ships with a critical bug, yank it from PyPI rather
than deleting it:

    python -m twine yank coordpy-ai==X.Y.Z --reason "<reason>"

Then publish a fix as ``X.Y.(Z+1)``.
