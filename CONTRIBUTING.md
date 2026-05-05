# Contributing to CoordPy

Thanks for thinking about contributing. CoordPy ships out of the
[Context Zero](https://github.com/adotdong29/context-zero) research
programme; this document is the short version of what we expect from
patches.

## Ground rules

- The **stable surface** is the SDK contract: ``coordpy.RunSpec`` /
  ``coordpy.run`` / ``RunReport`` and the CLIs (``coordpy``,
  ``coordpy-import``, ``coordpy-ci``, ``coordpy-capsule``). Don't
  break the byte-shape of the report or the four capsule-graph
  schemas (``coordpy.capsule_view.v1``, ``coordpy.provenance.v1``,
  ``phase45.product_report.v2``, the CI verdict schema) without an
  explicit version bump.
- Everything under ``coordpy.__experimental__`` is research
  surface. It can move, rename, or be withdrawn between releases.
- Capsule lifecycle invariants (T-1..T-7 in the team-coord audit and
  C1..C4 in ``capsule.py``) are load-bearing. Touching them needs a
  matching test and a results note.

## Development setup

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The distribution name on PyPI is ``coordpy-ai``; the import name is
``coordpy``.

Run the smoke test before you push anything:

```bash
coordpy --profile local_smoke --out-dir /tmp/cp-smoke
coordpy-ci --report /tmp/cp-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule verify --report /tmp/cp-smoke/product_report.json
```

Tests live under ``coordpy/tests/`` (when present) and are run with
``pytest``.

## Pull requests

1. Open against ``main``. Keep diffs focused — one logical change per
   PR.
2. Update or add tests for any behaviour change.
3. If you touch a capsule schema, update
   ``docs/CAPSULE_FORMALISM.md`` and bump the schema version.
4. Run ``ruff`` / ``black`` / ``mypy`` (all configured under
   ``[project.optional-dependencies].dev``) before requesting review.

## Releasing

Full runbook: [`RELEASING.md`](RELEASING.md).

Quick path (manual upload):

```bash
./scripts/release.sh check     # build + twine-check + smoke-test the wheel
./scripts/release.sh upload    # then push to PyPI
```

Recommended path: push a ``vX.Y.Z`` tag — the
``.github/workflows/release.yml`` Trusted Publisher workflow
builds and uploads to PyPI automatically. See
[`RELEASING.md`](RELEASING.md) for one-time PyPI Trusted
Publisher configuration.
