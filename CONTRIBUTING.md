# Contributing

Thanks for considering a contribution. The package is small enough
that the rules fit on one page.

## Stable surface

The public contract is `coordpy.RunSpec` / `coordpy.run` /
`RunReport` and the four CLIs (`coordpy`, `coordpy-import`,
`coordpy-ci`, `coordpy-capsule`). Don't change the report shape or
the four on-disk schemas (`coordpy.capsule_view.v1`,
`coordpy.provenance.v1`, `phase45.product_report.v2`, the CI
verdict schema) without a version bump.

The capsule lifecycle invariants (T-1..T-7 in the team-coord audit
and C1..C4 in `capsule.py`) are load-bearing. Touching them needs
a matching test.

`coordpy.__experimental__` can move, rename, or disappear between
releases.

## Setup

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

PyPI distribution name is `coordpy-ai`; import name is `coordpy`.

## Smoke checks

Before pushing:

```bash
coordpy --profile local_smoke --out-dir /tmp/cp-smoke
coordpy-ci --report /tmp/cp-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule verify --report /tmp/cp-smoke/product_report.json
python tests/test_smoke_full.py
```

The full smoke driver runs in under five seconds and exercises
every documented public symbol.

## Pull requests

- One logical change per PR; keep diffs focused.
- Update or add tests for any behaviour change.
- If you touch a capsule schema, update
  `docs/CAPSULE_FORMALISM.md` and bump the schema version.
- Run `ruff`, `black`, and `mypy` (configured under
  `[project.optional-dependencies].dev`) before requesting review.

## Releasing

Full runbook: [`RELEASING.md`](RELEASING.md).

```bash
./scripts/release.sh check     # build, twine-check, smoke-test
./scripts/release.sh upload    # then push to PyPI
```

The recommended path is to push a `vX.Y.Z` tag and let
`.github/workflows/release.yml` upload via PyPI Trusted Publisher.
