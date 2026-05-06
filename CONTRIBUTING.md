# Contributing

Thanks for considering a contribution. The package is small enough
that the rules fit on one page.

## Stable surface

The public contract is `coordpy.RunSpec` / `coordpy.run` /
`RunReport` and the four CLIs (`coordpy`, `coordpy-import`,
`coordpy-ci`, `coordpy-capsule`). Don't change the report shape
or the four on-disk schemas without a version bump.

The schemas and where they live:

| Schema constant | Defined in |
|---|---|
| `CAPSULE_VIEW_SCHEMA` (`coordpy.capsule_view.v1`) | `coordpy/capsule.py` |
| `PROVENANCE_SCHEMA` (`coordpy.provenance.v1`) | `coordpy/provenance.py` |
| `PRODUCT_REPORT_SCHEMA` (`phase45.product_report.v2`) | `coordpy/_internal/product/runner.py` |
| `CI_VERDICT_SCHEMA` | `coordpy/_internal/product/ci_gate.py` |
| `IMPORT_AUDIT_SCHEMA` | `coordpy/_internal/product/import_data.py` |

The capsule lifecycle invariants are load-bearing. Changes need a
matching test:

- Capsule construction invariants C1..C4 — `coordpy/capsule.py`,
  on `ContextCapsule.new`, `CapsuleLedger.admit`,
  `CapsuleLedger.seal`.
- Team-coordination lifecycle invariants T-1..T-7 —
  `coordpy/lifecycle_audit.py` and `coordpy/team_coord.py`.

`coordpy.__experimental__` (defined in `coordpy/__init__.py`) can
move, rename, or disappear between releases. Pin against the
experimental tuple if you depend on it.

## Setup

Supported Python versions: 3.10, 3.11, 3.12, 3.13, 3.14.

```bash
git clone https://github.com/adotdong29/context-zero.git
cd context-zero
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

PyPI distribution name is `coordpy-ai`; import name is `coordpy`.

The `[dev]` extra installs `ruff`, `black`, `mypy`, `pytest`,
`build`, and `twine`. Lint and type-check rules live under
`[tool.ruff]`, `[tool.black]`, `[tool.mypy]` in `pyproject.toml`.
Run the same commands CI runs:

```bash
ruff check .
black --check .
mypy coordpy
```

## Smoke checks

Before pushing:

```bash
./scripts/smoke.sh
```

Or run the steps individually:

```bash
coordpy --profile local_smoke --out-dir /tmp/cp-smoke
coordpy-ci --report /tmp/cp-smoke/product_report.json --min-pass-at-1 1.0
coordpy-capsule verify --report /tmp/cp-smoke/product_report.json
python tests/test_smoke_full.py
```

`local_smoke` writes ~7 small artefacts to the `--out-dir`
(`product_report.json`, `capsule_view.json`, `provenance.json`,
`meta_manifest.json`, `readiness_verdict.json`,
`product_summary.txt`, `sweep_result.json`). The directory is
disposable; nothing else writes there.

The full smoke driver runs in under five seconds and exercises
every documented public symbol.

## Pull requests

- One logical change per PR; keep diffs focused.
- Update or add tests for any behaviour change.
- If you touch a schema, bump the schema version in the file
  listed above and update the test that pins it.
- Run `ruff`, `black`, `mypy`, and the smoke driver before review.

## Releasing

The version lives in **one place**: `coordpy/_version.py`
(`__version__ = "X.Y.Z"`). `pyproject.toml` reads it dynamically.

Full runbook: [`RELEASING.md`](RELEASING.md). Quick path:

```bash
./scripts/release.sh check     # build, twine-check, smoke-test
./scripts/release.sh upload    # then push to PyPI
```

The recommended path is to push a `vX.Y.Z` tag and let
`.github/workflows/release.yml` upload via PyPI Trusted Publisher.
