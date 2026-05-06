# Contributing

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
`build`, and `twine`. Tool config lives under `[tool.ruff]`,
`[tool.black]`, `[tool.mypy]` in `pyproject.toml`.

## Pre-push gate

Before every push, run:

```bash
./scripts/smoke.sh
```

That runs `ruff`, then the four console scripts in order
(`coordpy --profile local_smoke`, `coordpy-ci`,
`coordpy-capsule verify`, `coordpy-import` against a bundled
fixture), then `tests/test_smoke_full.py`. CI runs the same
set, so a clean local `smoke.sh` should mean a green CI.

`black` and `mypy` are in the `[dev]` extra. They aren't required
to push; run them on files you touch if you want.

To run the full pytest suite or a single test:

```bash
pytest                              # whole suite
pytest -k smoke                     # just smoke tests
pytest tests/test_smoke_full.py -v  # one file, verbose
```

## Public surface

The public contract is `coordpy.RunSpec` / `coordpy.run` /
`RunReport` and the four console scripts.

The capsule lifecycle invariants are load-bearing. Touching one
needs a regression test that names the invariant in its
docstring.

Capsule construction and admission, all in `coordpy/capsule.py`:

- **C1**: a capsule's CID is a SHA-256 of its canonical content,
  so identical payloads have identical CIDs.
- **C2**: every capsule has a kind from `CapsuleKind.ALL`.
- **C3**: admission honours the capsule's byte / parent
  budget; oversize raises `CapsuleAdmissionError`.
- **C4**: `seal` is monotonic — sealed capsules are immutable.

Team-coordination lifecycle, in `coordpy/lifecycle_audit.py` and
`coordpy/team_coord.py`:

- **T-1..T-3**: every TEAM_HANDOFF declares one source role,
  one target role, and parents reachable from the source's
  ROLE_VIEW.
- **T-4..T-5**: every TEAM_DECISION has parents that include
  every TEAM_HANDOFF it adjudicates.
- **T-6..T-7**: ROLE_VIEW and TEAM_DECISION transitions are
  monotonic — once sealed, they stay reachable from the run's
  RUN_REPORT capsule.

The five on-disk schemas are pinned to specific files and tests:

| Schema constant | Defined in | Pinned by `tests/test_smoke_full.py` |
|---|---|---|
| `CAPSULE_VIEW_SCHEMA` (`coordpy.capsule_view.v1`) | `coordpy/capsule.py` | yes |
| `PROVENANCE_SCHEMA` (`coordpy.provenance.v1`) | `coordpy/provenance.py` | yes |
| `PRODUCT_REPORT_SCHEMA` (`phase45.product_report.v2`) | `coordpy/_internal/product/runner.py` | yes |
| `CI_VERDICT_SCHEMA` | `coordpy/_internal/product/ci_gate.py` | indirectly |
| `IMPORT_AUDIT_SCHEMA` | `coordpy/_internal/product/import_data.py` | indirectly |

Don't change any of these without a version bump (procedure below).

`coordpy.__experimental__` (defined in `coordpy/__init__.py`) is
a research surface. Pin against the `__experimental__` tuple if
you depend on it. Promotion to stable is a release-cycle
decision, not a PR-time one.

## Bumping a schema

Schemas use suffixed semantic versions like `coordpy.foo.v1`. The
recipe is the same in every case:

1. **Add a new constant.** Don't mutate the old one. For example,
   to bump `PROVENANCE_SCHEMA`:

   ```python
   # coordpy/provenance.py
   PROVENANCE_SCHEMA   = "coordpy.provenance.v2"   # new
   PROVENANCE_SCHEMA_V1 = "coordpy.provenance.v1"  # keep for readers
   ```

2. **Update producers.** Find the code that emits the schema
   and bump the string it writes. For `PROVENANCE_SCHEMA` that
   producer is `build_manifest` in `coordpy/provenance.py`; for
   the report and CI/import schemas, look in the matching file
   listed in the table above. If the on-disk shape changes, add
   a reader helper that accepts both versions during the
   migration window:

   ```python
   def parse_provenance(d: dict) -> dict:
       schema = d.get("schema")
       if schema == PROVENANCE_SCHEMA:        # new
           return _parse_v2(d)
       if schema == PROVENANCE_SCHEMA_V1:     # old
           return _parse_v1(d)
       raise ValueError(f"unknown provenance schema: {schema!r}")
   ```

3. **Update `tests/test_smoke_full.py`.** It pins each schema
   constant by literal value; bump those literals.

4. **Add a `CHANGELOG.md` entry** noting the schema bump and any
   migration steps for downstream consumers.

Backwards-incompatible schema changes need a major version
bump; see [`RELEASING.md`](RELEASING.md) for the release-side
procedure.

## Debugging a failing capsule chain

When a run reports `chain_ok=False` or `coordpy-capsule verify`
prints a verdict other than `OK`, work through these steps in
order:

1. **Read the verdict line.** `coordpy-capsule verify` shows four
   sub-checks (chain recompute embedded / on-disk view agreement
   / artefacts on disk / meta_manifest on disk). The first one
   to fail is usually the proximate cause.

2. **Check `--full` view.** `coordpy-capsule view --full --report
   <path>` prints every capsule with its CID, kind, lifecycle,
   parent count, and byte/token counts. A wrong parent count or
   an unexpected `RETIRED` lifecycle is often the smoking gun.

3. **Diff producer vs. consumer.** If the embedded chain is OK
   but the on-disk view disagrees, something rewrote the file
   between sealing and verification. Re-run with a fresh
   `--out-dir` to rule out a stale or shared output directory,
   then compare the embedded view against the on-disk view:

   ```bash
   diff <(jq -S '.capsules' product_report.json) \
        <(jq -S . capsule_view.json)
   ```

4. **Re-run with the smoke profile.** If `local_smoke` itself
   fails, treat it as a regression and bisect recent changes to
   `coordpy/capsule.py`, `coordpy/capsule_runtime.py`, or
   `coordpy/_internal/product/runner.py`.

5. **Run the lifecycle audit.** `coordpy-capsule audit --report
   <path>` checks eleven L-1..L-11 rules. Each rule is named in
   the module docstring at the top of
   `coordpy/lifecycle_audit.py`; the audit output names the
   capsule kind and CID that broke the invariant.

## Adding a backend

The supported backend layout is `coordpy.LLMBackend` (a Protocol
defined in `coordpy/llm_backend.py`). To add a new one:

1. Implement a class with the same call shape as
   `OllamaBackend` / `OpenAICompatibleBackend`.
2. Register it via `coordpy.make_backend(name, ...)` if you want
   it nameable from `COORDPY_BACKEND`.
3. Add tests under `tests/`. The smoke driver's section 16 is a
   useful template.

`coordpy.extensions` covers the same pattern for sandboxes
(`SandboxBackend`), report sinks (`ReportSink`), and task banks
(`TaskBankLoader`).

## Pull requests

- One logical change per PR; keep diffs focused.
- Update or add tests for any behaviour change.
- Run `./scripts/smoke.sh` and confirm it ends with
  `ALL SMOKE CHECKS PASSED`.
- Touch a schema → follow the bump recipe above.
- Touch a capsule lifecycle invariant → cite the C/T number in
  the PR description and add a regression test.

## Releasing

The version is single-source-of-truth in `coordpy/_version.py`.
`pyproject.toml` reads it dynamically.

Full runbook: [`RELEASING.md`](RELEASING.md). Quick path:

```bash
./scripts/release.sh check     # build, twine-check, smoke-test
./scripts/release.sh upload    # then push to PyPI
```

The recommended path is to push a `vX.Y.Z` tag and let
`.github/workflows/release.yml` upload via PyPI Trusted
Publisher.
