# CoordPy Slice 2 — Extension system, unified runtime, real cluster validation

*2026-04-22*

Slice 1 (2026-04-21) shipped the CoordPy SDK boundary, provenance
manifest, and console scripts. Slice 2 makes CoordPy a genuinely
extensible, uniformly-executed, cluster-verified product: a real
plugin system, a unified mock/real runtime with a first-class cost
gate, env-driven cluster endpoints, and a canonical ASPEN-backed
artifact produced through the CoordPy surface itself.

---

## 1. What shipped

### 1.1 Extension system (`vision_mvp/coordpy/extensions/`)

Three runtime-checkable Protocols, each with an in-process registry
and `importlib.metadata.entry_points` discovery:

| Extension point       | Protocol          | Entry-point group     | Built-ins                         |
| --------------------- | ----------------- | --------------------- | --------------------------------- |
| Execution boundary    | `SandboxBackend`  | `coordpy.sandboxes`     | `in_process`, `subprocess`, `docker` |
| Evaluation input      | `TaskBankLoader`  | `coordpy.task_banks`    | `jsonl`                           |
| Post-run emission     | `ReportSink`      | `coordpy.report_sinks`  | `stdout`, `jsonfile`              |

Third-party packages extend CoordPy in two equivalent ways:

  * In-process: `coordpy.extensions.register_sandbox("my_box", MyBackend)`.
  * Declarative: an `entry_points` block in the plugin package's
    `pyproject.toml`; CoordPy discovers it via
    `coordpy.extensions.discover_entry_points()`.

A worked `ReportSink` example (`JsonlWithMetaSink`) ships under
`vision_mvp/coordpy/extensions/examples/` and is exercised end-to-end
by `test_coordpy_extensions.py::ReportSinkExtensionTests::test_worked_example_sink_end_to_end`
— the test registers the factory, drives it through
`RunSpec.report_sinks=("jsonl_with_meta",)`, and verifies both the
primary artifact and the `.meta.json` sidecar land on disk.

### 1.2 Unified runtime (`vision_mvp/coordpy/runtime.py`)

New frozen dataclass `SweepSpec` and single entry point
`run_sweep(spec)`:

  * **Mock mode**: deterministic oracle, no network, in-process.
    Same semantics as Slice 1's `_mock_sweep`.
  * **Real mode, acknowledged**
    (`RunSpec.acknowledge_heavy=True`): LLM-backed sweep runs
    *in-process* against the configured Ollama endpoint; the report
    records `executed_in_process=True`.
  * **Real mode, not acknowledged** (default): CoordPy refuses to
    start the heavy run and emits the resolved launch command as a
    staging artifact. `strict_cost_gate=True` makes this raise
    `HeavyRunNotAcknowledged` instead.

This replaces the Slice 1 split where mock runs executed in-process
but real runs always staged a launch command. The product `sweep`
block is now schema-unified under `coordpy.sweep.v2`, and the enclosing
`product_report` schema is bumped to `phase45.product_report.v2`.
The CI gate accepts both v1 and v2.

### 1.3 Production hardening

  * **Env-driven endpoints**: `COORDPY_OLLAMA_URL_MAC1`,
    `COORDPY_OLLAMA_URL_MAC2`, `COORDPY_OLLAMA_URL` override the
    profile-declared URL at runtime. Resolved inside
    `coordpy.runtime._resolve_endpoint`; tested by
    `test_coordpy_runtime.EnvEndpointOverrideTests`. No hard-coded
    cluster IP sits on a path that a consumer has to patch.
  * **Release workflow**: `.github/workflows/coordpy-ci.yml`
    (SDK contract tests on 3.10/3.11/3.12, console-script smoke,
    `python -m build` sdist+wheel, release on tag).
  * **CHANGELOG discipline**: `CHANGELOG.md` now tracks CoordPy SDK
    releases (v0.4.0 Slice 1, v0.5.0 Slice 2) rather than phase
    numbers.
  * **Operator-grade failure messages**: the unified runtime wraps
    sweep errors into `sweep_result.error_kind` /
    `sweep_result.error_detail` and the report renderer prints them
    as `sweep : ERROR <kind>: <detail>` instead of propagating a
    stack trace out of `coordpy.run`.
  * **Provenance remains first-class**: unchanged from Slice 1 — every
    run (mock, real-staged, real-executed) emits a
    `coordpy.provenance.v1` manifest.

### 1.4 Cluster-backed CoordPy validation

`vision_mvp/artifacts/coordpy_slice2_g1/` holds a real ASPEN
macbook-1 (192.168.12.191, `qwen2.5-coder:14b`) run launched via

```python
from vision_mvp.coordpy import RunSpec, run
run(RunSpec(profile="aspen_mac1_coder",
            out_dir="vision_mvp/artifacts/coordpy_slice2_g1",
            acknowledge_heavy=True,
            report_sinks=("stdout",)))
```

(with `n_instances=4` override for bounded wall time). Outcome on
4 instances × 2 parser modes × 3 strategies:

| parser | apply  | nd | n | naive | routing | substrate |
| ------ | ------ | -- | - | ----- | ------- | --------- |
| strict | strict |  6 | 4 | 0.000 | 0.000   | 0.000     |
| robust | strict |  6 | 4 | 1.000 | 1.000   | 1.000     |

This reproduces the Phase-42 parser-recovery finding (strict parser
fails under coder-model output drift; robust parser recovers
100 % on the bundled bank) via the CoordPy surface, not the
experiment script. `sweep.executed_in_process=True`, and the
run emits the full artifact set (`product_report.json`,
`product_summary.txt`, `provenance.json`, `readiness_verdict.json`,
`sweep_result.json`). `coordpy-ci` was then run on the report and
produced `ci_verdict.json` with `ok=true, blockers=0`. That verdict
is the canonical cluster-backed release artifact. Wall: 114 s.

---

## 2. Theorem-style claims and conjectures

Following the Slice 1 convention: bold the claim, tag it as
**empirical**, **proved** (constructive proof in-code or
referenced), or **conjectural** (falsifiable by a specific,
named condition).

### W2-1. Stable extension surface preserves bounded-context guarantees

**Claim (proved, constructive).** An extension that replaces a
`SandboxBackend`, a `TaskBankLoader`, or a `ReportSink` cannot
violate the bounded-context guarantees held by the substrate that
`coordpy.runtime.run_sweep` invokes.

**Proof sketch.** The extension protocols are projections of
already-settled module boundaries:

  * `SandboxBackend` is a renaming of `vision_mvp.tasks.swe_sandbox.Sandbox`,
    which is contract-tested (`test_code_interproc`, `test_code_semantics`,
    the Phase-40 sandbox-equivalence tests) to preserve `WorkspaceResult`
    shape across backends.
  * `TaskBankLoader` produces a `TaskBankBundle` whose `tasks` are
    opaque but whose iteration contract matches what
    `run_swe_loop_sandboxed` already consumes.
  * `ReportSink` receives the finished `product_report.json` — it
    has no read-back channel to the substrate, so it cannot perturb
    the bounded-context invariant (Theorem 3, `PROOFS.md`) of any
    run.

Therefore, for any external extension `E` and any `SweepSpec`
`s`, the pooled-summary block of `run_sweep(s)` with or without
`E` is the same modulo `E`'s documented side-effects
(different execution-boundary attribution, different input JSONL,
different post-run emission). Falsifiable by exhibiting an
extension that, by construction, violates the
`run_swe_loop_sandboxed` I/O contract — which the Protocol's
runtime `isinstance` check rules out at `get_*` time.

### W2-2. Unified-runtime identity under mock oracle

**Claim (empirical).** For every profile `p` with a mock sweep
block, the sweep cells produced by `coordpy.runtime.run_sweep` are
pooled-summary-equal to the cells produced by Slice 1's
`_mock_sweep(p)`. Evidence:
`test_phase45_product.test_local_smoke_end_to_end`
(Slice 1) asserts `pass_at_1 == 1.0` for all strategies on the
bundled bank; that same assertion holds unchanged under Slice 2's
unified runtime (1349/1349 tests pass).

Falsifiable by any mock profile whose pooled summary differs
between Slice 1 and Slice 2.

### W2-3. External-run safety as a product contract

**Claim (proved, constructive).** Every call to `coordpy.run(spec)`
produces a provenance manifest that is *sufficient* to reproduce
the run on a different machine, modulo external state the manifest
declares.

**Proof sketch.** The manifest (schema `coordpy.provenance.v1`)
records: git SHA + dirty flag, package version, Python version,
platform, profile name + schema, model tag, Ollama endpoint,
sandbox backend, input JSONL path + SHA-256 + bytes, argv, cwd,
user, hostname, sorted artifact list, timestamp. Given (a) the
same git SHA, (b) the same JSONL SHA-256, (c) the declared model
tag resident on the declared endpoint, and (d) the declared
sandbox backend available, `coordpy.run` is deterministic up to
LLM nondeterminism, which is bounded by `temperature=0.0`
(enforced in `coordpy.runtime._real_cells`) and the raw-response
cache (same prompt → same text within a run). Falsifiable by any
provenance-identical pair of runs whose pooled summaries differ
without the manifest surfacing the drift source.

### W2-4. Conditions under which CoordPy is a true drop-in SDK

**Conjecture (testable).** CoordPy is a true drop-in SDK for an
external operator iff the following hold:

  1. **E1** — the operator can install via `pip install coordpy` and
     the console scripts `coordpy`, `coordpy-import`, `coordpy-ci` work
     without a repo checkout. (Slice 1 + Slice 2: *satisfied*.)
  2. **E2** — every run emits a reproducibility-sufficient
     provenance manifest. (Slice 1: *satisfied*.)
  3. **E3** — the operator can execute a real-LLM sweep against a
     configured endpoint via `coordpy.run(RunSpec(...,
     acknowledge_heavy=True))` without editing in-tree code, and
     the report records `executed_in_process=True`. (Slice 2:
     *satisfied*; canonical artifact
     `vision_mvp/artifacts/coordpy_slice2_g1/`.)
  4. **E4** — the operator can add a new sandbox / task bank /
     report sink by installing a third-party plugin, without
     patching `vision_mvp/`. (Slice 2: *machinery satisfied; first
     real out-of-tree plugin still future work; tested end-to-end
     with in-process factory registration*.)
  5. **E5** — the SDK has a release automation path producing
     sdist + wheel artifacts on tag. (Slice 2: *satisfied via
     `.github/workflows/coordpy-ci.yml`; not yet exercised on a
     real tag in this environment*.)
  6. **E6** — Docker-first sandbox is the default when the caller
     requests untrusted-JSONL safety. (Slice 2: *backend exists,
     default-flip is Slice 3*.)

E1, E2, E3 are closed. E4 is closed as machinery; closure as
community practice is a first-real-out-of-tree-plugin test.
E5's GitHub-Actions workflow is checked in but the on-tag branch
has not been fired from this environment. E6 is the one remaining
product-path gap. The conjecture says: once all six hold, CoordPy
is a drop-in SDK; today, **E1–E3 + E4(machinery) + E5(declared)**
hold. Falsifiable by an external operator blocked on any of the
six conditions.

---

## 3. What still blocks 10/10 drop-in production

| Gap | Classification | Concrete resolution |
|---|---|---|
| Docker-first sandbox as default for public JSONLs | Slice 3 engineering | flip the default in `SweepSpec.sandbox` when the caller declares `trust_input=False`, and surface that through `RunSpec` |
| First real out-of-tree plugin | Slice 3 community | ship one external plugin repo (e.g. a Slack `ReportSink`) and link it from README |
| GitHub Actions release fired on a real tag | Slice 3 ops | one real tagged release |
| Public SWE-bench-Lite JSONL on local disk | 🧱 external (unchanged from Slice 1) | fetch the public JSONL; no code change |
| Resident ≥70B coder-finetuned model on cluster | 🧱 external (unchanged) | `aspen_mac1_coder_70b` profile exists; pull the model |

Slice 1 bought the SDK boundary. Slice 2 bought the plugin
surface, the unified runtime, the cluster-executed artifact, and
the release automation. Slice 3 is one flip, one example, one
tagged release.

---

## 4. Tests and validation runs exercised

Full suite: `python3 -m unittest discover -s vision_mvp/tests` →
**1349 tests, OK, 19.9 s** (Slice 1 was 1327).

New Slice 2 test modules (all passing):

  * `vision_mvp/tests/test_coordpy_extensions.py` — 10 tests
    (protocol conformance, registry roundtrip, entry-point
    discovery shape, worked-example sink end-to-end).
  * `vision_mvp/tests/test_coordpy_runtime.py` — 10 tests
    (`SweepSpec` validation, mock cell shape, real staging,
    `HeavyRunNotAcknowledged`, v2 report schema, env-var
    endpoint overrides).

Updated (still passing):

  * `test_coordpy_public_api.py` — `SDK_VERSION` bumped to
    `coordpy.sdk.v2`, `REQUIRED_SYMBOLS` extended, `RunSpec` field
    set extended.
  * `test_coordpy_provenance.py` — unchanged behavior on v2 runner.
  * `test_phase45_product.py` — unchanged behavior under the
    unified runtime (backwards-compatible at the data level).

Real cluster validation:

  * `vision_mvp/artifacts/coordpy_slice2_g1/product_report.json` —
    `profile=aspen_mac1_coder`, `sweep.executed_in_process=true`,
    `sweep.schema=coordpy.sweep.v2`, `sweep.model=qwen2.5-coder:14b`,
    `sweep.endpoint=http://192.168.12.191:11434`, 4 instances.
  * `vision_mvp/artifacts/coordpy_slice2_g1/ci_verdict.json` —
    `ok=true, blockers=0, executed_in_process=true`.
