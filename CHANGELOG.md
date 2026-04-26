# Changelog

The Changelog now tracks **Wevra SDK** releases. The research
programme's phase-by-phase narrative lives in
`vision_mvp/RESULTS_PHASE*.md` and
`docs/context_zero_master_plan.md`.

## [SDK v3.4] — 2026-04-26 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + synthetic mode + cross-model parser-boundary research

*Strictly additive on SDK v3.3. Every v3.3 contract test (18) still
passes byte-for-byte; capsule view schema name unchanged
(`wevra.capsule_view.v1` — PROMPT / LLM_RESPONSE payloads are
additive). Full Wevra + capsule test suite green (199 tests).*

### Added
- **PROMPT capsule kind** (parent: SWEEP_SPEC; Theorem W3-42).
  Records prompt SHA-256 + byte length + bounded text snippet
  (≤ 4 KiB) + model_tag + prompt_style + coordinates.
  Idempotent on content (Capsule Contract C1) — byte-identical
  prompts collapse to one capsule.
- **LLM_RESPONSE capsule kind** (parent: PROMPT; Theorem W3-43).
  Records response SHA-256 + byte length + bounded snippet +
  elapsed milliseconds + coordinates. Admission rejects if
  prompt CID is not yet sealed (Capsule Contract C5).
- **`CapsuleNativeRunContext.seal_prompt`** /
  **`seal_llm_response`** runtime methods, plus
  **`seal_parse_outcome(llm_response_cid=...)`** optional
  argument. The end-to-end inner-loop chain is now five typed
  capsules: `PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
  PATCH_PROPOSAL → TEST_VERDICT`.
- **`capsule_from_prompt`**, **`capsule_from_llm_response`**
  adapters; `PROMPT_TEXT_CAP` / `LLM_RESPONSE_TEXT_CAP` constants.
- **Lifecycle audit invariants L-9 / L-10 / L-11** (Theorems
  W3-44 / W3-45):
  - L-9: PROMPT.parents == (SWEEP_SPEC,).
  - L-10: LLM_RESPONSE has exactly one parent, a sealed PROMPT.
  - L-11: PARSE_OUTCOME / LLM_RESPONSE coordinate consistency
    (instance_id / parser_mode / apply_mode / n_distractors;
    strategy may differ).
- **Synthetic-LLM mode**: `SweepSpec(mode="synthetic",
  synthetic_model_tag=<tag>)`. Uses a deterministic in-process
  `SyntheticLLMClient` instead of an Ollama endpoint. Seven
  calibrated distributions ship in
  `vision_mvp.wevra.synthetic_llm.SYNTHETIC_MODEL_PROFILES`:
  `clean`, `unclosed`, `prose`, `empty`, `fenced`,
  `multi_block`, `mixed`. The full PROMPT / LLM_RESPONSE /
  PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT chain seals
  end-to-end without network access.
- **Cross-model parser-boundary experiment** (Conjecture W3-C6,
  empirical):
  `vision_mvp.experiments.parser_boundary_cross_model`. Sweeps
  `(model_tag, parser_mode)` across the synthetic distribution
  library; reports cross-distribution PARSE_OUTCOME failure-kind
  TVD up to 1.000 and parser-mode (strict→robust) shift up to
  1.000 on `synthetic.unclosed`. Reproducible from CLI:
  `python3 -m vision_mvp.experiments.parser_boundary_cross_model`.
- **16 new contract tests** in
  `vision_mvp/tests/test_wevra_capsule_native_inner_loop.py`
  covering W3-42 / W3-43 / W3-44 / W3-45 / W3-C6.

### Changed
- **`SDK_VERSION`** bumped to `wevra.sdk.v3.4`.
- **`CapsuleKind.ALL`** now includes `PROMPT` and `LLM_RESPONSE`.
- **`render_view.payload_kinds_always`** extended to include
  PROMPT and LLM_RESPONSE (so on-disk audits can navigate the
  full inner-loop chain from `capsule_view.json` alone).
- **`CapsuleLifecycleAudit.RULES`** extended from 8 rules to 11.
- **W3-13** (DAG height ≤ 4 on canonical run pattern) is updated
  to ≤ 5 on canonical SDK v3.4 runs (the inner-loop chain adds
  one structural layer). Documented in
  `docs/CAPSULE_FORMALISM.md` § 4.J.
- **Conjecture W3-C5 (legacy SDK v3.3)** is **DISCHARGED** by
  Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **Conjecture W3-C4 (legacy SDK v3.3)** is **superseded** by the
  sharper synthetic reading W3-C6.

### Documentation
- New milestone note: **`docs/RESULTS_WEVRA_INNER_LOOP.md`**.
- `docs/CAPSULE_FORMALISM.md` § 4.J added (W3-42 / W3-43 / W3-44 /
  W3-45 / W3-C6 + W3-C5-discharged).
- `docs/THEOREM_REGISTRY.md`, `docs/RESEARCH_STATUS.md`,
  `docs/HOW_NOT_TO_OVERSTATE.md` updated for SDK v3.4.
- `docs/START_HERE.md` adds "What changed in SDK v3.4" section.
- `docs/context_zero_master_plan.md` § 4.21 added.
- `papers/wevra_capsule_native_runtime.md` strengthened —
  capsule-native execution is now its real centre, with strict
  claim taxonomy covering PROMPT / LLM_RESPONSE chain and the
  W3-C6 empirical anchor.
- README headline + stability matrix updated.

## [0.5.1] — 2026-04-22 — Wevra identity & clarity pass

*Documentation / exemplar milestone. No SDK-contract change; all 1349
Slice-2 tests still pass.*

### Added
- **`docs/START_HERE.md`** — canonical one-pass orientation for new
  readers. Classifies every top-level surface (Wevra SDK, CLI,
  extension protocols, unified runtime, legacy product path, core
  substrate, research shards, boundary). Meant to be the answer to
  "what is this repo?" without duplicating the README or the master
  plan.
- **`examples/out_of_tree_plugin/wevra-markdown-sink/`** — first
  in-repo exemplar of a standalone pip-installable Wevra plugin
  package. Declares `[project.entry-points."wevra.report_sinks"]`,
  registers a Markdown `ReportSink` via
  `importlib.metadata.entry_points`, and requires zero edit under
  `vision_mvp/`. Closes master-plan § 10.5 ledger item 2 at the
  machinery-plus-artifact level (only the "published by a third
  party" condition remains future).
- **`vision_mvp/RESULTS_WEVRA_IDENTITY.md`** — theory-forward results
  note with theorem-style claims (W-IDN-1 identity projection,
  W-IDN-2 orientation sufficiency, W-IDN-3 extension-surface
  reality) and three conjectures (W-IDN-C1 cold-agent
  classification, W-IDN-C2 stable-identity robustness, W-IDN-C3
  distinctiveness via composition rather than primitive novelty).

### Changed
- **README headline** now leads with **Wevra** (the shipped product)
  and positions CASR as original-substrate research; the scaling
  claims are preserved and re-anchored to Theorem 3 in `PROOFS.md`.
- **ARCHITECTURE.md headline** re-anchored to Wevra + Context Zero;
  a framing callout was added before the Phase 26–44 block so
  readers know that block is a historical incremental record and
  the durable architecture is the layered substrate diagram + § 3
  of the master plan.
- **`vision_mvp/__init__.py`** top-level docstring: Wevra is the
  shipped product; `CASRRouter` is explicitly research-grade code
  used by the SDK under the hood.
- **`vision_mvp/api.py`** `CASRRouter` docstring no longer says
  "Phase-3 hierarchical protocol" or "CASR-theoretic optimum" in
  places where a user would read them as current product contract;
  the O(log N) bound is now anchored to Theorem 3.
- **`vision_mvp/product/__init__.py`** retitled from "Phase-45
  product-grade orchestration surface" to "Legacy product modules
  (pre-Wevra import path)" — same code, correct framing.
- **`pyproject.toml`** — clearer comment on the `casr` legacy
  script; public CLI stays `wevra` / `wevra-import` / `wevra-ci`.
- **Master plan § 10** — short "Programme vs Product" callout near
  the top; § 10.1 stability matrix row for out-of-tree plugins
  updated from "boundary / next-slice" to "exemplar landed";
  § 10.3 B.6 note and § 10.5 ledger item 2 updated.

### Not changed (deliberately)
- The Wevra SDK contract (every Slice 2 public symbol remains).
- Any test; suite is green at 1349/1349.
- Docker-first-by-default flip for untrusted JSONLs (still Slice 3).
- GitHub Actions release-on-real-tag firing (workflow still declared,
  not yet exercised on a real tag).

## [0.5.0] — 2026-04-22 — Wevra SDK Slice 2

### Added
- **Extension system** (`vision_mvp/wevra/extensions/`). Three
  runtime-checkable Protocols — `SandboxBackend`, `TaskBankLoader`,
  `ReportSink` — each with an in-process registry and discovery via
  `importlib.metadata.entry_points` under groups
  `wevra.sandboxes`, `wevra.task_banks`, `wevra.report_sinks`.
  One worked example (`JsonlWithMetaSink`) and a contract test
  suite that exercises the full register→resolve→emit path.
- **Unified mock/real runtime** (`vision_mvp/wevra/runtime.py`).
  New `SweepSpec` dataclass; single `run_sweep(spec)` entry point
  dispatches mock and real runs through the same substrate
  primitives. Real runs execute in-process when
  `RunSpec.acknowledge_heavy=True`; otherwise the SDK refuses to
  start the heavy run and emits the resolved launch command.
- **`RunSpec.acknowledge_heavy`** and **`RunSpec.report_sinks`** —
  first-class cost gate and plugin hook on the top-level SDK spec.
- **`HeavyRunNotAcknowledged`** exception — strict cost-gate signal.
- **Env-driven endpoints**: `WEVRA_OLLAMA_URL_MAC1`,
  `WEVRA_OLLAMA_URL_MAC2`, `WEVRA_OLLAMA_URL` override profile-
  declared URLs at runtime. No hard-coded cluster IP is baked into
  code paths that a third-party consumer has to edit.
- **`--acknowledge-heavy` / `--report-sink`** flags on `wevra`.
- **Report schema bump**: `phase45.product_report.v2`. v1 remains
  accepted by `wevra-ci`; both listed in `EXPECTED_REPORT_SCHEMAS`.
- **GitHub Actions workflow** (`.github/workflows/wevra-ci.yml`):
  SDK contract tests on 3.10/3.11/3.12, console-script smoke,
  `python -m build` sdist+wheel, release on tag.
- **Cluster-backed validation artifact** under
  `vision_mvp/artifacts/wevra_slice2_g1/` — real ASPEN `mac1`
  `qwen2.5-coder:14b` run launched via `wevra.run(RunSpec(...,
  acknowledge_heavy=True))`, with provenance manifest and
  `wevra-ci` verdict.
- **Theory note**: `vision_mvp/RESULTS_WEVRA_SLICE2.md` —
  theorem-style claims W2-1 … W2-4.

### Changed
- `SDK_VERSION` bumped to `wevra.sdk.v2`. The bump is additive;
  every Slice 1 public symbol remains available.
- `CI gate` accepts v1 and v2 report schemas.
- `product/runner.py` now routes all sweeps through
  `wevra.runtime.run_sweep` instead of the legacy
  `_real_sweep_stub`.

### Deprecated
- `_real_sweep_stub` / `_mock_sweep` in `vision_mvp/product/runner.py`
  are private and will be removed in a future release; external code
  should use `wevra.run_sweep(SweepSpec(...))`.

### Next-slice (deferred, still honest)
- Docker-first sandbox as the default for public/untrusted JSONLs
  (backend exists; default-flip is Slice 3).
- Public SWE-bench-Lite JSONL on local disk (🧱 external).
- Resident ≥70B coder-finetuned model (🧱 external).

## [0.4.0] — 2026-04-21 — Wevra SDK Slice 1

See `docs/context_zero_master_plan.md` § 10.2.

- Introduced `vision_mvp/wevra/` stable SDK boundary.
- `RunSpec` / `run`, `WevraConfig`, `build_manifest`, schema
  constants, profile/report/ci_gate/import_data re-exports.
- Provenance manifest (`wevra.provenance.v1`) on every run.
- Console scripts: `wevra`, `wevra-import`, `wevra-ci`.
- Package renamed to `wevra` on PyPI; `SDK_VERSION = wevra.sdk.v1`.
- `sys.path.insert` hacks removed from product modules.
- Contract tests: `test_wevra_public_api.py`, `test_wevra_provenance.py`.

---

## [0.1.0] — 2026-04-16

Initial alpha release. One continuous research session.

### Added — Core library (`vision_mvp/`)

- **`CASRRouter`** — black-box public API. `step(observations) -> estimates`.
- Core primitives: `Bus`, `Agent`, `Manifold` (given basis),
  `StreamingPCA` (learned basis), `Stigmergy` (CRDT register),
  `Workspace` (top-k admission), `NeuralPredictor` and `PredictorBank`
  (vectorized across agents).
- Phase-6 additions: `MarketWorkspace` (VCG pricing),
  `SharedRNG`/`DeltaChannel` (pre-shared randomness), `AdaptiveScale` and
  `ContinuousScaleProjector` (continuous-scale projection).
- Six coordination protocols: `naive`, `gossip`, `manifold_only`,
  `full_stack`, `adaptive`, `hierarchical`, `holographic`, `swarm`, and
  `llm_protocols` (real LLM agents via Ollama).
- Two coordination tasks: `consensus` (static) and
  `drifting_consensus` (non-stationary with optional shock).

### Added — Experiments & results

- Phase 1 through Phase 5 runnable experiment harnesses under
  `vision_mvp/experiments/`.
- Measured scaling law: peak per-agent context = ⌈log₂ N⌉ exactly at
  every N ∈ {10, 50, 200, 1 000, 5 000, 10 000, 20 000, 50 000, 100 000}.
- Real LLM demonstration at N = 10 (local qwen2.5:0.5b via Ollama) showing
  34 % token savings with 100 % accuracy.

### Added — Theory

- **`PROOFS.md`** — twelve formal theorems, each with a proof and a
  machine-checkable empirical counterpart in `tests/`.
- **`EXTENDED_MATH_[1–7].md`** — 72-framework survey converging on the
  O(log N) bound from Information Bottleneck through Geometric Langlands.
- **`VISION_MILLIONS.md`** — the 10-idea paradigm shift for million-agent
  systems. 6 of 10 ideas implemented.

### Added — Tests

- **94 tests**, all passing (0.45 s total wall time):
  - 55 core-module unit tests.
  - 15 protocol integration & regression tests (including the scaling-law
    assertion `test_full_stack_peak_context_is_log_n`).
  - 13 Phase-6 tests (market, shared randomness, continuous scale).
  - 11 public-API (`CASRRouter`) tests.

### Added — Developer UX

- `pyproject.toml` (installable as `context-zero`).
- `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`, top-level `README.md`.
- `casr` CLI entry-point (`python -m vision_mvp demo|scale|phase|test|info`).
- Four runnable `examples/`:
  1. basic consensus
  2. drift tracking
  3. scaling demo
  4. local LLM coordination

### Not yet

- Real LLM tests at N > 10 (need bigger compute budget).
- Async variants (current protocol is synchronous).
- A formal peer-review cycle for the math.
- PyPI upload.

All the mathematics says O(log N). The code and the test suite confirm it.
The next step is to run it in anger on harder tasks and let skeptical
reviewers tear it apart.
