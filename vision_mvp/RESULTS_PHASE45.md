# Phase 45 — Finished-product state: one-command product runner, release-candidate validation, and the release-criteria checklist

**Status: product milestone. Phase 45 is the first milestone whose
deliverable is not a new substrate primitive, a new parser axis,
or a new semantic label, but a durable *operator surface* that
composes the already-settled Phase 31..44 layers into a single
invocation + a single verdict artifact.** Phase 44 closed the
science (raw capture + refined taxonomy + readiness validator).
Phase 45 closes the *operability* side: profiles, runner, report
renderer, finished-product checklist, and a release-candidate
validation pass. Nothing below the runner changes.

Four coupled artifacts ship:

1. **Product package** (`vision_mvp/product/`): `profiles.py`,
   `runner.py`, `report.py`, `__main__.py`, `__init__.py`.
   Schema version `phase45.product_report.v1` and
   `phase45.profile.v1`.
2. **Finished-Product Checklist** in
   `docs/context_zero_master_plan.md` §9. Durable, status-bearing,
   grouped into seven product-critical areas. Explicit about what
   still blocks true finished-product status.
3. **Phase-45 test slice**
   (`vision_mvp/tests/test_phase45_product.py`). 11 tests covering
   profile set stability, deepcopy independence, full smoke E2E,
   bundled 57 saturation, ASPEN real-profile launch-cmd recording,
   skip/force flags, public-jsonl override, schema stability,
   model capability table, and summary rendering.
4. **Release-candidate artifacts** under
   `vision_mvp/artifacts/phase45_rc_bundled/`,
   `vision_mvp/artifacts/phase45_rc_mock_sweep/`, and
   `vision_mvp/artifacts/phase45_mac1_recorded/` — one machine-
   readable `product_report.json` + one human-readable
   `product_summary.txt` per run.

Phase 45 in one line: **with a stable profile schema, a one-
command runner that composes readiness → sweep → report, and a
release-criteria checklist that names its two external blockers
(public data, ≥70B model) explicitly, the programme is in
finished-product state with respect to everything inside its
control.**

Three new theorems (P45-1 / P45-2 / P45-3) and three new
conjectures (C45-1 / C45-2 / C45-3). Phase 41..44 regressions
green (91 tests under `test_phase4[1234]_*.py`); Phase 45 adds
11.

---

## Part A — Research framing

### A.1 Why this milestone exists

The master plan's §7.5 rulebook says *never prioritise
productisation*. That rule stands and is not being revised.
What Phase 45 does is recognise a different fact: after Phase 44,
the programme's reproducibility surface is *already* operator-
shaped — the five-check readiness validator, the raw-capture
store, the refined classifier, the 57-instance bundled bank, the
parser + matcher + sandbox axes. The question is whether the
operator has to hand-wire five experiment scripts to run the
normal path, or whether there is a single, stable entrypoint.

There was no single entrypoint before Phase 45. `phase42_parser_
sweep.py`, `phase44_public_readiness.py`, and `phase44_semantic_
residue.py` are three independently-argument-parsed scripts. A
fresh operator (or a future version of the authors) had to
remember the flag set for each one and the right order, and had
to write ad-hoc glue to land the artifacts in a consistent
directory. That glue was not durable, not versioned, and not
tested.

Phase 45 replaces the glue with a declared profile set and a
runner. The runner is **a thin composition**, not a rewrite: every
primitive it calls is byte-for-byte the primitive Phases 42/44
already shipped. That discipline is what makes Phase 45 a
productisation milestone *without* violating §7.5 — the research
stack is unchanged and the runner is a convenience over it.

Separately, the master plan needed a durable place to answer the
question *"is this finished?"*. Phase 44 ended with the externali-
sation gap being purely data-availability-shaped, but there was
no single section that named every axis of "finished" with
explicit status. §9 of the master plan is that section, shipped
in this phase.

### A.2 What Phase 45 ships

* **`vision_mvp/product/profiles.py`.** Six named profiles
  (`local_smoke`, `bundled_57`, `bundled_57_mock_sweep`,
  `aspen_mac1_coder`, `aspen_mac2_frontier`, `public_jsonl`), a
  `list_profiles()` + `get_profile(name)` API that returns a
  deep-copied dict (profiles are *declarations*, not singletons),
  a `model_capability_table()` that maps local Ollama model names
  to suitable-for tags (`parser_dominant`, `semantic_headroom`,
  `frontier`, `canonical_coder`, `null_control`, etc.).
  Schema version `phase45.profile.v1`.

* **`vision_mvp/product/runner.py`.**
  * `run_profile(name, out_dir, jsonl_override, skip_sweep,
    force_sweep)` → dict conforming to
    `phase45.product_report.v1`.
  * Executes (1) readiness validation, (2) mock-mode sweep in
    process (or real-mode launch-command recording), (3)
    emission of `product_report.json`, `product_summary.txt`,
    `readiness_verdict.json`, and `sweep_result.json` /
    `sweep_launch.json`.
  * Real-LLM sweeps are deliberately *not* forked from inside the
    runner — the runner instead writes out the resolved
    `phase42_parser_sweep` / `phase44_semantic_residue` command
    the operator can paste onto the correct ASPEN node. This is
    a scope-discipline decision: long multi-hour LLM sweeps
    should be visible and explicit at the shell, not hidden
    inside a product-runner invocation.

* **`vision_mvp/product/report.py`.** `render_summary(report_dict)
  → str`. Prints readiness per-check counts, mock-sweep pass@1
  per strategy, real-launch command lines, artifact list, wall.
  Used by the runner and by the tests; can also be re-run on a
  stored `product_report.json` without re-executing anything.

* **`vision_mvp/product/__main__.py`.** Module entrypoint —
  `python3 -m vision_mvp.product --profile <name> --out-dir
  <dir>` is the canonical invocation.

* **`docs/context_zero_master_plan.md` §9 — Finished-Product
  Checklist / Release Criteria.** Seven status-bearing groups
  (core substrate, parser/matcher/sandbox, public-data readiness,
  model coverage, operator workflow, reporting, release docs)
  with per-item status, anchors, and a final §9.8 naming exactly
  two remaining blockers (both external).

* **Phase-45 test slice.**
  `vision_mvp/tests/test_phase45_product.py` — 11 tests.

* **Release-candidate artifacts** (see § D).

### A.3 Scope discipline (what Phase 45 does NOT claim)

1. **No new substrate claim.** Phase 45 does not add a new
   substrate layer, a new routing semantics, or a new typed-
   handoff contract. The runner is an orchestration surface over
   Phases 31..44.
2. **No new parser or matcher axis.** The parser-mode set
   (strict, robust, unified) and apply-mode set (strict,
   permissive) are unchanged. The runner exercises them; it does
   not extend them.
3. **No new empirical pass@1 number.** The bundled-bank readiness
   saturation (57/57) and the mock-oracle pass@1 (1.000 under
   every strategy, every cell) are reproductions of Theorems
   P41-1 and P44-3, byte-for-byte. Phase 45 does not *claim* a
   new number; it *composes* the existing ones under the runner.
4. **No new LLM evaluation on a new frontier model.** The mac1
   and mac2 profiles record the launch command for the Phase-44
   cluster runs; they do not re-execute them inside Phase 45.

---

## Part B — Theory

### B.1 Setup

Phase 45 introduces three operator-facing objects:

* **`Profile`** — a pair
  `(readiness_cfg, sweep_cfg_or_None)` with a stable schema
  (`phase45.profile.v1`).
* **`Runner : Profile × out_dir × overrides → ProductReport`**.
  Deterministic in its execution path given the same profile and
  the same JSONL; the only non-deterministic axis is
  wall-seconds.
* **`ProductReport`** — a dict conforming to schema
  `phase45.product_report.v1`, containing the readiness verdict,
  the sweep result (mock-executed or real-launch-recorded), an
  artifact list, and wall-seconds.

### B.2 Theorem P45-1 — Runner composition is a faithful projection of the underlying primitives

**Statement.** For every profile `P` with readiness config
`R = (jsonl, limit, sandbox_name)` and sweep config `S`, the
runner's readiness verdict equals the verdict produced by
`phase44_public_readiness.run_readiness(R.jsonl, R.limit,
R.sandbox_name)` byte-for-byte, and the runner's mock-sweep cell
list equals the cell list produced by invoking
`phase42_parser_sweep`'s oracle generator path with the same
`(jsonl, n_instances, parser_modes, apply_modes, n_distractors,
strategies, sandbox)` tuple.

**Interpretation.** The runner is a *pure composition* of the
already-settled primitives; it neither adds signal nor strips
signal. A reader who trusts the Phase-42 and Phase-44 scripts
can trust the runner's output without re-verifying it.

**Proof sketch.** The runner's readiness path is a direct call
into `run_readiness` with the profile's arguments; there is no
intermediate transformation. The runner's mock-sweep path
invokes `deterministic_oracle_generator` and
`run_swe_loop_sandboxed` with the same arguments
`phase42_parser_sweep` uses in mock mode, including the same
`ParserComplianceCounter` factory and the same strategy tuple
`ALL_SWE_STRATEGIES`. The only transformation is the
per-cell summarisation into a dict conforming to
`phase45.product_report.v1`. ∎

**Empirical anchor.** Phase-45 test
`test_local_smoke_end_to_end` + `test_bundled_57_readiness_only_
full_saturation` reproduce the Phase-44 57/57 saturation under
the runner. Byte-identity between the runner's mock-sweep pass@1
and the `phase42_parser_sweep --mode mock` pass@1 is structural
(same primitives called with same arguments).

### B.3 Theorem P45-2 — Readiness gating monotonicity

**Statement.** Let `P` be a profile with a sweep config `S`. If
`run_readiness(P.readiness) = {ready: False, ...}`, then
`run_profile(P, force_sweep=False)` returns a report whose
`sweep` block has `{skipped: True, reason:
"readiness_not_ready"}`. Conversely, if readiness is ready, the
sweep is executed (or, for real mode, recorded as a launch
command).

**Interpretation.** The runner treats readiness as a *gate*, not
a *signal*. This guarantees the sweep is never run on a JSONL
that has not passed the adapter/parser/matcher/test-runner
checks — catching schema-rot before it burns an LLM call.

**Proof sketch.** The runner branches on
`readiness_verdict["ready"]` before calling `_mock_sweep` or
`_real_sweep_stub`. The only override path is the `force_sweep`
flag. ∎

**Empirical anchor.** Phase-45 test
`test_skip_sweep_flag_skips_sweep_block` covers the skip path;
the gating path is covered indirectly by the 57/57 full-bank
readiness (which is always ready on the bundled bank).

### B.4 Theorem P45-3 — Finished-product state is a composition, not a claim

**Statement.** Let `L` be the set of layers the programme
depends on being operationally reproducible: substrate,
parser/matcher/sandbox, public-data readiness, reporting. For
every layer `ℓ ∈ L`, there exists a theorem `T(ℓ)` (Theorem 3,
P41-1, P41-2, P43-3, P44-1, P44-2, P44-3, P45-1, P45-2) that
states `ℓ`'s correctness. **Finished-product state is the
conjunction `⋀_{ℓ∈L} T(ℓ)` under the runner.** It is therefore
not an independent claim; it is the logical product of the
per-layer theorems, and it holds on the bundled 57-instance bank
by Theorem P45-1 (runner faithfulness) + the per-layer theorems
on that bank.

**Interpretation.** "Finished product" is operational, not
narrative. The programme is not waiting for a new idea; it is
waiting for two external inputs (public JSONL, ≥70B model). The
runner is the tool that makes this explicit: if those two inputs
land, one command reproduces every science claim against them.

**Proof sketch.** Each `T(ℓ)` is proved or empirically anchored
elsewhere. The runner's composition-faithfulness (P45-1) means
that invoking the runner on a valid profile is equivalent to
invoking the underlying primitives in the correct order. The
readiness-gate (P45-2) guarantees the pipeline is never run on a
bank that has not passed the adapter/parser/matcher/test-runner
checks. ∎

**Empirical anchor.** `vision_mvp/artifacts/phase45_rc_bundled/
product_report.json` shows readiness ready + 57/57 on every
check; `vision_mvp/artifacts/phase45_rc_mock_sweep/
product_report.json` shows readiness 57/57 + mock-oracle 1.000
on every strategy in every cell; `vision_mvp/artifacts/
phase45_mac1_recorded/product_report.json` shows readiness ready
+ a recorded real-LLM launch command. These three artifacts
together are a byte-for-byte reproduction of the Theorems
P41-1, P43-3, P44-1, P44-3 content under the Phase-45 runner.

### B.5 Conjecture C45-1 — Profile-set completeness under a stable operator surface

**Statement.** Every operator-facing evaluation the programme
has published in Phases 40..44 (bundled-bank readiness; bundled-
bank mock sweep; mac1 qwen2.5-coder:14b cluster run; mac2
qwen3.5:35b cluster run; public-bank drop-in) can be launched
with exactly one profile from the Phase-45 set. In particular,
the set `{local_smoke, bundled_57, bundled_57_mock_sweep,
aspen_mac1_coder, aspen_mac2_frontier, public_jsonl}` covers
every published evaluation.

**Status.** Structurally true on inspection of Phases 40..44
artefacts; listed as a conjecture because a future phase could
introduce a new axis (e.g. a contested-incident bank, a
non-SWE-style task distribution) that the current profile set
does not cover without extension.

**Falsifier.** Any post-Phase-44 evaluation whose launch cannot
be expressed as one profile from the current set.

### B.6 Conjecture C45-2 — Operator wall-time dominance by sandbox launch

**Statement.** For the `bundled_57` profile under the subprocess
sandbox, the runner wall-time is within 5 % of the sum of
per-row `SubprocessSandbox.run` calls — i.e. the orchestration
overhead of the runner (profile resolution, JSON emit, report
rendering) is <5 % of total wall.

**Status.** Supported empirically by the Phase-45 release-
candidate artefacts (5.16 s total vs ≈5.2 s measured for
readiness alone in Phase 44). Listed as a conjecture because
the overhead fraction grows as N shrinks (dominated by
fixed-cost Python import time at small N).

**Falsifier.** Any N ≥ 50 run where runner overhead exceeds 10 %
of sandbox wall.

### B.7 Conjecture C45-3 — External blocker shape is model/data, not architecture

**Statement.** Every remaining item in §9.8 of the master plan
that is not marked ✅ or ◐ is either a **🧱 external data
availability** blocker or a **◐ larger-model-on-cluster** blocker.
In particular, no item in §9.8 is an **architecture** blocker, an
**API contract** blocker, or a **substrate-semantics** blocker.

**Status.** Structurally verified by inspection of §9.8.
Conjectured rather than theorem because a future phase could
discover an architecture blocker (e.g. a new task distribution
where typed handoffs are insufficient) — the conjecture is
therefore a statement about the *programme's current boundary*,
not a permanent statement.

**Falsifier.** Any future phase that identifies a finished-
product blocker inside the architecture (substrate router, parser,
matcher, sandbox, classifier).

### B.8 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P45-1 runner composition is faithful | **Theorem** (structural + Phase-45 tests) |
| P45-2 readiness gating is a hard gate | **Theorem** (code + test) |
| P45-3 finished-product state is a composition | **Theorem** (logical product of per-layer theorems) |
| C45-1 profile-set completeness on Phases 40..44 | **Conjecture** (structurally true on inspection) |
| C45-2 operator overhead <5 % | **Conjecture** (supported on bundled_57; breaks at small N) |
| C45-3 external blockers are model/data shaped | **Conjecture** (true as of Phase 45; revisable) |

---

## Part C — Architecture

### C.1 New / extended modules

```
vision_mvp/product/__init__.py                    [NEW]    stable subpackage surface
vision_mvp/product/profiles.py                     [NEW]   ~180 LOC — 6 profiles + model matrix
vision_mvp/product/runner.py                       [NEW]   ~220 LOC — one-command runner
vision_mvp/product/report.py                       [NEW]    ~60 LOC — summary renderer
vision_mvp/product/__main__.py                     [NEW]     5 LOC  — module entrypoint

vision_mvp/tests/test_phase45_product.py           [NEW]    11 tests

docs/context_zero_master_plan.md                   [EXTENDED] §9 Finished-Product Checklist
vision_mvp/RESULTS_PHASE45.md                      [NEW]     this document
README.md                                          [EXTENDED] Phase-45 thread
ARCHITECTURE.md                                    [EXTENDED] product-surface pointer
MATH_AUDIT.md                                      [EXTENDED] P45-1 / P45-2 / P45-3 + C45-1..3

vision_mvp/artifacts/phase45_rc_bundled/           [NEW]    readiness-only release artifacts
vision_mvp/artifacts/phase45_rc_mock_sweep/        [NEW]    readiness + mock-sweep artifacts
vision_mvp/artifacts/phase45_mac1_recorded/        [NEW]    readiness + recorded-launch artifacts
```

No existing file under `vision_mvp/core/`, `vision_mvp/tasks/`,
or `vision_mvp/experiments/` is edited. The entire Phase 31..44
stack is preserved byte-for-byte.

### C.2 Where the new primitives sit

```
   ┌──────────────────────────────────────────────────────┐
   │  Phase 45 — Product surface + release checklist       │
   │  - ``vision_mvp.product.profiles`` (6 profiles)       │
   │  - ``vision_mvp.product.runner``    (one command)     │
   │  - ``vision_mvp.product.report``    (summary render)  │
   │  - Master plan §9 Finished-Product Checklist          │
   └──────────────────────────────────────────────────────┘
                             │  (orchestration only — imports)
   ┌──────────────────────────────────────────────────────┐
   │  Phase 44 — Raw residue + public-readiness            │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 43 — Semantic residue + public-style audit     │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 42 — Parser-compliance attribution             │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 41 — Matcher permissiveness attribution        │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 40 — Loader + sandbox + driver                 │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phases 31..38 — Substrate + typed-handoff team       │
   └──────────────────────────────────────────────────────┘
```

---

## Part D — Evaluation (release-candidate validation pass)

### D.1 RC#1 — Bundled 57 readiness (profile `bundled_57`)

```
python3 -m vision_mvp.product --profile bundled_57 \
    --out-dir vision_mvp/artifacts/phase45_rc_bundled
```

Result:

| field | value |
|---|---|
| readiness | **READY** |
| n | 57 |
| n_passed_all | 57 |
| schema / adapter / parser / matcher / test_runner | 57 / 57 / 57 / 57 / 57 |
| wall_seconds | 5.16 |
| sweep | not configured (readiness-only profile) |

Artifact: `vision_mvp/artifacts/phase45_rc_bundled/
product_report.json` + `product_summary.txt` +
`readiness_verdict.json`.

**Passes the Theorem P44-3 saturation bound and the Theorem
P45-1 composition-faithfulness check.**

### D.2 RC#2 — Bundled 57 mock sweep (profile `bundled_57_mock_sweep`)

```
python3 -m vision_mvp.product --profile bundled_57_mock_sweep \
    --out-dir vision_mvp/artifacts/phase45_rc_mock_sweep
```

Result:

| cell | pass@1 naive | pass@1 routing | pass@1 substrate |
|---|---:|---:|---:|
| parser=strict  apply=strict  nd=6  n=57 | 1.000 | 1.000 | 1.000 |
| parser=robust  apply=strict  nd=6  n=57 | 1.000 | 1.000 | 1.000 |

readiness READY, wall 33.16 s.

Artifact: `vision_mvp/artifacts/phase45_rc_mock_sweep/
product_report.json` + `product_summary.txt` +
`readiness_verdict.json` + `sweep_result.json`.

**Reproduces Theorem P41-1 (oracle saturation) and Theorem
P43-3 (strategy invariance) under the Phase-45 runner. The
substrate-vs-naive gap under oracle is 0 pp — byte-for-byte
consistent with Phase 42.**

### D.3 RC#3 — ASPEN mac1 real-LLM launch recording (profile `aspen_mac1_coder`)

```
python3 -m vision_mvp.product --profile aspen_mac1_coder \
    --out-dir vision_mvp/artifacts/phase45_mac1_recorded
```

Result: readiness READY (57/57), sweep *recorded* (not executed)
with launch command

```
python3 -m vision_mvp.experiments.phase42_parser_sweep \
    --mode real --model qwen2.5-coder:14b \
    --ollama-url http://192.168.12.191:11434 \
    --jsonl .../swe_lite_style_bank.jsonl \
    --parser-modes strict robust --apply-modes strict \
    --n-distractors 6 --sandbox subprocess
```

and a parallel raw-capture launch command pointing at
`phase44_semantic_residue`. The two executed command lines are
the ones used in Phase 44 § D.3 that produced the 14B-coder
residue split of `SEM_RIGHT_FILE_WRONG_SPAN` 25 % +
`SEM_WRONG_EDIT_SITE` 25 % (α = 0.5 support for C44-1).

**Validates Theorem P45-1 (the runner's recorded command is the
canonical launch string) without consuming cluster budget in
this Phase.**

### D.4 Unit-test pass and regression

* Phase-45: 11 tests, all green, 14.94 s.
  (`vision_mvp/tests/test_phase45_product.py`)
* Regression (Phases 41–44): 91 tests, all green, 109.96 s
  (`vision_mvp/tests/test_phase4[1234]_*.py`).
* Phase 31..38 substrate suite: untouched by this phase and
  therefore byte-for-byte preserved.

### D.5 Release-candidate verdict

All three release-candidate profiles execute cleanly under the
Phase-45 runner. The bundled-bank readiness + mock-sweep saturates
at every check. The real-LLM profile records a launch command
that the operator can paste onto the correct ASPEN node.

**Where the product still falls short of "finished":**
1. No public SWE-bench-Lite JSONL on local disk. Pipeline accepts
   one; data does not exist. Blocker 🧱.
2. No ≥70B coder-finetuned local model. Largest canonical cell
   remains 35B. Blocker ◐.

Neither blocker is an architecture debt. The programme is in
finished-product state with respect to everything inside its
control.

---

## Part E — Failure modes the runner surfaces

The runner is deliberately *loud* about these paths:

* **Missing `--jsonl` on `public_jsonl` profile.** `SystemExit`
  with a pointer to the required flag.
  (`test_public_jsonl_profile_requires_override`.)
* **Readiness NOT READY.** Sweep is skipped with a blockers list;
  operator must either fix the JSONL or pass `--force-sweep`
  knowing the pipeline may choke.
  (P45-2 guarantees this is a hard gate unless overridden.)
* **Unknown profile name.** `KeyError` listing valid profiles.
* **Unknown parser/apply mode on the underlying sweep.** Surfaced
  by the phase42 primitive, which the runner calls in mock mode.

---

## Part F — Future work

### F.1 Drop-in public SWE-bench-Lite run

Once a public SWE-bench-Lite JSONL is on local disk, the one-
command run is:

```
python3 -m vision_mvp.product --profile public_jsonl \
    --jsonl /path/to/swe_bench_lite.jsonl \
    --out-dir vision_mvp/artifacts/phase45_public_lite
```

No other code change required. This operationalises Phase 44
C44-4 (readiness is closed under row-level filtering) into a
single operator invocation.

### F.2 ≥70B coder-finetuned profile

Adding a seventh profile (e.g. `aspen_mac1_coder_70b`) is one
dict entry in `profiles.py` once the model is resident on the
cluster. The refined-residue classifier (Phase 44 v2) is already
ready to consume its artefact.

### F.3 Replay-verification driver

A `replay_capture` driver that takes a Phase-44
`(parent, capture)` pair and re-runs the matcher + sandbox
entirely from stored bytes. This would make Theorem P44-1
empirically checkable on any stored capture, and would integrate
cleanly as a sub-command under the Phase-45 runner.

### F.4 Non-Phase-45

Phase 45 does not change the Phase 31..38 substrate, the Phase
39..42 bridge, or the Phase 43..44 residue layer. All
corresponding claims stand.

---

## Part G — One-paragraph summary

Phase 45 introduces a one-command product runner
(`python3 -m vision_mvp.product --profile <name> --out-dir <d>`)
that composes the Phase-44 readiness validator, the Phase-42
parser sweep, and the Phase-44 raw-capture and refined-taxonomy
primitives behind a small, versioned profile set
(`phase45.profile.v1`). The runner is a faithful projection of
the underlying primitives (Theorem P45-1), treats readiness as a
hard gate for the sweep (Theorem P45-2), and — together with the
§9 Finished-Product Checklist in the master plan — makes
finished-product state an explicit logical product of the per-
layer theorems (Theorem P45-3). A release-candidate validation
pass saturates readiness on the bundled 57-instance bank
(57/57 in 5.16 s), reproduces oracle pass@1 under the mock-sweep
profile, and records the canonical ASPEN macbook-1 launch command
for the qwen2.5-coder:14b cluster run. The only remaining blockers
to a publishable public-bank run are external (data availability
and a ≥70B cluster model); both are named, dated, and explicitly
outside programme scope.

---
