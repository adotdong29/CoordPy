# Phase 46 — External-exercise readiness: public-data import, CI gate, and the frontier-model slot

**Status: deployment-readiness milestone. Phase 45 closed "finished
product within programme control"; Phase 46 closes the operator
surface *at the boundary between programme control and the outside
world*.** Nothing inside the substrate, parser/matcher/sandbox, or
residue-classifier layers changes. What changes is that the two
remaining external blockers named in master plan §9.8 — *public
JSONL availability* and *≥70B model availability* — now meet a
code path that is turnkey on their arrival.

Three coupled artifacts ship:

1. **Public-data import CLI**
   (`vision_mvp/product/import_data.py`). `audit_jsonl(...)` →
   `phase46.import_audit.v1`. Schema classification (native
   SWE-bench-Lite / hermetic / ambiguous / unusable), row-level
   diagnostics, duplicate-`instance_id` detection, non-object /
   decode-error / empty-bank / utf-8-decode enumeration, and a
   delegated Theorem-P44-3 readiness check. Exit codes:
   0 clean / 1 blocker / 2 file-not-found.
2. **CI / deployment consumer**
   (`vision_mvp/product/ci_gate.py`). `evaluate_report(...)` +
   `aggregate(...)` over one or more `product_report.json`
   files. Five checks: **schema**, **profile compatibility**,
   **readiness threshold**, **sweep outcome**, **artifact
   presence**. Machine-readable `phase46.ci_verdict.v1` +
   `.aggregate`. Pass/fail via Unix exit code.
3. **Frontier-model slot**. New profile
   `aspen_mac1_coder_70b` + `model_availability()` declarative
   check + capability-table entries (`qwen2.5-coder:70b`,
   `deepseek-coder-v3:70b` with the `slot_pending_availability`
   tag). The runner attaches a `model_metadata` block to the
   recorded real-LLM launch command so downstream tooling can
   reason about whether the model is resident on the cluster.

Phase 46 in one line: **with a turnkey import CLI, a real CI
consumer, and a frontier-model profile slot, the two external
blockers named in master plan §9.8 now meet the code at a
single-command boundary — adding the public JSONL is one
`--jsonl`, adding the 70B model is one config change.**

Three new theorems (P46-1 / P46-2 / P46-3) and three new
conjectures (C46-1 / C46-2 / C46-3). Phase-45 (11) + Phase-46
(12) tests all green; Phase-41..44 regression (91 tests) green.

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 45 §9.8 named two external blockers and called them
*named, dated, outside programme scope*. That was honest but
incomplete: **named** does not mean **code-reachable**. An
operator who obtained a public SWE-bench-Lite JSONL tomorrow
would still have to compose three separate scripts (validate /
run / interpret) and wire a CI hook themselves. An operator
who pulled a 70B coder into Ollama would still have to
hand-edit the Phase-45 profile set. Those are small
frictions, but they live exactly on the boundary between
"our finished product" and "someone else's deployment," and
that boundary is where Phase 46 lives.

Three distinct programme concerns motivate the three pieces:

1. **Public-data frictionlessness.** A public JSONL is not
   guaranteed to be SWE-bench-Lite-shape. A real import path
   needs to classify shapes, catch decode errors, detect
   duplicates, and surface the specific blocker before the
   operator tries to run the pipeline on a malformed file.
   This is the *row-level readiness report* requested in §A.
2. **Deployment signal.** `product_report.json` is
   CI-gate-shaped (Phase 45 § 9.6) but the CI gate is a
   *consumer*, not a *format*. Until a concrete consumer
   exists, the shape is a promise, not a contract.
3. **Frontier-model slot.** Phase 45's `aspen_mac*_*` profiles
   point at models currently in the Ollama cache. The 70B slot
   has a different status — the model may not be resident —
   and the profile set must distinguish *intended capability*
   from *resident-on-cluster*. That distinction is what
   `model_availability()` encodes.

### A.2 What Phase 46 ships

* **`vision_mvp/product/import_data.py`.**
  * `audit_jsonl(jsonl_path, *, limit, run_readiness_check,
    sandbox_name) → phase46.import_audit.v1` dict.
  * Five row-level checks (JSON decode, object-shape,
    native/hermetic schema classification, duplicate id
    detection, readiness delegation).
  * Failure modes: `file_not_found` (exit 2),
    `utf8_decode_error`, `empty_bank`,
    `json_decode_errors:<N>`, `non_object_rows:<N>`,
    `unusable_rows:<N>`, `duplicate_instance_ids:<N>`, and
    delegated `readiness:<blocker>` for P44-3 failures.
  * CLI:
    `python3 -m vision_mvp.product.import_data --jsonl <p>
    --out <o>`.

* **`vision_mvp/product/ci_gate.py`.**
  * `evaluate_report(report_path, ...) → phase46.ci_verdict.v1`.
  * `aggregate(verdicts) → phase46.ci_verdict.v1.aggregate`.
  * Threshold knobs: `--min-ready-fraction` (default 1.0),
    `--min-pass-at-1` (default 1.0),
    `--allow-not-ready`, `--require-sweep-executed`.
  * Whitelist: `--require-profile` / `--allow-profile`.
  * Multi-report aggregation (`--report A B C`) with a single
    top-level pass/fail.

* **Frontier-model slot.**
  * New profile `aspen_mac1_coder_70b` with
    `requires_model_availability: True` in `profiles.py`.
  * `model_availability(model)` → dict with
    `availability: pending_availability | assumed_resident |
    n/a` + capability tags.
  * `_real_sweep_stub` attaches `model_metadata` to the
    recorded launch payload so a CI consumer can detect the
    `slot_pending_availability` tag and downgrade expectations
    cleanly.

* **Phase-46 test slice**
  (`vision_mvp/tests/test_phase46_deployment.py`). 12 tests
  covering import-data (clean / broken / missing / empty), CI
  gate (RC accept / profile whitelist / sweep-required /
  mock-sweep threshold / missing report), frontier slot
  (registration / availability check / report metadata).

* **Phase-46 artifacts.**
  * `vision_mvp/artifacts/phase46_public_audit/
    bundled_bank_audit.json` — audit on the bundled bank
    (57/57 rows `ambiguous` shape, readiness READY).
  * `vision_mvp/artifacts/phase46_public_audit/
    missing_file_audit.json` — demonstrator of the
    `file_not_found` failure path (exit 2).
  * `vision_mvp/artifacts/phase46_ci_gate/
    aggregate_verdict.json` — aggregate CI verdict over the
    three Phase-45 RC artifacts (all ok).
  * `vision_mvp/artifacts/phase46_frontier_slot/` —
    `readiness_verdict.json`, `sweep_launch.json`
    (`model_metadata.availability ==
    pending_availability`), `product_report.json`,
    `product_summary.txt`.

### A.3 Scope discipline (what Phase 46 does NOT claim)

1. **No new substrate, parser, matcher, sandbox, classifier.**
   The Phase 31..45 stack is preserved byte-for-byte.
2. **No real 70B run.** The slot is declared. If the model
   lands the run is one config change, but no 70B numbers are
   claimed in this milestone.
3. **No real public SWE-bench-Lite run.** A real public JSONL
   is not on local disk; the codepath is exercised on a
   constructed broken JSONL + the bundled bank. The
   externalisation gap is still data-availability-shaped (§C.3
   of master plan §9.8).
4. **No new pass@1 number.** Every pass@1 reported here is a
   byte-for-byte reproduction of a Phase 44 / 45 number.

---

## Part B — Theory

### B.1 Setup

Phase 46 introduces three objects at the boundary between
programme-internal state and external inputs:

* **`V_import : path × (limit, sandbox) → ImportAudit`**.
  The import CLI. Pure function of JSONL bytes + the P44-3
  validator (when the file passes schema).
* **`V_ci : product_report × thresholds → CIVerdict`**.
  The CI gate. Pure function of the `product_report.json`
  bytes + the operator-supplied thresholds.
* **`M_avail : model_tag → {pending_availability,
  assumed_resident, n/a}`**. The declarative model-availability
  map. Trivially pure; serves as the hand-off point between
  the profile set and a future availability-probing upgrade.

### B.2 Theorem P46-1 — Import-audit saturation on bundled bank

**Statement.** `V_import(bundled_bank, limit=None,
sandbox=subprocess)` returns a report with
`ok == True`, `n_rows == 57`,
`shape_counts == {"ambiguous": 57}`,
`duplicate_instance_ids == []`, and
`readiness.ready == True` with all five Phase-44 checks
saturating at 57/57.

**Interpretation.** The bundled 57-instance bank is
SWE-bench-Lite-compatible under *both* shape classifications
(native + hermetic). Any strict subset of the bundled bank is
also `ok` — a direct consequence of the P44-3 row-level
pureness (Phase 44 C44-4) extended to the schema audit.

**Proof sketch.** The audit is a fold over the JSONL rows with
two pure predicates (`_classify_row_shape` and
`run_readiness` per-row check). The bundled bank was
constructed from SWE-bench-Lite-shape rows by design
(`swe_lite_style_bank.jsonl`), so every row satisfies both
`_NATIVE_REQUIRED` and `_HERMETIC_REQUIRED` → shape is
`ambiguous`. P44-3 handles the readiness half. ∎

**Empirical anchor.** `vision_mvp/artifacts/
phase46_public_audit/bundled_bank_audit.json` +
`test_import_bundled_bank_is_ready`.

### B.3 Theorem P46-2 — CI gate composition is faithful

**Statement.** For every `product_report.json` conforming to
`phase45.product_report.v1`, the CI gate's five-check verdict
is a pure function of the report bytes + the operator
thresholds. No hidden state, no network, no file-system access
beyond the report path itself. The gate's decision is therefore
deterministic and reproducible.

**Interpretation.** The CI gate is *faithful* to the Phase-45
product surface (Theorem P45-1) in the same sense that the
runner is faithful to the Phase-44 primitives. A reader who
trusts the product report can trust the gate's decision; a
reader who discovers a gate false-positive has discovered a
*report* bug, not a *gate* bug.

**Proof sketch.** Each of the five check predicates reads only
from the loaded JSON dict + the threshold arguments. The
schema check compares one string; the profile check tests
membership in a static set; the readiness check computes
`n_passed_all / n` and compares to the threshold; the sweep
check iterates over `cells[*].pooled[*].pass_at_1`; the
artifact check is set-difference against a static required
set. Each predicate is deterministic; the conjunction is
deterministic. ∎

**Empirical anchor.** `vision_mvp/artifacts/phase46_ci_gate/
aggregate_verdict.json` (3/3 ok) +
`test_ci_gate_accepts_bundled_57_rc`,
`test_ci_gate_rejects_non_whitelisted_profile`,
`test_ci_gate_requires_sweep_executed_fails_on_readiness_only`,
`test_ci_gate_local_smoke_sweep_passes_threshold`.

### B.4 Theorem P46-3 — Frontier-model slot separates capability declaration from residency

**Statement.** For every model tag `t` in the Phase-46
capability table:

* `M_avail(t).availability == "pending_availability"` iff
  `"slot_pending_availability" ∈ tags(t)`.
* Otherwise `M_avail(t).availability == "assumed_resident"`.

The runner's `_real_sweep_stub` attaches `model_metadata =
M_avail(t)` to every recorded real-LLM launch command. A
downstream consumer (CI gate, deployment controller) can
therefore distinguish between *"the profile names a model
that is canonical on this cluster"* and *"the profile names a
model whose slot exists but whose residency is unverified"*
without executing the model.

**Interpretation.** Adding a frontier model to the programme
is a two-step operation: (a) declare the slot — one dict
entry in `profiles.py` and one capability-table entry, done
in Phase 46 for the 70B coder class; (b) mark it resident
once the Ollama `ollama pull` completes on the cluster — one
string change per tag. The second step does not require any
architecture change.

**Proof sketch.** `model_availability` is a tagged-lookup
pure function. Residency semantics are declarative: the
programme does not *probe* Ollama; it *declares* what it
believes is resident. A future upgrade can replace the static
table with a probe against `http://<host>:11434/api/tags`
without changing the runner or the CI gate interface. ∎

**Empirical anchor.** `test_model_availability_declarative`,
`test_frontier_slot_sweep_metadata_attached_in_report` +
`vision_mvp/artifacts/phase46_frontier_slot/sweep_launch.json`
(`model_metadata.availability == "pending_availability"`).

### B.5 Conjecture C46-1 — External blockers are boundary-shaped, not programme-shaped

**Statement.** Every item in master plan §9.8 that is not
marked ✅ or ◐ is reachable from programme-internal state via
a *boundary* operation (file import, model residency check,
CI hook) rather than a *programme-internal* operation
(architecture change, theorem reproof, API reshape). In
particular, the two remaining blockers (public JSONL,
≥70B model) are both boundary-shaped and therefore bounded in
implementation cost to **one config change + one exercise run**
each.

**Status.** Supported by the Phase-46 shipped code: the import
CLI consumes a path argument only; the frontier slot consumes
a model-tag string only. Listed as a conjecture because a
future phase could uncover a boundary operation that is
*not* a pure config change (e.g. a public JSONL with
row-schema drift the current audit does not detect).

**Falsifier.** Any external blocker whose resolution requires
an architecture change inside `vision_mvp/core/`,
`vision_mvp/tasks/`, or `vision_mvp/experiments/`.

### B.6 Conjecture C46-2 — Semantic ceilings scale with model capability while substrate gap is invariant

**Statement.** For any coder-class model `f` that lands in the
70B capability slot (tags including `frontier`, `canonical_coder`,
`semantic_headroom`), the Phase-44 v2-refined residue
composition is expected to shift (fewer
`SEM_RIGHT_FILE_WRONG_SPAN`, fewer `SEM_WRONG_EDIT_SITE`, more
`SEM_NARROW_FIX_TEST_OVERFIT` / `SEM_RIGHT_SPAN_WRONG_LOGIC`),
but the substrate-vs-naive pass@1 gap remains 0 pp on every
cell — a strict extension of Conjecture C44-3 into the
70B-class regime.

**Status.** Open. Resolvable the moment the model is resident
on the cluster by re-running `aspen_mac1_coder_70b` + the
Phase-44 `phase44_semantic_residue` analyse-only driver.
**Falsifier.** Any 70B cell where the v2-refined substrate
gap is non-zero, or where the refined residue composition
matches the 35B composition byte-for-byte (no ceiling lift).

### B.7 Conjecture C46-3 — CI consumption is closed under profile extension

**Statement.** For any new profile added to `profiles.py`
conforming to `phase45.profile.v1`, the CI gate's five-check
verdict remains meaningful: the schema check, profile-
whitelist check, readiness threshold, and artifact-presence
check are profile-agnostic; the sweep check is profile-aware
but degrades cleanly (absent sweep → "no sweep configured"
which passes unless `--require-sweep-executed`).

**Status.** Structurally true for the current check set. The
conjecture is listed as a conjecture because a profile with a
radically new sweep *shape* (e.g. a multi-model ensemble cell
or a non-pass@1 metric) could require a new gate check.
**Falsifier.** Any future profile whose sweep block cannot be
described by `{mode, cells[].pooled[].pass_at_1}` under the
current gate schema.

### B.8 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P46-1 import-audit saturation on bundled bank | **Theorem** (empirical 57/57 + structural pureness) |
| P46-2 CI-gate composition faithfulness | **Theorem** (structural + tests) |
| P46-3 capability declaration ≠ residency | **Theorem** (pure-function lookup + metadata plumbing) |
| C46-1 external blockers are boundary-shaped | **Conjecture** (supported by §9.8 inventory) |
| C46-2 70B ceiling lift + substrate invariance | **Conjecture** (resolvable on model residency) |
| C46-3 CI consumption closed under profile extension | **Conjecture** (structurally expected under current schema) |

---

## Part C — Architecture

### C.1 New / extended modules

```
vision_mvp/product/import_data.py                  [NEW]   ~260 LOC
vision_mvp/product/ci_gate.py                      [NEW]   ~260 LOC
vision_mvp/product/profiles.py                     [EXTENDED] +~60 LOC
vision_mvp/product/runner.py                       [EXTENDED] +10 LOC (artifact snapshot + model_metadata)
vision_mvp/tests/test_phase46_deployment.py        [NEW]    12 tests
vision_mvp/RESULTS_PHASE46.md                      [NEW]    this document
docs/context_zero_master_plan.md                   [EXTENDED] §9 refresh, §9.9 boundary vs internal
README.md                                          [EXTENDED] Phase-46 thread
ARCHITECTURE.md                                    [EXTENDED] boundary surface pointer
MATH_AUDIT.md                                      [EXTENDED] P46-1..3 + C46-1..3

vision_mvp/artifacts/phase46_public_audit/         [NEW]    bundled + missing-file audits
vision_mvp/artifacts/phase46_ci_gate/              [NEW]    aggregate verdict over 3 Phase-45 RCs
vision_mvp/artifacts/phase46_frontier_slot/        [NEW]    frontier-slot recorded launch
```

No file under `vision_mvp/core/`, `vision_mvp/tasks/`, or
`vision_mvp/experiments/` is edited.

### C.2 Boundary surface diagram

```
  ┌──────────────────────────────────────────────────────────────┐
  │ External inputs                                              │
  │   - public SWE-bench-Lite JSONL (🧱 external)                 │
  │   - ≥70B coder-finetuned Ollama model (◐ engineering)         │
  │   - downstream CI pipeline                                    │
  └──────────────────────────────────────────────────────────────┘
                          │   (one-command boundary crossings)
                          ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ Phase 46 — Boundary surface                                  │
  │   - import_data.audit_jsonl   (schema + P44-3 gate)           │
  │   - ci_gate.evaluate_report   (5-check verdict)               │
  │   - profiles.model_availability (slot vs resident)            │
  └──────────────────────────────────────────────────────────────┘
                          │   (all consumers below are programme-internal)
                          ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ Phase 45 — Product runner + profiles                          │
  └──────────────────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────────────────┐
  │ Phases 40..44 — Bridge + sandbox + parser + matcher + taxonomy│
  └──────────────────────────────────────────────────────────────┘
                          │
  ┌──────────────────────────────────────────────────────────────┐
  │ Phases 31..38 — Substrate + typed-handoff team                │
  └──────────────────────────────────────────────────────────────┘
```

---

## Part D — Evaluation

### D.1 Public-data import on the bundled bank

```
python3 -m vision_mvp.product.import_data \
    --jsonl vision_mvp/tasks/data/swe_lite_style_bank.jsonl \
    --out   vision_mvp/artifacts/phase46_public_audit/bundled_bank_audit.json
```

Result: `ok=True`, 57 rows, shapes `{"ambiguous": 57}`,
0 decode errors, 0 non-object rows, 0 duplicates, readiness
READY with 57/57 on all five checks in 4.8 s. Exit code 0.

### D.2 Public-data import on a missing path

```
python3 -m vision_mvp.product.import_data \
    --jsonl /nonexistent/public_swe_bench_lite.jsonl
```

Result: `ok=False`, `error_kind=file_not_found`. **Exit code 2.**
This is the exact path an operator takes when they have not yet
obtained the public SWE-bench-Lite JSONL; the exit code is
distinct from a schema-blocker exit (1), so CI tooling can
distinguish *"your file is broken"* from *"your file does not
exist"*.

### D.3 Public-data import on a synthetic broken JSONL

Covered by `test_import_broken_jsonl_surfaces_blockers`. A
four-row file with one decode error, one array row, one
missing-keys row, and one duplicate id produces a blockers
list containing `json_decode_errors:1`, `non_object_rows:1`,
`unusable_rows:1`, `duplicate_instance_ids:1`.

### D.4 CI gate over the three Phase-45 release candidates

```
python3 -m vision_mvp.product.ci_gate \
    --report \
        vision_mvp/artifacts/phase45_rc_bundled/product_report.json \
        vision_mvp/artifacts/phase45_rc_mock_sweep/product_report.json \
        vision_mvp/artifacts/phase45_mac1_recorded/product_report.json \
    --out vision_mvp/artifacts/phase46_ci_gate/aggregate_verdict.json
```

Per-report results:

| report | profile | schema | profile | readiness | sweep | artifacts | **ok** |
|---|---|---|---|---|---|---|---|
| `phase45_rc_bundled` | `bundled_57` | ✅ | ✅ | ✅ 57/57 | ✅ none configured | ✅ 3/3 | **True** |
| `phase45_rc_mock_sweep` | `bundled_57_mock_sweep` | ✅ | ✅ | ✅ 57/57 | ✅ mock 1.000 × 2 cells | ✅ 3/3 | **True** |
| `phase45_mac1_recorded` | `aspen_mac1_coder` | ✅ | ✅ | ✅ 57/57 | ✅ real recorded | ✅ 3/3 | **True** |

Aggregate `ok=True`, n_reports=3, 0 aggregate blockers. Exit
code 0.

### D.5 Frontier-model slot exercise

```
python3 -m vision_mvp.product --profile aspen_mac1_coder_70b \
    --out-dir vision_mvp/artifacts/phase46_frontier_slot
```

Readiness READY (57/57 in ~4.5 s). Recorded launch command
points at `qwen2.5-coder:70b` on macbook-1 with raw capture
on. `sweep_launch.json["model_metadata"]["availability"]`
= `pending_availability`, tags include `frontier`,
`canonical_coder`, `semantic_headroom`,
`slot_pending_availability`. **No LLM was invoked.**

### D.6 Unit-test pass and regression

* Phase 46: 12/12 green (`test_phase46_deployment.py`).
* Phase 45: 11/11 green (regression, runner artifact-list
  fix picked up cleanly).
* Phase 41..44: 91/91 green (114.66 s).

---

## Part E — Operator path (post-Phase-46)

The two remaining blockers in master plan §9.8 now meet a
one-command path each.

**Public SWE-bench-Lite JSONL arrives on disk:**

```
# 1. Audit the shape + readiness.
python3 -m vision_mvp.product.import_data \
    --jsonl /path/to/swe_bench_lite.jsonl \
    --out   audit.json

# 2. If ok, run the product path.
python3 -m vision_mvp.product --profile public_jsonl \
    --jsonl /path/to/swe_bench_lite.jsonl \
    --out-dir vision_mvp/artifacts/public_lite

# 3. Gate a downstream CI pipeline on the report.
python3 -m vision_mvp.product.ci_gate \
    --report vision_mvp/artifacts/public_lite/product_report.json \
    --min-ready-fraction 0.95
```

**≥70B coder-finetuned model becomes resident:**

```
# 1. Mark the model resident in the capability table
#    (change slot_pending_availability → semantic_headroom only).
# 2. Run the frontier profile on the correct cluster node.
python3 -m vision_mvp.product --profile aspen_mac1_coder_70b \
    --out-dir vision_mvp/artifacts/mac1_70b_recorded
# 3. Kick off the recorded launch cmd on macbook-1.
# 4. Feed the resulting Phase-42/44 artifacts into the
#    existing phase44_semantic_residue --analyse-only driver
#    for the refined-residue update.
```

---

## Part F — Future work

### F.1 Live Ollama probe for `M_avail`

Replace the static `_MODEL_CAPABILITY[...]
"slot_pending_availability"` tag with a live HTTP check
against `http://<host>:11434/api/tags`. Keeps the
declarative interface; swaps the implementation. One pure
function.

### F.2 Public-JSONL row-schema-drift detector

Phase 46 classifies a row as `unusable` when it lacks keys.
A future detector could check *value* shape (e.g.
`problem_statement` is a string of minimum length,
`base_commit` is a 40-char SHA). Would close C46-1's
falsifier window on public-data drift.

### F.3 GitHub Actions workflow template

Ship a `.github/workflows/product-gate.yml` template that
runs the bundled_57 profile on every push and fails the build
on a non-zero `ci_gate` exit. Operator-facing, zero new
programme-internal code. Deferred because §7.5 still applies
to productisation.

### F.4 Non-Phase-46

Phase 46 does not change the Phase 31..45 stack. All
corresponding claims stand.

---

## Part G — One-paragraph summary

Phase 46 closes the boundary between the Phase-45 finished
product and the outside world. A turnkey public-data import
CLI classifies row shape, catches decode / object / duplicate
errors, and delegates to the Theorem-P44-3 readiness check;
exit codes distinguish missing file from schema blocker. A
CI consumer over `product_report.json` emits a deterministic
five-check verdict (schema / profile / readiness threshold /
sweep outcome / artifact presence) suitable for an external
CI pipeline. A frontier-model profile slot
(`aspen_mac1_coder_70b`) + the `model_availability()`
declarative check separate *capability declaration* from
*residency*, so adding a 70B coder model is a one-string
config change. Three new theorems (import-audit saturation,
CI-gate faithfulness, capability-vs-residency separation) and
three new conjectures (boundary-shape of remaining blockers,
70B semantic ceiling + substrate invariance, CI-closure under
profile extension). Every programme-internal theorem from
Phases 31..45 is preserved byte-for-byte; the only remaining
blockers are the arrival of the public JSONL and the
residency of the 70B model, both of which now meet the code
at a single command.

---
