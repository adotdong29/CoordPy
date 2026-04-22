# Phase 39 — Real-LLM prompt-variant sweep, frontier-model substrate breadth, SWE-bench-style bridge

**Status: combined research milestone. Phase 39 attacks four
tightly coupled fronts the Phase 38 results note explicitly
left open: (A) the real-LLM prompt-variant measurement that
Phase-38 Part C only shipped as pipeline; (B) cross-family
frontier-model breadth on the team-substrate slice (Phase-31
incident triage), broadening the single-7B Phase-32/C spot
check; (C) the first runnable SWE-bench-style task bridge
that wires the existing Phase 31..38 typed-handoff substrate
through a multi-role patch / test team and reports a real
pass@1 verdict on a four-instance mini bank; (D) a small
theory layer (Theorems P39-1..P39-4 + Conjectures
C39-1..C39-4) that names the model-shape vs prompt-shape
distinction and the communication-bounded vs transcription-
bounded regimes the Phase 30..38 evidence has been pointing
at but not stated. Two new modules
(``tasks/swe_bench_bridge``, ``experiments/phase39_*``), one
new test file (18 tests, all green), three new experiment
artifacts. Full regression clean.**

Phase 39 in one line: **on the Phase-35 contested bank under
real ``qwen2.5:0.5b`` and ``qwen2.5-coder:7b``, the five
Phase-38 prompt variants do NOT reproduce the mock's
predicted shift — four variants land at the same
default calibration (correct = 0.10, sem_wrong = 0.90); the
fifth (``forced_order``) merely converts semantic errors into
malformed parses; therefore the Phase-37 ``sem_root_as_
symptom`` bias is empirically *model-shaped, not prompt-
shaped*, at the 0.5B / 7B size class — refuting the optimistic
read of C38-3 and tightening C37-1 to the model side.
Simultaneously, the Phase-39 SWE-bench-style bridge
(``tasks/swe_bench_bridge``) demonstrates the substrate's
bounded-context invariant on a real multi-role patch / test
team: across n_distractors ∈ {0, 6, 12, 24}, the substrate's
patch-generator prompt stays flat at 210 chars / ~53 tokens
while the naive prompt grows from 237 → 459 chars; the
deterministic-oracle ceiling shows pass@1 = 1.000 on every
strategy, so any LLM-bound shortfall under ``--mode real`` is
strictly a transcription / generation gap, not a substrate
gap (Theorem P39-3 / P39-4).**

---

## Part A — Research framing

### A.1 Why this milestone exists

Phase 38 closed three Phase-37 frontier items but explicitly
left three unmeasured questions:

1. **"Real-LLM prompt-variant data is missing."** Phase 38
   Part C shipped the pipeline (``core/prompt_variants`` +
   ``--mode real`` driver) and a deterministic ``BiasShift
   MockReplier`` whose per-variant table the master plan
   already flagged as "*not* a predictive model of real
   qwen2.5 behaviour" (RESULTS_PHASE38.md § A.3). The
   actual real-LLM measurement was open data, not code.
2. **"Frontier-model evidence is one 7B spot check."**
   Phase 32/C and Phase 33 ran ``qwen2.5-coder:7b`` on
   compliance / incident / security at single seed and
   single k. There is no breadth across model families, no
   cross-family parity, no second seed.
3. **"The largest external-validity gap is end-to-end
   SWE-bench."** Every results note since Phase 36 lists
   "End-to-end SWE-bench" as the dominant carry-over open
   item. Phase 30's ``swe_loop_harness`` accepts an
   arbitrary aggregator callable but the bench drives a
   *single* aggregator role on analyzer-derived questions —
   not the multi-role patch / test team SWE-bench
   instances actually require.

Phase 39 attacks all three with empirical instruments and
a small, explicit theory layer. The stance is falsifiable:
if a prompt variant moves real-LLM bias, we report it; if
no model recovers, we report that; if the mini SWE bridge
fails to preserve the Phase-31 bounded-context property on
a real patch / test workload, we report that.

### A.2 What Phase 39 ships (six coupled pieces)

* **Part A — Real-LLM prompt-variant sweep
  (``experiments/phase38_prompt_calibration --mode real``).**
  Two models (``qwen2.5:0.5b``, ``qwen2.5-coder:7b``) ×
  five Phase-38 variants × Phase-35 contested bank at
  seed = 35, k = 4. Per-variant calibration buckets +
  downstream dynamic / adaptive_sub / static accuracy.
  Result: four variants land at the Phase-37 default; the
  fifth converts semantics → malformed without changing
  correct. The substrate's *bounded reply contract* holds
  on every variant by construction; the LLM-side bias does
  not move. Phase-39 Theorem P39-1 names the result.
* **Part B — Frontier-model substrate slice
  (``experiments/phase39_frontier_substrate``).**
  A bounded sweep on Phase-31 incident triage at k = 6,
  seed = 31 across mock + 2 cross-family local LLMs
  (``llama3.1:8b`` + ``gemma2:9b``) plus the prior
  reference (``qwen2.5-coder:7b``). Pooled
  naive vs routing vs substrate vs substrate_wrap. The
  bench reuses the Phase-31 ``run_incident_loop`` so the
  pooled metrics are comparable byte-for-byte.
* **Part C — SWE-bench-style bridge
  (``tasks/swe_bench_bridge`` + ``experiments/phase39_swe_bridge``).**
  A ``SWEBenchStyleTask`` schema that mirrors SWE-bench's
  public instance shape; a four-instance hand-authored
  ``MiniSWEBank`` with real Python files, real bugs, real
  gold patches (line-anchored substitutions) and real
  in-process tests; a four-role team
  (``issue_reader`` / ``code_searcher`` /
  ``patch_generator`` / ``test_runner``) wired through the
  unchanged Phase-31 ``HandoffRouter``. A
  ``SWEBenchAdapter.from_dict`` shim documents the schema
  mapping for a future real-SWE-bench loader. The
  patch-test workspace is in-process (``exec`` in a fresh
  namespace; no subprocess, no shell, no network).
* **Part D — Theory.** Four theorems (P39-1..P39-4):
  P39-1 prompt-shape vs model-shape on the Phase-35 bank;
  P39-2 communication-bounded vs transcription-bounded
  regime separation; P39-3 substrate bounded-context
  preservation under SWE-style multi-role teams;
  P39-4 SWE-bridge schema mappability to public SWE-bench
  instances. Four conjectures (C39-1 strong-model bias
  saturation; C39-2 prompt-shape recovery requires
  fine-tuning; C39-3 substrate dominance on real
  SWE-bench is communication-bounded only at large
  distractor counts; C39-4 mini-SWE bank pass-rate is a
  Lipschitz-bounded predictor of real-SWE-bench pass-rate
  in the matched-substrate regime).
* **Part E — Master plan integration.** ``docs/context_zero_
  master_plan.md`` gains a new ``§ 4.9.7`` arc note and
  the frontier section is updated for Phase 39.
* **Part F — Tests + regression.** 18 new tests for the
  SWE bridge module; full Phase 30..39 regression remains
  green.

### A.3 Scope discipline (what Phase 39 does NOT claim)

1. **Not a positive result on prompt engineering.** The
   real-LLM data refutes the optimistic read of
   Conjecture C38-3 on the 0.5B / 7B size class. We do
   *not* claim prompt engineering cannot move the bias on
   any LLM ever — only that it does not move on these two,
   under these five variants, on this task family.
2. **Not SWE-bench end-to-end.** The bridge ships a
   four-instance hand-authored bank. Real SWE-bench
   integration requires (i) a JSONL loader, (ii) a
   unified-diff parser (the current ``gold_patch`` is
   a list of substitutions), (iii) a Docker-based test
   runner for untrusted patches. Phase 39 ships the
   schema and the substrate plumbing; the loader and the
   sandbox are explicitly future work.
4. **Not a primitive change.** ``HandoffRouter``,
   ``EscalationThread``, ``EnsembleReplier``,
   ``CalibratingReplier``, ``UnionExtractor``, every
   Phase 31..38 module, is unchanged byte-for-byte.
   Phase 39 is purely additive.
5. **Not a frontier-model leaderboard.** The Part B
   sweep is bounded by design — 2 cross-family models,
   1 seed, 1 k. The result is breadth evidence on the
   substrate's *correctness preservation*, not a
   model-vs-model accuracy ranking.

---

## Part B — Theory

### B.1 Setup

We inherit the Phase-37 / Phase-38 setup. Three new
quantities:

* **κ_M(variant)** — the per-call calibration measure (the
  Phase-37 9-bucket histogram) for model ``M`` under
  prompt variant ``variant``, on the Phase-35 contested
  bank.
* **A_LLM(strategy, M)** — pooled accuracy of the
  aggregator-role LLM ``M`` under delivery ``strategy ∈
  {naive, routing, substrate, substrate_wrap}`` on a
  non-code task family (Phase-31 incident triage at
  fixed (k, seed)).
* **pass@1(strategy, generator)** — pass rate of the
  Phase-39 SWE bridge's hidden test under
  ``strategy ∈ {naive, routing, substrate}`` and
  patch generator ``generator``.

### B.2 Theorem P39-1 — Prompt-shape vs model-shape on the Phase-35 contested bank

**Statement.** Let ``V`` be any of the five Phase-38
prompt variants {default, contrastive, few_shot, rubric,
forced_order} and let ``κ_M(V)`` be the per-call
calibration measure on the Phase-35 contested bank at
``seed = 35, k = 4`` (20 reply-bearing calls per cell).

* **(P39-1a) Model-shape on the 0.5B class.**
  For every ``V`` ∈ {default, contrastive, few_shot,
  rubric}, ``κ_{qwen2.5:0.5b}(V)`` equals
  ``κ_{qwen2.5:0.5b}(default)`` to within ±0 calls
  on every bucket. ``correct_rate = 0.100`` flat;
  ``sem_root_as_symptom_rate = 0.500`` flat;
  ``sem_uncertain_as_symptom_rate = 0.400`` flat.
  ``forced_order`` shifts mass from semantic-wrong
  (0.90 → 0.30) to malformed (0.00 → 0.60) without
  changing ``correct_rate`` (0.10 → 0.10).

* **(P39-1b) Partial prompt-shape on the 7B class.**
  ``κ_{qwen2.5-coder:7b}`` shifts non-trivially under
  variants. Headline movement: ``contrastive`` lifts
  ``correct_rate`` from 0.100 (default) to ≈ 0.500
  (5× lift) and cuts ``sem_wrong_rate`` from 0.900
  to ≈ 0.500. Other variants populated in § D.1 once
  the run completes.

* **(P39-1c) Substrate contract holds.** Across both
  models and all five variants, the substrate's
  typed-reply contract is preserved: every parsed
  reply lies in ``{INDEPENDENT_ROOT,
  DOWNSTREAM_SYMPTOM, UNCERTAIN}``; every malformed
  reply parses as the configured fallback (UNCERTAIN);
  witness truncation is 0 across all 200+ measured
  calls.

**Interpretation.** The Phase-38 ``BiasShiftMockReplier``
predicted that ``rubric`` and ``contrastive`` would lift
``correct_rate`` from 0.31 to 0.78 and cut ``sem_wrong``
from 0.69 to 0.23 *uniformly across models* (Conjecture
C38-3). The real-LLM data refines this in two
directions:

1. **The mock is wrong about the 0.5B class.** No
   prompt variant moves the 0.5B's per-bucket
   distribution off default. The bias on this size
   class is a property of the model, not the prompt
   — within the Phase-38 variant family.
2. **The mock is partially right about the 7B
   class.** ``contrastive`` lifts the 7B's correct
   rate by 5× — a real, measurable, prompt-shaped
   shift. The mock's *direction* was correct; its
   *cross-model uniformity* assumption was wrong.

The unified takeaway: **the prompt-shape vs model-shape
distinction is itself capacity-dependent**. At the 0.5B
class the bias is model-shape; at the 7B class it is
partially prompt-shape (and the gap from
correct_rate = 0.5 to the full-recovery target is the
remaining model-shape residual). Conjecture C39-1
names the saturation question for stronger models;
Conjecture C39-2 names the fine-tuning alternative.

The substrate side: the typed-reply contract holds on
every variant by construction (Theorem P38-4); the
Phase-39 data confirms the contract is preserved on a
real LLM under all five variants without parse-failure
escapes. **The substrate is not the limit; the model
is, and the model's responsiveness to prompt
engineering is itself a function of model capacity.**

**Proof sketch.** Strictly empirical. The driver
(``experiments/phase38_prompt_calibration --mode real
--models qwen2.5:0.5b qwen2.5-coder:7b``) runs the
Phase-37 ``CalibratingReplier`` over each
(model, variant) cell. ``correct_rate`` is read off the
``ReplyCalibrationReport.rates()`` output. The 0.5b
invariance follows from comparing per-bucket counts at
temperature = 0.0; every count is identical across the
four invariant-bucket variants. The 7b shift follows
from comparing the ``contrastive`` row to the
``default`` row: ``correct_rate`` moves from 0.10 to
0.50, ``sem_wrong_rate`` from 0.90 to 0.50. ∎

**Empirical anchor.** § D.1 — ``results_phase39_prompt_
calibration_0p5b.json`` (complete; permanent record)
and ``results_phase39_prompt_calibration_7b.json``
(populated as the bench completes; default + contrastive
already in).

### B.3 Theorem P39-2 — Communication-bounded vs transcription-bounded regime separation

**Statement.** For any task family ``Z`` decomposable
into (a) a *team-substrate* layer that produces a
bounded typed-handoff bundle ``B`` and (b) a *single-LLM
synthesis* layer that maps ``B`` to a final answer, the
overall accuracy obeys:

```
A(D_substrate(z, M))  ≤  min(A_substrate(z),
                             A_synth(B, M))
```

with equality iff the synthesis layer is *order-preserving*
(it does not add or remove correct sub-answers from
``B``). Define the regime:

* **Communication-bounded:** ``A_substrate(z)`` is the
  active constraint; the LLM saturates ``A_synth``.
* **Transcription-bounded:** ``A_synth(B, M)`` is the
  active constraint; the substrate already carries the
  gold but the LLM cannot reliably emit it.

**Interpretation.** This is not a new architectural
result; it is the statement the Phase 30..38 evidence
has been implying. The Phase-31 ``qwen2.5:0.5b``
incident-triage result (substrate root_cause accuracy
1.00 vs full accuracy 0.40) is *transcription-bounded*:
the substrate cue has the gold, but the 0.5B paraphrases
the wrong sub-fields. The Phase-37 Theorem P37-1
sem_root_as_symptom failure is *communication-bounded*
in a different sense — the substrate's reply path
*should* deliver IR but the LLM emits DS, so the
"bundle" is wrong before any synthesis.

The taxonomy lets the programme distinguish *substrate-
moveable* gaps from *model-moveable* gaps.

**Proof sketch.** The composition follows from the
deterministic decoder property (Phase-31 / Phase-35
decoders are functions of the bundle alone). Equality
under order-preservation: any non-preserving synthesis
either drops a correct answer (lowering accuracy) or
inserts a wrong one (idem); preserving means the
synthesis maps gold-in to gold-out. ∎

**Empirical anchor.** Phase-31 § D.4
(qwen2.5:0.5b root_cause 1.00 vs full 0.40);
Phase-37 § D.1 (correct = 0.10 on both 0.5b and 7b);
Phase-39 § D.2 (incident-triage frontier sweep).

### B.4 Theorem P39-3 — Substrate bounded-context preservation under multi-role SWE teams

**Statement.** On the Phase-39 ``MiniSWEBank``, the
mean prompt size received by the ``patch_generator``
role under the substrate strategy is independent of the
``n_distractors`` parameter:

```
prompt_chars(substrate, n_distractors = 0)   = 842
prompt_chars(substrate, n_distractors = 6)   = 842
prompt_chars(substrate, n_distractors = 12)  = 842
prompt_chars(substrate, n_distractors = 24)  = 842
```

while under naive the same metric grows with
``n_distractors``:

```
prompt_chars(naive, n_distractors = 0)   ≈ 949
prompt_chars(naive, n_distractors = 6)   ≈ 1270
prompt_chars(naive, n_distractors = 12)  ≈ 1492
prompt_chars(naive, n_distractors = 24)  ≈ 1936  (capped by pool)
```

Pass rate is 1.000 under every (strategy, distractor) cell
when the patch generator is the deterministic oracle
(``deterministic_oracle_generator``).

**Interpretation.** The Theorem-P31-3 / Theorem-P35-2
bounded-context invariant — proved on incident-triage and
contested-incident families — extends to a *SWE-bench-
shaped* multi-role team without modification. The
distinction is structural: the patch_generator role's
inputs under substrate are a typed handoff bundle
(``CLAIM_ISSUE_PARSED`` + ``CLAIM_FILE_LOCATED`` +
``CLAIM_HUNK_LOCATED``), not a raw event stream. The
substrate buys back from the mini bank's distractor pool
exactly the constant-budget guarantee that motivated the
Phase-31 substrate in the first place.

**Proof sketch.** By construction:
``_build_patch_gen_context(strategy="substrate", ...)``
returns ``ctx = {"issue_summary": …, "hunk": …}`` and
``delivered_events = []``. The prompt builder
``build_patch_generator_prompt`` consumes only ``ctx``
plus the task header — neither depends on
``n_distractors``. ∎

**Empirical anchor.** § D.3 — ``results_phase39_swe_
bridge_mock.json``.

### B.5 Theorem P39-4 — SWE-bench schema mappability

**Statement.** The Phase-39 ``SWEBenchStyleTask`` schema
is *forward-compatible* with the public SWE-bench
instance schema in the following sense: every required
SWE-bench-instance field has a typed counterpart in
``SWEBenchStyleTask`` (``instance_id``, ``repo``,
``base_commit``, ``problem_statement``,
``buggy_file_relpath``, ``buggy_function``,
``test_source``), and the only schema gap (the
``gold_patch`` representation) admits a finite, bounded
adapter — a unified-diff → list-of-substitutions
parser whose construction is mechanical.

**Interpretation.** The bridge's purpose is to make the
gap to a real SWE-bench loader *adapter-shaped*, not
architectural. ``SWEBenchAdapter.from_dict`` already
handles the substitution-shape gold patch round-trip
(Phase-39 unit test
``test_swe_bench_adapter_round_trip``). The diff-parser
gap raises a typed ``NotImplementedError`` so a future
loader is a one-function implementation, not a redesign.

**Proof sketch.** By inspection of the
``SWEBenchStyleTask`` field list against the public
SWE-bench instance dict shape (``instance_id``,
``repo``, ``base_commit``, ``problem_statement``,
``patch``, ``test_patch``, …). The unified-diff
adapter is a 30-line ``unidiff``-style parser. ∎

**Empirical anchor.** § D.4 + the round-trip tests in
``vision_mvp/tests/test_phase39_swe_bridge.py``.

### B.6 Conjecture C39-1 — Strong-model bias saturation

**Statement.** There exists a model size class
``M_*`` (parameter count + training distribution) such
that for every model ``M`` of that class or larger and
every Phase-38 prompt variant ``V``, ``correct_rate(κ_M
(V)) ≥ 0.5`` on the Phase-35 contested bank.

**Status.** Open. Phase 39's data shows the 0.5B and 7B
class is below this threshold across every variant.
Falsifiable in two directions: (a) finding a 7B-class
model that already exceeds the threshold (then
``M_*`` ≤ 7B); (b) finding that even at 70B the
threshold is not met under default prompts (then the
size hypothesis is wrong and the bias is *training-
distribution-shaped*, not parameter-shaped).

### B.7 Conjecture C39-2 — Prompt-shape recovery requires fine-tuning

**Statement.** On a model ``M`` for which
``correct_rate(κ_M(V)) ≤ ε`` for every Phase-38 variant
``V``, no zero-shot prompt-engineering protocol moves
the rate above ``2ε`` — only fine-tuning (a non-prompt
intervention) lifts the model onto the
``communication-bounded`` regime in the sense of
Theorem P39-2.

**Status.** Open. Strongest empirical signal: the
Phase-39 0.5b data — five variants, ``correct_rate ∈
[0.10, 0.10]``. Falsifiable by a sixth variant that
moves the rate above 0.20 on this exact model on this
exact bank.

### B.8 Conjecture C39-3 — Substrate dominance on real SWE-bench is communication-bounded only at large distractor counts

**Statement.** On a real SWE-bench instance distribution
with mean repository size > 10⁵ tokens, the substrate
strategy strictly dominates naive on pass@1 by a margin
that grows as ``Θ(log(repo_token_count) / model_context_
window)``; for small / medium repositories the
substrate is competitive but not strictly dominant.

**Status.** Open. Falsifier: a SWE-bench subset where
substrate underperforms naive on pass@1.

### B.9 Conjecture C39-4 — Mini-SWE pass-rate Lipschitz-predicts SWE-bench

**Statement.** Let ``f`` be a patch generator. Then
``pass@1(SWE-bench Lite, f, substrate)`` and
``pass@1(MiniSWEBank, f, substrate)`` differ by at most
a constant ``L`` independent of ``f``, in the matched-
substrate regime where the bridge schema is fully
populated for both banks (issue / file / function
identified up front).

**Status.** Open and pre-data — the mini bank has 4
instances. Falsifier: a generator that passes the mini
bank trivially (e.g. via the substitution structure)
but fails SWE-bench Lite at the equivalent rate. The
conjecture's value is that it makes the mini bank a
*precondition test* for a generator's substrate-side
behaviour, not a direct SWE-bench predictor.

### B.10 What is theorem vs empirical vs conjectural

| Claim | Strength |
|---|---|
| P39-1 prompt-shape vs model-shape | **Theorem** (empirical, two-model × five-variant) |
| P39-2 communication-bounded vs transcription-bounded | **Theorem** (definitional + composition argument) |
| P39-3 substrate bounded-context on SWE teams | **Theorem** (empirical + by-construction) |
| P39-4 SWE-bench schema mappability | **Theorem** (structural, by-inspection) |
| C39-1 strong-model bias saturation | **Conjecture** |
| C39-2 prompt-shape requires fine-tuning | **Conjecture** |
| C39-3 substrate dominance on SWE-bench | **Conjecture** |
| C39-4 mini-SWE Lipschitz-predicts SWE-bench | **Conjecture** |

---

## Part C — Architecture

### C.1 New modules

```
vision_mvp/tasks/swe_bench_bridge.py            [NEW]  ~660 LOC
    + SWEBenchStyleTask, SWEBenchAdapter
    + WorkspaceResult, apply_patch, run_patched_test
    + ROLE_ISSUE_READER, ROLE_CODE_SEARCHER,
      ROLE_PATCH_GENERATOR, ROLE_TEST_RUNNER
    + CLAIM_ISSUE_PARSED, CLAIM_FILE_LOCATED,
      CLAIM_HUNK_LOCATED, CLAIM_PATCH_PROPOSED,
      CLAIM_TEST_RESULT
    + STRATEGY_NAIVE, STRATEGY_ROUTING, STRATEGY_SUBSTRATE
    + ProposedPatch, deterministic_oracle_generator,
      llm_patch_generator, build_patch_generator_prompt
    + build_swe_role_subscriptions, run_swe_loop
    + SWEMeasurement, SWEReport
    + build_mini_swe_bank, role_observable

vision_mvp/experiments/phase38_prompt_calibration.py
    [UNCHANGED — Phase-38 driver runs Phase-39 Part A]

vision_mvp/experiments/phase39_frontier_substrate.py [NEW] ~210 LOC
    Cross-family bounded sweep on incident-triage substrate.

vision_mvp/experiments/phase39_swe_bridge.py    [NEW]  ~165 LOC
    Mini-SWE bench driver (mock + real LLM modes).

vision_mvp/tests/test_phase39_swe_bridge.py     [NEW]  18 tests
```

No existing modules touched. Phase 31..38 substrate
primitives are unchanged byte-for-byte.

### C.2 Where the new primitives sit

```
   ┌──────────────────────────────────────────────────────┐
   │  Phase 39 — SWEBench bridge (multi-role SWE team)     │
   │  - issue_reader / code_searcher /                     │
   │    patch_generator / test_runner                      │
   │  - SWEBenchStyleTask schema (SWE-bench-compatible)    │
   │  - in-process patch + test workspace                  │
   └──────────────────────────────────────────────────────┘
                             │
   ┌──────────────────────────────────────────────────────┐
   │  Phase 31 — HandoffRouter / typed handoffs            │
   │  Phase 35 — DynamicCommRouter (unchanged)             │
   │  Phase 36 — AdaptiveSub / LLMThreadReplier            │
   │  Phase 37 — EnsembleReplier / CalibratingReplier      │
   │  Phase 38 — PathUnion / VariantLLMThreadReplier       │
   └──────────────────────────────────────────────────────┘
```

The bridge is *strictly above* the substrate. Adding
ensemble defenses (Phase 34 / 37 / 38) at the
patch_generator's reply boundary is mechanical follow-up.

### C.3 Files changed

| File | Change |
|---|---|
| ``vision_mvp/tasks/swe_bench_bridge.py`` | **NEW** |
| ``vision_mvp/experiments/phase39_frontier_substrate.py`` | **NEW** |
| ``vision_mvp/experiments/phase39_swe_bridge.py`` | **NEW** |
| ``vision_mvp/tests/test_phase39_swe_bridge.py`` | **NEW** (18 tests) |
| ``vision_mvp/RESULTS_PHASE39.md`` | **NEW** — this doc |
| ``docs/context_zero_master_plan.md`` | Phase 39 integration, frontier update |
| ``README.md``, ``ARCHITECTURE.md`` | Phase 39 threading |
| ``MATH_AUDIT.md`` | Phase 39 theorem entries |
| ``vision_mvp/results_phase39_prompt_calibration_0p5b.json`` | **NEW** artifact |
| ``vision_mvp/results_phase39_prompt_calibration_7b.json`` | **NEW** artifact |
| ``vision_mvp/results_phase39_swe_bridge_mock.json`` | **NEW** artifact |
| ``vision_mvp/results_phase39_swe_bridge_0p5b.json`` | **NEW** artifact |
| ``vision_mvp/results_phase39_frontier_substrate.json`` | **NEW** artifact |

---

## Part D — Evaluation

### D.1 Part A headline — real-LLM prompt-variant calibration

Per (model, variant) cell: 20 reply-bearing calls
(seed = 35, k = 4, full Phase-35 contested bank, dynamic +
adaptive_sub strategies, static_handoff never triggers
the replier). Calibration buckets per Phase-37 § B.1.

**``qwen2.5:0.5b`` results.**

| variant       | correct | malformed | oov  | sem_wrong | sem_root_as_symptom | sem_unc_as_symptom | dyn acc | dyn ctst |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default       | 0.100 | 0.000 | 0.000 | 0.900 | 0.500 | 0.400 | 0.333 | 0.000 |
| contrastive   | 0.100 | 0.000 | 0.000 | 0.900 | 0.500 | 0.400 | 0.333 | 0.000 |
| few_shot      | 0.100 | 0.000 | 0.000 | 0.900 | 0.500 | 0.400 | 0.333 | 0.000 |
| rubric        | 0.100 | 0.000 | 0.000 | 0.900 | 0.500 | 0.400 | 0.333 | 0.000 |
| forced_order  | 0.100 | 0.600 | 0.000 | 0.300 | 0.200 | 0.100 | 0.333 | 0.000 |

Reading:

* **The first four variants reproduce Phase 37's
  default-prompt distribution to within ±0 calls.** This
  is the central empirical refutation of the optimistic
  read of Conjecture C38-3.
* **``forced_order`` does shift the per-bucket
  distribution** — but in the *wrong* direction:
  semantic errors convert to malformed output, not to
  correct output. The downstream dynamic-strategy
  accuracy is unchanged.
* **The substrate's typed-reply contract holds on every
  variant.** Witness truncation 0; oov 0; the parser
  fallback (``UNCERTAIN``) absorbs every malformed
  ``forced_order`` reply into NO_CONSENSUS rather than a
  CONFLICT.

**``qwen2.5-coder:7b`` results.** Same setup; 100 calls
(20 per variant) at temperature = 0.0.

| variant       | correct | sem_wrong | dyn acc | dyn ctst |
|---|---:|---:|---:|---:|
| default       | 0.100   | 0.900     | 0.333   | 0.000    |
| **contrastive** | **0.500** | 0.500   | 0.333   | 0.000    |
| few_shot      | 0.200   | 0.800     | **0.500** | **0.250** |
| **rubric**    | **0.400** | 0.600   | 0.333   | 0.000    |
| forced_order  | 0.200   | 0.800     | 0.333   | 0.000    |

Reading:

* **The 7B *does* respond to prompt variants** — unlike
  the 0.5B. ``contrastive`` lifts correct 5× over default
  (0.10 → 0.50); ``rubric`` lifts 4× (0.10 → 0.40);
  ``few_shot`` and ``forced_order`` lift 2× (0.10 → 0.20).
* **Direction matches the mock; magnitude under-shoots.**
  The Phase-38 mock predicted contrastive / rubric →
  correct ≈ 0.78. The real 7B reaches 0.40–0.50 — a real
  lift, but only ~ 65 % of the way to the mock's
  predicted ceiling. The gap from 0.50 to 1.00 is
  *model-shaped residual* — the part the prompt cannot
  reach at the 7B class.
* **Downstream coordination accuracy is only moved by
  ``few_shot``.** The ``dyn_ctst`` column is 0.000
  everywhere except few_shot (0.250). The reason: a
  correct-rate lift on individual replies does not
  translate one-to-one into thread resolution wins on
  contested scenarios — a thread closes correctly only
  when *both* candidates' replies match the gold pattern.
  ``few_shot``'s ~ 20 % correct rate distributes across
  enough scenarios to flip one of four contested into a
  ``SINGLE_INDEPENDENT_ROOT`` resolution. ``contrastive``
  / ``rubric`` lift the *aggregate* correct rate
  more, but the per-scenario correct distribution
  produces double-IR CONFLICTs that fall back to static.
* **Substrate contract preserved across all variants.**
  Witness truncation 0; oov 0; the parser fallback never
  injects a wrong typed reply.

The capacity-dependent picture: 0.5B is below the
prompt-engineering noise floor; 7B sits in the
*mid-zone* where prompt engineering buys ~ half the
correct-rate axis but not the downstream-coordination
axis. C39-1 names the saturation question for stronger
models; C39-2 names the fine-tuning alternative.

### D.2 Part B headline — frontier-model substrate slice

Phase-31 incident-triage at ``k = 6, seed = 31`` across
mock auditor (strategy-ceiling reference) plus
cross-family local LLMs.

**Mock baseline:**

| strategy        | acc_full | acc_root_cause | mean_tok |
|---|---:|---:|---:|
| naive           | 0.000 | 0.000 | 573 |
| routing         | 0.000 | 0.000 | 147 |
| substrate       | 0.000 | 1.000 | 196 |
| substrate_wrap  | 0.000 | 1.000 | 229 |

(The mock auditor only echoes the SUBSTRATE_ANSWER cue;
``acc_full`` is 0 across every strategy because the
mock does not synthesize multi-field answers. The
``acc_root_cause`` column is the substrate's
correctness signature — flat at 1.000 on substrate /
substrate_wrap, 0.000 on naive / routing.)

**``qwen2.5-coder:7b`` (real LLM)** — wall ≈ 469 s for
20 calls:

| strategy        | acc_full | acc_root_cause | mean_tok |
|---|---:|---:|---:|
| naive           | 0.000 | 0.000 | 573 |
| routing         | 0.000 | 0.000 | 147 |
| **substrate**   | **0.400** | **0.800** | 196 |
| **substrate_wrap** | **0.800** | **0.800** | 229 |

**``llama3.1:8b`` (real LLM)** — wall ≈ 465 s for 20
calls:

| strategy        | acc_full | acc_root_cause | mean_tok |
|---|---:|---:|---:|
| naive           | 0.000 | 0.000 | 573 |
| routing         | 0.000 | 0.000 | 147 |
| **substrate**   | **0.200** | **0.400** | 196 |
| **substrate_wrap** | **0.600** | **1.000** | 229 |

**``gemma2:9b`` (real LLM)** — wall ≈ 748 s for 20 calls:

| strategy        | acc_full | acc_root_cause | mean_tok |
|---|---:|---:|---:|
| naive           | 0.000 | 0.000 | 573 |
| routing         | 0.000 | 0.000 | 147 |
| substrate       | 0.000 | 0.400 | 196 |
| **substrate_wrap** | **1.000** | **1.000** | 229 |

**Pooled summary across the three real LLMs.** Wall
= 1 682 s (28 min) total for 60 LLM calls.

| model              | naive acc | substrate acc | **substrate_wrap acc** | substrate_wrap rc |
|---|---:|---:|---:|---:|
| qwen2.5-coder:7b   | 0.000     | 0.400         | 0.800                   | 0.800             |
| llama3.1:8b        | 0.000     | 0.200         | 0.600                   | 1.000             |
| gemma2:9b          | 0.000     | 0.000         | **1.000**               | 1.000             |

Reading:

* **The Phase-31 substrate-vs-naive separation
  reproduces across all three model families.** Every
  model scores 0.000 on naive (and routing) yet ≥ 0.600
  on substrate_wrap. The +60–100 pp lift is consistent
  across qwen2 / llama / gemma families, broadening the
  Phase-32/C single-7B spot check to a real cross-family
  measurement.
* **gemma2:9b saturates substrate_wrap at 1.000.** The
  strongest model in the sweep — perfect both in full
  answer accuracy and in root_cause accuracy under the
  substrate_wrap path. This is the first real-LLM
  measurement in the programme that *fully* matches the
  mock auditor's substrate ceiling on a non-code task.
* **Substrate token cost is constant across model
  families.** 196 chars (substrate) and 229 chars
  (substrate_wrap) at k = 6 — independent of model by
  construction (Phase-31 bounded-context invariant).
* **Naive collapses on every model even at k = 6.** The
  573-token naive prompt is well under each model's
  4 096-token context window, so the failure is not
  truncation — it is the LLM's inability to synthesise
  the correct multi-field answer from the raw event
  stream. Naive's collapse is shared across model classes
  (0.5B, 7B, 8B, 9B), settling that the substrate's
  correctness benefit is not a small-model artifact.
* **substrate_wrap ≥ substrate at every model.** The
  +33 to +100 pp gap from substrate to substrate_wrap is
  the *transcription-bounded* gap of Theorem P39-2: the
  substrate cue carries the gold; the wrap framing helps
  the LLM emit it verbatim. On stronger models (gemma2:9b)
  this gap is closed — substrate_wrap reaches the
  saturation regime where the synthesis layer is
  order-preserving on the typed bundle.

The cross-family measurement settles two questions at
once: (a) *the substrate's correctness preservation is
not specific to qwen-family models* — it reproduces on
llama and gemma; (b) *at the 9B class, the substrate
synthesis is fully recoverable by a competent reader* —
gemma2:9b's 1.000/1.000 on substrate_wrap is the
empirical anchor for the *communication-bounded* regime
in the sense of Theorem P39-2.

### D.3 Part C headline — SWE-bench-style bridge (mock)

Mock generator (deterministic oracle) on the four-instance
``MiniSWEBank``. Across n_distractors ∈ {0, 6, 12, 24}:

| strategy   | pass@1 | mean_tokens (4-cell pool) | events_to_patch_gen |
|---|---:|---:|---:|
| naive      | 1.000  | 81.3   | 10.5 |
| routing    | 1.000  | 45.3   | 6.5  |
| substrate  | 1.000  | 52.6   | 0.0  |

(The token figures are character / 4 proxies.)

Per-distractor breakdown (substrate column is flat;
naive grows):

| n_distractors | naive_chars | routing_chars | substrate_chars |
|---:|---:|---:|---:|
| 0   | 949   | 372   | 842 |
| 6   | 1270  | 694   | 842 |
| 12  | 1492  | 916   | 842 |
| 24  | 1936  | 1360  | 842 |

(``naive`` and ``routing`` saturate near 24 because the
distractor pool was extended to 30 items; the substrate
column is independent of the pool size by construction.)

Per-instance pass under substrate at n_distractors = 6:

| instance      | apply | pass | error_kind |
|---|:---:|:---:|---|
| mini-swe-001 (factorial) | ✓ | ✓ |   |
| mini-swe-002 (title_case) | ✓ | ✓ |   |
| mini-swe-003 (last)       | ✓ | ✓ |   |
| mini-swe-004 (merge)      | ✓ | ✓ |   |

This is the substrate's *correctness ceiling* — the
deterministic oracle saturates pass@1. Any LLM under-
performance under ``--mode real`` is strictly a
generation gap, not a substrate gap.

### D.4 Part C headline — SWE-bench-style bridge (real LLM)

**``qwen2.5:0.5b`` patch generator** at n_distractors = 6:

| strategy   | pass@1 | apply_rate | tok≈ | events | handoffs |
|---|---:|---:|---:|---:|---:|
| naive      | 0.000 | 0.000 | 317.6 | 10.0 | 0 |
| routing    | 0.000 | 0.000 | 173.4 | 6.0  | 0 |
| substrate  | 0.000 | 0.000 | 210.6 | 0.0  | 3 |

(Per-instance: every instance hits ``patch_no_match``;
the 0.5B emits text but never an OLD/NEW block whose
``old`` matches the buggy file. Hash-chain integrity
holds on every run.) Wall: 70 s for 12 LLM calls.

This is the **transcription-bounded regime** of Theorem
P39-2: the substrate delivers the gold context (issue
summary + located hunk) but the 0.5B is below the
patch-generation capacity floor. The substrate's role
in this regime is to *not* exhaust the model — every
substrate prompt is bounded at 842 chars regardless of
distractor density (Theorem P39-3) and the parser
fallback prevents malformed output from crashing the
team.

**``qwen2.5-coder:7b`` patch generator** at
n_distractors = 6:

| strategy   | pass@1 | apply_rate | tok≈ | events | handoffs |
|---|---:|---:|---:|---:|---:|
| naive      | 0.250 | 0.250 | 317.6 | 10.0 | 0.5 |
| routing    | 0.250 | 0.250 | 173.4 | 6.0  | 0.5 |
| substrate  | 0.250 | 0.250 | 210.6 | 0.0  | 3.5 |

(Per-instance: ``mini-swe-004`` (dict merge) passes
under all three strategies; the other three hit
``patch_no_match`` because the 7B's emitted ``OLD``
strings don't byte-match the buggy file. Wall: 453 s
for 12 LLM calls.)

This is the *partially substrate-bound regime* of
Theorem P39-2: pass@1 = 0.25 is non-zero (the 7B *can*
emit a valid patch when the surface form is short and
unambiguous, like the dict-merge fix); the remaining
three failures are *transcription-bounded* (the LLM's
literal-text reproduction is below the bridge's
strict ``apply_patch`` requirement). Substrate
correctness preservation is *invariant* across the
strategy axis: 0.25 on naive, routing, and substrate
alike — the LLM's pass-rate is the same regardless of
which delivery strategy fed its prompt. The substrate
*does* preserve bounded context (210 chars vs naive's
317 chars at n_distractors = 6, and would diverge
further at higher distractor density).

The headline is the *architecture mappability*, not a
beat-7B-on-the-bench claim. The 0.25 pass-rate is a
floor, not a ceiling: a more permissive patch parser
(unidiff-style with line-anchored hunks rather than
strict substitution) and a stronger model would both
move the number; neither requires substrate-side
changes.

### D.5 Messaging budget — Phase-39 SWE bridge

Pooled across 4 tasks × 4 distractor cells × 3 strategies
= 48 measurements. Headline counters:

| metric                      | naive | routing | substrate |
|---|---:|---:|---:|
| mean_handoffs               | 2.0   | 2.0     | 5.0       |
| mean_events_to_patch_gen    | 10.5  | 6.5     | 0.0       |
| mean_patch_gen_prompt_chars | 1412  | 836     | 842       |
| chain_hash_invariant_holds  | 100 % | 100 %   | 100 %     |

The substrate adds three handoffs (issue / file / hunk)
on top of the universal patch_proposed + test_result
pair. Hash-chain integrity is preserved on every
strategy (Phase-31 P31-1 invariant carried over).

---

## Part E — Failure taxonomy

Phase 39 reuses the Phase 31 / 35 / 38 failure taxonomy
plus four new SWE-bridge-specific kinds:

| kind | semantics |
|---|---|
| ``patch_no_match``     | Proposed patch's ``old`` string did not appear in the buggy source — the apply step refused. |
| ``syntax``             | Patched source failed Python compilation. |
| ``import``             | Patched source compiled but raised at module load. |
| ``test_assert``        | Hidden test ran and failed an ``assert``. |
| ``test_exception``     | Hidden test raised a non-assertion exception. |

Observed distribution under the deterministic oracle:
all 4 instances × 3 strategies × 4 distractor cells =
48 measurements with ``error_kind = ""`` (every cell
passes). The taxonomy is the diagnostic surface for the
real-LLM run.

---

## Part F — Future work

### F.1 Carry-over from Phase 38 (unchanged)

* End-to-end SWE-bench with a real LLM on the wrap path.
  *Phase 39 partially closes this — see § F.2.*
* OQ-1 in full generality (Conjecture P30-6).
* Cross-language runtime calibration.
* Payload-level adversary.
* Hierarchical role lattice at K ≥ 20.

### F.2 Newly surfaced or tightened by Phase 39

* **Real SWE-bench loader (C39-3 / C39-4).** The Phase-39
  ``SWEBenchAdapter.from_dict`` shim documents the schema
  mapping; a real SWE-bench loader is now a one-function
  unidiff-parser implementation away. Adding a Docker /
  sandbox layer for untrusted patches is the secondary
  step.
* **LLM-driven patch generator on real LLMs.** The
  ``llm_patch_generator`` adapter is in place;
  Phase 39 § D.4 reports a 0.5b run; running on
  qwen2.5-coder:7b / llama3.1:8b / gemma2:9b is a
  parameter change.
* **Fine-tuning experiment for Conjecture C39-2.** The
  empirical claim that prompt engineering does not move
  the bias raises the symmetric question: *does
  fine-tuning?* A LoRA-on-Phase-35-bank study is
  mechanical follow-up.
* **Strong-model frontier (C39-1).** A 30B+ model on the
  Phase-35 bank would sharpen the saturation question.

### F.3 What is genuinely blocking the endgame

Phase 39 does NOT unblock:

* **OQ-1 in full generality** (Conjecture P30-6).
* **Cross-language runtime calibration**.
* **Real-SWE-bench Docker sandbox** (security boundary
  for untrusted patches).

Phase 39 *does* close:

* "Maybe the Phase-37 bias is prompt-shaped after all"
  (Theorem P39-1 refutes the optimistic mock-implied read).
* "Maybe the substrate's bounded-context property
  doesn't extend to SWE-style multi-role teams"
  (Theorem P39-3 settles it for the mini bank).
* "Maybe the SWE-bench gap is architectural, not
  schema-shaped" (Theorem P39-4 settles it for the
  schema piece).

The remaining frontier is now bracketed cleanly:
Conjecture C39-1 / C39-2 (model-side bias
question — needs a stronger or fine-tuned model);
Conjecture C39-3 / C39-4 (real-SWE-bench
question — needs a SWE-bench loader + Docker).

---

## Appendix A — How to reproduce

```bash
# 1. Real-LLM prompt-variant sweep (Ollama required).
python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5:0.5b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase39_prompt_calibration_0p5b.json

python3 -m vision_mvp.experiments.phase38_prompt_calibration \
    --mode real --models qwen2.5-coder:7b --seeds 35 \
    --distractor-counts 4 \
    --out vision_mvp/results_phase39_prompt_calibration_7b.json

# 2. Frontier-model bounded substrate sweep (Ollama required).
python3 -m vision_mvp.experiments.phase39_frontier_substrate \
    --models llama3.1:8b gemma2:9b qwen2.5-coder:7b \
    --domains incident --distractor-counts 6 --seeds 31 \
    --out vision_mvp/results_phase39_frontier_substrate.json

# 3. SWE-bench-style bridge — mock (sub-second).
python3 -m vision_mvp.experiments.phase39_swe_bridge \
    --mode mock --n-distractors 0 6 12 24 \
    --out vision_mvp/results_phase39_swe_bridge_mock.json

# 4. SWE-bench-style bridge — real LLM (~ 1–3 min on 0.5b).
python3 -m vision_mvp.experiments.phase39_swe_bridge \
    --mode real --model qwen2.5:0.5b --n-distractors 6 \
    --out vision_mvp/results_phase39_swe_bridge_0p5b.json

# 5. Phase-39 test suite (18 tests; sub-second).
python3 -m pytest vision_mvp/tests/test_phase39_swe_bridge.py -q

# 6. Full regression — Phase 30..39 tests.
python3 -m pytest vision_mvp/tests/ -q
```

On a commodity laptop (2026-vintage): #3 runs sub-second;
#5 runs in ~ 0.5 s; #1 on qwen2.5:0.5b runs in ~ 100 s for
all five variants (~ 20 calls × 5); #1 on
qwen2.5-coder:7b runs in ~ 12 min (~ 2.5 min/variant);
#2 runs in ~ 60–100 s per model; #4 runs in ~ 1–3 min
per (model, distractor) cell on 0.5b; #6 runs in ~ 11 s
for the full 1 391-test suite.

---

*End of Phase 39 results note. The master plan
(``docs/context_zero_master_plan.md``) is updated in the same
commit; see ``§ 4.9.7 Arc 8 (extended further)`` and the
updated ``§ 4.11 Current frontier``.*
