# RUNBOOK — W134: deployable complexity witness + held-out complexity-only eval + conditional frontier rerun

**Status: LOCKED before any NIM.** This document pre-commits the entire W134 contract
(α/β/γ branch logic, the complexity-only split rule, the no-leakage / deployable-witness
rule, the deployable-witness slate, the self-test/regression-fixture rule, the same-budget
evaluation rule, the held-out earn rule, the frontier-target rule, the primary-source
research rule, the graphify deliverables, and the W135 branch logic). Constants here are
frozen; any drift is a refusal, not a tuning knob.

`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · advanced work explicit-import only · `ultracode` OFF ·
`COO-9` stays lead.

---

## 0. One-line thesis

W133 proved the **exact-oracle EW2 complexity witness is REAL and load-bearing** (C2/C3 beat
the blind W132 stack B0 by **+6.06 pp** and A1 by **+12.12 pp** on held-out DEV; MLB-2 60 %;
4 algorithmic COMPLEXITY rescues) but the gain was **single-mode** (COMPLEXITY_BLIND), so the
≥2-mode mixed-family frontier gate correctly FAILED. The EW2 gain, however, **consumed the
oracle's timing** (`big_ref_runtime_s`: "a correct reference finishes in 0.2 s"). W134 asks the
one honest follow-up: **can CoordPy distill that exact-oracle complexity witness into a
DEPLOYABLE public-signal witness that needs NO oracle answer and still preserves most of the
gain — and does it earn a targeted frontier rerun on a dedicated complexity-only field?**

W134 is **NOT** another official-benchmark supply hunt, **NOT** another selector retry, **NOT**
another broad witness-family lap, **NOT** any bounded-context / compaction / summary trick
(those remain anti-patterns). It is exactly three things: (1) a deployable complexity-witness
instrument + a dedicated complexity-only corpus; (2) a held-out complexity-only mechanism
bench; (3) a targeted frontier complexity rerun **only if (2) earns it**.

---

## 1. Branch logic (α → β → γ)

```
α  build deployable witness instrument + complexity-only corpus ($0)
   ├─ corpus hits ≥36/36/36 train/dev/eval + ≥30 frontier, all gates pass
   │     └─ AND the $0 naive/ref separation characterization is clean → go to β
   └─ corpus or self-tests FAIL the floor
         └─ land the instrument anyway; register the machine-checkable blocker; STOP at $0
β  held-out complexity-only mechanism bench (DEV first)
   ├─ DEV gate clears (§7a)  → run EVAL on the LOCKED eval slice
   │     ├─ EVAL earn rule clears (§7b) → γ frontier rerun EARNED
   │     └─ EVAL earn rule FAILS → register the deployable-complexity cap; $0 frontier
   └─ DEV gate FAILS          → register the deployable-complexity cap; $0 eval, $0 frontier
γ  primary-source research (always) + stronger-model gate recheck (always) + graphify
   └─ frontier rerun runs IFF β earned it (locked frontier slice, target llama-3.3-70b)
```

No expensive NIM is spent on a branch whose gate did not clear. Frontier spend is **earned**,
never automatic.

---

## 2. Target-mode rule (LOCKED before generating anything)

**The W134 target mode is `COMPLEXITY_BLIND` ONLY.** W133 localised the W132 cap: the complexity
sub-mode is FEEDBACK-fixable (EW2 lifted it); the value/algorithm sub-modes (EW1) are
CAPABILITY-bound (+0.00 over B0). W134 attacks ONLY the feedback-fixable sub-mode, on a field
that is entirely that sub-mode, so the W133 single-mode limitation is by-construction out of
scope (every eval problem IS the mode the witness addresses). No HIDDEN_EDGE / WRONG_ALGORITHM /
SEARCH_ENUM problem is admitted to the W134 corpus.

---

## 3. Complexity-only split rule (LOCKED before results)

* **Source slate:** `RBC_SLATE_V1` filtered to `mode == MODE_COMPLEXITY_BLIND` → the 9 `cb_*`
  templates (all `DISC_TIMEOUT`: the naive is value-correct but O(N²) and TLEs on the large
  hidden stress case; the reference is O(N log N)/O(N)). No new template families are minted in
  W134 (the 9 are already W132/W133-gate-validated); diversity comes from **9 distinct
  algorithm families × multiple fresh seeds**.
* **Multi-seed-per-split:** each split is minted from a DISJOINT set of mint seeds; one
  `(template, seed)` pair → one instance. Problem ids are namespaced `rbc_<name>__s<seed>` so
  same-family instances across seeds never collide. Content is addressed by `content_cid`
  (statement+samples+secret+ref); seed-disjoint instances have different `content_cid` by
  construction (seeded public samples + hidden cases differ).
* **Seeds (LOCKED):**
  * train  = `(134011, 134012, 134013, 134014, 134015)`
  * dev    = `(134021, 134022, 134023, 134024, 134025)`
  * eval   = `(134031, 134032, 134033, 134034, 134035)`
  * frontier = `(134041, 134042, 134043, 134044, 134045)`
  All 20 seeds distinct ⇒ splits are seed-disjoint. (9 families × 5 seeds = up to 45
  candidate instances/split; the ≥36 / ≥30 floors leave admission margin.)
* **Admission:** every instance must pass the W132 per-problem gates (ref-solvable,
  brute↔ref small-case agreement, discriminating TIMEOUT, public/hidden split integrity) AND
  the W132 novelty filter. A non-admitting instance is dropped; the split keeps the admitted
  remainder.
* **Floors (Lane-α SUCCESS):** train ≥ 36, dev ≥ 36, eval ≥ 36, frontier ≥ 30 admitted.
* **Content addressing + locking:** `corpus_cid` (all four split CIDs), `eval_split_cid`, and
  `frontier_slice_cid` are computed in the $0 Lane-α build and **LOCKED in this runbook before
  any eval/frontier spend**. The eval slice and the frontier slice are NEVER used in mechanism
  design. The bench script asserts the locked CID prefixes before eval / frontier NIM and
  refuses on drift. **LOCKED values (from the $0 build at `timeout_s=8.0`, predating all β NIM):**
  * `corpus_cid          = 191d995487d6cb09db6dba7683413661c69b1cefa82036a3fc339d5b0bb54a55`
  * `eval_split_cid      = 748dd6faa8a82b80bfaac79fe506c4a5ed5dc4e504b312dc7839305e03ccb284`
  * `frontier_slice_cid  = 31a813041e333dc8498ddc5121fef174011d084014bfb1168a87972394cf62be`
  * admitted (≥ floors): train 45 / dev 45 / eval 45 / frontier 45 (slice 30); 9 families × 5 seeds;
    held-out integrity TRUE (content-CID + seed pairwise-disjoint across all four splits).
* **Disjointness audit:** pairwise-disjoint per-instance `content_cid` across all four splits +
  seed disjointness ⇒ held-out integrity (the W133 `split_disjointness_report_v1` basis;
  residual per-secret-INPUT recurrence is only seed-independent canonical stress/boundary cases
  carrying no answer signal — the model never sees secret cases and the mechanism is
  pre-committed, never tuned on a split).

---

## 4. No-leakage / deployable-witness rule (LOCKED before results)

The deployable witness is **deployable** iff it is reconstructible from ONLY: (a) the **public
problem statement** (to parse the size constraint), (b) **public-format inputs** synthesised
from the public samples, and (c) the **observed runtime of the candidate's own program**.

* **NEVER model-facing / NEVER consumed by the deployable witness:** `ref_source`,
  `brute_source`, `naive_source`, the graded `secret_cases`, or any hidden-case bank. The
  deployable arm's witness builder takes ONLY `(candidate_code, statement, public_samples)` —
  it has no parameter through which a template/ref/naive/secret could enter (enforced by the
  function signature + a structural test).
* **No hidden outputs emitted.** The deployable witness carries NO `expected_output` (it cannot
  compute one without an oracle — that is the entire point). It emits ONLY: the parsed size
  constraint, the synthesised ladder sizes, the candidate's measured runtimes, a fitted growth
  exponent, an extrapolated time at `N_max`, and a target-complexity recommendation. Structurally
  leakage-clean (strictly weaker disclosure than EW2, which executed the reference).
* **Contrast with EW2 (the exact-oracle witness W133 used, retained here as the C0 UPPER
  BOUND):** EW2 executes `ref_source` to establish "fast is possible" (`big_ref_runtime_s`) and
  fires when the candidate is ≥8× slower than that reference. The deployable witness replaces
  `big_ref_runtime_s` with a **constraint-derived budget** (DW1) + the candidate's **own measured
  growth curve** (DW2). No reference is ever executed in any D-arm.
* **Genuinely-different guard:** `deployable_witness_is_genuinely_new_v1` machine-checks that the
  witness carries (i) ≥ 2 distinct measured `(size, runtime)` points, (ii) a derived growth
  exponent + a target-complexity class, and (iii) NO oracle output — information the blind B0
  reject bit structurally lacks AND that EW2 obtains only by running the oracle. A witness that
  collapses to "your code timed out" (no curve, no exponent) is NOT genuinely-new and does not
  count (that would be B0's bit re-stated = fake-different prompt decoration).

---

## 5. Deployable witness slate (LOCKED) — `coordpy.deployable_complexity_witness_v1`

All constants frozen. Witness GENERATION is $0 NIM (subprocess timing only; never a model call).

* **DW1 — constraint-derived budget witness.** `n_max = parse_max_constraint_v1(statement)`
  (reused from `public_signal_selection_oracle_v1`). Ops budget `OPS_BUDGET = 5e8` (reused
  `COMPLEXITY_OPS_BUDGET`; ~a few-second ICPC limit). Admissible exponent ceiling
  `p_adm = log(OPS_BUDGET)/log(n_max)` (at `n_max=1e5`, `p_adm≈1.74`, so O(N²) is inadmissible,
  O(N log N) admissible). Emits the budget fact, the size ladder, and the violation signal.
  Never emits a hidden output. If `n_max` is unparseable the budget verdict abstains (DW2's
  growth-ratio test can still fire).
* **DW2 — stress-growth witness.** A public-format **ladder generator** parses the public sample
  structure (line 1 = header whose FIRST token is the size N + optional params; line 2 = N
  integers) and synthesises a GEOMETRIC ladder `LADDER = [1000, 2000, 4000, 8000]` (base 1000,
  4 rungs, doubling) of spec-consistent inputs across a fixed shape set
  `SHAPES = ("random", "descending", "constant")` (random = typical; descending = strictly
  decreasing distinct, the monotonic-stack/short-circuit worst case; constant = all-equal). Value
  range parsed from the statement (`a_i <= X`), default `[1, 10^9]`. A `BASELINE_SIZE = 64`
  micro-input measures interpreter/parse overhead `t0`, subtracted from every rung so the fit is
  not flattened by startup. For each rung: `t_compute(m) = max_over_shapes(wall(m) - t0)`, with a
  per-run timeout `PER_RUN_TIMEOUT_S = 2.0` (a TLE rung is a super-linear lower bound).
  `NOISE_FLOOR_S = 0.005`. A log-log least-squares fit over the ≥ `MIN_LADDER_POINTS = 3` finite
  rungs above the noise floor gives the empirical exponent `p_emp`. **Inadmissible iff**
  (any rung TLEs) OR (`p_emp ≥ P_SUPERLINEAR = 1.7`) OR (extrapolated
  `t(n_max) = t_compute(top)·(n_max/top)^p_emp > WALL_BUDGET_S = 2.0`). The witness FIRES
  (`kind = COMPLEXITY`) only on an inadmissible verdict; else `NONE`.
* **DW3 — complexity rewrite prompt.** Feeds the model ONLY: the public statement, its own
  latest code, and the DW block (DW1 budget ⊕ DW2 measured curve + exponent + extrapolation +
  target class). Requires a FRESH complete rewrite (not a rerank/selection). Reuses the
  `build_gg2_rewrite_prompt_v1` / `build_worstcase_rewrite_prompt_v1` scaffolds from
  `stronger_generator_slate_v1` where useful.
* **DW4 — complexity-gated controller.** Combines the W130/`stronger_generator_slate_v1`
  complexity gate (`complexity_admissible_v1` on the measured `p_emp` against `n_max`) with the
  DW2 witness: routes **KEEP** (admissible measured growth ⇒ do not waste a rewrite; fall back to
  blind reflexion) / **REWRITE** (inadmissible ⇒ emit the DW block) / **ABSTAIN** (unmeasurable —
  e.g., candidate crashes on every ladder input ⇒ fall back to the blind W120 reflexion bit, so
  the arm is never WORSE than B0).
* **DW5 — constrained witness-action policy.** Uses `constrained_policy_optimisation_v1` ONLY if
  Lane α produces enough real KEEP/REWRITE/ABSTAIN decision data to meet a data floor
  (`DW5_MIN_DECISIONS = 200` labelled decisions spanning all three actions). **Default: the data
  floor is NOT expected to be met from a single dev seed ⇒ DW5 is documented but NOT forced.**
  If the floor is not met, DW5 is not built into an arm and that is recorded, not papered over.

---

## 6. Same-budget evaluation rule (LOCKED)

All arms: `K = 5`, same model, `sampling_temperature = 0.7`, `max_tokens_per_call = 1536`,
`executor_timeout_s = 8.0`. Per-problem model-call counts:

* A0 = 1 (single shot, the standard initial prompt at temp 0.7, scored pass@1 on attempt-0)
* A1 = K (i.i.d. samples; oracle pass@K)
* B0 = K (blind W120/W132 reflexion: judge-reject bit + stderr + public-sample results)
* C0 = K (exact-oracle EW2 arm = W133 `ARM_C2_COMPLEXITY`; the UPPER BOUND — uses the oracle)
* D1 = K (deployable rewrite, DW3)
* D2 = K (deployable + complexity gate, DW4 KEEP/REWRITE without ABSTAIN fallback distinction)
* D3 = K (deployable controller, full DW4 KEEP/REWRITE/ABSTAIN — the LEAD candidate)

Every D-arm and C0 is a **strict same-budget swap** of the B0 feedback object (attempt-0 =
the standard initial prompt; K attempts; one model call per attempt; no early stop, no selective
retry). Each arm is scored in the "B" slot so `arm − A1` is byte-identical to `B − A1` (the W108
`_evaluate_phase2_gates` / `_mlb_rates`, the verbatim code that scored W89/W105/W120/W132/W133).
Witness generation (the ladder timing) is $0 NIM and is paid OUTSIDE the K budget — exactly as
EW2's oracle execution was in W133 — so no model-facing step expands the budget. The W129 NIM-free
selector discipline is held FIXED unless a deployable arm makes it genuinely obsolete (it does
not: D-arms are reflexion arms, not selection arms).

---

## 7. Gates

### 7a. DEV gate (go/no-go for EVAL spend) — pre-committed

On the DEV complexity slice the LEAD deployable arm must satisfy ALL of:

1. `(lead − B0) ≥ +3.33 pp`, AND
2. lead rescues-vs-B0 span ≥ 2 distinct templates or seed families, AND
3. `(C0 − lead) ≤ +3.33 pp` (the deployable witness tracks the exact-oracle upper bound within a
   dev tolerance — it preserves most of the oracle gain, not a fraction of it).

**Lead selection (pre-committed):** lead = argmax over {D3, D1, D2} of `(arm − B0)` among arms
whose dev rescues-vs-B0 span ≥ 2 distinct templates/seed families; ties → min `(C0 − arm)`; then
arm order D3 > D1 > D2. If the DEV gate FAILS ⇒ register `W134-L-DEPLOYABLE-COMPLEXITY-*-CAP`,
**$0 eval, $0 frontier.**

### 7b. EVAL earn rule (frontier-rerun trigger) — pre-committed, operator-locked

On the LOCKED eval complexity slice the lead deployable arm earns the frontier rerun iff ALL of:

1. `(lead − B0) ≥ +5.00 pp` (beats the blind stack by the retirement-relevant margin), AND
2. `(C0 − lead) ≤ +2.00 pp` (stays within 2 pp of the exact-oracle UPPER BOUND), AND
3. lead rescues-vs-B0 span ≥ **3 distinct templates or seed families** (one lucky template is
   NOT an earn), AND
4. a per-rescue audit classifies **every** counted rescue as ALGORITHMIC/complexity (a
   formatting-only or parsing-only gain is NOT an earn).

If the EVAL earn rule FAILS ⇒ register the deployable-complexity cap honestly; **$0 frontier**;
do NOT hand-wave a close miss into a mechanism story.

### 7c. FRONTIER outcome (single seed; NEVER a retirement by itself)

On the LOCKED frontier 30-slice, target `meta/llama-3.3-70b-instruct`, exact-oracle grader,
pass-fail only: `DEPLOYABLE_PASS_MECHANISM_DRIVEN` iff `lead − A1 ≥ +5 pp` AND MLB-1 ≥ 33 % AND
MLB-2 ≥ 33 %. A single-seed pass ⇒ W135 multi-seed confirmation toward retirement-grade — it is
NOT a retirement. W89 (+5.56) + W105 (+7.00) remain the only two retirements unless and until a
later clean MULTI-SEED `PASS_MECHANISM_DRIVEN`.

---

## 8. Self-test + regression-fixture rule (LOCKED, all $0)

**Lane-α quality gates (must pass before β):**
1. **Witness reproducibility** — same `(code, statement, samples, ladder_seed)` ⇒ byte-identical
   DW block + identical fired/not-fired verdict (timing-robust: the verdict is from the growth
   SHAPE, asserted on the ratio/TLE structure, not a host-fragile absolute wall).
2. **Deterministic ladder generation** — same `(statement, samples, ladder_seed)` ⇒ identical
   ladder inputs (bytes).
3. **Public-spec-consistent stress** — every synthesised ladder input parses under the public
   format (line-1 first token = the requested size; line-2 has exactly that many integers in the
   parsed value range).
4. **Naive/ref separation on admitted complexity tasks** — on EVERY admitted train-split problem,
   the deployable witness FIRES inadmissible on `naive_source` (the O(N²) trap) and does NOT fire
   on `ref_source` (the O(N log N) reference). This is the deployability-faithfulness gate; its
   pass-rate vs the exact-oracle EW2 on the same programs is recorded.
5. **Deterministic split regeneration** — re-minting from the locked seeds reproduces the same
   `corpus_cid` / `eval_split_cid` / `frontier_slice_cid`.

**Regression fixtures (replayed before any fresh run):**
* the W132 B-unique complexity rescue family `cb_pairs_absdiff_le_d`,
* the W133 four complexity-rescue families `cb_distinct_in_windows`, `cb_pairs_sum_eq_t`,
  `cb_subarrays_sum_eq_k`, `cb_pairs_absdiff_le_d`,
* the W133 zero-gain counterexample modes (the deployable COMPLEXITY witness must NOT fire on a
  value-wrong-but-fast program — a negative control that it is complexity-specific, not a
  generic "rewrite" nudge).
The deployable witness must fire inadmissible on each rescued family's `naive_source` and stay
silent on each negative control; recorded as a machine-checkable fixture set.

---

## 9. Bench slice rule (LOCKED) + spend discipline

* **DEV bench slice:** all 9 complexity families × `DEV_BENCH_PER_FAMILY` instances, family-
  stratified, deterministic by `(family, seed)` order. Default `DEV_BENCH_PER_FAMILY = 2`
  (18 problems). A **pre-committed latency contingency:** if a $0 NIM-latency probe (one tiny
  call) shows median > 12 s/call (the W133 throttle regime), `DEV_BENCH_PER_FAMILY` drops to 1
  (9 problems) so the dev bench stays bounded; the ≥3-family earn condition is preserved either
  way. No other reduction is permitted.
* **EVAL bench slice:** the LOCKED eval 30-slice (deterministic, family-stratified across the 9
  families). Arms: A0/A1/B0/C0/lead. Runs ONLY if §7a clears.
* **FRONTIER slice:** the LOCKED frontier 30-slice. Arms: A0/A1/B0/lead (+ C0 only if cheap).
  Runs ONLY if §7b earns.
* Corpus construction + deployable-witness self-tests come FIRST ($0). Held-out DEV NIM is
  allowed. EVAL NIM is gated on §7a. FRONTIER NIM is gated on §7b. Maverick cross-scale is an
  OPTIONAL separate push-button (W134 does NOT block on Maverick). No exposed-frontier-control
  spend by default. No new seed-chasing on old official benchmarks. No stronger-model frontier
  spend unless the primary-cutoff gate genuinely opens. A close edge, a contaminated witness, or
  a prompt-decoration effect is NOT a success.

---

## 10. Frontier-target rule (LOCKED)

* Default frontier target = `meta/llama-3.3-70b-instruct` (the W105 retirement model; primary-
  KNOWN cutoff ~Dec-2023; reachable ~7 s/call). The frontier run is on the LOCKED complexity-only
  frontier 30-slice — NOT the mixed W132 core slice.
* Maverick remains an OPTIONAL push-button cross-scale check on the same slice if its NIM
  deployment recovers; W134 does NOT block on Maverick.
* A stronger-than-Maverick model is used ONLY if the §3 primary-cutoff gate
  (`stronger_model_cutoff_certification_v1`, decision CID `258b6ed7`) genuinely opens — re-checked
  in γ, expected to stay CLOSED `{KNOWN:1, UNKNOWN:4}`. No 405B unless reachability changes and a
  pre-committed gate clears.

---

## 11. Primary-source research rule (LOCKED, γ)

Actual external research, primary sources ONLY (arXiv / OpenReview / official ACL/EMNLP/NAACL/
COLM/ICLR/ICML/NeurIPS). Re-check + use primary work on: performance-bug / complexity-bug
diagnosis; verifier-guided code improvement; execution-feedback code generation; counterexample-
guided repair where directly relevant; empirical/measured time-complexity inference. The lane
answers: which complexity-witness mechanisms are EXECUTABLE-HERE (public statement + public-format
inputs + candidate self-timing, no oracle/training/infra), which require infra we do NOT honestly
have, which would LEAK hidden supervision and must be rejected. Use the literature ONLY if it
changes the mechanism — no literature-summary-as-output, no "inspired by paper X" without an
executable implementation.

---

## 12. graphify deliverables (LOCKED)

* START: confirm the graph is built from current HEAD (done: `cd1e3d40`).
* `graphify explain` on the eight named modules; `graphify path` from
  `exact_oracle_witness_v1` to the battlefield + reflexion bench; secondary `graphify query`.
* The new `deployable_complexity_witness_v1` must be a REAL bridge: a 1-hop `imports_from` edge
  to `public_signal_selection_oracle_v1` (parse_max_constraint / stress) AND to
  `stronger_generator_slate_v1` (complexity gate / rewrite prompt) AND to the witness/bench path,
  not a trivial-string node hop. END: `graphify update .` after material changes; record the
  commit the refreshed graph was built from.

---

## 13. W135 branch logic (pre-committed)

* **β DEV gate FAILS** ⇒ W135 = register `W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-DEV-CAP` (the
  instrument + corpus STAND as reusable assets); remaining levers are a genuinely different axis,
  the Maverick cross-scale push-button, or a primary-KNOWN stronger model when the gate opens.
* **β EVAL earn FAILS (dev passed)** ⇒ W135 = register
  `W134-L-DEPLOYABLE-COMPLEXITY-WITNESS-EVAL-CAP`; the deployable witness is real on dev but
  does not clear the held-out earn bar; accept the bounded complexity ceiling.
* **γ FRONTIER single-seed PASS** ⇒ W135 = operator-greenlit MULTI-SEED confirmation toward
  W89/W105 retirement-grade on the locked frontier slice (NOT a retirement by itself).
* **γ FRONTIER FAIL** ⇒ W135 = register `W134-L-DEPLOYABLE-COMPLEXITY-FRONTIER-CAP`; do not
  hand-wave the miss.
* In every branch: W89 (+5.56) + W105 (+7.00) STAND unless a later clean MULTI-SEED
  `PASS_MECHANISM_DRIVEN`; `COO-9` stays lead; bounded-context / compaction remain anti-patterns.

---

## 14. Carry-forwards preserved (unless new evidence genuinely changes them)

W123 official-supply cap · W124 local-encoder cap · W125 re-routing cap · W126 deterministic-
synthesis cap · W127 scaffold cap · W128 selection cap · W129 generation cap · W130 sub-bar
generator result · W131 disclosure-bound stronger-model cap · W132 resistant-by-construction
pilot cap · W133 witness single-mode cap. Stronger-model gate CLOSED, decision CID `258b6ed7`
invariant `{KNOWN:1, UNKNOWN:4}`. No reopening MBPP+ V2 / frozen cross-modal lines / the closed
Llama-3.1 rescue branch / APPS main-lane NIM. No dirty synthetic benchmark sold as a frontier
win; no official-task paraphrase sold as resistant-by-construction.
