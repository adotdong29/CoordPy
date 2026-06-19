# RUNBOOK — W136: machine-structured ALGORITHM-STATE TRACE instrument + real-trace learned-memory/controller check + held-out NON-COMPLEXITY bench + conditional frontier rerun

**Status: LOCKED before any NIM.** This document pre-commits the entire W136 contract (α/β/γ branch
logic, the algorithm-state split rule, the no-leakage / trace-state rule, the trace slate, the
self-test / regression-fixture rule, the same-budget evaluation rule, the held-out earn rule, the
frontier-target rule, the primary-source research rule, the graphify deliverables, and the W137 branch
logic). Constants here are frozen; any drift is a refusal, not a tuning knob. The CID values from the
$0 Lane-α build are recorded in §3 **before any β/eval/frontier NIM**.

`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · advanced work explicit-import only · `ultracode` OFF ·
`COO-9` stays lead.

---

## 0. One-line thesis

W133 proved the exact-oracle **EW1 counterexample witness adds +0.00 pp over blind reflexion** on the
WRONG_ALGORITHM / SEARCH_ENUM modes; W135 then proved the **PROSE solution-structure witness S4 TIES the
counterexample (S4−C1 = +0.00) AND blind reflexion (S4−B0 = +0.00)** on the held-out non-complexity dev
field — the wrong-algorithm ceiling is *structure-unliftable in prose at 70B*. W135's structure ladder
was a **flat list of optimal values** rendered in prose. W136 asks the one level deeper the W133→W135 arc
was building toward: **can CoordPy expose machine-structured algorithm STATE — the optimal AND the naive
trajectories through the sub-instance state space, the exact step where they diverge, and the
increment/branching deltas — and does representing solver state DIRECTLY (not describing it in prose)
unlock a mechanism class — possibly the dormant learned-memory/controller line — that prose witnesses
could not, WITHOUT leaking the answer?**

W136 is **NOT** another official-benchmark supply hunt, **NOT** another selector retry, **NOT** another
prose-witness rewrite lap, **NOT** any bounded-context / compaction / summary trick (those remain
anti-patterns). It is exactly four things: (1) a machine-structured algorithm-state trace instrument + a
dedicated fresh-seed corpus; (2) an honest real-trace learned-memory/controller applicability check; (3)
a held-out non-complexity trace-conditioned mechanism bench; (4) a targeted frontier non-complexity rerun
**only if (3) earns it**.

---

## 1. Branch logic (α → β → γ)

```
α  build machine-structured trace instrument + fresh-seed corpus ($0)
   ├─ corpus hits ≥36/36/36 train/dev/eval + ≥30 frontier, all 5 self-tests pass
   │     └─ AND the trace fires genuinely-new-vs-S4 + leakage-clean on the traps → go to β
   └─ corpus or self-tests FAIL the floor
         └─ land the instrument anyway; register the machine-checkable blocker; STOP at $0
β  real-trace learned-memory/controller check ($0) + held-out non-complexity trace bench (DEV first)
   ├─ learned-memory modules consume real traces without synthetic fakery? (honest check; kill if not)
   ├─ DEV gate clears (§7a)  → run EVAL on the LOCKED W136 eval slice (airtight, fresh baselines)
   │     ├─ EVAL earn rule clears (§7b) → γ frontier rerun EARNED
   │     └─ EVAL earn rule FAILS → register the trace-state cap; $0 frontier
   └─ DEV gate FAILS          → register the trace-state cap; $0 eval, $0 frontier
γ  primary-source research (always, done before lock) + stronger-model gate recheck (always) + graphify
   └─ frontier rerun runs IFF β earned it (locked frontier slice, target llama-3.3-70b)
```

No expensive NIM is spent on a branch whose gate did not clear. Frontier spend is **earned**, never
automatic.

---

## 2. Target-mode rule (LOCKED before generating anything)

**The W136 core target modes are `WRONG_ALGORITHM_ADMISSIBLE` and `SEARCH_ENUM`** (the two non-complexity
output-mismatch families on the W132 `RBC_SLATE_V1`) — exactly the modes where W133's EW1 counterexample
AND W135's S4 prose structure were flat (+0.00), the capability ceiling W136 attacks with machine state.

* **Core targets:** the 8 `wa_*` (greedy-vs-DP traps) + 8 `se_*` (wrong-recurrence counting) families.
* **`HIDDEN_EDGE_STATE_MISS`:** admitted ONLY if it yields a structured state trace genuinely distinct
  from a plain counterexample. It does not (a hidden-edge witness collapses to a concrete corner-case
  counterexample = the EW1/C1 channel, with no distinct state trajectory), so it is **excluded** as a
  W136 target (documented, not silently dropped) — same exclusion logic as W135.
* **`COMPLEXITY_BLIND`:** kept ONLY as the **negative control** (a value-correct-but-slow naive produces
  NO counterexample ⇒ the trace is correctly NONE; the W134 D0 lever's job, not the trace's). No
  complexity problem is admitted to the W136 corpus splits.

---

## 3. Algorithm-state split rule (LOCKED before results)

* **Source slate:** `RBC_SLATE_V1` filtered to `mode in {WRONG_ALGORITHM, SEARCH_ENUM}` → the 16
  `wa_*`/`se_*` templates (the SAME field W135 used, so the trace-vs-prose contrast is clean). No new
  template families are minted; diversity = **16 distinct algorithm families × FRESH W136 seeds**.
* **FRESH W136 seeds (LOCKED, all distinct AND disjoint from the W135 135_0xx seeds):**
  * train    = `(136011, 136012, 136013, 136014, 136015)`
  * dev      = `(136021, 136022, 136023, 136024, 136025)`
  * eval     = `(136031, 136032, 136033, 136034, 136035)`
  * frontier = `(136041, 136042, 136043, 136044, 136045)`
* **Content addressing + locking:** `corpus_cid`, `eval_split_cid` (`= corpus.eval.split_cid`), and
  `frontier_slice_cid` are computed in the $0 Lane-α build and **LOCKED here before any eval/frontier
  spend**. The eval slice and the frontier slice are NEVER used in mechanism design. The bench asserts
  the locked CID prefixes before eval / frontier NIM and refuses on drift.
  **LOCKED values (filled from the $0 build at `timeout_s=8.0`, predating all β NIM):**
  * `corpus_cid          = ce1a6bc6541250ee98dd97be631c02da957734844c06c53c67412ad68b31a68a`
  * `eval_split_cid      = 135193533d6b57335c421514b96929bfeebbfab505dd3f55f273c98e1e3314c7`
  * `frontier_slice_cid  = 3f75b3020271ada89c99a7b7b47c210fc9cc200a0920c37e5b6ff61c92f8f756`
  * admitted (≥ floors): train **80** / dev **79** / eval **79** / frontier **78** (slice 30 = 14 SE +
    16 WA, both modes); held-out integrity **TRUE** (content-CID + seed pairwise-disjoint).
* **Floors (Lane-α SUCCESS):** train ≥ 36, dev ≥ 36, eval ≥ 36, frontier ≥ 30 admitted (the W132-gate
  admitted counts). If a floor is missed, land the instrument anyway and register the machine-checkable
  blocker (§1).
* **DEV bench reuse (spend discipline):** the W136 **dev bench reuses the EXACT W135 dev problems** (the
  16 from `select_dev_bench_slice_v1(W135_dev, per_family=1)`, seeds 135_02x) and their already-paid
  A0/A1/B0/C1/S4 baselines (all 81.25 %, the SAME 3 capability-bound traps `se_lattice_paths_blocked` /
  `wa_knapsack_01` / `wa_weighted_interval_scheduling`). Rationale: (a) the field is fully characterised
  there (W135), (b) it gives a byte-identical "machine-structured trace (T1) vs prose structure (S4)"
  contrast on the same problems, (c) it spends NIM ONLY on the genuinely-new trace arms. The fresh,
  locked W136 eval/frontier slices (136_03x / 136_04x) are where the earn is **re-validated airtight**
  with all-fresh baselines — that is exactly where seed-disjointness gates the earn, so the dev reuse is
  honest and disclosed.

---

## 4. No-leakage / trace-state rule (LOCKED before results)

The trace is **leakage-clean** iff it is reconstructible from ONLY: (a) the **candidate's own program**,
(b) the **public problem statement + public samples**, and (c) **executions of the owned oracle
(`ref_source` / `naive_source`) on FRESH disjoint sub-instances of the counterexample, byte-disjoint from
the graded `secret_cases`** — emitting only oracle OUTPUTS and derived state-transition summaries, never
the oracle PROGRAMS, never a hidden answer, never the recurrence/algorithm.

* **NEVER model-facing:** `ref_source` / `brute_source` / `naive_source` text; the graded `secret_cases`;
  any hidden-case input or output; the recurrence formula or the augmented DP state. The builder EXECUTES
  the sources (subprocess), never renders them (a structural test asserts `to_capsule_block()` contains
  no `def `/`import `/`class ` solver source).
* **AT4 (invariant/internal-state) is EXCLUDED for leakage:** a clean loop-invariant / augmented-DP-state
  trace cannot be derived from oracle OUTPUTS; it requires INSTRUMENTING the reference solver to emit its
  internal state, which renders the recurrence itself (the human insight that, per Pu OOPSLA 2011, IS the
  algorithm). That is answer-adjacent leakage (it hands over the algorithm, not the structure), so AT4 is
  **documented-and-excluded**. The W136 trace is **output-derived only** (the dual optimal/naive
  trajectory + divergence + deltas), never an instrumented internal-state dump.
* **All revealed inputs are disjoint:** the counterexample `X` AND every sub-instance row are asserted
  byte-disjoint from the graded `secret_cases` (`leakage_clean`).
* **Bounded by construction:** ≤ `TRACE_MAX_ROWS = 6` rows on SMALL sub-instances — minimal per the
  primary-source finding that LONGER execution traces HURT repair (arXiv:2505.04441); a fat trace is the
  documented falsification risk, not a feature.
* **Grading is the anti-leakage guarantee:** the model is graded by `grade_on_secret_v1` on the DISJOINT
  hidden bank (pass iff ALL secret cases pass). Memorising the shown table CANNOT pass — the trace tests
  GENERALISATION, not memorisation (the W133/W135 discipline).
* **Rendering emphasis is mode-routed (oracle-side, never shown):** a counting problem gets count-talk
  (AT3) and a greedy-vs-DP optimisation gets divergence-talk (AT1/AT2); the `mode` is a property of the
  owned battlefield, never rendered to the model — it only makes the prose phrasing correct (the
  count-vs-scalar-gap ambiguity is otherwise indistinguishable from outputs alone). The trace CONTENT is
  purely behaviour-derived.
* **Genuinely-new guard (`trace_is_genuinely_new_vs_structure_v1`):** a trace is genuinely-new-vs-S4 iff
  it carries the DUAL trajectory (≥1 row with BOTH optimal AND naive values + a `diverges` flag) AND a
  transition signal (a marked first-divergence step OR an increment/delta trajectory) — the structure
  S4's flat optimal-only ladder lacked. A trace that reduces to S4's ladder is NOT genuinely-new (it is
  "a prose witness in JSON clothing") and does not count. **If a trace collapses into answer leakage, it
  is killed.**

---

## 5. Trace slate (LOCKED) — `coordpy.algorithm_state_trace_v1`

All constants frozen. Trace GENERATION is $0 NIM (oracle + executor subprocess only; never a model call).
Anchored on a token-minimal fresh discriminating counterexample `X` (reusing W133 EW1 + the W135 typed
sub-instance generator, extended for structured shapes); if the candidate is value-correct on every small
probe (no counterexample), the trace is `NONE` (correctly silent — the complexity negative control).

* **AT1 — decision-path trace** (WRONG_ALGORITHM greedy-vs-opt): per-sub-instance optimal V(i) AND naive
  G(i), with the marked first-divergence step i* = min{i : V(i)≠G(i)} (the machine-readable point where
  the naive's decision path departs from the optimum).
* **AT2 — subproblem-state trace** (DP optimal-substructure): the optimal-value state table with the
  per-step increment trajectory `optimal_delta(i) = V(i)−V(i−1)` (the transition structure the flat
  ladder collapsed).
* **AT3 — search-frontier trace** (SEARCH_ENUM counting): the exact-count C(i) and the naive count Cn(i)
  per disjoint sub-instance with the branching delta (the wrong recurrence exposed as a SEQUENCE of
  (correct, wrong) counts).
* **AT5 — typed trace capsule:** the `AlgorithmStateTraceV1` object itself — bounded, content-addressed,
  reproducible from the sanctioned oracle-execution API. The unifying object T1/T2 consume.
* **AT4 — invariant-state trace: EXCLUDED for leakage** (§4).
* **Family-aware sub-instances (`_typed_subinstances_v1`):** the W135 generic 1D slicer produced MALFORMED
  sub-instances for the structured-input traps (knapsack `N W`+pairs, weighted-interval `N`+triples,
  lattice `R C`+grid) — exactly the 3 capability-bound traps. W136 adds typed sub-instances: stride
  detection (count = header int s.t. body%count==0) for tuple/array shapes + top-left `r'×c'` sub-grids
  for 2D grids, all VALIDATED by executing the oracle (a malformed parse yields rows the oracle rejects,
  so it is safe), falling back to the generic ladder for unknown shapes. This is what gives the trace a
  FAIR shot on the exact traps (verified at $0: all 3 traps now yield genuinely-new, leakage-clean,
  multi-row dual-trajectory traces).

The trace arm (`run_trace_arm_v1`) is a strict **same-budget** swap of the W120 reflexion arm (identical
K / model / temperature / attempt-0 prompt; the ONLY change is the between-attempt feedback object),
scored in the "B" slot so `T − A1 ≡ B − A1` (the verbatim W108 evaluator).

---

## 5b. Mechanism slate (LOCKED) — the arms

* **A0** = 1 (single shot, scored pass@1 on attempt-0)
* **A1** = K (i.i.d. samples; oracle pass@K)
* **B0** = K (blind W120/W132 reflexion: judge-reject bit + stderr + public-sample results)
* **C1** = K (exact-oracle EW1 counterexample = W133 `ARM_C1_COUNTEREXAMPLE`)
* **S4** = K (W135 PROSE structure controller — the flat optimal-only ladder; the lever T must beat)
* **T1** = K (machine-structured algorithm-state TRACE rewrite — full dual-trajectory capsule; **LEAD**)
* **T2** = K (forward-only trace-conditioned CONTROLLER — routes capsule vs counterexample via
  `route_trace_action_v1`, weightless, bridging `ControllerAction` + `FailureDigestV1`; **staged** on a
  T1 crack — per the W129 generation-ceiling result a routing layer cannot exceed T1's generation
  ceiling, so if T1 does not beat S4, T2 cannot).
* **T3 — trace-conditioned learned-memory/controller adjunct: KILLED at $0 (recorded, not papered over).**
  `differentiable_memory_substrate_v1` / `composed_learned_memory_v1` / `live_composed_memory_training_v1`
  are random-until-trained nets benched only on synthetic `rng.standard_normal` recall data
  (`build_content_addressed_recall_dataset_v1` / `build_composed_long_horizon_dataset_v1`); the live one
  hard-requires GPU+transformers (`LiveTrainingBlockedOnHardwareError`); `constrained_policy_optimisation_v1`
  needs a learned MLP + a simulator reward. NONE can consume a real task trace without fabricating
  synthetic supervision or running a training loop we honestly do not have (corroborates W124's
  `TOO_SYNTHETIC_NOT_WARRANTED`). The ONLY honestly-usable controller — the forward-only, weightless
  `controller_native_code_mechanism_v1` digest-router (zero learned weights) — IS exercised, as the **T2**
  route. So the learned-memory line is re-opened HONESTLY and the verdict is recorded: the trace does not
  rescue the trainable nets from being synthetic-only at inference.
* **T4 — constrained action policy: NOT warranted (documented).** It would need a labelled KEEP/REWRITE/
  ABSTAIN decision floor (the W135 `SW5_MIN_DECISIONS = 200`) not met from one dev seed; not forced.

> **EXECUTION AMENDMENT (transparent, post-lock, operator-directed).** After T1 cracked 0/3 traps (the full
> 2-D DP table + recurrence scaffold included), the milestone did NOT stop at the apparent null. Per an
> explicit operator directive to root-cause the *true reason* and iterate, a diagnostic captured the model's
> ACTUAL code + the execution diff vs the oracle and found the apparent "wrong-algorithm ceiling" is an
> **I/O-FORMAT CONFOUND**: the model writes correct algorithms but misparses the W132 battlefield's
> whitespace-flattened input. This added (a) the `run_w136_trap_diagnostic_v1` root-cause harness, (b) the
> `$0` same-DP-two-parsings proof + the corpus-wide I/O-format characterisation, (c) a new execution-grounded
> arm **T_IO** (`run_io_grounded_trace_arm_v1`; weightless, no-leakage, fires only on the model's own public
> failure) that cracks 3/3 traps, and (d) the STANDARD-I/O A0 confirmation (one-shots 3/3 with no feedback).
> The earn discipline is UNCHANGED and binding: T_IO's gains are parsing-only ⇒ §7b condition 4 excludes
> them ⇒ **$0 frontier**. The locked α/β/γ gates, CIDs, and earn rule are untouched; this amendment only
> records the post-T1 root-cause iteration. See `docs/RESULTS_W136_ALGORITHM_STATE_TRACE_BENCH_V1.md`.

---

## 6. Same-budget evaluation rule (LOCKED)

All arms: `K = 5`, same model, `sampling_temperature = 0.7`, `max_tokens_per_call = 1536`,
`executor_timeout_s = 8.0`. Per-problem model-call counts: A0=1; A1=K; B0=K; C1=K; S4=K; T1=K; T2=K.
Every T/S/C arm is a **strict same-budget swap** of the B0 feedback object (attempt-0 = the standard
initial prompt; K attempts; one model call per attempt; no early stop, no selective retry). Each arm is
scored in the "B" slot so `arm − A1` is byte-identical to `B − A1` (the verbatim W108
`_evaluate_phase2_gates` / `_mlb_rates` that scored W89/W105/W120/W132/W133/W134/W135). Trace generation
(oracle execution on the disjoint sub-instances) is $0 NIM and is paid OUTSIDE the K budget — exactly as
EW1's oracle execution was in W133. The W129 NIM-free selector discipline is held FIXED (the T-arms are
reflexion arms, not selection arms).

**DEV spend (§3):** the dev bench REUSES the W135 A0/A1/B0/C1/S4 baselines (same problems) — NIM is spent
ONLY on the trace arms (T1 lead; T2 staged on a T1 crack). **EVAL/FRONTIER spend:** the full fresh
baseline stack (A0/A1/B0/C1/S4) + T1, airtight, run ONLY if the prior gate cleared. Latency contingency:
if a $0 NIM-latency probe shows the representative call regime is slow, the dev pass stays the decisive
lead subset (T1 only); T2 is the $0-staged follow-up on a T1 crack.

---

## 7. Gates

### 7a. DEV gate (go/no-go for EVAL spend) — pre-committed

On the DEV non-complexity bench the LEAD trace arm (T1) must satisfy ALL of:

1. `(T1 − B0) ≥ +3.33 pp`, AND
2. `(T1 − S4) ≥ +3.33 pp` (machine-structured state beats the prose structure witness — the W136
   thesis), AND
3. T1 rescues-vs-B0 span ≥ **2 distinct modes** (WA ∧ SE) OR ≥ **3 distinct template families**, AND
4. every counted rescue is STRUCTURAL (the trace fired genuinely-new on it — not a parse/format fix), AND
5. no regression turns a B0-solved problem into a T1 failure that nets the gain below the bar.

If the DEV gate FAILS ⇒ register `W136-L-ALGORITHM-STATE-TRACE-DEV-CAP`, **$0 eval, $0 frontier.**

### 7b. EVAL earn rule (frontier-rerun trigger) — pre-committed, operator-locked

On the LOCKED W136 eval non-complexity 30-slice (fresh baselines, airtight) the lead trace arm earns the
frontier rerun iff ALL of:

1. `(T1 − B0) ≥ +5.00 pp`, AND
2. `(T1 − S4) ≥ +5.00 pp` (machine-structured state beats the prose structure lever by the
   retirement-relevant margin), AND
3. T1 rescues span ≥ **2 distinct non-complexity modes** OR ≥ **3 distinct template families** (a
   single-family/single-template blip is NOT an earn), AND
4. a per-rescue audit classifies **every** counted rescue as STRUCTURAL/algorithmic (a formatting/parsing
   gain is NOT an earn; a complexity-only gain does NOT count).

If the EVAL earn rule FAILS ⇒ register the trace-state cap honestly; **$0 frontier**; do NOT hand-wave a
close miss into a mechanism story.

### 7c. FRONTIER outcome (single seed; NEVER a retirement by itself)

On the LOCKED W136 frontier 30-slice, target `meta/llama-3.3-70b-instruct`, exact-oracle grader,
pass-fail only: `TRACE_PASS_MECHANISM_DRIVEN` iff `T1 − A1 ≥ +5 pp` AND MLB-1 ≥ 33 % AND MLB-2 ≥ 33 %. A
single-seed pass ⇒ W137 multi-seed confirmation toward retirement-grade — it is NOT a retirement. W89
(+5.56) + W105 (+7.00) remain the only two retirements unless and until a later clean MULTI-SEED
`PASS_MECHANISM_DRIVEN`.

---

## 8. Self-test + regression-fixture rule (LOCKED, all $0)

**Lane-α quality gates (must pass before β):**
1. **Trace reproducibility** — same `(code, problem, witness_seed)` ⇒ byte-identical capsule + verdict.
2. **Deterministic typed sub-instances** — same input ⇒ same ladder (bytes).
3. **Naive/ref separation** — on EVERY admitted train problem the trace fires genuinely-new on
   `naive_source` and is `NONE` on `ref_source` (faithfulness); the trace-admissibility + the
   genuinely-new-vs-S4 rate are recorded (problems with no extractable state reported, not hidden).
4. **Genuinely-new vs S4** — the trace carries the dual trajectory + transition beyond S4's ladder.
5. **Deterministic split regeneration** — re-mint ⇒ same `corpus_cid`.

**Regression fixtures (replayed before any fresh run):**
* the 3 W135 capability-bound traps (`se_lattice_paths_blocked` / `wa_knapsack_01` /
  `wa_weighted_interval_scheduling`): the trace must fire GENUINELY-NEW (dual trajectory + divergence)
  where S4's prose ladder was flat/not-genuinely-new — proving the trace carries strictly more (whether
  that converts is the β question);
* **negative control (W134 complexity):** on a `COMPLEXITY_BLIND` `naive_source` (value-correct-but-slow),
  the trace must be `NONE` (no counterexample ⇒ no state to trace);
* **positive control:** the correct `ref_source` as candidate ⇒ `NONE`.

---

## 9. Bench slice rule (LOCKED) + spend discipline

* **DEV bench:** the W135 dev problems (16 = 16 families × 1 seed; spans WA ∧ SE). Arms: reused
  A0/A1/B0/C1/S4 + fresh T1 (lead); T2 staged on a T1 crack. ≤ ~160 NIM (T1 80 + T2 80).
* **EVAL bench:** the LOCKED W136 eval 30-slice. Arms: A0/A1/B0/C1/S4/T1 (all fresh). Runs ONLY if §7a clears.
* **FRONTIER slice:** the LOCKED W136 frontier 30-slice. Arms: A0/A1/B0/C1/S4/T1. Runs ONLY if §7b earns.
* Corpus + trace self-tests come FIRST ($0). Held-out DEV NIM is allowed. EVAL NIM is gated on §7a.
  FRONTIER NIM is gated on §7b. Maverick cross-scale is an OPTIONAL separate push-button (W136 does NOT
  block on Maverick). No exposed-frontier-control spend by default. No new seed-chasing on old official
  benchmarks. No stronger-model frontier spend unless the primary-cutoff gate genuinely opens. A close
  edge, a contaminated trace, or a prompt-decoration effect is NOT a success.

---

## 10. Frontier-target rule (LOCKED)

* Default frontier target = `meta/llama-3.3-70b-instruct` (the W105 retirement model; primary-KNOWN
  cutoff Dec-2023). The frontier run is on the LOCKED W136 non-complexity frontier 30-slice.
* Maverick remains an OPTIONAL push-button cross-scale check on the same slice if its NIM deployment
  recovers (primary-KNOWN cutoff Aug-2024); W136 does NOT block on Maverick.
* A stronger-than-Maverick model is used ONLY if the §3 primary-cutoff gate
  (`stronger_model_cutoff_certification_v1`, decision CID `258b6ed7`) genuinely opens — re-checked in γ,
  **CONFIRMED STILL CLOSED** this milestone. No 405B unless reachability changes and a pre-committed gate
  clears.

---

## 11. Primary-source research rule (LOCKED, γ) — DONE before lock; it CHANGED the mechanism

Actual external research, primary sources ONLY (arXiv / OpenReview / official ACL/EMNLP/NAACL/COLM/ICLR/
ICML/NeurIPS). Findings that CHANGED the W136 mechanism FORM (wired into the executable trace, §4–§5):

* **Self-Debugging (arXiv:2304.05128) / LDB (arXiv:2402.16906) / Scratchpad (arXiv:2112.00114)** feed the
  model its OWN program's runtime trace to localise its own bug ⇒ the ORACLE-derived trace of the CORRECT
  algorithm's STATE (the dual trajectory) is **genuinely new in FORM** relative to the closest primary
  work.
* **NAR / CLRS (arXiv:2205.15659) + TransNAR (arXiv:2406.09308)** model correct-algorithm execution state
  but only by TRAINING a GNN/LLM on it ⇒ NOT applicable to a frozen LLM at inference (justifies T3's
  kill).
* **NTM (arXiv:1410.5401) / DNC** are gradient-trained controller+memory architectures ⇒ unusable with a
  frozen LLM at inference (corroborates the T3 kill of the differentiable-memory line).
* **CEGIS / oracle-guided synthesis (arXiv:2502.07786)** is the only frozen-LLM oracle-coupled line but
  needs an SMT/MaxSAT solver and feeds a COUNTEREXAMPLE, not algorithm STATE.
* **Trace-repair (arXiv:2505.04441)** found LONGER/raw execution traces HURT repair ⇒ the trace MUST be
  bounded/minimal (`TRACE_MAX_ROWS = 6` on small sub-instances) — a fat DP table is the falsification
  risk, baked into the no-leakage rule (§4).

No literature-summary-as-output; the findings are wired into the executable trace FORM and the kills.

---

## 12. graphify deliverables (LOCKED)

* START: confirm the graph is built from current HEAD (done; refreshed at milestone start).
* `graphify explain` on the named modules; `graphify path` from `algorithm_state_trace_v1` to
  `exact_oracle_witness_v1` and `solution_structure_witness_v1`; secondary `graphify query`.
* The new `algorithm_state_trace_v1` must be a REAL bridge: 1-hop `imports_from` edges to
  `exact_oracle_witness_v1` (counterexample reuse), `solution_structure_witness_v1` (sub-instance +
  scalar-gap reuse), `resistant_by_construction_battlefield_v1` (exec), `icpc_reflexion_bench_v1`
  (bench), AND `controller_native_code_mechanism_v1` / `executor_grounded_patcher_v1` (the T2 controller
  bridge) — not a trivial-string node hop. END: `graphify update .` after material changes; record the
  commit the refreshed graph was built from.

---

## 13. W137 branch logic (pre-committed)

* **β DEV gate FAILS** ⇒ W137 = register `W136-L-ALGORITHM-STATE-TRACE-DEV-CAP` (the instrument + corpus
  STAND as reusable assets); the wrong-algorithm ceiling is confirmed capability-bound even under
  machine-structured oracle-derived state feedback at 70B; remaining levers are a code-competent local
  model, a primary-KNOWN stronger model when the gate opens, the Maverick cross-scale push-button, or a
  genuinely different mechanism axis.
* **β EVAL earn FAILS (dev passed)** ⇒ W137 = register `W136-L-ALGORITHM-STATE-TRACE-EVAL-CAP`; the trace
  is real on dev but does not clear the held-out earn bar; accept the bounded trace-state ceiling.
* **γ FRONTIER single-seed PASS** ⇒ W137 = operator-greenlit MULTI-SEED confirmation toward W89/W105
  retirement-grade on the locked frontier slice (NOT a retirement by itself).
* **γ FRONTIER FAIL** ⇒ W137 = register `W136-L-ALGORITHM-STATE-TRACE-FRONTIER-CAP`; do not hand-wave.
* In every branch: W89 (+5.56) + W105 (+7.00) STAND unless a later clean MULTI-SEED `PASS_MECHANISM_DRIVEN`;
  `COO-9` stays lead; bounded-context / compaction remain anti-patterns.

---

## 14. Carry-forwards preserved (unless new evidence genuinely changes them)

W123 official-supply cap · W124 local-encoder cap · W125 re-routing cap · W126 deterministic-synthesis
cap · W127 scaffold cap · W128 selection cap · W129 generation cap · W130 sub-bar generator result ·
W131 disclosure-bound stronger-model cap · W132 resistant-by-construction pilot cap · W133 witness
single-mode cap · W134 deployable-complexity dev cap · W135 structure-witness dev cap. Stronger-model gate
CLOSED, decision CID `258b6ed7` invariant. No reopening MBPP+ V2 / frozen cross-modal lines / the closed
Llama-3.1 rescue branch / APPS main-lane NIM. No dirty synthetic benchmark sold as a frontier win; no
official-task paraphrase sold as resistant-by-construction.
