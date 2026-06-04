# RUNBOOK — W135: oracle-derived solution-STRUCTURE witness + held-out NON-COMPLEXITY eval + conditional frontier rerun

**Status: LOCKED before any NIM.** This document pre-commits the entire W135 contract
(α/β/γ branch logic, the non-complexity split rule, the no-leakage / structure-witness rule,
the structure-witness slate, the self-test/regression-fixture rule, the same-budget evaluation
rule, the held-out earn rule, the frontier-target rule, the primary-source research rule, the
graphify deliverables, and the W136 branch logic). Constants here are frozen; any drift is a
refusal, not a tuning knob. CID values from the $0 Lane-α build are recorded in §3 **before any
β/eval/frontier NIM**.

`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · advanced work explicit-import only · `ultracode` OFF ·
`COO-9` stays lead.

---

## 0. One-line thesis

W133 proved the **exact-oracle EW1 counterexample witness adds +0.00 pp over blind reflexion** on
the WRONG_ALGORITHM / SEARCH_ENUM / HIDDEN_EDGE modes (0 rescues vs B0) = the **wrong-algorithm
capability ceiling**, while only the EW2 complexity witness helped. W134 then showed even the
complexity lever is single-family and sub-oracle. So the live blocker is **no longer complexity
detection — it is wrong-algorithm / solution-structure capability**. W135 asks the one honest
follow-up the whole W120→W134 arc was building toward: **can CoordPy derive SOLUTION STRUCTURE
from the exact oracle it owns — a greedy-failure certificate, an optimal-substructure / recurrence
witness, a search-frontier witness — and can that structure break the wrong-algorithm capability
ceiling that a bare counterexample (EW1/C1) could not, WITHOUT leaking the answer?**

W135 is **NOT** another official-benchmark supply hunt, **NOT** another selector retry, **NOT**
another complexity-only lap, **NOT** any bounded-context / compaction / summary trick (those remain
anti-patterns). It is exactly three things: (1) a solution-structure witness instrument + a
dedicated NON-complexity corpus; (2) a held-out non-complexity mechanism bench; (3) a targeted
frontier non-complexity rerun **only if (2) earns it**.

---

## 1. Branch logic (α → β → γ)

```
α  build structure-witness instrument + non-complexity corpus ($0)
   ├─ corpus hits ≥36/36/36 train/dev/eval + ≥30 frontier, all 5 self-tests pass
   │     └─ AND the $0 naive/ref separation + genuinely-new characterization is clean → go to β
   └─ corpus or self-tests FAIL the floor
         └─ land the instrument anyway; register the machine-checkable blocker; STOP at $0
β  held-out non-complexity mechanism bench (DEV first)
   ├─ DEV gate clears (§7a)  → run EVAL on the LOCKED eval slice
   │     ├─ EVAL earn rule clears (§7b) → γ frontier rerun EARNED
   │     └─ EVAL earn rule FAILS → register the structure-witness cap; $0 frontier
   └─ DEV gate FAILS          → register the structure-witness cap; $0 eval, $0 frontier
γ  primary-source research (always, done before lock) + stronger-model gate recheck (always) + graphify
   └─ frontier rerun runs IFF β earned it (locked frontier slice, target llama-3.3-70b)
```

No expensive NIM is spent on a branch whose gate did not clear. Frontier spend is **earned**,
never automatic.

---

## 2. Target-mode rule (LOCKED before generating anything)

**The W135 target modes are `WRONG_ALGORITHM_ADMISSIBLE` and `SEARCH_ENUM` ONLY** (the two
non-complexity, output-mismatch failure families on the W132 `RBC_SLATE_V1`). These are exactly the
modes where W133's EW1 counterexample witness was flat (+0.00 over B0) — the capability ceiling
W135 attacks with structure.

* **Core targets:** `MODE_WRONG_ALGORITHM` (8 families: greedy-vs-DP traps — coin-change, house-robber,
  weighted-interval, balanced-partition, LIS, knapsack, max-product, LCS) + `MODE_SEARCH_ENUM`
  (8 families: wrong-recurrence counting — subset-sum, fib-no-adjacent, stairs, tribonacci, Catalan,
  partitions, lattice-paths, change-making).
* **Excluded as main target — `COMPLEXITY_BLIND`:** kept ONLY as a negative-control / regression
  family (the structure witness must stay SILENT on a value-correct-but-slow program — no
  counterexample exists ⇒ no structure witness; complexity is the W134 D0 lever's job, not
  structure's). No complexity problem is admitted to the W135 corpus splits.
* **Excluded as main target — `HIDDEN_EDGE_STATE_MISS`:** a HIDDEN_EDGE witness collapses to a
  concrete corner-case counterexample = exactly the EW1/C1 channel W133 proved flat, with no
  distinct *structure* to render that is not already EW1. The operator rule admits it "only if the
  witness can be rendered cleanly without leaking" and adds *distinct* structure; it does not, so it
  is excluded (documented, not silently dropped). SW4-invariant is therefore NOT built as an arm.

---

## 3. Non-complexity split rule (LOCKED before results)

* **Source slate:** `RBC_SLATE_V1` filtered to `mode in {MODE_WRONG_ALGORITHM, MODE_SEARCH_ENUM}`
  → the 16 `wa_*` / `se_*` templates (all `DISC_OUTPUT_MISMATCH`: the naive passes every public
  sample and produces a WRONG ANSWER on ≥1 hidden case; each ships an INDEPENDENT exhaustive
  `brute_source` distinct from `naive_source`). No new template families are minted in W135 (the 16
  are already W132/W133-gate-validated); diversity comes from **16 distinct algorithm families ×
  multiple fresh seeds**.
* **Multi-seed-per-split:** each split is minted from a DISJOINT set of mint seeds; one
  `(template, seed)` pair → one instance, id `rbc_<name>__s<seed>`; content is addressed by
  `content_cid` (name+mode+statement+samples+secret_cases+ref_source). Seed-disjoint instances have
  different `content_cid` by construction (seeded public samples + hidden cases differ).
* **Seeds (LOCKED):**
  * train    = `(135011, 135012, 135013, 135014, 135015)`
  * dev      = `(135021, 135022, 135023, 135024, 135025)`
  * eval     = `(135031, 135032, 135033, 135034, 135035)`
  * frontier = `(135041, 135042, 135043, 135044, 135045)`
  All 20 seeds distinct ⇒ splits are seed-disjoint. (16 families × 5 seeds = up to 80 candidate
  instances/split; the ≥36 / ≥30 floors leave wide admission margin.)
* **Admission:** every instance must pass the W132 per-problem gates (passfail-only, ref-solvable,
  brute↔ref small-case agreement with `n_checked ≥ 1`, discriminating OUTPUT_MISMATCH, public/hidden
  split integrity) AND the W132 within-seed novelty filter. A non-admitting instance is dropped; the
  split keeps the admitted remainder.
* **Floors (Lane-α SUCCESS):** train ≥ 36, dev ≥ 36, eval ≥ 36, frontier ≥ 30 admitted.
* **Content addressing + locking:** `corpus_cid` (all four split CIDs), `eval_split_cid`
  (`= corpus.eval.split_cid`), and `frontier_slice_cid` are computed in the $0 Lane-α build and
  **LOCKED here before any eval/frontier spend**. The eval slice and the frontier slice are NEVER
  used in mechanism design. The bench script asserts the locked CID prefixes before eval / frontier
  NIM and refuses on drift.
  **LOCKED values (filled from the $0 build at `timeout_s=8.0`, predating all β NIM):**
  * `corpus_cid          = 306610aee0819ac9e40244e2de09538e85ae71ceba2b87909b4c81bdc567ca18`
  * `eval_split_cid      = 3f6e3e599dc0abf37ee422b3e709989bac178a434d6cfd189894622189d3b87d`
  * `frontier_slice_cid  = 8aa535644d934f3c894fd826f253099124f9b735e5f4d1bac479429864101f64`
  * admitted (≥ floors): train 80 / dev 79 / eval 80 / frontier 78 (slice 30); 16 families × 5 seeds;
    held-out integrity **TRUE** (content-CID + seed pairwise-disjoint, after cross-split dedup of the
    seed-independent `fib_no_adjacent` collisions — 1 dev + 2 frontier instances dropped, floors
    untouched). Both modes present in every split (WA 40 / SE 38–40).
* **Disjointness audit:** pairwise-disjoint per-instance `content_cid` across all four splits + seed
  disjointness ⇒ held-out integrity (the W133/W134 basis; residual per-secret-INPUT recurrence is
  only seed-independent canonical boundary/stress cases carrying no answer signal — the model never
  sees secret cases and the mechanism is pre-committed, never tuned on a split).

---

## 4. No-leakage / structure-witness rule (LOCKED before results)

The structure witness is **leakage-clean** iff it is reconstructible from ONLY: (a) the
**candidate's own program**, (b) the **public problem statement + public samples**, and (c)
**executions of the owned oracle (`ref_source` / `brute_source` / `naive_source`) on FRESH
witness-seed probe inputs that are byte-disjoint from the graded `secret_cases`** — emitting only
oracle OUTPUTS and derived structural summaries, never the oracle PROGRAMS, never a hidden answer,
and never the recurrence/algorithm itself.

* **NEVER model-facing:** `ref_source`, `brute_source`, `naive_source` text; the graded
  `secret_cases`; any hidden-case input or output; the recurrence formula or the "clever"/augmented
  DP state (per Pu, OOPSLA 2011, the state design IS the human insight — disclosing it would hand
  over the algorithm). The witness builder takes `(candidate_code, problem, template)` and the
  `template`'s sources are EXECUTED (subprocess), never rendered into the prompt (enforced by a
  structural test: `to_prompt_block()` contains no `def `/`import `/`class ` solver source).
* **All revealed inputs are disjoint:** the counterexample input `X` AND every sub-instance used in
  a sub-value ladder are asserted NOT byte-equal to any graded secret-case input (`leakage_clean`).
* **Grading is the anti-leakage guarantee:** the model is graded by `grade_on_secret_v1` on the
  DISJOINT hidden bank (pass iff ALL secret cases pass). A model that merely memorises the shown
  counterexample value or the shown small-N ladder CANNOT pass the full bank — it must write a
  GENERAL algorithm. The witness therefore tests GENERALISATION, not memorisation (the W133
  discipline). Sub-value ladders are capped to SMALL sub-instances of `X`; the hidden bank carries
  larger/disjoint cases the ladder cannot pin.
* **Genuinely-different guard:** `structure_witness_is_genuinely_new_v1` machine-checks that the
  witness carries information the blind B0 reject bit AND the EW1 counterexample structurally lack:
  either (i) a sub-value ladder with ≥ 2 disjoint sub-instances + their oracle-optimal values, OR
  (ii) a naive-vs-exact ATTRIBUTION contrast (the obvious/greedy value `V_greedy` AND the gap to the
  optimum `V*`, or the wrong-count vs exact-count). A witness that collapses to EW1's bare
  `(X, expected, observed)` triple is NOT genuinely-new (it is C1 re-stated = fake-different
  decoration) and does not count. **If a witness collapses into answer leakage (a revealed input
  collides with a graded case, or the block would carry solver source / the recurrence), it is
  killed.**

**Research-grounded witness FORM (primary sources, γ, applied here — these CHANGE the mechanism):**
1. **Minimal by input-token count** (PGS, arXiv:2506.18315): every counterexample is shrunk to the
   token-minimal still-failing input (reusing `_shrink_counterexample`). Raw I/O counterexamples
   cause repair failure via cognitive load — the most parsimonious explanation of W133's C1 +0.00.
2. **ATTRIBUTION, not just the failing point** (loop-invariant repair, TOPLAS 2025; Dolcetti,
   arXiv:2412.14841): the delta over the bare counterexample. SW1 states the greedy-vs-optimal
   decision divergence + the objective gap; SW3 states which count the recurrence over/under-counts.
3. **Property/invariant phrasing over I/O phrasing** (PGS): the block is phrased "your solution
   violates [no-local-greedy / optimal-substructure / correct-counting] here," with the instance as
   evidence, not as a naked (input, expected) row.
4. **Shift-left edge-case enumeration** (SolidCoder, arXiv:2604.19825 — the largest oracle-free
   repair lever): the rewrite prompt first asks the model to enumerate worst-case / adversarial
   inputs, then rewrite. Oracle-free and complementary; applied in every S-arm rewrite.
5. **SW2 is EXPLORATORY + leak-constrained** (Pu, OOPSLA 2011; KNARsack, arXiv:2509.15239): inducing
   a recurrence from a sub-value table is itself a synthesis problem whose demonstrated solutions
   need a constraint solver (we lack) or training (we lack). SW2 tabulates optimal sub-values over
   the OBVIOUS parameterisation only (never the augmented state), with no recurrence formula, and is
   pre-registered as the riskiest arm. SW1 (greedy-failure certificate) and SW3 (search-frontier
   exact-count) are the load-bearing bets (best literature support; safest on leakage).

---

## 5. Structure-witness slate (LOCKED) — `coordpy.solution_structure_witness_v1`

All constants frozen. Witness GENERATION is $0 NIM (oracle + executor subprocess only; never a model
call). The witness is anchored on a token-MINIMAL fresh discriminating counterexample `X`
(candidate(X) ≠ ref(X), shrunk); if the candidate is value-correct on every small probe (no
counterexample), the witness is `NONE` (structure has nothing to teach — correctly silent).

* **SW1 — greedy-failure certificate** (PRIMARY; leakage-safest). On `X`: the optimum `V* = ref(X)`,
  the obvious/greedy value `V_greedy = naive(X)` (the template's canonical admissible-wrong rule),
  the objective gap `Δ = |V* − V_greedy|`, and an ATTRIBUTION line: "a locally-greedy / obvious
  approach yields V_greedy here; the optimum is V*; the local choice is dominated — a global method
  is required." No solver source; no recurrence. Emits the disjoint `X`, `V*`, `V_greedy`, `Δ`.
* **SW2 — optimal-substructure / recurrence witness** (EXPLORATORY; leak-constrained). On `X`: a
  compact LADDER of optimal sub-values over OBVIOUS sub-instances of `X` (array problems → prefixes;
  single-integer-N problems → the exact-value sequence over `1..N`), each sub-instance disjoint from
  the secret bank, each value computed by the owned oracle. Plus a property hint: "the optimum on
  each sub-instance reuses optima of smaller sub-instances (optimal substructure); a greedy local
  extension is not optimal — recover the recurrence." NEVER states the recurrence/state.
* **SW3 — search-frontier witness** (SEARCH_ENUM). On `X`: the exact count `V* = ref(X)`, the naive's
  wrong count `naive(X)`, an ATTRIBUTION line ("your recurrence over/under-counts by `Δ` — it counts
  [ordered-vs-unordered / with-reuse / wrong-state]"), and the exact-count SEQUENCE over the small
  disjoint sub-instance ladder (the search-frontier structure that exposes the wrong recurrence).
* **SW4 — structure-to-rewrite controller** (LEAD). Consumes the richest applicable structure (SW1
  attribution ⊕ SW2 ladder ⊕ SW3 count-contrast, routed from OBSERVED behaviour — counting contrast
  if `naive(X)` is a different scalar than `V*` on a count-style answer; greedy gap + ladder
  otherwise) and produces a FRESH complete rewrite (NOT a rerank/selection), with the shift-left
  edge-case step prepended. Falls back to the EW1 counterexample if no structure is extractable, so
  it is never WORSE than C1.
* **SW5 — constrained structure-action policy.** Uses `constrained_policy_optimisation_v1` ONLY if
  Lane α produces enough KEEP/REWRITE/ABSTAIN decision data to meet a floor
  (`SW5_MIN_DECISIONS = 200` labelled decisions). **Default: NOT expected to be met from one dev
  seed ⇒ SW5 is documented but NOT forced into an arm** (recorded, not papered over).

---

## 6. Same-budget evaluation rule (LOCKED)

All arms: `K = 5`, same model, `sampling_temperature = 0.7`, `max_tokens_per_call = 1536`,
`executor_timeout_s = 8.0`. Per-problem model-call counts:

* A0 = 1 (single shot, temp 0.7, scored pass@1 on attempt-0)
* A1 = K (i.i.d. samples; oracle pass@K)
* B0 = K (blind W120/W132 reflexion: judge-reject bit + stderr + public-sample results)
* C1 = K (exact-oracle EW1 counterexample = W133 `ARM_C1_COUNTEREXAMPLE`; the FLAT BASELINE the earn
  rule must beat by ≥ +5 pp)
* D0 = K (W134 deployable complexity witness, controller `ARM_D3_CONTROLLER`; NEGATIVE control —
  expected to ≈ B0 on non-complexity, where the complexity witness almost never fires)
* S1 = K (greedy-failure certificate rewrite)
* S2 = K (optimal-substructure ladder rewrite)
* S3 = K (search-frontier witness rewrite)
* S4 = K (structure controller — the LEAD candidate)

Every S/C/D arm is a **strict same-budget swap** of the B0 feedback object (attempt-0 = the standard
initial prompt; K attempts; one model call per attempt; no early stop, no selective retry). Each arm
is scored in the "B" slot so `arm − A1` is byte-identical to `B − A1` (the verbatim W108
`_evaluate_phase2_gates` / `_mlb_rates` that scored W89/W105/W120/W132/W133/W134). Witness
generation (oracle execution on the disjoint probe) is $0 NIM and is paid OUTSIDE the K budget —
exactly as EW1's oracle execution was in W133. The W129 NIM-free selector discipline is held FIXED
(the S-arms are reflexion arms, not selection arms — the selector is not touched).

---

## 7. Gates

### 7a. DEV gate (go/no-go for EVAL spend) — pre-committed

On the DEV non-complexity slice the LEAD structure arm must satisfy ALL of:

1. `(lead − B0) ≥ +3.33 pp`, AND
2. `(lead − C1) ≥ +3.33 pp` (structure beats a bare counterexample — the whole W135 thesis), AND
3. lead rescues-vs-C1 span ≥ **2 distinct modes** (WA ∧ SE) OR ≥ **3 distinct template families**.

**Lead selection (pre-committed):** lead = argmax over {S4, S2, S3, S1} of `(arm − C1)` among arms
whose dev rescues-vs-C1 span ≥ 2 modes or ≥ 3 families; ties → arm order S4 > S2 > S3 > S1. If the
DEV gate FAILS ⇒ register `W135-L-SOLUTION-STRUCTURE-WITNESS-DEV-CAP`, **$0 eval, $0 frontier.**

### 7b. EVAL earn rule (frontier-rerun trigger) — pre-committed, operator-locked

On the LOCKED eval non-complexity slice the lead structure arm earns the frontier rerun iff ALL of:

1. `(lead − B0) ≥ +5.00 pp`, AND
2. `(lead − C1) ≥ +5.00 pp` (the structure lever beats the flat counterexample baseline by the
   retirement-relevant margin — a structure witness that only matches C1 is NOT an earn), AND
3. lead rescues-vs-C1 span ≥ **2 distinct non-complexity modes** OR ≥ **3 distinct template
   families** (a single-family or single-template blip is NOT an earn), AND
4. a per-rescue audit classifies **every** counted rescue as STRUCTURAL/algorithmic (a
   formatting-only or parsing-only gain is NOT an earn; a complexity-only gain does NOT count).

If the EVAL earn rule FAILS ⇒ register the structure-witness cap honestly; **$0 frontier**; do NOT
hand-wave a close miss into a mechanism story.

### 7c. FRONTIER outcome (single seed; NEVER a retirement by itself)

On the LOCKED frontier 30-slice, target `meta/llama-3.3-70b-instruct`, exact-oracle grader,
pass-fail only: `STRUCTURE_PASS_MECHANISM_DRIVEN` iff `lead − A1 ≥ +5 pp` AND MLB-1 ≥ 33 % AND
MLB-2 ≥ 33 %. A single-seed pass ⇒ W136 multi-seed confirmation toward retirement-grade — it is NOT
a retirement. W89 (+5.56) + W105 (+7.00) remain the only two retirements unless and until a later
clean MULTI-SEED `PASS_MECHANISM_DRIVEN`.

---

## 8. Self-test + regression-fixture rule (LOCKED, all $0)

**Lane-α quality gates (must pass before β):**
1. **Witness reproducibility** — same `(code, problem, witness_seed)` ⇒ byte-identical structure
   block + identical found/NONE verdict.
2. **Deterministic shrink + bounded derivation trace** — same input ⇒ same token-minimal
   counterexample + same sub-value ladder (bytes).
3. **Naive/ref separation on admitted train problems** — on EVERY admitted train-split problem, the
   structure witness FIRES genuinely-new on `naive_source` (the wrong algorithm) and is `NONE` on
   `ref_source` (the correct reference). Faithfulness gate; its pass-rate vs the EW1 counterexample
   on the same programs is recorded.
4. **Witness genuinely-new vs EW1/B0** — on every admitted train problem where the witness fires, it
   carries a ≥ 2-rung sub-value ladder OR a naive-vs-exact attribution contrast beyond EW1's triple;
   the genuinely-new RATE is recorded (problems with no extractable structure are reported, not
   hidden).
5. **Deterministic split regeneration** — re-minting from the locked seeds reproduces the same
   `corpus_cid` / `eval_split_cid` / `frontier_slice_cid`.

**Regression fixtures (replayed before any fresh run):**
* the W133 zero-gain modes (WRONG_ALGORITHM / SEARCH_ENUM): the structure witness must fire
  GENUINELY-NEW on the `naive_source` of each (where EW1 was flat) — proving it carries strictly
  more than the flat counterexample channel (whether that converts to rescues is the β question);
* **negative control (W134 complexity):** on a `cb_*` COMPLEXITY_BLIND `naive_source` (value-correct
  but O(N²)), the structure witness must be `NONE` (no counterexample exists ⇒ no structure witness —
  structure is value/algorithm-specific, NOT a complexity nudge); symmetrically, the W134 deployable
  complexity witness (D0) must be SILENT on the WA/SE `naive_source` (no super-linear growth) — the
  clean structure-vs-complexity dissociation;
* **positive control:** the correct `ref_source` as candidate ⇒ `NONE` (no counterexample);
* the W132 capability-bound traps: the structure witness fires genuinely-new on each.

---

## 9. Bench slice rule (LOCKED) + spend discipline

* **DEV bench slice:** the WA + SE families × `DEV_BENCH_PER_FAMILY` instances, family-stratified,
  deterministic by `(family, seed)` order. Default `DEV_BENCH_PER_FAMILY = 1` (16 problems — spans
  both modes and all 16 families). Arms (full): A0/A1/B0/C1/D0/S1/S2/S3/S4. **Latency contingency:**
  if a $0 NIM-latency probe (one tiny call) shows median > `LATENCY_THROTTLE_S = 12 s` (the W133/W134
  throttle regime), the dev arm set drops to A0/A1/B0/C1/S2/S4 (the two cleanest structural arms +
  baselines; D0/S1/S3 dropped) so the dev bench stays bounded; the ≥2-mode/≥3-family earn condition
  is preserved either way. No other reduction is permitted.
  * **EXECUTION AMENDMENT (transparent, post-lock, $0-discipline):** the tiny 8-token latency probe
    measured 0.5 s and did NOT trip the throttle, but the REPRESENTATIVE 1536-token calls ran at
    ~20 s (base) / ~58 s (oracle-heavy witness arms) — well above the 12 s threshold the throttle was
    meant to catch (the probe under-measured real per-call latency). The full 9-arm slate projects to
    ~9 h, an unattended-reliability risk. Per the throttle's INTENT, the dev pass is therefore run as
    a tighter **DECISIVE LEAD subset** — `A0/A1/B0/C1/S4` (S4 renders the SW1⊕SW2⊕SW3 union, so if
    S4 does not beat C1 no single structure arm would) — and the `S1/S2/S3` ablations + the D0
    negative control are a **$0-staged follow-up run ONLY if S4 clears the §7a gate vs C1**. The $0
    Lane-α self-tests already establish the structure-vs-complexity dissociation D0 would re-confirm
    (structure witness silent on complexity naive; complexity witness silent on WA/SE), so D0 on dev
    is confirmatory, not decisive.
* **EVAL bench slice:** the LOCKED eval 30-slice (deterministic, family-stratified). Arms:
  A0/A1/B0/C1/lead. Runs ONLY if §7a clears.
* **FRONTIER slice:** the LOCKED frontier 30-slice. Arms: A0/A1/B0/C1/lead. Runs ONLY if §7b earns.
* Corpus construction + structure-witness self-tests come FIRST ($0). Held-out DEV NIM is allowed.
  EVAL NIM is gated on §7a. FRONTIER NIM is gated on §7b. Maverick cross-scale is an OPTIONAL
  separate push-button (W135 does NOT block on Maverick). No exposed-frontier-control spend by
  default. No new seed-chasing on old official benchmarks. No stronger-model frontier spend unless
  the primary-cutoff gate genuinely opens. A close edge, a contaminated witness, or a
  prompt-decoration effect is NOT a success.

---

## 10. Frontier-target rule (LOCKED)

* Default frontier target = `meta/llama-3.3-70b-instruct` (the W105 retirement model; primary-KNOWN
  cutoff Dec-2023 per Meta's `llama3_3/MODEL_CARD.md`; reachable ~7 s/call). The frontier run is on
  the LOCKED non-complexity frontier 30-slice — NOT the mixed W132 core slice and NOT the W134
  complexity-only slice.
* Maverick remains an OPTIONAL push-button cross-scale check on the same slice if its NIM deployment
  recovers (primary-KNOWN cutoff Aug-2024); W135 does NOT block on Maverick.
* A stronger-than-Maverick model is used ONLY if the §3 primary-cutoff gate
  (`stronger_model_cutoff_certification_v1`, decision CID `258b6ed7`) genuinely opens — re-checked in
  γ, **CONFIRMED STILL CLOSED `{KNOWN:1, UNKNOWN:4}`** this milestone (Qwen3-Coder-480B /
  DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5 all primary-UNDISCLOSED; the new entrant
  MiniMax-M3 [Jun-2026] has no published card/cutoff). No 405B unless reachability changes and a
  pre-committed gate clears.

---

## 11. Primary-source research rule (LOCKED, γ) — DONE before lock; it CHANGED the mechanism

Actual external research, primary sources ONLY (arXiv / OpenReview / official ACL/EMNLP/NAACL/COLM/
ICLR/ICML/NeurIPS). Findings that CHANGED the W135 mechanism (see §4 form-rules 1–5):

* **PGS (arXiv:2506.18315, ICML 2026)** — minimal-by-input-token + property-oriented feedback;
  raw I/O counterexamples cause repair failure via cognitive load ⇒ explains W133's C1 +0.00 and
  prescribes minimality + attribution + property phrasing.
* **Loop-invariant repair (TOPLAS 2025) + Dolcetti (arXiv:2412.14841)** — typed/localized
  ATTRIBUTION is the active ingredient ⇒ SW1/SW3 must attribute (decision divergence + gap; which
  count is over/under).
* **SolidCoder (arXiv:2604.19825)** — shift-left edge-case enumeration is the largest oracle-free
  repair lever ⇒ prepended to every S-arm rewrite.
* **Pu, "Synthesis of First-Order DP Algorithms" (OOPSLA 2011) + KNARsack (arXiv:2509.15239)** —
  recovering a recurrence from a sub-value table needs a constraint solver (we lack) or training
  (we lack); the DP-table signal is real but its demonstrated exploitation is training-time ⇒ SW2 is
  EXPLORATORY + leak-constrained (obvious parameterisation only, never the augmented state).
* **Closest neighbours that do NOT match** (novelty): LDB (arXiv:2402.16906) shows the model its OWN
  program's runtime values, not oracle sub-values; CEGIS / oracle-guided synthesis (ICSE 2010) uses
  distinguishing I/O pairs + an SMT learner; Reflexion (= our B0). No primary work feeds
  oracle-derived, attributed, minimal greedy-failure / state-count witnesses to a frozen LLM at
  inference ⇒ SW1/SW3 are genuinely new in FORM, with the literature endorsing the design and
  flagging SW2's risk.

No literature-summary-as-output; the findings are wired into the executable witness FORM (§4–§5) and
into the genuinely-new + leakage guards.

---

## 12. graphify deliverables (LOCKED)

* START: confirm the graph is built from current HEAD (done: `b023ee4`).
* `graphify explain` on the named modules; `graphify path` from
  `solution_structure_witness_v1` to `exact_oracle_witness_v1` and to the battlefield;
  `graphify affected solution_structure_witness_v1.py`; secondary `graphify query`.
* The new `solution_structure_witness_v1` must be a REAL bridge: a 1-hop `imports_from` edge to
  `exact_oracle_witness_v1` (reuse `WitnessProbeSetV1` / `build_witness_probe_set_v1` /
  `find_counterexample_witness_v1` / `_shrink_counterexample`) AND to
  `resistant_by_construction_battlefield_v1` (`_exec_capture_v1` / `_tok_count`) AND to the
  reflexion/bench path (`run_witness_arm_v1` scaffold / `IcpcArmOutcomeV1`), not a trivial-string
  node hop. END: `graphify update .` after material changes; record the commit the refreshed graph
  was built from.

---

## 13. W136 branch logic (pre-committed)

* **β DEV gate FAILS** ⇒ W136 = register `W135-L-SOLUTION-STRUCTURE-WITNESS-DEV-CAP` (the instrument
  + non-complexity corpus STAND as reusable assets); the wrong-algorithm ceiling is confirmed
  capability-bound even under oracle-derived structure feedback at 70B; remaining levers are a
  code-competent local model, a primary-KNOWN stronger model when the gate opens, the Maverick
  cross-scale push-button, or a genuinely different mechanism axis.
* **β EVAL earn FAILS (dev passed)** ⇒ W136 = register
  `W135-L-SOLUTION-STRUCTURE-WITNESS-EVAL-CAP`; the structure witness is real on dev but does not
  clear the held-out earn bar; accept the bounded structure ceiling.
* **γ FRONTIER single-seed PASS** ⇒ W136 = operator-greenlit MULTI-SEED confirmation toward
  W89/W105 retirement-grade on the locked frontier slice (NOT a retirement by itself).
* **γ FRONTIER FAIL** ⇒ W136 = register `W135-L-SOLUTION-STRUCTURE-FRONTIER-CAP`; do not hand-wave
  the miss.
* In every branch: W89 (+5.56) + W105 (+7.00) STAND unless a later clean MULTI-SEED
  `PASS_MECHANISM_DRIVEN`; `COO-9` stays lead; bounded-context / compaction remain anti-patterns.

---

## 14. Carry-forwards preserved (unless new evidence genuinely changes them)

W123 official-supply cap · W124 local-encoder cap · W125 re-routing cap · W126 deterministic-
synthesis cap · W127 scaffold cap · W128 selection cap · W129 generation cap · W130 sub-bar
generator result · W131 disclosure-bound stronger-model cap · W132 resistant-by-construction pilot
cap · W133 witness single-mode cap · W134 deployable-complexity dev cap. Stronger-model gate CLOSED,
decision CID `258b6ed7` invariant `{KNOWN:1, UNKNOWN:4}`. No reopening MBPP+ V2 / frozen cross-modal
lines / the closed Llama-3.1 rescue branch / APPS main-lane NIM. No dirty synthetic benchmark sold
as a frontier win; no official-task paraphrase sold as resistant-by-construction.
