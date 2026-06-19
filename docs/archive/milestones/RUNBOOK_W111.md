# RUNBOOK — W111 (different-MECHANISM tournament on contamination-resistant code + bounded-claim fallback only if deserved)

**Pre-commit contract. Locked BEFORE any expensive NIM call** (the only NIM
spend W111 contemplates is a *small, smallest-decisive* M3 cheap probe, and
only if M3 honestly earns it under § 4). `COO-9` REMAINS the lead path. No
version bump; no PyPI; `coordpy/__init__.py` untouched; advanced work
explicit-import only.

This milestone executes the W110 pre-commit (`docs/RUNBOOK_W110.md` § 8 FAIL
branch + `docs/RESULTS_W110_BIGCODEBENCH_PHASE2_70B_V1.md` § 6; `COO-34` close):
**the W110 BigCodeBench resistant FAIL answered "is the W108 LCB FAIL
LCB-specific or general?" → GENERAL.** So a multi-seed de-noise of either
resistant FAIL is NOT WARRANTED, and the live options are exactly: **(a) a
genuinely DIFFERENT mechanism that beats same-budget self-consistency (A1) on
contamination-resistant code, or (b) acceptance of the tightly-bounded
two-retirement contamination-EXPOSED-HumanEval-family claim as the honest code
ceiling.** W111 attacks (a) aggressively and falls back to (b) only if the
different-mechanism search dies honestly.

W111 is NOT a new benchmark-family tournament; NOT another reflexion rerun; NOT
APPS reopening; NOT a 405B/Llama-3.1 reopening; NOT a bounded-context /
compaction / summarization / token-compression drift (those remain
anti-patterns, not the frontier path). It is exactly THREE lanes: a
different-mechanism main lane (α), a benchmark/claim-discipline lane (β), and a
graphify/truth-tightening lane (γ).

**The scientific question (stated once, sharply):** the W89 *sequential
reflexion* mechanism fails on contamination-resistant code (0/2: LCB −3.33 pp,
BigCodeBench +0.00 pp). Is that a property of *reflexion specifically*, or of
*all same-budget multi-call mechanisms* at 70B on resistant code? **Is there a
genuinely different mechanism that beats A1 on contamination-resistant code at
70B?** If yes → that is the new frontier. If every serious candidate dies
cheaply → the bounded contamination-exposed claim is the honest ceiling.

---

## Linear

* `COO-9` (second/next code benchmark battlefield) — High, **lead path**. W111
  carries its "is the resistant FAIL mechanism-specific?" question to a
  different-mechanism test.
* `COO-35` (to be created at end-of-milestone) — the W111 milestone sub-issue
  under `COO-6`.
* Sync discipline: GitHub canonical; Linear synced at end-of-milestone via
  `scripts/sync_linear_github_v1.py` + `linear_github_mapping.json`.

---

## What is NOT in scope (anti-drift)

* No reopening MBPP+ V2 (`W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`).
* No reopening frozen cross-modal lines (RealWorldQA @ 11B / MathVista / ChartQA).
* No reopening the closed Llama-3.1 rescue branch
  (`W106-L-…-MARGIN-CAP-…-NOT-EARNED-CAP`).
* No 405B run (gate CLOSED at the 5th 404; W111 does not re-probe it).
* No LLM-as-judge anywhere; functional/`unittest`-based subset only.
* **No reopening APPS** (2021 contamination-EXPOSED; control evidence only;
  $0 further APPS NIM).
* **No more reflexion (B-mechanism) NIM** on either resistant FAIL — a
  multi-seed de-noise of a +0.00/−3.33 pp weak-MLB-2 point is NOT WARRANTED.
* **No mechanism that needs the hidden test source** — that is oracle leakage,
  not a fair same-budget mechanism. A candidate sees ONLY the visible
  docstring/spec + the executor's pass/fail + its 800-char `stderr_tail`
  (exactly the B-mechanism's information regime), never the `test` source.
* No claim that a contamination-RESISTANT PASS, if obtained at cheap-pilot
  scale, is a retirement until a Phase-3 multi-seed bench clears (W112+).

---

## Operational state (pre-W111 facts)

| Fact | Value |
|---|---|
| Confirmed retirements | EXACTLY TWO, both `meta/llama-3.3-70b-instruct` @ 70B: **W89** base HumanEval +5.56 pp; **W105** HumanEval+ +7.00 pp. Both contamination-EXPOSED HumanEval-family (2021). |
| Resistant column | **0/2** via the W89 reflexion mechanism: W108 LiveCodeBench 2025 B−A1=−3.33 pp (MLB-2 25%); W110 BigCodeBench 2024 B−A1=+0.00 pp (MLB-1 40% PASS, MLB-2 25% FAIL). The W108 FAIL is GENERAL (W110). |
| Exposed column | 3/3 PASS (W89 +5.56, W105 +7.00, W109 APPS +16.67 non-mech). |
| Contamination-confound | **STRENGTHENED toward a finding, NOT proven** (two single-seed resistant points; orthogonal difficulty not excluded). |
| 405B gate | CLOSED — HTTP 404 ×5. Not re-probed. |
| Operational reachability | `NVIDIA_API_KEY` present; BigCodeBench corpus cached + SHA-pinned (`ca4f352e…`); bcb_venv present; W110 slice CID `b69bf3a0…` reproducible. |
| Stable boundary | `coordpy.__version__ == 0.5.20`; `SDK_VERSION == coordpy.sdk.v3.43` |

---

## § 1 — α / β / γ branch logic (LOCKED)

* **Lane α — different-mechanism MAIN lane.** (1) Run a full NIM-free
  mechanism-mining pass (§ 2) over the EXISTING W110 (and, as needed, W108)
  resistant pilots to characterise WHY the resistant FAILs happen and WHICH
  weakness each candidate attacks. (2) Build the candidate slate (§ 3) with
  pre-results hypotheses. (3) Run the NIM-free cheap probes + adversarial
  ablations (§ 4); kill candidates whose attacked weakness is absent from the
  resistant-failure distribution. (4) If a candidate honestly earns a live
  probe (§ 4.3), run the *smallest-decisive* probe; if that is promising, run
  the fair 30-slice cheap pilot (§ 5) — within W111 if feasible, else
  pre-commit it to W112. (5) If all serious candidates die cheaply, register
  that honestly and trigger the bounded-claim fallback (§ 6).
* **Lane β — benchmark / claim-discipline lane.** Pre-commit (§ 6) the exact
  rule for when the bounded contamination-EXPOSED claim is accepted as the
  current code ceiling (a LAST resort, not the default). Pre-commit the
  cheap-pilot gates (§ 5) and the primary/secondary benchmark rule (§ 2.4)
  BEFORE any verdict. Spend NO NIM on APPS or the old reflexion mechanism.
* **Lane γ — graphify / truth-tightening lane.** Refresh graphify from HEAD at
  start + end; use `query`/`affected`/`explain`/`path` for file selection +
  claim surfaces; tighten the truth surfaces so the
  contamination-EXPOSED-specific-at-70B boundary is defensible after W111
  whatever happens (§ 7).

The only NIM spend is the Lane α M3 cheap probe, and ONLY if earned under § 4.3.

---

## § 2 — mechanism-mining + candidate-selection rule (LOCKED — NIM-free; applied BEFORE the slate)

**The rule.** A candidate mechanism is admitted to the slate only if it is (i)
a *genuinely different mechanism* from sequential reflexion (B) — not a prompt
variant — and (ii) operating within the SAME K=5 model-call budget and the SAME
information regime (docstring + executor pass/fail + 800-char `stderr_tail`; no
test source). A candidate is killed at $0 NIM if a NIM-free probe shows the
weakness it attacks accounts for **< 33 %** of the observed resistant-failure
distribution ("weakness-coverage floor"), OR if it is strictly dominated by a
slate-mate.

### § 2.1 — the mechanism-mining pass (executed; $0 NIM)

`scripts/mine_w111_resistant_failure_modes_v1.py` re-executes EVERY A1+B
candidate from the W110 BigCodeBench pilot (300 executions) through the real
`unittest` executor (headless Agg, bcb_venv) and classifies each failing
candidate's executor signal. **Result (`results/w111/mechanism_mining/
w110_bcb_failure_census.json`):**

| Failure class | Count | % of 114 failures |
|---|---|---|
| **SEMANTIC_LOGIC** (AssertionError / wrong output) | 93 | **81.6 %** |
| TIMEOUT | 9 | 7.9 % |
| ENV_HARNESS (FileNotFoundError, mock files) | 7 | 6.1 % |
| OTHER | 3 | 2.6 % |
| **API_GROUNDING** (Import/Attribute/Name/sig-Type) | **2** | **1.8 %** |

Both API_GROUNDING failures are on a SINGLE problem (BigCodeBench/51) that
reflexion already rescued. **The resistant failure is overwhelmingly SEMANTIC
(the model writes correctly-importing, plausible code that produces the wrong
output), NOT API/library-grounding.**

A hard-core ablation (the 8 problems where BOTH A1 and B fail — where a
different mechanism MUST win) refines this:

| hard-core regime | count | fair-regime reachable? |
|---|---|---|
| **MOCK_COUPLING** (traceback through `unittest.mock`; fix needs the hidden test's mock setup — NOT in `stderr_tail`) | **6/8** | NO (would need test source = oracle leak) |
| **OUTPUT_VALUE** (the expected value IS printed in `stderr_tail`; targetable) | **2/8** (BigCodeBench/15, /20) | YES, in principle |

**Finding (`W111-T-RESISTANT-FAILURE-IS-SEMANTIC-HIDDEN-TEST-COUPLING`):** the
contamination-resistant difficulty is dominated by hidden-`unittest`-test
coupling + spec under-specification — the hidden suite mocks specific library
functions and asserts exact output formats/values not determined by the visible
docstring. This is largely information-unavailable to any *fair* same-budget
mechanism at 70B.

### § 2.2 — candidate-selection verdict (driven by § 2.1)

* **M2 (tool-augmented local symbol/doc introspection)** attacks API_GROUNDING
  = **1.8 %** of resistant failures (and the only 2 are on an already-rescued
  problem). Local stdlib introspection CANNOT reveal hidden-test conventions.
  **KILLED at $0 NIM** (weakness-coverage 1.8 % ≪ 33 % floor).
* **M1 (library/spec-grounded planner→coder)** attacks spec-comprehension. The
  failures are NOT comprehension failures (the model already produces plausible
  logic) — they are hidden-test-convention failures unreachable by planning
  over the *visible* spec; and M1 spends 1 of its 5 calls on a plan with NO
  executor grounding, reducing self-consistency sampling from 5 to 4. It is
  strictly dominated by M3 (same SEMANTIC target, but executor-grounded).
  **KILLED at $0 NIM** (real mechanism reach ≈ 0 on the hard core; dominated).
  *(The raw census coverage metric lumps M1 with all SEMANTIC = 83.3 %, but
  that is an upper bound on a weakness M1's planning mechanism does not actually
  reach; the hard-core ablation is the binding refinement.)*
* **M3 (executor-grounded structured-failure patcher)** attacks SEMANTIC_LOGIC
  = **81.6 %** — the dominant class — and is the ONLY candidate that uses the
  executor failure signal (the information that actually distinguishes a
  resistant failure). It is a genuinely different mechanism from reflexion
  (§ 3). **ADMITTED to a live probe**, subject to the § 4 gates. Its fair reach
  is bounded by the ablation to the OUTPUT_VALUE subset.

### § 2.3 — why M3 is a *different* mechanism, not "reflexion in name only"

Sequential reflexion (B): each turn re-prompts the model with the cumulative
prose history `(candidate_k, raw stderr_tail_k)` and asks it to "diagnose the
bug class and produce a new corrected COMPLETE solution." M3 differs on three
load-bearing axes:

1. **Typed failure digest, not prose history.** M3 parses `stderr_tail` into a
   structured record `{failing_test_name, exception_type, expected_repr,
   actual_repr}` and presents it as an explicit contract.
2. **Explicit target-value contract.** M3 instructs: "Test `T` requires the
   function to yield EXPECTED=`…`; your code yielded ACTUAL=`…`. Produce code
   whose output for that case equals EXPECTED." Reflexion never states an
   explicit expected/actual contract.
3. **Minimal targeted patch, not full rewrite.** M3 holds the latest candidate
   fixed and asks for the smallest change that makes the failing assertion pass,
   reducing the regression rate that cancelled reflexion's rescues (W110 net 0).

Same model, same K=5 byte-exact budget, same information regime (only
`stderr_tail`, never the test source). It is exactly the "executor-guided
patching materially different from plain reflexion" candidate named in the
charter.

### § 2.4 — primary / secondary benchmark usage (LOCKED)

* **BigCodeBench is the PRIMARY battlefield** (W110 gave the stronger resistant
  negative; MLB-1 PASS proves the mechanism is genuinely exercised; the harness
  is real, SHA-pinned, bug-fixed). All W111 probes/pilots run on the pinned
  W110 BigCodeBench corpus (`ca4f352e…`) and gold-green pool.
* **LiveCodeBench is the SECONDARY resistant cross-check** — used ONLY if M3
  survives the BigCodeBench fair 30-slice cheap pilot with a clean
  mechanism-driven PASS, as a pre-committed cross-check (then it is honestly
  warranted). NOT used speculatively.
* APPS is NOT reopened as a main lane. The old reflexion mechanism gets no more
  NIM.

---

## § 3 — candidate slate (LOCKED — hypotheses written BEFORE any live result)

| ID | Mechanism | Hypothesis (pre-results) | Exact weakness attacked | Why it could beat A1 on resistant code | Cheap-probe / kill rule |
|---|---|---|---|---|---|
| **M1** | library/spec-grounded planner→coder (1 plan call + 4 code samples) | An explicit typed API/behaviour plan before coding reduces semantic-misread failures | spec-comprehension / library-composition | If failures were comprehension/composition errors, planning would front-load them | **KILLED NIM-free (§ 2.2)** — failures are hidden-convention, not comprehension; sacrifices a self-consistency sample; dominated by M3 |
| **M2** | tool-augmented local symbol/doc introspection (local `inspect`/`help`, then code) | Grounding real library APIs locally reduces hallucinated API use | API hallucination / wrong signatures | If failures were API-grounding errors, local introspection would fix them | **KILLED NIM-free (§ 2.2)** — API_GROUNDING is 1.8 % of failures; introspection can't reveal hidden-test conventions |
| **M3** | executor-grounded structured-failure patcher (1 initial + 4 typed minimal-patch turns) | A typed expected/actual contract + minimal-patch framing rescues output-value mismatches that prose reflexion could not | failure-feedback actionability on SEMANTIC failures | Targets the dominant 81.6 % SEMANTIC class with executor grounding; explicit expected-value contract is information reflexion stated only implicitly | **ADMITTED** to a live probe (§ 4.3); KILL if it rescues 0 OUTPUT_VALUE hard-core problems |

For each surviving candidate the discipline is: (1) hypothesis before results
[done above]; (2) exact weakness [above]; (3) why-beats-A1 [above]; (4) cheap
probe first [§ 4]; (5) ≥ 1 adversarial ablation [§ 4.2]; (6) kill if weak [§ 4.3].

---

## § 4 — cheap-probe + ablation rules (LOCKED)

### § 4.1 — NIM-free cheap probes (executed before any NIM)

1. **Weakness-coverage probe** (`mine_w111_resistant_failure_modes_v1.py`): the
   re-executed failure census (§ 2.1). KILLS any candidate with attacked-class
   coverage < 33 %. → M2 killed (1.8 %).
2. **Hard-core recoverable-surface ablation** (§ 2.1): mock-coupling vs
   output-value on the 8 both-fail problems. → M1 killed (real reach ≈ 0;
   dominated); caps M3's fair reach to the 2 OUTPUT_VALUE problems.

### § 4.2 — M3 information-sufficiency adversarial ablation (NIM-free; executed)

The strongest argument AGAINST M3: reflexion (B) ALREADY received the same
800-char `stderr_tail` containing the `AssertionError: expected != actual`, and
rescued only 25 %. So M3's incremental information over B is ZERO — its only
lever is *structuring + explicit-contract + minimal-patch*. **Ablation result:**
of the 8 hard-core failures, 6 are MOCK_COUPLING (the expected behaviour is NOT
in `stderr_tail` — it is hidden in the test's mock setup, which a fair M3 never
sees), and only 2 (BigCodeBench/15, /20) are OUTPUT_VALUE (expected value IS in
`stderr_tail`). So M3's *entire fair upside on the hard core is 2 problems*, and
both are cases reflexion failed WITH the expected value already in hand. This is
a strong negative prior — but it is non-zero, and M3's structuring/contract
lever is genuinely untested, so M3 earns a *smallest-decisive* live probe.

### § 4.3 — M3 smallest-decisive live cheap probe (the ONLY NIM spend; gates LOCKED)

M3 earns the probe iff (both hold, both true): weakness-coverage ≥ 33 % (M3 =
81.6 %) AND hard-core fair recoverable surface ≥ 1 (M3 = 2). Both hold.

**Probe design (smallest-decisive, ~130 NIM calls):** on a hard-core-focused
slice of the pinned W110 BigCodeBench gold-green pool — the 8 both-fail problems
+ BigCodeBench/51 (the B-rescue, a hold check) + BigCodeBench/26 (the
B-regression) + 3 both-pass controls = **13 problems**; single seed (111001);
K=5 byte-exact. Run **A1 (re-sampled, paired)** + **M3** on each (A0 reused
from W110 context). NIM = 13 × (5 + 5) = **130 calls** at
`meta/llama-3.3-70b-instruct`. A ≤2-problem canary runs first.

This slice is **rescue-concentrated by construction** → it yields an UPPER
BOUND on M3, never a PASS claim (the W102/W106 rescue-concentrated-slice
discipline). Its job is to KILL or EARN, not to score.

* **KILL M3** (→ bounded-claim fallback, § 6) iff M3 rescues **0** of the 2
  OUTPUT_VALUE hard-core problems (/15, /20). Rationale: if M3 cannot beat
  reflexion even where the expected value is in hand, its structuring lever is
  not load-bearing, and the 6 mock-coupling problems are unreachable, so a fair
  30-slice pilot cannot clear +5 pp (the arithmetic: A1=21/30; +5 pp needs
  ≥23/30; M3's only fair rescue targets beyond /51 are /15,/20).
* **M3 EARNS the fair 30-slice cheap pilot** (§ 5) iff it rescues **≥ 1**
  OUTPUT_VALUE hard-core problem AND holds BigCodeBench/51 AND regresses **≤ 1**
  of the 3 both-pass controls. If earned and time/NIM permit, run § 5 within
  W111; else pre-commit § 5 as W112.

---

## § 5 — M3 fair 30-slice cheap-pilot gates (LOCKED — only if § 4.3 EARNS)

**Slice (G1):** the EXACT pinned W110 BigCodeBench 30-problem gold-green slice
(slice CID `b69bf3a0…`) — so M3 is directly comparable to W110's A1=B=70 %.

**Arms (byte-identical budget/regime to W110):** A0 (reuse W110), **A1**
(re-sampled, K=5 first-pass-among-K), **M3** (1 initial + 4 executor-grounded
structured patches, K=5). Same model, same executor (headless Agg, `unittest`
oracle, no LLM judge), budget byte-exact.

**The 9 Phase-2 gates + MLB sub-gates** (verbatim from W103–W110, with M3
replacing B), evaluated by `coordpy.contamination_resistant_interpretation_v1.
evaluate_phase2_gates_v1`:

| Gate | Pass condition |
|---|---|
| G1 slice pre-committed | slice CID `b69bf3a0…` |
| G2 A1 < 90 % | non-saturated |
| G3 M3 > A1 | strict |
| G4 (M3 − A1) ≥ +5 pp | margin bar |
| G5 (M3 − A0) ≥ +5 pp | vs single-shot |
| G6 per-problem majority (≥ 16/30) | M3 did not regress vs A1 |
| G7 budget byte-exact | A1/M3 both K=5 |
| G8 audit chain re-derives | per-call CIDs + Merkle |
| G9 executor clean | no-LLM-judge `unittest` subprocess |
| MLB-1 | patch loop invoked on ≥ 33 % of problems |
| MLB-2 | of invocations, ≥ 33 % rescued by the patch loop |

**Verdict labels:** `PASS_MECHANISM_DRIVEN` iff 9/9 + both MLB;
`PASS_NON_MECHANISM_DRIVEN` iff 9/9 but an MLB fails; `FAIL` otherwise.

**What an M3 BigCodeBench fair-slice `PASS_MECHANISM_DRIVEN` entitles:** exactly
"a genuinely DIFFERENT mechanism (executor-grounded structured patching) beats
same-budget self-consistency on a contamination-RESISTANT code benchmark at
cheap-pilot scale at 70B — the W89-reflexion resistant FAIL is reflexion-
specific, not universal." It does NOT add a retirement (Phase-3 multi-seed is
W112+), and it triggers the § 2.4 LiveCodeBench secondary cross-check pre-commit.

---

## § 6 — bounded-claim fallback rule (LOCKED — LAST resort, not default)

The bounded contamination-EXPOSED claim is accepted as the programme's current
honest code ceiling **iff ALL** hold:

1. M1 and M2 are killed NIM-free (§ 2.2) — DONE.
2. M3 is either (a) killed in its smallest-decisive live probe (§ 4.3 KILL), or
   (b) earns the fair pilot but the fair 30-slice pilot returns `FAIL` (§ 5).
3. No other genuinely-different, fair, in-budget candidate survives the § 2
   mining (none does — see § 2.2).

If triggered, register `W111-L-NO-DIFFERENT-MECHANISM-BEATS-A1-ON-RESISTANT-
CODE-AT-70B-CHEAP-CAP` and accept: **"At 70B, neither the W89 reflexion
mechanism NOR a genuinely-different executor-grounded patcher beats same-budget
self-consistency on contamination-resistant code at cheap-pilot scale; the two
confirmed retirements are contamination-EXPOSED-HumanEval-family-specific at
70B."** This is a SHARPER bounded claim, not a weaker one. It does NOT prove the
contamination confound (still single-seed resistant points); it does NOT retire
anything; it does NOT close `COO-9` (W112 may probe a stronger-model scale or a
third resistant benchmark with a different mechanism).

The fallback is explicitly NOT the default: it fires only after M3 has been
given a fair, smallest-decisive empirical shot.

---

## § 7 — graphify deliverables (LOCKED)

* Refresh at start (`graphify update .`; confirm built from current HEAD —
  DONE: `d41265d5`, "No code-graph topology changes detected" ⇒ already current).
* Re-ingest after adding the M3 module + probe script + tests.
* Use concretely (DONE / to-do): `explain run_bigcodebench_reflexion_bench_v1`
  + `interpret_second_resistant_result_v1` (graph-wired confirmation); `path`
  bigcodebench↔livecodebench / bigcodebench↔apps benches (4-hop sibling check);
  `query` for the resistant/retirement claim surfaces; `affected` on the new M3
  module + the truth surfaces after they land.
* Refresh at end; record exactly what graphify changed in file selection /
  understanding (`docs/RESULTS_W111_*` + the milestone summary).

---

## § 8 — W112 branch logic (LOCKED)

Driven by the Lane α outcome:

* **M3 smallest-decisive probe EARNS + fair 30-slice pilot run in W111 →
  `PASS_MECHANISM_DRIVEN`** ⇒ **W112 = BigCodeBench Phase-3 multi-seed M3
  retirement bench** (3 seeds × 100 × K=5; mirrors W103→W105) — the earned path
  to a contamination-RESISTANT THIRD retirement — preceded by the § 2.4
  LiveCodeBench secondary cross-check (cheap pilot). 405B + Llama-3.1 stay closed.
* **M3 probe EARNS but fair pilot NOT run in W111 (NIM/time)** ⇒ **W112 = run
  the earned M3 fair 30-slice cheap pilot** (no new earning work needed).
* **M3 probe KILLS M3 (or fair pilot FAILs)** ⇒ bounded-claim fallback fires
  (§ 6); **W112 = the honest post-fallback move** — either a cross-scale-UP
  probe of M3/reflexion on a stronger code model if one becomes reachable on
  NIM, or a THIRD genuinely-different resistant benchmark with a different
  mechanism. NOT a reflexion rerun; NOT APPS; NOT 405B/Llama-3.1.
* **M3 probe could not launch (NIM unavailable)** ⇒ W111 closes
  probe-earned-but-operator-gated; **W112 = launch the earned M3 probe**.

`COO-9` stays the lead path in all branches unless the evidence forces a
different code-line move.

---

## § 9 — Stable boundary preservation (LOCKED)

* `coordpy.__version__ == "0.5.20"`; `coordpy.SDK_VERSION ==
  "coordpy.sdk.v3.43"`; no PyPI publish; `coordpy/__init__.py` untouched.
* New work is explicit-import only: `coordpy.executor_grounded_patcher_v1` (M3,
  the structured-failure patcher bench, A0/A1/M3 byte-identical-budget to the
  W110 line). New scripts: `scripts/mine_w111_resistant_failure_modes_v1.py`
  (NIM-free mining; landed), `scripts/run_w111_m3_probe.py` (the M3 probe/pilot
  runner). The interpretation evaluator
  `coordpy.contamination_resistant_interpretation_v1` is REUSED (the 9-gate+MLB
  evaluator is mechanism-agnostic).

---

## Honest framing

W111 attacks the honest aggressive move — a genuinely DIFFERENT mechanism on
contamination-resistant code — rather than accepting defeat by default. The
mechanism-mining shows the resistant failure is overwhelmingly SEMANTIC
hidden-test coupling (81.6 %), not API-grounding (1.8 %), so M1 (planner) and M2
(introspection) attack near-absent weaknesses and die at $0 NIM. M3 (an
executor-grounded structured-failure patcher — genuinely different from prose
reflexion) is the one aligned candidate and earns a smallest-decisive live
probe, with a strong but non-zero negative prior (its fair upside is the 2
OUTPUT_VALUE hard-core problems, where reflexion already failed with the
expected value in hand). If M3 rescues there → a fair 30-slice pilot and, on a
clean PASS, the new resistant-code frontier + a LiveCodeBench cross-check. If M3
dies → the bounded contamination-EXPOSED-HumanEval-family-at-70B claim is the
honest ceiling, registered SHARPER, not weaker. Either way: NOT a retirement by
itself, NOT proof of the confound, NOT a reopening of any closed line. The two
confirmed retirements remain W89 + W105. `COO-9` stays lead.
