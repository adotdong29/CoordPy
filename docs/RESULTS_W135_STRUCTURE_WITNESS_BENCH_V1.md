# RESULTS — W135: oracle-derived solution-STRUCTURE witness + held-out NON-COMPLEXITY eval + conditional frontier rerun

Executes the pre-committed `docs/RUNBOOK_W135.md` (locked before any NIM). COO-9 sibling (COO-60).
`coordpy.__version__ == "0.5.20"` · `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` · no PyPI ·
`coordpy/__init__.py` untouched · `ultracode` OFF.

## One line

W133 proved the exact-oracle **EW1 counterexample witness adds +0.00 pp over blind reflexion** on the
WRONG_ALGORITHM / SEARCH_ENUM modes (the wrong-algorithm capability ceiling). **W135 asks whether
oracle-derived SOLUTION STRUCTURE — a greedy-failure certificate, an optimal-substructure / recurrence
witness, a search-frontier exact-count witness — can break that ceiling a bare counterexample could
not, without leaking the answer.** The instrument is REAL (Lane α SUCCESS: fires genuinely-new +
leakage-clean, 80/80 naive/ref separation), but the held-out dev verdict is sharp and clean: **on the
dedicated non-complexity field at 70B, every arm — A0, A1, B0 (blind reflexion), C1 (counterexample),
and S4 (the full structure witness) — lands at exactly 81.25 %**, the structure witness rescues **0**
problems over the counterexample (S4−C1 = +0.00 pp, 0 modes / 0 families), so the **§7a dev gate FAILS
⇒ $0 eval, $0 frontier**. The wrong-algorithm ceiling is **CAPABILITY-bound, not feedback-form-bound,
at 70B**. **W89 (+5.56) + W105 (+7.00) remain the only two retirements; W135 retires none.**

All numbers below are filled ONLY from emitted verdict JSON (`results/w135/**`).

## Lane α — structure-witness instrument + non-complexity corpus ($0 NIM) — **SUCCESS**

New modules (explicit-import only): `coordpy.solution_structure_witness_v1` (the SW1–SW4 structure
slate + the deterministic content-addressed `StructureWitnessV1` + the same-budget arm
`run_structure_witness_arm_v1` + the `structure_witness_is_genuinely_new_v1` guard) +
`coordpy.noncomplexity_structure_corpus_v1` (the seed-disjoint train/dev/eval/frontier corpus over the
16 `wa_*`/`se_*` templates).

**Instrument.** Anchored on a token-minimal fresh disjoint counterexample (reusing the W133 EW1
machinery), the witness adds, on that counterexample `X`: **SW1** the optimum `V*=ref(X)` + the
obvious/greedy value `V_greedy=naive(X)` + the objective gap + an attribution line; **SW2** a compact
LADDER of optimal sub-values over the OBVIOUS sub-instances of `X` (array → prefixes; integer-N → the
exact-value sequence over small `1..N`), each disjoint from the graded bank, each value from the owned
oracle, with a property hint (never the recurrence/state); **SW3** the exact-vs-naive count contrast +
the small-instance exact-count sequence for SEARCH_ENUM; **SW4** (LEAD) the richest applicable union +
a shift-left edge-case step. No-leakage is enforced + tested: the witness EXECUTES `ref/naive/brute`
but never renders solver source (a structural test asserts `to_prompt_block` carries no
`def`/`import`/`class`), the counterexample + every sub-instance are byte-disjoint from `secret_cases`,
and the recurrence/clever-state is never stated; grading on the DISJOINT hidden bank makes memorising
the shown ladder unable to pass — the witness tests GENERALISATION.

**Corpus** `coordpy.noncomplexity_structure_corpus_v1` — the 8 `wa_*` (greedy-vs-DP) + 8 `se_*`
(wrong-recurrence counting) templates × 5 seed-disjoint seeds/split. Admitted **train 80 / dev 79 /
eval 80 / frontier 78** (slice 30) — all ≥ the 36/36/36/30 floors; both modes in every split
(WA 40 / SE 38–40). **LOCKED CIDs** (from the $0 build at `timeout_s=8.0`, predating all β NIM):
`corpus_cid 306610ae…`, `eval_split_cid 3f6e3e59…`, `frontier_slice_cid 8aa53564…`. **Held-out
integrity TRUE** after a principled cross-split dedup: a handful of SEARCH_ENUM `fib_no_adjacent`
instances have a seed-INDEPENDENT tiny case set ⇒ byte-identical problems across seeds (whole-problem
`content_cid` collision); keeping each `content_cid` in the first split it appears (1 dev + 2 frontier
instances dropped, floors untouched) makes the splits truly content-disjoint.

**Self-tests + faithfulness gate (all $0, ALL PASS):** witness reproducibility ✓, deterministic
shrink + ladder ✓, **naive/ref separation 80/80** (the structure witness fires on every admitted
train `naive_source` and is silent — NONE — on every `ref_source`; agreement with the EW1
counterexample 80/80), **genuinely-new-vs-EW1 54/80 (67.5 %)** (a ≥2-rung optimal-substructure ladder,
or a canonical-greedy datapoint distinct from the candidate's output, beyond EW1's bare triple — the
remaining 26/80 are families whose OBVIOUS parameterisation is thin: knapsack/LCS/grid/subset, which
fall back to the counterexample + greedy contrast; reported, not hidden), **all leakage-clean**,
deterministic split regeneration ✓. **Regression fixtures** (4 clean-ladder representatives spanning
both modes — `wa_max_nonadjacent_sum` / `wa_min_coins` / `se_count_stair_climbings` /
`se_count_bsts_catalan`) all fire genuinely-new (3–5 ladder rungs). **Negative control:** the
structure witness is SILENT (NONE) on a COMPLEXITY_BLIND `naive_source` (value-correct-but-slow ⇒ no
counterexample ⇒ no structure — structure is value/algorithm-specific, not a complexity nudge);
symmetric to the W134 deployable complexity witness being silent on WA/SE. **Positive control:** the
correct `ref_source` yields NONE. ⇒ `W135-T-SOLUTION-STRUCTURE-WITNESS-INSTRUMENT-MINTABLE`.

Artifacts: `results/w135/corpus/{corpus_build_v1,separation_characterization_v1,selftest_v1}.json` +
`corpus_cache.pkl` (deterministic cache so the bench never re-mints).

## Lane β — held-out non-complexity mechanism bench — **DEV GATE FAILS ⇒ $0 eval, $0 frontier**

Executed the DEV bench: 16 held-out problems (16 families × 1 seed = **8 SEARCH_ENUM + 8
WRONG_ALGORITHM**), `meta/llama-3.3-70b-instruct`, 1 seed × K=5, **336 NIM calls, wall 8 507 s**. Arms
same-budget K=5, each scored in the "B" slot so `arm − A1 ≡ B − A1` (verbatim W108 evaluator).

**Decisive LEAD subset** (`A0/A1/B0/C1/S4`) — a transparent post-lock execution amendment (RUNBOOK
§9): the tiny latency probe measured 0.5 s but the representative 1536-token calls ran ~20 s (base) /
~58 s (oracle-heavy witness arms), above the 12 s throttle threshold (the probe under-measured), so
the full 9-arm slate projected to ~9 h. Per the throttle's intent the dev pass ran the decisive subset
— C1 (the flat counterexample baseline) + S4 (the LEAD, which renders the SW1⊕SW2⊕SW3 union, so if S4
does not beat C1 no single structure arm would). The `S1/S2/S3` ablations + the D0 negative control
were a $0-staged follow-up gated on S4 clearing — **not earned**, so not run.

| arm | pass@1 | − A1 | − B0 | − C1 | rescues vs C1 (modes / fams) | MLB-2 |
| -- | -- | -- | -- | -- | -- | -- |
| A0 (single-shot, temp 0) | 81.25 % | — | — | — | — | — |
| A1 (self-consistency K=5) | 81.25 % | — | — | — | — | — |
| **B0** (blind reflexion) | **81.25 %** | +0.00 | — | +0.00 | — | 0 % |
| **C1** (EW1 counterexample) | **81.25 %** | +0.00 | **+0.00** | — | — | 25 % |
| **S4** (structure controller, LEAD) | **81.25 %** | +0.00 | **+0.00** | **+0.00** | **0 (0 m / 0 f)** | 0 % |

**Three clean findings.**
1. **The field has near-ZERO feedback headroom at 70B.** A0 single-shot already solves **13/16
   (81.25 %)**: Llama-3.3-70B writes the correct textbook DP / counting algorithm directly for most of
   these generated families (house-robber, LIS, coin-change, max-product, Fibonacci/Tribonacci/Catalan
   counting), so the admissible-wrong naive trap does not fool the model's OWN generation. A1, B0, C1,
   and S4 all land on the SAME 13/16 — `B0−A1 = +0.00`, so even blind reflexion adds nothing.
2. **The same 3 problems are CAPABILITY-bound for EVERY arm.** `se_lattice_paths_blocked`
   (blocked-grid path counting), `wa_knapsack_01` (ratio-greedy → 0/1 DP), and
   `wa_weighted_interval_scheduling` (count-greedy → weighted-interval DP) resist A0, A1, B0, C1, AND
   S4 — none of single-shot, self-consistency, blind reflexion, an exact counterexample, or the full
   oracle-derived structure witness cracks them.
3. **Structure does NOT beat the counterexample.** `S4 − C1 = +0.00 pp`, `S4 − B0 = +0.00 pp`, **0
   rescues over C1** (0 modes, 0 families). The structure witness fired genuinely-new + leakage-clean
   on exactly the 4/16 problems where the model failed an attempt (the 13 it one-shots never trigger a
   witness), but on the capability-bound traps the model could NOT convert the optimal-substructure
   ladder / exact-count sequence into a correct algorithm ⇒ 0 conversions. This EXTENDS W133's EW1
   +0.00: not only a bare counterexample but a full attributed solution-structure witness fails to
   break the wrong-algorithm ceiling at 70B.

**§7a DEV gate (pre-committed: lead beats B0 ≥ +3.33 ∧ beats C1 ≥ +3.33 ∧ rescues span ≥2 modes or
≥3 families): FAILS** (S4−B0 = +0.00 < +3.33; S4−C1 = +0.00 < +3.33; 0 modes/0 families) ⇒ the locked
rule fires: **$0 eval, $0 frontier — the frontier rerun is NOT earned** ⇒
`W135-L-SOLUTION-STRUCTURE-WITNESS-DEV-CAP`. No close miss is hand-waved into an earn. Artifacts:
`results/w135/dev/w135_dev_meta_llama-3.3-70b-instruct_20260604T031806Z/w135_dev_report.json`.

## Lane γ — research + stronger-model gate + frontier

Primary-source research (arXiv/OpenReview/official venues) was decisive and CHANGED the mechanism
FORM (wired into the executable witness, §4–§5 of the runbook), not summarized as output:
**PGS (arXiv:2506.18315)** — minimal-by-input-token + property-oriented feedback; raw I/O
counterexamples cause repair failure via cognitive load (the most parsimonious explanation of W133's
C1 +0.00) ⇒ shrink-to-minimal + property phrasing + explicit ATTRIBUTION (the delta over a bare
counterexample), corroborated by counterexample-guided loop-invariant repair (TOPLAS 2025) and Dolcetti
(arXiv:2412.14841). **SolidCoder (arXiv:2604.19825)** — shift-left edge-case enumeration is the largest
oracle-free repair lever ⇒ prepended to every S-arm rewrite. **Pu, "Synthesis of First-Order DP
Algorithms" (OOPSLA 2011) + KNARsack (arXiv:2509.15239)** — recovering a recurrence from a sub-value
table is itself a synthesis problem whose demonstrated solutions need a constraint solver (we lack) or
training (we lack) ⇒ SW2 pre-registered as the EXPLORATORY + leak-constrained arm; **LDB
(arXiv:2402.16906)** shows the model its OWN program's runtime values, not oracle sub-values, so the
oracle-derived attributed structure witness is genuinely new in FORM. The decisive empirical caveat the
literature predicted — that a sub-value table does not by itself let a frozen LLM recover the algorithm
without a solver/training — is exactly what the held-out dev result observed (0 conversions on the
capability-bound traps).

Stronger-model gate re-derived `NO_CERTIFIABLE_STRONGER_MODEL`, **decision CID `258b6ed7` invariant**
({KNOWN:1, UNKNOWN:4}): Llama-3.3-70B (Dec-2023) + Llama-4-Maverick (Aug-2024) are the only
primary-KNOWN cutoffs; Qwen3-Coder-480B / DeepSeek-V4-Pro / Mistral-Small-4-119B-2603 / GLM-5 all
primary-UNDISCLOSED; the new entrant MiniMax-M3 (2026-06-01) has no published card/cutoff. Gate
**CLOSED**; frontier target stays `meta/llama-3.3-70b-instruct`. Artifact
`results/w135/stronger_model_gate/gate_recheck_v1.json`.

**Frontier: NOT launched.** The §7a dev gate failed, so per the locked spend rule the frontier
non-complexity rerun was NOT earned and **$0 frontier NIM** was spent. **No Maverick cross-scale check
was run** (the frontier was not earned; W135 does not block on Maverick). No stronger-than-Maverick
model became primary-KNOWN/certifiable. graphify START (`b023ee4`) + END refreshed; the new
`solution_structure_witness_v1` is a REAL 1-hop `imports_from` bridge to `exact_oracle_witness_v1`
(probe + counterexample reuse) AND `resistant_by_construction_battlefield_v1` (exec) AND the
reflexion/bench path.

## Net

W135 LANDS a real, executable, leakage-clean oracle-derived solution-structure witness instrument + a
dedicated held-out non-complexity corpus (Lane α SUCCESS), and the held-out dev bench gives a sharp,
honest verdict: on the dedicated WRONG_ALGORITHM / SEARCH_ENUM field at 70B the structure witness is
REAL but **adds NOTHING over the counterexample (S4−C1 = +0.00) or blind reflexion (S4−B0 = +0.00)** —
the model one-shots the 13 easy families (A0 = 81.25 %) and the 3 hard ones are CAPABILITY-bound for
every feedback form (single-shot, self-consistency, reflexion, counterexample, and full structure
alike). So the §7a dev gate fails and the frontier rerun is genuinely NOT earned (`$0` eval, `$0`
frontier). This SHARPENS the W133 localisation: the wrong-algorithm sub-mode is not merely
counterexample-unliftable, it is **structure-unliftable at 70B** — the residual cap is a generation
CAPABILITY limit, not a feedback-FORM limit. **W89 (+5.56) + W105 (+7.00) remain the only two
retirements; W135 retires none.** COO-9 stays lead. The structure-witness instrument + non-complexity
corpus STAND as reusable, push-button assets (eval slice `3f6e3e59…` + frontier slice `8aa53564…`
locked + cached) for a code-competent local model / a primary-KNOWN stronger model when the gate opens.
No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.

W135 (per RUNBOOK §13, dev-fail branch) ⇒ **W136** = accept
`W135-L-SOLUTION-STRUCTURE-WITNESS-DEV-CAP`; the wrong-algorithm ceiling is confirmed CAPABILITY-bound
even under oracle-derived attributed structure feedback at 70B; the remaining levers are a
code-competent local model (better generation), a primary-KNOWN stronger model when the §3 gate opens,
the Maverick cross-scale push-button, or a genuinely different mechanism axis.

## Carry-forwards added

* `W135-T-SOLUTION-STRUCTURE-WITNESS-INSTRUMENT-MINTABLE` — the oracle-derived greedy-failure /
  optimal-substructure / search-frontier witness is buildable + reusable; 80/80 naive/ref separation,
  genuinely-new-vs-EW1 54/80, all leakage-clean, neg/pos controls + clean-ladder fixtures pass; $0.
* `W135-T-NONCOMPLEXITY-STRUCTURE-CORPUS-MINTABLE` — a dedicated WA+SE seed-disjoint corpus (80/79/80/78
  ≥ floors; held-out integrity TRUE after cross-split dedup of the seed-independent `fib_no_adjacent`
  collisions); `corpus_cid 306610ae…`, eval `3f6e3e59…`, frontier `8aa53564…` locked + cached.
* `W135-T-NONCOMPLEXITY-FIELD-LOW-HEADROOM-AT-70B` — A0 single-shot already solves 13/16 (81.25 %);
  A1 = B0 = C1 = S4 = 81.25 %; the 3 residual (`se_lattice_paths_blocked`, `wa_knapsack_01`,
  `wa_weighted_interval_scheduling`) are capability-bound for every feedback form ⇒ the field offers no
  feedback-mechanism headroom at 70B (the WA/SE analogue of W134's high-B0 complexity field).
* `W135-T-STRUCTURE-WITNESS-TIES-COUNTEREXAMPLE-ON-HELD-OUT-DEV` — S4 − C1 = +0.00 pp, 0 rescues over
  C1 (0 modes/0 families); the structure witness fired genuinely-new + leakage-clean on the 4/16
  failed problems but the model converted NONE on the capability-bound traps; oracle-derived attributed
  structure adds nothing over a bare counterexample on WA/SE at 70B.
* `W135-L-SOLUTION-STRUCTURE-WITNESS-DEV-CAP` — §7a dev gate fails (S4−B0 = S4−C1 = +0.00 < +3.33;
  0 modes/families) ⇒ $0 eval, $0 frontier; the wrong-algorithm ceiling is CAPABILITY-bound, not
  feedback-form-bound, at 70B. The deployable structure lever does NOT earn a frontier rerun.
