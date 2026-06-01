# RUNBOOK W130 — generation-ceiling attack on the hard ICPC clusters + same-family EXPOSED dev bench + targeted resistant probe only if earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W130 NIM call and BEFORE any dev-bench
or probe result is interpreted. The Lane α generator-failure atlas (§ 2) is a $0-NIM
reconstruction of stored generations and is allowed before this lock (it spends no NIM and only
re-grades already-paid candidates — the W129 $0-recon discipline). Fill `docs/RESULTS_W130_*`
ONLY from emitted verdict JSON (`feedback_never_prewrite_results_before_data`). The pre-committed
code rules below are the branch authority, not any prior or hope.

`ultracode` OFF. W130 is a bounded generator-mechanism / conditional-probe milestone, not a
repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly expands into a genuine
dynamic-workflow problem (multiple generator FAMILIES all earn live runs at once / repo-wide
generator-oracle integration / broad multi-surface external verification at once) — and only
after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py` untouched,
advanced work explicit-import only.

---

## § 0 — Why W130 is NOT another selector/oracle/scaffold lap

W129 attacked the SELECTOR directly and proved the binding cap is **GENERATION, not selection**:
a selector can only commit candidates the pool CONTAINS, so committed ≤ pool ceiling
(3/11 = baseline+1) < the +2 earn bar regardless of selector quality. The honest remaining lever
is a **stronger GENERATOR that creates genuinely new candidate headroom under the same budget**.

W130 is NOT another battlefield pivot, NOT another selector/oracle retry, NOT another scaffold
retry. It is: (1) a generation-failure atlas refinement; (2) a stronger same-budget generator
line; (3) with the W129 selector HELD FIXED downstream; (4) and a resistant probe only if the
generator actually creates new headroom. Bounded-context / compaction / summarization / "cram
less / truncate better" remain anti-patterns, explicitly NOT the path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — generator-failure atlas refinement (MAIN diagnosis, NIM-free).** Build
  `coordpy.generator_failure_atlas_v1` (explicit-import only). Reconstruct the FULL old
  W128/W129 candidate pool (plain ∪ scaffold ∪ rda) per hard-cluster dev target from the stored
  W128 sidecar, grade every candidate with a mechanical failure signature, cross-check the
  OFFLINE accepted-algorithm reference (NEVER model-facing), and classify each problem's dominant
  generator-failure mode (§ 2). Quantify pool-bearing vs pool-dead, selector-fixable vs
  generator-fixable, and the dominant pool-dead modes. Emits
  `results/w130/atlas/generator_failure_atlas_v1.json`. **$0 NIM.**
* **Lane β — stronger same-budget generator line (MAIN mechanism, EXPOSED dev spend ALLOWED).**
  Build `coordpy.stronger_generator_slate_v1` (explicit-import only): the GG1–GG4 + GGLEAD slate
  (§ 5) with NIM-free realness controls + the honest hosted-controller examination. Run the slate
  on the SAME W128/W129 hard-cluster EXPOSED dev bench (11 targets) at MATCHED K=5 budget, with
  the **W129 selector held FIXED** downstream (`public_signal_selection_oracle_v1.select_so_v1`
  SOLEAD, NIM-free). Apply the R2W earn gate (§ 7). Run the stored-regression trio (§ 6) first.
  Kill fake/weak arms sharply. Emits `results/w130/dev_bench/gg_dev_bench_verdict.json`.
* **Lane γ — targeted resistant probe / stronger-model gate / truth.** Re-check primary cutoffs
  (§ 9). A targeted resistant probe is earned ONLY iff T1 ∧ T2 (§ 8). If earned, run the smallest
  honest cluster-matched probe first. Else **$0 resistant NIM**, register the cap. Keep the
  W123–W129 caps closed unless new evidence genuinely changes them. Refresh graphify START + END
  (§ 10); land executable code, not docs only.

Branch order: α ($0 atlas) → β (dev bench, EXPOSED dev spend) → γ (T1∧T2 → targeted resistant
probe; else $0).

---

## § 2 — Generator-failure atlas schema (LOCKED before results)

Per-problem dominant failure mode (`coordpy.generator_failure_atlas_v1.GENERATOR_FAILURE_MODES`,
locked in code):

| mode | definition |
|---|---|
| `SOLVED` | pool-bearing; the committed candidate is correct (no generation gap) |
| `SELECTION_FIXABLE` | pool-bearing; ≥1 hidden-correct AND ≥1 hidden-wrong public survivor (W129 domain) |
| `HIDDEN_EDGE_STATE_MISS` | pool-DEAD; a candidate passes ALL public but fails secret (near-correct algo / state/invariant/edge bug) |
| `COMPLEXITY_BLIND` | pool-DEAD; dominant non-pass failure is TLE (algorithm asymptotically too slow) |
| `WRONG_ALGORITHM_ADMISSIBLE` | pool-DEAD; 0 public survivors but ≥1 sketch matches the accepted approach on a SPECIFIC idiom (idea proposed, derivation/impl wrong) |
| `WRONG_ALGORITHM_NO_SKETCH` | pool-DEAD; 0 public survivors AND no sketch matches the accepted approach (capability failure) |
| `PARSE_IO_FAILURE` | pool-DEAD; dominant failure is parse/crash (trivial IO/format bug) |

Mechanical per-candidate failure typing (`FAIL_TYPES`): `PASS` / `HIDDEN_FAIL` / `WRONG_ANSWER`
/ `TLE` / `CRASH` / `PARSE_ERR`, from the official execution path (`_run_capture_stdout_v1`,
whitespace-collapsed exact compare — all targets are KIND_PASSFAIL). `generator_fixable` =
mode ∈ {`HIDDEN_EDGE_STATE_MISS`, `COMPLEXITY_BLIND`, `WRONG_ALGORITHM_ADMISSIBLE`,
`PARSE_IO_FAILURE`}; `selector_fixable` = mode == `SELECTION_FIXABLE`. The accepted-algorithm
admissibility check is a TRANSPARENT idiom-overlap HEURISTIC (the W127 47%-theme-classifier
lesson) and an **UPPER BOUND** on generator-fixability (a named technique ≠ a correct algorithm);
it is OFFLINE-only and NEVER model-facing.

---

## § 3 — No-leakage + teacher/target-disjointness rule (LOCKED, enforced in code)

1. **NEVER** expose a target's accepted solution, secret input, secret answer, or validator
   internals to ANY model-facing generator prompt. The GG ANALYZE / IMPLEMENT / REWRITE prompts
   contain ONLY: the target's PUBLIC statement, PUBLIC samples, public-signal-derived cases
   (model DERIVED counterexamples + the candidate's own outputs), candidate SOURCE code, and
   typed failure digests (`parse_failure_digest_v1`). GG2/GG4 rewrite feedback uses ONLY a PUBLIC
   sample (known-correct I/O) or a model-DERIVED case — NEVER a secret case.
2. The GG3 family anti-pattern coach is GENERIC family-level advice + de-identified idiom NAMES
   from the EXPOSED teacher library; it is NEVER a same-problem scaffold and NEVER carries the
   target's accepted bytes. Mechanically guarded: a coach card that reproduces an accepted block
   (`reproduces_accepted_block_v1`) → `coach_is_scaffold=True` → GG3 killed.
3. Teacher corpus (for the GG3 library) = all EXPOSED problems whose short-name is NOT a hard dev
   target; teacher/dev disjointness asserted; `teacher_corpus_cid` + `hard_dev_target_cid` pinned.
4. Every committed/pool candidate passes the W126/W127 provenance-aware leakage guard (the
   contiguous-block `reproduces_accepted_block_v1` tripwire over the target's accepted texts,
   provenance = statement + samples). A run reproducing an accepted block is NOT counted as a win.
   Positive control preserved (a planted accepted solution is caught).
5. If ANY leakage check fails on the EARNING set ⇒ the earn is INVALID and the lane is killed
   honestly; resistant spend is NOT earned.

---

## § 4 — Same-budget accounting rule (LOCKED)

* The comparison budget is the W128/W129 hard-cluster shape: **K = 5 model calls per target**
  (== the W120/W121/W127 plain baseline A1 and the W128 role-diverse search).
* EVERY GG arm spends EXACTLY ≤ K model calls per target (asserted: `n_calls ≤ K`). The
  complexity gate (GG1/GGLEAD), the failing-case finder + digest (GG2/GG4/GGLEAD), the digest
  router (GG4), and ALL grading/selection are NIM-free ($0) and do NOT consume budget.
* Any MODEL-facing verifier/critic/rewrite step COUNTS inside the K budget (e.g. GG2 = 1 ANALYZE
  + (K-2) implements + 1 rewrite = K; GGLEAD = 1 ANALYZE + (K-2) implements + 1 rewrite = K). No
  silent budget expansion: a rewrite slot is REALLOCATED from an implement slot, not added.
* The W129 selector held fixed downstream is the **NIM-free** SOLEAD (SO1→SO2; `gen=None`), so it
  costs $0 and the only variable across arms is GENERATION.

---

## § 5 — Generator slate (LOCKED before results)

New module `coordpy.stronger_generator_slate_v1` (explicit-import only). Arms:

* **GG1 — complexity-gated role handoff.** 1 ANALYZE (per-sketch Big-O REQUIRED, stated-bound
  injected) + (K-1) implements over the ADMISSIBLE sketches; sketches whose parsed complexity
  cannot meet the parsed N bound are REJECTED before implementation; freed slots reallocate to a
  worst-case-hardened re-implement of the best admissible sketch. NIM-free gate. Attacks
  `COMPLEXITY_BLIND` + the `WRONG_ALGORITHM_ADMISSIBLE` sketch→impl handoff (W129 proved the model
  CAN reason about O(N²) vs O(N); GG1 brings that into GENERATION, not post-hoc SELECTION).
* **GG2 — counterexample-to-rewrite.** 1 ANALYZE + (K-2) implements + 1 in-loop REWRITE driven by
  a typed PUBLIC/derived failure digest (a FRESH candidate, not a rerank). If the best candidate
  fails nothing public/derived, the rewrite slot funds an extra diverse impl. Attacks
  `HIDDEN_EDGE_STATE_MISS` + `PARSE_IO_FAILURE` + `WRONG_ANSWER`.
* **GG3 — family anti-pattern coach.** 1 coached ANALYZE (generic family anti-pattern/complexity/
  invariant card + de-identified idiom names) + (K-1) implements. KILLED if the coach collapses
  to a same-problem scaffold (`coach_is_scaffold`) — it must be family-level, not W127 scaffold
  retry, not answer material.
* **GG4 — planner/coordinator budget policy.** 1 ANALYZE + 2 seed implements, then the W125
  PATCH/REPLAN/ABSTAIN digest-router allocates the remaining K-3 (PATCH=rewrite then widen,
  REPLAN=new diverse sketches). The hosted CACHE-aware planner is EFFICIENCY-only (KV-prefix
  savings), recorded honestly — NOT a capability lever; the substrate handoff/coordinator literal
  bridges are KILLED as substrate-trust-specific fake-different
  (`examine_hosted_controller_applicability_v1`, machine-checkable — the W128/W129 W79 lesson).
* **GGLEAD = GG1 → GG2.** 1 GG1-ANALYZE + (K-2) admissible/hardened implements + 1 counterexample
  rewrite = K. The LEAD arm.

**Realness controls / kill rules (NIM-free).** `gg1_gate_control_v1` (an O(N²) sketch at N=1e6
MUST be inadmissible, O(N log N) admissible, unstated unjudgeable). `gg2_rewrite_control_v1` (the
failing-case finder returns a case for a public-failing candidate, None for a passing one). An
arm is KILLED if: (i) its only gains are trivial parse/format fixes that the selector would not
commit; (ii) on the dev bench it creates no NEW pool solve over the old W128/W129 pool (generation
= prompt decoration); (iii) a win depends on same-problem leakage; (iv) GG3 collapses to a
scaffold; (v) GG4's router never varies its route (decoration); (vi) GG2's rewrite never produces
a structurally-new candidate.

---

## § 6 — Stored-regression-trio rule (LOCKED)

Before / alongside the fresh dev run, over the stored pool-bearing fixtures
`blueberrywaffle` / `pawnshop` / `sunandmoon` with the FIXED W129 NIM-free SOLEAD selector:

* **R-KEEP-BLUE**: the fixed selector must still COMMIT a hidden-correct `blueberrywaffle`
  candidate when the pool contains one (the W128 unique win is not regressed by the generator).
* **R-KEEP-SUN**: `sunandmoon` (both-correct) stays committed/safe.
* **R-NO-NEW-MISCOMMIT-PAWN**: the fixed selector must NOT commit the hidden-wrong `pawnshop` A0
  (the W129 abstain discipline holds — abstain the under-determined tie, never re-commit the
  wrong candidate). New SELECTION mis-commits over the trio MUST be 0.

The trio is a SAFETY guard: the stronger generator must not regress the now-understood selector
behavior. A regression on the trio is a kill signal for that arm's generation changes.

---

## § 7 — EXPOSED hard-cluster dev-bench earn rule (LOCKED; R2W)

Dev set = the SAME 11 hard-cluster EXPOSED dev targets (W128 `hard_dev_target_cid`;
simulation_grid 4 / adhoc_math 6 / greedy_scheduling 1; graph_flow EXPOSED supply = 0). Arms at
MATCHED K=5 budget on `meta/llama-4-maverick-17b-128e-instruct`, the W129 selector fixed
downstream, graded on official secret (public prescreen). Budget ceiling ≤ **~300** NIM
(11 × 5 arms × 5 + canary). Canary first.

The **old W128/W129 pool** solved exactly {`pawnshop`, `sunandmoon`, `blueberrywaffle`} on secret
(the 3 pool-bearing); the other 8 are pool-DEAD. A **NEW pool solve** = a pool-DEAD problem that a
GG arm now produces a secret-passing, leakage-clean candidate for.

**R2W EARNED iff ALL hold** (`apply_gg_dev_bench_earn_gate_v1`):
* **R2a** some arm's `new_pool_solves ≥ 2` (≥ 2 pool-DEAD problems newly solved on secret, absent
  from the ENTIRE old W128/W129 pool), AND
* **R2b** the new solves span ≥ 2 distinct FAMILIES **or** ≥ 2 distinct atlas FAILURE-MODES, AND
* **R2c** every new-solve run is realness-REAL (its arm's realness controls pass) + leakage-clean.

Merely MATCHING the old pool ceiling does NOT earn. Merely committing old pool wins more reliably
does NOT earn (W129 already showed selection can do some of that — the trio guards it). A close
edge / one-trick parse fix / same-problem leak is NOT an earn. If R2W FAILS ⇒ register
`W130-L-GENERATION-CEILING-*-CAP`; **$0 resistant NIM**. Record which arm + which generator lever
(complexity gate / counterexample rewrite / anti-pattern coach / budget router) was load-bearing.

---

## § 8 — Targeted resistant-probe earn rule (LOCKED; T1 ∧ T2)

Fresh resistant hosted spend is earned ONLY iff BOTH:
* **T1** — Lane β: a W130 generator arm EARNS on the EXPOSED dev bench (R2W EARNED — genuinely new
  ceiling headroom), AND
* **T2** — the generator-failure atlas identifies a RESISTANT subset whose failure mode MATCHES
  the winning generator mechanism (e.g. if GG1's complexity gate earned, a resistant
  `COMPLEXITY_BLIND` / hard-cluster subset; the EXPOSED-earned mode ∩ resistant hard target
  failure-modes ≠ ∅).

If T1 ∧ T2: run the **smallest honest failure-mode-matched targeted resistant probe** first
(the resistant hard problems whose atlas mode matches the winning lever, ≤ 1 seed, the winning GG
arm), grading committed + pool on the official secret cases vs the old W120/W126 pool (0 on the 22
uniformly-unsolved). Probe budget ceiling ≤ **~45** NIM. `targeted_new_solves` = #(matched-subset
problems the GG arm newly solves on secret, leakage-clean). If ≥ 1 ⇒ define whether a broader
resistant pilot is earned (separate, explicitly-flagged; NOT auto-run). If = 0 ⇒ register the
resistant generation cap; no broader pilot.

If T1 ∧ T2 do NOT both hold ⇒ **$0 additional resistant NIM**; register the exact blocker. No new
n=30 seed-chasing. No stronger-model spend unless § 9 opens. No 405B. No reopening MBPP+ V2 /
frozen cross-modal / the closed Llama-3.1 rescue / APPS main-lane NIM. No dirty exposed benchmark
sold as a frontier win. No exposed frontier-control spend by default.

---

## § 9 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID `258b6ed7`,
invariant W114→W129). Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B, DeepSeek-V4-pro,
Mistral-Small-4-119B-2603, GLM-5, and any newly reachable same-budget-comparable model. A model
SUPERSEDES Maverick as the hosted target ONLY if it becomes primary-KNOWN (disclosed cutoff) AND
certifiable on the matched ICPC family. Standing prior: **{KNOWN:1 (Maverick, Aug-2024),
UNKNOWN:4}** ⇒ Maverick is the only certifiable hosted target. No 405B run unless reachability
changes and a pre-committed gate clears. Emit `results/w130/stronger_model_gate/gate_recheck_v1.json`.

---

## § 10 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `4a55d53`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `public_signal_selection_oracle_v1`,
  `role_diverse_algorithm_search_v1`, `controller_native_code_mechanism_v1`,
  `hosted_cache_aware_planner_v12`, `hosted_real_handoff_coordinator_v11`,
  `multi_agent_substrate_coordinator_v15`, `resistant_capability_atlas_v1`,
  `family_scaffold_generation_v1`, and the NEW `generator_failure_atlas_v1` +
  `stronger_generator_slate_v1`.
* `graphify path public_signal_selection_oracle_v1 role_diverse_algorithm_search_v1` +
  `graphify path controller_native_code_mechanism_v1 hosted_real_handoff_coordinator_v11` +
  `graphify affected public_signal_selection_oracle_v1`. `graphify query` only as a secondary
  claim-surface finder.
* The new `stronger_generator_slate_v1` must create the FIRST semantic bridge from the GG
  generation path to BOTH the W129 selector (held fixed) AND the executor digest / W125 router;
  the END graph must show its edges.

---

## § 11 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless the targeted
  resistant probe earns AND a (separately-defined) broader pilot clears the +5.00pp clean-
  superiority bar. W130 retires none unless the JSON says so.
* On R2W fail (no arm creates ≥2 new pool solves): register
  `W130-L-GENERATION-CEILING-DEV-BENCH-CAP` — a stronger same-budget generator does NOT create
  ≥2 new EXPOSED hard-cluster solves with the W129 selector fixed ⇒ the generation ceiling is not
  liftable at this model scale + budget (the GENERATION-lever sibling of the W123→W129 cap
  taxonomy: battlefield → encoder → re-routing → synthesis → scaffold-gen → role-diverse-search →
  selection-oracle → **generator-line**). Record which lever was closest.
* On R2W earn + T2 fail: register `W130-L-EXPOSED-GENERATION-EARNS-NO-RESISTANT-MATCH`.
* On T1 ∧ T2 with `targeted_new_solves = 0`: register `W130-L-RESISTANT-GENERATION-LINE-CAP`.
* On T1 ∧ T2 with `targeted_new_solves ≥ 1`: register the new-solve evidence + broader-pilot
  decision (NOT a retirement by itself).
* Always carry forward `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` +
  `W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN`. Named claims filled ONLY from the
  emitted verdict JSON.

---

## § 12 — W131 branch logic (pre-committed)

* If R2W fails (generation cap holds) ⇒ W131 = accept the registered generation-ceiling cap; the
  honest remaining lever is NOT more same-budget generator engineering but a **code-COMPETENT
  local model** (new trajectories at $0) or a **primary-KNOWN reachable stronger-than-Maverick
  model** (better generation AND selection), or a genuinely different axis. Bounded-context /
  compaction remain anti-patterns.
* If R2W earns but T2 fails ⇒ W131 = the generator is a real same-family lever but the resistant
  field's missing capability is not generation-addressable in the matched cluster; pursue the
  matched mode on the resistant field via a different instrument.
* If T1 ∧ T2 and `targeted_new_solves = 0` ⇒ W131 = accept the resistant generation cap;
  stronger / code-competent model.
* If T1 ∧ T2 and `targeted_new_solves ≥ 1` ⇒ W131 = define + (operator-greenlit) run the broader
  failure-mode-matched resistant pilot; retire iff a clean +5.00pp multi-seed same-budget margin.
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
