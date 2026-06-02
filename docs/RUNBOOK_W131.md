# RUNBOOK W131 — code-competent MODEL supply census + stronger-generator hard-cluster dev bench (W129 selector FIXED) + targeted resistant probe only if earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W131 hosted-NIM dev-bench call and BEFORE any
dev-bench or probe result is interpreted. The Lane α census (§ 2) and the Lane γ cutoff-gate recheck
(§ 9) are $0-NIM (reachability GET + local-Ollama smoke on local compute + a deterministic
read-only certification re-derivation) and are allowed before this lock — the W129/W130 $0-recon
discipline. They are already emitted (census `census_cid d360c117…`; gate decision CID `258b6ed7…`
invariant). Fill `docs/RESULTS_W131_*` ONLY from emitted verdict JSON
(`feedback_never_prewrite_results_before_data`). The pre-committed code rules below are the branch
authority, not any prior or hope.

`ultracode` OFF. W131 is a bounded model-supply / conditional-probe milestone, not a repo-wide
dynamic-workflow job. It turns ON only if the work unexpectedly expands into a genuine
dynamic-workflow problem (multiple code-competent models all earn live runs at once / repo-wide
adapter-model integration / broad multi-surface external verification at once) — and only after an
explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py` untouched,
advanced work explicit-import only.

---

## § 0 — Why W131 is NOT another selector / prompt-only / battlefield lap

W129 moved the binding cap from SELECTION to GENERATION. W130 attacked the GENERATOR directly under
the same budget (Maverick, K=5, W129 selector fixed): a stronger same-budget generator created
EXACTLY 1 new pool solve (GG2 cracked `doubleup`) — below the +2 earn bar — and the dominant
`WRONG_ALGORITHM_ADMISSIBLE` failures were capability-bound (`W130-T-ADMISSIBLE-SKETCH-IS-CAPABILITY-
NOT-GENERATION-FIXABLE`). So the next honest lever is **stronger MODEL capability**, not a better
selector and not more prompt-only generator engineering.

W131 is: (1) a real code-competent MODEL supply census; (2) a stronger-generator hard-cluster dev
bench that swaps the base model while holding the W129 selector FIXED downstream; (3) a targeted
resistant probe ONLY if a stronger generator genuinely creates new headroom AND a frontier-eligible
target exists. Bounded-context / compaction / summarization / "cram less / truncate better" remain
explicit anti-patterns, NOT the path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — code-model supply + capability census (MAIN diagnosis, $0).** Build
  `coordpy.code_model_supply_census_v1` (explicit-import only). Census the THREE supply surfaces
  (local-HF transformer runtime, local-Ollama OpenAI-compatible, hosted-NIM), run the tiny
  same-family code smoke gate on reachable local code models, cross-cut with the cutoff gate, and
  emit a machine-checkable matrix to `results/w131/census/model_supply_census_v1.json`. Identify the
  best honest dev candidate(s). **$0** (hosted smoke deferred to the dev-bench canary).
* **Lane β — stronger-generator hard-cluster dev bench, W129 selector FIXED (MAIN empirical;
  EXPOSED dev spend ALLOWED).** Build `coordpy.generator_model_bench_v1` (explicit-import only): the
  B0–B4 slate (§ 5) over a swappable model (local $0 + hosted NIM), each arm at MATCHED K=5 with the
  W129 NIM-free SOLEAD selector held FIXED downstream. Run the stored-regression trio guard (§ 6)
  via the trio targets. Apply the R2W earn gate (§ 7) across ALL model rungs × arms. Kill fake/weak
  arms sharply. Emits `results/w131/dev_bench/verdict_<model>.json` per rung.
* **Lane γ — targeted resistant probe / stronger-model gate / truth.** Re-check primary cutoffs
  (§ 9). A targeted resistant probe is earned ONLY iff T1 ∧ T2 (§ 8). If earned, run the smallest
  honest cluster-matched probe first on a FRONTIER_ELIGIBLE target. Else **$0 resistant NIM**,
  register the cap. Keep W123–W130 caps closed unless new evidence genuinely changes them. Refresh
  graphify START + END (§ 10); land executable code, not docs only.

Branch order: α ($0 census) → γ-gate ($0 recheck) → β (dev bench: local $0 rung + hosted NIM rung,
EXPOSED dev spend) → γ-probe (T1∧T2 → targeted resistant probe; else $0).

---

## § 2 — Model-supply census schema (LOCKED before results)

`coordpy.code_model_supply_census_v1.CodeModelSupplyRecordV1`, one record per reachable/blocked
model, fields LOCKED in code:

| field | meaning |
|---|---|
| `model_id` | model identifier |
| `access_path` | `LOCAL_HF` / `LOCAL_OLLAMA` / `HOSTED_NIM` |
| `code_prior` | `CODE_TUNED` / `REASONING` / `GENERAL_LM` / `EMBED_OR_OTHER` (name-keyed, transparent UPPER BOUND) |
| `param_hint`, `context_hint` | size + advertised context relevant to ICPC |
| `load_success`, `blocked_reason` | local: weights/endpoint reachable; hosted: in `/v1/models` |
| `smoke_ran`, `smoke_pass`, `smoke_detail` | the same-family code smoke gate result |
| `cutoff_disclosure` | `PRIMARY_KNOWN` / `UNKNOWN` (primary-source) |
| `cutoff_boundary` | the disclosed cutoff or "" |
| `stronger_than_maverick` | True / False / None |
| `usage_class` | `FRONTIER_ELIGIBLE` / `DEV_ONLY` / `SETTLED` / `NOT_A_GENERATOR` |
| `realistic_for_dev_bench` | code-competent + reachable + smoke-pass (or smoke deferred) |

**Code-competence rigor:** a model is NOT a dev candidate just because it loads/is reachable. It
must pass `code_smoke_gate_v1` — emit a runnable program that solves a trivial stdin/stdout task on
BOTH `SMOKE_CASES` (the 2-case guard rejects constant-output cheats). Reachable-but-not-emitting ⇒
`realistic_for_dev_bench=False`.

**Fence-format normalization (`normalize_fence_v1`, LOCKED):** some code models emit the fence
info-string on its own line (` ```\npython\n<code> `), leaving a stray bare `python` first line that
crashes BOTH this smoke extractor AND the audited `extract_candidate_code_v1` (NameError → RC:1).
`normalize_fence_v1` moves a misplaced `python` tag onto the fence line. It is a **parsing-fairness
fix, applied uniformly at the generation seam, NEVER a capability lever and NEVER a change to any
algorithm**; the call sidecar preserves the RAW model output. (W130 honesty rule: a parsing fix is
reported AS a parsing fix; it removes an UNDER-statement of capability, it does not add capability.)

---

## § 3 — No-leakage + teacher/target-disjointness rule (LOCKED, enforced in code)

1. **NEVER** expose a target's accepted solution, secret input, secret answer, or validator
   internals to ANY model-facing generator prompt. The plain/analyze/implement/rewrite prompts
   contain ONLY: the target's PUBLIC statement, PUBLIC samples, public-signal-derived cases (model
   DERIVED counterexamples + the candidate's own outputs), candidate SOURCE code, and typed PUBLIC
   failure digests. Rewrite feedback (GG2/GG4/GGLEAD) uses ONLY a PUBLIC sample or a model-DERIVED
   case — NEVER a secret case.
2. Every committed/pool candidate passes the W126/W127 provenance-aware leakage guard
   `reproduces_accepted_block_v1` (contiguous-block tripwire, `min_block=3`, over the target's
   accepted texts; provenance = statement + samples). A run reproducing an accepted block is NOT
   counted as a win. Positive control preserved (a planted accepted solution is caught).
3. Teacher corpus (for the GG3 coach library, unused in the core slate) = all EXPOSED problems whose
   short-name is NOT a hard dev target; teacher/dev disjointness asserted; `teacher_corpus_cid` +
   `hard_dev_target_cid` pinned (dev `546c1466…`, byte-identical to W128/W129/W130; teacher
   `ffa027db…`).
4. The `normalize_fence_v1` transform (§ 2) touches FORMAT only and is leakage-irrelevant (it cannot
   introduce accepted bytes).
5. If ANY leakage check fails on the EARNING set ⇒ the earn is INVALID and the lane is killed
   honestly; resistant spend is NOT earned.

---

## § 4 — Same-budget accounting rule (LOCKED)

* The comparison budget is the W128/W129/W130 hard-cluster shape: **K = 5 model calls per target**
  (== the W120/W121/W127 plain baseline A1 and the W128/W130 slates).
* EVERY B-arm spends EXACTLY ≤ K model calls per target (`assert_same_budget_v1(outcome, K)`; any
  violation is counted in `same_budget_violations` and fails the run). The complexity gate, the
  failing-case finder + digest, the digest router, and ALL grading/selection are NIM-free ($0) and
  do NOT consume budget.
* The W129 selector held fixed downstream is the **NIM-free** SOLEAD (`select_so_v1(..., gen=None)`),
  so it costs $0 and **GENERATION (the base model) is the only variable across arms/rungs**.
* **Honest caveat (recorded, not hidden):** a larger model is more FLOPs/call, so a win at a higher
  capability rung is a "stronger-model" win at the SAME call budget — NOT a same-compute win. This is
  exactly the W131 question (does stronger model capability move the ceiling?), and it is stated as
  such. A win that is only formatting/parsing, or only re-cashes old pool wins, does NOT count
  (§ 7).

---

## § 5 — Generator / model slate (LOCKED before results)

New module `coordpy.generator_model_bench_v1` (explicit-import only). The B-slate maps onto the
audited W128/W130 arms so the ONLY new variable is the MODEL; every arm ends in the fixed W129
SOLEAD selector via `stronger_generator_slate_v1._finalize_arm`:

* **B0** — Maverick `meta/llama-4-maverick-17b-128e-instruct` baseline. **REUSED** from
  `results/w130` (old pool ceiling 3/11; plain baseline 2/11; GG2 cracked `doubleup`). NOT re-run.
* **B1_PLAIN** — plain same-budget generation: K i.i.d. full-solution implements (NO analyze role),
  then the fixed selector. The new arm (`run_plain_arm_v1`).
* **B2_RDIV** — role-diverse multi-sketch generation (W128 shape) via `run_gg1_v1`
  (complexity-gated role handoff) + fixed selector.
* **B3_GG2** — counterexample-to-rewrite (the W130 winning lever) via `run_gg2_v1` + fixed selector.
* **B4_GGLEAD** — GG1→GG2 composite handoff/planner via `run_gglead_v1` + fixed selector.

**Capability ladder (the model swap is the experiment):**
`Maverick-17B-128e` (B0, reuse, $0) → `qwen2.5-coder:32b` (LOCAL Ollama, $0, the best $0
code-competent local candidate) → `qwen/qwen3-coder-480b-a35b-instruct` (HOSTED NIM, the strongest
reachable code-competent model; EXPOSED dev spend).

**Spend plan (LOCKED, staged, canary first):**
* LOCAL rung `qwen2.5-coder:32b` — arms `{B1_PLAIN, B3_GG2}` × 11 targets × K=5 (**$0**, the
  free code-specialized rung).
* HOSTED rung `qwen3-coder-480b-a35b` — arms `{B1_PLAIN, B3_GG2}` × 11 × K=5 (**~110 NIM**, the
  strongest-model core). **Escalate** to `{B2_RDIV, B4_GGLEAD}` (+~110 NIM; hosted ceiling ≤ **~275
  NIM** ≈ the W130 envelope) ONLY if the core arms produce ≥ 1 NEW pool solve. Canary (1 plain gen)
  before any rung; abort if the canary emits no code.

**Realness / kill rules (NIM-free).** Reuse the slate's `gg1_gate_control_v1` /
`gg2_rewrite_control_v1` / `examine_hosted_controller_applicability_v1`. An arm/rung is KILLED if:
(i) its only gains are trivial parse/format fixes the selector would not commit (the fence
normalization is logged as parsing, not a win); (ii) on the dev bench it creates no NEW pool solve
over the old W128/W129/W130 pool (generation = decoration); (iii) a win depends on same-problem
leakage; (iv) a model is stronger but outside the same-budget spirit (recorded, § 4).

---

## § 6 — Stored-regression-trio rule (LOCKED)

Over the stored pool-bearing fixtures `blueberrywaffle` / `pawnshop` / `sunandmoon` with the FIXED
W129 NIM-free SOLEAD selector, on every rung:

* **R-KEEP-BLUE**: the fixed selector must still COMMIT a hidden-correct `blueberrywaffle` candidate
  when the pool contains one.
* **R-KEEP-SUN**: `sunandmoon` (both-correct) stays committed/safe (abstain is safe, never a
  mis-commit).
* **R-NO-NEW-MISCOMMIT-PAWN**: the fixed selector must NOT commit a hidden-wrong `pawnshop` A0
  candidate (the W129 abstain discipline holds). New SELECTION mis-commits over the trio MUST be 0.

The trio is a SAFETY guard: a stronger generator must not regress the now-understood selector
behavior. A new mis-commit on the trio is a kill signal for that rung's generation changes.

---

## § 7 — EXPOSED hard-cluster dev-bench earn rule (LOCKED; R2W — IDENTICAL to W130)

Dev set = the SAME 11 hard-cluster EXPOSED dev targets (W128 `hard_dev_target_cid 546c1466…`;
simulation_grid 4 / adhoc_math 6 / greedy_scheduling 1; graph_flow EXPOSED supply = 0). Arms at
MATCHED K=5, W129 selector fixed downstream, graded on official secret (public prescreen). The earn
gate is the W130 `apply_gg_dev_bench_earn_gate_v1` (re-exported as `apply_dev_bench_earn_gate_v1`).

The **old W128/W129/W130 pool** solved exactly {`pawnshop`, `sunandmoon`, `blueberrywaffle`} on
secret (3 pool-bearing) PLUS W130's `doubleup` crack. A **NEW pool solve** = a pool-DEAD problem for
which a W131 arm (any rung) now produces a secret-passing, leakage-clean candidate the fixed
selector did not already have. (`doubleup` is the W130 crack; a W131 arm re-solving `doubleup` is
NOT new — it must be a problem absent from the entire old pool ∪ {doubleup}.)

**R2W EARNED iff ALL hold** (evaluated over ALL rungs × arms; the BEST arm decides):
* **R2a** some arm's `new_pool_solves ≥ 2` (≥ 2 pool-DEAD problems newly solved on secret), AND
* **R2b** the new solves span ≥ 2 distinct FAMILIES **or** ≥ 2 distinct atlas FAILURE-MODES, AND
* **R2c** every new-solve run is leakage-clean (and not a fence-format-only artifact).

Merely MATCHING the old pool ceiling does NOT earn. Merely committing old pool wins more reliably
does NOT earn (the trio guards it). Re-solving only `doubleup` does NOT earn. A close edge /
one-trick parse fix / same-problem leak does NOT earn. If R2W FAILS ⇒ register
`W131-L-MODEL-AXIS-GENERATION-CEILING-DEV-BENCH-CAP`; **$0 resistant NIM**. Record which model rung +
which arm/lever was closest.

---

## § 8 — Targeted resistant-probe earn rule (LOCKED; T1 ∧ T2)

Fresh resistant hosted spend is earned ONLY iff BOTH:
* **T1** — Lane β: a W131 model/arm EARNS on the EXPOSED dev bench (R2W EARNED — genuinely new
  ceiling headroom that is **not explained by EXPOSED-problem memorization**), AND
* **T2** — the EARNING model is **FRONTIER_ELIGIBLE** (§ 9) **OR** the winning generation METHOD can
  be translated honestly onto a FRONTIER_ELIGIBLE target.

**Contamination/memorization rule (LOCKED, the W127 lesson).** Every B1–B4 model except Maverick is
DEV_ONLY (UNKNOWN cutoff). The 11 hard-cluster dev targets are EXPOSED (pre-Aug-2024 official ICPC),
so a DEV_ONLY frontier model very likely TRAINED on them. Therefore an EXPOSED earn by a DEV_ONLY
model is **contamination-confounded** and, by itself, CANNOT license a resistant claim: T2 fails
because (a) the model is not FRONTIER_ELIGIBLE, and (b) raw model CAPABILITY does not "translate"
onto Maverick (only a METHOD can). A DEV_ONLY EXPOSED earn ⇒ register
`W131-L-MODEL-CAPABILITY-LIFTS-EXPOSED-BUT-RESISTANT-INELIGIBLE` and **$0 resistant NIM**.

If T1 ∧ T2 BOTH hold: run the **smallest honest failure-mode-matched targeted resistant probe** on
the FRONTIER_ELIGIBLE target first (the resistant hard problems whose atlas mode matches the winning
lever, ≤ 1 seed), grading committed + pool on the official secret cases vs the old W120/W126 pool
(0 on the 22 uniformly-unsolved). Probe budget ceiling ≤ **~45 NIM**. If `targeted_new_solves ≥ 1`
⇒ define whether a broader resistant pilot is earned (separate, explicitly-flagged; NOT auto-run).
If = 0 ⇒ register the resistant generation cap; no broader pilot.

If T1 ∧ T2 do NOT both hold ⇒ **$0 additional resistant NIM**; register the exact blocker. No new
n=30 seed-chasing. No stronger-model RESISTANT spend unless § 9 opens. No 405B. No reopening MBPP+
V2 / frozen cross-modal / the closed Llama-3.1 rescue / APPS main-lane NIM. No dirty exposed
benchmark sold as a frontier win. No exposed frontier-control spend by default.

---

## § 9 — Frontier-eligible vs dev-only rule + per-model disclosure (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID `258b6ed7`,
invariant W114→W130). A model is **FRONTIER_ELIGIBLE** (honest for a resistant claim) iff:
its training-cutoff is **PRIMARY_KNOWN** AND ≤ the resistant instrument frontier (Maverick's Aug-2024;
the W120 resistant slice is post-cutoff) AND it is strictly stronger-than-70B-class AND reachable.
Else it is **DEV_ONLY** (reachable code-competent but UNKNOWN-cutoff or weaker — EXPOSED-dev valid,
resistant-ineligible). Maverick is PRIMARY_KNOWN Aug-2024 but **SETTLED** (exhausted as the resistant
anchor; W120/W121 already ran it).

Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B-2603,
GLM-5, and any newly reachable same-budget-comparable model (the W131 census surfaced
qwen3.5-397b-a17b, mistral-large-3-675b, deepseek-v4-flash, nemotron-4-340b, codellama-70b, …).
Standing prior re-affirmed by the $0 recheck: **{KNOWN:1 (Maverick, Aug-2024), UNKNOWN:4}**;
**FRONTIER_ELIGIBLE = NONE** (the census found 13 stronger-than-Maverick code models reachable, ALL
UNKNOWN-from-primary ⇒ DEV_ONLY). A model SUPERSEDES Maverick ONLY if it becomes primary-KNOWN AND
certifiable on the matched ICPC family. No 405B run unless reachability changes and a pre-committed
gate clears. Emit `results/w131/stronger_model_gate/gate_recheck_v1.json`.

---

## § 10 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `4a55d536`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `public_signal_selection_oracle_v1`,
  `stronger_generator_slate_v1`, `role_diverse_algorithm_search_v1`, `code_substrate_v1`,
  `transformers_runtime_v1`, `substrate_adapter_v25`, `real_task_bench_adapter_v1`,
  `controller_native_code_mechanism_v1`, `hosted_cache_aware_planner_v12`,
  `hosted_real_handoff_coordinator_v11`, and the NEW `code_model_supply_census_v1` +
  `generator_model_bench_v1`.
* `graphify path stronger_generator_slate_v1 public_signal_selection_oracle_v1` +
  `graphify path code_substrate_v1 substrate_adapter_v25` +
  `graphify affected stronger_generator_slate_v1`. `graphify query` only as a secondary
  claim-surface finder.
* The new `code_model_supply_census_v1` must create the FIRST semantic bridge unifying the three
  model-supply surfaces (`nim_frontier_text_runtime_v1` + the local-HF runtime probe + the cutoff
  gate `stronger_model_cutoff_certification_v1`); `generator_model_bench_v1` must bridge the supply
  census `build_openai_compat_gen_v1` to BOTH the W130 generator arms AND the W129 fixed selector.
  The END graph must show their edges.

---

## § 11 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless the targeted
  resistant probe earns AND a (separately-defined) broader pilot clears the +5.00pp clean-
  superiority bar. W131 retires none unless the JSON says so.
* On R2W fail (no model/arm creates ≥2 new pool solves): register
  `W131-L-MODEL-AXIS-GENERATION-CEILING-DEV-BENCH-CAP` — a stronger code model (local 32B and/or
  hosted 480B), W129 selector fixed, does NOT create ≥2 new EXPOSED hard-cluster solves at K=5 ⇒ the
  generation ceiling is not model-axis-liftable at the reachable rungs (the MODEL-axis sibling of
  the W123→W130 cap taxonomy: battlefield → encoder → re-routing → synthesis → scaffold-gen →
  role-diverse-search → selection-oracle → generator-line → **model-axis**). Record the closest rung.
* On R2W earn by a DEV_ONLY model with T2 fail: register
  `W131-L-MODEL-CAPABILITY-LIFTS-EXPOSED-BUT-RESISTANT-INELIGIBLE` (the contamination/memorization
  rule, § 8) — the supply gap is now cutoff-DISCLOSURE-bound, not model-existence-bound.
* On T1 ∧ T2 with `targeted_new_solves = 0`: register `W131-L-RESISTANT-MODEL-AXIS-CAP`.
* On T1 ∧ T2 with `targeted_new_solves ≥ 1`: register the new-solve evidence + broader-pilot
  decision (NOT a retirement by itself).
* Always carry forward `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` +
  `W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN` +
  `W130-L-GENERATION-CEILING-DEV-BENCH-CAP`. Named claims filled ONLY from the emitted verdict JSON.

---

## § 12 — W132 branch logic (pre-committed)

* If R2W fails (model-axis generation cap holds at the reachable rungs) ⇒ W132 = accept the
  registered model-axis cap; the honest remaining lever is a **PRIMARY-KNOWN-cutoff stronger model
  on the ICPC family** (none currently exists — the supply gap is now cutoff DISCLOSURE) or a
  genuinely different axis. Bounded-context / compaction remain anti-patterns.
* If R2W earns by a DEV_ONLY model + T2 fails ⇒ W132 = the EXPOSED ceiling IS model-capability-
  liftable but resistant-ineligible (contamination); pursue a FRONTIER_ELIGIBLE supply or a
  resistant-by-construction instrument dated after a primary-KNOWN strong model's cutoff.
* If T1 ∧ T2 and `targeted_new_solves = 0` ⇒ W132 = accept the resistant model-axis cap.
* If T1 ∧ T2 and `targeted_new_solves ≥ 1` ⇒ W132 = define + (operator-greenlit) run the broader
  failure-mode-matched resistant pilot; retire iff a clean +5.00pp multi-seed same-budget margin.
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
