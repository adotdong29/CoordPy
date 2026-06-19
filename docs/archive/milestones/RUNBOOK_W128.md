# RUNBOOK W128 — role-diverse algorithm SEARCH on the non-scaffoldable resistant ICPC clusters + same-family hard-cluster dev bench + targeted resistant probe only if earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE any W128 NIM call and BEFORE any dev-bench
or probe result is interpreted. Fill `docs/RESULTS_W128_*` ONLY from emitted verdict JSON (the
"never pre-write results" discipline — `feedback_never_prewrite_results_before_data`). The
pre-committed code rules below are the branch authority, not any prior or hope.

`ultracode` OFF. W128 is a bounded mechanism-search / conditional-probe milestone, not a
repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly expands into a
genuine dynamic-workflow problem (multiple role-diverse mechanisms all earn live runs at once /
repo-wide role-search integration / broad multi-surface external verification at once) — and
only after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py` untouched,
advanced work explicit-import only.

---

## § 0 — Why W128 is NOT another lap

W120–W127 closed every prior lever against the resistant ICPC field:

| milestone | lever | result |
|---|---|---|
| W120–W123 | battlefield | resistant +0.00 / exposed +3.33 FAIL; n=30 unresolvable; ≥100/field supply-UNREACHABLE |
| W124 | local transformer encoder | distilgpt2 hidden state adds nothing; no code-competent local model |
| W125 | $0 controller re-routing | `blind_selection_headroom = 0` ⇒ GENERATION-CAPPED |
| W126 | $0 deterministic synthesis | 894 leakage-clean candidates, **oracle ceiling 0/22** ⇒ SYNTHESIS-CAPPED |
| W127 | family-SCAFFOLD fresh generation | EXPOSED dev EARNED +2 (weak/confounded) but resistant probe **0/6** ⇒ `RESISTANT_SCAFFOLD_FRESH_GEN_CAP` |

W127's capability atlas (`results/w127/atlas/capability_atlas_v1.json`) is the W128 starting
point: the 22 uniformly-unsolved resistant problems are **95% wrong-algorithm** (19 visible / 3
hidden; ≈0 TLE/crash) and **algorithmically DIVERSE** (top-2 family concentration ≈ 45–50%; no
dominant cluster). W127 already proved a better SCAFFOLD does not transfer. **W128's question is
genuinely different:** can a *role-diverse algorithm SEARCH* mechanism — generate → verify →
select → abstain — attack the **non-scaffoldable** clusters (`graph_flow`, `simulation_grid`)
that a skeleton cannot cover, validated on a disjoint same-family EXPOSED hard-cluster dev bench
first, and probed on the resistant field ONLY if it earns the right?

W128 is NOT another battlefield pivot, NOT another scaffold retry, NOT another deterministic
synthesis lap. Bounded-context / compaction / summarization / "cram less / truncate better"
remain anti-patterns, explicitly NOT the frontier path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — non-scaffoldable cluster hardening + role-diverse mechanism (MAIN diagnosis +
  mechanism design, NIM-free).** Derive the hard-cluster target set (§ 2) from the W127 atlas;
  write the strict per-cluster protocol (cluster id / dominant evidence / why scaffold transfer
  is the wrong mechanism / what search signal helps). Build the role-diverse mechanism slate
  RDA1–RDA4 (§ 4) with a NIM-free **fake-diversity** structural test + a positive control. Mine
  the underused role-diverse coordination/consensus/synthesis stack HONESTLY: bridge to the
  W41/W42 synthesis decisions (`select_role_invariance_decision`,
  `select_integrated_synthesis_decision`) for abstain-on-divergence, and to the
  `executor_grounded_patcher_v1` digest for counterexample elimination; the W79 substrate
  controllers (`team_consensus_controller_v14` / `consensus_fallback_controller_v25` /
  `hosted_cost_planner_v12` / `hosted_real_handoff_coordinator_v11`) are EXAMINED and the
  literal-bridge KILLED as fake-inapplicable (machine-checkable). Lane α emits
  `results/w128/cluster_protocol/cluster_protocol_v1.json`. **$0 NIM.**
* **Lane β — same-family EXPOSED hard-cluster dev bench (MAIN validation, EXPOSED dev spend
  ALLOWED).** Build the disjoint same-family EXPOSED hard-cluster dev bench (§ 5) under the
  no-leakage rule (§ 3). Three arms at MATCHED budget: plain baseline / W127 scaffold /
  role-diverse search. Apply the R1′ earn gate (§ 5). Kill the mechanism sharply if it is fake
  or weak. Lane β emits `results/w128/dev_bench/hard_cluster_dev_bench_verdict.json`.
* **Lane γ — targeted resistant probe / stronger-model gate / truth.** Re-check primary cutoffs
  (§ 8). A targeted resistant probe is earned ONLY iff T1 ∧ T2 (§ 6). If earned, run the
  smallest honest cluster-matched probe first. If not, **$0 resistant NIM**, register the cap.
  Keep the W123/W124/W125/W126/W127 caps closed unless new evidence genuinely changes them.
  Refresh graphify START + END (§ 9); land executable code, not docs only.

Branch order: α (diagnosis + mechanism, $0) → β (dev bench, dev spend) → γ (T1∧T2 → targeted
resistant probe, conditional spend; else $0).

---

## § 2 — Hard-cluster target rule (LOCKED)

* **Resistant hard-cluster target = the W127-atlas entries whose public
  `dominant_algorithm_family ∈ {graph_flow, simulation_grid}`** (the operator-named minimum) —
  ALL of which carry `scaffoldable_flag = False`. From the atlas this is the **8** problems:
  `graph_flow` = {`andor`, `balancingart`, `bigand`, `buyingjerseys`}; `simulation_grid` =
  {`brownianbears`, `chesssolitaire`, `enchantedmaze`, `spiesvsspies`} (2 of the
  simulation_grid are HIDDEN-failures — looks-right-fails-hidden). Pinned by the atlas CID +
  the W120 resistant 30-slice CID `01bf9ef8…`.
* The atlas reference cross-check (`reference_family_signal`) is OFFLINE-ONLY and NEVER
  model-facing (the mechanism module does not import `classify_reference_family_v1`).

---

## § 3 — No-leakage + teacher/target-disjointness rule (LOCKED, enforced in code)

1. **NEVER** expose a target's accepted solution, secret input, secret answer, or validator
   internals to ANY model-facing prompt. The role-diverse mechanism's ANALYZE + IMPLEMENT
   prompts contain ONLY the target's PUBLIC statement + PUBLIC samples.
2. The model's DERIVED counterexamples are PUBLIC-signal-derived (generated from the statement
   + samples only); they are used as a candidate-AGREEMENT oracle (RDA2) and, where the model
   supplies a predicted-expected, as a trust axis (RDA4) — never as ground truth from secret.
3. The SCAFFOLD arm reuses the W127 G1/G2/G3 line; teacher/target disjointness is mechanically
   enforced by short-name (a teacher whose short-name equals the target is dropped), plus the
   W127 near-duplicate retrieval guard. Teacher corpus = all EXPOSED problems whose short-name
   is NOT a hard dev target; `teacher_corpus_cid` + `hard_dev_target_cid` pinned.
4. Every committed/pool candidate passes the W126/W127 provenance-aware leakage guard
   (`SynthesisLeakageGuardV1` + the contiguous-block `reproduces_accepted_block_v1` tripwire;
   the accepted text is a TRIPWIRE, never an input). A run failing any guard is
   `RDA_RUN_INVALID_LEAKAGE`, dropped, never counted as a win. Positive control preserved (a
   planted accepted solution is caught — tested).
5. If ANY leakage check fails on the EARNING set ⇒ the earn is INVALID and the lane is killed
   honestly; resistant spend is NOT earned.

---

## § 4 — Role-diverse mechanism slate (LOCKED before results)

New module `coordpy.role_diverse_algorithm_search_v1` (explicit-import only). One target = **K=5
model calls = 1 ANALYZE + 4 IMPLEMENT** (matched to baseline A1's K=5; the analyze call COSTS
one generation, so the mechanism implements 4 enforced-distinct sketches vs baseline's 5 i.i.d.
— any win is therefore at a generation DISADVANTAGE, making it more convincing). All RDA1–RDA4
SELECTION variants are computed NIM-free over the SAME 5 generations (so the slate costs NIM
once and each component's load-bearingness is measurable).

* **RDA1 — role-diverse algorithm-sketch search.** ANALYZE → SPEC + 2–4 INVARIANTS + COMPLEXITY
  + `n_sketches` MATERIALLY-DIFFERENT sketches + 3–6 DERIVED counterexamples; one IMPLEMENT per
  sketch (low temp; diversity from the sketch, not sampling). Select = first public-sample
  passer (naive).
* **RDA2 — counterexample-guided elimination.** Run public-survivors on the DERIVED
  counterexamples (executor-grounded via `parse_failure_digest_v1`); group by behaviour
  signature; commit a representative of the MAJORITY (agreement) class, eliminating outliers.
* **RDA3 — role-invariant abstain (REAL bridge to `role_invariant_synthesis`).** Feed the
  survivor agreement to `select_role_invariance_decision`; `INVARIANCE_DIVERGED_ABSTAINED` with
  no strict-majority quorum ⇒ **ABSTAIN** rather than commit a coin-flip.
* **RDA4 — two-axis consensus + fallback (REAL bridge to `integrated_synthesis`).** Combine the
  producer axis (impl-consensus) with a trust axis (candidate matching the model's
  predicted-expected on the derived cases) via `select_integrated_synthesis_decision`; commit
  on both-axes agreement, ABSTAIN on divergence, producer-only fallback when no
  predicted-expected exists. **RDA4 is the FULL mechanism and the earn arm.**

**Fake-diversity kill (NIM-free, machine-checkable — the W125 `MechanismFingerprintV1`
analogue).** A run is `diversity_real = REAL` iff: ≥2 sketches AND max pairwise sketch-outline
Jaccard < 0.80 AND ≥2 distinct AST-normalized implementations AND ≥1 derived counterexample is
NEW (⊄ public samples) AND invariants non-empty. Else `FAKE_DIVERSE`. `fake_diversity_control_v1`
(identical sketches + samples-as-counterexamples) MUST classify `FAKE_DIVERSE` (positive
control). **A win on a `FAKE_DIVERSE` run does NOT count as a mechanism win** (R1c).

**Honest mining record (RDA4 — which candidate died and why).** The W79 substrate controllers
are EXAMINED by `examine_substrate_controller_applicability_v1`: their decision logic is
parameterised over substrate-trust quantities (`replacement_then_restart_after_long_delay` /
trust floors), NOT code-candidate consensus ⇒ a literal bridge would be fake-different ⇒ the
literal-controller bridge is KILLED; the consensus/abstain role is filled by the W41/W42
synthesis decisions (genuinely aligned). Recorded in the cluster-protocol JSON.

**Kill rules (honest):** a slate member / the whole mechanism is killed if (i) the role
artifacts collapse to fake diversity (detector); (ii) the role-diverse arm is no better than
plain on the dev bench (search = prompt decoration); (iii) the only gains are trivial parse/IO
fixes; (iv) a win depends on same-problem leakage.

---

## § 5 — EXPOSED hard-cluster dev-bench earn rule (LOCKED; R1′)

Dev set = the EXPOSED problems (`/tmp/w121_icpc`, the W121 family) whose public family ∈
`NON_SCAFFOLDABLE_FAMILIES = {graph_flow, simulation_grid, adhoc_math, search_enumeration,
greedy_scheduling}`, with **`simulation_grid` as the PRIORITY named-present cluster**. From the
NIM-free census: `graph_flow` EXPOSED supply = **0** (registered `graph_flow` exposed-supply cap
⇒ graph_flow is resistant-probe-only, never exposed-dev-validated); `simulation_grid` = 4,
`adhoc_math` = 6, `greedy_scheduling` = 1 ⇒ **n ≈ 11** hard dev targets. Pinned by
`hard_dev_target_cid`. The role-diverse arm is TEACHER-FREE (searches from the statement);
teacher disjointness (§ 3) applies only to the SCAFFOLD reference arm.

Three arms, MATCHED budget, `pass` graded on the official secret cases (public-sample prescreen
→ secret only for sample-passers), same model `meta/llama-4-maverick-17b-128e-instruct`:

* **`plain`** baseline — K=5 i.i.d. plain generations at T=0.7 (== W120/W121/W127 A1).
* **`scaffold`** — W127 G2→G3, K=5 at T=0.7 (reference; expected weak on non-scaffoldable
  families — that is the point).
* **`rda`** — role-diverse search: 5 calls = 1 ANALYZE (T=0.5) + 4 IMPLEMENT (T=0.2). The earn
  arm = **RDA4 committed**. Diagnostics: `rda_pool_pass` (any impl passes secret = generation
  ceiling), per-variant committed/abstain (RDA1–RDA4 load-bearingness), diversity classify.

Earn metrics: `rda_unique_solves` = #(RDA4-committed PASS ∧ baseline FAIL);
`rda_regressions` = #(baseline PASS ∧ RDA4-committed FAIL); `net_rda_gain` = unique − regr;
`net_vs_scaffold` = net_rda_gain − net_scaffold_gain.

**R1′ EARNED iff ALL hold:**
* **R1a′** `net_rda_gain ≥ DEV_MIN_NET_GAIN` (LOCKED = **+2** net), AND
* **R1b′** the unique solves span ≥ 2 hard families OR include ≥ 1 `simulation_grid` solve
  (the named present cluster), AND
* **R1c′** every unique-solve RDA run is `diversity_real = REAL` (mechanism not fake) AND
  leakage-clean, AND
* **R1d′** `gain_is_nontrivial` (unique solves are not exclusively parse/IO fixes), AND
* **R1e′** `net_rda_gain ≥ net_scaffold_gain` (the role-diverse line beats the scaffold line on
  the hard clusters — else it is not the mechanism that matters).

If R1′ FAILS ⇒ the role-diverse line is **not earned**; register
`W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP`; **$0 resistant NIM**. A close edge is NOT
sufficient. If the dev bench shows real value, record which clusters / roles / intermediate
artifacts were load-bearing (which RDA variant carried each win; was RDA2/RDA3/RDA4 ever
decisive vs RDA1).

Dev-bench budget ceiling (LOCKED): ≤ **~200** NIM calls (≈ 11 targets × 3 arms × ~5 + canary).
Canary first (harness validation), then the dev bench.

---

## § 6 — Targeted resistant probe earn rule (LOCKED; T1 ∧ T2)

Fresh resistant hosted spend is earned ONLY iff BOTH:
* **T1** — Lane β: the role-diverse line shows REAL dev-bench value on the hard clusters
  (R1′ = EARNED), AND
* **T2** — the resistant atlas identifies a cluster-matched subset where the mechanism is
  specifically relevant: a hard family (`graph_flow` or `simulation_grid`) that intersects the
  dev-earned families (the EXPOSED-earned family ∩ the resistant hard target families ≠ ∅).

If T1 ∧ T2:
* Run the **smallest honest cluster-matched targeted resistant probe** first — the resistant
  hard problems in the earned family (e.g. the 4 `simulation_grid` resistant problems if
  simulation_grid is the earned cluster), NOT a full 30-rerun, ≤ 1 seed, the role-diverse RDA4
  mechanism (5 calls/problem). Grade committed + pool on the official secret cases; compare
  against the old 11-generation pool (0 on these).
* Probe budget ceiling (LOCKED): cluster-matched resistant subset × 5 calls (+ canary), ≤ **~45**
  NIM calls. No full-30 rerun by default.
* `targeted_new_solves` = #(cluster-subset problems the RDA4 mechanism solves on secret that the
  old pool never solved, leakage-clean). If `≥ 1` ⇒ define whether a broader resistant pilot is
  earned (a separate, explicitly-flagged decision; NOT auto-run in W128). If `= 0` ⇒ the
  resistant field resists role-diverse search too; register the cap; no broader pilot.

If T1 ∧ T2 do NOT both hold ⇒ **$0 additional resistant NIM**; register the exact blocker. No
new n=30 seed-chasing. No stronger-model spend unless § 8 opens. No 405B. No reopening MBPP+ V2
/ frozen cross-modal / the closed Llama-3.1 rescue / APPS main-lane NIM. No dirty exposed
benchmark sold as a frontier win. A close blip, same-problem leak, or one-trick parse fix is NOT
a win.

---

## § 7 — Exposed-control earn / no-earn rule (LOCKED)

The matched exposed-frontier *control* pilot (W121-style; distinct from the EXPOSED *dev bench*
which IS authorized by § 5) is downstream and NOT automatic. Buy it ONLY if a targeted
resistant probe is RUN AND produces a real interpretation-changing result that an exposed
control would resolve (mechanism-vs-exposure). If the probe is not earned/not run, or is a clean
negative ⇒ exposed control NOT earned and NOT bought (resistant-first).

---

## § 8 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID `258b6ed7`,
invariant W114→W127). Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B, DeepSeek-V4-pro,
Mistral-Small-4-119B-2603, GLM-5, and any newly reachable same-budget-comparable model. A model
SUPERSEDES Maverick as the hosted target ONLY if it becomes primary-KNOWN (disclosed cutoff) AND
certifiable on the matched ICPC family. Standing prior: **{KNOWN:1 (Maverick, Aug-2024),
UNKNOWN:4}** ⇒ Maverick is the only certifiable hosted target. No 405B run unless reachability
changes and a pre-committed gate clears. Emit
`results/w128/stronger_model_gate/gate_recheck_v1.json`.

---

## § 9 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `0b323e6`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `resistant_capability_atlas_v1`,
  `family_scaffold_generation_v1`, `multi_agent_substrate_coordinator_v15`,
  `team_consensus_controller_v14`, `consensus_fallback_controller_v25`, `integrated_synthesis`,
  `role_invariant_synthesis`, `hosted_cost_planner_v12`, `hosted_real_handoff_coordinator_v11`.
* `graphify path family_scaffold_generation_v1 multi_agent_substrate_coordinator_v15` +
  `graphify path role_diverse_algorithm_search_v1 role_invariant_synthesis` (END should be
  1-hop `imports_from`); `graphify affected role_diverse_algorithm_search_v1`. `graphify query`
  only as a secondary claim-surface finder.
* The new module `role_diverse_algorithm_search_v1` must create the FIRST semantic bridge
  between the role-diverse synthesis stack (communities 0/35) and the ICPC resistant-code path
  (communities 174/329); the END graph must show the new module edges (degree + 1-hop paths).

---

## § 10 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless the targeted
  resistant probe earns AND a (separately-defined) broader pilot clears the +5.00pp clean-
  superiority bar. W128 retires none unless the JSON says so.
* On a fake/weak mechanism (Lane α detector fires on the slate, or R1′ fail): register
  `W128-L-ROLE-DIVERSE-HARD-CLUSTER-DEV-BENCH-CAP` — a role-diverse algorithm-search line does
  not beat plain hosted generation on held-out EXPOSED hard-cluster problems by a real margin ⇒
  the mechanism is not validated ⇒ no resistant spend earned. (If the mechanism is REAL but
  merely does not earn, say so distinctly from "fake".)
* Always register `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP` — the EXPOSED hard-cluster supply has
  ZERO `graph_flow` problems ⇒ graph_flow is resistant-probe-only and cannot be exposed-dev-
  validated at this corpus.
* On T1 ∧ T2 with `targeted_new_solves = 0`: register
  `W128-L-RESISTANT-ROLE-DIVERSE-SEARCH-CAP` — a dev-validated role-diverse search line, run
  fresh on the cluster-matched resistant subset, creates ZERO new resistant solves ⇒ the
  resistant capability gap is not closed by role-diverse search at this model scale (the
  search-lever sibling of the W123→W127 cap taxonomy).
* On T1 ∧ T2 with `targeted_new_solves ≥ 1`: register the new-solve evidence + the broader-pilot
  decision (NOT a retirement by itself).
* Named claims filled ONLY from the emitted verdict JSON.

---

## § 11 — W129 branch logic (pre-committed)

* If the mechanism is FAKE (Lane α detector fires) ⇒ W129 = the role-diverse search idea is not
  realisable at this scale; accept the bounded ceiling + caps; fire only on a code-COMPETENT
  local model / a primary-KNOWN reachable stronger-than-Maverick model / a genuinely different
  mechanism axis.
* If the mechanism is REAL but R1′ fails on the EXPOSED hard-cluster dev bench ⇒ W129 = accept
  the bounded ceiling + the registered hard-cluster dev-bench cap; the honest remaining lever is
  a stronger/code-competent model, not more role-search prompt engineering.
* If R1′ holds but T2 fails (real value but no matched resistant cluster) ⇒ W129 = the
  role-diverse line is a real same-family mechanism but the resistant field's missing
  capabilities are not search-addressable in the matched cluster; pursue the remaining
  non-scaffoldable clusters / a different mechanism.
* If T1 ∧ T2 and `targeted_new_solves = 0` ⇒ W129 = accept the resistant role-diverse-search
  cap; the honest remaining lever is a stronger/code-competent model.
* If T1 ∧ T2 and `targeted_new_solves ≥ 1` ⇒ W129 = define + (operator-greenlit) run the broader
  cluster-matched resistant pilot and carry the verdict (retire iff a clean +5.00pp multi-seed
  same-budget margin).
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
