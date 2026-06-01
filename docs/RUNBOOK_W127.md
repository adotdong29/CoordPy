# RUNBOOK W127 — Resistant capability atlas + family-specific algorithm-scaffold generation + targeted resistant fresh-generation probe only if earned

**Status: LOCKED PRE-REGISTRATION.** Written BEFORE the capability atlas is run, BEFORE
any scaffold-generation result is interpreted, and BEFORE any NIM call. Fill
`docs/RESULTS_W127_*` ONLY from emitted verdict JSON (the "never pre-write results"
discipline — `feedback_never_prewrite_results_before_data`). The pre-committed code rules
below are the branch authority, not any prior or hope.

`ultracode` OFF. W127 is a bounded capability-research / targeted-fresh-generation
milestone, not a repo-wide dynamic-workflow job. It turns ON only if the work unexpectedly
expands into a genuine dynamic-workflow problem (multiple scaffold mechanisms all earn live
runs at once / repo-wide scaffold-controller integration / broad multi-surface external
verification at once) — and only after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests): `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI publish, `coordpy/__init__.py`
untouched, advanced work explicit-import only.

---

## § 0 — Why W127 is NOT another lap

W120–W126 closed every **$0** in-repo lever against the resistant ICPC field:

| milestone | lever | result |
|---|---|---|
| W120/W121/W122/W123 | battlefield | resistant +0.00 / exposed +3.33 FAIL; n=30 unresolvable; ≥100-per-field supply-UNREACHABLE |
| W124 | local transformer encoder | distilgpt2 hidden state adds nothing; no code-competent local model |
| W125 | $0 controller **re-routing** | `blind_selection_headroom = 0`; pool-union 8/30 ⇒ GENERATION-CAPPED |
| W126 | $0 deterministic **synthesis** | 894 leakage-clean candidates, **oracle ceiling 0/22** ⇒ SYNTHESIS-CAPPED; the 22 are **capability failures** |

W126's sharp finding: the 22 uniformly-unsolved resistant problems have an oracle ceiling of
**0** — there is no headroom to miss; deterministic recombination/repair/consensus over
capability-failed generations cannot manufacture a correct algorithm at $0. The remaining
honest lever is therefore NOT more deterministic post-processing — it is **fresh algorithmic
trajectory generation informed by a real diagnosis of what capability is missing.**

W127 is explicitly **operator-greenlit** for (a) a capability atlas, (b) a family-specific
algorithm-scaffold generation line validated on a disjoint same-family **EXPOSED development
bench** (development spend ALLOWED for mechanism validation), and (c) a tightly-scoped
**targeted resistant fresh-generation probe** *only if the mechanism actually earns it*.
W127 is NOT another battlefield pivot, NOT another reranking lap, NOT another deterministic
recombination memo. Bounded-context / compaction / summarization / "cram less / truncate
better" remain anti-patterns, NOT the frontier path.

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α — resistant capability atlas (MAIN diagnosis, NIM-free).** Reconstruct the 22
  uniformly-unsolved resistant problems from the W126 grade cache + the W120 resistant
  30-slice; build the machine-checkable capability atlas (§ 2). Cluster the 22 by algorithm
  family; quantify concentration; identify the top capability clusters. Lane α is NOT
  complete if it only says "capability failure" — it must say WHICH capabilities, HOW MANY
  problems per capability, and which are plausibly scaffoldable.
* **Lane β — family-specific scaffold generation line (MAIN mechanism, EXPOSED dev spend
  ALLOWED).** Build the scaffold slate G1/G2/G3/G4 (§ 4) under the no-leakage rule (§ 3);
  validate it on a disjoint same-family EXPOSED dev bench (§ 5). EXPOSED-family development
  spend is allowed for mechanism validation only. Apply the dev-bench earn rule (§ 5: R1).
  Kill fake mechanisms honestly.
* **Lane γ — targeted resistant probe / stronger-model gate / truth.** Re-check primary
  cutoffs (§ 8). Fresh resistant hosted spend is earned ONLY iff R1 ∧ R2 (§ 6). If earned,
  run the smallest honest cluster-matched targeted probe first. If not, **$0 resistant NIM**,
  register the blocker. Keep the W123/W124/W125/W126 caps closed unless new evidence
  genuinely changes them. Refresh graphify START + END (§ 9); land executable code, not docs
  only.

Branch order: α (diagnosis, $0) → β (build + EXPOSED dev bench, dev spend) → γ
(R1∧R2 → targeted resistant probe, conditional spend; else $0).

---

## § 2 — Capability atlas schema (LOCKED before atlas results)

New module `coordpy.resistant_capability_atlas_v1` (explicit-import only). For EACH of the
22 uniformly-unsolved resistant problems it emits a `CapabilityAtlasEntryV1` with EXACTLY
these fields (the schema is locked here; the VALUES come only from the emitted JSON):

**Hard, re-executable signals (from the grade cache + re-execution; not heuristic):**
1. `problem_id`, `short_name`, `surface`, `contest_date`
2. `n_samples`, `n_secret`
3. `failure_visibility` ∈ {`visible`, `hidden`} — `hidden` iff ≥1 generation passes ALL
   public samples but fails secret; else `visible`.
4. `n_generations`, `n_distinct_codes`, `n_distinct_digests`
5. `digest_distribution` — counts over {`ok`, `wrong_answer`, `timeout`, `runtime_error`,
   `parse_error`} across the (up to) 11 generations (re-derived via
   `executor_grounded_patcher_v1.parse_failure_digest_v1` on PUBLIC samples only).
6. `best_sample_pass_frac` — max over generations of (public-sample pass count / total).

**Soft, transparent, evidence-recorded heuristic layer (machine-checkable = deterministic +
auditable; NOT ground truth):**
7. `dominant_algorithm_family` ∈ the LOCKED taxonomy (below), = argmax of a deterministic
   lexicon+code-signal score over PUBLIC inputs only (statement + samples + the model's own
   generations). Ties broken by the taxonomy order.
8. `family_scores` — the full per-family score vector (auditable).
9. `family_evidence` — the exact signal hits that drove the label.
10. `likely_missing_technique` — derived deterministically from `dominant_algorithm_family`
    + failure pattern.
11. `complexity_mismatch_evidence` — `{max_constraint_seen, any_timeout, naive_signal}`
    (large stated constraint + TLE digest + brute-force code signal).
12. `parsing_impl_mismatch_evidence` — `{any_parse_error, sample_shape_mismatch}`.
13. `teacher_family_coverage` — # EXPOSED teacher problems whose accepted solution maps to
    the SAME `dominant_algorithm_family` (same classifier on teacher reference code).
14. `scaffoldable_flag` + `scaffoldable_reasons` — see the LOCKED rule below.

**Independent analyst cross-check (offline only; NEVER model-facing):**
15. `reference_family_signal` — the algorithm family inferred from the resistant target's
    OWN accepted-solution structure (imports / filenames / idioms), computed ONLY to report
    `atlas_label_agreement` (a validation metric of the public-signal label). This field and
    its source are NEVER passed to any G1/G2/G3 generation path (enforced: the scaffold
    module does not import the cross-check function).

**LOCKED algorithm-family taxonomy** (the SAME taxonomy is used by Lane β's retriever and the
"spans ≥ 2 capability families" earn metric):
`graph_flow`, `dp_optimization`, `geometry`, `number_theory_math`, `string_processing`,
`greedy_scheduling`, `simulation_grid`, `search_enumeration`, `data_structure`, `adhoc_math`.

**LOCKED `scaffoldable_flag` rule** (a transparent heuristic, not a guarantee — the dev bench
EMPIRICALLY tests scaffold value):
`scaffoldable_flag = (teacher_family_coverage >= 2) AND (dominant_algorithm_family ∈
SCAFFOLDABLE_FAMILIES)` where `SCAFFOLDABLE_FAMILIES = {graph_flow, dp_optimization,
geometry, number_theory_math, string_processing, data_structure, greedy_scheduling}` (the
families with a well-defined reusable skeleton; `adhoc_math`, `search_enumeration`,
`simulation_grid` are NOT auto-scaffoldable because they lack a transferable algorithmic
skeleton). `scaffoldable_reasons` records each clause.

**Atlas clustering / concentration outputs (LOCKED):**
`cluster_counts` (problems per family), `top_clusters` (the families sorted by count),
`dominant_cluster` (top-1 family by count; ties → taxonomy order),
`concentration_top2_frac` (fraction of the 22 in the top-2 families),
`scaffoldable_count`, `scaffoldable_by_family`.

Lane α emits `results/w127/atlas/capability_atlas_v1.json`. **$0 NIM.**

---

## § 3 — No-leakage + teacher/target-disjointness rule (LOCKED, enforced in code)

1. **NEVER** expose a resistant (or EXPOSED dev) **target's** accepted solution, secret
   input, secret answer, or validator internals to ANY model-facing prompt or scaffold
   generator. The scaffold generator opens neither the target's `submissions/` nor its
   `data/secret/`.
2. EXPOSED-side accepted solutions are usable ONLY as **FAMILY-LEVEL teacher material**
   (G1). They are normalized into structural skeletons (literals/identifiers stripped) before
   they may inform any prompt.
3. **Teacher/target disjointness is mechanically enforced** by problem short-name: a teacher
   solution whose problem short-name equals the (dev or resistant) target's short-name is
   dropped from the teacher corpus for that target. The corpus identity is pinned by a
   `teacher_corpus_cid`.
4. **Retrieval-leakage guard:** for every retrieved scaffold used on a target, assert
   (a) source problem ≠ target problem, and (b) the scaffold-vs-target statement n-gram
   overlap is below `MAX_SCAFFOLD_TARGET_OVERLAP` (a near-duplicate teacher is dropped).
5. Every generated candidate passes the W126 `SynthesisLeakageGuardV1` (provenance-aware:
   a secret byte-run already present in PUBLIC provenance is a coincidence, not an
   injection; a planted secret ABSENT from provenance still bites — verified positive
   control). A scaffold or candidate failing any guard ⇒ that target's scaffold result is
   `SCAFFOLD_INVALID_LEAKAGE`, dropped, and never counted as a win.
6. If ANY leakage check fails on the EARNING set ⇒ the earn is INVALID and the lane is killed
   honestly; resistant spend is NOT earned.

---

## § 4 — Family-specific scaffold-generation slate (LOCKED before results)

New module `coordpy.family_scaffold_generation_v1` (explicit-import only). It wires the
EXPOSED teacher corpus + the LOCKED family taxonomy + the executor digest + (optionally) the
policy arsenal onto a FRESH-generation prompt that calls the hosted model. This is fresh
trajectory creation, NOT reranking and NOT recombining dead resistant outputs.

* **G1 — algorithm-family scaffold library** (`build_scaffold_library_v1`). Normalize each
  EXPOSED accepted `.py` into a reusable `AlgorithmScaffoldV1`: classify its family (LOCKED
  taxonomy, on the reference code), strip problem-specific literals/identifiers down to a
  STRUCTURAL skeleton (control-flow + stdlib-idiom outline: I/O shape, core loops/recursion,
  data structures used, key library calls), and record `family`, `skeleton`, `idioms`,
  `source_problem`, `source_sha`. The library is keyed by family.
* **G2 — scaffold retriever** (`retrieve_scaffolds_v1`). Given a target's PUBLIC signals
  (statement family-classification + constraint signals + first-attempt failure digest),
  retrieve top-`R` family-level scaffolds from G1 by (family match) → (idiom/constraint
  affinity) → (skeleton diversity), enforcing § 3 disjointness + the retrieval-leakage guard.
  Retrieval is FAMILY-level, never same-problem.
* **G3 — scaffolded fresh-generation controller** (`scaffolded_generate_v1`). Build a prompt
  = target statement + public samples + the retrieved STRUCTURAL skeleton(s) (as an
  "approach outline", explicitly marked as a template from OTHER problems, never a solution)
  + (optionally) the typed first-attempt failure digest. Call the hosted model for K fresh
  candidates. This is NEW generation conditioned on a family scaffold.
* **G4 — constrained scaffold policy** (`scaffold_action_policy_v1`, CONDITIONAL). Only if
  the dev data warrants it (enough labelled scaffold-action events): use
  `constrained_policy_optimisation_v1` / `learned_economics_controller_v1` to choose among
  {scaffold-family A, scaffold-family B, abstain, plain} per target. If the labelled corpus
  is too small (W124 precedent: chance at n≈14) ⇒ registered `NOT_WARRANTED`, G4 is a
  deterministic family-match heuristic, no learned policy claimed.

**Kill rules (honest):** a candidate slate member is KILLED if (i) G3 with the scaffold is
no better than G3 without it on the dev bench (scaffold = prompt decoration); (ii) retrieval
leaks or collapses to same-problem memorization (§ 3); (iii) the only gains are on trivial
parsing fixes (said sharply, counted separately).

---

## § 5 — EXPOSED-family development-bench earn rule (LOCKED; R1)

Build a disjoint same-family EXPOSED dev bench from `/tmp/w121_icpc` (38 gradeable EXPOSED
problems with accepted `.py`; the W121 family — RMRC 2021 / ECNA 2022-23 / RMRC 2022-23 /
ECNA 2023-24):

* Deterministic split by short-name hash into **TEACHER problems** (their accepted `.py` →
  G1 library) and **DEV-TARGET problems** (held out; graded; accepted solution NEVER shown
  to the generator). Disjointness asserted; `teacher_corpus_cid` + `dev_target_cid` pinned.
* **Baseline arm `A_dev`** — plain hosted generation: K=5 candidates at T=0.7, `pass@5` on
  the official secret cases (identical discipline to W120/W121 A1; same model
  `meta/llama-4-maverick-17b-128e-instruct`, same grader, same K).
* **Scaffold arm `G_dev`** — G2-retrieved family scaffold → G3 fresh generation: K=5
  candidates at T=0.7, `pass@5` on secret. **Same K=5 budget** (no extra generations beyond
  the at-most-one first-attempt the digest needs, accounted explicitly).
* Per dev target record `baseline_pass`, `scaffold_pass`, family, retrieval provenance,
  leakage-clean.

Earn metrics (on the dev set):
* `scaffold_unique_solves` = # targets `baseline_FAIL ∧ scaffold_PASS`.
* `scaffold_regressions` = # targets `baseline_PASS ∧ scaffold_FAIL`.
* `net_scaffold_gain` = `scaffold_total_pass − baseline_total_pass` (= unique_solves −
  regressions).
* `gain_distinct_families` = # distinct LOCKED families among the unique-solve targets.
* `gain_is_nontrivial` = NOT all unique solves are pure parsing/IO fixes (the unique-solve
  target's failure family is not exclusively `parse_error`/IO).

**R1 EARNED iff ALL hold:**
* **R1a** `net_scaffold_gain ≥ DEV_MIN_NET_GAIN` (LOCKED = **+2** problems net — a real
  margin, not a single fluke), AND
* **R1b** `gain_distinct_families ≥ 2` (spans ≥ 2 capability families, not a single narrow
  trick), AND
* **R1c** every unique-solve scaffold is leakage-clean + disjoint (§ 3), AND
* **R1d** `gain_is_nontrivial` (not only trivial parsing fixes).

If R1 FAILS (net ≤ +1, or single-family, or leaking, or trivial-only) ⇒ the scaffold line is
**not earned**; register the cap; **$0 resistant NIM**. A close edge is NOT sufficient.

Dev-bench budget ceiling (LOCKED): ≤ ~260 NIM calls total (≈ ≤ 18 dev targets × ≤ 2 arms ×
K=5 + canary). Canary first (harness validation), then the dev bench.

Lane β emits `results/w127/dev_bench/exposed_dev_bench_verdict.json`.

---

## § 6 — Targeted resistant fresh-generation probe earn rule (LOCKED; R1 ∧ R2)

Fresh resistant hosted spend is earned ONLY iff BOTH:
* **R1** — Lane β shows real fresh-generation value on the held-out EXPOSED dev bench (§ 5),
  AND
* **R2** — the capability atlas (§ 2) identifies a dominant resistant cluster (a family with
  ≥ `R2_MIN_CLUSTER` = **3** of the 22, `scaffoldable_flag = True`) that the scaffold line is
  specifically targeting — i.e. the atlas `dominant_cluster` (or any top-2 scaffoldable
  cluster) ∩ the dev-bench earned families ≠ ∅.

If R1 ∧ R2:
* Run the **smallest honest targeted resistant probe** first — the resistant problems in the
  earned scaffoldable cluster (the cluster-matched subset), NOT a full 30-rerun, ≤ 1 seed,
  K=5. Generate G3 scaffolded fresh candidates with NIM; grade on the official secret cases;
  compare against the old pool (which was 0/those problems).
* Probe budget ceiling (LOCKED): the cluster-matched resistant subset × K=5 (+ ≤ 1
  first-attempt each), ≤ ~80 NIM calls. No full-30 rerun by default.
* `targeted_new_solves` = # cluster-subset problems the scaffold line solves on secret that
  the entire old 11-generation pool did NOT. If `targeted_new_solves ≥ 1` (a REAL new solve,
  leakage-clean, on a problem the old pool never solved) ⇒ define whether a broader resistant
  pilot is earned (a separate, explicitly-flagged decision; NOT auto-run in W127). If
  `targeted_new_solves = 0` ⇒ the resistant field resists fresh scaffolded generation too;
  register the cap; no broader pilot.

If R1 ∧ R2 do NOT both hold ⇒ **$0 additional resistant NIM**; register the exact blocker
honestly. No new n=30 seed-chasing. No stronger-model spend unless § 8 opens. No 405B. No
reopening MBPP+ V2 / frozen cross-modal / the closed Llama-3.1 rescue / APPS main-lane NIM.
No dirty exposed benchmark sold as a frontier win. A close blip, same-problem leakage, or a
one-trick parsing fix is NOT a win.

---

## § 7 — Exposed-control earn / no-earn rule (LOCKED)

The matched exposed-frontier *control* pilot (the W121-style matched control, distinct from
the EXPOSED *dev bench* which IS authorized by § 5) is downstream and NOT automatic. Buy it
ONLY if a targeted resistant probe is RUN AND produces a real interpretation-changing result
that an exposed control would resolve (mechanism-vs-exposure). If the targeted probe is not
earned or not run ⇒ exposed control NOT earned and NOT bought (resistant-first).

---

## § 8 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID
`258b6ed7`, invariant W114→W126). Re-check PRIMARY sources for: Maverick, Qwen3-Coder-480B,
DeepSeek-V4-pro, Mistral-Small-4-119B-2603, GLM-5, and any newly reachable
same-budget-comparable model. A model SUPERSEDES Maverick as the hosted target ONLY if it
becomes primary-KNOWN (disclosed cutoff) AND certifiable on the matched ICPC family (a KNOWN
cutoff ≤ the resistant instrument frontier). Standing prior: **{KNOWN:1 (Maverick, Aug-2024),
UNKNOWN:4}** ⇒ Maverick is the only certifiable hosted target. No 405B run unless reachability
changes and a pre-committed gate clears. Emit
`results/w127/stronger_model_gate/gate_recheck_v1.json`.

---

## § 9 — graphify deliverables (LOCKED)

* Refresh `graphify update .` at START (built from HEAD `596ddf9`) and END (record END HEAD).
* `graphify explain` on the mined arsenal: `family_adapted_repair_synthesis_v1`,
  `controller_native_code_mechanism_v1`, `executor_grounded_patcher_v1`,
  `adversarial_consensus_repair_v1`, `compose_repair_integrity_pipeline_v1`,
  `constrained_policy_optimisation_v1`.
* `graphify path family_adapted_repair_synthesis_v1 adversarial_consensus_repair_v1` +
  `graphify affected family_adapted_repair_synthesis_v1`; `graphify query` only as a
  secondary claim-surface finder.
* The new W127 modules must create the first semantic bridge between the EXPOSED teacher
  corpus + the family taxonomy + the scaffold-generation path; the END graph must show the
  new module edges.

---

## § 10 — Carry-forward registration (LOCKED shape; filled ONLY from JSON)

* **W89 (+5.56) + W105 (+7.00)** remain the only two confirmed retirements unless the
  targeted resistant probe earns AND a (separately-defined) broader pilot clears the +5.00pp
  clean-superiority bar. W127 retires none unless the JSON says so.
* On a NOT-EARNED dev bench (R1 fail): register `W127-L-EXPOSED-SCAFFOLD-DEV-BENCH-CAP` — the
  family scaffold line does not beat plain hosted generation on held-out EXPOSED-family
  problems by a real multi-family margin ⇒ the fresh-generation mechanism is not validated ⇒
  no resistant spend earned.
* On R1 ∧ R2 with `targeted_new_solves = 0`: register
  `W127-L-RESISTANT-SCAFFOLD-FRESH-GEN-CAP` — even a dev-validated family scaffold line, run
  fresh on the cluster-matched resistant subset, creates ZERO new resistant solves ⇒ the
  resistant capability gap is not closed by family scaffolds at this model scale.
* On R1 ∧ R2 with `targeted_new_solves ≥ 1`: register the new-solve evidence + the
  broader-pilot decision (NOT a retirement by itself).
* Named claims filled ONLY from the emitted verdict JSON.

---

## § 11 — W128 branch logic (pre-committed)

* If R1 fails (scaffold dead on EXPOSED dev) ⇒ W128 = accept the bounded HumanEval-family
  ceiling + the registered scaffold-dev cap; fire only on a code-COMPETENT local model, a
  primary-KNOWN reachable stronger-than-Maverick model, or a genuinely different mechanism
  axis.
* If R1 holds but R2 fails (scaffold real but no scaffoldable resistant cluster) ⇒ W128 =
  the scaffold line is a real same-family mechanism but the resistant field's missing
  capabilities are not scaffold-addressable; accept the bounded ceiling / pursue the
  atlas's non-scaffoldable clusters with a different mechanism.
* If R1 ∧ R2 and `targeted_new_solves = 0` ⇒ W128 = accept the resistant scaffold cap; the
  honest remaining lever is a stronger/code-competent model.
* If R1 ∧ R2 and `targeted_new_solves ≥ 1` ⇒ W128 = define + (operator-greenlit) run the
  broader cluster-matched resistant pilot and carry the verdict (retire iff a clean
  +5.00pp multi-seed same-budget margin).
* `COO-9` stays the lead path unless the evidence genuinely forces a different code-line move.
