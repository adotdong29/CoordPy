# Theorem registry — canonical status

> Canonical, single-source-of-truth registry for every named
> theorem-style claim in the Context Zero / Wevra programme.
> If this file disagrees with any other doc on the *status* of a
> claim, this file is right and the other file is stale. Last
> touched: SDK v3.5, 2026-04-26.
>
> Status vocabulary (definitions in `docs/HOW_NOT_TO_OVERSTATE.md`):
>
> - **proved** — mathematical proof or proof-by-inspection.
> - **proved-conditional** — proof depends on a stated assumption.
> - **mechanically-checked** — runtime audit on every run; soundness by inspection.
> - **empirical** — measured on a published bench from a published seed.
> - **conjectural** — stated, falsifiable; not yet proved or systematically tested.
> - **retracted** — earlier reading withdrawn; replaced by a more honest reading.

## Capsule algebra (W3-7 .. W3-16)

| Claim   | One-line description                                                  | Status                | Code/proof anchor                                                                 |
| ------- | --------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------- |
| W3-7    | CID is parent-permutation invariant                                    | proved                | `_capsule_cid` sorts parents; test `test_c1_parent_order_is_canonicalised`        |
| W3-8    | Budget monotonicity of admissibility                                   | proved                | `CapsuleLedger.admit`; `_capsule_cid` byte check                                  |
| W3-9    | Ledger DAG is acyclic; append order is topo                            | proved                | `CapsuleLedger.admit` parental check; `ancestors_of` BFS terminates               |
| W3-10   | Chain tamper-evidence under SHA-256 second-preimage                    | proved-conditional    | `_chain_step`, `verify_chain`; `test_c5_hash_chain_detects_tamper`                |
| W3-11   | Capsule subsumption — partial (4 cases)                                | proved (partial)      | adapters in `vision_mvp/wevra/capsule.py`                                          |
| W3-12   | Capsule view is a faithful header projection                            | proved                | `as_header_dict`, `render_view`                                                  |
| W3-13   | DAG height ≤ 4 on canonical run pattern                                  | proved                | inspection of `build_report_ledger`                                              |
| W3-14   | Per-capsule budget locality — negative                                  | proved (negative)     | `test_phase47_cohort_subsumption.py::test_per_capsule_budget_cannot_bound_total_count` |
| W3-15   | Cohort lift                                                              | proved                | `capsule_from_cohort`; cohort tests                                              |
| W3-16   | Relational limitation post-cohort-lifting — negative                    | proved (negative)     | cohort relational tests                                                          |

## Decoder frontier (W3-17 .. W3-31)

| Claim   | Description                                                                | Status                | Anchor                                                                                                  |
| ------- | -------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------- |
| W3-17   | Admission-only rules ≤ priority-decoder ceiling under ceiling-forcing      | proved-conditional    | `phase47_bundle_learning.py`                                                                            |
| W3-18   | Plurality > priority on coherent-majority bundles                          | proved-conditional    | `test_phase48_bundle_decoding.py`                                                                       |
| W3-19   | Learned bundle decoder breaks 0.200 ceiling at +15pp on $n=80$             | empirical             | `phase48_bundle_decoding.py` reproducible from default seeds                                            |
| W3-20   | Deep Sets sufficiency (capacity statement)                                 | proved                | `capsule_decoder_v2.py::_phi_capsule`                                                                   |
| W3-21   | Linear-class sign-flip asymmetry — negative                                | proved                | `phase49_symmetric_transfer.py`                                                                         |
| W3-22   | Pooled-multitask symmetric transfer at $n=80$                              | empirical             | Phase 49 result (0.350 / 0.350)                                                                         |
| W3-23   | DeepSet best-cell 0.425 at $n=80$                                          | empirical             | Phase 49 result                                                                                         |
| W3-24   | Post-search winner's-curse bias                                            | proved                | Phase 50                                                                                                 |
| W3-25   | Tail-replication / hold-out generalisation gap                             | proved-conditional    | Phase 50                                                                                                |
| W3-26   | DeepSet best-cell 0.362 at $n=320$                                         | empirical             | Phase 50 — falsifies W3-C7 strict reading                                                               |
| W3-27   | 6-family zero-shot max-penalty +0.112 at $n=320$                           | empirical             | Phase 50 — falsifies W3-C7 penalty reading                                                              |
| W3-28   | Sign-stable DeepSet zero-shot gap = 0.000 at level 0.237                   | empirical             | Phase 50 — direction-invariance, not level-matching                                                     |
| W3-29   | Bayes-divergence zero-shot risk lower bound on linear family               | proved                | Phase 50 structural argument                                                                            |
| W3-30   | Strict separation: relational decoder vs. magnitude-monoid                 | proved                | Phase 51                                                                                                |
| W3-31   | Empirical relational-decoder level-lift                                    | empirical             | Phase 51                                                                                                |

## Capsule-native execution (W3-32 .. W3-41)

| Claim                  | Description                                                                  | Status                            | Anchor                                                                          |
| ---------------------- | ---------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| W3-32                  | Lifecycle ↔ execution-state correspondence (spine kinds)                     | proved                            | `capsule_runtime.py`; `test_wevra_capsule_native.py`                            |
| W3-32-extended         | W3-32 carry-over to PATCH_PROPOSAL / TEST_VERDICT                            | proved                            | `test_wevra_capsule_native_intra_cell.py`                                       |
| W3-33                  | Content addressing at artifact creation time                                  | proved                            | `seal_and_write_artifact`; cross-validation test                               |
| W3-34                  | In-flight ↔ post-hoc CID equivalence on non-ARTIFACT spine kinds              | proved                            | `test_w3_34_spine_equivalence_holds`                                            |
| W3-35                  | Parent-CID gating is the execution contract                                   | proved                            | runtime gate raises `CapsuleLifecycleError`                                     |
| W3-36                  | Meta-artifact circularity is sharp; detached-witness corollary                | proved (negative + constructive)  | `test_meta_manifest_seals_in_secondary_ledger`                                  |
| W3-37                  | Chain-from-headers verification                                                 | proved                            | `verify_chain_from_view_dict`; tamper tests                                     |
| W3-38                  | ARTIFACT audit-time on-disk re-hash                                            | proved                            | `verify_artifacts_on_disk`                                                       |
| **W3-39 (SDK v3.3)**   | **PARSE_OUTCOME lifecycle gate + parse → patch → verdict DAG chain**          | **proved**                        | `test_wevra_capsule_native_deeper.py::ParseOutcomeLifecycleTests`               |
| **W3-40 (SDK v3.3)**   | **Lifecycle-audit soundness on L-1..L-8**                                     | **proved + mechanically-checked** | `lifecycle_audit.py`; `LifecycleAuditTests`                                     |
| **W3-41 (SDK v3.3)**   | **Deterministic-mode CID determinism on full DAG**                            | **proved + empirical**            | `DeterministicModeTests::test_w3_41_two_runs_collapse_to_identical_cids`        |
| **W3-42 (SDK v3.4)**   | **PROMPT lifecycle gate (parent = SWEEP_SPEC; idempotent on content)**         | **proved**                        | `test_wevra_capsule_native_inner_loop.py::PromptCapsuleLifecycleTests`         |
| **W3-43 (SDK v3.4)**   | **Prompt → response parent gate (LLM_RESPONSE has 1 parent = sealed PROMPT)**  | **proved**                        | `test_wevra_capsule_native_inner_loop.py::LLMResponseCapsuleLifecycleTests`     |
| **W3-44 (SDK v3.4)**   | **PARSE_OUTCOME → LLM_RESPONSE chain coordinate consistency**                  | **proved + mechanically-checked** | `lifecycle_audit.py::_check_l11`; `LifecycleAuditExtendedTests`                |
| **W3-45 (SDK v3.4)**   | **Lifecycle-audit soundness extends to L-1..L-11**                             | **proved + mechanically-checked** | `lifecycle_audit.py`; `LifecycleAuditExtendedTests::test_full_chain_audit_is_ok` |

## Team-level capsule coordination (W4-1 .. W4-3) — SDK v3.5

| Claim   | Description                                                                  | Status                            | Anchor                                                                          |
| ------- | ---------------------------------------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------- |
| **W4-1 (SDK v3.5)** | **Team-lifecycle audit soundness on T-1..T-7**                                | **proved + mechanically-checked** | `team_coord.py::audit_team_lifecycle`; `TeamLifecycleAuditTests`                |
| **W4-2 (SDK v3.5)** | **Coverage-implies-correctness on the deterministic team decoder**            | **proved-conditional**            | `team_coord.py::TeamCoordinator`; `TeamLevelCorrectnessTests::test_w4_2_*`     |
| **W4-3 (SDK v3.5)** | **Local-view limitation: per-role budget below causal-share floor cannot be rescued by any admission policy** | **proved-negative**               | `TeamLevelCorrectnessTests::test_w4_3_*`; `phase52_team_coord.run_phase52_budget_sweep` |

## Conjectures (W3-C*)

| Claim   | Description                                                              | Status                                       |
| ------- | ------------------------------------------------------------------------ | -------------------------------------------- |
| W3-C1   | General subsumption: every Phase-N bounded-context theorem subsumes      | conjectural                                  |
| W3-C2   | Capsule contract preserves all Phase-N substrate guarantees              | conjectural                                  |
| W3-C3   | Honest-falsification frontier (table-level invariants)                   | retracted by W3-15 cohort lift               |
| W3-C4 (legacy) | Earlier "decoder paradigm shift candidate at 0.400 / strict zero-shot" | **retracted** by W3-26 / W3-27       |
| **W3-C4 (new SDK v3.3)** | **PARSE_OUTCOME failure_kind distribution stable across LLM tags** | **superseded by W3-C6 (sharper synthetic reading)** |
| W3-C5 (legacy) | Relational-axis extension closes W3-16                              | conjectural                                  |
| **W3-C5 (new SDK v3.3)** | **Sub-intra-cell PROMPT/LLM_RESPONSE capsule slice closes the inner-loop boundary without breaking W3-34 spine equivalence** | **DISCHARGED by W3-42/W3-43/W3-44/W3-45 in SDK v3.4** |
| **W3-C6 (new SDK v3.4)** | **Synthetic-LLM cross-distribution PARSE_OUTCOME failure-kind TVD ≥ 0.5; strict→robust parser-mode shift on synthetic.unclosed = 1.000** | **empirical (reproducible from synthetic distribution library)** |
| W3-C6   | Plurality + auxiliary signal closes 0.200 priority-decoder ceiling       | partial — replaced by learned-decoder W3-19  |
| W3-C7 (strict) | Strict point-estimate Gate-1 ($\hat p \ge 0.400$) + zero-shot Gate-2 strict penalty ≤ 5pp | **retracted** by W3-26, W3-27   |
| W3-C8   | (deferred — relational decoder closes Gate 2 strictly)                   | conjectural                                  |
| W3-C9   | Refined paradigm-shift reading: $n=80$ point-estimate + zero-shot gap    | conjectural (candidate)                      |
| W3-C10  | Relational decoder level-ceiling                                                | conjectural                                  |
| **W4-C1 (SDK v3.5)** | **Learned per-role admission policy admits strictly fewer handoffs (12/12 seeds) and improves pooled team-decision accuracy on most train seeds (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean +0.032) over the strongest fixed admission baseline (coverage-guided) on the Phase-52 default config (K_auditor=8, noise=(0.10, 0.30, 0.05), PYTHONHASHSEED=0)** | **empirical** (budget-efficiency dominance is robust per-seed; accuracy advantage is mean-positive but not strict per-seed; reverses at higher noise — see § Cross-seed reading in `docs/RESULTS_WEVRA_TEAM_COORD.md`) |
| **W4-C2 (SDK v3.5)** | **Cohort-lifted role view closes W4-3 on a sub-class of scenarios**           | **conjectural** (Phase 53 candidate; falsifier: a scenario whose causal-share floor exceeds a single COHORT's max_parents) |
| **W4-C3 (SDK v3.5)** | **The capsule-layer admission rule subsumes the Phase-36 AdaptiveSubscriptionTable route-edit primitive** | **conjectural** (open) |

## Phase-substrate (P19, P31, P35, P36, P39..P44)

The substrate-side phase-by-phase theorems (P19-L2, P31-1..P31-5,
P35-1..P35-3, P36-*, P39-*, P40-*, P41-*, P42-*, P43-*, P44-*)
are stated and proved in their respective `RESULTS_PHASE*.md`
notes. They are inputs to W3-11 (capsule subsumption) but are
not maintained in this registry — see `vision_mvp/RESULTS_PHASE*.md`
or `PROOFS.md`.

## How to add a claim

1. Pick the lowest unused W3-* / W3-C* number in the appropriate
   section.
2. Add the row here with status, one-line description, and
   anchor.
3. Add the formal statement to `docs/CAPSULE_FORMALISM.md` (for
   W3-* in the capsule algebra / runtime axes).
4. If empirical, add a falsifiability test to
   `vision_mvp/tests/test_wevra_*.py` or a research shard.
5. If conjectural, add the falsifier (what would refute it).
6. Cross-link from `docs/RESEARCH_STATUS.md`.
