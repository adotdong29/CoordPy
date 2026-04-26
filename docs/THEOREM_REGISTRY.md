# Theorem registry — canonical status

> Canonical, single-source-of-truth registry for every named
> theorem-style claim in the Context Zero / Wevra programme.
> If this file disagrees with any other doc on the *status* of a
> claim, this file is right and the other file is stale. Last
> touched: SDK v3.8, 2026-04-26.
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

## Two-Mac distributed inference + real cross-LLM (W5-1 .. W5-3) — SDK v3.6

| Claim                | Description                                                                                                                                                                         | Status                            | Anchor                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **W5-1 (SDK v3.6)**  | **Real cross-LLM parser-boundary saturation: ``qwen3.5:35b`` (36B-MoE Q4) under strict parsing yields ``failure_kind=unclosed_new`` 10/10; ``qwen2.5:14b-32k`` yields ``ok`` 10/10; cross-model TVD on strict = 1.000; robust mode collapses TVD to 0.000** | **proved-empirical (real LLM)**    | `parser_boundary_real_llm.py`; result JSON `/tmp/wevra-distributed/real_cross_model_n10.json`         |
| **W5-2 (SDK v3.6)**  | **Backend integration: ``run_sweep(..., llm_backend=<duck-typed>)`` routes inner-loop calls through the backend; PROMPT/LLM_RESPONSE/PARSE_OUTCOME/PATCH_PROPOSAL/TEST_VERDICT chain seals byte-for-byte equivalently regardless of backend** | **proved**                        | `test_wevra_llm_backend.py::RunSweepBackendIntegrationTests`                                          |
| **W5-3 (SDK v3.6)**  | **``MLXDistributedBackend`` wire shape: OpenAI-compatible POST /v1/chat/completions with ``{model, messages, max_tokens, temperature, stream:false}``; parses ``choices[0].message.content``** | **proved**                        | `test_wevra_llm_backend.py::MLXDistributedBackendWireShapeTests`                                      |

## Model-scale vs capsule-structure on multi-agent coordination (W6-1 .. W6-4) — SDK v3.7

| Claim                | Description                                                                                                                                                                         | Status                            | Anchor                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **W6-1 (SDK v3.7)**  | **Capsule-team lifecycle audit T-1..T-7 holds for every (regime × strategy × scenario) cell of the Phase-53 scale-vs-structure benchmark when driven by a real-LLM producer-role extractor (3 regimes: synthetic / qwen2.5:14b-32k / qwen3.5:35b × 4 capsule strategies × 5 scenarios = 60/60).** | **proved + mechanically-checked** | `phase53_scale_vs_structure.py::audit_ok_grid`; `docs/data/phase53_scale_vs_structure_K4_n5.json`; `test_wevra_scale_vs_structure.py::Phase53AuditOkGridTests` |
| **W6-2 (SDK v3.7)**  | **Phase-53 driver accepts any duck-typed ``LLMBackend`` substitute as the producer-role extractor backend; the team-coord pipeline seals TEAM_HANDOFF / ROLE_VIEW / TEAM_DECISION capsules end-to-end against an arbitrary backend, with no spine modification.** | **proved**                        | `test_wevra_scale_vs_structure.py::LLMExtractorBackendDuckTypingTests`                                |
| **W6-3 (SDK v3.7)**  | **``parse_role_response`` is robust on the closed-vocabulary claim grammar: accepts ``KIND <sep> payload`` for ``sep ∈ {|, :, -, –, —}``, rejects kinds outside the allowed list, deduplicates by kind (first wins), strips preamble noise, and treats ``NONE`` as zero claims (skip-not-early-return semantics).** | **proved + mechanically-checked** | `test_wevra_scale_vs_structure.py::ParseRoleResponseRobustnessTests` (16 cases)                       |
| **W6-4 (SDK v3.7)**  | **Phase-53 default config (K_auditor=4, T_auditor=128, n_eval=5, prompt-style identical, ``temperature=0``) yields ``accuracy_full = 0.800`` for substrate / capsule_fifo / capsule_priority / capsule_coverage in every model regime. ``capsule_learned`` (trained OOD on Phase-52 synthetic+noise) yields 0.400 on synthetic / 14B and 0.800 on 35B. ``structure_gain[regime]`` is non-positive at every regime (-0.4, -0.4, 0.0); ``scale_gain[capsule_learned] = +0.4``; ``scale_gain[fixed] = 0.0``; ``delta_with_scale = +0.4``. Cross-model candidate-kind TVD (14B vs 35B) = 0.167.** | **proved-empirical (real LLM, n=5 saturated)** | `docs/data/phase53_scale_vs_structure_K4_n5.json`; `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`         |

## Cross-role cohort-coherence multi-agent coordination (W7-1 .. W7-3) — SDK v3.8

| Claim                | Description                                                                                                                                                                         | Status                            | Anchor                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **W7-1 (SDK v3.8)**  | **FIFO unbeatability under low surplus: when the producer's emission stream satisfies ``|candidates(scenario)| ≤ K_role`` for every scenario in the bench, every fixed admission policy (FIFO, claim_priority, coverage_guided, cohort_coherence, learned) is permutation-equivalent to ``admit_all``; ``structure_gain`` over substrate FIFO is identically zero by construction.** Phase-53 default config (``mean_n_admitted_auditor < K_auditor=4`` in every regime) is the empirical anchor in the *positive* direction. | **proved-empirical**              | `docs/data/phase53_scale_vs_structure_K4_n5.json`; `phase53_scale_vs_structure.py`                    |
| **W7-1-aux (SDK v3.8)** | **Streaming cohort-coherence admission is unstable under candidate-arrival permutation: under a stream where the first auditor-routed candidate carries a foreign-service tag, ``CohortCoherenceAdmissionPolicy(fixed_plurality_tag=None)`` admits the foreign tag and rejects every subsequent gold-tag candidate, producing ``mean_n_admitted_auditor`` strictly less than the gold cohort size.** On Phase-54 default, streaming cohort ties FIFO at ``accuracy_full = 0.000`` across 10/10 scenarios. The buffered variant (W7-2 anchor) corrects this by pre-fitting the plurality off-line. | **proved-empirical (n=10 saturated)** | `test_wevra_cross_role_coherence.CohortPolicyUnitTests::test_streaming_locks_to_first_admitted_tag`; `Phase54DefaultConfigTests::test_streaming_cohort_does_not_beat_fifo_at_K4` |
| **W7-2 (SDK v3.8)**  | **Cross-role cohort-coherence structural win under gold-plurality: when the candidate stream satisfies the gold-plurality property (gold service tag has strictly more auditor-routed candidates than any decoy service tag) AND ``|candidates| > K_auditor``, the buffered ``CohortCoherenceAdmissionPolicy`` (constructed via ``from_candidate_payloads``) achieves ``accuracy_full = 1.000`` while substrate FIFO, ``CapsuleFifoAdmissionPolicy``, ``ClaimPriorityAdmissionPolicy``, ``CoverageGuidedAdmissionPolicy``, and the streaming cohort variant all produce ``accuracy_full = 0.000``. Headline gap = +1.000 on the pre-committed Phase-54 default (K_auditor=4, T_auditor=128, n_eval=10), stable across 5/5 bank seeds (11, 17, 23, 29, 31).** | **proved-empirical (n=50 saturated)** | `docs/data/phase54_cross_role_coherence_K4_n10.json`; `Phase54DefaultConfigTests::test_buffered_cohort_strictly_beats_fifo_at_K4`; `Phase54DefaultConfigTests::test_result_stable_across_bank_seeds` |
| **W7-2-conditional (SDK v3.8)** | **W7-2 holds iff K_auditor is in the structure-win window: ``gold_count_min ≤ K_auditor < min(K_auditor+1, total_candidates)``. The Phase-54 budget sweep observes ``+0.400`` gap at K∈{2,3} (gold cohort partially fits) and ``+1.000`` gap at K∈{4,5,6,8} (gold cohort fully fits; foreign-service decoys filtered out by cohort regardless of K). Sharper observation: the decoder's ``services_correct`` set-equality check makes service-blind admission unable to ever achieve services_correct on foreign-service-decoy benches, even with no budget pressure.** | **proved-empirical (K-sweep, K∈{2,3,4,5,6,8})** | `docs/data/phase54_cross_role_coherence_budget_sweep.json` |
| **W7-3 (SDK v3.8)**  | **Extraction floor (proved-negative): no per-role admission policy (FIFO, priority, coverage, learned, streaming or buffered cohort coherence) can recover a missing causal claim that the producer never emitted into the candidate stream. Proof: ``capsule_role_view.parents`` is constructed from ledger CIDs (Capsule Contract C5); a never-emitted claim has no CID; therefore no ROLE_VIEW can include it.** Empirical anchor: Phase-53 ``deadlock_pool_exhaustion`` 1/5 miss across all 5 admission strategies in every model regime; ``DEADLOCK_SUSPECTED`` is never emitted by 14B / 35B from role-local events. Implication: separates admission-fixable failures from extraction-fixable failures; the producer's claim coverage is a strict upper bound on every downstream admission strategy. | **proved-negative**               | Capsule Contract C5; `docs/data/phase53_scale_vs_structure_K4_n5.json` (failure_hist on `deadlock_pool_exhaustion`) |

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
| **W4-C1 (SDK v3.5; SDK v3.7 conditional reading)** | **Learned per-role admission policy admits strictly fewer handoffs (12/12 seeds) and improves pooled team-decision accuracy on most train seeds (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean +0.032) over the strongest fixed admission baseline (coverage-guided) on the Phase-52 default config (K_auditor=8, noise=(0.10, 0.30, 0.05), PYTHONHASHSEED=0).** | **conditional**: empirical-positive on its anchor distribution (Phase-52 synthetic+noise default config, K=8, spurious=0.30 — unchanged from SDK v3.5); **falsified** at higher symbolic noise (spurious=0.50, disclosed in SDK v3.5); **falsified out-of-distribution on the Phase-53 real-LLM regime** (capsule_learned 0.400 vs fixed 0.800 on synthetic and qwen2.5:14b-32k, gap = -0.40; ties at qwen3.5:35b at 0.800/0.800; SDK v3.7, see `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` § 4.4) |
| **W4-C2 (SDK v3.5)** | **Cohort-lifted role view closes W4-3 on a sub-class of scenarios**           | **conjectural** (Phase 53 candidate; falsifier: a scenario whose causal-share floor exceeds a single COHORT's max_parents) |
| **W4-C3 (SDK v3.5)** | **The capsule-layer admission rule subsumes the Phase-36 AdaptiveSubscriptionTable route-edit primitive** | **conjectural** (open) |
| **W5-C1 (SDK v3.6)** | **Parser-boundary instability is a (model architecture × prompt-format) interaction, not a model-capacity artefact. Concretely, scaling 14.8B-dense → 36B-MoE under fixed prompt format on the bundled bank flips strict-mode parser failure rate from 0/10 to 10/10.** | **empirical-research** (n=10 saturation; falsifier = a bank where the larger model strict-parses ok > 50%) |
| **W5-C2 (SDK v3.6)** | **Robust-mode parser recovery (specifically ``recovery=closed_at_eos``) is the load-bearing safety net that makes the capsule-native runtime model-class-agnostic on the bundled prompt format. Cross-model TVD on robust = 0.000 in W5-1.** | **empirical-research** (n=10 saturation; falsifier = a model whose ``unclosed_new`` cannot be salvaged) |
| **W5-C3 (SDK v3.6)** | **The capsule-native runtime's closed-vocabulary ``PARSE_OUTCOME.failure_kind`` is a *minimum sufficient* typed witness of cross-model behaviour differences — strict-mode failure-kind TVD captures in one number what manual byte-level diff over N responses would surface.** | **conjectural** (research; falsifier = a model pair with identical strict-mode `failure_kind` distribution but materially different downstream test-pass rate) |
| **W6-C1 (SDK v3.7; FALSIFIED on Phase-53)** | **Drafted conjecture: ``structure_gain`` (capsule_learned − substrate, accuracy_full) is preserved or grows when the underlying LLM scales up. Falsifier: ``structure_gain[35B] ≤ 0`` with ``substrate ≥ capsule_learned − 0.05``.** | **falsified-empirical** on Phase-53 default: ``structure_gain[14B] = -0.4``; ``structure_gain[35B] = 0.0``. Honest revised reading: structure_gain is **non-positive at every regime tested**; scale narrows a *deficit* (not a surplus). |
| **W6-C2 (SDK v3.7; FALSIFIED on Phase-53)** | **Drafted conjecture: the per-role admission scorer trained on Phase-52 synthetic+noise data transfers usefully (capsule_learned > capsule_fifo) to real-LLM candidate streams. Falsifier: capsule_learned beats capsule_fifo by < 0.05 on average across all model regimes.** | **falsified-empirical** on Phase-53 default: capsule_learned − capsule_fifo = -0.40 / -0.40 / 0.00 across (synthetic, 14B, 35B); average gap = -0.267. The W4-C1 (SDK v3.5) advantage **does not transfer OOD** to the real-LLM regime on this bench. |
| **W6-C3 (SDK v3.7)** | **36B-MoE producer roles emit a different candidate-handoff distribution than 14.8B-dense producer roles on the same scenario bank under matched prompt and temperature; falsifier: per-role TVD between candidate-kind histograms < 0.10.** | **empirical-research (positive)** — TVD = 0.167 on the pooled (source_role × claim_kind) histogram across 5 scenarios; modest but above the 0.10 falsifier; cf. W5-1's parser-axis TVD = 1.000 (the multi-agent extractor axis is more model-class-stable than the parser axis on the same hardware). |
| **W6-C4 (SDK v3.7; new)** | **Substrate FIFO admission is competitive with every capsule admission policy (FIFO, priority, coverage, learned) in real-LLM-driven multi-agent benchmarks at sufficient per-role budget. Falsifier: a (model, scenario, K_auditor) configuration where ``substrate_acc[m, s] < min_capsule_acc[m, s] - 0.05``.** | **conjectural-empirical** — anchored by 0.000 gap at K_auditor=4 on Phase-53 default; falsifier search direction = lower K_auditor (1, 2, 3) where substrate FIFO must admit non-causal head-of-arrival emissions. |
| **W6-C5 (SDK v3.7; new)** | **Model scale narrows the OOD generalisation gap of the per-role admission scorer trained on synthetic noise: ``scale_gain[capsule_learned] > 0 ≥ scale_gain[every fixed capsule policy] = 0`` whenever the synthetic training distribution carries non-trivial noise (``spurious_prob ≥ 0.20``). Falsifier: a synthetic→real transfer where ``scale_gain[capsule_learned] ≤ 0``.** | **conjectural-empirical** — anchored by ``scale_gain[capsule_learned] = +0.400`` against ``scale_gain[fixed] = 0`` on Phase-53 default. |
| **W7-C1 (SDK v3.8; new)** | **Multi-service-gold extension: a cohort policy that admits the *top-2* plurality tags (rather than the strict plurality) strictly improves accuracy_full on multi-service-gold scenarios over single-plurality cohort. Falsifier: a multi-service Phase-55-style bench where top-2 cohort ties or loses to single-plurality cohort.** | **conjectural** (Phase-55 candidate; not yet measured). |
| **W7-C2 (SDK v3.8; new)** | **Bundle-aware decoder + cohort admission strictly outperforms cohort admission alone on benches with weak / partial service-tag coherence (e.g. some gold candidates lacking a service tag). Falsifier: a bench where the decoder modification provides ≤ +0.05 on accuracy_full over cohort_buffered alone.** | **conjectural** (decoder modification not yet shipped). |
| **W7-C3 (SDK v3.8; new)** | **The W7-2 structural win transfers to a real-LLM regime when the LLM is prompted with a multi-service event mix (foreign-service decoys come from the prompt, not from synthetic injection). Falsifier: a real-LLM run where buffered cohort coherence achieves ``cohort_buffered − fifo accuracy_full < 0.20`` after surplus is verified.** | **conjectural** (Phase-56 candidate; requires Mac-2 or wider real-LLM runs). |

## Phase-substrate (P19, P31, P35, P36, P39..P44)

The substrate-side phase-by-phase theorems (P19-L2, P31-1..P31-5,
P35-1..P35-3, P36-*, P39-*, P40-*, P41-*, P42-*, P43-*, P44-*)
are stated and proved in their respective `RESULTS_PHASE*.md`
notes. They are inputs to W3-11 (capsule subsumption) but are
not maintained in this registry — see `vision_mvp/RESULTS_PHASE*.md`
or `docs/archive/pre-wevra-theory/PROOFS.md`.

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
