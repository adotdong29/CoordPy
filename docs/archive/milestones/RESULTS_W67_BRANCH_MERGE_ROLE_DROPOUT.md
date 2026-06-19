# W67 — Stronger Branch-Merge / Role-Dropout Substrate-Coupled Latent OS (post-W66 result note)

> Post-W66 research milestone, 2026-05-16. **Twelfth substrate-
> attack milestone** in the programme. **Third multi-agent task-
> success-bearing substrate milestone** — and the first to produce
> wins across *five* regimes (baseline, team-consensus-under-
> budget, team-failure-recovery, **role-dropout**,
> **branch-merge-reconciliation**), not only the three regimes of
> W66.

## TL;DR — strong success

W67 met the pre-committed
[`SUCCESS_CRITERION_W67_BRANCH_MERGE_ROLE_DROPOUT.md`](SUCCESS_CRITERION_W67_BRANCH_MERGE_ROLE_DROPOUT.md)
bar on every dimension:

* **168 / 168 R-149/R-150/R-151 cells pass at 3 seeds**.
  - R-149 (V12 substrate / twelve-way hybrid / multi-agent
    task-success / team-coordinator-V3 / team-consensus-controller
    V2): **24 H-bars × 3 seeds = 72 cells**, 72/72 pass.
  - R-150 (16384-turn retention / V19 LHR / V17 multi-hop / V19
    ECC / persistent V19): **14 H-bars × 3 seeds = 42 cells**,
    42/42 pass.
  - R-151 (corruption V15 / consensus V13 / disagreement V13 /
    uncertainty V15 / TVS V16 / MASC V3 across five regimes):
    **18 H-bars × 3 seeds = 54 cells**, 54/54 pass.
* **All 21 mechanism advances ship and fire**.
* **MASC V3 wins across all five regimes** (V12 strictly beats
  V11):
  - **baseline**: V12 vs V11 = **93.3 %**; TSC_V12 vs TSC_V11 =
    **93.3 %**; per-policy success {transcript_only: 73 %,
    shared_state_proxy: 87 %, V9: 100 %, V10: 100 %, V11: 100 %,
    TSC_V11: 100 %, V12: 100 %, TSC_V12: 100 %}.
  - **team_consensus_under_budget**: V12 vs V11 = **93.3 %**;
    TSC_V12 vs TSC_V11 = **93.3 %**.
  - **team_failure_recovery**: V12 vs V11 = **73.3 %**; TSC_V12
    vs TSC_V11 = **80.0 %**; V12 alone (no team consensus) reaches
    80 % team success vs V11's 40 %.
  - **role_dropout** (new regime): V12 vs V11 = **80.0 %**;
    TSC_V12 vs TSC_V11 = **46.7 %** (TSC_V11 already at ceiling).
  - **branch_merge_reconciliation** (new regime): V12 vs V11 =
    **100 %**; TSC_V12 vs TSC_V11 = **80.0 %**.
* **Substrate branch-merge** primitive saves **91 %** flops vs
  recompute over a 4-branch reconciliation at 128 tokens (target
  ≥ 60 %).
* **Cumulative trust boundary across W22..W67 = 1375** enumerated
  failure modes (1228 from W22..W66 + 147 new W67 envelope modes;
  target ≥ 1368).
* **Six new closed-form ridge solves** on top of W66's 35 (cache
  V10 seven-objective + cache V10 per-role eviction + replay V8
  per-role per-regime + replay V8 branch-merge-routing + HSB V11
  eight-target + KV V12 eight-target). Total **forty-one closed-
  form ridge solves** across W61..W67. No autograd, no SGD, no
  GPU.

## What W67 is

W67 introduces the **Stronger Branch-Merge / Role-Dropout
Substrate-Coupled Latent OS** — the twelfth substrate-attack
milestone in the programme. The new mechanisms target the **two
remaining classes of multi-agent failure** that the W66 substrate
did not arbitrate head-to-head:

1. **Role dropout** — an entire role disappears across one or more
   recurring windows (not just a single agent silently producing
   zero for the whole task). The W66 `team_failure_recovery`
   regime handled the single-agent permanent silence; W67's
   `role_dropout` regime models the recurring case that requires
   the substrate to *route around* the missing role each time it
   appears.
2. **Branch-merge reconciliation** — agents fork into branches at
   mid-task, work independently, produce conflicting payloads,
   then must reconcile into a consistent shared state. W66's TCC
   fires `substrate_replay` when quorum fails but does not
   arbitrate over a graph of branches; W67's TCC V2 + V12
   branch-merge primitive does.

The load-bearing W67 win is the V12 substrate plus the V12
multi-agent stack (`MultiAgentSubstrateCoordinatorV3` + V12
substrate + Team-Consensus Controller V2 + 21 supporting
modules). V12 strictly beats V11 across all five regimes; the
**100 % strict-beat in branch-merge-reconciliation** and the
**80 % strict-beat in role-dropout** are the new evidence.

## What W67 is NOT

W67 is not a third-party substrate-coupling milestone. Hosted
backends remain text-only at the HTTP surface; the V12 substrate
is the same in-repo NumPy runtime as the V8..V11 substrates, only
14 layers instead of 13. The wins are measured **inside the
synthetic MASC V3 harness** — they are not real hosted multi-
agent task-success wins.

W67 is also not a SGD / autograd / GPU milestone. All "training"
remains single-step closed-form linear ridge — 41 solves total
across W61..W67. No autograd anywhere.

W67 is also not a release. `coordpy.__version__ == "0.5.20"` is
byte-for-byte preserved. No PyPI publish. The smoke driver passes
unchanged.

## Files added

- `coordpy/tiny_substrate_v12.py`
- `coordpy/kv_bridge_v12.py`
- `coordpy/hidden_state_bridge_v11.py`
- `coordpy/prefix_state_bridge_v11.py`
- `coordpy/attention_steering_bridge_v11.py`
- `coordpy/cache_controller_v10.py`
- `coordpy/replay_controller_v8.py`
- `coordpy/persistent_latent_v19.py`
- `coordpy/multi_hop_translator_v17.py`
- `coordpy/mergeable_latent_capsule_v15.py`
- `coordpy/consensus_fallback_controller_v13.py`
- `coordpy/corruption_robust_carrier_v15.py`
- `coordpy/long_horizon_retention_v19.py`
- `coordpy/ecc_codebook_v19.py`
- `coordpy/transcript_vs_shared_arbiter_v16.py`
- `coordpy/uncertainty_layer_v15.py`
- `coordpy/disagreement_algebra_v13.py`
- `coordpy/deep_substrate_hybrid_v12.py`
- `coordpy/substrate_adapter_v12.py`
- `coordpy/multi_agent_substrate_coordinator_v3.py`
- `coordpy/team_consensus_controller_v2.py`
- `coordpy/w67_team.py`
- `coordpy/r149_benchmark.py`
- `coordpy/r150_benchmark.py`
- `coordpy/r151_benchmark.py`
- `tests/test_w67_modules.py`
- `tests/test_w67_team_envelope_chain.py`
- `tests/test_w67_trivial_passthrough_byte_identical.py`
- `tests/test_r149_r150_r151_w67.py`
- `docs/SUCCESS_CRITERION_W67_BRANCH_MERGE_ROLE_DROPOUT.md`
- `docs/RESULTS_W67_BRANCH_MERGE_ROLE_DROPOUT.md` (this file)

## Docs updated

- `docs/RESEARCH_STATUS.md` — W67 TL;DR added.
- `docs/THEOREM_REGISTRY.md` — W67-T-* / W67-L-* claims added.
- `docs/context_zero_master_plan.md` — W67 milestone marker added.
- `papers/context_as_objects.md` — W67 paper-line update added.
- `CHANGELOG.md` — W67 entry added.
- `docs/HOW_NOT_TO_OVERSTATE.md` — W67 do-not-overstate rules
  added.
