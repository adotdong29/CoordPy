# W67 — Stronger Branch-Merge / Role-Dropout Substrate-Coupled Latent OS (pre-committed success criterion)

> Post-W66 research milestone, 2026-05-16. **Twelfth** substrate-attack
> milestone in the programme. **Third** multi-agent task-success-bearing
> milestone (the first to produce wins under *five* failure-mode
> regimes, not three). The bar is written BEFORE any code change so the
> result can be checked against this file. No version bump; no PyPI
> release.

## What W67 is supposed to do

W66 produced multi-agent task wins under three regimes (baseline +
team-consensus-under-budget + team-failure-recovery). That was the
breakthrough. The remaining failure modes of multi-agent teams that
W66 does NOT yet address head-to-head are:

1. **Role dropout** — an entire role disappears (silent for many
   turns, not just one), forcing other roles to absorb its
   responsibilities. The W66 `team_failure_recovery` regime only
   handles a single agent dropping out for the whole task; a
   *role-dropout* regime models a recurring dropout that the team
   must recover from across turns.
2. **Branch-merge reconciliation** — agents branch, work
   independently, then must reconcile *conflicting* updates into a
   consistent shared state. W66's TCC fires `substrate_replay` when
   quorum fails but it does not arbitrate over a *graph* of
   branches.
3. **Disagreement under hostile channels** — capsules disagree AND
   the latent channel is being corrupted, so the consensus
   controller must decide which branch to trust *and* repair the
   corrupted carrier.

W67's bar is that the W67 substrate must produce **multi-agent task
wins under five failure-mode regimes** (the three from W66 + the two
new) and that the new mechanisms must materially help on at least the
two new regimes. The substrate has to keep doing more work, not less.

Concretely W67 must:

1. Add a **seventh policy** `substrate_routed_v12` that strictly
   beats `substrate_routed_v11` on the existing synthetic MASC task
   on ≥ 50 % of seeds at matched-budget across all five regimes
   (the three W66 regimes plus the two new W67 regimes).
2. Add an **eighth policy** `team_substrate_coordination_v12` that
   couples the W67 team-consensus controller V2 with
   substrate-routed-V12 and beats `team_substrate_coordination_v11`
   on ≥ 50 % of seeds across all five regimes (load-bearing in the
   two new W67 regimes).
3. Add **two new MASC regimes**:
   - `role_dropout` — one role drops out at random points (multiple
     dropout windows over the task, not just one mid-task event); the
     V12 policy must beat V11 by ≥ 0.05 success rate.
   - `branch_merge_reconciliation` — agents fork into branches at
     mid-task, each branch produces a conflicting payload, and the
     team must reconcile into one consensus; the V12 policy must
     beat V11 by ≥ 0.05 success rate.
4. Add a real **team-consensus controller V2** (`TeamConsensusControllerV2`)
   with branch-merge arbiter, role-dropout-repair head, and
   conflict-resolution policy; measured on real branch-merge /
   role-dropout regimes.
5. Add a trainable **role_dropout_regime** in the replay controller
   V8 with a closed-form ridge head and a
   **branch_merge_reconciliation_regime** also closed-form ridge.
6. Add a richer **substrate branch-merge primitive** in the V12
   substrate, with a measurable substrate-branch-merge saving.
7. Preserve all W66 mechanisms byte-for-byte; W67 is strictly
   additive on top of W66.

If any of these fail or only fire as architectural probes, W67 is
partial success, not strong success.

## Pre-committed quantitative bars

### A. Twelve+ mechanism advances over W66

W67 must ship at least twelve substantial mechanism advances on top
of W66. The pre-committed list:

- M1 `coordpy.tiny_substrate_v12`: 14 layers (vs V11's 13); four
  new V12 axes: per-(L, H, T) **branch-merge witness tensor**,
  per-role-pair **role-dropout-recovery flag**, per-branch
  **substrate snapshot-fork primitive**, per-layer **V12
  composite gate score**.
- M2 `coordpy.kv_bridge_v12`: 8-target stacked ridge (7 V11 + 1
  branch-merge-reconciliation target); substrate-measured
  branch-merge margin probe; role-pair fingerprint;
  branch-merge-reconciliation falsifier.
- M3 `coordpy.hidden_state_bridge_v11`: 8-target stacked ridge with
  role-dropout target; substrate-measured per-(L, H)
  hidden-vs-branch-merge probe.
- M4 `coordpy.prefix_state_bridge_v11`: K=128 drift curve (vs V10's
  K=96); role+task+team+branch 40-dim fingerprint; six-way
  prefix/hidden/replay/team/recover/branch comparator.
- M5 `coordpy.attention_steering_bridge_v11`: seven-stage clamp
  (V10 + per-(L, H) branch-merge attention bias); substrate-
  measured attention fingerprint with branch-conditioned weighting.
- M6 `coordpy.cache_controller_v10`: seven-objective ridge (adds
  branch-merge); per-role 8-dim eviction priority head.
- M7 `coordpy.replay_controller_v8`: 12 regimes (adds
  `role_dropout_regime` and `branch_merge_reconciliation_regime`);
  per-role per-regime ridge; trained branch-merge-routing head
  (closed-form ridge).
- M8 `coordpy.deep_substrate_hybrid_v12`: twelve-way bidirectional
  loop with the new V12 substrate axes + branch-merge axis.
- M9 `coordpy.substrate_adapter_v12`: new `substrate_v12_full`
  tier; V12 capability axes.
- M10 `coordpy.persistent_latent_v19`: 18 layers (vs V18's 17);
  sixteenth skip carrier `role_dropout_recovery_carrier`;
  `max_chain_walk_depth=16384`.
- M11 `coordpy.multi_hop_translator_v17`: 40 backends (vs V16's
  36); chain-length 30; 12-axis composite adding
  `branch_merge_reconciliation_trust`.
- M12 `coordpy.mergeable_latent_capsule_v15`: adds
  `role_dropout_recovery_witness_chain` and
  `branch_merge_reconciliation_witness_chain`.
- M13 `coordpy.consensus_fallback_controller_v13`: 20-stage chain
  (vs V12's 18) inserting `role_dropout_arbiter` and
  `branch_merge_reconciliation_arbiter`.
- M14 `coordpy.corruption_robust_carrier_v15`: 32768-bucket
  fingerprint (vs V14's 16384); 35-bit adversarial burst (vs V14's
  33); branch-merge-reconciliation recovery probe.
- M15 `coordpy.long_horizon_retention_v19`: 18 heads (vs V18's
  17), max_k=384 (vs V18's 320); branch-merge-conditioned head.
- M16 `coordpy.ecc_codebook_v19`: K1..K18 = 2^31 = 2 147 483 648
  codes; **33.333 bits/visible-token** at full emit.
- M17 `coordpy.uncertainty_layer_v15`: 14-axis composite adding
  `branch_merge_reconciliation_fidelity`.
- M18 `coordpy.disagreement_algebra_v13`: Bregman-equivalence
  identity + falsifier.
- M19 `coordpy.transcript_vs_shared_arbiter_v16`: 17-arm comparator
  adding `branch_merge_reconciliation` arm.
- M20 `coordpy.multi_agent_substrate_coordinator_v3`: eight-policy
  MASC V3 across five regimes; V12 strictly beats V11; TSC_V12
  strictly beats TSC_V11.
- M21 `coordpy.team_consensus_controller_v2`: regime-aware
  weighted quorum + branch-merge arbiter + role-dropout repair +
  substrate-replay fallback + transcript fallback.

### B. Three benchmark families

W67 must ship at least three benchmark families with ≥ 18 H-bars
each (≥ 54 total).

- R-149: V12 substrate / twelve-way hybrid / multi-agent
  task-success V3 / team-coordinator-V3 / team-consensus-controller
  V2 (≥ 24 H-bars).
- R-150: Long-horizon retention V19 / aggressive ECC V19 /
  persistent V19 / multi-hop V17 (≥ 14 H-bars).
- R-151: Corruption V15 / disagreement V13 / consensus V13 /
  branch-merge / role-dropout (≥ 18 H-bars).

### C. Five regime multi-agent wins

- R-149 H285b — V12 strictly beats V11 on ≥ 50 % of seeds in the
  baseline regime, matched-budget.
- R-149 H285c — TSC_V12 strictly beats TSC_V11 on ≥ 50 % of seeds
  in the baseline regime.
- R-149 H285d — V12 strictly beats V11 on ≥ 50 % of seeds in
  team_consensus_under_budget.
- R-149 H285e — V12 strictly beats V11 on ≥ 50 % of seeds in
  team_failure_recovery.
- R-149 H285f — V12 strictly beats V11 on ≥ 50 % of seeds in
  role_dropout.
- R-149 H285g — V12 strictly beats V11 on ≥ 50 % of seeds in
  branch_merge_reconciliation.

### D. Long-horizon and compression bars

- R-150 H287 — Persistent V19 chain-walk depth ≥ 16384.
- R-150 H287b — Persistent V19 has sixteenth skip carrier.
- R-150 H287c — Long-horizon retention V19 max_k=384.
- R-150 H287d — ECC V19 ≥ 33.0 bits/visible-token at full emit.

### E. Corruption / disagreement / consensus bars

- R-151 H290 — CRC V15 32768-bucket fingerprint single-byte detect
  rate ≥ 0.95.
- R-151 H290b — CRC V15 35-bit adversarial burst detect rate
  ≥ 0.4.
- R-151 H291 — Consensus V13 20-stage chain.
- R-151 H291b — Consensus V13 `branch_merge_reconciliation_arbiter`
  fires under conflicting branches.
- R-151 H291c — Consensus V13 `role_dropout_arbiter` fires when at
  least one parent is silent.
- R-151 H292 — Disagreement Algebra V13 Bregman-equivalence identity
  holds iff argmax preserved AND Bregman ≤ floor.

### F. Falsifiers and limitation reproductions

- W67-F-V12-NO-AUTOGRAD — V12 ridge solves remain closed-form
  linear (no SGD, no autograd, no GPU).
- W67-F-V12-NO-THIRD-PARTY-SUBSTRATE — V12 substrate is in-repo
  NumPy; not bridging to third-party hosted models.
- W67-F-BRANCH-MERGE-IN-REPO — branch-merge primitive operates on
  the in-repo V12 cache only.
- W67-L-MULTI-AGENT-COORDINATOR-V3-SYNTHETIC — MASC V3 is a
  synthetic deterministic harness.

### G. Trust boundary

- ≥ 140 disjoint failure modes enumerated in the W67 envelope
  verifier (vs W66's 123).
- ≥ 54 H-bars across R-149/R-150/R-151 (162/162 cells at 3 seeds
  → strong success).
- Cumulative trust boundary across W22..W67 ≥ 1368 enumerated
  failure modes.

### H. Stable-boundary preservation

- `coordpy.__version__ == "0.5.20"` (unchanged).
- SDK_VERSION unchanged.
- No PyPI release.
- Trivial passthrough preserved: W66 envelope CID identical
  byte-for-byte when W67Params.build_trivial() is used.

## Verdict labels

- **Strong success** — ≥ 12 mechanism advances + 3 benchmark
  families + ≥ 54 H-bars at 3 seeds pass + ≥ 4 regime wins fire +
  long-horizon + compression + corruption bars all pass.
- **Partial success** — ≥ 8 mechanism advances + 2 benchmark
  families + ≥ 30 H-bars pass + at least 2 regime wins fire.
- **Failure** — fewer than 8 mechanism advances OR ≥ 2 benchmark
  families fail OR no new regime wins fire.

W67 is engineered so that strong success is the expected outcome
on the in-repo V12 substrate. Failure to reach strong success would
demonstrate that the W66 substrate is the local optimum and that
the next move must be the third-party transformer-internal bridge.
