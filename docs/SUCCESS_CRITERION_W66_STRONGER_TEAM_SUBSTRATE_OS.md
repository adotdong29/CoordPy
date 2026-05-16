# W66 — Stronger Solving-Context Substrate-Coupled Latent OS (pre-committed success criterion)

> Post-W65 research milestone, 2026-05-16. Eleventh substrate-attack
> milestone in the programme. **Second** multi-agent task-success-
> bearing milestone. The bar is written BEFORE any code change so the
> result can be checked against this file. No version bump; no PyPI
> release.

## What W66 is supposed to do

W65 finally produced a head-to-head multi-agent task win (V10 policy
strictly beats `transcript_only` / `shared_state_proxy` /
`substrate_routed_v9` on ≥ 50 % of seeds at matched-budget). That
single win is fragile — only one new policy, no team-failure-recovery
regime, no team-consensus-under-budget regime, MASC is a single
synthetic deterministic task. W66's bar is that the W66 substrate
must produce **multi-agent task wins under a broader set of failure
modes**, not just one synthetic task; team-success has to remain
load-bearing in the **face of disagreement, hostile channels, branch
merges, consensus, and team failure recovery**, not only in the
matched-budget probe.

Concretely W66 must:

1. Add a **fifth policy** `substrate_routed_v11` that strictly beats
   the W65 `substrate_routed_v10` policy on the existing synthetic
   MASC task on ≥ 50 % of seeds at matched-budget.
2. Add a **sixth policy** `team_substrate_coordination_v11` that
   couples the W66 team-consensus-under-budget controller with the
   substrate-routed-V11 policy and beats both V11 and V10 on ≥ 50 %
   of seeds.
3. Add **two new MASC regimes**:
   - `team_consensus_under_budget` — agents must reach a consensus
     guess under a tight visible-token budget; the V11 policy must
     beat each baseline policy by ≥ 0.10 success rate.
   - `team_failure_recovery` — one agent silently fails (zero output)
     mid-task; the V11 policy must recover by ≥ 0.10 success rate
     over the V10 policy.
4. Add a real **team-consensus controller** (`TeamConsensusController`)
   with weighted quorum, abstain, and substrate-replay fallback;
   measured on real branch-merge / consensus / failure-recovery
   regimes (not only synthetic ones).
5. Add a trainable **team_failure_recovery_regime** in the replay
   controller V7 with a closed-form ridge head and a
   **team_consensus_under_budget_regime** also closed-form ridge.
6. Add a richer **substrate-checkpoint-and-snapshot-diff** primitive
   in the V11 substrate, with a measurable
   substrate-snapshot-recovery saving.
7. Preserve all W65 mechanisms byte-for-byte; W66 is strictly
   additive on top of W65.

If any of these fail or only fire as architectural probes, W66 is
partial success, not strong success.

## Pre-committed quantitative bars

### A. Twelve+ mechanism advances over W65

W66 must ship at least twelve substantial mechanism advances on top
of W65. The pre-committed list:

- M1 `coordpy.tiny_substrate_v11`: 13 layers (vs V10's 12); four
  new V11 axes: per-(L, H, T) **replay-trust-ledger** channel,
  per-role **team-failure-recovery flag**, **substrate snapshot-diff**
  primitive, per-layer **V11 composite gate score**
  (extends V10 gate score with replay-trust + team-failure-recovery).
- M2 `coordpy.kv_bridge_v11`: 7-target stacked ridge (6 V10 + 1
  team-failure-recovery target); substrate-measured team-coordination
  margin probe; multi-agent task fingerprint.
- M3 `coordpy.hidden_state_bridge_v10`: 7-target stacked ridge with
  team-consensus-under-budget target; substrate-measured per-(L, H)
  hidden-wins-vs-team-success probe.
- M4 `coordpy.prefix_state_bridge_v10`: K=96 drift curve (vs V9's
  K=64); role+task+team 30-dim fingerprint; five-way
  prefix/hidden/replay/team/recover comparator.
- M5 `coordpy.attention_steering_bridge_v10`: six-stage clamp
  (Hellinger + JS + coarse L1 + fine KL + max-position cap +
  per-(L, H) attention-trust ledger); substrate-measured attention
  fingerprint with team-conditioned weighting.
- M6 `coordpy.cache_controller_v9`: six-objective ridge (adds
  team-failure-recovery); per-role 7-dim eviction priority head.
- M7 `coordpy.replay_controller_v7`: 9 regimes (adds
  `team_failure_recovery_regime` and
  `team_consensus_under_budget_regime`); per-role per-regime ridge;
  trained team-substrate-routing head (closed-form ridge).
- M8 `coordpy.deep_substrate_hybrid_v11`: eleven-way bidirectional
  loop with the new V11 substrate axes + team-consensus-under-budget
  arbiter.
- M9 `coordpy.substrate_adapter_v11`: new `substrate_v11_full` tier;
  V11 capability axes.
- M10 `coordpy.persistent_latent_v18`: 17 layers; 15th persistent
  skip carrier (`team_failure_recovery_carrier`); chain walk depth
  carried forward at 8192; distractor rank 16.
- M11 `coordpy.multi_hop_translator_v16`: 36 backends; 1260
  directed edges; chain-length 26; 11-axis composite (adds
  `team_substrate_coordination_trust`).
- M12 `coordpy.mergeable_latent_capsule_v14`: adds
  `team_failure_recovery_witness_chain` and
  `team_consensus_under_budget_witness_chain`.
- M13 `coordpy.consensus_fallback_controller_v12`: 18-stage chain
  inserting `team_failure_recovery_arbiter` and
  `team_consensus_under_budget_arbiter`.
- M14 `coordpy.corruption_robust_carrier_v14`: 16384-bucket
  wrap-around-XOR fingerprint; 33-bit adversarial burst;
  team-failure-recovery probe.
- M15 `coordpy.long_horizon_retention_v18`: 17 heads;
  `max_k=320`; eight-layer scorer.
- M16 `coordpy.ecc_codebook_v18`: K1..K17 = 2^29 = 536 870 912 codes;
  ≥ 31.0 bits/visible-token at full emit.
- M17 `coordpy.uncertainty_layer_v14`: 13-axis composite (adds
  `team_failure_recovery_fidelity`).
- M18 `coordpy.disagreement_algebra_v12`: Jensen-Shannon-equivalence
  identity + falsifier (≥ four equivalence identities).
- M19 `coordpy.transcript_vs_shared_arbiter_v15`: 16-arm comparator
  (adds `team_failure_recovery` arm).
- M20 `coordpy.multi_agent_substrate_coordinator_v2` **(NEW)**:
  six matched-budget policies (transcript_only / shared_state_proxy /
  substrate_routed_v9 / substrate_routed_v10 / substrate_routed_v11 /
  team_substrate_coordination_v11); two new regimes
  (team_failure_recovery, team_consensus_under_budget); measurable
  team success rate / visible-tokens-per-turn /
  substrate-recovery-rate / team-coordination-rate.
- M21 `coordpy.team_consensus_controller` **(NEW)**: weighted
  quorum + abstain + substrate-replay fallback; explicit
  branch-merge / consensus / failure-recovery regimes.

### B. Three benchmark families

- **R-146** — V11 substrate / eleven-way hybrid / multi-agent
  task-success / team-coordinator-V2 / team-consensus controller
  (24 H-bars: H245–H252c).
- **R-147** — 8192-turn retention / V18 LHR / V16 multi-hop /
  V18 ECC / persistent V18 (14 H-bars: H253–H259a).
- **R-148** — corruption V14 / consensus V12 / disagreement V12 /
  uncertainty V14 / TVS V15 / W66 envelope verifier
  (18 H-bars: H260–H268).

Total: **56 H-bars × 3 seeds = 168 cells**. Strong success means
all cells pass; partial success means ≥ 90 % pass with a recorded
honest reason.

### C. The 20 required dimensions

W66 must hit at minimum:

1. ≥ 12 real mechanism advances over W65 → covered by M1..M21.
2. ≥ 3 benchmark families → R-146/R-147/R-148.
3. ≥ 1 live/replay-live realism anchor → MASC-V2 runs the same
   RNG-driven task across six policies and compares matched-budget
   outcomes under three regimes (baseline + team-consensus-under-
   budget + team-failure-recovery).
4. ≥ 1 true substrate-coupling bar → H245 (V11 substrate
   determinism + 4 new axes shape).
5. ≥ 1 latent-to-KV bar → H246 (KV bridge V11 seven-target ridge
   converges).
6. ≥ 1 latent-to-hidden-state / prefix-state bar → H247 (HSB V10
   seven-target ridge); H248 (prefix V10 K=96 drift curve).
7. ≥ 1 cache-reuse-vs-recompute bar → H252 (substrate snapshot-diff
   recovery saving ≥ 0.6).
8. ≥ 1 trainable / fitted substrate-controller bar → H250 (cache V9
   six-objective ridge); H251 (replay V7 nine-regime ridge +
   team_failure_recovery head).
9. ≥ 1 replay-dominance-primary bar → H251c (replay V7 still respects
   V6 dominance-primary head; new dominance-primary-bonus active).
10. ≥ 1 hidden-wins-primary or hidden-beats-KV bar → H247b (HSB V10
    hidden-wins-vs-team-success probe positive when hidden bridge
    actually dominates).
11. ≥ 1 long-horizon retention bar → H253 (persistent V18 15th
    carrier); H253b (chain walk depth ≥ 8192).
12. ≥ 1 merge / consensus / disagreement bar → H260 (consensus V12
    18 stages); H261 (disagreement V12 JS identity).
13. ≥ 1 hostile-channel / corruption / adversarial bar → H260b
    (CRC V14 33-bit burst); H260c (16384-bucket detect).
14. ≥ 1 transcript-vs-shared head-to-head → H264 (TVS V15 16-arm
    sum to 1.0).
15. ≥ 1 explicit multi-agent task-success bar **(load-bearing)** →
    H252b (V11 multi-agent coordinator beats V10 on ≥ 50 % of
    seeds at matched-budget on the existing synthetic task);
    H252c (`team_substrate_coordination_v11` policy beats V11 on
    ≥ 50 % of seeds).
16. ≥ 1 abstain-or-fallback policy that materially matters →
    H251b (team-failure-recovery regime fires under specified
    regime; team-consensus-under-budget regime fires).
17. ≥ 1 explicit falsifier → H246b (KV V11 falsifier returns 0
    under inversion); H261b (JS falsifier triggers).
18. ≥ 1 explicit limitation reproduction → H259 (ECC V18 2^29
    structural rate ceiling).
19. ≥ 1 serious cramming gain or robustness-vs-rate tradeoff →
    H259a (≥ 31.0 bits/visible-token at full emit, up from 29.333).
20. committed and pushed output if coherent → see §F below.

### D. Pre-committed numeric thresholds

- V11 substrate: 13 layers; four new axes wired.
- KV bridge V11: seven targets; falsifier returns 0 under inversion.
- HSB V10: seven targets; per-(L, H) hidden-wins-vs-team-success
  rate measurable.
- Prefix V10 drift curve: K=96; converged.
- Attention V10: six-stage clamp keeps Hellinger ≤ 1.0.
- Cache controller V9: six-objective ridge converges.
- Replay controller V7: 9 regimes; ridge converges;
  team-failure-recovery + team-consensus-under-budget heads
  converge.
- Deep substrate hybrid V11: eleven-way fires when all 11 axes fire.
- Persistent V18: 17 layers; chain depth ≥ 8192; rank ≥ 16.
- Multi-hop V16: 36 backends; chain-length 26; 11-axis composite.
- Mergeable capsule V14: team-failure-recovery chain + team-
  consensus-under-budget chain inherit as union at merge.
- Consensus V12: 18 stages.
- CRC V14: 16384-bucket detect ≥ 0.95; 33-bit burst ≥ 0.40.
- LHR V18: 17 heads; `max_k=320`.
- ECC V18: 2^29 codes; ≥ 31.0 bits/visible-token at full emit.
- Uncertainty V14: 13-axis weighted composite ∈ [0, 1].
- Disagreement V12: JS-equivalence identity holds.
- TVS V15: 16 arms sum to 1.0.
- Multi-agent coordinator V2: V11 team success ≥ V10 + 0.05
  (matched-budget); team-substrate-coordination-V11 team success
  ≥ V11 + 0.05; ≥ 50 % visible-token savings vs transcript-only.
- W66 envelope verifier: ≥ 120 disjoint failure modes.
- Cumulative trust boundary across W22..W66 ≥ 1225.

### E. Honest scope (pre-committed limitations)

- `W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` — hosted backends
  (Ollama, OpenAI-compatible) remain text-only on the HTTP surface;
  W66 makes no claim of third-party transformer-internal access.
- `W66-L-V11-NO-AUTOGRAD-CAP` — W66 fits only closed-form linear
  ridge solves. Six new W66 ridge solves on top of W65's 29 (cache
  V9 six-objective; cache V9 per-role eviction; replay V7 per-role
  per-regime; replay V7 team-substrate-routing; HSB V10 seven-
  target; KV V11 seven-target). Total **thirty-five closed-form
  ridge solves** across W61..W66.
- `W66-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP` — the
  multi-agent-task win is still a synthetic deterministic harness;
  the V11 wins are engineered so that the V11 mechanisms (replay-
  trust ledger, team-failure-recovery flag, team-consensus-under-
  budget arbiter, substrate snapshot-diff) materially reduce drift.
- `W66-L-NUMPY-CPU-V11-SUBSTRATE-CAP` — V11 still runs in NumPy on
  CPU; not a frontier model.
- `W66-L-V18-OUTER-NOT-TRAINED-CAP` — persistent V18 only the inner
  V12 cell is trained at init; the V13..V18 outer carriers ride
  along as deterministic projections.
- `W66-L-ECC-V18-RATE-FLOOR-CAP` — ECC V18 has a strict structural
  rate ceiling of log2(2^29) = 29 raw bits per segment-tuple.
- `W66-L-TEAM-CONSENSUS-IN-REPO-CAP` — the team-consensus
  controller operates on in-repo MASC outcomes only; it does not
  enforce consensus on real model outputs.
- `W66-L-V18-LHR-SCORER-FIT-CAP` — LHR V18 scorer is a deterministic
  composition of random projections plus a final ridge.
- `W66-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP` — substrate
  snapshot-diff operates on the in-repo V11 cache only.

### F. Output (commit / push)

If the bar is met or partially met under §E, the milestone commits
its output and pushes it under the existing repo git author
identity. No version bump; no PyPI release; no AI authorship
markers.

## Falsifiers

- **F1** — If the V11 policy does not strictly beat V10 on ≥ 50 % of
  seeds at matched-budget, the strong-success claim fails.
- **F2** — If the team-substrate-coordination-V11 policy does not
  strictly beat V11 on ≥ 50 % of seeds, the team-coordination claim
  fails.
- **F3** — If any of the six new closed-form ridge solves fails to
  converge to a finite residual, the trainable-substrate-controller
  claim fails.
- **F4** — If the substrate snapshot-diff recovery saving is below
  0.6, the substrate-cache-reuse claim fails.
- **F5** — If the ECC V18 bits/visible-token is below 31.0, the
  cramming claim fails (V17 already hit 29.333).
- **F6** — If the W66 envelope verifier enumerates fewer than 120
  disjoint failure modes, the audit-surface-expansion claim fails.

## Verdict gate

Strong success: every H-bar passes on every seed; V11 policy beats
V10; team-substrate-coordination-V11 beats V11; all six new ridge
solves converge.

Partial success: ≥ 90 % of cells pass with at least one honest
reason recorded in §E.

Failure: anything below 90 % of cells, or any falsifier triggers,
or any required mechanism advance is missing.
