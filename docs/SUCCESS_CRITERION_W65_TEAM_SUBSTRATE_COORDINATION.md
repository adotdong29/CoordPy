# W65 — Team-Substrate-Coordination Substrate-Coupled Latent OS (pre-committed success criterion)

> Post-W64 research milestone, 2026-05-16. Tenth substrate-attack
> milestone in the programme. The bar is written BEFORE any code
> change so the result can be checked against this file. No version
> bump; no PyPI release.

## What W65 is supposed to do

W64 made the in-repo substrate deeper and added a nine-way hybrid
loop, but it stopped short of measuring whether substrate control
*actually changes multi-agent coordination outcomes*. W65's bar is
that the W65 substrate has to demonstrate **measurable team-level
wins** under a tight transcript budget — not just internal probes
firing. Concretely:

1. A **MultiAgentSubstrateCoordinator** that runs a real N-agent
   loop where role-typed agents pass latent carriers through the
   V10 substrate and the W65 controllers decide
   reuse/recompute/fallback/abstain per turn.
2. A **head-to-head comparator** that runs the same synthetic
   multi-agent task under four matched-budget policies:
   transcript-only, shared-state-proxy (W48 baseline),
   substrate-routed-V9 (W64 baseline), substrate-routed-V10
   (W65 new). The team success rate of the V10 policy must be
   *strictly higher* than each baseline under the matched
   transcript budget — and the V10 policy must not be reached by
   simply enabling more bridges; the win must be attributable to
   *new W65 mechanisms* (multi-agent abstain head, role-conditioned
   KV bank, team-task-success target, substrate checkpoint).
3. A **trainable team-substrate-coordination regime** in the replay
   controller V6 with a closed-form ridge head and a multi-agent
   abstain head — also closed-form ridge — that the comparator
   actually queries on every turn.
4. A real **substrate checkpoint / restore** primitive in the V10
   substrate, with a measurable cache-reuse-vs-recompute saving.

If any of these fail or only fire as architectural probes, W65 is
partial success, not strong success.

## Pre-committed quantitative bars

### A. Twelve+ mechanism advances over W64

W65 must ship at least twelve substantial mechanism advances on
top of W64. The pre-committed list:

- M1 `coordpy.tiny_substrate_v10`: 12 layers (vs V9's 11); four
  new V10 axes: per-(L, H, T) **hidden-write-merit** channel,
  per-role **KV bank**, **substrate checkpoint/restore** primitive,
  per-layer **V10 composite gate score**.
- M2 `coordpy.kv_bridge_v10`: 6-target stacked ridge (5 V9 + 1
  team-task-routing target); substrate-measured per-target margin
  probe; **substrate cache-reuse saving** measurement.
- M3 `coordpy.hidden_state_bridge_v9`: 6-target stacked ridge with
  team-coordination target; substrate-measured per-(L, H)
  hidden-wins-rate probe.
- M4 `coordpy.prefix_state_bridge_v9`: K=64 drift curve (vs V8's
  K=32); role+task 20-dim fingerprint; substrate-measured per-step
  drift probe.
- M5 `coordpy.attention_steering_bridge_v9`: five-stage clamp
  (Hellinger + JS + coarse L1 + fine KL + max-position cap);
  substrate-measured attention-map fingerprint per-(L, H).
- M6 `coordpy.cache_controller_v8`: five-objective ridge (adds
  team-task-success); per-role eviction priority head.
- M7 `coordpy.replay_controller_v6`: 8 regimes (adds
  team-substrate-coordination); per-role per-regime ridge;
  **multi-agent abstain head** (closed-form ridge).
- M8 `coordpy.deep_substrate_hybrid_v10`: ten-way bidirectional
  loop with hidden-write-merit + role-conditioned-KV +
  substrate-checkpoint axes.
- M9 `coordpy.substrate_adapter_v10`: new `substrate_v10_full`
  tier; V10 capability axes.
- M10 `coordpy.persistent_latent_v17`: 16 layers; 14th persistent
  skip carrier (team-task-success EMA); `max_chain_walk_depth =
  8192`; distractor rank 14.
- M11 `coordpy.multi_hop_translator_v15`: 35 backends; 1190
  directed edges; chain-length 25; 10-axis composite (adds
  `team_coordination_trust`).
- M12 `coordpy.mergeable_latent_capsule_v13`: adds
  `team_substrate_witness_chain` and `role_conditioned_witness_chain`.
- M13 `coordpy.consensus_fallback_controller_v11`: 16-stage chain
  inserting `team_substrate_coordination_arbiter` and
  `multi_agent_abstain_arbiter`.
- M14 `coordpy.corruption_robust_carrier_v13`: 8192-bucket
  wrap-around-XOR fingerprint; 31-bit adversarial burst; team
  coordination recovery probe.
- M15 `coordpy.long_horizon_retention_v17`: 16 heads; `max_k=256`;
  seven-layer scorer.
- M16 `coordpy.ecc_codebook_v17`: K1..K16 = 2^27 = 134 217 728
  codes; ≥ 29.0 bits/visible-token at full emit.
- M17 `coordpy.uncertainty_layer_v13`: 12-axis composite (adds
  `team_coordination_fidelity`).
- M18 `coordpy.disagreement_algebra_v11`: Wasserstein-equivalence
  identity + falsifier.
- M19 `coordpy.transcript_vs_shared_arbiter_v14`: 15-arm
  comparator (adds `team_substrate_coordination` arm).
- M20 `coordpy.multi_agent_substrate_coordinator` **(NEW)**: real
  N-agent multi-agent harness with four matched-budget policies
  (transcript_only / shared_state_proxy / substrate_routed_v9 /
  substrate_routed_v10) and measurable team success rate /
  visible-tokens-per-turn / substrate-recovery-rate.

### B. Three benchmark families

- **R-143** — V10 substrate / ten-way hybrid / multi-agent
  task-success / team coordinator (18 H-bars: H223–H230c).
- **R-144** — 8192-turn retention / V17 LHR / V15 multi-hop /
  V17 ECC / persistent V17 (12 H-bars: H231–H236).
- **R-145** — corruption / consensus V11 / disagreement V11 /
  uncertainty V13 / TVS V14 / W65 envelope verifier (16 H-bars:
  H237–H244b).

Total: **46 H-bars × 3 seeds = 138 cells**. Strong success means
all cells pass; partial success means ≥ 90 % pass with a recorded
honest reason.

### C. The 20 required dimensions

W65 must hit at minimum:

1. ≥ 12 real mechanism advances over W64 → covered by M1..M20.
2. ≥ 3 benchmark families → R-143/R-144/R-145.
3. ≥ 1 live/replay-live realism anchor → multi-agent comparator
   runs the same RNG-driven task across all four policies and
   compares matched-budget outcomes.
4. ≥ 1 true substrate-coupling bar → H223 (V10 substrate
   determinism + 4 new axes shape).
5. ≥ 1 latent-to-KV bar → H224 (KV bridge V10 six-target ridge
   converges; substrate-measured per-target margin).
6. ≥ 1 latent-to-hidden-state / prefix-state bar → H225 (HSB V9
   six-target ridge); H226 (prefix V9 K=64 drift curve).
7. ≥ 1 cache-reuse-vs-recompute bar → H230 (substrate checkpoint
   yields a measurable flop saving over recompute).
8. ≥ 1 trainable / fitted substrate-controller bar → H228 (cache
   controller V8 five-objective ridge); H229 (replay controller
   V6 eight-regime ridge + multi-agent abstain head).
9. ≥ 1 replay-dominance-primary bar → H229c (replay V6 dominance
   primary head still converges).
10. ≥ 1 hidden-wins-primary or hidden-beats-KV bar → H225b (HSB
    V9 hidden-wins-rate probe positive when hidden bridge actually
    dominates).
11. ≥ 1 long-horizon retention bar → H231 (persistent V17 14th
    skip carrier); H231b (max_chain_walk_depth ≥ 8192).
12. ≥ 1 merge / consensus / disagreement bar → H237 (consensus
    V11 16 stages); H239 (disagreement V11 Wasserstein identity).
13. ≥ 1 hostile-channel / corruption / adversarial bar → H237b
    (CRC V13 31-bit burst); H237c (8192-bucket detect).
14. ≥ 1 transcript-vs-shared head-to-head → H243 (TVS V14 15-arm
    sum to 1.0).
15. ≥ 1 explicit multi-agent task-success bar **(load-bearing)** →
    H230b (V10 multi-agent coordinator beats transcript-only,
    shared-state-proxy, and substrate-routed-V9 on matched-budget
    team success rate).
16. ≥ 1 abstain-or-fallback policy that materially matters →
    H229d (multi-agent abstain head fires under specified regime).
17. ≥ 1 explicit falsifier → H224b (KV V10 substrate-margin
    falsifier returns 0 under inversion); H239b (Wasserstein
    falsifier triggers).
18. ≥ 1 explicit limitation reproduction → H236b (ECC V17 2^27
    structural rate ceiling).
19. ≥ 1 serious cramming gain or robustness-vs-rate tradeoff →
    H236 (≥ 29.0 bits/visible-token at full emit, up from 27.333).
20. committed and pushed output if coherent → see §F below.

### D. Pre-committed numeric thresholds

- V10 substrate: 12 layers; four new axes wired.
- KV bridge V10: six targets; falsifier returns 0 under inversion.
- HSB V9: six targets; per-(L, H) hidden-wins rate measurable.
- Prefix V9 drift curve: K=64; converged.
- Attention V9: five-stage clamp keeps Hellinger ≤ 1.0.
- Cache controller V8: five-objective ridge converges.
- Replay controller V6: 8 regimes; ridge converges; multi-agent
  abstain head converges.
- Deep substrate hybrid V10: ten-way fires when all 10 axes fire.
- Persistent V17: 16 layers; chain depth ≥ 8192; rank ≥ 14.
- Multi-hop V15: 35 backends; chain-length 25; 10-axis composite.
- Mergeable capsule V13: team-substrate chain + role-conditioned
  chain inherit as union at merge.
- Consensus V11: 16 stages.
- CRC V13: 8192-bucket detect ≥ 0.95; 31-bit burst ≥ 0.40.
- LHR V17: 16 heads; `max_k=256`.
- ECC V17: 2^27 codes; ≥ 29.0 bits/visible-token at full emit.
- Uncertainty V13: 12-axis weighted composite ∈ [0, 1].
- Disagreement V11: Wasserstein-equivalence identity holds.
- TVS V14: 15 arms sum to 1.0.
- Multi-agent coordinator: V10 team success ≥ each baseline + 0.1
  (matched-budget); ≥ 50 % visible-token savings vs transcript-only.
- W65 envelope verifier: ≥ 100 disjoint failure modes.
- Cumulative trust boundary across W22..W65 ≥ 1100.

### E. Honest scope (pre-committed limitations)

- `W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` — hosted backends
  (Ollama, OpenAI-compatible) remain text-only on the HTTP surface;
  W65 makes no claim of third-party transformer-internal access.
- `W65-L-V10-NO-AUTOGRAD-CAP` — W65 fits only closed-form linear
  ridge solves. Six new W65 ridge solves on top of W64's 23 (cache
  V8 five-objective; replay V6 eight-regime; replay V6 multi-agent
  abstain head; replay V6 per-role per-regime; HSB V9 hidden-wins
  rate; KV V10 substrate-margin) = **twenty-nine total closed-form
  linear ridge solves**. No SGD, no autograd, no GPU.
- `W65-L-NUMPY-CPU-V10-SUBSTRATE-CAP` — V10 substrate is 12
  layers / d_model=64 / byte-vocab / max_len=128 / untrained NumPy
  on CPU. Not a frontier model.
- `W65-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP` — the multi-agent
  coordinator runs synthetic deterministic task instances; the
  team-success win is *measurable inside the W65 harness*, not on
  third-party hosted models.
- `W65-L-MULTI-HOP-V15-SYNTHETIC-BACKENDS-CAP` — the 35 multi-hop
  backends are NAMED, not EXECUTED.
- `W65-L-CRC-V13-FINGERPRINT-SYNTHETIC-CAP` — the 8192-bucket
  fingerprint is wrap-around XOR over the in-repo substrate cache,
  not third-party hosted cache state.
- `W65-L-ECC-V17-RATE-FLOOR-CAP` — structural rate ceiling
  log2(2^27) = 27 raw data bits per segment-tuple.
- `W65-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP` — the substrate
  checkpoint/restore primitive operates on the in-repo V10 cache
  only; it does not touch third-party hosted cache bytes.
- `W65-L-V6-REPLAY-NO-AUTOGRAD-CAP` — Replay controller V6 fits
  one 11-dim ridge regime head per role × 8 regimes + a 4×10
  multi-agent abstain head. No autograd.
- `W65-L-V8-CACHE-CONTROLLER-NO-AUTOGRAD-CAP` — Cache controller
  V8 fits a 5×4 stacked ridge + a per-role 6-dim eviction head.
  No autograd.

### F. What "shipped" means for W65

- Modules at `coordpy.tiny_substrate_v10`,
  `coordpy.kv_bridge_v10`, `coordpy.hidden_state_bridge_v9`,
  `coordpy.prefix_state_bridge_v9`,
  `coordpy.attention_steering_bridge_v9`,
  `coordpy.cache_controller_v8`,
  `coordpy.replay_controller_v6`,
  `coordpy.deep_substrate_hybrid_v10`,
  `coordpy.substrate_adapter_v10`,
  `coordpy.persistent_latent_v17`,
  `coordpy.multi_hop_translator_v15`,
  `coordpy.mergeable_latent_capsule_v13`,
  `coordpy.consensus_fallback_controller_v11`,
  `coordpy.corruption_robust_carrier_v13`,
  `coordpy.long_horizon_retention_v17`,
  `coordpy.ecc_codebook_v17`,
  `coordpy.uncertainty_layer_v13`,
  `coordpy.disagreement_algebra_v11`,
  `coordpy.transcript_vs_shared_arbiter_v14`,
  `coordpy.multi_agent_substrate_coordinator`,
  `coordpy.w65_team`,
  `coordpy.r143_benchmark`, `coordpy.r144_benchmark`,
  `coordpy.r145_benchmark` — reachable only through explicit
  imports. `coordpy.__version__` remains `0.5.20`. No PyPI
  release. Existing SDK contract is byte-for-byte unchanged.
- Tests at `tests/test_w65_modules.py`,
  `tests/test_w65_team_envelope_chain.py`,
  `tests/test_w65_trivial_passthrough_byte_identical.py`,
  `tests/test_r143_r144_r145_w65.py`.
- Docs: `docs/RESULTS_W65_TEAM_SUBSTRATE_COORDINATION.md`,
  this file. `RESEARCH_STATUS.md`, `THEOREM_REGISTRY.md`,
  `HOW_NOT_TO_OVERSTATE.md`, `context_zero_master_plan.md`,
  `papers/context_as_objects.md`, `CHANGELOG.md`,
  `docs/START_HERE.md` updated coherently.
- Committed and pushed if everything is green, with no Claude /
  AI authorship markers anywhere in the commit metadata or body.
