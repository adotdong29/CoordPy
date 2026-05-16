# Changelog

Release history for the coordpy SDK. For installation and usage,
see [`README.md`](README.md).

## Post-release research milestones (NOT releases; no version bump; no PyPI)

These are research-only additions on top of the released
0.5.20 line. Each ships at an explicit-import path and is NOT
re-exported through `coordpy.__init__` or
`coordpy.__experimental__`. The released SDK contract
(`coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, the smoke driver,
the public symbols) is byte-for-byte unchanged.

- **W69 Stronger Solving-Context Two-Plane Multi-Agent Substrate**
  (post-W68, 2026-05-16) — *fourteenth substrate-attack milestone;
  fifth **multi-agent task-success-bearing** substrate milestone
  (first to win across **nine** regimes: W68's seven plus
  ``multi_branch_rejoin_after_divergent_work`` plus
  ``silent_corruption_plus_member_replacement``); first milestone to
  **operationalise the two-plane split with a Plane A↔B handoff
  coordinator** that records content-addressed handoff envelopes
  per turn while preserving the W68 wall as an invariant*. Twenty-
  eight orthogonal advances on top of W68 (22 Plane B + 5 Plane A
  V2 + 1 new handoff coordinator). Plane B: (M1) **Tiny Substrate
  V14** (16 layers; four new V14 axes — per-(L, H, T) multi-branch-
  rejoin witness tensor, per-role silent-corruption witness with
  member-replacement flag, substrate self-checksum CID, per-layer
  V14 composite gate score); (M2) **KV Bridge V14** (ten-target
  stacked ridge; multi-branch-rejoin margin probe; 60-dim silent-
  corruption fingerprint; multi-branch-rejoin falsifier); (M3)
  **HSB V13** (ten-target stacked ridge with multi-branch-rejoin
  target; per-(L, H) hidden-vs-multi-branch-rejoin probe); (M4)
  **Prefix V13** (K=256 drift curve; role+task+team+branch+
  corruption+rejoin 60-dim fingerprint; eight-way prefix/hidden/
  replay/team/recover/branch/contradict/multi_branch_rejoin
  comparator); (M5) **Attention V13** (nine-stage clamp adding
  multi-branch-rejoin attention bias; multi-branch-conditioned
  fingerprint); (M6) **Cache Controller V12** (nine-objective
  stacked ridge adding multi-branch-rejoin; per-role 10-dim
  silent-corruption priority head); (M7) **Replay Controller V10**
  (16 regimes adding multi-branch-rejoin and silent-corruption;
  per-role per-regime ridge; trained multi-branch-rejoin-routing
  head); (M8) **Deep Substrate Hybrid V14** (fourteen-way
  bidirectional loop with the four new V14 axes + team-consensus-
  controller-V4 axis); (M9) **Substrate Adapter V14** (4 new V14
  capability axes; new ``substrate_v14_full`` tier); (M10)
  **Persistent Latent V21** (20 layers; eighteenth skip carrier
  ``multi_branch_rejoin_carrier``; max_chain_walk_depth=65536;
  distractor rank 20); (M11) **Multi-Hop Translator V19** (48
  backends; 2256 directed edges; chain-length 38; 14-axis
  composite adding ``multi_branch_rejoin_reconciliation_trust``);
  (M12) **Mergeable Latent Capsule V17** (``multi_branch_rejoin_
  witness_chain`` and ``silent_corruption_witness_chain``); (M13)
  **Consensus Fallback Controller V15** (24-stage chain inserting
  ``multi_branch_rejoin_arbiter`` and
  ``silent_corruption_plus_member_replacement_arbiter``); (M14)
  **Corruption-Robust Carrier V17** (131072-bucket fingerprint;
  37-bit adversarial burst; silent-corruption recovery probe);
  (M15) **Long-Horizon Retention V21** (20 heads, max_k=512,
  eleven-layer scorer adding random+swish); (M16) **ECC Codebook
  V21** (K1..K20 = 2^35 = 34 359 738 368 codes; **37+ bits/visible-
  token** at full emit); (M17) **Uncertainty Layer V17** (16-axis
  composite adding ``multi_branch_rejoin_resolution_fidelity``);
  (M18) **Disagreement Algebra V15** (multi-branch-rejoin-
  equivalence identity + falsifier); (M19) **Nineteen-arm TVS
  Arbiter V18** (19 arms with ``multi_branch_rejoin_resolution``
  arm); (M20) **Multi-Agent Substrate Coordinator V5** (the load-
  bearing W69 mechanism — twelve matched-budget policies across
  nine regimes; V14 strictly beats V13 across all nine regimes,
  ≥ 60 % of seeds per regime, peaking at 86.7 % in
  multi_branch_rejoin); (M21) **Team-Consensus Controller V4**
  (regime-aware weighted quorum + multi-branch-rejoin arbiter +
  silent-corruption-plus-replacement arbiter + substrate-replay
  fallback + transcript fallback). Plane A V2: (H1) **Hosted
  Router V2** (weighted score + sticky + blacklist + cost-per-
  success bookkeeping); (H2) **Hosted Logprob Router V2**
  (Bayesian Dirichlet fusion + per-provider trust + tiebreak
  fallback); (H3) **Hosted Cache-Aware Planner V2** (per-role
  staggered prefix; ≥ 60 % savings on 6×8 at hit_rate=1.0);
  (H4) **Hosted Provider Filter V2** (compositional ALL/ANY
  chaining); (H5) **Hosted Cost Planner V2** (multi-turn schedule
  + cost-per-success ratio); (H6) **Hosted/Real Substrate
  Boundary V2** (19 blocked axes — W68's 15 + 4 V14; 3 frontier-
  blocked axes; falsifier). And the new (H7) **Hosted-Real Handoff
  Coordinator** — the load-bearing operational W69 advance — per-
  turn content-addressed handoff envelopes routing to
  ``hosted_only`` / ``real_substrate_only`` /
  ``hosted_with_real_substrate_audit`` / ``abstain``; falsifier;
  ≥ 60 % cross-plane token savings at default workload. R-156
  (10 H-bars) + R-157 (17) + R-158 (18) + R-159 (9) + R-160 (8)
  deliver **62 H-bars × 3 seeds = 186 cells, all pass**.
  Cumulative trust boundary across W22..W69 = **1461 enumerated
  failure modes** (1417 from W22..W68 + 44 new W69 envelope
  verifier modes). Ships at ``coordpy.tiny_substrate_v14``,
  ``coordpy.kv_bridge_v14``,
  ``coordpy.hidden_state_bridge_v13``,
  ``coordpy.prefix_state_bridge_v13``,
  ``coordpy.attention_steering_bridge_v13``,
  ``coordpy.cache_controller_v12``,
  ``coordpy.replay_controller_v10``,
  ``coordpy.persistent_latent_v21``,
  ``coordpy.multi_hop_translator_v19``,
  ``coordpy.mergeable_latent_capsule_v17``,
  ``coordpy.consensus_fallback_controller_v15``,
  ``coordpy.corruption_robust_carrier_v17``,
  ``coordpy.long_horizon_retention_v21``,
  ``coordpy.ecc_codebook_v21``,
  ``coordpy.transcript_vs_shared_arbiter_v18``,
  ``coordpy.uncertainty_layer_v17``,
  ``coordpy.disagreement_algebra_v15``,
  ``coordpy.deep_substrate_hybrid_v14``,
  ``coordpy.substrate_adapter_v14``,
  ``coordpy.multi_agent_substrate_coordinator_v5``,
  ``coordpy.team_consensus_controller_v4``,
  ``coordpy.w69_team``, ``coordpy.hosted_router_controller_v2``,
  ``coordpy.hosted_logprob_router_v2``,
  ``coordpy.hosted_cache_aware_planner_v2``,
  ``coordpy.hosted_provider_filter_v2``,
  ``coordpy.hosted_cost_planner_v2``,
  ``coordpy.hosted_real_substrate_boundary_v2``,
  ``coordpy.hosted_real_handoff_coordinator``,
  ``coordpy.r156_benchmark`` / ``coordpy.r157_benchmark`` /
  ``coordpy.r158_benchmark`` / ``coordpy.r159_benchmark`` /
  ``coordpy.r160_benchmark``, reachable only via explicit import.
  ``coordpy.__version__`` remains ``0.5.20``; SDK contract is
  byte-for-byte unchanged; no PyPI release. Honest scope:
  ``W69-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` (hosted backends
  still text-only); ``W69-L-NUMPY-CPU-V14-SUBSTRATE-CAP`` (16-
  layer in-repo NumPy substrate, not a frontier model);
  ``W69-L-V14-NO-AUTOGRAD-CAP`` (closed-form ridge throughout);
  ``W69-L-MULTI-AGENT-COORDINATOR-V5-SYNTHETIC-CAP`` (multi-agent
  wins are synthetic-harness-load-bearing);
  ``W69-L-HANDOFF-NOT-CROSSING-WALL-CAP`` (handoff coordinator
  preserves the wall as an invariant — it does NOT cross the
  substrate boundary); ``W69-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-
  CAP`` (third-party hosted-model substrate access remains the
  unsolved research-line wall).

- **W68 Two-Plane Substrate-Coupled Latent Operating System**
  (post-W67, 2026-05-16) — *thirteenth substrate-attack milestone;
  fourth **multi-agent task-success-bearing** substrate milestone
  (first to win across **seven** regimes: W67's five +
  partial-contradiction-under-delayed-reconciliation +
  agent-replacement-warm-restart); first milestone to **explicitly
  split the architecture into two planes** (hosted control plane
  vs real substrate plane) and codify the hosted ↔ real-substrate
  wall as a first-class artefact*. Twenty-eight orthogonal advances
  on top of W67 (22 Plane B + 6 Plane A). Plane B: (M1) **Tiny
  Substrate V13** (15 layers; four new V13 axes — per-(L, H, T)
  partial-contradiction witness tensor, per-role
  agent-replacement-flag with warm-restart window, substrate
  prefix-reuse counter, per-layer V13 composite gate score); (M2)
  **KV Bridge V13** (nine-target stacked ridge; partial-
  contradiction margin probe; 50-dim agent-replacement fingerprint;
  partial-contradiction falsifier); (M3) **HSB V12** (nine-target
  stacked ridge with agent-replacement target; per-(L, H)
  hidden-vs-agent-replacement probe); (M4) **Prefix V12** (K=192
  drift curve; role+task+team+branch+agent 50-dim fingerprint;
  seven-way prefix/hidden/replay/team/recover/branch/contradict
  comparator); (M5) **Attention V12** (eight-stage clamp adding
  partial-contradiction attention bias; agent-conditioned
  fingerprint); (M6) **Cache Controller V11** (eight-objective
  stacked ridge adding partial-contradiction; per-role 9-dim
  agent-replacement priority head); (M7) **Replay Controller V9**
  (14 regimes adding `partial_contradiction_under_delayed_
  reconciliation_regime` and `agent_replacement_warm_restart_
  regime`; per-role per-regime ridge; trained
  **agent-replacement-routing head** with 6 routing labels × 11
  features ridge); (M8) **Deep Substrate Hybrid V13** (thirteen-way
  bidirectional loop adding partial-contradiction-witness axis);
  (M9) **Substrate Adapter V13** (new `substrate_v13_full` tier;
  V13 capability axes); (M10) **Persistent V20** (19 layers;
  seventeenth skip carrier `partial_contradiction_carrier`;
  max_chain_walk_depth=32768; distractor rank 19); (M11) **Multi-
  Hop V18** (44 backends; 44×43=1892 directed edges; chain-len 34;
  13-axis composite adding
  `partial_contradiction_reconciliation_trust`); (M12) **Mergeable
  Latent Capsule V16** (partial-contradiction + agent-replacement
  witness chains); (M13) **Consensus V14** (22 stages: adds
  `partial_contradiction_arbiter` and
  `agent_replacement_warm_restart_arbiter`); (M14) **CRC V16**
  (65536-bucket fingerprint; 36-bit adversarial burst; partial-
  contradiction recovery probe); (M15) **LHR V20** (19 heads;
  max_k=448; ten-layer scorer); (M16) **ECC V20** (2^33 = 8 589
  934 592 codes; **≥ 35.0 bits/visible-token** at full emit);
  (M17) **Uncertainty V16** (15-axis composite adding
  `partial_contradiction_resolution_fidelity`); (M18) **Disagreement
  V14** (agent-replacement-equivalence identity + falsifier);
  (M19) **TVS V17** (18 arms with `partial_contradiction_
  resolution`); (M20) **MASC V4** (ten-policy harness across
  **seven** regimes; V13 strictly beats V12 across all seven);
  (M21) **Team-Consensus Controller V3** (partial-contradiction
  arbiter + agent-replacement-warm-restart arbiter); (M22)
  **W68Team** orchestrator. Plane A: (H1) **HostedRouterController**
  (content-addressed provider registry + capability-aware routing
  graph); (H2) **HostedLogprobRouter** (honest top-k logprob fusion
  on shared top-k + text-only quorum fallback); (H3)
  **HostedCacheAwarePlanner** (per-turn prefix-CID planner +
  cross-plane bridge to V13 prefix-reuse counter); (H4)
  **HostedProviderFilter** (data-policy- and tier-aware registry
  filter); (H5) **HostedCostPlanner** (cost/latency-aware provider
  selection under matched-quality constraint); (H6)
  **HostedRealSubstrateBoundary** (explicit architecture wall +
  falsifier + wall report). W68 fits **six** new closed-form ridge
  solves on top of W67's 41 (cache V11 eight-objective; cache V11
  per-role agent-replacement; replay V9 per-role per-regime;
  replay V9 agent-replacement-routing; HSB V12 nine-target; KV V13
  nine-target). Total **47 ridge solves across W61..W68**. No SGD
  / autograd / GPU. **R-152** (10 H-bars Plane A) + **R-153** (16
  H-bars Plane B) + **R-154** (14 H-bars seven-regime multi-agent
  task success) + **R-155** (6 H-bars hosted-vs-real wall) deliver
  46/46 H-bars at 3/3 seeds (138/138 cells). Cumulative trust
  boundary across W22..W68 = **1417 enumerated failure modes**.
  Ships at `coordpy.tiny_substrate_v13`, `coordpy.kv_bridge_v13`,
  `coordpy.hidden_state_bridge_v12`,
  `coordpy.prefix_state_bridge_v12`,
  `coordpy.attention_steering_bridge_v12`,
  `coordpy.cache_controller_v11`,
  `coordpy.replay_controller_v9`,
  `coordpy.persistent_latent_v20`,
  `coordpy.multi_hop_translator_v18`,
  `coordpy.mergeable_latent_capsule_v16`,
  `coordpy.consensus_fallback_controller_v14`,
  `coordpy.corruption_robust_carrier_v16`,
  `coordpy.long_horizon_retention_v20`,
  `coordpy.ecc_codebook_v20`,
  `coordpy.transcript_vs_shared_arbiter_v17`,
  `coordpy.uncertainty_layer_v16`,
  `coordpy.disagreement_algebra_v14`,
  `coordpy.deep_substrate_hybrid_v13`,
  `coordpy.substrate_adapter_v13`,
  `coordpy.multi_agent_substrate_coordinator_v4`,
  `coordpy.team_consensus_controller_v3`, `coordpy.w68_team`,
  `coordpy.hosted_router_controller`,
  `coordpy.hosted_logprob_router`,
  `coordpy.hosted_cache_aware_planner`,
  `coordpy.hosted_provider_filter`,
  `coordpy.hosted_cost_planner`,
  `coordpy.hosted_real_substrate_boundary`,
  `coordpy.r152_benchmark` / `coordpy.r153_benchmark` /
  `coordpy.r154_benchmark` / `coordpy.r155_benchmark`. Limitations:
  `W68-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`,
  `W68-L-V13-NO-AUTOGRAD-CAP`,
  `W68-L-HOSTED-NO-SUBSTRATE-CAP`,
  `W68-L-HOSTED-ESTIMATES-CALLER-CAP`,
  `W68-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP`,
  `W68-L-MULTI-AGENT-COORDINATOR-V4-SYNTHETIC-CAP`,
  `W68-L-TEAM-CONSENSUS-V3-IN-REPO-CAP`. Public SDK unchanged:
  `coordpy.__version__ == "0.5.20"`. No PyPI release.

- **W67 Stronger Branch-Merge / Role-Dropout Substrate-Coupled
  Latent Operating System** (post-W66, 2026-05-16) — *twelfth
  substrate-attack milestone; third **multi-agent task-success-
  bearing** substrate milestone (first to win across **five**
  regimes: baseline + team-consensus-under-budget +
  team-failure-recovery + role-dropout +
  branch-merge-reconciliation)*. Twenty-one orthogonal advances
  on top of W66: (M1) **Tiny Substrate V12** (14 layers; four new
  V12 axes — per-(L, H, T) branch-merge witness tensor,
  per-role-pair role-dropout-recovery flag, substrate snapshot-
  fork primitive, per-layer V12 composite gate score); (M2)
  **KV Bridge V12** (eight-target stacked ridge; branch-merge
  margin probe; role-pair 40-dim fingerprint;
  branch-merge-reconciliation falsifier); (M3) **HSB V11**
  (eight-target stacked ridge; per-(L, H) hidden-vs-branch-merge
  probe; branch-merge margin); (M4) **Prefix V11** (K=128 drift
  curve; role+task+team+branch 40-dim fingerprint; six-way
  prefix/hidden/replay/team/recover/branch comparator); (M5)
  **Attention V11** (seven-stage clamp adding branch-merge
  attention bias; branch-conditioned fingerprint); (M6) **Cache
  Controller V10** (seven-objective stacked ridge adding
  branch-merge; per-role 8-dim eviction head); (M7) **Replay
  Controller V8** (12 regimes adding `role_dropout_regime` and
  `branch_merge_reconciliation_regime`; per-role per-regime
  ridge; **trained branch-merge-routing head**); (M8) **Deep
  Substrate Hybrid V12** (twelve-way bidirectional loop); (M9)
  **Substrate Adapter V12** (new `substrate_v12_full` tier);
  (M10) **Persistent V19** (18 layers; sixteenth skip carrier
  `role_dropout_recovery_carrier`; max_chain_walk_depth=16384;
  distractor rank 18); (M11) **Multi-Hop V17** (40 backends; 1560
  directed edges; chain-len 30; 12-axis composite); (M12)
  **Mergeable Latent Capsule V15** (role-dropout-recovery +
  branch-merge-reconciliation witness chains); (M13) **Consensus
  V13** (20 stages: adds `role_dropout_arbiter` and
  `branch_merge_reconciliation_arbiter`); (M14) **CRC V15**
  (32768-bucket fingerprint; 35-bit adversarial burst; branch-
  merge reconciliation probe); (M15) **LHR V19** (18 heads;
  max_k=384; nine-layer scorer); (M16) **ECC V19** (2^31 = 2 147
  483 648 codes; **33.333 bits/visible-token** at full emit);
  (M17) **Uncertainty V15** (14-axis composite adding
  `branch_merge_reconciliation_fidelity`); (M18) **Disagreement
  V13** (Bregman-equivalence identity + falsifier); (M19)
  **TVS V16** (17 arms with `branch_merge_reconciliation`); (M20)
  **MASC V3** (eight-policy harness across **five** regimes; V12
  strictly beats V11 across all five); (M21) **Team-Consensus
  Controller V2** (branch-merge arbiter + role-dropout repair).
  W67 fits **six** new closed-form ridge solves on top of W66's
  35 (cache V10 seven-objective; cache V10 per-role eviction;
  replay V8 per-role per-regime; replay V8 branch-merge-routing;
  HSB V11 eight-target; KV V12 eight-target). Total **41 ridge
  solves across W61..W67**. No SGD / autograd / GPU. **R-149**
  (24 cell families) + **R-150** (14) + **R-151** (18) deliver
  56/56 H-bars at 3/3 seeds (168/168 cells). Cumulative trust
  boundary across W22..W67 = **1375 enumerated failure modes**.
  Ships at `coordpy.tiny_substrate_v12`, `coordpy.kv_bridge_v12`,
  `coordpy.hidden_state_bridge_v11`,
  `coordpy.prefix_state_bridge_v11`,
  `coordpy.attention_steering_bridge_v11`,
  `coordpy.cache_controller_v10`,
  `coordpy.replay_controller_v8`,
  `coordpy.persistent_latent_v19`,
  `coordpy.multi_hop_translator_v17`,
  `coordpy.mergeable_latent_capsule_v15`,
  `coordpy.consensus_fallback_controller_v13`,
  `coordpy.corruption_robust_carrier_v15`,
  `coordpy.long_horizon_retention_v19`,
  `coordpy.ecc_codebook_v19`,
  `coordpy.transcript_vs_shared_arbiter_v16`,
  `coordpy.uncertainty_layer_v15`,
  `coordpy.disagreement_algebra_v13`,
  `coordpy.deep_substrate_hybrid_v12`,
  `coordpy.substrate_adapter_v12`,
  `coordpy.multi_agent_substrate_coordinator_v3`,
  `coordpy.team_consensus_controller_v2`, `coordpy.w67_team`,
  `coordpy.r149_benchmark` / `coordpy.r150_benchmark` /
  `coordpy.r151_benchmark`. Limitations: `W67-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`, `W67-L-V12-NO-AUTOGRAD-CAP`,
  `W67-L-SUBSTRATE-BRANCH-MERGE-IN-REPO-CAP`,
  `W67-L-TEAM-CONSENSUS-V2-IN-REPO-CAP`,
  `W67-L-MULTI-AGENT-COORDINATOR-V3-SYNTHETIC-CAP`. Public SDK
  unchanged: `coordpy.__version__ == "0.5.20"`. No PyPI release.

- **W66 Stronger Solving-Context Substrate-Coupled Latent
  Operating System** (post-W65, 2026-05-16) — *eleventh substrate-
  attack milestone; second **multi-agent task-success-bearing**
  substrate milestone (first to win across multiple regimes)*.
  Twenty-one orthogonal advances on top of W65: (M1) **Tiny
  Substrate V11** (13 layers; four new V11 axes — per-(L, H, T)
  replay-trust ledger, per-role team-failure-recovery flag,
  substrate snapshot-diff primitive, per-layer V11 composite gate
  score); (M2) **KV Bridge V11** (seven-target stacked ridge;
  team-coordination margin probe; multi-agent task fingerprint;
  team-failure-recovery falsifier); (M3) **HSB V10** (seven-target
  stacked ridge; per-(L, H) hidden-wins-vs-team-success probe;
  team-consensus margin); (M4) **Prefix V10** (K=96 drift curve;
  role+task+team 30-dim fingerprint; five-way prefix/hidden/replay/
  team/recover comparator); (M5) **Attention V10** (six-stage clamp
  adding per-(L, H) attention-trust ledger; team-conditioned
  fingerprint); (M6) **Cache Controller V9** (six-objective stacked
  ridge adding team-failure-recovery; per-role 7-dim eviction
  head); (M7) **Replay Controller V7** (10 regimes adding
  `team_failure_recovery_regime` and `team_consensus_under_budget_regime`;
  per-role per-regime ridge; **trained team-substrate-routing
  head**); (M8) **Deep Substrate Hybrid V11** (eleven-way
  bidirectional loop); (M9) **Substrate Adapter V11** (new
  `substrate_v11_full` tier); (M10) **Persistent V18** (17 layers;
  fifteenth skip carrier `team_failure_recovery_carrier`;
  max_chain_walk_depth=8192; distractor rank 16); (M11)
  **Multi-Hop V16** (36 backends; 1260 directed edges; chain-len
  26; 11-axis composite); (M12) **Mergeable Latent Capsule V14**
  (team-failure-recovery + team-consensus-under-budget witness
  chains); (M13) **Consensus V12** (18 stages: adds
  `team_failure_recovery_arbiter` and
  `team_consensus_under_budget_arbiter`); (M14) **CRC V14**
  (16384-bucket fingerprint; 33-bit adversarial burst; team-
  failure-recovery probe); (M15) **LHR V18** (17 heads; max_k=320;
  eight-layer scorer); (M16) **ECC V18** (2^29 = 536 870 912 codes;
  **31.0 bits/visible-token**); (M17) **Uncertainty V14** (13-axis
  composite); (M18) **Disagreement Algebra V12** (Jensen-Shannon
  equivalence identity); (M19) **TVS V15** (16 arms); (M20)
  **Multi-Agent Substrate Coordinator V2** (the load-bearing W66
  multi-agent mechanism — six matched-budget policies and three
  regimes; V11 policy strictly beats V10 on ≥ 93 % of seeds at
  baseline regime; TSC_V11 policy strictly beats V11 on ≥ 80 % of
  baseline seeds and ≥ 73 % of failure-recovery seeds); (M21)
  **Team-Consensus Controller** (weighted quorum + abstain +
  substrate-replay fallback + transcript fallback, regime-aware).
  W66 fits **six new closed-form ridge solves** on top of W65's 29
  (35 total across W61..W66). 123 new W66 envelope verifier
  failure modes (cumulative trust boundary across W22..W66 =
  **1228 enumerated failure modes**). R-146 (24 H-bars) + R-147
  (14 H-bars) + R-148 (18 H-bars) at 3 seeds = **168/168 cells
  pass** (strong success). Ships at `coordpy.tiny_substrate_v11`,
  `coordpy.kv_bridge_v11`, `coordpy.hidden_state_bridge_v10`,
  `coordpy.prefix_state_bridge_v10`,
  `coordpy.attention_steering_bridge_v10`,
  `coordpy.cache_controller_v9`,
  `coordpy.replay_controller_v7`,
  `coordpy.deep_substrate_hybrid_v11`,
  `coordpy.substrate_adapter_v11`,
  `coordpy.persistent_latent_v18`,
  `coordpy.multi_hop_translator_v16`,
  `coordpy.mergeable_latent_capsule_v14`,
  `coordpy.consensus_fallback_controller_v12`,
  `coordpy.corruption_robust_carrier_v14`,
  `coordpy.long_horizon_retention_v18`,
  `coordpy.ecc_codebook_v18`,
  `coordpy.uncertainty_layer_v14`,
  `coordpy.disagreement_algebra_v12`,
  `coordpy.transcript_vs_shared_arbiter_v15`,
  `coordpy.multi_agent_substrate_coordinator_v2`,
  `coordpy.team_consensus_controller`,
  `coordpy.w66_team`, `coordpy.r146_benchmark`,
  `coordpy.r147_benchmark`, `coordpy.r148_benchmark` — reachable
  only through explicit imports. `coordpy.__version__` remains
  `0.5.20`. Pre-committed bar:
  [`docs/SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md`](docs/SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md).
  Result note:
  [`docs/RESULTS_W66_STRONGER_TEAM_SUBSTRATE_OS.md`](docs/RESULTS_W66_STRONGER_TEAM_SUBSTRATE_OS.md).

- **W65 Team-Substrate-Coordination Substrate-Coupled Latent
  Operating System** (post-W64, 2026-05-16) — *tenth substrate-
  attack milestone; first **multi-agent task-success-bearing**
  substrate milestone in the programme*. Twenty orthogonal
  advances on top of W64: (M1) **Tiny Substrate V10** (12 layers;
  four new V10 axes — per-(L, H, T) hidden-write-merit, per-role
  KV bank with FIFO eviction, substrate checkpoint/restore
  primitive, per-layer V10 composite gate score); (M2) **KV
  Bridge V10** (six-target stacked ridge fit; substrate-measured
  per-target margin probe; team-task falsifier); (M3) **HSB V9**
  (six-target stacked ridge; per-(L, H) hidden-wins-rate probe;
  team-coordination margin); (M4) **Prefix V9** (K=64 drift curve;
  role+task 20-dim fingerprint; four-way prefix/hidden/replay/team
  comparator); (M5) **Attention V9** (five-stage clamp: Hellinger
  + JS + coarse L1 + fine KL + max-position cap; substrate-measured
  attention-map fingerprint); (M6) **Cache Controller V8**
  (five-objective stacked ridge adding team-task-success; per-role
  eviction head); (M7) **Replay Controller V6** (8 regimes; per-
  role per-regime ridge; **multi-agent abstain head**); (M8)
  **Deep Substrate Hybrid V10** (ten-way bidirectional loop);
  (M9) **Substrate Adapter V10** (new `substrate_v10_full` tier);
  (M10) **Persistent V17** (16 layers; fourteenth skip carrier;
  max_chain_walk_depth=8192; distractor rank 14); (M11) **Multi-Hop
  Translator V15** (35 backends; 1190 directed edges; chain-len
  25; 10-axis composite); (M12) **Mergeable Latent Capsule V13**
  (team-substrate + role-conditioned witness chains); (M13)
  **Consensus V11** (16 stages: adds `team_substrate_coordination_arbiter`
  and `multi_agent_abstain_arbiter`); (M14) **CRC V13** (8192-bucket
  fingerprint; 31-bit adversarial burst); (M15) **LHR V17** (16
  heads; max_k=256; seven-layer scorer); (M16) **ECC V17** (2^27 =
  134 217 728 codes; **29.333 bits/visible-token**); (M17)
  **Uncertainty V13** (12-axis composite); (M18) **Disagreement
  Algebra V11** (Wasserstein-equivalence identity); (M19) **TVS
  V14** (15 arms); (M20) **Multi-Agent Substrate Coordinator**
  (the load-bearing W65 mechanism — real N-agent multi-agent
  harness with four matched-budget policies; V10 policy strictly
  beats `transcript_only`, `shared_state_proxy`, and
  `substrate_routed_v9` on ≥ 50 % of seeds at ≤ 17 % of the
  transcript-only token budget). 103 new W65 envelope verifier
  failure modes (cumulative trust boundary across W22..W65 =
  **1105 enumerated failure modes**). Ships at
  `coordpy.tiny_substrate_v10`, `coordpy.kv_bridge_v10`,
  `coordpy.hidden_state_bridge_v9`,
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
  `coordpy.w65_team`, `coordpy.r143_benchmark`,
  `coordpy.r144_benchmark`, `coordpy.r145_benchmark` — reachable
  only through explicit imports. `coordpy.__version__` remains
  `0.5.20`. Pre-committed bar:
  [`docs/SUCCESS_CRITERION_W65_TEAM_SUBSTRATE_COORDINATION.md`](docs/SUCCESS_CRITERION_W65_TEAM_SUBSTRATE_COORDINATION.md).
  Result note:
  [`docs/RESULTS_W65_TEAM_SUBSTRATE_COORDINATION.md`](docs/RESULTS_W65_TEAM_SUBSTRATE_COORDINATION.md).

- **W64 Replay-Dominance-Primary Hidden-Wins-Primary 6144-Turn
  Nine-Way Substrate-Coupled Latent Operating System** (post-W63,
  2026-05-15) — *ninth substrate-attack milestone; first
  **per-(layer, head, slot) hidden-wins-primary tensor** inside
  an in-repo NumPy substrate (V9), first **per-(layer, head,
  slot) replay-dominance witness channel**, first **per-layer
  attention-entropy probe** (sigmoid of attention-distribution
  Shannon entropy), first **per-(layer, head, slot, slot) cache-
  similarity matrix** (cosine over cache_keys), first **per-
  (layer, head) hidden-state-trust ledger** (EMA over HSB V8
  decisions), first **trained four-way bridge classifier** (4×9
  closed-form ridge over 9-dim regime features against `kv_wins` /
  `hidden_wins` / `prefix_wins` / `replay_wins`), first **trained
  seven-regime per-regime replay head** (replay controller V5,
  seven regimes × 10×4 ridge head + 9-dim regime gate, includes
  `replay_dominance_primary_regime`), first **trained replay-
  dominance-primary head** (replay controller V5, 4×9 ridge head
  that favours REUSE under high replay-dominance), first **trained
  similarity-aware eviction head** (cache controller V7, 6-dim
  ridge over [flag_count, hidden_write, replay_age,
  attention_receive_l1, cache_key_norm, mean_similarity_to_others]),
  first **four-objective stacked ridge fit** (cache controller V7,
  drop-oracle + retrieval-relevance + hidden-wins + replay-
  dominance simultaneously), first **token+role-conditional
  drift-curve predictor** (prefix V8, stacked 12×K ridge over
  [reuse_len, recompute_len, drop_len, 4-D SHA256 token
  fingerprint, 4-D SHA256 role fingerprint, drift-acceleration,
  1] against K-step drift curve target), first **four-stage
  attention clamp** (attention V8, Hellinger budget + JS budget +
  coarse L1-mass clamp + fine per-(L,H,Q,K) KL clamp), first
  **nine-way bidirectional substrate ↔ V9 ↔ cache controller V7
  ↔ replay controller V5 ↔ retrieval head ↔ attention-steering V8
  ↔ four-way bridge classifier ↔ prefix-state-bridge V8 ↔ hidden-
  state-bridge V8** hybrid loop (deep substrate hybrid V9), first
  **hidden-wins-primary falsifier** (`probe_kv_bridge_v9_hidden_
  wins_primary_falsifier` returns 0 exactly when inverting the
  flag flips the decision), first **multi-hop nine-axis composite
  trust** (multi-hop V14, 27 backends × 26 directed edges = 702,
  chain-length-21, compromise threshold ∈ [1, 9]), first **15-
  layer LHR** (V16: 15 reconstruction heads + max_k=192 + six-layer
  scorer with random+silu fifth layer), first **2^25 = 33 554 432
  ECC codebook** (V16: K1..K15, 27.333 bits/visible-token at full
  emit ≥ 27.0 target), first **15-layer persistent latent** (V16
  with thirteenth skip carrier (replay-dominance-witness EMA),
  max_chain_walk_depth=6144, distractor rank 12), first
  **fourteen-arm TVS** (V13: V12's thirteen + replay_dominance_
  primary), first **eleven-axis uncertainty** (V12 adds replay_
  dominance_primary_fidelity), first **TV-equivalence identity +
  falsifier** (disagreement V10), first **fourteen-stage consensus
  chain** (V10: V9's thirteen + `replay_dominance_primary_arbiter`
  between `hidden_wins_arbiter` and `best_parent`), first
  **mergeable capsule V12** (adds `replay_dominance_primary_witness_
  chain`, `hidden_state_trust_witness_chain`, `disagreement_total_
  variation_distance`, two new algebra signatures), first **CRC
  V12 4096-bucket fingerprint + 23-bit adversarial burst + replay-
  dominance recovery probe + substrate-corruption blast-radius
  probe**, first **substrate adapter V9** with five new capability
  axes and `substrate_v9_full` tier. R-137 (20 cell families) +
  R-138 (12) + R-139 (19) deliver **51/51 H-bars (H203..H222b)
  pass 3/3 seeds (153/153 cells)**. ``W64-L-V9-NO-AUTOGRAD-CAP``
  documents that all "training" remains single-step closed-form
  linear ridge over a small subspace; no SGD / autograd / GPU.
  Cumulative trust boundary across W22..W64 = **1002 enumerated
  failure modes** (916 from W22..W63 + 86 new W64 envelope verifier
  modes). W64 ships at ``coordpy.tiny_substrate_v9``,
  ``coordpy.kv_bridge_v9``, ``coordpy.hidden_state_bridge_v8``,
  ``coordpy.prefix_state_bridge_v8``,
  ``coordpy.attention_steering_bridge_v8``,
  ``coordpy.cache_controller_v7``,
  ``coordpy.replay_controller_v5``,
  ``coordpy.persistent_latent_v16``,
  ``coordpy.multi_hop_translator_v14``,
  ``coordpy.mergeable_latent_capsule_v12``,
  ``coordpy.consensus_fallback_controller_v10``,
  ``coordpy.corruption_robust_carrier_v12``,
  ``coordpy.long_horizon_retention_v16``,
  ``coordpy.ecc_codebook_v16``,
  ``coordpy.transcript_vs_shared_arbiter_v13``,
  ``coordpy.uncertainty_layer_v12``,
  ``coordpy.disagreement_algebra_v10``,
  ``coordpy.deep_substrate_hybrid_v9``,
  ``coordpy.substrate_adapter_v9``, ``coordpy.w64_team``,
  ``coordpy.r137_benchmark`` / ``coordpy.r138_benchmark`` /
  ``coordpy.r139_benchmark`` — reachable only through explicit
  imports; the released v0.5.20 wheel's public surface is byte-
  for-byte unchanged; no PyPI release.*

- **W63 Stronger Replay-Dominance Hidden-Wins 4096-Turn
  Substrate-Coupled Latent Operating System** (post-W62,
  2026-05-15) — *eighth substrate-attack milestone; first
  **per-(layer, head, slot) hidden-vs-KV contention tensor**
  inside an in-repo NumPy substrate (V8), first **per-layer
  hidden-state confidence probe**, first **per-(layer, head,
  slot) replay-determinism channel**, first **per-(layer, head)
  prefix-reuse trust ledger**, first **per-(L, H, L, H) cross-
  layer-head coupling matrix**, first **trained three-way
  bridge classifier** (closed-form ridge over 7-dim regime
  features against `kv_wins` / `hidden_wins` / `prefix_wins`),
  first **trained six-regime per-regime replay head** (replay
  controller V4, six regimes × 8×4 ridge head + 7-dim regime
  gate), first **trained retrieval-repair head** (cache
  controller V6, 5-dim ridge over [flag_count, hidden_write,
  replay_age, attention_receive_l1, cache_key_norm]), first
  **three-objective stacked ridge fit** (cache controller V6,
  drop-oracle + retrieval-relevance + hidden-wins
  simultaneously), first **token-content-conditional drift-
  curve predictor** (prefix V7, stacked 8×K ridge over
  [reuse_len, recompute_len, drop_len, 4-D SHA256 token
  fingerprint, 1] against K-step drift curve target), first
  **three-stage attention clamp** (attention V7, JS budget +
  coarse L1-mass clamp + fine per-(L,H,Q,K) KL clamp), first
  **eight-way bidirectional substrate ↔ V8 ↔ cache controller
  V6 ↔ replay controller V4 ↔ retrieval head ↔ attention-
  steering V7 ↔ three-way bridge classifier ↔ prefix-state-
  bridge V7** hybrid loop, and first **hidden-wins falsifier**
  that returns 0 exactly when inverting residual roles flips
  the decision.* Ships at `coordpy.tiny_substrate_v8` (10-layer
  / 8-query-head / 4-KV-head GQA / d_model=64 / RMSNorm /
  SwiGLU / per-(layer, head, slot) hidden-vs-KV contention
  tensor / per-layer hidden-state confidence probe / per-(layer,
  head, slot) replay-determinism channel / per-(layer, head)
  prefix-reuse trust ledger / per-(L, H, L, H) cross-layer-head
  coupling matrix), `coordpy.kv_bridge_v8` (V8 layer-e
  correction additive on top of V7 layer-d; four-target stacked
  ridge fit with explicit hidden-wins target; contention
  coupling into V8 substrate; hidden-wins falsifier),
  `coordpy.hidden_state_bridge_v7` (four-target stacked ridge
  fit; V8 contention coupling; V3 recovery audit with two-stage
  basin width), `coordpy.prefix_state_bridge_v7` (token-content-
  conditional drift-curve predictor; prefix-vs-hidden three-way
  comparator; V8 reuse-trust coupling),
  `coordpy.attention_steering_bridge_v7` (three-stage clamp;
  per-bucket cosine-aligned falsifier),
  `coordpy.cache_controller_v6` (three-objective ridge head;
  retrieval-repair head; composite_v6 7-head ridge mixture),
  `coordpy.replay_controller_v4` (six-regime per-regime ridge
  head; 7-dim regime gate; three-way bridge classifier; replay-
  determinism bonus on REUSE),
  `coordpy.deep_substrate_hybrid_v8` (eight-way bidirectional
  loop), `coordpy.persistent_latent_v15` (14 layers; twelfth-
  and-thirteenth skip carriers; max_chain_walk_depth=4096;
  distractor rank 10), `coordpy.multi_hop_translator_v13`
  (24 backends, 552 directed edges, chain-length-19, 8-axis
  composite trust adding `hidden_wins_trust`),
  `coordpy.mergeable_latent_capsule_v11` (hidden_wins +
  prefix_reuse witness chains; Jensen-Shannon disagreement
  distance; two new algebra signatures),
  `coordpy.consensus_fallback_controller_v9` (13-stage chain
  inserting `hidden_wins_arbiter`),
  `coordpy.corruption_robust_carrier_v11` (2048-bucket wrap-
  around-XOR fingerprint; 19-bit adversarial burst; hidden-state
  recovery L2-ratio probe), `coordpy.long_horizon_retention_v15`
  (14 heads; max_k=160; five-layer scorer; hidden-wins head),
  `coordpy.ecc_codebook_v15` (K1..K14 = 2^24 = 16 777 216 codes;
  26.333 bits/visible-token at full emit ≥ 26.0 target),
  `coordpy.uncertainty_layer_v11` (10-axis composite with
  `hidden_wins_fidelity`), `coordpy.disagreement_algebra_v9`
  (JS-equivalence identity + falsifier),
  `coordpy.transcript_vs_shared_arbiter_v12` (13-arm pick rates
  summing to 1.0 within 1e-6 with `hidden_wins` arm),
  `coordpy.substrate_adapter_v8` (`substrate_v8_full` tier),
  and `coordpy.w63_team` (22 module witness CIDs sealed into a
  `W63HandoffEnvelope` whose `w62_outer_cid` carries forward
  the W62 envelope byte-for-byte; verifier enumerates 72
  disjoint failure modes; trivial passthrough preserved byte-
  for-byte). R-134 + R-135 + R-136 at 3 seeds verify
  H181..H202b. All **49 H-bars pass 3/3 seeds (147/147 cells)**.
  ``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged; ``W63-L-V8-NO-AUTOGRAD-CAP`` (only
  seventeen closed-form linear ridge solves across W61+W62+W63;
  no SGD, no autograd, no GPU). ``coordpy.__version__`` remains
  ``0.5.20``; SDK contract is byte-for-byte unchanged. No PyPI
  release.

- **W62 Trainable Replay-Dominance Hidden-vs-KV Substrate-
  Coupled Latent Operating System** (post-W61, 2026-05-15) —
  *seventh substrate-attack milestone; first **per-(layer,
  head, slot) cache-write ledger axis** inside an in-repo NumPy
  substrate (V7), first **per-layer logit-lens probe**, first
  **per-(layer, head, position) attention-receive delta**, first
  **per-(layer, head) replay-trust ledger**, first **trained
  hidden-vs-KV regime classifier** (closed-form ridge over a
  5-dim regime feature against a 3-class label), first
  **trained per-regime replay head** (replay controller V3,
  4 regimes × 6×4 ridge per-regime softmax head + nearest-
  centroid regime gate), first **trained corruption-repair head**
  (cache controller V5, 4-dim ridge over [flag_count,
  hidden_write, replay_age, attention_receive_l1] against
  repair amount), first **two-objective stacked ridge fit**
  (cache controller V5, drop-oracle + retrieval-relevance
  simultaneously), first **drift-curve predictor** (prefix V6,
  stacked 4×K ridge over [reuse_len, recompute_len, drop_len, 1]
  against K-step drift curve target), first **two-stage
  attention clamp** (attention V6, coarse L1-mass clamp + fine
  per-(L,H,Q,K) KL clamp), and first **seven-way bidirectional
  substrate ↔ V7 ↔ cache controller V5 ↔ replay controller V3
  ↔ retrieval head ↔ attention-steering V6 ↔ hidden-vs-KV
  classifier** hybrid loop.* Ships at
  `coordpy.tiny_substrate_v7` (9-layer / 8-query-head / 4-KV-head
  GQA / d_model=64 / RMSNorm / SwiGLU / per-(layer, head, slot)
  cache-write ledger / per-layer logit-lens probe / per-(layer,
  head, position) attention-receive delta / per-(layer, head)
  replay-trust ledger),
  `coordpy.kv_bridge_v7` (V7 layer-d correction additive on
  top of V6 layer-c; three-target stacked ridge fit; ledger
  write into V7 cache; hidden-vs-KV decision tap),
  `coordpy.hidden_state_bridge_v6` (three-target stacked ridge
  fit; V7 ledger write; V2 recovery audit with post-recovery
  margin),
  `coordpy.prefix_state_bridge_v6` (drift-curve predictor:
  stacked 4×K closed-form ridge over [reuse_len, recompute_len,
  drop_len, 1] against K-step drift curve target),
  `coordpy.attention_steering_bridge_v6` (two-stage clamp;
  per-bucket signed-coefficient falsifier),
  `coordpy.cache_controller_v5` (three new policies:
  `two_objective_v5` / `trained_repair_v5` / `composite_v5`;
  two-objective stacked ridge over drop-oracle + retrieval-
  relevance; trained-repair head 4-dim ridge; composite_v5
  6-head mixture ridge),
  `coordpy.replay_controller_v3` (per-regime 6×4 ridge head
  trained per regime ∈ {synthetic_corruption,
  crc_passed_low_drift, hidden_write_heavy, transcript_only};
  nearest-centroid regime gate; hidden-vs-KV 5×3 ridge
  classifier with synthetic supervision target accuracy ≥ 0.8;
  replay-dominance scalar = softmax margin),
  `coordpy.deep_substrate_hybrid_v7` (seven-way V7 bidirectional
  loop), `coordpy.substrate_adapter_v7` (4 new axes —
  `cache_write_ledger` / `logit_lens_probe` /
  `attention_receive_delta` / `replay_trust_ledger` — and a new
  top tier `substrate_v7_full` satisfied only by the V7 runtime),
  `coordpy.persistent_latent_v14` (**12-layer** outer skin over
  V13, **decuple** persistent skip-link with replay-dominance-
  EMA, max_chain_walk_depth=2048, distractor rank=8),
  `coordpy.multi_hop_translator_v12` (**20 backends**, 380
  directed edges, chain-length-17, **seven-axis** trust
  composite adding `replay_dominance_trust`, 1 ≤ threshold ≤ 7
  detector),
  `coordpy.mergeable_latent_capsule_v10` (adds
  `replay_dominance_witness_chain`,
  `disagreement_wasserstein_distance`; two new algebra
  signatures {`replay_dominance_propagation`,
  `wasserstein_disagreement`}),
  `coordpy.consensus_fallback_controller_v8` (**12-stage**
  chain; inserts `trained_repair` between
  `attention_pattern_consensus` and `best_parent`),
  `coordpy.corruption_robust_carrier_v10` (**1024-bucket**
  wrap-around-XOR fingerprint; **17-bit** adversarial burst
  family; post-repair top-K Jaccard floor),
  `coordpy.long_horizon_retention_v14` (**13 heads**, max_k=128,
  replay-dominance-conditioned head, **four-layer** scorer:
  random+ReLU → random+tanh → random+tanh-2 → closed-form ridge),
  `coordpy.ecc_codebook_v14` (**13-level**; K1..K13 = 2^23 =
  8 388 608 codes; **25.333 bits/visible-token** at full emit;
  4096-bit/token falsifier reproduces honestly),
  `coordpy.transcript_vs_shared_arbiter_v11` (**12-arm**
  adding `replay_dominance`),
  `coordpy.uncertainty_layer_v10` (**9th** weighting axis
  `replay_dominance_fidelity`),
  `coordpy.disagreement_algebra_v8` (adds Wasserstein-1
  equivalence identity with falsifier),
  `coordpy.w62_team` (W62 envelope with `w61_outer_cid` chain
  forward; verifier enumerates **68 disjoint failure modes**;
  trivial passthrough preserved byte-for-byte),
  `coordpy.r131_benchmark` (13 cell families — substrate V7 /
  latent bridge V7 / cache controller V5 / replay controller V3
  / attention V6), `coordpy.r132_benchmark` (12 cell families —
  long-horizon retention / persistent V14 / ECC V14 / multi-hop
  V12), `coordpy.r133_benchmark` (20 cell families — corruption
  V10 / consensus V8 / uncertainty V10 / disagreement V8 / TVS
  V11 / capsule V10 / substrate V7 axes / hybrid V7 / adapter
  V7). R-131 + R-132 + R-133 at 3 seeds verify
  **H163..H180 — 45 of 45 H-bars pass 3/3 seeds (135/135 cells,
  strong success per the W62 success criterion)**. W62 envelope
  chain end-to-end: `W61 envelope CID == W62.w61_outer_cid`
  (verified live; the test_w62_team_envelope_chain test
  preserves the supplied W61 outer CID byte-for-byte). Honest
  scope: `W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries
  forward the W56..W61 cap unchanged on hosted backends;
  `W62-L-V7-NO-AUTOGRAD-CAP` is the new ridge-only cap (W62
  fits closed-form ridge parameters in **five new places** on
  top of W61's seven, totalling **twelve closed-form ridge
  solves**: cache controller V5 two-objective stacked head;
  cache controller V5 trained-repair head; cache controller V5
  composite_v5 mixture; replay controller V3 per-regime head ×
  4 regimes; replay controller V3 hidden-vs-KV 3-class
  classifier; HSB V6 three-target stack delegates to V5;
  prefix V6 drift-curve predictor — no SGD, no autograd, no
  GPU); `W62-L-V14-OUTER-NOT-TRAINED-CAP`,
  `W62-L-ECC-V14-RATE-FLOOR-CAP`,
  `W62-L-V14-LHR-SCORER-FIT-CAP`,
  `W62-L-NUMPY-CPU-V7-SUBSTRATE-CAP`,
  `W62-L-V6-PREFIX-DRIFT-CURVE-LINEAR-CAP`,
  `W62-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`,
  `W62-L-V3-REPLAY-NO-AUTOGRAD-CAP`,
  `W62-L-V6-HSB-NO-AUTOGRAD-CAP`,
  `W62-L-V6-ATTN-NO-AUTOGRAD-CAP`,
  `W62-L-MULTI-HOP-V12-SYNTHETIC-BACKENDS-CAP`,
  `W62-L-CRC-V10-FINGERPRINT-SYNTHETIC-CAP`,
  `W62-L-CONSENSUS-V8-REPAIR-STAGE-SYNTHETIC-CAP` document the
  new honest caps. SDK contract byte-for-byte unchanged.
  `coordpy.__version__` remains `"0.5.20"`. NO PyPI release.

- **W61 Trainable Hidden-State Substrate-Coupled Latent
  Operating System** (post-W60, 2026-05-15) — *sixth substrate-
  attack milestone; first **content-addressable cache-key axis**
  inside an in-repo NumPy substrate, first **bilinear retrieval
  head fit by closed-form ridge over the (query ⊗ cache_key)
  outer-product feature** (cache controller V4), first **trained
  ridge replay-threshold head** (replay controller V2, 4-way
  decision softmax with abstain-on-confidence threshold), first
  **per-(layer, head, query, key) 4-D attention-budget tensor with
  a signed-coefficient falsifier**, first **multi-target stacked
  HSB fit** (per-(layer, head, position) 3-D δ tensor against m
  stacked target logit directions), first **attention-pattern-
  target KV fit** (steers the substrate's last-row attention map
  toward a reference pattern by ridge), and first **six-way
  bidirectional substrate ↔ V6 ↔ cache controller V4 ↔ replay
  controller V2 ↔ retrieval head ↔ attention-steering V5** hybrid
  loop.* Ships at
  `coordpy.tiny_substrate_v6` (8-layer / 8-query-head / 4-KV-head
  GQA / d_model=64 / RMSNorm / SwiGLU / per-(layer, position, d_key=8)
  content-addressable cache key tensor / per-(layer, head)
  cumulative hidden-write trace / per-(layer, position) replay-age
  channel / forward counter / per-(layer_i, layer_j) cross-layer
  attention-coupling diagnostic),
  `coordpy.kv_bridge_v6` (matrix-valued multi-target ridge fit
  with worst-residual reduction; attention-pattern target fit;
  V6 cache_key 128-bucket fingerprint),
  `coordpy.hidden_state_bridge_v5` (per-(layer, head, position)
  3-D δ tensor, multi-target stacked fit, V4 recovery delegation,
  HSB→V6-cache hidden-write coupling),
  `coordpy.prefix_state_bridge_v5` (chain-of-chains over V6
  substrates; trained 3-feature ridge drift predictor; V6 cache-
  key fingerprint surfaced in the witness),
  `coordpy.attention_steering_bridge_v5` (per-(layer, head, query,
  key) **4-D budget tensor**; signed-coefficient falsifier;
  attention-map L1 + L2 + top-K Jaccard observables),
  `coordpy.cache_controller_v4` (four new policies:
  `bilinear_retrieval_v6` / `trained_corruption_floor` /
  `two_stage_v4` / `composite_v4`; bilinear M ridge over the
  (query_feature ⊗ cache_key) outer-product feature; quadratic
  corruption-floor fit; closed-form two-stage L1 threshold),
  `coordpy.replay_controller_v2` (closed-form linear ridge over
  a 6-dim feature against a 4-class label one-hot; softmax
  decision-confidence; abstain-on-confidence threshold; hidden-
  write-cap gate),
  `coordpy.deep_substrate_hybrid_v6` (six-way V6 bidirectional
  loop), `coordpy.substrate_adapter_v6` (7 new axes —
  `cache_key_axis` / `hidden_write_trace` /
  `replay_age_channel` / `cross_layer_coupling` /
  `bilinear_retrieval_head` / `trained_replay_thresholds` /
  `attention_pattern_target` — and a new top tier
  `substrate_v6_full` satisfied only by the V6 runtime),
  `coordpy.persistent_latent_v13` (**11-layer** outer skin over
  V12, **nonuple** persistent skip-link with replay-confidence-EMA,
  max_chain_walk_depth=1536, distractor rank=6),
  `coordpy.multi_hop_translator_v11` (**18 backends**, 306
  directed edges, chain-length-16, **six-axis** trust composite
  adding `attention_pattern_fidelity`, 1 ≤ threshold ≤ 6 detector),
  `coordpy.mergeable_latent_capsule_v9` (adds
  `attention_pattern_witness_chain`, `cache_retrieval_witness_chain`,
  per-(layer, head) trust matrix; two new algebra signatures
  {`attention_pattern_steer`, `cache_retrieval_query`}),
  `coordpy.consensus_fallback_controller_v7` (**11-stage** chain;
  inserts `attention_pattern_consensus` between `replay_controller`
  and `best_parent`),
  `coordpy.corruption_robust_carrier_v9` (**512-bucket**
  wrap-around-XOR fingerprint; **13-bit** adversarial burst family;
  post-replay top-K Jaccard floor),
  `coordpy.long_horizon_retention_v13` (**12 heads**, max_k=128,
  attention-pattern-conditioned head, **three-layer** scorer:
  random+ReLU → random+tanh → closed-form ridge),
  `coordpy.ecc_codebook_v13` (**12-level**; K1..K12 = 2^22 =
  4 194 304 codes; **24.333 bits/visible-token** at full emit;
  2048-bit/token falsifier reproduces honestly),
  `coordpy.transcript_vs_shared_arbiter_v10` (**11-arm**
  adding `attention_pattern_steer`),
  `coordpy.uncertainty_layer_v9` (**8th** weighting axis
  `attention_pattern_fidelity`),
  `coordpy.disagreement_algebra_v7` (adds attention-pattern-
  equivalence identity with falsifier),
  `coordpy.w61_team` (W61 envelope with `w60_outer_cid` chain
  forward; verifier enumerates **61 disjoint failure modes**;
  trivial passthrough preserved byte-for-byte),
  `coordpy.r128_benchmark` (25 cell families — substrate V6 /
  latent bridge V6 / cache controller V4 / replay controller V2
  / attention pattern), `coordpy.r129_benchmark` (15 cell
  families — long-horizon retention / persistent / ECC / multi-hop
  / mergeable capsule), `coordpy.r130_benchmark` (12 cell families
  — corruption / consensus / uncertainty / disagreement / TVS
  arbiter). R-128 + R-129 + R-130 at 3 seeds verify
  **H144..H162c — 52 of 52 H-bars pass 3/3 seeds (156/156 cells,
  strong success per the W61 success criterion)**. W61 envelope
  chain end-to-end: `W60 envelope CID == W61.w60_outer_cid`
  (verified live; the test_w61_team_envelope_chain test
  preserves the supplied W60 outer CID byte-for-byte). Honest
  scope: `W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries
  forward the W56..W60 cap unchanged on hosted backends;
  `W61-L-V6-NO-AUTOGRAD-CAP` is the new ridge-only cap (W61 fits
  closed-form ridge parameters in **seven** places: KV bridge V6
  multi-target + attention-pattern; HSB V5 multi-target stack;
  cache controller V4 bilinear retrieval head; cache controller
  V4 trained corruption floor; replay controller V2 threshold
  head; LHR V13 third-layer scorer — no SGD, no autograd, no
  GPU); `W61-L-V13-OUTER-NOT-TRAINED-CAP`,
  `W61-L-ECC-V13-RATE-FLOOR-CAP`, `W61-L-V13-LHR-SCORER-FIT-CAP`,
  `W61-L-NUMPY-CPU-V6-SUBSTRATE-CAP`,
  `W61-L-V5-DRIFT-PREDICTOR-LINEAR-CAP`,
  `W61-L-KV-V6-MULTI-TARGET-REDUCTION-CAP`,
  `W61-L-V5-ATTN-NO-AUTOGRAD-CAP`,
  `W61-L-V5-HSB-NO-AUTOGRAD-CAP`,
  `W61-L-V2-REPLAY-NO-AUTOGRAD-CAP`,
  `W61-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-CAP`,
  `W61-L-ATTENTION-PATTERN-TARGET-SYNTHETIC-CAP`, and
  `W61-L-MULTI-HOP-V11-SYNTHETIC-BACKENDS-CAP` document the new
  honest caps. SDK contract byte-for-byte unchanged.
  `coordpy.__version__` remains `"0.5.20"`. NO PyPI release.

- **W60 Trainable Cache-Control Substrate-Coupled Latent
  Operating System** (post-W59, 2026-05-14) — *fifth substrate-
  attack milestone; first **multi-direction multi-target closed-
  form ridge fits** (KV bridge V5 over `n_directions` orthogonal
  correction directions; HSB V4 per-(layer, head) δ tensor;
  cache controller V3 `trained_eviction` + `composite_v3`
  mixture); first **first-class state-reuse-vs-recompute-vs-
  fallback-vs-abstain ReplayController**; first **five-way
  substrate ↔ V6 ↔ cache controller V3 ↔ replay controller ↔
  retrieval head** hybrid loop; first **hidden-vs-KV head-to-
  head** harness on a fixed target logit direction (falsifiable:
  at least one arm wins or it's a tie).* Ships at
  `coordpy.tiny_substrate_v5` (7-layer / 8-query-head / 4-KV-head
  GQA / d_model=64 / RMSNorm / SwiGLU / per-(layer, head,
  position) cumulative attention-receive matrix / per-(layer,
  head) linearised logit Jacobian table / per-(layer, position)
  corruption flag channel / multi-segment partial-prefix reuse
  with reuse + recompute + drop kinds), `coordpy.kv_bridge_v5`
  (multi-direction closed-form ridge fit, logit-direction fit,
  two correction layers, all-bank fingerprint, reverse-extract
  via least-squares with residual L2 ≤ 2.4e-7 typical),
  `coordpy.hidden_state_bridge_v4` (per-(layer, head) closed-
  form ridge fit of inject-scale tensor, recovery from
  adversarial per-head perturbation, KV-vs-Hidden head-to-head
  harness), `coordpy.prefix_state_bridge_v4` (multi-segment
  partial reuse on V5; chain forward with per-step drift L2 +
  cumulative envelope; ~46% flop saving on standard split),
  `coordpy.attention_steering_bridge_v4` (per-(layer, head,
  query) 3-D budget tensor; measurable attention-map L1 mass
  shift; negative-budget falsifier),
  `coordpy.cache_controller_v3` (four new policies:
  `learned_attention_receive`, `learned_corruption_aware`,
  `trained_eviction`, `composite_v3`; trained_eviction reduces
  residual against the V1 leave-one-out drop oracle from 28.4
  → 0.39, ≈73x improvement), `coordpy.replay_controller` (first
  first-class state-reuse-vs-recompute-vs-fallback-vs-abstain
  policy with audit log + flop-vs-drift trade-off),
  `coordpy.persistent_latent_v12` (**10-layer**, **octuple**
  skip-link with replay-EMA, max_chain_walk_depth=1024,
  distractor-resistant projection),
  `coordpy.deep_substrate_hybrid_v5` (**five-way** V6 ↔
  substrate V5 ↔ cache controller V3 ↔ replay controller ↔
  retrieval head; `five_way=True` in real runs),
  `coordpy.multi_hop_translator_v10` (16 backends, 240 directed
  edges, chain-length-15, substrate × hidden × attention ×
  retrieval × replay **five-axis** trust composite + compromise-
  threshold detector), `coordpy.mergeable_latent_capsule_v8`
  (`replay_witness_chain` + `substrate_witness_chain` +
  `provenance_trust_table` all union-inherited; two new algebra
  signatures {`replay_choice`, `substrate_state_inject`}),
  `coordpy.consensus_fallback_controller_v6` (**10-stage** chain
  adding `replay_controller_choice` between `retrieval_replay`
  and `best_parent`), `coordpy.corruption_robust_carrier_v8`
  (**256-bucket** wrap-around-XOR fingerprint with guaranteed
  single-byte detection at any blob length, `recover_v8_kv_cache`
  operator, adversarial 11-bit burst, post-replay top-K
  agreement floor ≥ pre-replay),
  `coordpy.long_horizon_retention_v12` (11-head V12 with
  replay-conditioned head, max_k=96, two-layer scorer:
  random projection + frozen ReLU + closed-form ridge over the
  post-ReLU features), `coordpy.ecc_codebook_v12` (**11-level**;
  K1..K11 = 2 097 152 codes = 2^21; 23.333 bits/visible-token
  at full emit; 2048 bits/token falsifier reproduces honestly),
  `coordpy.transcript_vs_shared_arbiter_v9` (10-arm: adds
  `replay_controller_choice`),
  `coordpy.uncertainty_layer_v8` (**7th** weighting axis
  `replay_fidelity`), `coordpy.disagreement_algebra_v6` (adds
  replay-controller-equivalence identity with falsifier),
  `coordpy.substrate_adapter_v5` (7 new capability axes,
  `substrate_v5_full` tier on V5 only), `coordpy.w60_team`
  (W60 envelope with `w59_outer_cid` chain forward; verifier
  enumerates 52 disjoint failure modes; trivial passthrough
  preserved). R-125 + R-126 + R-127 at 3 seeds verify
  H125..H143b — 45 of 45 H-bars pass 3/3 seeds (135/135 cells).
  W60 envelope chain end-to-end: `W59 envelope CID ==
  W60.w59_outer_cid` (verified live). Honest scope:
  `W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` documents that
  hosted backends remain text-only at the HTTP surface
  unchanged; `W60-L-V5-NO-AUTOGRAD-CAP` is the new ridge-only
  cap (no SGD, no autograd, no GPU);
  `W60-L-V12-OUTER-NOT-TRAINED-CAP`,
  `W60-L-ECC-V12-RATE-FLOOR-CAP`,
  `W60-L-LHR-V12-SCORER-FIT-CAP`, and
  `W60-L-CORRUPTION-FLAG-CHANNEL-CAP` document the new honest
  caps. SDK contract byte-for-byte unchanged. NOT a release;
  `coordpy.__version__` remains `"0.5.20"`. NO PyPI release.

- **W59 Trainable Substrate-Conditioned Latent Operating
  System** (post-W58, 2026-05-14) — *fourth substrate-attack
  milestone; first **closed-form ridge fit of a real substrate-
  facing matrix** (the cache-controller V2 bilinear retrieval
  matrix, fit by closed-form ridge over a d²-dim outer-product
  feature against the substrate's leave-one-out drop oracle) and
  first **four-way substrate ↔ V6 ↔ cache-controller ↔
  retrieval-head** hybrid loop.* Ships at
  `coordpy.tiny_substrate_v4` (6-layer / 8-query-head / 4-KV-head
  GQA / d_model=64 / RMSNorm / SwiGLU / cumulative-EMA KV
  importance / partial-prefix split-and-replay / per-(layer,
  head) hidden-state tap / 128-bucket fingerprint / logit-
  Jacobian probe), `coordpy.kv_bridge_v4` (four role banks
  bank_a/bank_b/bank_c/bank_d, closed-form ridge fit of a 1-D
  correction α along a fixed random direction over an N-point
  carrier/target set, 128-bucket readback fingerprint),
  `coordpy.hidden_state_bridge_v3` (target-logit-shift fit by
  closed-form ridge on a 1-D inject-scale α, per-(layer, head)
  scale tensor), `coordpy.prefix_state_bridge_v3` (partial-prefix
  reuse byte-identical to full recompute on the matched span ≤
  4.4e-16; K-seed drift spectrum mean/max/min/var; empirical
  Lipschitz certificate ratio; 128-bucket fingerprint),
  `coordpy.attention_steering_bridge_v3` (per-(layer, head)
  KL-budget clip fit by iterative shrink; per-head dominance
  ablation), `coordpy.cache_controller_v2` (keeps W58's three
  policies, adds **learned_hidden** closed-form ridge on a cross-
  layer hidden-state feature and **learned_retrieval** bilinear
  M-matrix closed-form ridge over a d² outer-product feature,
  pre/post fit residual >4 OOM reduction routinely on R-122),
  `coordpy.persistent_latent_v11` (**9-layer**, **septuple**
  skip-link with retrieval-EMA, max_chain_walk_depth=768),
  `coordpy.deep_substrate_hybrid_v4` (**four-way** V6 ↔
  substrate V4 ↔ cache-controller V2 ↔ retrieval head;
  `four_way=True` and `retrieval_used=True` in real runs),
  `coordpy.multi_hop_translator_v9` (14 backends, 182 directed
  edges, chain-length-13, substrate × hidden × attention ×
  retrieval **four-axis** trust composite),
  `coordpy.mergeable_latent_capsule_v7`
  (retrieval_witness_chain inheritance,
  controller_witness_cid propagation, two new algebra signatures
  {retrieval_replay, partial_prefix_reuse}),
  `coordpy.consensus_fallback_controller_v5` (**9-stage** chain
  adding `retrieval_replay` between `cache_reuse_replay` and
  `best_parent`), `coordpy.corruption_robust_carrier_v7`
  (**128-bucket** Reed-Solomon fingerprint + cache-retrieval
  top-K agreement under non-target corruption + **9-bit
  adversarial burst** detection), `coordpy.long_horizon_retention_v11`
  (**10 heads**, max_k=80, retrieval-conditioned head,
  closed-form ridge retention scorer), `coordpy.ecc_codebook_v11`
  (K1..K10 = 1 048 576 codes = 2^20, **22.333 bits/visible-token**
  at full emit), `coordpy.transcript_vs_shared_arbiter_v8`
  (**9-arm** policy adding `retrieval_replay`),
  `coordpy.uncertainty_layer_v7` (6th axis retrieval_fidelity),
  `coordpy.disagreement_algebra_v5` (retrieval-equivalence
  identity), `coordpy.substrate_adapter_v4` (five new capability
  axes — partial_prefix_reuse / cache_retrieval /
  closed_form_ridge / per_head_kl_fit / hidden_target_fit — and
  new top tier `substrate_v4_full`), `coordpy.w59_team` (W59Team
  orchestrator, W59HandoffEnvelope with 49 disjoint verifier
  failure modes), `coordpy.r122_benchmark`, `coordpy.r123_benchmark`,
  `coordpy.r124_benchmark`. R-122 (15 cell families) + R-123 (12
  cell families) + R-124 (11 cell families) at 3 seeds verify
  **38/38 H-bars (H107..H124) pass 3/3 seeds** — strong success
  per the W59 success criterion. Cumulative trust boundary
  across W22..W59 = **663 enumerated failure modes**. W59
  modules are reachable only through explicit imports;
  `coordpy.__version__` remains `0.5.20`; SDK contract byte-for-
  byte unchanged. No PyPI release.
  `W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries forward
  unchanged; `W59-L-V4-NO-AUTOGRAD-CAP` documents that all W59
  "training" is single-step closed-form linear ridge over a
  small subspace (no SGD / autograd / GPU); `W59-L-V11-OUTER-
  NOT-TRAINED-CAP`, `W59-L-ECC-V11-RATE-FLOOR-CAP`, `W59-L-LHR-
  V11-SCORER-FIT-CAP`, `W59-L-MULTI-HOP-V9-SYNTHETIC-BACKENDS-
  CAP`, and `W59-L-V11-PERMUTATION-INVARIANCE-CAP` are the
  additional honest caps; `W59-C-DEEP-TRANSFORMER-COUPLING` and
  `W59-C-FRONTIER-SCALE-SUBSTRATE-LIFT` carry forward as
  conjectures on frontier-scale model substrate access.

- **W58 Deep Cache-Reuse Substrate-Coupled Latent Operating
  System** (post-W57, 2026-05-13) — *third substrate-attack
  milestone; first **three-way** substrate breach with **real
  fp64 flop savings** as a benchmark-load-bearing axis.* Ships
  at `coordpy.tiny_substrate_v3` (5-layer / 8-query-head / 4-KV-
  head GQA / d_model=64 / RMSNorm / SwiGLU / per-token KV
  importance tracking / real fp64 flop counter / partial-forward
  / 64-bucket KV fingerprint), `coordpy.kv_bridge_v3` (fitted
  per-(layer, head) inject scales via coordinate descent, role-
  conditioned KV banks bank_a/bank_b, 64-bucket readback
  fingerprint), `coordpy.hidden_state_bridge_v2` (multi-layer
  fitted injection), `coordpy.prefix_state_bridge_v2` (real
  flop-saved counter — **66.7% savings** on H100b; redundant
  copy CID; cross-seed drift L2),
  `coordpy.attention_steering_bridge_v2` (KL-budget enforcement
  by coordinate descent on global bias clip; per-head ablation),
  `coordpy.cache_controller` (uniform / importance / **learned**
  closed-form ridge over a leave-one-out drop oracle, with real
  flop savings ratio in the witness),
  `coordpy.persistent_latent_v10` (8-layer, **sextuple** skip
  with attention-pattern EMA, max_chain_walk_depth=512),
  `coordpy.deep_substrate_hybrid_v3` (**three-way** V6 ↔
  substrate V3 ↔ cache controller; `three_way=True` in real
  runs), `coordpy.multi_hop_translator_v8` (12 backends, 132
  directed edges, chain-length-11, substrate × hidden ×
  attention three-axis trust composite),
  `coordpy.mergeable_latent_capsule_v6` (attention_witness_chain
  inheritance, cache_reuse_witness_cid propagation, two new
  algebra signatures {cache_reuse_replay, attention_steer}),
  `coordpy.consensus_fallback_controller_v4` (**8-stage** chain
  adding `cache_reuse_replay` between `logit_lens` and
  `best_parent`), `coordpy.corruption_robust_carrier_v6`
  (64-bucket Reed-Solomon fingerprint + prefix-state corruption
  detection + adversarial 7-bit burst detection),
  `coordpy.long_horizon_retention_v10` (9 heads, max_k=72,
  attention-conditioned head), `coordpy.ecc_codebook_v10`
  (K1..K9 = 524 288 codes, **21.333 bits/visible-token** at full
  emit), `coordpy.transcript_vs_shared_arbiter_v7` (**8-arm**
  policy adding `cache_reuse_replay`),
  `coordpy.uncertainty_layer_v6` (5th axis cache_reuse_fidelity),
  `coordpy.disagreement_algebra_v4` (cache-reuse equivalence
  identity), `coordpy.substrate_adapter_v3` (five new capability
  axes — kv_importance_track / flop_counter / partial_forward /
  fitted_inject_scale / cache_controller — and new top tier
  `substrate_v3_full`), `coordpy.w58_team` (W58Team orchestrator,
  W58HandoffEnvelope with 46 disjoint verifier failure modes),
  `coordpy.r119_benchmark`, `coordpy.r120_benchmark`,
  `coordpy.r121_benchmark`. R-119 (16 cell families) + R-120 (12
  cell families) + R-121 (12 cell families) at 3 seeds verify
  **40/40 H-bars (H86..H106) pass 3/3 seeds** — strong success
  per the W58 success criterion. Cumulative trust boundary
  across W22..W58 = **614 enumerated failure modes**. W58
  modules are reachable only through explicit imports;
  `coordpy.__version__` remains `0.5.20`; SDK contract byte-for-
  byte unchanged. No PyPI release.
  `W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries forward
  unchanged; `W58-L-V3-NO-BACKPROP-CAP` documents that W58 fits
  only KV-bridge / HSB inject scales (coordinate descent) and a
  single linear cache-controller scoring head (closed-form
  ridge), with no end-to-end backprop; `W58-L-V10-OUTER-NOT-
  TRAINED-CAP`, `W58-L-ECC-V10-RATE-FLOOR-CAP`, `W58-L-CACHE-
  CONTROLLER-LINEAR-CAP`, `W58-L-MULTI-HOP-V8-SYNTHETIC-
  BACKENDS-CAP`, and `W58-L-V10-PERMUTATION-INVARIANCE-CAP` are
  the additional honest caps; `W58-C-DEEP-TRANSFORMER-COUPLING`
  and `W58-C-FRONTIER-SCALE-SUBSTRATE-LIFT` carry forward as
  conjectures on frontier-scale model substrate access.

- **W57 Deep Substrate-Coupled Latent Operating System**
  (post-W56, 2026-05-13) — *second substrate-attack milestone;
  first **bidirectional** substrate breach.* Ships at
  `coordpy.tiny_substrate_v2` (richer 4-layer / 8-head /
  d_model=64 / RoPE / per-layer logit lens / KV cache eviction
  / prefix-state extraction / per-head pre-softmax attention
  bias hook), `coordpy.kv_bridge_v2` (per-(layer, head)
  projection + readback CID), `coordpy.hidden_state_bridge`
  (residual injection at any layer), `coordpy.prefix_state_bridge`
  (save / load / detect-corruption; reuse-vs-recompute
  byte-identical to ≤ 5e-16), `coordpy.attention_steering_bridge`
  (per-(layer, head, query, key) bias tensor; mean-KL ≈ 5.59
  nats/layer), `coordpy.persistent_latent_v9` (7 layers,
  quintuple persistent skip-link, max_chain_walk_depth = 384,
  substrate-fidelity damping), `coordpy.multi_hop_translator_v7`
  (10 backends, chain-length-9, substrate-hidden-trust composite
  arbitration), `coordpy.mergeable_latent_capsule_v5`
  (hidden-state-witness-chain, per-head trust, two new algebra
  signatures), `coordpy.consensus_fallback_controller_v3`
  (7-stage chain — adds logit-lens-conditioned tiebreaker),
  `coordpy.corruption_robust_carrier_v5` (3-D interleave 4×4×4,
  9-of-13 majority, 32-bucket KV cache fingerprint),
  `coordpy.long_horizon_retention_v9` (8 heads, max_k=64),
  `coordpy.ecc_codebook_v9` (262144 codes, **20.333
  bits/visible-token**), `coordpy.transcript_vs_shared_arbiter_v6`
  (7-arm policy — adds substrate_hidden_inject),
  `coordpy.uncertainty_layer_v5` (hidden-state-fidelity 4th
  axis, adversarial brackets), `coordpy.disagreement_algebra_v3`
  (hidden-projection identity), `coordpy.deep_substrate_hybrid_v2`
  (**bidirectional substrate ↔ V6 hybrid stack**),
  `coordpy.substrate_adapter_v2` (4 new capability axes,
  substrate_v2_full tier), `coordpy.w57_team`,
  `coordpy.r116_benchmark`, `coordpy.r117_benchmark`,
  `coordpy.r118_benchmark`. R-116 (14 cell families) + R-117
  (14 cell families) + R-118 (15 cell families) at 3 seeds verify
  **43/43 H-bars (H43..H85) pass 3/3 seeds** — strong success
  per the W57 success criterion. Cumulative trust boundary
  across W22..W57 = **568 enumerated failure modes**.
  W57 modules are reachable only through explicit imports;
  `coordpy.__version__` remains `0.5.20`; SDK contract
  byte-for-byte unchanged. No PyPI release.
  `W57-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` records that
  hosted backends remain text-only at the HTTP surface;
  `W57-C-DEEP-TRANSFORMER-COUPLING` is a sharper restatement of
  the open question on frontier-scale models;
  `W57-C-FRONTIER-SCALE-SUBSTRATE-LIFT` is a new conjecture
  that the W57 bridges would scale-monotonically improve
  usefulness if frontier runtimes exposed compatible hooks.

- **W56 Substrate-Coupled Latent Operating System** (post-W55,
  2026-05-13) — *first substrate-attack milestone in the
  programme.* Ships at `coordpy.tiny_substrate`,
  `coordpy.substrate_adapter`, `coordpy.kv_bridge`,
  `coordpy.persistent_latent_v8`,
  `coordpy.multi_hop_translator_v6`,
  `coordpy.mergeable_latent_capsule_v4`,
  `coordpy.consensus_fallback_controller_v2`,
  `coordpy.corruption_robust_carrier_v4`,
  `coordpy.deep_substrate_hybrid`,
  `coordpy.long_horizon_retention_v8`,
  `coordpy.ecc_codebook_v8`,
  `coordpy.transcript_vs_shared_arbiter_v5`,
  `coordpy.uncertainty_layer_v4`,
  `coordpy.disagreement_algebra_v2`, `coordpy.w56_team`,
  `coordpy.r113_benchmark`, `coordpy.r114_benchmark`, and
  `coordpy.r115_benchmark`. Twelve orthogonal advances over
  W55: M1 a tiny in-repo executable NumPy transformer substrate
  with REAL multi-head causal self-attention, REAL per-layer KV
  cache, REAL hidden states, REAL logits, REAL layer norm, REAL
  GeLU FF, REAL byte-vocab embedding + unembedding; M2
  substrate adapter honestly classifying each backend across 8
  capability axes into one of {substrate_full, embeddings_only,
  logits_only, text_only, unreachable}; M3 KV bridge projecting
  capsule-layer latent carriers into per-layer (K,V) slot pairs
  and injecting them into the substrate's KV cache —
  replay-deterministic, content-addressed, measurable logit
  perturbation (max abs ≈ 0.86 mean across 3 seeds); M4 6-layer
  persistent latent state V8 with *quad* persistent skip-link
  (turn-0 anchor + fast EMA + slow EMA + substrate-conditioned
  EMA) and `max_chain_walk_depth=256`; M5 8-backend
  (A,B,C,D,E,F,G,H) oct multi-hop translator V6 over 56 directed
  edges with chain-length-7 transitivity and
  substrate-trust-weighted arbitration; M6 Mergeable Latent
  State Capsule V4 (MLSC V4) extending V3 with
  `substrate_witness` field, `algebra_signature`, and per-fact
  provenance chain walking back to root capsule; M7 Consensus
  Fallback Controller V2 with **6-stage decision chain**
  {K-of-N → trust-weighted → substrate-conditioned →
  best-parent → transcript → abstain}; M8 Corruption-Robust
  Carrier V4 with BCH(31,16) **triple-bit correction** (real
  minimum-distance bounded decoder over a 65536-codeword
  codebook), **7-of-9 majority repetition**, and **2-D row-column
  interleaving**; M9 Deep Substrate Hybrid Stack replacing the
  top of W55 V6 with the **real tiny-substrate attention block**
  — reads / writes the real KV cache, adaptive abstain
  threshold; M10 7-head Long-Horizon Reconstruction V8 (causal +
  branch + cycle + merged-branch + cross-role + cross-cycle +
  substrate-conditioned) at `max_k=48`; M11 7-level ECC Codebook
  V8 (K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2 × K7=2 = 131072
  codes) — achieves **19.333 bits/visible-token at full emit**
  (≥ 19.0 target); M12 6-arm Transcript-vs-Shared-vs-Substrate
  Arbiter V5 over {transcript, shared, merge_consensus,
  trust_weighted_merge, **substrate_replay**, abstain} with
  per-arm budget allocator — the **first capsule-vs-substrate
  head-to-head** in the programme. Plus Uncertainty Layer V4
  (substrate-fidelity-weighted composite) and Disagreement
  Algebra V2 (V1 identities + substrate-projection identity).
  R-113 (12 cell families) + R-114 (11 cell families) + R-115
  (19 cell families) at 3 seeds verify H1..H42; **38/42 H-bars
  pass 3/3 seeds (strong success)**; **4 H-bars reproduce as
  honest caps** (H8 V8 outer untrained, H26 BCH 4-bit pathology
  on small probes, H31 V8 permutation invariance, H32 5-bit
  burst silent failure). 38 new disjoint envelope failure modes
  at W56; cumulative trust boundary across W22..W56 =
  **524 enumerated failure modes**.
  ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` documents that
  hosted backends (Ollama, OpenAI-compatible) remain text-only
  on their HTTP surface; ``W56-C-DEEP-TRANSFORMER-COUPLING``
  carries forward the frontier-model substrate-blocked
  conjecture unchanged.
  W56 is the **first executable substrate-coupling milestone**
  in the Context Zero programme; it does NOT claim third-party
  transformer-internal access.

- **W55 Deep Trust-Weighted Disagreement-Algebraic Latent
  Operating System** (post-W54, 2026-05-12) —
  `coordpy.persistent_latent_v7`,
  `coordpy.multi_hop_translator_v5`,
  `coordpy.mergeable_latent_capsule_v3`,
  `coordpy.trust_weighted_consensus_controller`,
  `coordpy.corruption_robust_carrier_v3`,
  `coordpy.deep_proxy_stack_v6`,
  `coordpy.long_horizon_retention_v7`,
  `coordpy.ecc_codebook_v7`,
  `coordpy.transcript_vs_shared_arbiter_v4`,
  `coordpy.uncertainty_layer_v3`,
  `coordpy.disagreement_algebra`,
  `coordpy.w55_team`, and `coordpy.r110_benchmark` /
  `coordpy.r111_benchmark` / `coordpy.r112_benchmark`.
  Eleven orthogonal capsule-native advances over W54: M1 5-layer
  V7 stacked GRU with *triple* persistent skip-link (turn-0
  anchor + fast EMA + slow EMA carriers) + disagreement-
  algebraic merge head (merged + low-bound + high-bound +
  disagreement) + max chain walk depth 128;
  M2 7-backend (A,B,C,D,E,F,G) hept translator V5 with chain-
  length-6 transitivity + trust-weighted compromise arbitration
  (per-backend trust × pairwise agreement);
  M3 Mergeable Latent State Capsule V3 (MLSC V3) with
  disagreement algebra primitives ⊕/⊖/⊗ + per-fact confirmation
  count + trust signature decay (`trust ← decay × trust` each
  turn unless reinforced);
  M4 Trust-Weighted Consensus Controller with continuous trust-
  weighted quorum (`Σ trust_i ≥ trust_threshold`) and 5-stage
  decision chain {K-of-N → trust-weighted → best-parent →
  transcript → abstain};
  M5 Corruption-Robust Carrier V3 with BCH(15,7) double-bit
  correction + 5-of-7 majority repetition + bit-interleaving
  (3-bit burst recovery rate 1.0);
  M6 depth-14 Deep Proxy Stack V6 with trust-projected residual
  gating + disagreement-algebra head + adaptive abstain
  threshold (scales monotonically with input L2 norm);
  M7 6-head Long-Horizon Reconstruction V7 (causal + branch +
  cycle + merged-branch + cross-role + cross-cycle) at
  `max_k=36` + degradation curve probe to `k=72`;
  M8 six-level ECC Codebook V7 (K1=32 × K2=16 × K3=8 × K4=4 ×
  K5=2 × K6=2 = 65536 codes) + BCH(15,7) per-segment double-bit
  correction → **18.333 bits/visible-token** at full emit;
  M9 5-arm Transcript-vs-Shared Arbiter V4 over {transcript,
  shared, merge_consensus, trust_weighted_merge, abstain-with-
  fallback} with per-arm budget allocator (retention × trust
  softmax);
  M10 Uncertainty Layer V3 with per-fact-tag uncertainty
  propagation (geometric mean over contributors) + adversarial
  calibration check (worst-case bounded perturbation) + trust-
  weighted composite (each component's confidence scaled by its
  trust scalar);
  M11 Disagreement Algebra first-class capsule-native module
  exposing ⊕/⊖/⊗ as content-addressed primitives with algebraic
  identities by inspection (idempotent ⊕ on a==b; ⊖ self-
  cancellation; ⊗ distributivity on agreement subspace).
  R-110 (12 cell families) + R-111 (10 cell families) + R-112
  (16 cell families) at 3 seeds verify H1..H38; cumulative trust
  boundary across W22..W55 = **486 enumerated failure modes**.
  W55-L-OVERDEPTH-V6-CAP, W55-L-ECC-V7-RATE-FLOOR-CAP, W55-L-
  BCH-FIVE-BIT-PATHOLOGY, W55-L-HEPT-TRANSLATOR-COMPROMISE-CAP,
  W55-L-V7-DISTRIBUTION-CAP, W55-L-V7-OUTER-NOT-TRAINED-CAP,
  W55-L-TRUST-WEIGHTED-NOT-STRICT-DOMINANCE, W55-L-ALGEBRA-
  IDENTITIES-ARE-EXACT-ONLY-ON-AGREEMENT, W55-L-TRUST-DECAY-NOT-
  RECOVERABLE-WITHOUT-REINFORCEMENT all reproduce honestly.
  The W54-C-DEEP-TRANSFORMER-COUPLING conjecture is further-
  bounded (now W55-C-DEEP-TRANSFORMER-COUPLING) but not closed.
  Reachable via explicit `from coordpy.w55_team import
  W55Team`. No version bump; no PyPI release.

- **W54 Deep Mergeable Disagreement-aware Latent Operating System**
  (post-W53, 2026-05-12) — `coordpy.persistent_latent_v6`,
  `coordpy.multi_hop_translator_v4`,
  `coordpy.mergeable_latent_capsule_v2`,
  `coordpy.consensus_quorum_controller`,
  `coordpy.corruption_robust_carrier_v2`,
  `coordpy.deep_proxy_stack_v5`,
  `coordpy.long_horizon_retention_v6`,
  `coordpy.ecc_codebook_v6`,
  `coordpy.transcript_vs_shared_arbiter_v3`,
  `coordpy.uncertainty_layer_v2`,
  `coordpy.w54_team`, and `coordpy.r107_benchmark` /
  `coordpy.r108_benchmark` / `coordpy.r109_benchmark`.
  Ten orthogonal capsule-native advances over W53: M1 4-layer
  V6 stacked GRU with *dual* persistent skip-link
  (turn-0 anchor + running EMA carrier) + disagreement-tagged
  merge head + max chain walk depth 64;
  M2 6-backend (A,B,C,D,E,F) hex translator V4 with chain-
  length-5 transitivity + disagreement-aware compromise
  arbitration;
  M3 MLSC V2 with per-dim disagreement metadata + provenance
  fact graph + per-parent trust signatures;
  M4 first-class K-of-N consensus quorum controller with
  explicit abstain-with-fallback policy + content-addressed
  audit trail;
  M5 corruption-robust carrier V2 with Hamming(7,4) single-bit
  *correction* per segment + 3-of-5 majority repetition (vs
  W53 V1's detect-only parity + 3-of-3 majority);
  M6 depth-12 deep proxy stack V5 with disagreement-aware
  head + uncertainty-projected residual gating + abstain
  short-circuit;
  M7 5-head long-horizon reconstruction V6 (causal + branch +
  cycle + merged-branch + cross-role) at max_k=24 (vs V5's 16)
  with per-dim degradation score; curve probe to k=48;
  M8 5-level ECC V6 codebook (K1×K2×K3×K4×K5 = 32768) +
  Hamming(7,4) on all 5 segments; ≥ 16 bits/visible-token
  target (achieved 18.0);
  M9 4-arm transcript-vs-shared arbiter V3 over {transcript,
  shared, merge_consensus, abstain-with-transcript-fallback};
  M10 uncertainty layer V2 with per-component noise injection +
  calibration-under-noise check + per-decision rationale +
  disagreement-weighted composite. Composes into the
  `W54Team` orchestrator that produces a content-addressed
  envelope chain `w47 → w48 → w49 → w50 → w51 → w52 → w53 →
  w54`. Trivial passthrough preserved byte-for-byte. 30 new
  enumerated failure modes (cumulative W22..W54 = 453).
  R-107 (12 families) × R-108 (10 families) × R-109 (14
  families) at 3 seeds each verify H1..H36. **No version
  bump; no PyPI release.**

- **W53 Persistent Mergeable Corruption-Robust Latent Operating System**
  (post-W52, 2026-05-12) — `coordpy.persistent_latent_v5`,
  `coordpy.multi_hop_translator_v3`,
  `coordpy.mergeable_latent_capsule`,
  `coordpy.deep_proxy_stack_v4`,
  `coordpy.ecc_codebook_v5`,
  `coordpy.long_horizon_retention_v5`,
  `coordpy.branch_merge_memory_v3`,
  `coordpy.corruption_robust_carrier`,
  `coordpy.transcript_vs_shared_arbiter_v2`,
  `coordpy.uncertainty_layer`,
  `coordpy.w53_team`, `coordpy.r104_benchmark`,
  `coordpy.r105_benchmark`, `coordpy.r106_benchmark`. Ten
  orthogonal capsule-native advances layered on top of W52:
  (M1) a 3-layer persistent latent state V5 with persistent
  identity-init skip-link applied at every step + a state-merge
  head; chain walks past 32 turns. (M2) a 5-backend (A,B,C,D,E)
  multi-hop translator V3 over 20 directed edges with chain-
  length-4 transitivity scoring and uncertainty-aware
  arbitration that returns per-dim 1-sigma confidence intervals.
  (M3) the **Mergeable Latent State Capsule (MLSC)** load-bearing
  new abstraction: content-addressed mergeable capsules with
  explicit ``MergeOperator`` + content-addressed
  ``MergeAuditTrail``; supports K-of-N consensus quorum with
  abstain semantics. (M4) a depth-10 deep proxy stack V4
  wrapping V3 with merge-aware + corruption-aware (per-layer
  L2-pathology) heads. (M5) a four-level ECC codebook V5
  K1=32 × K2=16 × K3=8 × K4=4 with XOR parity bits per segment;
  ≥ 14.5 bits/visible-token at full emit (empirically 17.67) +
  single-bit corruption detection. (M6) a four-headed long-
  horizon reconstruction V5 (causal + branch + cycle + merged-
  branch) at ``max_k=16`` (vs W52's 12) with a degradation curve
  probe across ``k ∈ {1..32}``. (M7) a branch merge memory V3
  with consensus pages populated by K-of-N quorum + content-
  addressed consensus audit + abstain when no quorum. (M8) a
  corruption-robust carrier composing ECC parity + 3-of-3
  majority repetition over the bits payload; reports detect /
  partial-correct / abstain / silent-failure rates. (M9) a
  transcript-vs-shared arbiter V2 with explicit per-turn policy
  over {transcript, shared, abstain} + 3-arm comparison with
  oracle-correctness rate. (M10) an uncertainty / confidence
  layer that composes per-component confidences into a single
  composite scalar + a calibration check.
  Composed by ``coordpy.w53_team.W53Team`` over the W52 envelope
  to produce ``W53HandoffEnvelope`` (closing the chain
  ``w47 → w48 → w49 → w50 → w51 → w52 → w53``).
  Carries forward W52's full verifier surface plus 30 new
  disjoint W53 envelope failure modes —
  **cumulative trust boundary across W22..W53 = 423
  enumerated failure modes** (393 from W22..W52 + 30 at W53).
  Verified by R-104 (12 cell families) + R-105 (10 cell
  families) + R-106 (12 cell families) at 3 seeds each
  (H1..H34 success criterion). Zero changes to released
  CoordPy SDK 0.5.20 surface, smoke driver, or runtime.
  W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY sharpened to
  W53-C-CROSS-TOKENIZER-QUINT-CAP. The
  W47-C-DEEP-TRANSFORMER-COUPLING ..
  W52-C-DEEP-TRANSFORMER-COUPLING line is *further-bounded*
  by W53 (chain length 4, persistent skip-link, MLSC
  abstraction, ECC parity) but remains substrate-blocked. See
  `docs/SUCCESS_CRITERION_W53_PMCRLOS.md`,
  `docs/RESULTS_W53_PMCRLOS.md`,
  `docs/RESEARCH_STATUS.md` for empirical results and the
  capsule-layer storyline; `docs/THEOREM_REGISTRY.md` for the
  theorem table; `docs/HOW_NOT_TO_OVERSTATE.md` for the
  do-not-overstate rules.

- **W52 Quantised Persistent Multi-Hop Latent Coordination**
  (post-W51, 2026-05-11) — `coordpy.persistent_latent_v4`,
  `coordpy.multi_hop_translator`,
  `coordpy.deep_proxy_stack_v3`,
  `coordpy.quantised_compression`,
  `coordpy.long_horizon_retention_v4`,
  `coordpy.branch_cycle_memory_v2`,
  `coordpy.role_graph_transfer`,
  `coordpy.transcript_vs_shared_state`,
  `coordpy.w52_team`, `coordpy.r102_benchmark`,
  `coordpy.r103_benchmark`. Eight orthogonal capsule-native
  advances layered on top of W51:
  (M1) a **stacked two-layer persistent latent state V4**
  with identity-init signal skip-link that carries the
  turn-0 signal through mid-sequence distractors; chain
  walks up to depth 24;
  (M2) a **multi-hop quad-backend translator** over four
  backend tags ``(A, B, C, D)`` with 12 directed edges,
  trained with length-2 and length-3 transitivity losses,
  plus a **disagreement-weighted arbitration** mechanism
  with per-edge confidence calibrated from training
  residuals (strictly beats naive equal-weight arbitration
  on perturbed-edge regimes);
  (M3) a depth-eight **deep proxy transformer stack V3**
  (vs W51's ``L=6``) with **role-conditioned KV banks**
  and **per-layer residual gate**;
  (M4) a **three-level quantised codebook V4** with
  coarse ``K1=32`` + fine ``K2=16`` + ultra-fine ``K3=8``
  (= 4096 codes ≈ 12 bits/triple) plus a learned adaptive
  budget allocator — achieves ≥ 14 bits/visible-token at
  full emit (vs W51's 12.0);
  (M5) a **three-headed long-horizon reconstruction V4**
  (causal + branch + cycle) at ``max_k=12`` (vs W51's
  ``max_k=8``) with a degradation-curve probe across
  ``k ∈ {1..24}``;
  (M6) a **branch/cycle memory V2** with trainable merge +
  importance-weighted evict heads + joint
  ``(branch, cycle)`` pages and a content-addressed merge
  audit trail;
  (M7) a new **role-graph conditioned cross-role transfer**
  module with per-edge ``(src_role, dst_role)`` linear
  projections that strictly beats equal-weight cross-role
  averaging on direction-dependent regimes;
  (M8) a **transcript-vs-shared-state matched-budget
  comparator** — the first capsule-native ablation that
  compares transcript truncation against shared-latent
  quantised encoding under a fixed visible-token budget;
  reports the strict retention gap and bit-density gap.
  Carries forward W51's full verifier surface plus 26 new
  disjoint W52 envelope failure modes —
  **cumulative trust boundary across W22..W52 = 393
  enumerated modes**. R-102 (12 cell families, 3 seeds) +
  R-103 (10 cell families, 3 seeds) verify the H1..H22
  success criterion — **22/22 H bars pass**. Headline
  results: 24-turn corrupted V4 retention 0.995 vs trained
  V3 0.757 (Δ +0.238); length-3 transitive fidelity 0.924,
  transitivity gap 0.058; disagreement-weighted
  arbitration 0.724 vs naive 0.375 (Δ +0.348); L=8 V3
  accuracy 0.764 vs L=6 V2 0.681 (Δ +0.083); role-graph
  transfer 0.806 vs equal-weight 0.080 (Δ +0.727);
  transcript-vs-shared retention gap +0.253 at matched
  B=3 budget; quantised 15.67 bits/visible-token; BCM V2
  joint recall 0.991 vs V1 0.657 (Δ +0.334); reconstruction
  V4 MSE at k=8 = 0.417, at k=12 = 0.369; verifier 1.000;
  replay determinism 1.000. Honest non-claims: L=8 does
  NOT strictly improve over L=6 V3 on shallow regimes
  (H21: -0.056); multi-hop translator under
  identity-friendly init preserves ~0.55 of clean signal
  even under forged training (H11 protect rate 0.43);
  ``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`` sharpened
  forward as ``W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY``.
  W52 ships at the explicit-import paths above; the
  released v0.5.20 wheel's public surface is byte-for-byte
  unchanged. See
  `docs/RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md` and
  `docs/SUCCESS_CRITERION_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`
  for the full result + pre-committed H bars.

- **W51 Persistent Cross-Backend Latent Coordination** (post-W50,
  2026-05-11) — `coordpy.persistent_shared_latent`,
  `coordpy.cross_backend_translator`,
  `coordpy.deep_proxy_stack_v2`,
  `coordpy.hierarchical_compression`,
  `coordpy.long_horizon_retention`,
  `coordpy.branch_cycle_memory`,
  `coordpy.w51_team`,
  `coordpy.r100_benchmark`, `coordpy.r101_benchmark`. Six
  orthogonal capsule-native advances layered on top of W50:
  (M1) a trainable **GRU-style persistent shared latent state
  V3** with the update rule
  ``s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ tanh(W_h · [s_{t-1}; x_t])``
  and a content-addressed ``PersistentLatentStateChain``
  recoverable from the envelope chain alone, plus a learned
  **cross-role mixer** producing per-role views of the team
  state with a learned blend coefficient;
  (M2) a **triple-backend translator** over three backend
  tags ``(A, B, C)`` with direct translators ``A→B``, ``A→C``,
  ``B→C`` plus a trainable transitivity loss penalising
  disagreement between ``A→C`` and ``A→B→C``, with a
  best-effort real-LLM triple-Ollama realism anchor when
  ``COORDPY_W51_OLLAMA_REACHABLE=1``;
  (M3) a depth-six **deep proxy transformer stack V2** (vs
  W50's ``L=4``) with branch-specialised heads, cycle-
  specialised heads, and per-layer learned temperature;
  (M4) a **hierarchical adaptive compression V3** with a
  coarse ``K1=32`` codebook + per-cluster fine ``K2=16``
  sub-codebooks plus a degradation-curve probe — achieves
  ≥ 12 bits/visible-token at full emit (vs W50's 8.0);
  (M5) a **two-headed long-horizon reconstruction V3**
  (causal + branch) at ``max_k=8`` (vs W50's ``max_k=3``)
  with a degradation-curve probe across ``k ∈ {1..16}``;
  (M6) a **branch/cycle-specialised memory head** with
  separate per-branch and per-cycle storage pages plus
  learned cross-branch consensus + cross-cycle merger.
  Carries forward W50's full 22-mode verifier surface plus
  24 new disjoint W51 envelope failure modes —
  **cumulative trust boundary across W22..W51 = 367 enumerated
  modes**. R-100 (11 cell families, 3 seeds) + R-101 (8 cell
  families, 3 seeds) verify the H1..H18 success criterion —
  **18/18 H bars pass**. Headline results: persistent state
  long-horizon recall 0.707 vs W50 baseline -0.237 (Δ +0.945);
  triple-backend direct fidelity 0.887 with transitivity gap
  0.087; branch/cycle memory recall 0.993 vs generic 0.785
  (Δ +0.208); hierarchical compression 13 bits/visible-token
  at full emit; 12-turn cosine retention 0.707; 16-turn stretch
  0.796; reconstruction V3 MSE at k=5 0.409, at k=8 0.462;
  verifier 1.000; replay determinism 1.000. Honest non-claims:
  L=6 does NOT strictly improve over L=4 under pure-Python
  autograd (structural floor + non-regression H4 met instead;
  the V2 win comes from branch/cycle specialisation H5: +0.056);
  ``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`` sharpened forward
  as ``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``. W51 ships
  at the explicit-import paths above; the released v0.5.20
  wheel's public surface is byte-for-byte unchanged. See
  `docs/RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md` and
  `docs/SUCCESS_CRITERION_W51_PERSISTENT_LATENT_COORDINATION.md`
  for the full result + pre-committed H bars.

- **W50 Cross-Backend Latent Coordination** (post-W49,
  2026-05-11) — `coordpy.cross_backend_alignment`,
  `coordpy.deep_proxy_stack`,
  `coordpy.adaptive_compression`,
  `coordpy.cross_bank_transfer`,
  `coordpy.shared_latent_carrier`,
  `coordpy.w50_team`,
  `coordpy.r98_benchmark`, `coordpy.r99_benchmark`. Five
  orthogonal capsule-native advances layered on top of W49:
  (M1) a trainable cross-backend latent projector that maps
  the W49 ``SharedLatentCapsule`` chain between two backend
  behaviors via a shared lingua-franca code, with a
  best-effort real-LLM realism anchor when
  ``COORDPY_W50_OLLAMA_REACHABLE=1``; (M2) a deeper proxy
  transformer stack at ``L=4`` (vs W49's ``L_p=2``) with
  per-layer learned mask gates + per-layer residual scales;
  (M3) an adaptive K=16 prototype codebook + learned per-bit
  emit-mask gate, packed as a ``LATENT_CTRL_V3`` control block
  with a ``CrammingWitnessV2`` recording structured-bits /
  visible-token ratio (target ≥ 8.0; W49 baseline 5.0);
  (M4) a role-pair-conditioned ``CrossBankTransferLayer`` that
  moves slot keys/values between role banks via a learned
  linear projection, paired with an
  ``AdaptiveEvictionPolicyV2`` (5-feature sigmoid scorer);
  (M5) a chain-walkable ``SharedLatentCarrierV2`` with a
  trainable ``ReconstructionV2Head`` recovering turn ``t-k``
  flat features for ``k ≤ 3``. The W50 envelope chain
  ``w47_outer → w48_proxy_outer → w49_multi_block_outer →
  w50_outer`` is verified by 20 disjoint failure modes
  (cumulative W22..W50 = 343 modes). R-98 (10 cell families)
  and R-99 (7 cell families) at 3 seeds each verify the
  H1..H16 success criterion. W50 ships at explicit-import
  paths and is NOT re-exported through `coordpy.__init__`;
  the released SDK contract (`coordpy.__version__ ==
  "0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`,
  smoke driver, public symbols) is byte-for-byte unchanged.
  Honest scope: W50 does NOT touch transformer-internal
  hidden states, KV cache bytes, attention weights,
  embeddings, or real tokenizers. The realism anchor
  **bounds** — not closes — the cross-tokenizer conjecture.
  Pure-Python only; reuses the W47 ``Variable`` +
  ``AdamOptimizer`` autograd engine.

- **W49 Multi-Block Cross-Bank Coordination** (post-W48,
  2026-05-11) — `coordpy.multi_block_proxy`,
  `coordpy.r96_benchmark`, `coordpy.r97_benchmark`. The
  strongest *executable proxy* for deep, role-aware,
  retention-headed transformer-internal coupling at the capsule
  layer: an `L_p`-stacked **multi-block proxy transformer**
  (`L_p = 2` default; each block = multi-head attention +
  position-wise tanh feed-forward + trainable residual scale);
  **role-conditioned multi-bank pseudo-KV** (one bank per role
  plus a shared team bank, with a trainable `BankRouter` for
  writes and a trainable `BankMixGate` for reads); a trainable
  **`EvictionPolicy`** that scores slots over
  `(age, role_match, write_gate)` and replaces FIFO under
  capacity pressure; a separate trainable **`RetentionHead`**
  answering binary "was this fact stored?" questions against the
  multi-bank read; a trainable **`DictionaryCodebook`**
  (`K = 8` default) that quantises the latent-control payload
  to a packed `LATENT_CTRL_V2` block carrying
  `code=<int>/<bits>b mask=... bits=...`; a content-addressed
  **`SharedLatentCapsule`** evolving per turn (chain-walk
  recovers all prior latent states from the envelope chain
  alone); a per-turn **`CrammingWitness`** recording structured-
  bits / visible-token frontier; a 22-mode envelope verifier.
  Cumulative trust boundary W22..W49 = **323 named failure
  modes**. R-96 + R-97 benchmarks (3 seeds × 16 cell families):
  multi_block_depth +0.292 acc on three-way XOR composition;
  multi_bank_recall +0.208 cosine; learned_eviction +0.859
  cosine on overflow; retention_head +0.500 accuracy;
  dictionary saves 25% control-block tokens; shared-latent
  chain-walk 1.000; cross-bank interference 0.000 (proves
  isolation); replay determinism 1.000; long-branch retention
  +0.268 cosine; cycle reconstruction +0.461 cosine; cramming
  bits/token 5.0 vs 3.0 (1.67×); live-anchor task-correct rate
  1.000 vs 0.000; aggressive compression +0.500 info/token;
  multi_block_distribution_cap 0.000 (limitation reproduces).
  Does NOT close W43..W48 substrate-blocked conjectures. New
  limitations: `W49-L-NO-REAL-KV-CAP`,
  `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP` (strengthens
  `W48-L-PROXY-DISTRIBUTION-CAP`). New conjectures:
  `W49-C-DEEP-TRANSFORMER-COUPLING` (carry-forward, bounds W48
  further), `W49-C-CROSS-MODEL-LATENT-TRANSFER`. See
  `docs/RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md` and
  `docs/SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md`.

- **W48 Shared-State Transformer-Proxy** (post-W47, 2026-05-11) —
  `coordpy.shared_state_proxy`, `coordpy.r95_benchmark`. The
  strongest *executable proxy* for transformer-internal
  coupling at the capsule layer: a single team-shared base
  state (`SharedStateCapsule`) stable across turns and roles, a
  per-role rank-`r` LoRA-style delta, a trainable **pseudo-KV
  factor bank** that reproduces `softmax(Q·K^T/sqrt(d))·V` with
  strict causal masking at the capsule layer, an `H=2` (default)
  **multi-head proxy attention block** with its own per-head
  `(W_Q, W_K, W_V)` + trainable output projection, a
  **slot-memory write head**, a **reconstruction decoder** that
  recovers prior-turn flat channel features from
  `(shared_state, flat_channels, pseudo_kv_read)`, a trainable
  **branch/cycle-aware bias matrix**, a bijective **branch-
  history compressor** (`BRANCH_HIST: <int> over <n_b>x<n_c>`),
  a learned **latent control serializer** (`LATENT_CTRL:
  SHARED_STATE_HASH=... mask=... bits=...`), content-addressed
  `TrainingTraceWitness` (reused from W47), and a 22-mode
  envelope verifier. Cumulative trust boundary W22..W48 = 301
  named failure modes. Does NOT close substrate-blocked W43
  conjectures or `W47-C-DEEP-TRANSFORMER-COUPLING`. New
  limitations: `W48-L-NO-REAL-KV-CAP`,
  `W48-L-PROXY-DISTRIBUTION-CAP` (strengthens
  `W47-L-AUTOGRAD-DISTRIBUTION-CAP`). New conjectures:
  `W48-C-REAL-KV-COUPLED-PROXY`, `W48-C-MULTI-HOST-SHARED-STATE`.
  See `docs/RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md` and
  `docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md`.

- **W47 Autograd Manifold Stack** (post-W46, 2026-05-10) —
  `coordpy.autograd_manifold`, `coordpy.r94_benchmark`. First
  end-to-end-trainable capsule-layer manifold-memory stack:
  pure-Python reverse-mode `Variable` autograd engine, Adam
  optimiser, trainable multi-layer tanh stack, trainable
  rank-r LoRA-style role adapter, trainable K-prototype
  dictionary, trainable QKV memory head, trainable
  packed-control serializer, content-addressed
  `TrainingTraceWitness`, 21-mode envelope verifier. Closes
  `W46-C-AUTOGRAD-DEEP-STACK` under the explicit "pure-Python
  reverse-mode AD + Adam SGD" reading. See
  `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md` and
  `docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`.

- **W46 Manifold Memory Controller** (post-W45, 2026-05-10) —
  `coordpy.manifold_memory`, `coordpy.r93_benchmark`. See
  `docs/RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md`.

- **W45 Learned Manifold Controller** (post-W44, 2026-05-10) —
  `coordpy.learned_manifold`, `coordpy.r92_benchmark`. See
  `docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md`.

- **W44 Live Manifold-Coupled Coordination** (post-W43,
  2026-05-10) — `coordpy.live_manifold`,
  `coordpy.r91_benchmark`. See
  `docs/RESULTS_COORDPY_W44_LIVE_MANIFOLD.md`.

- **W43 Product-Manifold Capsule** (post-0.5.20, 2026-05-10) —
  `coordpy.product_manifold`, `coordpy.r90_benchmark`. See
  `docs/RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md`.

## [0.5.20] coordpy-team CLI front door, replay-fidelity audit trail, curated presets

This release turns the lightweight `AgentTeam` surface into a real
front door for new users: a CLI you can drive without writing
Python, three curated multi-role presets, and a sealed manifest
shape that survives replay against a different backend.

### CLI

- New `coordpy-team` console script with four subcommands:
  - `coordpy-team run` — drive a curated preset over a task file
    (or stdin), dump a four-file replayable bundle.
  - `coordpy-team replay` — re-run a sealed `team_result.json`
    against any backend at the original sampling settings; fails
    loudly on per-turn `prompt_sha256` mismatch.
  - `coordpy-team sweep` — run the same task at multiple
    `max_visible_handoffs` settings; emit a side-by-side savings
    table.
  - `coordpy-team compare` — run on backend A, replay on backend
    B, report side-by-side telemetry and `ACTION` agreement.
- New `coordpy._pretty` TTY-aware presentation helpers (stdlib
  only) for the new CLI; degrades gracefully to plain ASCII when
  piped or redirected.

### Replay fidelity

This was the load-bearing audit fix. Pre-0.5.20, replaying a
sealed manifest could silently substitute the loader's defaults
(`temperature=0.2`, `max_tokens=256`) for the original sampling
settings, so a `quant_desk_team` run sealed at `temperature=0.0`
would not reproduce on replay. As of 0.5.20:

- `AgentTurn` persists `temperature`, `max_tokens`, `model_tag`,
  `prompt_sha256`, `prompt_words`, `naive_prompt_words`,
  `visible_handoffs`, `wall_ms`, real `prompt_tokens` /
  `output_tokens`, and `backend_base_url`.
- `TeamResult.dump(out_dir)` writes
  `team_result.json` (schema `coordpy.team_result.v1`),
  `team_capsule_view.json`, `team_report.md`, and
  `final_output.txt` — the four-file bundle that
  `coordpy-team replay` reads.
- `coordpy.replay_team_result(...)` re-applies the per-turn
  `temperature` and `max_tokens` from the manifest, not the
  loader's defaults; verifies the per-turn `prompt_sha256` to
  detect tampering; and seals fresh `TEAM_HANDOFF` capsules
  carrying the *new* model tag with the *original* prompt SHA so
  an auditor can prove "same prompt, different model."
- `capsule_team_handoff(...)` now accepts optional
  `prompt_sha256` / `prompt_bytes` / `model_tag` kwargs, stored
  in both the capsule body and metadata so `coordpy-capsule
  view --full` surfaces them in the audit view.

### Curated presets

- New `coordpy.presets` module with three opinionated multi-role
  teams that all share the bounded-context handoff story:
  - `presets.quant_desk_team(...)` — four-role US-equity quant
    desk; emits an `ACTION` line that
    `TeamResult.parse_action()` extracts as a structured
    `ActionDecision`.
  - `presets.code_review_team(...)` — three-role code-review
    team; emits an `APPROVE / REQUEST-CHANGES / BLOCK` verdict.
  - `presets.research_writer_team(...)` — three-role
    plan/research/write team for general Q&A.

### `AgentTeam` improvements

- `AgentTeam(task_summary=...)` and `AgentTeam.from_env(...,
  task_summary=...)` — agents 1..N see the one-line summary
  instead of the full task, so the bounded-context savings story
  is operational, not aspirational.
- `AgentTeam.run(task, *, progress=callable, should_continue=callable)`
  — a `ProgressCallback` fires once per finalised turn
  (essential UX for slow local LLMs), and a `ShouldContinue`
  callback can stop the team early (e.g. "the risk manager
  rejected all signals — abort").
- Per-turn handoff capsule budget is now auto-sized to the
  agent's `max_tokens` plus headroom (default `max(member.max_tokens,
  output_words+32, 128)`). The 0.5.18 `handoff_budget` knob still
  pins an explicit budget when set.
- `TeamResult.cramming_estimate()` reports the bounded vs naive
  token saving in real numbers; surfaced in `team_report.md`.
- `TeamResult.parse_action()` extracts a synthesizer's `ACTION:`
  line into a structured `ActionDecision`.
- `TeamResult.render_markdown()` produces a polished single-page
  report.

### Public surface

New top-level re-exports (visible in `dir(coordpy)`):
`ActionDecision`, `ProgressCallback`, `ShouldContinue`,
`TEAM_RESULT_SCHEMA`, `replay_team_result`, and the `presets`
submodule.

### Onboarding

- README now leads with the capsules + token-cramming framing and
  a five-minute first-run path through `coordpy-team`.
- `docs/START_HERE.md` drops the stale "not on PyPI yet" wording,
  the broken `vision_mvp.coordpy.*` import paths, and the
  references to the deleted `examples/04..06` files. The example
  ladder is rebuilt around the public API
  (`01_quickstart.py`, `02_quant_desk.py`, `03_replay_and_audit.py`)
  with a bundled `scenario_bullish.txt` for the quant-desk path.

### Compatibility

- Existing `team_result.json` manifests written by 0.5.16..0.5.19
  still replay; the new per-turn generation params are read when
  present and fall back to the lightweight defaults when absent.
- `coordpy.__init__` re-export list grows; nothing existing was
  removed or renamed.
- `capsule_team_handoff` keyword-only signature is additive (new
  optional kwargs); existing callers are unaffected.

## [0.5.18] AgentTeam: usable default budgets, agent() validation, friendlier CLI

A subagent build-test against `coordpy-ai 0.5.17` from PyPI hit
`CapsuleAdmissionError: capsule has 73 tokens but
budget.max_tokens=64` on the very first agent's first turn of an
otherwise routine three-agent code-review pipeline. A second
hostile build-test on the 0.5.18-rc surfaced four more papercuts
in adjacent surfaces. This release closes all of them.

### Defaults

- `_default_budget_for(TEAM_HANDOFF)`: `max_tokens` 64 → 4096,
  `max_bytes` 2 KiB → 64 KiB.
- `_default_budget_for(ROLE_VIEW)`: `max_tokens` 1024 → 32k,
  `max_bytes` 16 KiB → 256 KiB.

### `AgentTeam` / `create_team` knobs

- `AgentTeam(handoff_budget=...)` constructor knob, also wired
  through `AgentTeam.from_env(...)`, `AgentTeam.from_config(...)`,
  and the convenience `create_team(...)` function. Pass any
  `coordpy.CapsuleBudget` to override the per-handoff capsule
  budget. Tighten for benchmarks; widen for very long turns.
- `capsule_team_handoff(budget=...)` accepts the same kwarg for
  callers building handoff capsules directly.

### Validation

- `agent("")` and `agent(None, ...)` now raise `ValueError`
  immediately with a clear message. Previously they constructed
  an `Agent` with an empty/None name and only failed later (or
  silently produced an unusable agent).

### Test ergonomics

- `coordpy.SyntheticLLMClient` and `make_synthetic_response_fn`
  are re-exported at the top level (and visible via
  `dir(coordpy)`), so tests and examples don't need the
  `coordpy.synthetic_llm` submodule path.

### CLI: `coordpy-capsule`

- `view` / `verify` / `cid` / `audit` now accept either a
  `product_report.json` **or** a bare `capsule_view.json` /
  `TeamResult.capsule_view` dump. Previously the CLI required
  the wrapper shape (`{"capsules": ...}`), which made
  `team.run()` outputs unusable through the documented default
  command.
- `cid` on a view with no `RUN_REPORT` capsule (typical for
  `AgentTeam.run()` outputs, which seal `TEAM_HANDOFF` chains)
  now exits 2 with a clear message instead of printing a blank
  line.

Backwards-compatible: any code that admitted under the old
defaults still admits under the new ones; any CLI invocation
that worked before still works.

## [0.5.17] Curate the public surface

`dir(coordpy)` now returns the 82-name stable surface instead of
the 687 names that were re-exported at module level. Research /
experimental names live behind `coordpy.__experimental__` (a
tuple of name strings) and remain importable as
`coordpy.<name>` for back-compat, but they no longer pollute
autocomplete or `help(coordpy)`.

`__all__` is unchanged, so `from coordpy import *` keeps working
byte-for-byte for any code that already depended on it.

## [0.5.16] First PyPI release as `coordpy-ai`

Prepares the SDK for PyPI publication. Distribution name is
`coordpy-ai`; import name is still `coordpy`. The runtime
contract (one `RunSpec` in, one `RunReport` out with a sealed
capsule graph) is unchanged.

Packaging
- Distribution renamed to `coordpy-ai`. `pip install coordpy-ai`
  exposes `import coordpy`.
- PEP 621 / PEP 639 metadata. SPDX `license = "MIT"` plus
  `license-files = ["LICENSE"]`. Dropped the now-redundant
  `License :: OSI Approved` classifier.
- Single-source version: `dynamic = ["version"]` reads
  `coordpy._version.__version__`.
- PEP 561 `py.typed` marker in the wheel and a `Typing :: Typed`
  classifier.
- Added `maintainers`, broader keywords, the CPython
  implementation classifier, and config blocks for `ruff`,
  `black`, `mypy`, and `pytest`.
- `CITATION.cff` for citation discovery.
- Source layout flattened. The SDK is at top-level `coordpy/`;
  runtime-internal modules live under `coordpy/_internal/`. The
  legacy `vision_mvp` namespace is gone. Wheel went from ~2.4 MB
  to ~770 KB.

Fixes
- `CapsuleLedger.admit_and_seal` is now idempotent on CID, as the
  docstring already claimed.
- `ContextCapsule.new` raises `CapsuleAdmissionError` (not a bare
  `ValueError`) on budget violations.
- `register_report_sink`, `register_sandbox`, and
  `register_task_bank` accept either a factory callable or an
  already-built instance.
- `coordpy-capsule view|verify|cid|audit` prints a clean
  `error: report not found: ...` and exits 2 on a missing or
  malformed report path instead of dumping a traceback.

Tests and examples
- `tests/test_smoke_full.py`: 20 sections, ~80 checks against the
  installed wheel. Runs in under five seconds with no network.
- `examples/build_with_coordpy.py`: eight-step demo using only
  the public SDK and the synthetic backend.

## [0.5.16 / 3.43] — 2026-05-03 — SDK v3.43 — first public CoordPy release; final release of the SDK v3.4x line

Headline: cross-role-invariant synthesis + manifest-v12 CID +
role-handoff-signature axis + composite-collusion bounding (W42
family). New benchmark family R-89. The new theorem
`W42-L-FULL-COMPOSITE-COLLUSION-CAP` records the residual capsule-
layer wall. First measured cross-host paraphrase-invariance live
probe in the programme.

*Strictly additive on SDK v3.42 RC2.  The stable CoordPy product/runtime
(`RunSpec → run report`) is byte-for-byte unchanged.  The W42 surface
lives under `__experimental__`.  This release is the **final release**
of the SDK v3.4x research line — the **end-of-line for the
capsule-layer-only research programme** in Context Zero. The
remaining open frontiers (`W42-C-NATIVE-LATENT`,
`W42-C-MULTI-HOST`) are explicitly out of capsule-layer scope and
are next-programme work, not v3.43 blockers.*

**Install/use today:** install from a clone with `pip install -e .`,
then use the stable import surface `vision_mvp.coordpy` and the public
CLIs `coordpy`, `coordpy-import`, `coordpy-ci`, and
`coordpy-capsule`.  `pip install coordpy` / `pipx install coordpy`
remain the intended published install paths once the package is on
PyPI.

### Added — W42 family (cross-role-invariant synthesis + composite-collusion bounding)

* **`vision_mvp/coordpy/role_invariant_synthesis.py`** — new module
  implementing the W42 third-axis bounding mechanism on top of
  W41 integrated synthesis.  Adds:
  - ``RoleInvariantSynthesisRatificationEnvelope`` (sealed
    manifest-v12 envelope binding the W41 parent CID + 4 W42
    component CIDs + role-handoff signature CID).
  - ``RoleInvariancePolicyEntry`` /
    ``RoleInvariancePolicyRegistry`` (controller-side honest
    mapping ``role_handoff_signature_cid -> expected_services``).
  - ``RoleInvariantSynthesisRegistry`` (W42 orchestrator
    registry with ``invariance_enabled``, ``manifest_v12_disabled``,
    and ``abstain_on_invariance_diverged`` knobs).
  - ``W42RoleInvariantResult`` (per-cell W42 result record).
  - ``RoleInvariantSynthesisOrchestrator`` (closed-form,
    deterministic, zero-parameter wrapper of
    ``IntegratedSynthesisOrchestrator``).
  - ``compute_role_handoff_signature_cid`` (deterministic
    SHA-256 over canonical sorted ``(role, kind, payload)``
    tuples; permutation-invariant; payload-canonicalised;
    namespaced ``w42_role_handoff_signature``).
  - ``select_role_invariance_decision`` (closed-form,
    zero-parameter classifier returning one of seven invariance
    branches: ``trivial_invariance_passthrough`` /
    ``invariance_disabled`` / ``invariance_rejected`` /
    ``invariance_no_trigger`` / ``invariance_no_policy`` /
    ``invariance_ratified`` /
    ``invariance_diverged_abstained``).
  - ``verify_role_invariant_synthesis_ratification`` (pure-
    function verifier enumerating 14 disjoint W42 failure modes
    namespaced disjoint from the W22..W41 cumulative 182 modes;
    cumulative trust boundary across W22..W42 = **196 enumerated
    failure modes**).
  - ``build_trivial_role_invariant_registry`` /
    ``build_role_invariant_registry`` (registry builders).

* **`vision_mvp/experiments/phase89_role_invariant_synthesis.py`**
  — new R-89 driver with five banks (``trivial_w42`` /
  ``role_invariant_agrees`` / ``role_invariant_recover`` /
  ``full_composite_collusion`` /
  ``insufficient_invariance_policy``) plus a 5-seed sweep entry
  point.  Adds a synthetic per-cell incident-marker handoff
  (``incident_dispatcher`` role, ``w42_incident_id`` claim_kind)
  on non-trivial banks to simulate the upstream cell-schema
  dispatcher tag a real deployment would emit; the marker uses
  an unknown ``claim_kind`` and is ignored by the W21..W41 inner
  chain by construction.

* **`vision_mvp/experiments/phase89_xllm_role_invariance_probe.py`**
  — bounded live cross-host paraphrase-invariance probe at
  temperature 0 on the two-Mac topology (``localhost``
  gemma2:9b + ``192.168.12.191`` qwen2.5:14b) across K=4
  paraphrases of one closed-vocabulary arithmetic prompt.

* **`vision_mvp/tests/test_phase89_role_invariant_synthesis.py`**
  — 40 new W42 unit tests covering signature determinism +
  permutation-invariance + payload canonicalisation, decision
  selector branches, policy registry, CID determinism, all 14
  verifier failure modes, all 5 R-89 banks, and orchestrator
  passthrough.

* **`docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`**
  — pre-committed success criterion with H1..H12 hard gates +
  S1..S7 soft gates + named falsifiers + final release verdict
  structure.

* **`docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`** —
  results note with empirical headlines, density / efficiency
  measurements, trust boundary enumeration, theoretical claims,
  hard-gate / soft-gate aggregate, forced release verdict, and
  end-of-line declaration.

* **`docs/THEOREM_REGISTRY.md`** — appended W42-1..W42-4 +
  W42-L-FULL-COMPOSITE-COLLUSION-CAP + W42-L-TRIVIAL-PASSTHROUGH
  + W42-L-INSUFFICIENT-INVARIANCE-POLICY + W42-C-NATIVE-LATENT +
  W42-C-MULTI-HOST + W42-LIVE-CROSS-HOST-PARAPHRASE entries.

* **`docs/RESEARCH_STATUS.md`** — new TL;DR for SDK v3.43 final
  release with the strict +0.500 gain headline + final-release
  declaration + cumulative trust boundary update.

* **`docs/START_HERE.md`** — header table updated with the W42
  milestone as the latest milestone.

* **`docs/context_zero_master_plan.md`** — appended SDK v3.43
  W42 family section with the eight forced master-plan synthesis
  questions answered for W42 + the end-of-line declaration.

* **`README.md`** — headline updated to W42 final release.

### Headline empirical results

* **R-89-ROLE-INVARIANT-RECOVER (load-bearing)**:
  ``trust_precision_w42 = 1.000`` strictly improving over
  ``trust_precision_w41 = 0.500``;
  ``Δ_trust_precision_w42_w41 = +0.500`` across 5/5 seeds (min =
  max).  The first measured strict trust-precision recovery on a
  regime where W41 tied at 0.500.  Aggregate branch
  distribution: 40 ``invariance_ratified`` (prefix half) + 40
  ``invariance_diverged_abstained`` (recovery half).

* **R-89-TRIVIAL-W42**: byte-for-W41 across 5/5 seeds; W42
  overhead = 0; aggregate branch distribution: 80
  ``trivial_invariance_passthrough``;
  ``all_byte_equivalent_w42_w41 = True``.

* **R-89-ROLE-INVARIANT-AGREES**: no regression; all 80 cells
  ``invariance_ratified``;
  ``Δ_trust_precision_w42_w41 = 0.000``.

* **R-89-FULL-COMPOSITE-COLLUSION**:
  ``W42-L-FULL-COMPOSITE-COLLUSION-CAP`` fires; W42 ratifies on
  the wrong colluded set on every cell;
  ``Δ_trust_precision_w42_w41 = 0.000``.

* **R-89-INSUFFICIENT-INVARIANCE-POLICY**: W42 falls through
  ``invariance_no_policy``; preserves W41 byte-for-W40 on the
  answer.

* **W42 cross-host paraphrase-invariance live probe (2026-05-03)**:
  4/4 paraphrases × 2 hosts × 2 model architectures (gemma2:9b
  + qwen2.5:14b), all 4/4 gold-correct on each host;
  cross-host normalised agreement = 1.000; both hosts paraphrase-
  invariant (1 distinct normalised answer per host).

### Regression confidence

* 738/738 phase69-89 focused W22..W42 regression pass (was
  698/698 phase69-88 at SDK v3.42 RC2; W42 added 40 cleanly).
* 889/889 phase4-7 broad spot-check pass (excluding pre-existing
  ``test_phase50_ci_and_zero_shot.py`` collection-time hang
  carried forward unchanged from W41).

### Versioning

* ``SDK_VERSION = coordpy.sdk.v3.43``.
* ``vision_mvp.__version__ = 0.5.16``.
* ``pyproject.toml version = 0.5.16`` (alignment maintained).

### Final release verdict

The SDK v3.43 line ships as the **final release** of the CoordPy
SDK v3.4x research line.  Every hard gate (H1..H12) and every
load-bearing soft gate (S3, S5, S6, S7) of the W42 success
criterion passes.  The strict +0.500 trust-precision gain on
R-89-ROLE-INVARIANT-RECOVER is reproducible across 5/5 seeds
(min = max).  The live two-Mac paraphrase-invariance probe is
recorded honestly.

This is the **end-of-line for the capsule-layer-only research
programme** in the Context Zero project.  The remaining open
frontiers (``W42-C-NATIVE-LATENT``, ``W42-C-MULTI-HOST``) are
explicitly out of capsule-layer scope and require new
architectural substrate (transformer-internal access, K+1-host
topology, or both).

## [0.5.15 / 3.42] — 2026-05-03 — SDK v3.42 RC2 — integrated multi-agent context synthesis + manifest-v11 CID + cross-axis witness CID + producer-axis x trust-axis decision selector + R-88 Phase-88 benchmark family + W41-L-COMPOSITE-COLLUSION-CAP limitation theorem + W41-INFRA-1 (.101 retracted: AirPlay receiver, not Mac) + RC2 declaration

*Strictly additive on SDK v3.41 RC1.  The stable CoordPy product/runtime
(`RunSpec → run report`) is byte-for-byte unchanged.  W41 surface lives
under `__experimental__`.  This release is the **second release-
candidate (RC2)** of the SDK v3.4x line.*

### Added — W41 family (integrated multi-agent context synthesis)

* **`IntegratedSynthesisOrchestrator`** wraps W40
  (`CrossHostResponseHeterogeneityOrchestrator`) with a cross-axis
  classification layer that jointly binds the strongest old-line
  explicit-capsule trust-adjudication chain (W21..W40) AND the
  strongest cross-role / multi-round bundle decoder family (W7..W11)
  into a single auditable end-to-end path with one **manifest-v11**
  envelope binding both axes plus a content-addressed cross-axis
  witness.  W41 is closed-form, zero-parameter, and capsule-layer; it
  does NOT add a transformer-internal mechanism, does NOT close
  `W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`, and does NOT close its
  own new `W41-L-COMPOSITE-COLLUSION-CAP` limitation theorem.

* **Cross-axis decision selector**
  `select_integrated_synthesis_decision`: deterministic 8-branch
  classifier returning one of: `trivial_integrated_passthrough`,
  `integrated_disabled`, `integrated_rejected`,
  `integrated_producer_only`, `integrated_trust_only`,
  `integrated_both_axes`, `integrated_axes_diverged_abstained`,
  `integrated_neither_axis`.

* **Manifest-v11 CID** binds five component CIDs: `parent_w40_cid`,
  `synthesis_state_cid`, `synthesis_decision_cid`,
  `synthesis_audit_cid`, and `cross_axis_witness_cid` (an explicit
  per-cell witness namespaced as `w41_cross_axis_witness` so
  substituting a W22..W40 audit / witness for it is mechanically
  rejected).

* **W41 verifier** (`verify_integrated_synthesis_ratification`)
  enumerates 14 failure modes disjoint from W22..W40 (cumulative 182
  across W22..W41).

* **R-88 Phase-88 benchmark family** (5 banks): `trivial_w41`,
  `both_axes`, `trust_only_safety`, `composite_collusion`,
  `insufficient_response_signature`.

* **W41-L-COMPOSITE-COLLUSION-CAP** (NEW limitation theorem): when
  the adversary coordinates BOTH the producer-side admission AND the
  trust-side W40 ratification on the wrong set, W41 cannot recover
  at the capsule layer.

* **W41-INFRA-1** (lab topology retraction): `192.168.12.101` is an
  Apple TV / AirPlay receiver (`AirTunes/860.7.1` banner on port
  5000; locally-administered MAC `36:1c:eb:dc:9a:04`), NOT a Mac
  running Ollama.  The W37..W40 "TCP-up + HTTP-broken Ollama Mac"
  framing is retracted at this milestone.  Honest live topology:
  two-Mac pair (`localhost` + `192.168.12.191`).  `192.168.12.248`
  recorded as gone per user instruction.

### Empirical results (R-88, 5/5 seeds × 16 cells/seed)

* `trivial_w41`: W41 = W40 byte-for-byte; overhead = 0.
* `both_axes`: correctness_w41 = correctness_w40 = 1.000;
  trust_precision_w41 = trust_precision_w40 = 1.000; delta = 0.
* `trust_only_safety`: trust_precision_w41 = trust_precision_w40 =
  1.000 on the safety branch (W41 routes through the
  INTEGRATED_TRUST_ONLY safety branch on the recovery half);
  correctness_w41 = correctness_w40 = 0.500.
* `composite_collusion`: trust_precision_w41 = trust_precision_w40
  = 0.500; W41-L-COMPOSITE-COLLUSION-CAP fires; delta = 0.
* `insufficient_response_signature`: W41 = W40 byte-for-W39 on the
  answer; delta = 0.

All five banks: `all_w41_verified_ok = True`; W41 overhead = 1
visible token/cell (= 0 on `trivial_w41`); ~15.5k structured bits
per cell of cross-axis state under manifest-v11.

### Tests

698/698 phase69-88 focused W22..W41 stack regression (was 661/661
at W40 RC1; W41 added 37 cleanly) + 364/364 phase11-39 broad spot
check + 205/205 phase40-51 + phase6 broad spot check (1267 tests
pass excluding the named pre-existing
`test_phase50_ci_and_zero_shot` collection-time hang carried
forward unchanged from W40).

### Versioning

* `SDK_VERSION` = `coordpy.sdk.v3.42`
* `vision_mvp.__version__` = `0.5.15`
* `pyproject.toml` version = `0.5.15`

## [0.5.14 / 3.41] — 2026-05-03 — SDK v3.41 RC1 — cross-host response-signature heterogeneity ratification + manifest-v10 CID + cross-host response-text Jaccard divergence guard + R-87 Phase-87 benchmark family + W40-L-COORDINATED-DIVERSE-RESPONSE-CAP limitation theorem + W40-INFRA-1 (.101 TCP-up + HTTP-broken) + RC1 declaration

*Strictly additive on SDK v3.40.  The stable CoordPy product/runtime
(`RunSpec → run report`) is byte-for-byte unchanged.  W40 surface lives
under `__experimental__`.  This release is the **first official release-
candidate (RC1)** of the SDK v3.4x line.*

### Added — W40 family

* **`CrossHostResponseHeterogeneityOrchestrator`** wraps W39
  (`MultiHostDisjointQuorumOrchestrator`) with a cross-host
  response-signature heterogeneity layer that operates on an evidence
  axis ORTHOGONAL to top_set: the per-member response **text bytes**
  themselves.  Even if K colluders coordinate their declared top_set
  (the W39 full-quorum-collusion attack), naturally-independent K
  probes should produce heterogeneous response text bytes.  When the
  K member probes' mean pairwise Jaccard divergence over canonical
  sorted token bags falls strictly below
  `response_text_diversity_min`, W40 abstains via
  `RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`.

* **Cross-host response-heterogeneity (Jaccard divergence) score**:
  closed-form, zero-parameter, deterministic
  `1 - |inter|/|union|` over canonical sorted whitespace token bags
  (lower-cased, punctuation-stripped, deduplicated).  Permutation-
  and case-invariant.

* **Manifest-v10 CID** binds six component CIDs:
  `parent_w39_cid`, `response_signature_state_cid`,
  `response_signature_audit_cid`,
  `response_signature_topology_cid`,
  `response_signature_decision_cid`, and the new
  `response_heterogeneity_witness_cid` (an explicit per-pair-
  intersection witness namespaced as
  `w40_response_heterogeneity_witness` so swapping a W39 mutual-
  disjointness witness for a W40 heterogeneity witness or vice-versa
  is mechanically rejected).

* **W40 verifier**
  (`verify_cross_host_response_heterogeneity_ratification`)
  enumerates 14 failure modes disjoint from W22..W39:
  `empty_w40_envelope`, `w40_schema_version_unknown`,
  `w40_schema_cid_mismatch`, `w39_parent_cid_mismatch`,
  `w40_projection_branch_unknown`,
  `w40_response_probe_unregistered_host`,
  `w40_response_probe_unregistered_oracle`,
  `w40_response_disjoint_topology_violation`,
  `w40_response_mutual_disjointness_violation`,
  `w40_response_thresholds_invalid`,
  `w40_response_state_cid_mismatch`,
  `w40_response_decision_cid_mismatch`,
  `w40_response_topology_cid_mismatch`,
  `w40_manifest_v10_cid_mismatch` (with `w40_outer_cid_mismatch`
  and the heterogeneity-witness swap-detection co-defined).
  Cumulative trust boundary across W22 + W29 + W30 + W31 + W32 +
  W33 + W34 + W35 + W36 + W37 + W38 + W39 + W40 = **168 enumerated
  failure modes**.

* **`ResponseSignatureProbe`**: one W40 member probe, carrying
  `(member_index, host_ids, oracle_ids, response_token_bag,
  response_signature_cid, cell_idx)`.  Closed-form, deterministic,
  zero-parameter, audited capsule-layer artefact.

* **`MultiHostResponseHeterogeneityProbe`**: a frozen tuple of K
  `ResponseSignatureProbe` member probes plus
  `response_text_diversity_min` and
  `min_response_signature_probes`.  Mechanically enforces both
  trajectory-disjointness AND mutual-disjointness preconditions
  inherited from W39.

* **`select_cross_host_response_heterogeneity_decision`**:
  closed-form zero-parameter cross-host response-heterogeneity
  selector.  Returns one of:
  `RESPONSE_SIGNATURE_NO_REFERENCES`,
  `RESPONSE_SIGNATURE_NO_TRIGGER`,
  `RESPONSE_SIGNATURE_INSUFFICIENT`,
  `RESPONSE_SIGNATURE_INCOMPLETE`,
  `RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`,
  `RESPONSE_SIGNATURE_DIVERSE`.

* **R-87 / Phase-87 benchmark family**: five regimes —
  `trivial_w40`, `response_signature_collapse` (load-bearing),
  `no_regression_diverse_agrees`, `coordinated_diverse_response`
  (W40-L-COORDINATED-DIVERSE-RESPONSE-CAP),
  `insufficient_response_signature`.

* **W40 unit tests**: 36/36 passing in
  `vision_mvp/tests/test_phase87_cross_host_response_heterogeneity.py`.

### Measured

* On **R-87-RESPONSE-SIGNATURE-COLLAPSE** (the load-bearing
  regime): W40 wins **+0.500 trust precision** over W39 across
  5/5 seeds × 16 cells/seed (W39 trust precision 0.5; W40 trust
  precision 1.000).  W40 abstains via
  `RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED` on 8 cells/seed in the
  recovery half (the W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP
  regime where W39 RATIFIES the wrong top_set), preserves W39
  behaviour on the prefix half (W39 didn't RATIFY → W40 fires
  RESPONSE_SIGNATURE_NO_TRIGGER).  W40 overhead: 1 visible
  token/cell.  Structured state density: ~14.5k bits per visible
  W40 token.

* On **R-87-TRIVIAL-W40**: W40 = W39 byte-for-byte across 5/5
  seeds; overhead = 0.

* On **R-87-NO-REGRESSION-DIVERSE-AGREES**: W40 = W39 (delta = 0)
  across 5/5 seeds; W40 ratifies via RESPONSE_SIGNATURE_DIVERSE
  on 8 cells/seed.

* On **R-87-COORDINATED-DIVERSE-RESPONSE**:
  `W40-L-COORDINATED-DIVERSE-RESPONSE-CAP` fires; delta_trust =
  0 across 5/5 seeds.  Closure requires native-latent evidence
  outside the capsule layer or a K+1-host disjoint topology with
  a new genuinely uncompromised pool.

* On **R-87-INSUFFICIENT-RESPONSE-SIGNATURE**: W40 returns
  `RESPONSE_SIGNATURE_INSUFFICIENT` on every recovery cell where
  fewer than `min_response_signature_probes=2` member probes are
  present; delta = 0 across 5/5 seeds.

### Lab topology — W40-INFRA-1

* `192.168.12.101` re-probed (2026-05-03):
  - `ping -c 2`: **0% packet loss**, 30 ms RTT (a strict
    improvement over the W39 end-state of 100% packet loss).
  - `nc -zv 11434`: TCP connect succeeds.
  - `nc -zv 22`: TCP SSH connect succeeds (auth methods
    advertised: publickey, password, keyboard-interactive).
  - `curl /api/version`: still "Empty reply from server" /
    "Connection reset by peer" across 5 attempts at 60-second
    timeout (the W39-INFRA-1 hung-listener pattern).
  - SSH attempted via `nobody@`, `qdong@`, `root@`, `admin@`:
    all rejected for lack of credentials.
* Verdict: **W40-INFRA-1**.  `192.168.12.101` is now **TCP-up +
  HTTP-broken** at the Ollama layer; the host network stack has
  recovered (a strict improvement over W39); the Ollama listener
  remains in the W39-INFRA-1 hung-listener state.  Restoration
  requires SSH credentials still unavailable in this environment.
* `192.168.12.248` remains ARP-incomplete (32nd milestone in a
  row).

### Newly named

* **W40-L-COORDINATED-DIVERSE-RESPONSE-CAP**: new limitation
  theorem (the W40 analog of W34/W37/W38/W39 collusion-CAPs).
* **W40-L-INSUFFICIENT-RESPONSE-SIGNATURE**: new falsifier.
* **W40-L-NATIVE-LATENT-GAP**: new native-latent gap.
* **W40-C-NATIVE-LATENT**: open conjecture.
* **W40-C-MULTI-HOST**: open conjecture (partially discharged at
  the topology layer via .101 TCP-up; still open at the
  live-inference layer via W40-INFRA-1).
* **W40-C-LIVE-RESPONSE-HETEROGENEITY** (NEW): empirical-
  validated at the synthetic-bench layer; conjectural at the
  live-inference layer pending a dedicated live collusion-
  recovery experiment.

### Release readiness — RC1 declaration

* **RC1 declared.**  H1..H12 + S3 of the W40 success criterion
  pass.  The SDK v3.41 line is the **first official release-
  candidate** of the CoordPy SDK v3.4x line.
* `coordpy.sdk.v3.41`; pyproject `0.5.14`;
  `vision_mvp.__version__ == "0.5.14"` (alignment maintained).
* Stable runtime contract is byte-for-byte unchanged.
* Stable-vs-experimental boundary is final for RC1: every
  W22..W40 symbol is exported under `__experimental__`.
* Open-conjectures + limitation-theorems cut-list pinned in
  `THEOREM_REGISTRY.md`.
* 36/36 W40 unit tests + 661/661 phase69-87 focused W22..W40
  stack regression (was 625/625 phase69-86 at SDK v3.40; W40
  added 36 cleanly) + 364/364 phase11-39 broad spot check +
  205/205 phase40-51 + phase6 broad spot check (1230 total
  tests pass excluding test_phase50_ci_and_zero_shot.py which
  has a pre-existing collection-time hang independent of W40 —
  carried forward unchanged from W39).

## [0.5.13 / 3.40] — 2026-05-02 — SDK v3.40 — multi-host disjoint quorum consensus-reference ratification + manifest-v9 CID + mutually-disjoint physical-host topology + R-86 Phase-86 benchmark family + W39 multi-host quorum-bounding limitation theorem + bounded 5-host live disjoint quorum probe + historical `.101` third-host candidate reading (later retracted at W41)

*Strictly additive on SDK v3.39.  The stable CoordPy product/runtime
(`RunSpec → run report`) is byte-for-byte unchanged.  W39 surface lives
under `__experimental__`.*

### Added — W39 family

* **`MultiHostDisjointQuorumOrchestrator`** wraps W38
  (`DisjointConsensusReferenceOrchestrator`) with a multi-host
  disjoint quorum layer.  At every cell where W37/W38 reroutes,
  W39 cross-checks the candidate top_set against a
  controller-pre-registered `MultiHostDisjointQuorumProbe` consisting
  of K member probes, each sourced from a physically-distinct host
  pool that is both mechanically disjoint from the W37 trajectory
  hosts AND mutually disjoint from every other quorum probe's pool.
  When at least `quorum_min` of the K member probes diverge from the
  W37/W38 candidate top_set, W39 abstains via the
  `QUORUM_DIVERGENCE_ABSTAINED` branch.

* **Mechanical mutual-disjointness enforcement**: the
  `MultiHostDisjointQuorumRegistry.__post_init__` raises
  `MutuallyDisjointTopologyError` if any two registered quorum host
  pools have non-empty intersection.  The verifier additionally
  rejects envelopes claiming an overlapping pool pair
  (`w39_quorum_mutual_disjointness_violation`).  This is a STRICT
  generalisation of W38-4: in addition to enforcing disjointness from
  the trajectory host set (inherited from W38), W39 enforces mutual
  disjointness across all registered quorum pools.

* **Manifest-v9 CID** binds six component CIDs:
  `parent_w38_cid`, `quorum_state_cid`, `quorum_audit_cid`,
  `quorum_topology_cid`, `quorum_decision_cid`, and the new
  `mutual_disjointness_cid` (an explicit per-pair-intersection
  witness so the verifier can defense-in-depth check that the
  declared topology is mutually disjoint, even if the outer
  topology CID happens to match an unrelated registered topology of
  equal canonical form).

* **W39 verifier** (`verify_multi_host_disjoint_quorum_ratification`)
  enumerates 14 failure modes disjoint from W22..W38:
  `empty_w39_envelope`, `w39_schema_version_unknown`,
  `w39_schema_cid_mismatch`, `w38_parent_cid_mismatch`,
  `w39_projection_branch_unknown`, `w39_quorum_probe_unregistered_host`,
  `w39_quorum_probe_unregistered_oracle`,
  `w39_quorum_disjoint_topology_violation`,
  `w39_quorum_mutual_disjointness_violation`,
  `w39_quorum_thresholds_invalid`, `w39_quorum_state_cid_mismatch`,
  `w39_quorum_decision_cid_mismatch`,
  `w39_quorum_topology_cid_mismatch`, `w39_manifest_v9_cid_mismatch`
  (with `w39_outer_cid_mismatch` co-defined).  Cumulative trust
  boundary across W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 +
  W36 + W37 + W38 + W39 = **154 enumerated failure modes**.

* **`MultiHostDisjointQuorumProbe`**, with K member
  `ConsensusReferenceProbe` instances, each carrying
  `(top_set, consensus_host_ids, consensus_oracle_ids,
  consensus_strength, cell_idx)`.  Closed-form, deterministic,
  zero-parameter, audited capsule-layer artefact.

* **`select_multi_host_disjoint_quorum_decision`**: closed-form
  zero-parameter K-of-N quorum selector.  Returns one of:
  `QUORUM_NO_REFERENCES`, `QUORUM_NO_TRIGGER`,
  `QUORUM_INSUFFICIENT`, `QUORUM_REFERENCE_WEAK`,
  `QUORUM_DIVERGENCE_ABSTAINED`, `QUORUM_RATIFIED`,
  `QUORUM_SPLIT`.

* **R-86 / Phase-86 benchmark family**: five regimes —
  `trivial_w39`, `multi_host_colluded_consensus` (load-bearing),
  `no_regression_quorum_agrees`, `full_quorum_collusion`
  (W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP), `insufficient_quorum`.

* **W39 unit tests**: 31/31 passing in
  `vision_mvp/tests/test_phase86_multi_host_disjoint_quorum.py`.
  Focused W22..W39 regression: **625/625 phase69-86** passing
  (was 594/594 phase69-85 at SDK v3.39; W39 added 31 cleanly).
  Broader spot checks: 364/364 phase11-39 + 205/205 phase40-51 +
  phase6 (excluding `test_phase50_ci_and_zero_shot.py` collection-
  time hang carried forward from earlier milestones).

### Added — Lab topology resolution + live evidence

* **Historical W39 topology reading (superseded at W41).**
  ``192.168.12.248`` remained ARP-incomplete for the 31st milestone
  in a row.  At W39 time, ``192.168.12.101`` was treated as the
  reachable third physical host candidate, **partially discharging
  W38-C-MULTI-HOST at the topology layer**: preflight-OK on cold
  contact with `qwen3.5:35b` and `qwen2.5:14b-32k` model files
  visible.  This specific `.101` interpretation was later retracted
  at W41, which identified `.101` as an Apple TV / AirPlay receiver,
  not a Mac.

* **`W39-INFRA-1`** named: ``.101`` Ollama inference path is
  bounded under capsule-layer one-word probe budget after first
  model-load contact.  Diagnosed across `/api/chat`, `/api/generate`,
  `/api/show`, `/api/ps`, `/api/tags`, ping; no recovery in
  5-minute polling; SSH access unavailable to restart the service.

* **W39-INFRA-1 robust fallback** in
  `phase86_xllm_quorum_probe.py`: when ``.101`` is unreachable,
  `mac_off_cluster_a` swaps to localhost running `llama3.1:8b` (a
  model class genuinely different from the trajectory's
  `gemma2:9b`), so the live K=2 quorum becomes `(localhost
  llama3.1:8b, .191 qwen2.5-coder:14b-32k)` -- two physically
  distinct hosts, each running a different model class from the
  trajectory pair AND from the W38 single consensus reference.

* **First measured 5-host live W39 disjoint-quorum probe in the
  programme**: 8/8 responsive on all 5 hosts, 7/8
  trajectory_pair_agrees, 7/8 W38 consensus agreements, 8/8
  quorum_a gold-correlated, 8/8 quorum_b gold-correlated, 8/8 K=2
  quorum size simultaneously responsive.

* **`W39-C-LIVE-TRUNCATION-RECOVERY`** (new conjecture): on the
  ``h2o`` probe the trajectory pair disagreed (`mac1=h2o` vs
  `mac_remote=h` due to `num_predict=4` truncation) but BOTH
  quorum members got `h2o` correct -- empirical-suggestive
  evidence for a recovery axis at the live layer distinct from the
  collusion bound.

### Added — empirical headlines

* **R-86-MULTI-HOST-COLLUDED-CONSENSUS (load-bearing)**: across 5/5
  seeds, W38 trust precision = 0.500 (W38-L-CONSENSUS-COLLUSION-CAP
  fires); W39 trust precision = **1.000**;
  `Δ_trust_precision_w39_w38 = +0.500` (min and max equal).  W39
  abstains via QUORUM_DIVERGENCE on 8 cells/seed, ratifies 0,
  no_references 0, insufficient 0, split 0.  Overhead = 1 visible
  token/cell.  Density: **24.4k structured bits per visible W39
  token** (~2.7x denser than W38's 9.07k bits/token on R-85-COLLUDED-
  CROSS-HOST-TRAJECTORY -- the densest milestone-level capsule-layer
  audited proxy in the programme to date).

* **R-86-TRIVIAL-W39**: W39 = W38 byte-for-byte across 5/5 seeds;
  `all_byte_equivalent_w39_w38 = True`; overhead = 0.

* **R-86-NO-REGRESSION-QUORUM-AGREES**: across 5/5 seeds,
  `Δ_correctness_w39_w38 = Δ_trust_precision_w39_w38 = 0.000`; W39
  ratifies via QUORUM_RATIFIED on 8 cells/seed.

* **R-86-FULL-QUORUM-COLLUSION**: across 5/5 seeds,
  `Δ_trust_precision_w39_w38 = 0.000`; the new
  `W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP` limitation theorem
  fires.

* **R-86-INSUFFICIENT-QUORUM**: across 5/5 seeds, `Δ = 0`; W39
  returns `QUORUM_INSUFFICIENT` on the recovery half; W39 audit
  envelope still records the QUORUM_INSUFFICIENT branch.

### Versioning

* `SDK_VERSION` bumped to `coordpy.sdk.v3.40`.
* `vision_mvp.__version__` bumped to `0.5.13`.
* `pyproject.toml` `project.version` bumped to `0.5.13`.
* Alignment maintained between `vision_mvp.__version__` and
  `pyproject.toml`.

### Stable / Experimental boundary

* W39 surface (`MultiHostDisjointQuorumProbe`,
  `MultiHostDisjointQuorumRatificationEnvelope`,
  `MultiHostDisjointQuorumRegistry`,
  `W39MultiHostDisjointQuorumResult`,
  `MultiHostDisjointQuorumOrchestrator`,
  `MutuallyDisjointTopologyError`,
  `verify_multi_host_disjoint_quorum_ratification`,
  `select_multi_host_disjoint_quorum_decision`,
  `build_trivial_multi_host_disjoint_quorum_registry`,
  `build_multi_host_disjoint_quorum_registry`, plus W39 schema
  version + branch constants) is exported under `__experimental__`.
* Stable runtime contract (`RunSpec` → run report, capsule
  primitives, lifecycle audit) byte-for-byte unchanged.

### Honest open walls after W39

* `W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP` remains open at the
  capsule layer (closure requires native-latent evidence outside
  the capsule layer or a K+1-host disjoint topology with a new
  uncompromised pool).
* `W39-C-NATIVE-LATENT` remains open and architecture-dependent.
* `W39-C-MULTI-HOST` is partially discharged at the topology layer
  (``.101`` reachable on cold contact) and still open at the
  live-inference layer (``W39-INFRA-1``).
* `W39-C-LIVE-TRUNCATION-RECOVERY` is a new conjecture observed
  suggestively on a single live probe; sharper validation requires
  a dedicated live truncation-recovery bench.

## [0.5.12 / 3.39] — 2026-05-02 — SDK v3.39 — disjoint cross-source consensus-reference trajectory-divergence adjudication + manifest-v8 CID + R-85 Phase-85 benchmark family + W38 collusion-bounding limitation theorem + bounded 3-host live consensus probe + version reconciliation

*Strictly additive on SDK v3.38.  The stable CoordPy product/runtime
(`RunSpec → run report`) is byte-for-byte unchanged.  W38 surface lives
under `__experimental__`.*

### Added — W38 family

* **`DisjointConsensusReferenceOrchestrator`** wraps W37
  (`CrossHostBasisTrajectoryOrchestrator`) with a disjoint
  cross-source consensus-reference layer.  At every cell where W37
  reroutes on a trajectory-anchored top_set, W38 cross-checks the
  candidate top_set against a controller-pre-registered
  `ConsensusReferenceProbe` whose host topology is mechanically
  disjoint from W37's trajectory hosts.  If the W37 candidate
  diverges from the consensus reference by ≥ `divergence_margin_min`
  (Jaccard), W38 abstains via the `CONSENSUS_DIVERGENCE_ABSTAINED`
  branch.

* **Mechanical disjoint-topology enforcement**: the
  `DisjointConsensusReferenceRegistry.__post_init__` raises
  `DisjointTopologyError` if `consensus_host_ids ∩ trajectory_host_
  ids ≠ ∅`.  The verifier additionally rejects envelopes claiming an
  overlapping topology (`w38_disjoint_topology_violation` failure
  mode).

* **`ConsensusReferenceProbe`**: closed-form, zero-parameter,
  controller-pre-registered audited capsule-layer probe carrying
  `(top_set, consensus_host_ids, consensus_oracle_ids,
  consensus_strength, cell_idx)`.  Explicitly NOT a runtime
  ground-truth oracle.

* **Manifest-v8 CID** over five component CIDs (`parent_w37_cid`,
  `consensus_reference_state_cid`, `divergence_audit_cid`,
  `consensus_topology_cid`, `consensus_probe_cid`) detecting
  cross-component swaps that the W37 manifest-v7 alone cannot detect.

* **`verify_disjoint_consensus_reference_ratification`** enumerates
  14 disjoint W38 failure modes
  (`empty_w38_envelope`, `w38_schema_version_unknown`,
   `w38_schema_cid_mismatch`, `w37_parent_cid_mismatch`,
   `w38_projection_branch_unknown`,
   `w38_consensus_host_unregistered`,
   `w38_consensus_oracle_unregistered`,
   `w38_disjoint_topology_violation`,
   `w38_consensus_strength_out_of_range`,
   `w38_divergence_threshold_invalid`,
   `w38_consensus_state_cid_mismatch`,
   `w38_consensus_probe_cid_mismatch`,
   `w38_consensus_topology_cid_mismatch`,
   `w38_manifest_v8_cid_mismatch` (and `w38_outer_cid_mismatch`
   co-defined)).
  Cumulative W22..W38 trust boundary: **140 enumerated failure modes**.

### Added — R-85 Phase-85 benchmark family

* `vision_mvp/experiments/phase85_disjoint_consensus_reference.py`
  with five regimes:

  * `trivial_w38` — byte-for-W37 preservation.
  * `colluded_cross_host_trajectory` — load-bearing W38 collusion
    bound.  Across 5 seeds × 16 cells/seed,
    **Δ_trust_precision_w38_w37 = +0.500** (min and max equal),
    W38 trust precision = **1.000**, W38 reroutes 0 cells, abstains
    via DIVERGENCE on 8 cells/seed; overhead = 1 visible token/cell.
  * `no_collusion_consensus_agrees` — Δ = 0 across 5 seeds.
  * `consensus_also_compromised` — `W38-L-CONSENSUS-COLLUSION-CAP`
    fires; Δ = 0 across 5 seeds.
  * `no_consensus_reference` — W38 returns
    `CONSENSUS_NO_REFERENCE`; Δ = 0 across 5 seeds.

* Artifacts in
  `vision_mvp/experiments/artifacts/phase85/{trivial_w38, colluded_cross_host_trajectory, no_collusion_consensus_agrees, consensus_also_compromised, no_consensus_reference}_seed_sweep.json`.

### Added — bounded live W38 cross-source consensus probe

* `vision_mvp/experiments/phase85_xllm_consensus_probe.py` --
  bounded 3-host probe: `mac1` (localhost gemma2:9b) and
  `mac_remote` (192.168.12.191 qwen2.5:14b) as trajectory hosts;
  `mac_consensus` (192.168.12.191 qwen2.5-coder:14b) as the
  disjoint consensus host (different model class on the same
  physical host -- defensible weak proxy for capsule-layer
  disjointness, NOT a true 3-host disjoint topology).

* Result: **8/8 responsive on all 3 hosts**, **7/8 trajectory-pair
  agreements** (the one disagreement is a `num_predict=4`
  truncation: gold "h2o" → mac_remote answered "h"),
  **7/8 cross-source consensus agreements**,
  **8/8 consensus-gold correlation** at temperature 0.

* This is the strongest honest 3-host bounded consensus evidence
  the current infrastructure supports.  It does NOT close
  `W38-C-MULTI-HOST` (true 3-host disjoint topology -- which
  requires Mac 2 or another physical host) and does NOT close
  `W38-L-CONSENSUS-COLLUSION-CAP` (no live colluded scenario was
  attempted; the live probe is gold-correlated agreement evidence,
  not an attack-recovery measurement).

* Honest infra note: the qwen3.5:35b MoE host on Mac 1 was
  empirically non-responsive at temperature 0 + `num_predict=4`
  (the W34-INFRA pattern) -- recorded as `W38-INFRA-1`
  (consensus host model availability under one-word prompt budget;
  qwen2.5-coder:14b was used as the available substitute disjoint
  consensus model).

* Artifact:
  `vision_mvp/experiments/artifacts/phase85/xllm_consensus_probe_2026_05_02.json`.

### Theory

* **W38-1** — verifier boundary: 14 disjoint W38 failure modes
  mechanically tested.  *Proved by inspection + mechanically
  checked*.
* **W38-2** — trivial reduction: disabled consensus + disabled
  divergence-abstain + disabled manifest-v8 reduces to W37
  byte-for-byte.  *Empirical*.
* **W38-3** — disjoint-consensus collusion bound (load-bearing):
  *Proved-conditional + empirical*.
* **W38-4** — disjoint-topology mechanical enforcement.  *Proved by
  inspection + mechanically tested*.
* **W38-L-CONSENSUS-COLLUSION-CAP** — proved-conditional limitation
  theorem.  *Empirical on R-85-CONSENSUS-ALSO-COMPROMISED*.
* **W38-L-DISJOINT-CONSENSUS-REQUIRED** — falsifier.  *Proved by
  inspection + empirical on R-85-NO-CONSENSUS-REFERENCE*.
* **W38-C-NATIVE-LATENT** — open conjecture (architecture-dependent).
* **W38-C-MULTI-HOST** — open conjecture (hardware-bounded; Mac 2
  ARP-incomplete for the 31st milestone in a row).

### Mac 2 status

* `192.168.12.248:11434/api/tags` times out at 5 s; `ping` reports
  "Host is down"; ARP entry incomplete.  Mac 2 has been ARP-
  incomplete for the **31st milestone in a row**.

### Tests / regression

* New: `vision_mvp/tests/test_phase85_disjoint_consensus_reference.py`
  — 31 unit tests covering 3 divergence-score tests, 5 selector
  tests, 16 verifier tests (clean envelope + 14 enumerated failure
  modes + outer CID mismatch), 2 registry tests, 5 bank tests.
* Focused W22..W38 regression: **594/594 phase69-85** tests pass
  (was 563/563 phase69-84 at SDK v3.38; W38 added 31).
* Broad regression: see `pytest vision_mvp/tests` count recorded in
  the milestone report.

### Versioning / release

* `SDK_VERSION` bumped to `coordpy.sdk.v3.39`.
* `vision_mvp.__version__` bumped to `0.5.12`.
* `pyproject.toml` `project.version` bumped to `0.5.12`.  The
  lingering `0.5.9` (vision_mvp) vs `0.5.11` (pyproject) misalignment
  from earlier milestones is now closed.
* W38 surface exported under `__experimental__`; stable runtime
  contract byte-for-byte unchanged.

### Documentation

* `docs/RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md` (new).
* `docs/SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md` (new).
* `docs/RESEARCH_STATUS.md` — TL;DR updated to SDK v3.39.
* `docs/THEOREM_REGISTRY.md` — W38 theorems and limitation theorem
  added.
* `docs/HOW_NOT_TO_OVERSTATE.md` — W38 do-not-overstate rules added.
* `docs/context_zero_master_plan.md` — milestone marker updated.
* `papers/context_as_objects.md` — milestone marker updated.
* `README.md` and `docs/START_HERE.md` — current-milestone summary
  updated.

---

## [0.5.11 / 3.38] — 2026-05-02 — SDK v3.38 — anchor-cross-host basis-trajectory ratification + manifest-v7 CID + R-84 Phase-84 benchmark family + bounded live cross-host trajectory probe

*Strictly additive on SDK v3.37.  The stable CoordPy product/runtime
contract is unchanged.  The W37 surface is experimental and wraps the
W36 `HostDiverseTrustSubspaceOrchestrator` with a closed-form
per-(host, oracle, top_set) EWMA over anchored historical
observations.  W37 converts a W36 host-diversity abstention into a
single-host trajectory-trusted reroute iff the supporting host has a
cross-host anchored trajectory above threshold across at least
``min_anchored_observations`` historical cells with at least
``min_trajectory_anchored_hosts`` distinct anchor hosts.  Without an
anchored trajectory, W37 preserves W36 behavior byte-for-byte.  It is
explicitly not native latent transfer, not transformer-internal
hidden-state projection, and not a KV-cache transplant.*

**New experimental surface.**

`CrossHostBasisTrajectoryEntry`,
`CrossHostBasisTrajectoryRatificationEnvelope`,
`CrossHostBasisTrajectoryRegistry`, `W37CrossHostTrajectoryResult`,
`CrossHostBasisTrajectoryOrchestrator`,
`select_cross_host_trajectory_projection`,
`verify_cross_host_trajectory_ratification`,
`build_trivial_cross_host_trajectory_registry`,
`build_cross_host_trajectory_registry`, plus W37 schema/branch/default
constants.

**Headline empirical results.**

* **R-84-SINGLE-HOST-TRAJECTORY-RECOVER.**  W37 improves over W36 from
  0.500 to **1.000** correctness (`Δ_correctness_w37_w36 = +0.500`,
  min and max equal across 5/5 seeds) at trust precision **1.000**.

* **R-84-TRIVIAL-W37.**  Trajectory disabled + single-host reroute
  disabled + manifest-v7 disabled reduces W37 to W36 byte-for-byte
  across 5/5 seeds.

* **R-84-NO-TRAJECTORY-HISTORY.**  Falsifier: empty trajectory ⇒
  W37 = W36 abstention (16/16 cells/seed).

* **R-84-POISONED-TRAJECTORY.**  Falsifier: single-host trajectory
  fails ``min_trajectory_anchored_hosts`` ⇒ W37 does not reroute.

* **R-84-TRAJECTORY-DISAGREEMENT.**  Falsifier: current basis
  disagrees with anchored trajectory ⇒ W37 does not reroute.

**Trust boundary.**  14 new W37 failure modes in
`verify_cross_host_trajectory_ratification`, all covered by W37 unit
tests; cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 +
W36 + W37 trust boundary = **126 enumerated failure modes**.

**Live fallback.**  Fresh preflight 9/10 hosts OK,
`192.168.12.248` ARP-incomplete for the 30th milestone.  Bounded
W37 cross-host trajectory probe on 2026-05-02 across local
`gemma2:9b` and remote `qwen2.5:14b` produced **8/8 responsive
probes, 8/8 cross-host anchored agreements, and 8/8 gold-correlated
agreements** at temperature 0.

**Tests.**  26/26 W37 unit tests pass; 563/563 phase69-84 focused
regression slice green; full ``pytest vision_mvp/tests`` runs to
completion during the milestone.

## [0.5.10 / 3.37] — 2026-05-02 — SDK v3.37 — host-diverse trust-subspace guard + manifest-v6 CID + R-83 Phase-83 benchmark family + live two-reachable-host probe

*Strictly additive on SDK v3.36.  The stable CoordPy product/runtime
contract is unchanged.  The W36 surface is experimental and wraps the
W35 `TrustSubspaceDenseControlOrchestrator` with a host-diverse
trust-subspace guard.  W36 requires dense-control projection support
to be independently attested by distinct registered healthy hosts;
unsafe, unverifiable, spoofed, or no-live-attestation branches reject
or abstain.  It is explicitly not native latent transfer, not
transformer-internal hidden-state projection, and not a KV-cache
transplant.*

**New experimental surface.**

`HostDiverseBasisEntry`, `HostDiverseRatificationEnvelope`,
`HostDiverseRegistry`, `W36HostDiverseResult`,
`HostDiverseTrustSubspaceOrchestrator`,
`select_host_diverse_projection`,
`verify_host_diverse_ratification`,
`build_trivial_host_diverse_registry`,
`build_host_diverse_registry`, plus W36 schema/branch/default
constants.

**Headline empirical results.**

* **R-83-HOST-DIVERSE-RECOVER.**  W36 improves over W35 from 0.625 to
  **0.9375** correctness (**+0.3125**) across 5/5 seeds and restores
  trust precision from 0.6667 to **1.000**; overhead is one visible
  token/cell.

* **R-83-HOST-SPOOFED-CONSENSUS.**  W36 does not improve correctness
  (0.625 remains 0.625), but trust precision rises from 0.625 to
  **1.000** by abstaining on 6 unsafe W35 ratifications per seed.

* **R-83-TRIVIAL-W36.**  W36 disabled + manifest-v6 disabled reduces
  to W35 byte-for-byte across 5/5 seeds.

* **R-83-NO-LIVE-ATTESTATION.**  Hard falsifier: W36 without live
  attestations abstains on every cell and drops correctness from
  W35's 1.000 to 0.000.

**Trust boundary.**  14 new W36 failure modes in
`verify_host_diverse_ratification`, all covered by W36 unit tests;
cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 + W36 trust
boundary = **112 enumerated failure modes**.

**Live fallback.**  Fresh preflight found local Ollama and
`192.168.12.191:11434` reachable, with `192.168.12.248:11434`
timing out.  Bounded two-reachable-host probe on 2026-05-02 across
local `qwen2.5:0.5b` and remote `qwen2.5:14b` produced 10/10
responsive probes, 4/5 cross-host disagreements, and 4/4
gold-correlated disagreement winners.

**Versioning.**  SDK_VERSION `coordpy.sdk.v3.37`; package version
`0.5.10`; W36 exports live under `__experimental__`.

## [0.5.9 / 3.36] — 2026-05-02 — SDK v3.36 — trust-subspace dense-control proxy + basis-history projection + W35 manifest-v5 CID + R-82 Phase-82 benchmark family + W34 abstention converted to verified reroute where basis history is stable

*Strictly additive on SDK v3.35.  The stable CoordPy product/runtime
contract is unchanged.  The W35 surface is experimental and wraps the
W34 `LiveAwareMultiAnchorOrchestrator` with a controller-verified
trust-subspace dense-control proxy.  W35 derives one basis entry per
oracle from W21 probe top_sets, W33 EWMA trust, W34
live-attestation/response-feature state, top-set stability, and host
health, then uses that basis only when it can safely convert W34
NO_CONSENSUS abstention into a verified reroute.  It is explicitly not
native latent transfer, not transformer-internal hidden-state
projection, and not a KV-cache transplant.*

**New experimental surface.**

`TrustSubspaceBasisEntry`, `TrustSubspaceDenseRatificationEnvelope`,
`TrustSubspaceDenseRegistry`, `W35TrustSubspaceResult`,
`TrustSubspaceDenseControlOrchestrator`,
`select_trust_subspace_projection`,
`verify_trust_subspace_dense_ratification`,
`build_trivial_trust_subspace_registry`,
`build_trust_subspace_dense_registry`, plus W35 schema/branch/default
constants.

**Headline empirical results.**

* **R-82-TRUST-SUBSPACE-SHIFT.**  W34 abstains on 6 disputed cells;
  W35 reroutes 5/6 through the stable `change_history` basis
  direction.  Correctness rises from 0.625 to **0.9375**
  (**+0.3125**) across 5/5 seeds; trust precision remains **1.000**;
  overhead is one visible token/cell.

* **R-82-TRIVIAL-W35.**  W35 disabled + manifest-v5 disabled reduces
  to W34 byte-for-byte across 5/5 seeds.

* **R-82-ALL-ANCHOR-COMPROMISED.**  Hard falsifier: when every basis
  direction moves wrong together, W35 cannot recover; W35-W34 delta =
  0.000 and trust precision remains 0.625.

**Trust boundary.**  14 new W35 failure modes in
`verify_trust_subspace_dense_ratification`, all covered by W35 unit
tests; cumulative W22 + W29 + W30 + W31 + W32 + W33 + W34 + W35 trust
boundary = **98 enumerated failure modes**.

**Live fallback.**  Full W34 xLLM pilot exceeded practical turn-time
budget and was stopped.  Bounded two-host live probe on 2026-05-02
across local `qwen2.5:0.5b` and remote `qwen2.5:14b` produced 10/10
responsive probes, 3/5 cross-host disagreements, and 3/3
gold-correlated disagreement winners.  `192.168.12.248` still timed
out on `/api/tags`; true broader multi-host evidence remains open.

**Versioning.**  SDK_VERSION `coordpy.sdk.v3.36`; package version
`0.5.9`; W35 exports live under `__experimental__`.

## [0.5.8 / 3.35] — 2026-05-01 — SDK v3.35 — live-aware multi-anchor adjudication + native-latent audited response-feature proxy + W34 manifest-v4 CID + W33 infra-blocker closure (preflight `/api/tags` + chat-template + `num_predict=4`) + R-81 Phase-81 benchmark family + W34 family + W33 single-anchor fragility closed via multi-anchor consensus + W33-INFRA-1 + W33-INFRA-2 jointly discharged

*Strictly additive on SDK v3.34.  The CoordPy single-run product
runtime contract is byte-for-byte unchanged.  The W34 surface wraps
the **W33 ``TrustEWMATrackedMultiOracleOrchestrator``** (the
EWMA-tracked multi-oracle line) with six NEW audited proxies at
the capsule layer: a **multi-anchor consensus reference**
(intersection of K registered anchors' top_sets when at least
``anchor_quorum_min`` non-abstaining anchors agree), a **NO_CONSENSUS
abstention branch** (when anchors disagree, W34 drops services
from the answer — anchor disagreement is itself a trust signal), a
**closed-form 64-bit response-feature signature** (SHA-256 prefix
over first_token_class + length_bucket + structural_hash; the W34
audited proxy for native-latent), a **content-addressed
LiveOracleAttestation** (host_id, model_id,
response_feature_signature, latency_ms_bucket, preflight_ok), a
**closed-form host-aware EWMA decay** (multiplicative
host_decay_factor in [0.5, 1.0] applied to oracles whose host is
unhealthy), and a **manifest-v4 CID** over four component CIDs
(parent_w33_cid, live_attestation_cid, multi_anchor_cid,
host_topology_cid) that detects cross-component swaps the W33
manifest-v3 alone cannot catch.

The W34 milestone also closes two named W33 infrastructure
follow-ups load-bearing in the live pilot:

* **W33-INFRA-1 closure** — closed-form preflight ``/api/tags``
  check.  *Honest empirical correction*: the W33 milestone diagnosed
  qwen3.5:35b on 192.168.12.191 as "model not loaded" but a fresh
  ``/api/tags`` curl on 2026-05-01 confirms the model IS loaded.
  The real W33 infra failure was timeout-budget exhaustion +
  chat-template mismatch, NOT model absence.

* **W33-INFRA-2 closure** — chat-template (``/api/chat`` with
  system+user messages) + ``num_predict=4`` + stop tokens.  Stops
  chain-of-thought emit at temperature 0 for mixtral:8x7b within
  the first 4 tokens.  Adaptive timeout per host: small models 30 s,
  medium 60 s, large (>= 30B) 240 s.

The new "live-aware multi-anchor / response-feature signature /
live oracle attestation / host-aware EWMA decay / manifest-v4 CID
/ preflight discipline" vocabulary is added at the **capsule layer
as audited proxy** — explicitly NOT a learned feature-signature
model in the deep-learning sense, NOT transformer-internal subspace
projection, NOT a runtime hidden-state transplant.*

**New surface (W34 family, multi-agent-coordination research slice).**

``derive_multi_anchor_consensus_reference``,
``compute_response_feature_signature``, ``apply_host_decay``,
``LiveOracleAttestation``,
``LiveAwareMultiAnchorRatificationEnvelope``,
``LiveAwareMultiAnchorRegistry`` (with
``multi_anchor_quorum_min: int``,
``live_attestation_disabled: bool``,
``manifest_v4_disabled: bool``,
``host_decay_factor: float``,
``registered_hosts: dict[str, HostRegistration]``),
``HostRegistration``,
``W34LiveAwareResult``,
``LiveAwareMultiAnchorOrchestrator``,
``verify_live_aware_multi_anchor_ratification``,
``build_trivial_live_aware_registry``,
``build_live_aware_registry``,
plus W34_*_SCHEMA_VERSION / W34_DEFAULT_* constants and
W34_BRANCH_* labels.

**Headline empirical results (W34 family).**

* **R-81-DOUBLE-ANCHOR-COMPROMISE (H6).**  Three-phase oracle
  compromise where the W33 single-anchor (service_graph) itself
  becomes compromised in the final phase.  W33: trust precision
  0.625 (commits to wrong decoy when sg flips).  W34 with K=2
  (sg+ch) anchors + ``anchor_quorum_min=2``: intersection of
  anchor top_sets is empty in the double-compromise phase ⇒
  NO_CONSENSUS branch fires ⇒ W34 abstains.  **Δ_trust_precision
  = +0.375** across **5/5 seeds at trust_precision_w34 = 1.000**;
  no correctness regression; max overhead = 1 token/cell.

* **R-81-MANIFEST-V4-TAMPER (H3).**  400/400 = 1.000 reject rate
  across 5/5 seeds × 16 cells × 5 named tampers per cell.

* **R-81-RESPONSE-FEATURE-SIGNATURE (H8).**  10 fixtures × 3 runs
  = 30/30 byte-equal calls.  Signature length = 16 hex chars (64
  bits).

* **R-81-TRIVIAL-W34 (H2).**  Byte-for-W33 equivalent across
  5/5 seeds.

* **R-81-NO-ANCHOR-DISAGREEMENT (W34-Λ-no-anchor-disagreement).**
  Δ = 0.000 (all anchors agree ⇒ W34 = W33).

* **R-81-FROZEN-HOST-DECAY (W34-Λ-frozen-host-decay).**
  Δ = 0.000 (host_decay_factor=1.0 ⇒ no decay).

* **W33-INFRA-1 + W33-INFRA-2 closure** in the live xLLM pilot.

**Five named falsifiers, all empirically observed:**
W34-Λ-trivial-multi-anchor; W34-Λ-no-anchor-disagreement;
W34-Λ-anchor-betrays; W34-Λ-frozen-host-decay;
W34-Λ-mis-feature-signature.

**Trust boundary: 14 enumerated W34 failure modes**, disjoint from
W22..W33's 70 ⇒ **84 cumulative across W22+W29+W30+W31+W32+W33+W34**.

**Limitation theorem proved**: **W34-L-MULTI-ANCHOR-CAP** — when
all K anchors are simultaneously compromised at the capsule layer,
no multi-anchor mechanism (including W34) can recover.  Native-
latent (architecture-dependent) is required to break this.

**Conjectures inheriting forward**:
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE (open),
W33-C-NATIVE-LATENT (open),
W33-C-MULTI-HOST (open; Mac 2 ARP-incomplete 29th milestone),
W33-C-LATENT-CROSS-AGENT-TRUST (open),
W34-C-CROSS-HOST-LIVE-MULTI-ANCHOR (new; live anchor compromise
detection on real cross-host topology),
W34-C-MULTI-HOST (new; ≥ 3 reachable hosts).

**Tests + regression**: 48/48 W34 unit tests + 494/494 phase69-81
regression + 211/211 wider coordpy suite = **753 tests pass**.
Backward-compat byte-for-byte preserved on the trivial-W34 path
(W34-Λ-trivial-multi-anchor falsifier).

**Stable-vs-experimental boundary**: ``__experimental__`` extended
with W34 symbols; ``SDK_VERSION = coordpy.sdk.v3.35``; pyproject
``0.5.8``.  Stable runtime contract byte-for-byte unchanged from
v3.34.

See ``docs/RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md`` for the
full milestone note + ``docs/SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md``
for the pre-committed bar.

## [0.5.7 / 3.34] — 2026-05-01 — SDK v3.34 — trust-EWMA-tracked multi-oracle adjudication + per-oracle agreement signal + anchor-oracle reference + content-addressed oracle-trust-state + trust-trajectory CID + W33 manifest-v3 CID + single-partition long-window strict-gain regime + R-80 Phase-80 benchmark family + W33 family + W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST + W32-C-LONG-WINDOW-STRICT-GAIN jointly discharged at a single milestone

*Strictly additive on SDK v3.33.  The CoordPy single-run product
runtime contract is byte-for-byte unchanged.  The W33 surface wraps
the **OLD W21 ``TrustWeightedMultiOracleDisambiguator``** (the
explicit multi-oracle adjudication line, dormant since W21) with
six NEW audited proxies at the capsule layer: a **closed-form
per-oracle EWMA trust accumulator** (``ewma_new = (1 - α) *
ewma_prev + α * obs`` with α=W32_DEFAULT_EWMA_ALPHA=0.20), a
**per-oracle agreement signal** that compares each oracle's probe
top_set to a registered **anchor-oracle reference** (a
trust-by-construction reference immune to quorum-flip), a
**closed-form trust-threshold gate** (default 0.50; oracles with
EWMA below threshold are excluded from the effective tally), a
**content-addressed oracle-trust-state CID** + **content-addressed
trust-trajectory CID** + **content-addressed anchor-oracle-set
CID**, and a **manifest-v3 CID** over six component CIDs that
detects cross-component swaps that per-component CIDs alone miss.
The new "trust-EWMA-tracked / per-oracle agreement signal /
anchor-oracle reference / oracle-trust-state CID / trust-trajectory
CID / manifest-v3 CID / single-partition strict-gain bench"
vocabulary is added at the **capsule layer as audited proxy** —
explicitly NOT a learned trust model in the deep-learning sense,
NOT transformer-internal subspace projection, NOT a runtime
hidden-state transplant; the per-oracle agreement signal is a
deterministic top-set comparison against a registered closed-vocab
anchor, NOT a runtime oracle.*

**New surface (W33 family, multi-agent-coordination research slice).**

``derive_per_oracle_agreement_signal``,
``TrustTrajectoryEntry``, ``TrustEWMARatificationEnvelope``,
``TrustEWMARegistry`` (with ``anchor_oracle_ids: frozenset[str]``),
``W33TrustEWMAResult``,
``TrustEWMATrackedMultiOracleOrchestrator``,
``verify_trust_ewma_ratification`` (14 enumerated failure modes —
disjoint from W22 + W29 + W30 + W31 + W32's 14-mode sets;
cumulative 70-mode trust boundary across W22 + W29 + W30 + W31 +
W32 + W33), ``build_trivial_trust_ewma_registry``,
``build_trust_ewma_registry``, W33 branch constants
(``W33_BRANCH_TRUST_EWMA_RESOLVED``,
``W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH``,
``W33_BRANCH_TRUST_EWMA_REJECTED``,
``W33_BRANCH_TRUST_EWMA_DISABLED``,
``W33_BRANCH_TRUST_EWMA_NO_TRIGGER``,
``W33_BRANCH_TRUST_EWMA_DETRUSTED_ABSTAIN``,
``W33_BRANCH_TRUST_EWMA_DETRUSTED_REROUTE``),
``W33_TRUST_EWMA_SCHEMA_VERSION``,
``W33_DEFAULT_TRUST_THRESHOLD = 0.50``,
``W33_DEFAULT_TRUST_TRAJECTORY_WINDOW = 16``,
``W33_DEFAULT_EWMA_ALPHA = 0.20``.

**New benchmark family (R-80 + R-79 single-partition).**

``vision_mvp.experiments.phase80_trust_ewma_tracked`` ships six
sub-banks: R-80-TRIVIAL-W33 (H2 byte-for-W21 anchor),
R-80-COMPROMISED-SHIFT (H6 strict-gain on a three-phase oracle-
compromise regime: K1=3N/8 calibration / K2=5N/8 single compromise
/ K3=N double compromise; W33 detrusts ch + oc and falls back to
the anchor sg ⇒ +0.375 trust precision strict gain), R-80-NO-
TRUST-SHIFT (W33-Λ-no-trust-shift falsifier), R-80-FROZEN-TRUST-
THRESHOLD (W33-Λ-frozen-threshold falsifier), R-80-MIS-TRUST-SHIFT
(W33-Λ-mis-trust-shift falsifier — honest empirical correction:
the anchor-oracle design is more robust than predicted),
R-80-MANIFEST-V3-TAMPER (H8 cross-component swap detection at
400/400 = 1.000 reject rate).
``vision_mvp.experiments.phase79_long_window_convergent`` adds the
new R-79-SINGLE-PARTITION sub-bank (a prefix-then-shift regime
over a single-partition signature space that exceeds the
W32-L-CYCLE-CAP cycle-capped Δ_max ≤ 0.0625 bound by
construction).  Standalone live cross-architecture trust-
calibration probe at
``vision_mvp/experiments/scripts/phase80_xllm_trust_pilot.py``.

**Headline measurements (5/5 seeds — byte-for-byte stable).**

* R-80-COMPROMISED-SHIFT: ``W33 trust precision = 1.000 vs W21 =
  0.625 ⇒ Δ = +0.375`` across 5/5 seeds × 16 cells/seed,
  correctness tied at 0.625, max overhead 1 token/cell.
  **Jointly discharges W21-C-CALIBRATED-TRUST AND W32-C-OLD-LINE-
  EWMA-TRUST** in a single milestone.
* R-79-SINGLE-PARTITION: ``Δ(W32 - W31) = +0.100`` across 5/5
  seeds × 80 cells.  **Discharges W32-C-LONG-WINDOW-STRICT-GAIN**
  on a regime that exceeds the W32-L-CYCLE-CAP cycle-capped bound.
* R-80-MANIFEST-V3-TAMPER: ``400/400 = 1.000 cross-component
  tamper rejection`` across 5/5 seeds × 16 ratified cell-positions
  × 5 named tampers per cell.
* R-80-TRIVIAL-W33: ``W33 = W21 byte-for-byte across 5/5 seeds``
  on the trivial-passthrough config.

**Live cross-architecture LLM trust-calibration probe (S1 best-
effort, honestly null on infrastructure).**

mixtral:8x7b on localhost + qwen3.5:35b on 192.168.12.191 across
20 trust-calibration prompts at temperature 0.  qwen3.5:35b
returned empty/timeout on every prompt (model not actually loaded
on the remote host); mixtral:8x7b ignored the "EXACTLY one word"
constraint and emitted full chain-of-thought past the
``num_predict=60`` budget.  Two named infrastructure-fix items
recorded (W33-INFRA-1: pre-flight ``/api/tags`` model verification;
W33-INFRA-2: stricter token-budget / chat-template for one-word
probes).  The W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE conjecture
**remains open**; the W33 mechanism's discharge claims do not
depend on this live probe.

**Falsifiers (all empirically observed at temp 0 on synthetic).**

* W33-Λ-trivial-trust-ewma ⇒ byte-for-W21 passthrough on
  R-80-TRIVIAL-W33;
* W33-Λ-no-trust-shift ⇒ all EWMA stay at 1.0 on
  R-80-NO-TRUST-SHIFT (no compromise event);
* W33-Λ-frozen-threshold ⇒ gate never fires when the threshold is
  pinned at 1.0 on R-80-FROZEN-TRUST-THRESHOLD;
* W33-Λ-mis-trust-shift ⇒ honest empirical correction: the
  anchor-oracle design is more robust than the falsifier
  predicted (anchor-relative agreement signal is immune to
  quorum-flip).

**Theory.**

* **W33-1** through **W33-5** named theorems (envelope determinism,
  manifest-v3 cross-component soundness, joint discharge of
  W21-C-CALIBRATED-TRUST + W32-C-OLD-LINE-EWMA-TRUST,
  W32-C-LONG-WINDOW-STRICT-GAIN single-partition discharge, and
  the cumulative 70-mode trust boundary across
  W22 + W29 + W30 + W31 + W32 + W33).
* **Three named open conjectures discharged**:
  W21-C-CALIBRATED-TRUST, W32-C-OLD-LINE-EWMA-TRUST,
  W32-C-LONG-WINDOW-STRICT-GAIN.
* **Open conjectures carried forward**:
  W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE (live infrastructure-
  bounded), W33-C-NATIVE-LATENT (architecture-dependent —
  transformer-internal trust subspace projection vs the W33
  audited proxy; out of capsule-layer scope), W33-C-MULTI-HOST
  (3+ host topology, hardware-bounded), W33-C-LATENT-CROSS-AGENT-
  TRUST (latent-state trust-EWMA tracking — not capsule-layer).

**Files changed.**

* New: `vision_mvp/coordpy/team_coord.py` (W33 surface ~600 lines
  appended).
* Modified: `vision_mvp/coordpy/__init__.py` (W33 imports;
  ``SDK_VERSION = "coordpy.sdk.v3.34"``;
  ``__experimental__`` = 98 entries; ``__all__`` = 440 entries).
* New: `vision_mvp/experiments/phase80_trust_ewma_tracked.py`.
* New: `vision_mvp/experiments/scripts/phase80_xllm_trust_pilot.py`.
* New: `vision_mvp/tests/test_phase80_trust_ewma_tracked.py`
  (31 tests).
* Modified: `vision_mvp/experiments/phase79_long_window_convergent.py`
  (new ``single_partition`` sub-bank).
* Modified: `vision_mvp/tests/test_phase79_long_window_convergent.py`
  (new ``test_h7b_single_partition_strict_gain``).
* New: `docs/SUCCESS_CRITERION_W33_TRUST_EWMA_TRACKED.md`.
* New: `docs/RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md`.
* Modified: `docs/THEOREM_REGISTRY.md`,
  `docs/RESEARCH_STATUS.md`, `README.md`, `CHANGELOG.md`,
  `pyproject.toml` (0.5.7).

**Tests.**

31/31 W33 unit tests + 46/46 phase79 (45 prior + 1 new
single_partition) + 446/446 phase69-80 regression + 133/133 wider
coordpy suite pass.

## [0.5.6 / 3.33] — 2026-05-01 — SDK v3.33 — long-window convergent online geometry-aware dense control + EWMA prior accumulator + Page CUSUM change-point detector + gold-correlated disagreement-routing + W32 manifest-v2 CID + first measured live cross-architecture LLM gold-verifiable agreement at temperature 0 + W31-C-LONG-WINDOW-CONVERGENCE discharged on the scaling-stability axis + W32-L-CYCLE-CAP limitation theorem + W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on the prompt-class-dependent agreement frontier

*Strictly additive on SDK v3.32. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W32 surface wraps
the W31 ``OnlineCalibratedOrchestrator`` with four NEW audited
proxies at the capsule layer: an EWMA prior accumulator (closed-form
``ewma_new = (1-α) * ewma_prev + α * obs``), a Page two-sided CUSUM
change-point detector (closed-form ``cusum_pos / cusum_neg``
accumulators bounded by registered ``cusum_max``), a gold-
correlated disagreement-routing primitive against a registered
closed-vocabulary ``GoldCorrelationMap``, and a sealed convergence-
state trajectory CID + cross-component manifest-v2 CID. The new
"long-window convergent / EWMA / Page CUSUM / gold-correlation
lookup / manifest-v2 CID" vocabulary is added at the **capsule
layer as audited proxy** — explicitly NOT a learned model in the
deep-learning sense, NOT transformer-internal subspace projection,
NOT a runtime KV transplant; the gold-correlation map is a
registered closed-vocabulary table, NOT a runtime ground-truth
observation. W32 also records the **first measured live cross-
architecture LLM gold-verifiable agreement at temperature 0**
in 29 milestones (gemma2:9b vs qwen2.5:14b on 19/20 = 0.950 of
gold-verifiable structured-decision prompts) — the **honest
converse** of W31's R-78-XLLM-LIVE result (mostly-disagreement on
operational-decision prompts) — characterising the prompt-class-
dependent cross-architecture disagreement frontier.*

**New surface (W32 family, multi-agent-coordination research slice).**

``GoldCorrelationMap``, ``build_gold_correlation_map``,
``ConvergenceStateEntry``,
``LongWindowConvergentRatificationEnvelope``,
``LongWindowConvergentRegistry``, ``W32LongWindowResult``,
``LongWindowConvergentOrchestrator``,
``verify_long_window_convergent_ratification`` (14 enumerated failure
modes — disjoint from W29/W30/W31's 14-mode sets; cumulative
56-mode trust boundary across W29 + W30 + W31 + W32),
``update_ewma_prior``, ``update_cusum_two_sided``,
``detect_change_point``,
``build_trivial_long_window_registry``,
``build_long_window_convergent_registry``,
W32 branch constants (``W32_BRANCH_*``),
``W32_LONG_WINDOW_SCHEMA_VERSION``,
``W32_DEFAULT_EWMA_ALPHA = 0.20``,
``W32_DEFAULT_CUSUM_THRESHOLD = 1.5``,
``W32_DEFAULT_CUSUM_K = 0.10``,
``W32_DEFAULT_CUSUM_MAX = 16.0``,
``W32_DEFAULT_LONG_WINDOW = 64``,
``W32_DEFAULT_GOLD_CORRELATION_MIN = 0.50``.

**New benchmark family (R-79).**

``vision_mvp.experiments.phase79_long_window_convergent`` ships eight
sub-banks: R-79-TRIVIAL-W32 (H2 byte-for-W31 anchor), R-79-DRIFT-
RECOVER (H6 partial-honest-null per W32-L-CYCLE-CAP limitation
theorem), R-79-LONG-WINDOW (H7 scaling sweep at long_window ∈
{16, 32, 64, 128}), R-79-MANIFEST-V2-TAMPER (H8 cross-component
swap detection at 1525/1525 = 1.000 reject rate),
R-79-NO-CHANGE-POINT (W32-Λ-no-change-point), R-79-FROZEN-EWMA
(W32-Λ-frozen-ewma honest empirical correction), R-79-MIS-
CORRELATED-GOLD (W32-Λ-mis-correlated-gold gate-bounded),
R-79-XLLM-LIVE-GOLD (S1/S2 best-effort live cross-architecture
probe on gold-verifiable prompts).  Standalone live cross-
architecture gold-verifiable probe at
``vision_mvp/experiments/scripts/phase79_xllm_gold_pilot.py``.

**Headline measurements (5/5 seeds — byte-for-byte stable).**

* R-79-LONG-WINDOW: ``W32 ≥ W31 byte-for-byte across 5/5 seeds ×
  4/4 windows = 20/20 cell-window pairs``; trust precision = 1.000;
  zero degradation as long_window grows from 16 to 128.
  **Discharges W31-C-LONG-WINDOW-CONVERGENCE on the scaling-
  stability axis**.
* R-79-MANIFEST-V2-TAMPER: ``1525/1525 = 1.000 cross-component
  tamper rejection rate`` across 5/5 seeds × 61 ratified cell-
  positions × 5 named tampers per cell.
* R-79-XLLM-LIVE-GOLD: ``19/20 = 0.950 agreement on gold-
  verifiable prompts at temperature 0`` (gemma2:9b localhost vs
  qwen2.5:14b on 192.168.12.191).  **First measured live cross-
  architecture LLM gold-verifiable agreement at temp 0 in the
  programme**.
* R-79-DRIFT-RECOVER: ``Δ(W32 - W31) = 0.000`` (honest-null per the
  **W32-L-CYCLE-CAP limitation theorem**: the strict-gain bar
  Δ ≥ +0.10 is bounded above by ``min(c_p/4, c_s)/N`` on cycle-
  capped dispatcher regimes; mechanism empirically validated by
  ``n_change_points = 1`` firing exactly at the shift boundary).

**New theorem-style claims (W32-1..W32-5; W32-L-CYCLE-CAP)**.
W32-1 (proved + mechanically-checked) trust-boundary soundness;
W32-2 (proved + empirical) trivial-long-window byte-for-W31
reduction; W32-3 (proved-conditional + empirical) long-window
convergent scaling-stability discharging W31-C-LONG-WINDOW-
CONVERGENCE; W32-4 (proved + empirical) Page CUSUM change-point
detection; W32-5 (proved-conditional + empirical) manifest-v2
cross-component tamper detection; W32-L-CYCLE-CAP (proved-
conditional limitation theorem) bounding strict gain on cycle-
capped dispatcher regimes.  Four named falsifiers (all empirically
observed): W32-Λ-trivial-long-window (byte-for-W31 reductive),
W32-Λ-no-change-point (stationary regime ⇒ no help, 0 change-
points), W32-Λ-frozen-ewma (honest empirical correction: did NOT
regress at α=1.0; the available regime is non-noisy), W32-Λ-mis-
correlated-gold (gate-bounded on synthetic; gate never opens
without real cross-host disagreement).

**Open conjectures inheriting forward.**
W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE (renamed from
W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE on the gold-
correlation axis; honestly null on available LLMs at temp 0 on
gold-verifiable prompts).
W32-C-LONG-WINDOW-STRICT-GAIN (renamed from H6 partial; requires a
regime that exceeds the W32-L-CYCLE-CAP limitation).
W32-C-NATIVE-LATENT (architecture-dependent; the next true wall).
W32-C-MULTI-HOST (hardware-bounded; Mac 2 ARP-incomplete for **27
consecutive milestones**).
W32-C-OLD-LINE-EWMA-TRUST (sharpens W21-C-CALIBRATED-TRUST on the
EWMA-tracked online axis; primitives now available for integration).

**Tests + regression**: 45/45 W32 unit tests + 414/414 phase69-79
regression (was 437 in v3.32; the new W32 stack passes byte-for-byte
on phase69-78 + adds 45 new W32 tests) + 77/77 wider coordpy suite
= **536 tests pass** across the W22..W32 stack + capsule + public
API + runtime + LLM backend + provenance.

---

## [0.5.5 / 3.32] — 2026-05-01 — SDK v3.32 — online self-calibrated geometry-aware dense control + sealed prior trajectory + adaptive threshold + W31 manifest CID + first measured live cross-architecture LLM disagreement at temperature 0 + W30-C-PRIOR-LEARNING discharged + W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on the infrastructure axis

*Strictly additive on SDK v3.31. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W31 surface wraps
the W30 ``CalibratedGeometryOrchestrator`` with three NEW audited
proxies at the capsule layer: an online closed-loop running-mean
update for the per-partition calibration prior, a closed-form
clipped-median adaptive threshold, and a sealed prior + threshold
trajectory CID + cross-component manifest CID.  The new "online
running-mean / adaptive threshold / sealed trajectory / manifest
CID" vocabulary is added at the **capsule layer as audited proxy**
— explicitly NOT a learned model in the deep-learning sense, NOT
transformer-internal subspace projection, NOT a runtime KV
transplant.  W31 also records the **first measured live cross-
architecture LLM disagreement at temperature 0** in 28 milestones
(gemma2:9b vs qwen2.5:14b on 2/8 = 0.250 of structured-decision
prompts, reproducible byte-for-byte).*

**New surface (W31 family, multi-agent-coordination research slice).**

``PriorTrajectoryEntry``,
``OnlineCalibratedRatificationEnvelope``,
``OnlineCalibratedRegistry``, ``W31OnlineResult``,
``OnlineCalibratedOrchestrator``,
``verify_online_calibrated_ratification`` (14 enumerated failure
modes — disjoint from W29's 14 and W30's 14; cumulative 42-mode
trust boundary across W29 + W30 + W31),
``derive_per_cell_agreement_signal``,
``compute_adaptive_threshold``,
``build_trivial_online_registry``,
``build_online_calibrated_registry``,
W31 branch constants (``W31_BRANCH_*``),
``W31_ONLINE_SCHEMA_VERSION``,
``W31_DEFAULT_THRESHOLD_MIN = 0.20``,
``W31_DEFAULT_THRESHOLD_MAX = 0.80``,
``W31_DEFAULT_TRAJECTORY_WINDOW = 16``.

**New benchmark family (R-78).**

``vision_mvp.experiments.phase78_online_calibrated_dense_control``
ships seven sub-banks: R-78-TRIVIAL-W31 (H2 byte-for-W30 anchor),
R-78-NONSTATIONARY-PRIOR (H6 main load-bearing claim — discharges
W30-C-PRIOR-LEARNING), R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-
THRESHOLD (H7 isolating axis), R-78-NO-DRIFT (W31-Λ-no-drift
falsifier), R-78-MANIFEST-TAMPER (H8 cross-component swap detection),
R-78-XLLM-LIVE (S1/S2 best-effort live cross-architecture probe
running gemma2:9b on localhost + qwen2.5:14b on 192.168.12.191).
Standalone live cross-architecture probe at
``vision_mvp/experiments/scripts/phase78_xllm_live_probe.py``.

**Headline measurements (5/5 seeds — byte-for-byte stable).**

* R-78-NONSTATIONARY-PRIOR: ``Δ(W31 − W30) = +0.125`` correctness
  gain across 5/5 seeds at trust precision 1.000; W30 baseline
  0.750 → W31 online 0.875.  Discharges
  **W30-C-PRIOR-LEARNING** on the magnitude axis.
* R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD: adaptive Δ =
  +0.125, frozen Δ = 0.000; difference = +0.125 ≥ +0.05 across 5/5
  seeds.  Isolates the adaptive-threshold contribution.
* R-78-MANIFEST-TAMPER: 65/65 = 1.000 tamper rejection rate across
  five named tampers per cell-position (cross-cell trajectory swap +
  manifest CID corruption + observed_agreement out of range +
  threshold value out of range + outer w31_cid corruption).
* R-78-TRIVIAL-W31: byte-for-W30 invariant on 5/5 seeds.
* R-78-NO-DRIFT: stationary regime ⇒ Δ = 0.000; W31-Λ-no-drift
  confirmed.
* R-78-XLLM-LIVE: gemma2:9b (localhost) vs qwen2.5:14b
  (192.168.12.191) systematically disagree on 2/8 = 0.250 of
  structured-decision prompts at temperature 0, reproducible
  byte-for-byte across two runs (Q5: db_query vs logs_pipeline;
  Q7: api vs storage).  **First measured live cross-architecture
  LLM disagreement at temp 0 in the programme** (28th milestone).
* Mean overhead w31/w30 ≤ 1.0 token/cell across all R-78 sub-banks;
  max overhead = 1; mean cumulative overhead w31/w28 ≤ 3.0.

**Theorem-style claims added to the registry.**

W31-1 (proved + mechanically-checked), W31-2 (proved + empirical),
W31-3 (proved-conditional + empirical — W30-C-PRIOR-LEARNING
discharge), W31-4 (proved-conditional + empirical — adaptive vs
frozen), W31-5 (proved-conditional + empirical — manifest tamper
detection).  Three named falsifiers (W31-Λ-trivial-online,
W31-Λ-no-drift, W31-Λ-frozen-threshold) all empirically confirmed.
Three new W31-named open conjectures inherit forward to v3.33:
W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE (gold-correlation
axis), W31-C-NATIVE-LATENT (architecture-dependent next true wall),
W31-C-MULTI-HOST (hardware-bounded; Mac 2 ARP-incomplete 26
milestones).

**Tests + validation runs.**

437/437 phase69-78 regression tests pass byte-for-byte (was 396 in
v3.31; the new 41 tests in ``test_phase78_online_calibrated.py``
cover every enumerated H1 failure mode + falsifiers + bench
discharge).  68/68 coordpy_team_coord/runtime/public_api/extensions/
provenance tests pass.

**Honest scope (what W31 does NOT claim).**

* W31 does NOT claim "we solved context."
* W31 does NOT claim a learned model in the deep-learning sense.
  The "online learning" is a closed-form Bayesian-style running
  mean over a per-cell agreement signal; zero parameters.
* W31 does NOT claim transformer-internal latent control.
* W31 does NOT bring up Mac 2 (192.168.12.248 still ARP-incomplete).
* W31 does NOT close W30-C-NATIVE-LATENT (architecture-dependent).

See `docs/RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md` and
`docs/SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md` for the
full milestone note + pre-committed bar.

## [0.5.4 / 3.31] — 2026-05-01 — SDK v3.31 — calibrated geometry-aware dense control + multi-stride basis history + per-partition calibration prior + cross-host disagreement-routing + ancestor-chain causal binding + simultaneous discharge of W29-C-CRAM-AMPLIFICATION AND W29-C-PARTITION-CALIBRATION

*Strictly additive on SDK v3.30. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W30 surface wraps
the W29 ``GeometryPartitionedOrchestrator`` with four NEW audited
proxies at the capsule layer: a multi-stride basis-history
accumulator, a per-partition calibration prior, a sorted ancestor
chain, and a cross-host disagreement-routing decision. The new
"calibration / multi-stride basis history / cross-host disagreement-
routing / ancestor-chain" vocabulary is added at the **capsule
layer as audited proxy** — explicitly NOT a learned model, NOT
transformer-internal subspace projection, NOT a Riemannian
curvature, NOT a temporal-ordering proof at the model layer.*

**New surface (W30 family, multi-agent-coordination research slice).**

``BasisHistory``, ``AncestorChain``, ``PartitionCalibrationVector``,
``CalibratedGeometryRatificationEnvelope``,
``CalibratedGeometryRegistry``, ``W30CalibratedResult``,
``CalibratedGeometryOrchestrator``,
``verify_calibrated_geometry_ratification`` (14 enumerated failure
modes — disjoint from W29's 14, cumulative 28-mode trust boundary
across W29 + W30), ``update_partition_calibration_running_mean``,
``build_trivial_calibrated_registry``, ``build_calibrated_registry``,
W30 branch constants (``W30_BRANCH_*``),
``W30_CALIBRATED_SCHEMA_VERSION``,
``W30_DEFAULT_CALIBRATION_PRIOR_THRESHOLD = 0.5``.  W29's
``GeometryPartitionedOrchestrator`` extended with one new optional
``partition_classifier_hook`` field — the W30 extension point that
lets the calibrated orchestrator inject calibration / disagreement-
route overrides into the inner W29's pre-dispatch path.  New
benchmark phase ``vision_mvp.experiments.phase77_calibrated_dense_control``.

``__experimental__`` tuple in ``vision_mvp.coordpy.__init__`` extended
with all new W30 symbols.  ``SDK_VERSION`` bumped to
``coordpy.sdk.v3.31``.  ``pyproject.toml`` version ``0.5.4``.

**Headline empirical results (W30 family).**

* **R-77-CHAIN-CRAM (H6 cram-factor anchor).** First empirical
  discharge of **W29-C-CRAM-AMPLIFICATION**:
  ``cram_factor_w30 / cram_factor_w28 = 8.74`` AND
  ``cram_factor_w30 / cram_factor_w29 = 3.80`` across **5/5 seeds**
  (11, 17, 23, 29, 31) at ``calibration_stride = 28``,
  ``ancestor_window = 12``.  Pre-committed bar: 8.0× over W28 ✓ and
  2.0× over W29 ✓.  Max overhead = 1 token/cell.  W30's multi-stride
  basis-history accumulator packs 28 basis CIDs + 12 ancestor CIDs
  + 3 calibration priors onto one ``<calibrated_ref:DDDD>`` wire
  token.
* **R-77-CALIBRATION-PRIOR (H7 calibration anchor).** First empirical
  discharge of **W29-C-PARTITION-CALIBRATION**:
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29 =
  +0.250 across 5/5 seeds`` with ``trust_precision_w30 = 1.000``.
  Calibrated priors (0.95, 0.95, 0.30) trigger reroute on 8 CYCLIC
  cells (low historical agreement) → routed to LINEAR (high agreement,
  full oracle) → strict correctness recovery.  W29 baseline 0.750,
  W30 1.000.  Sharpens **W21-C-CALIBRATED-TRUST** on the per-partition
  axis.
* **R-77-XHOST-DISAGREE (H8 disagreement-routing anchor).**  Sharpens
  **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE**:
  ``correctness_ratified_rate_w30 - correctness_ratified_rate_w29 =
  +0.250 across 5/5 seeds`` with ``trust_precision_w30 = 1.000``.
  Synthetic disagreement probe (simulating a more-pessimistic
  cross-host LLM probe at temperature 0) systematically rejects;
  cross-host variance witness fires on every ratified cell; W30
  reroutes 15 cells to high-trust = CYCLIC (full oracle).
* **R-77-TRIVIAL-CALIBRATION (H2 byte-for-W29).** W30 = W29
  byte-for-byte; every cell yields
  ``W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH``.
* **R-77-CALIBRATED-TAMPERED (H3).** 100% rejection rate across five
  named tamper modes (basis_history_cid, calibration_vector,
  ancestor_chain_cid, disagreement_route_target, calibrated_cid).
* **R-77-NON-CALIBRATABLE (W30-Λ-non-calibratable).** Uniform priors
  (1.0, 1.0, 1.0) ⇒ no override fires ⇒ W30 = W29 on correctness;
  ``n_calibration_rerouted = 0`` correctly reported.
* **R-77-DEGENERATE-HISTORY (W30-Λ-degenerate-history).**  ``stride =
  1`` ⇒ no real cram amplification; ``cram_ratio_w30/w29 = 0.86 ≤
  1.20`` correctly reported.

Trust boundary: tampered envelopes rejected **100%** across every
sub-bank, every seed, every named mode.  Cumulative 28 enumerated
trust-boundary failure modes across W29 + W30.  **36/36 W30 unit
tests + 273/273 phase69-77 tests + 84/84 wider coordpy_* tests pass**.

**Honest scope (what W30 does NOT claim).**

* W30 does NOT claim "we solved context."
* W30 does NOT claim a learned model.  Calibration vector is a
  vector of floats in [0, 1] registered at construction; the
  running-mean update is closed-form arithmetic.
* W30 does NOT claim transformer-internal latent control.  The
  basis history is a capsule-layer accumulator over W29's
  deterministic basis CIDs; honest **proxy** for the LatentMAS
  shared-substrate direction, not a runtime KV transplant.
* W30 does NOT claim a temporal-ordering proof at the model layer.
  The ancestor chain is a sorted tuple of registered ancestor CIDs;
  the chain CID is SHA-256 over canonical bytes.
* W30 does NOT bring up Mac 2 (192.168.12.248 ARP-incomplete, 25th
  consecutive milestone).  Two reachable hosts (localhost +
  192.168.12.191) suffice for the synthetic discharge.
* W30 does NOT solve full live LLM disagreement reduction.  The H8
  strict gain is on synthetic disagreement; live-LLM extension
  remains open as **W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE**.
* W30 does NOT close ``W29-C-NATIVE-LATENT`` (architecture-dependent;
  the next true wall).

**Discharges (in this milestone).**

* **W29-C-CRAM-AMPLIFICATION** (H6).
* **W29-C-PARTITION-CALIBRATION** (H7).
* **W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** (H8 synthetic axis;
  live axis carried forward as
  W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE).
* **W21-C-CALIBRATED-TRUST** sharpened (per-partition calibrated
  priors are the natural land for the W21 conjecture).

**Verdict against pre-committed `SUCCESS_CRITERION_W30_*.md`:**
**STRONG SUCCESS** — 10/10 hard gates met AND ≥ 4/5 soft gates
PASS or honestly-null with explanation.

## [0.5.3 / 3.30] — 2026-04-30 — SDK v3.30 — geometry-partitioned product-manifold dense control + audited subspace-basis payload + factoradic Lehmer routing index + causal-validity gate + cross-host variance witness + first empirical discharge of W28-C-CROSS-HOST-VARIANCE on the magnitude axis

*Strictly additive on SDK v3.29. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W29 surface wraps
the W28 ensemble-verified multi-chain orchestrator with a structural
geometry-partitioning step (linear / hierarchical / cyclic) that
optionally dispatches each cell through a per-partition inner W28
stack with its own oracle / probe topology. The new geometry /
Grassmannian / factoradic / Lehmer / mixed-curvature / causal-
validity vocabulary is added at the **capsule layer as audited
proxy** — explicitly NOT a transformer-internal subspace projection,
NOT a Riemannian curvature, NOT a learned manifold.*

**New surface (W29 family, multi-agent-coordination research slice).**

`SubspaceBasis`, `verify_subspace_basis`,
`compute_structural_subspace_basis`,
`encode_permutation_to_factoradic`, `decode_factoradic_to_permutation`,
`CrossHostVarianceWitness`,
`GeometryPartitionedRatificationEnvelope`,
`PartitionRegistration`, `GeometryPartitionRegistry`,
`W29PartitionResult`, `GeometryPartitionedOrchestrator`
(supports optional `inner_per_partition` per-partition dispatch +
`pre_dispatch_by_partition`), `classify_partition_id_for_cell`,
`verify_geometry_partition_ratification` (14 enumerated failure
modes), `build_trivial_partition_registry`,
`build_three_partition_registry`. W29 branch constants
(`W29_BRANCH_*`). New benchmark phase
`vision_mvp.experiments.phase76_geometry_partitioned_product_manifold`.

`__experimental__` tuple in `vision_mvp.coordpy.__init__` extended
with all new W29 symbols. `SDK_VERSION` bumped to
`coordpy.sdk.v3.30`. `pyproject.toml` version `0.5.3`.

**Headline empirical results (W29 family).**

* **R-76-XHOST-DRIFT (H6 load-bearing).** First synthetic regime
  where the dense-control synthesis strictly improves correctness
  over BOTH W27 AND W28 baselines:
  `correctness_ratified_rate_w27 =
  correctness_ratified_rate_w28 = 0.500` and
  `correctness_ratified_rate_w29 = 0.750`, **Δ = +0.250 across 5/5
  seeds (11, 17, 23, 29, 31)**, `trust_precision_w29 = 1.000`,
  `mean_overhead_w29_vs_w28_per_cell = 0.75`, `max_overhead = 1`.
  **Discharges W28-C-CROSS-HOST-VARIANCE on the empirical-magnitude
  axis.**
* **R-76-CROSS-HOST-LIVE (S1/S2/S3).** Same H6 +0.250 gain holds on
  the live two-host LLM topology (localhost gemma2:9b +
  192.168.12.191 qwen2.5:14b); 16 cross-host probe calls; 710 LAN
  bytes; `trust_precision_w29 = 1.000`. Structural partition signal
  alone drives the gain even when the LLM ensemble agrees.
* **R-76-TRIVIAL-PARTITION (H2 byte-for-W28).** W29 = W28
  byte-for-byte (`byte_equivalent_w29_w28 = true`); every cell yields
  `W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH`.
* **R-76-NON-ORTHOGONAL-BASIS (H5).** 12/12 tampered envelopes
  rejected with `subspace_basis_non_orthogonal`.
* **R-76-PARTITION-TAMPERED (H3).** 60/60 tampers rejected across
  five named modes.
* **R-76-COORDINATED-DRIFT-XHOST (H5).** When per-partition pools
  share the same oracle, partition routing cannot improve
  correctness; `Δ(W29-W28) = 0.000` correctly reported.
* **Cram-factor (H7).** Measured 2.30× on R-76-CHAIN-SHARED — short
  of the pre-committed 8× bar. Mechanism real, magnitude below bar;
  becomes `W29-C-CRAM-AMPLIFICATION` (open).

Trust boundary: tampered envelopes rejected **100%** across every
sub-bank, every seed, every named mode. **935/935 + 6 subtests pass**
across W3..W29 + capsule + public API + runtime + LLM backend.

**Honest scope (what W29 does NOT claim).**

* W29 does NOT claim transformer-internal subspace projection. The
  basis lives at the capsule layer; verifier checks orthogonality,
  finiteness, content-address.
* W29 does NOT claim Riemannian curvature. The "geometry partition"
  is a structural label.
* W29 does NOT claim a learned manifold. Basis and classifier are
  pure functions over deterministic structural inputs.
* Mac 2 (192.168.12.248) remains ARP-incomplete (24th consecutive
  milestone).
* W29 does NOT solve `W22-C-CACHE-AMPLIFICATION` or full live LLM
  disagreement reduction. Both retained as named open conjectures.

## [0.5.2 / 3.29] — 2026-04-30 — SDK v3.29 — ensemble-verified cross-model multi-chain pivot ratification + Phase-75 R-75 benchmark family + W28 family + first cross-host live LLM evidence in 23 milestones + stable-vs-experimental boundary tightened

*Strictly additive on SDK v3.28. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W28 surface
composes the W21 trust-weighted multi-oracle quorum (the **old**
explicit-capsule line) with the W27 multi-chain salience-keyed
pool (the **new** dense-control line) inside one decision, behind
a controller-verified ratification envelope with 11 new enumerated
failure modes.*

**New surface (W28 family, multi-agent-coordination research slice):**

* `vision_mvp.coordpy.team_coord.ProbeVote` — frozen probe-vote
  dataclass with strict invariants (ratify ⊕ reject; trust_weight ≥ 0).
* `vision_mvp.coordpy.team_coord.EnsembleProbe` — duck-typed Protocol
  surface for any object with `probe_id` + `vote(...)`.
* `vision_mvp.coordpy.team_coord.EnsembleProbeRegistration` — mirrors
  `OracleRegistration` (the W21 entry); carries `trust_prior`,
  `role_label`, `host_id` for cross-host telemetry.
* `vision_mvp.coordpy.team_coord.DeterministicSignatureProbe` —
  locally-recomputable probe; trivially trustworthy;
  `wire_required = False` (the K=1-byte-for-W27 path).
* `vision_mvp.coordpy.team_coord.OracleConsultationProbe` — wraps any
  W20/W21 `OutsideWitnessOracle` (composes with `ServiceGraphOracle`,
  `ChangeHistoryOracle`, etc.).
* `vision_mvp.coordpy.team_coord.LLMSignatureProbe` — wraps any
  `LLMBackend` (Ollama or MLX-distributed); designed for the
  two-host topology with cross-host round-trip-bytes telemetry.
* `vision_mvp.coordpy.team_coord.EnsemblePivotRatificationEnvelope`
  — content-addressed ensemble decision envelope (signature_cid,
  probe_votes, quorum_threshold, quorum_weight, ratified flag,
  ratification_cid). 11 enumerated failure modes via
  `verify_ensemble_pivot_ratification`.
* `vision_mvp.coordpy.team_coord.EnsembleRatificationRegistry` —
  controller-side ratification registry with cross-host telemetry.
* `vision_mvp.coordpy.team_coord.EnsembleVerifiedMultiChainOrchestrator`
  — the load-bearing W28 wrapper around
  `MultiChainPersistedFanoutOrchestrator`.
* `vision_mvp.coordpy.team_coord.W28EnsembleResult` — per-cell audit
  record with probe vote summary, quorum weight, ratification CID,
  cross-host bytes.
* `vision_mvp.coordpy.team_coord.verify_ensemble_pivot_ratification`
  — 11-mode pure verifier (rejects: empty, schema-version, schema-cid,
  signature-cid-empty, signature-cid-mismatch, probe-table-empty,
  probe-id-unregistered, probe-vote-malformed, trust-weight-negative,
  hash-mismatch, quorum-below-threshold, quorum-recompute-mismatch).
* Convenience factories: `build_default_ensemble_registry`,
  `build_two_probe_oracle_ensemble_registry`,
  `build_cross_host_llm_ensemble_registry`.
* W28 branch vocabulary (8 branches): `ratified`,
  `ratified_passthrough`, `quorum_below_threshold`, `probe_rejected`,
  `no_ratify_needed`, `fallback_w27`, `no_trigger`, `disabled`.

**New benchmark family (R-75):**

* `vision_mvp/experiments/phase75_ensemble_verified_multi_chain.py`
  — eight pre-committed sub-banks (single_probe, chain_shared,
  cross_model_drift, coordinated_drift, trust_zero,
  ratification_tampered, pool_exhausted, cross_host_live) +
  `discover_two_host_topology()` + cross_regime + seed_sweep CLI.
* `vision_mvp/tests/test_phase75_ensemble_verified_multi_chain.py`
  — 34 unit + integration tests covering every probe, every
  verifier failure mode, every named falsifier, byte-for-byte W27
  equivalence, disabled/no-trigger paths, two-host topology.
* `vision_mvp/experiments/artifacts/phase75/` — 9 result JSONs
  (1 cross_regime, 7 seed sweeps, 1 topology, 1 cross_host_live).

**Headline empirical results.**

* **R-75-SINGLE-PROBE (H2 anchor)**: W28 = W27 byte-for-byte across
  5/5 seeds; `byte_equivalent_w28_w27 = true`; 16/16 cells ratified
  via the `ratified_passthrough` branch with 0 token overhead.
* **R-75-CROSS-MODEL-DRIFT (S3 / W28-3 headline, synthetic)**: W28
  overhead = 1.00 token/cell across 5/5 seeds (within S4 ≤ 2 budget);
  16/16 ratified; trust precision 1.000.
* **R-75-RATIFICATION-TAMPERED (H3 trust falsifier)**: 16/16
  tampered envelopes rejected per seed (5/5 seeds), reject reason
  `quorum_recompute_mismatch`.
* **R-75-CROSS-HOST-LIVE (S1/S2, FIRST CROSS-HOST EVIDENCE IN 23
  MILESTONES)**: 16-cell live run on localhost (gemma2:9b) +
  192.168.12.191 (qwen2.5:14b); 128 cross-host probe calls; 5592
  bytes serialised over LAN; **10/16 ratified by ensemble; 6/16
  fell to quorum_below_threshold (real LLM disagreement);
  trust_precision 1.000; W28 correctness 1.000**.

**Falsifiers all empirically confirmed:** W28-Λ-single-probe,
W28-Λ-coordinated-drift, W28-Λ-trust-zero, W28-Λ-spoofed-probe,
W28-Λ-quorum-tampered, W28-Λ-pool-exhausted-passthrough.

**Conjectures introduced:** W28-C-CROSS-HOST-VARIANCE (variance
reduction magnitude on a regime where W27 itself makes mistakes —
open; the synthetic bench is already 1.000-correct under W27 so the
S3 headline is null but honest); W28-C-CALIBRATED-TRUST (calibrated
trust priors strictly outperform uniform; not exercised in this
milestone — natural follow-up).

**Old-line discharges:** the W21 / W27 *synthesis target* named in
the master plan post-W27 next-steps section is operational —
`OracleConsultationProbe` makes the W21 oracle interface a
first-class W28 probe; W21 trust priors thread directly into W28
quorum weights; the same `OutsideWitnessOracle` duck-type drives
both W21 quorum and W28 ratification.

**Stable-vs-experimental boundary tightened (H7 release-readiness):**

* `vision_mvp.coordpy.__init__.SDK_VERSION` bumped to
  `"coordpy.sdk.v3.29"`.
* `vision_mvp.coordpy.__init__.__experimental__` — new explicit tuple
  listing every dense-control symbol (W22..W28); external callers
  should pin a specific SDK version when depending on these.
* `pyproject.toml` version bump 0.5.1 → 0.5.2.

**Regression:** 222/222 W23..W28 stack tests + 534/534 wider
focused regression (W3 capsules, W4 team, W12-W15 packing/decoder
ladder, W18-W21 explicit-capsule trust line, W22-W28 dense-control
line, public API, runtime, LLM backend) — **all preserved
byte-for-byte**.

**Two-host topology used:**
- `localhost` → `gemma2:9b` (Gemma2 family)
- `192.168.12.191` → `qwen2.5:14b` (Qwen2.5 family)
- `192.168.12.248` → ARP-incomplete (23rd consecutive milestone).

## [3.28] — 2026-04-30 — SDK v3.28 — multi-chain salience-keyed dense-control fanout + per-signature scoping + Phase-74 R-74 benchmark family + W27 family + W26-C-DIVERGENCE-RECOVERY discharged (first capsule-native multi-agent-coordination method that simultaneously improves both efficiency AND correctness over W26 on a regime where W26's single-stack scope architecturally limits correctness — measured −76.27% total token reduction AND +0.500 correctness gain over W26 on R-74-XORACLE-RECOVER at 5/5 seeds, trust boundary sound via 12 enumerated failure modes across 2 new verify_* functions, four named W27-Λ falsifiers all empirically confirmed)

*Strictly additive on SDK v3.27. The CoordPy single-run product
runtime contract is byte-for-byte unchanged.*

**New surface (W27 family, multi-agent-coordination research slice):**

* `vision_mvp.coordpy.team_coord.SalienceSignatureEnvelope` — content-
  addressed signature over canonical compact state (4 enumerated
  failure modes via `verify_salience_signature`).
* `vision_mvp.coordpy.team_coord.ChainPivotEnvelope` — per-cell pivot
  to an existing chain in the multi-chain pool (8 enumerated
  failure modes via `verify_chain_pivot`).
* `vision_mvp.coordpy.team_coord.MultiChainPersistedFanoutRegistry` —
  controller-side multi-chain pool maintained inside the audited
  disambig wrapper.
* `vision_mvp.coordpy.team_coord.MultiChainPersistedFanoutDisambiguator`
  — audited W27 wrapper on top of one W26 stack.
* `vision_mvp.coordpy.team_coord.SharedMultiChainPool` — team-wide
  pool shared across producer + K consumers; one independent W26
  disambiguator per (signature, agent).
* `vision_mvp.coordpy.team_coord.MultiChainPersistedFanoutOrchestrator`
  — load-bearing W27 implementation that routes cells via
  `compute_input_signature_cid` to the matching slot in the team-
  wide pool.
* `vision_mvp.coordpy.team_coord.compute_input_signature_cid` —
  deterministic SHA-256 over canonical input handoffs.
* W27 branch vocabulary (7 branches): `pivoted`, `anchored_new`,
  `pool_exhausted`, `pivot_rejected`, `fallback_w26`, `no_trigger`,
  `disabled`.

**New benchmark family (R-74):**

* `vision_mvp/experiments/phase74_multi_chain_pivot.py` — six
  pre-committed sub-banks + cross_regime + signature_period sweep.
* `vision_mvp/tests/test_phase74_multi_chain_pivot.py` — 22 unit +
  integration tests covering the W27 surface end-to-end.
* `docs/data/phase74_cross_regime.json`,
  `docs/data/phase74_xoracle_seed_sweep.json`,
  `docs/data/phase74_signature_period_sweep.json`.

**Headline empirical result (R-74-XORACLE-RECOVER).** W27
**simultaneously** reduces total visible tokens by −76.27 % over
W26 AND raises correctness_ratified_rate from 0.500 to 1.000 at
T_decoder ∈ {None, 24}, K=3, 16 cells. Stable across 5/5 seeds.

**Falsifiers all empirically confirmed:** W27-Λ-single-signature,
W27-Λ-pool-exhausted, W27-Λ-pivot-tampered, W27-Λ-signature-drift.

**Conjectures introduced:** W27-C-MULTI-SIGNATURE-SCALING (M → ∞
asymptote unverified), W27-C-CROSS-HOST (gated on Mac-2 return —
22nd consecutive ARP-incomplete).

**Regression:** 508/508 in the focused suite — all preserved
byte-for-byte.

## [3.27] — 2026-04-30 — SDK v3.27 — chain-persisted dense-control fanout + per-consumer projections + Phase-73 R-73 benchmark family + W26 family + W25-C-K-SCALING discharge (first capsule-native multi-agent-coordination method that amortises the producer's per-cell salience-token cost across cells via 1-token chain-advance references while preserving the W25 multi-consumer fanout floor — measured −68.79% total token reduction over W25 and −90.60% over W24 on K=3 at 5/5 seeds, trust boundary sound on tampered + projection-mismatch falsifiers, W26-Λ-no-chain falsifier confirmed, W25-C-K-SCALING discharged at K∈{3,5,8,10})

*Strictly additive on SDK v3.26. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W26 surface adds
two new content-addressed signed envelopes
(``ChainAnchorEnvelope``, ``ChainAdvanceEnvelope``), one
projection slot (``ProjectionSlot``), three verification
functions (``verify_chain_anchor`` with 6 enumerated failure modes,
``verify_chain_advance`` with 8, ``verify_projection_subscription``
with 2), one controller-side registry
(``ChainPersistedFanoutRegistry``), one audit record
(``W26ChainResult``), and one wrapping decoder
(``ChainPersistedFanoutDisambiguator``) — purely additive in the
multi-agent coordination research slice.*

**Headline (W26-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 T_decoder).** Wrapping the W25
``SharedFanoutDisambiguator`` in the new
``ChainPersistedFanoutDisambiguator`` (W26: producer pays the full
W25 cost ``C ≈ 14.6`` tokens at the chain anchor; subsequent
in-window cells emit a single ``<chain_advance:DDDD>`` token + a
hash-chained ``ChainAdvanceEnvelope``; consumers subscribe at the
anchor via per-consumer ``ProjectionSlot`` and emit a single
``<chain_consumer:DDDD>`` token per cell) over the same
R-69-CACHE-FANOUT oracle ecology
(1 producer + K=3 consumers, 16 cells, ``chain_persist_window =
16``) strictly reduces ``mean_total_w26_visible_tokens`` over
``mean_total_w25_visible_tokens`` by **−12.125 tokens / cell
(−68.79 %)** at ``T_decoder = None``, AND over
``mean_total_w24_visible_tokens`` by **−53.00 tokens / cell
(−90.60 %)**. ``correctness_ratified_rate = 1.0000`` byte-for-byte;
``chain_consumer_resolved_rate = 1.0000``;
``registry_n_anchors = 1``, ``registry_n_advances = 15``. Identical
results at ``T_decoder = 24``. Stable across **5/5** seeds.

**K-scaling discharge (W25-C-K-SCALING).** The W25-C-K-SCALING
conjecture (savings should grow as K×(C−1)) was empirically
discharged at K∈{3,5,8,10}: W25 saving over W24 grows from 69.87 %
at K=3 to 84.69 % at K=10 (close to the conjectured 88 %); W26
saving over W24 grows from 90.60 % at K=3 to 92.23 % at K=10.

**Trust-boundary anchors (W26-3).** ``verify_chain_anchor``
enumerates 6 failure modes; ``verify_chain_advance`` enumerates 8;
``verify_projection_subscription`` enumerates 2. On
R-73-CHAIN-TAMPERED, 14/16 advances rejected via
``parent_mismatch``; correctness preserved via W25 fall-through.
On R-73-PROJECTION-MISMATCH, all 16 cells reject for the
mismatched consumer via ``projection_unauthorized``; the other 2
consumers still resolve.

**Named falsifiers** (all proved-empirical). **W26-Λ-no-chain**:
``chain_persist_window = 1`` reduces W26 to W25 byte-for-byte.
**W26-Λ-tampered**: tampered advances rejected. **W26-Λ-projection-mismatch**:
cross-projection access rejected. **W26-Λ-divergent**: when gold
subset flips at the bench midpoint, the inner W25 fires
``no_trigger`` and W26 falls through; correctness drops to 0.5
by construction.

**Backward-compat (W26-3-A, W26-3-B).** With ``enabled = False``
OR ``chain_registry = None``, W26 reduces to W25 byte-for-byte.
Full pre-existing W22..W25 + IS-1 / IS-2 test surfaces preserved
byte-for-byte (180/180 in the focused regression).

**Theoretical (W26-L, proved by inspection).** Any capsule-native
multi-agent coordination strategy whose producer emits only its
own compact state and whose consumers reference it via 1-token-
per-cell tokens has a per-cell total visible cost ≥ 1 + K. W26
attains this floor on every in-window advance cell.

**Mac 2 status: ARP-incomplete (21st consecutive milestone)**;
all results Mac-1 only. The W26 surface inherits the W24
``CrossProcessProducerDecoderWire`` as the strongest cross-
process honesty validated end-to-end on this repo. The
wire-bytes vs token-cost tradeoff is named **W26-C-MULTI-HOST**;
remains conjectural.

**New tests:** 63/63 pass on
``test_phase73_chain_persisted_fanout.py``. Full focused
regression on W22..W26 + IS-1 / IS-2: **180/180 + 6 subtests
pass in 15.6s**.

See ``docs/RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md`` for the
milestone note, ``docs/THEOREM_REGISTRY.md`` for the 13 new
theorems / falsifiers / conjectures (W26-1 through W26-C-MULTI-HOST,
plus W25-C-K-SCALING discharge), and
``docs/context_zero_master_plan.md`` § 4.44 for the
master-plan-level audit board.

## [3.26] — 2026-04-29 — SDK v3.26 — shared-fanout dense-control + cross-agent state reuse + Phase-72 R-72 benchmark family + W25 family (first capsule-native multi-agent-coordination method that extends W24 single-agent compaction to the multi-agent case — one producer computes 1 FanoutEnvelope for K named consumers, each consumer resolves via 1 ``<fanout_ref:DDDD>`` token, measured −69.87% total token reduction on K=3 at 5/5 seeds, trust boundary sound on poisoned-consumer falsifier, W25-Λ-disjoint named falsifier confirmed)

*Strictly additive on SDK v3.25.* The W25 surface adds one new
content-addressed signed envelope (``FanoutEnvelope``), one
controller-side registry (``SharedFanoutRegistry``), one
verification function (``verify_fanout``), one audit record
(``W25FanoutResult``), and one wrapping decoder
(``SharedFanoutDisambiguator``). On R-72-FANOUT-SHARED (1
producer + K=3 consumers, 16 cells, R-69-CACHE-FANOUT oracle
ecology), W25 strictly reduces ``mean_total_w25_visible_tokens``
over ``mean_total_w24_visible_tokens`` by **−40.875 tokens / cell
(−69.87 %)**; ``correctness_ratified_rate = 1.0000``;
``fanout_consumer_resolved_rate = 1.0000``. Stable across **5/5**
seeds. See ``docs/RESULTS_COORDPY_W25_SHARED_FANOUT.md``.

## [3.25] — 2026-04-29 — SDK v3.25 — bounded-window session compaction + intra-cell resample-quorum + real cross-process producer/decoder wire + Phase-71 R-71 benchmark family + W24 family (first capsule-native multi-agent-coordination method that combines bounded-window session compaction with intra-cell resample-quorum mitigation and a real OS-level cross-process producer/decoder wire — measured efficiency gain on long sessions, empirical discharge of W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell drift axis, and the first real OS-level subprocess pipe in the programme)

*Strictly additive on SDK v3.24. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W24 surface adds
one new content-addressed signed envelope
(``SessionCompactEnvelope``), one verification function
(``verify_session_compact``), one audit record
(``W24CompactionResult``), one wrapping decoder
(``MultiCellSessionCompactor``), one intra-cell resample-quorum
caching adapter (``ResampleQuorumCachingOracleAdapter``), one
real cross-process producer/decoder wire
(``CrossProcessProducerDecoderWire``), and one synthetic intra-
cell drift oracle (``IntraCellFlippingOracle``) — purely additive
in the multi-agent coordination research slice.*

**Headline (W24-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 budgets).** Pairing the W23
``CrossCellDeltaDisambiguator`` with the new
``MultiCellSessionCompactor`` over a registered ``SchemaCapsule``
on R-71-LONG-SESSION (16-cell session, ``compact_window = 4``)
strictly reduces ``mean_n_w24_visible_tokens_to_decider`` over
the W23 baseline by **−6.81 tokens / cell (−18.0 %)** at
``T_decoder = None`` and by **−6.81 tokens / cell (−20.5 %)** at
``T_decoder = 24``. ``compact_verifies_ok_rate = 0.812`` (13/16
cells beyond the window); ``correctness_ratified_rate = 1.000``
byte-for-byte vs W22; ties W22 / W23 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values: savings ≥ 6.69 tokens/cell on every seed,
mean savings 6.79 tokens/cell.

**W24-2 mitigation (proved-empirical n=8 saturated synthetic +
proved-empirical-research n=4 live mixtral).** On
R-71-INTRA-CELL-FLIP (synthetic ``IntraCellFlippingOracle``
registered in isolation so its vote is decisive in W21 quorum),
the W23 PER_CELL_NONCE baseline ties FIFO at ``acc_full = 0.000``
(each cell's first consult is the bad one); the W24
``ResampleQuorumCachingOracleAdapter`` (M=3, T=2) achieves
``acc_full = 0.500`` — **+0.500 strict mitigation advantage**.
**Empirically discharges W23-C-MITIGATION-LIVE-VARIANCE on the
intra-cell drift axis**. Live transfer to ``mixtral:8x7b`` on
Mac-1 Ollama (n=4): W23 quorum-keyed = 0.500, W24 resample =
**0.750** — **+0.250 strict gain on a fresh live LLM stream**.

**W24-3 trust-boundary soundness + real cross-process wire
(proved-empirical n=16 + proved by inspection).** On
R-71-COMPACT-TAMPERED, every tampered window is rejected (12/16
cells fire ``window_cids_mismatch`` → fall through to W23
byte-for-byte; ``correctness_ratified_rate = 1.000``). On
R-71-CROSS-PROCESS, the ``CrossProcessProducerDecoderWire`` spawns
a real Python subprocess and round-trips JSON envelopes via
stdin/stdout pipes: **12 861 bytes round-tripped on n=16, 0
failures** — a strictly stronger cross-process honesty proxy than
the W23 within-process round-trip.

**W24-Λ-no-compact (named falsifier).** On R-71-NO-COMPACT (chain
reset every cell), ``n_w24_compact_resolved_cells = 0`` AND W24
reduces to W23 byte-for-byte. Names the structural limit when the
chain length stays below the window.

**W24-Λ-real (proved-conditional + empirical-research n=4).**
Live mixtral 8x7b probe on R-71-INTRA-CELL-FLIP yields +0.250
strict mitigation advantage; the synthetic +0.500 does not fully
transfer because the live LLM does not perfectly match the
deterministic IntraCellFlippingOracle pattern. Names
**W24-C-LIVE-VARIANCE-COMPLETE** as the follow-up conjecture
frontier.

**Backward-compat.** 121/121 phase-69/70/71 + capsule tests pass;
33/33 new W24 tests pass; 619/619 coordpy-anchor + capsule + recent
phases pass. With ``enabled = False`` OR ``schema = None`` OR no
multi-cell window, W24 reduces to W23 byte-for-byte.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — **18th milestone in a row**.
**No two-Mac sharded inference happened in SDK v3.25.** The W24-3
``CrossProcessProducerDecoderWire`` upgrades the W23 within-
process round-trip to a real OS-level subprocess pipe — the
strongest cross-process honesty this repo can validate end-to-end
on Mac-1 alone. When Mac 2 returns the same JSON-canonical
interface drops in over a real socket with no W24 code changes.

See ``docs/RESULTS_COORDPY_W24_SESSION_COMPACTION.md`` for the
theory-forward results note,
``docs/THEOREM_REGISTRY.md`` for the W24 theorem family entries,
``docs/HOW_NOT_TO_OVERSTATE.md`` § "W24 forbidden moves" for the
canonical do-not-overstate rules, and
``vision_mvp/experiments/phase71_session_compaction.py`` for the
R-71 driver.

## [3.23] — 2026-04-29 — SDK v3.23 — capsule + audited latent-state-sharing hybrid + R-69 Phase-69 benchmark family + W22 family (first capsule-native multi-agent-coordination method that combines explicit-capsule passing with audited proxies for the LatentMAS direction — schema-passing, delta execution, shared-read cache, controller-verified latent digest envelope — measured efficiency gain on a regime where the W21 wire-cost concern actually applies)

*Strictly additive on SDK v3.22. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W22 surface adds
one new content-addressed dataclass (``SchemaCapsule``), one
typed signed envelope (``LatentDigestEnvelope``), one verification
result (``LatentVerificationOutcome``), one wire-side
write-once-read-many cache (``SharedReadCache``), one drop-in
oracle adapter (``CachingOracleAdapter``), one falsifier-test
primitive (``EnvelopeTamperer``), one audit record
(``W22LatentResult``), one controller verifier
(``verify_latent_digest``), and one wrapping decoder
(``LatentDigestDisambiguator``) — purely additive in the
multi-agent coordination research slice.*

**Headline (W22-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 cells).** Pairing the W21
``TrustWeightedMultiOracleDisambiguator`` with the new
``LatentDigestDisambiguator`` over a registered ``SchemaCapsule``
+ a shared ``SharedReadCache`` (every oracle wrapped in
``CachingOracleAdapter``) on R-69-CACHE-FANOUT strictly reduces
``mean_n_visible_tokens_to_decider`` over the W21 baseline by
**−7 tokens / cell (−14.51 %)** at ``T_decoder = None`` and by
**−7 tokens / cell (−16.09 %)** at ``T_decoder = 24``, AND
records ``cache_tokens_saved_total = 88`` over n=8 (oracle-side
wire savings), AND ties W21 byte-for-byte on
``accuracy_full = 1.000``. Stable across 5/5 alternate
``bank_seed`` values (savings exactly +7 tokens / cell on every
seed; cache_tokens_saved=88 on every seed; correctness ratified
rate=1.000 on every seed). The first capsule-native multi-agent-
coordination method that combines explicit-capsule passing with
audited proxies for the LatentMAS direction (collective KV
pooling / latent hidden-state transfer / super-token side
channels). Three named falsifiers (W22-Λ-no-cache,
R-69-POISONED-DIGEST, R-69-SCHEMA-DRIFT) and one backward-compat
anchor (R-69-NO-TRIGGER) make the W22-1 conditionality sharp.

**Live LLM transfer (W22-Λ-real, empirical n=4 × 2 models,
partially discharged).** Mac-1 mixtral 8x7b on cache_fanout:
visible-tokens savings **+39.08 %** (W21=87, W22=53 tokens / cell);
cache_tokens_saved_total=120 over 4 cells; correctness ratified
rate=0.750 — newly named conjecture **W22-C-CACHE-AMPLIFICATION**
(the cache freezes a probabilistic LLM oracle's first reply
across all matching cells; cell-1's variance amplifies). gemma2:9b
ties W21 byte-for-byte at 0.250 (gemma2's closure-landing rate is
the structural bound).

**Backward-compat preserved byte-for-byte.** With ``enabled=False``
OR no ``SchemaCapsule`` registered OR an inner W21 branch outside
``trigger_branches``, the W22 layer reduces to W21 byte-for-byte
on the answer field. 633 prior coordpy-suite tests + 32 new W22
tests + 10 misc = **675/675** pass.

**Audit T-1..T-7 preserved** on every cell of every regime.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` (16th milestone in a row); no two-Mac sharded
inference. The W22 surface is naturally a producer / cache-
controller separation (wire-compatible with cross-host
deployment) — no W22 code changes required when Mac-2 returns.

**Closes** the wire-cost half of the SDK v3.22
W21-C-CALIBRATED-TRUST conjecture (the *correctness* half remains
open and orthogonal). See
`docs/RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md` for the full
SDK v3.23 milestone note; `vision_mvp/experiments/phase69_capsule_latent_hybrid.py`
for the bench driver; `vision_mvp/tests/test_phase69_capsule_latent_hybrid.py`
for the 32 new tests.

## [3.22] — 2026-04-29 — SDK v3.22 — trust-weighted multi-oracle adjudicator + R-68 multi-oracle benchmark family + W21 family (first capsule-native multi-agent-coordination method that crosses the W20-Λ-compromised wall on a regime where single-oracle reasoning is structurally insufficient by adjudicating across N registered outside oracles under bounded context)

*Strictly additive on SDK v3.21. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W21 surface adds
one new dataclass (``OracleRegistration``), four oracle adapters
(``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle``), two
audit dataclasses (``W21OracleProbe``, ``W21MultiOracleResult``),
six new branch constants (``W21_BRANCH_QUORUM_RESOLVED``,
``W21_BRANCH_NO_QUORUM``, ``W21_BRANCH_SYMMETRIC_QUORUM``,
``W21_BRANCH_NO_ORACLES``, ``W21_BRANCH_NO_TRIGGER``,
``W21_BRANCH_DISABLED``), and one wrapping decoder
(``TrustWeightedMultiOracleDisambiguator``). All purely additive
in the multi-agent coordination research slice.*

**Headline (W21-1, proved-conditional + proved-empirical n=80
saturated × 5 seeds × 2 cells, also n=12).** Pairing the W19
``BundleContradictionDisambiguator`` with the new
``TrustWeightedMultiOracleDisambiguator`` over a three-oracle
registered set ``(compromised_registry first, service_graph,
change_history)`` under default ``quorum_min = 2`` achieves
``accuracy_full = 1.000`` on R-68-MULTI-MAJORITY-LOOSE
(``T_decoder = None``) AND R-68-MULTI-MAJORITY-TIGHT
(``T_decoder = 24``), strictly improving over **every** non-W21
capsule baseline including W20 (which trusts the first-registered
compromised oracle and FAILS at 0.000) by **+1.000**, stable across
**5/5** alternate ``bank_seed`` values (11, 17, 23, 29, 31). The
first capsule-native multi-agent-coordination method that crosses
the W20-Λ-compromised wall on a regime where the wall actually
applies. Three named falsifiers (W21-Λ-no-quorum,
W21-Λ-all-compromised, W21-Λ-partial) make the W21-1 conditionality
sharp; the conditional W21-C-PARTIAL-RECOVERY (with override
``quorum_min = 1`` on R-68-MULTI-PARTIAL) is empirically discharged
at 1.000 — the quorum-strictness trade-off is real.

**Live LLM transfer (SDK v3.22, W21-Λ-real / W21-C-LIVE-WITH-REGISTRY,
empirical n=4 × 2 models, partially discharged).** Two regimes:

* **Mixed-registry (registry-anchored, easy)** — four-oracle
  registry pairing deterministic ``service_graph`` +
  ``change_history`` with ``ollama_mixtral:8x7b``: W21 = 1.000,
  +1.000 over W20. ``W21-C-LIVE-WITH-REGISTRY`` partially
  discharged.
* **Coalition (LLM-vote-required, hard)** — three-oracle registry
  with one honest deterministic + one LLM + one compromised,
  ``quorum_min = 2`` (LLM vote required for quorum on gold):
  ``ollama_mixtral:8x7b`` (47B-MoE) lands gold tokens through the
  W18/W19 closure on 3/4 cells → W21 = **0.750**, **+0.750 over
  W20**; ``ollama_gemma2:9b`` (9.2B-dense) lands decoy tokens
  through the closure → W21 = **0.000**, **+0.000 over W20**.
  Cross-model split (47B-MoE / 9.2B-dense) sharp; **scale + general
  knowledge matter for the W21-Λ-real escape on the LLM-vote-
  required regime**.

**Two-Mac status (SDK v3.22).** Mac 2 (192.168.12.248) ARP
``incomplete`` — same status as SDK v3.6 through SDK v3.21 (15th
milestone in a row). **No two-Mac sharded inference happened in SDK
v3.22.** The W21 oracle Protocol is *naturally* a producer / multi-
adjudicator separation; cross-host deployment (registry on Mac-1,
LLM adjudicator on Mac-2) is wire-compatible — no W21 code changes
required when Mac-2 returns. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7B-MoE Q4) on Mac 1
Ollama.

**Bounded-context honesty (SDK v3.22).** The W21 layer issues
*exactly N = ``len(oracle_registrations)``* outside queries per
cell, each bounded by ``max_response_tokens``. The inner W15
``tokens_kept`` is byte-for-byte identical between W19, W20 AND
W21 on R-68-MULTI-MAJORITY-TIGHT (mechanically verified). Total
context delivered to the final decider on the 3-oracle stack:
``tokens_kept (≤ T_decoder) + 3 × n_outside_tokens (each ≤
max_response_tokens)``.

**Backward-compat (W21-3-A / W21-3-B) preserved byte-for-byte.**
With ``enabled = False`` OR no oracles registered, W21 reduces to
W19 byte-for-byte. With ``quorum_min = 1`` AND a single registered
honest oracle, W21 ties W20 byte-for-byte on R-67-OUTSIDE-RESOLVES.
Full SDK regression: **633 / 633 coordpy tests pass** (= 585 prior +
48 new W21 tests).

**Closes the named SDK v3.21 conjectures.** W20-C-MULTI-ORACLE
(named conjectural in SDK v3.21) is **discharged** by W21-1 on
R-68-MULTI-MAJORITY. W20-C-LIVE-WITH-REGISTRY (named conjectural
in SDK v3.21) is **partially discharged** by the live mixed-
registry probe on Mac-1 mixtral 8x7b.

**SDK v3.22 mint files:**

* New W21 surface in ``vision_mvp/coordpy/team_coord.py`` (purely
  additive on top of W20).
* New experiment:
  ``vision_mvp/experiments/phase68_multi_oracle_adjudication.py``
  (R-68 driver: 5 sub-banks + cross-regime synthetic + 5-seed
  stability sweep + live mixed-registry / coalition probes).
* New tests:
  ``vision_mvp/tests/test_coordpy_multi_oracle_adjudication.py`` (48
  tests).
* New artifacts: ``docs/data/phase68_*.json`` (7 files).
* New milestone results note:
  ``docs/RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md``.
* Master plan (§ 4.39), THEOREM_REGISTRY (W21 family with 11
  entries), SUCCESS_CRITERION (Bar 18), START_HERE, README,
  RESEARCH_STATUS, papers/context_as_objects.md (§ 14.2 escape
  ladder) all updated.
* ``SDK_VERSION = "coordpy.sdk.v3.22"``.

## [3.21] — 2026-04-29 — SDK v3.21 — outside-witness acquisition disambiguator + R-67 outside-information benchmark family + W20 family (first capsule-native multi-agent-coordination method that crosses the W19-Λ-outside wall on a regime where bundle-only reasoning is structurally insufficient by acquiring asymmetric outside information)

*Strictly additive on SDK v3.20. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W20 surface
adds one new Protocol (``OutsideWitnessOracle``), three new
dataclasses (``OutsideQuery``, ``OutsideVerdict``,
``W20OutsideResult``), four oracle adapters
(``ServiceGraphOracle`` / ``CompromisedServiceGraphOracle`` /
``AbstainingOracle`` / ``LLMAdjudicatorOracle``), one default
service-graph (``build_incident_triage_service_graph``), one
wrapping decoder (``OutsideWitnessAcquisitionDisambiguator``),
and six W20 branch constants. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase67_outside_information.py`` driver
ships as a research-slice addition with one positive anchor
sub-bank (R-67-OUTSIDE-RESOLVES under loose AND tight
``T_decoder``), one backward-compat sub-bank
(R-67-OUTSIDE-REQUIRED-BASELINE), and three named falsifier
sub-banks (R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED,
R-67-JOINT-DECEPTION).*

**The W20 family — outside-witness acquisition disambiguator
(SDK v3.21).** On a synthetic R-67-OUTSIDE-RESOLVES regime
(the same R-66-OUTSIDE-REQUIRED bundle shape — deceptive
primary mentions decoy only AND symmetric secondary witness
mentions all three — but with a deterministic
:class:`ServiceGraphOracle` registered as the outside
information source), every closed-form scorer in the SDK
pre-W20 — substrate FIFO, ``capsule_fifo``, ``capsule_priority``,
``capsule_coverage``, W7-2 cohort, W8 corroboration, W9 multi-
service, W11 multi-round, W12 robust-multi-round, W13 layered,
W15 ``AttentionAwareBundleDecoder``, W18
``RelationalCompatibilityDisambiguator``, AND **W19
``BundleContradictionDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W19-Λ-outside extends verbatim:
W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``). The W20
method, with the deterministic ServiceGraphOracle, achieves
``accuracy_full = 1.000`` on R-67-OUTSIDE-RESOLVES-LOOSE
(``T_decoder = None``) AND R-67-OUTSIDE-RESOLVES-TIGHT
(``T_decoder = 24``), strictly improving over every non-W20
capsule baseline by **+1.000**, stable across **5/5**
alternate ``bank_seed`` values (11, 17, 23, 29, 31). Three
named falsifiers (R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED,
R-67-JOINT-DECEPTION) make the W20-1 conditionality sharp:
no signal → abstain → tie FIFO; adversarial signal → trust →
fail at 0.000; jointly compromised → tie W19 at 0.000.
Bounded-context honesty: the W20 layer adds *exactly one*
outside query per cell, bounded by ``max_response_tokens =
24``; the W15 ``tokens_kept`` is byte-for-byte identical
between W19 and W20 on R-67-OUTSIDE-RESOLVES-TIGHT. Backward-
compat (W20-3) preserved byte-for-byte: 545 / 545 prior coordpy
tests pass + 40 new W20 tests pass = **585 / 585**.

**Live LLM extension (W20-Λ-real, partial pass).** A
:class:`LLMAdjudicatorOracle` over a fresh live Mac-1 Ollama
backend produces *measured*, not claimed, results. Partial
live advance: ``mixtral:8x7b`` (47B-MoE) free-form reply
mentions gold tokens asymmetrically and achieves ``acc_full
= 0.750`` (3/4 cells, ``+0.750`` strict gain over W19) on
``R-67-OUTSIDE-RESOLVES`` at ``n_eval = 4``,
``K_auditor = 12``. Honest negative: ``qwen2.5-coder:7b``
trusts the deceptive primary on every fired cell
(``services=cache``) and ties FIFO at 0.000 — the W20-Λ-real
under-scaled-model failure mode. Cross-model split: scale +
general knowledge correlates with W20-Λ-real escape;
smaller / coding-specialised models can fall into the
deception. Artifacts:
``docs/data/phase67_live_mixtral_8x7b_n4.json``,
``docs/data/phase67_live_qwen2_5_coder_7b_n4.json``.

**Two-Mac status (SDK v3.21).** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK
v3.6 through SDK v3.20. **No two-Mac sharded inference
happened in SDK v3.21.** The W20 ``OutsideWitnessOracle``
Protocol is *infrastructure-ready* for cross-host deployment
(producer roles on Mac 1 + adjudicator on Mac 2) when Mac 2
returns; the ``MLXDistributedBackend`` adapter is byte-for-
byte unchanged.

**The W20 theorem family.** Twelve W20 statements:
**W20-Λ-outside-extension** (proved-empirical n=8 saturated
+ structural sketch); **W20-1** (proved-conditional + proved-
empirical n=80 across 5 seeds × 2 budgets, also n=12);
**W20-2** (proved by inspection + mechanically-checked);
**W20-3** (proved-empirical full programme regression);
**W20-Λ-none** / **W20-Λ-compromised** /
**W20-Λ-joint-deception** (each proved-empirical n=8
saturated); **W20-Λ-real** (proved-conditional + empirical-
research, n=4 × 2 models on Mac-1); **W20-C-LEARNED** /
**W20-C-MULTI-ORACLE** / **W20-C-LIVE-WITH-REGISTRY** /
**W20-C-CROSS-BENCH** (conjectural). Honest scope:
``R-67-OUTSIDE-RESOLVES`` is a *synthetic* regime — the
producer is :class:`IdentityExtractor` AND the oracle is a
deterministic :class:`ServiceGraphOracle`. The W20 closure is
bounded by the same closed-vocabulary discipline that bounds
W19 / W18 / W13. The W20 escape is **partial** by design,
bounded above by oracle integrity (W20-Λ-compromised) and by
joint-N-oracle compromise (W20-Λ-joint-deception).

**SDK version bumped: ``coordpy.sdk.v3.20`` → ``coordpy.sdk.v3.21``.**

## [3.20] — 2026-04-28 — SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator + deceptive-ambiguity-under-trust benchmark family + W19 family (first capsule-native multi-agent-coordination method to cross the deceptive-ambiguity wall on a regime where bundle-only relational compatibility is structurally insufficient)

*Strictly additive on SDK v3.19. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W19 surface adds
one new dataclass (``W19TrustResult``), one wrapping decoder
(``BundleContradictionDisambiguator``), one canonical-role-for-kind
table (``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND``), one
asymmetric-witness counter (``_w19_witness_counts``), one
canonical-primary-index helper (``_w19_canonical_primary_index``),
and seven W19 branch constants. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase66_deceptive_ambiguity.py`` driver
ships as a research-slice addition with one positive
ratification anchor sub-bank (R-66-CORROBORATED), two strict-win
sub-banks (R-66-DECEIVE-NAIVE under loose AND tight ``T_decoder``,
R-66-CONFOUND-RESOLVABLE), and two named falsifier sub-banks
(R-66-DECEIVE-TOTAL, R-66-OUTSIDE-REQUIRED).*

On the new Phase-66 *deceptive-ambiguity-under-trust* benchmark:

* **R-66-CORROBORATED** (no adversarial signal; corroborating
  witnesses present; **W19-3 ratification anchor**; n=8 ×
  bank_replicates=2): W19 ties the W18
  ``RelationalCompatibilityDisambiguator`` byte-for-byte at
  ``accuracy_full = 1.000``. W19 reduces to W18 when no
  contradicting witnesses exist (asymmetric witness scores are
  uniformly zero ⇒ W18 trust path fires unchanged).
* **R-66-DECEIVE-NAIVE-LOOSE** (round-2 disambiguator
  adversarially mentions DECOY but NOT gold; secondary
  specific-tier witnesses corroborate gold; **W19-1 deceive
  anchor**, ``T_decoder = None``; n=8 × bank_replicates=2): every
  closed-form salience scorer including W18 ties FIFO at
  ``accuracy_full = 0.000`` (W18 trusts the adversarial
  disambiguator and picks decoy; W18-Λ-deceive extends to
  R-66 verbatim by W19-Λ-deceive-extension). The new W19
  ``BundleContradictionDisambiguator`` achieves
  ``capsule_bundle_contradiction = 1.000``. **+1.000 strict
  separation** vs every non-W19 capsule baseline.
* **R-66-DECEIVE-NAIVE-TIGHT** (same regime under decoder-side
  budget pressure ``T_decoder = 24``): same headline as loose.
  The W19 method composes cleanly with the W15 attention-aware
  pack; ``tokens_kept_sum`` is byte-for-byte identical to W18's
  on this regime (bounded-context honesty preserved).
* **R-66-CONFOUND-RESOLVABLE** (round-2 disambiguator mentions
  BOTH gold AND decoy symmetrically — the W18-Λ-confound wall;
  secondary specific-tier witnesses break the tie asymmetrically
  toward gold; **W19-1 confound anchor**, n=8 × bank_replicates=2):
  W18 abstains at 0.000 (symmetric round-2 disambiguator); the
  new W19 method achieves ``capsule_bundle_contradiction = 1.000``.
  **+1.000 strict separation** vs W18 and every other baseline.
* **5-seed stability** on R-66-DECEIVE-NAIVE-LOOSE,
  R-66-DECEIVE-NAIVE-TIGHT, AND R-66-CONFOUND-RESOLVABLE: gap
  ``w19 − w18 = +1.000`` on every seed in
  ``{11, 17, 23, 29, 31}`` (saturated; well above the 0.50
  strong-bar threshold).
* **R-66-DECEIVE-TOTAL** (W19-Λ-total falsifier): no witnesses
  anywhere (round-2 disambiguator adversarial; secondary handoffs
  silent); aw uniformly zero ⇒ W19 reduces to W18 byte-for-byte
  ⇒ W19 ties FIFO at 0.000 on 8/8 cells. Names the structural
  limit no bundle-only closed-form scorer can escape when the
  bundle carries no exonerating evidence at all.
* **R-66-OUTSIDE-REQUIRED** (W19-Λ-outside falsifier): witnesses
  are symmetric across gold and decoy (each side gets the same
  asymmetric witness count); W19's tiebreak is a wash ⇒ W19
  abstains; ties FIFO at 0.000 on 8/8 cells. Names the structural
  limit no bundle-only closed-form scorer can escape when the
  symmetry inside the bundle is total. The named research move
  beyond it is W19-C-OUTSIDE (conjectural; requires
  outside-the-bundle information — a learned scorer, an external
  knowledge base, an extra round of evidence).
* **W19-3 backward-compat**: on R-58 default the W19 method ties
  W18 byte-for-byte on the answer field; on every R-65 default
  bank (compat / no_compat / confound / deceive) W19 ties W18
  byte-for-byte. With ``enabled = False`` the W19 method reduces
  to W18 byte-for-byte.
* **Audit T-1..T-7** OK on every capsule strategy of every cell of
  every regime.

The W19 surface is purely additive on top of the W18 surface (one
new dataclass + one canonical-role table + one asymmetric-witness
counter + one canonical-primary-index helper + one wrapping
decoder + seven branch constants). The SDK v3.19 runtime contract
is byte-for-byte unchanged. New tests cover the W19 unit
semantics, the Phase-66 bench-property witnesses, the W19-1
strict-win anchor on three deceptive regimes, the 5-seed
stability across all three, the two named falsifiers, the
backward-compat smoke (R-58 + every R-65 default bank), the
token-budget honesty, and the cross-regime synthetic summary —
all 555/555 coordpy tests pass.

### Added

- **W19 surface** (``vision_mvp/coordpy/team_coord.py``):
  - ``BundleContradictionDisambiguator`` (wraps W18; consumes
    asymmetric specific-tier witness counts to either ratify
    W18's verdict or override it when the bundle's witness
    distribution contradicts W18's choice; reduces to W18
    byte-for-byte when no witnesses exist).
  - ``W19TrustResult`` dataclass (decision + branch + witness
    counts per tag + token bookkeeping).
  - ``_w19_witness_counts(...)``, ``_w19_canonical_primary_index(...)``
    helpers + ``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND`` table
    + ``W19_SYMMETRIC_NOISE_KINDS`` frozenset + seven
    ``W19_BRANCH_*`` constants.
  - Re-exported via ``vision_mvp.coordpy.__all__``.
- **Phase-66 driver**
  (``vision_mvp/experiments/phase66_deceptive_ambiguity.py``):
  five-bank synthetic benchmark + cross-regime summary +
  5-seed stability sweep + closed-vocabulary secondary-witness
  routing extension (``_P66_SECONDARY_ROUTES``).
- **Tests** (``vision_mvp/tests/test_coordpy_bundle_contradiction.py``):
  45 tests across 10 test classes — unit semantics, bench
  properties, default config, 5-seed stability, two named
  falsifiers, backward-compat, token efficiency, cross-regime,
  invariants.
- **Docs**: ``docs/RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md`` (new
  milestone note, ~12KB) + R-66 anchor + W19 family in
  ``docs/THEOREM_REGISTRY.md`` + bar 16 in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` +
  ``docs/HOW_NOT_TO_OVERSTATE.md`` (W19 honest-scope sections)
  + ``docs/RESEARCH_STATUS.md`` SDK v3.20 frontier +
  ``docs/context_zero_master_plan.md`` § 4.37 + four data files
  in ``docs/data/phase66_*.json``.

### Discharged conjectures

- **W18-Λ-deceive** (SDK v3.19 falsifier; named limit on bundle-
  relational scorers that *trust* round-2 disambiguator evidence):
  **PARTIALLY DISCHARGED** by W19-1 on R-66-DECEIVE-NAIVE — the
  bundle carries asymmetric secondary witnesses W18 ignored;
  W19's witness counter consumes them. Remaining limit
  W19-Λ-total (no witnesses anywhere) is genuinely beyond
  bundle-only closed-form scorers.
- **W18-Λ-confound** (implicit SDK v3.19 falsifier; W18 abstains
  on symmetric round-2 disambiguator): **PARTIALLY DISCHARGED**
  by W19-1 on R-66-CONFOUND-RESOLVABLE — secondary witnesses
  break the inside-bundle symmetry asymmetrically. Remaining
  limit W19-Λ-outside (symmetric witnesses) is genuinely beyond
  bundle-only closed-form scorers.

### Preserved

- **SDK v3.19 multi-agent surface.** Every fixed admission
  policy, every closed-form salience scorer, every bundle-aware
  decoder, every layered normaliser, every producer protocol,
  every attention-aware pack, every relational-compatibility
  disambiguator — all byte-for-byte unchanged from SDK v3.19.
- **CoordPy single-run product report v2 schema:** byte-for-byte
  identical from SDK v3.19.

See [`docs/RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md`](docs/RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md)
for the full milestone note.

## [3.19] — 2026-04-28 — SDK v3.19 — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family + W18 family (first capsule-native multi-agent-coordination method to cross the symmetric-corroboration wall on a regime where the wall actually applies)

*Strictly additive on SDK v3.18. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W18 surface adds
one new dataclass (``W18CompatibilityResult``), one wrapping
decoder (``RelationalCompatibilityDisambiguator``), one
deterministic tokeniser (``_disambiguator_payload_tokens``), and
one closed-form scorer (``_relational_compatibility_score``) with
contiguous-subsequence semantics for compound targets. Re-exported
via the SDK ``__init__``. The new
``vision_mvp/experiments/phase65_relational_disambiguation.py``
driver ships as a research-slice addition with one positive-
anchor sub-bank (R-65-COMPAT under loose AND tight
``T_decoder``) plus three named falsifier sub-banks
(R-65-NO-COMPAT, R-65-CONFOUND, R-65-DECEIVE).*

On the new Phase-65 *relational-compatibility disambiguation
under symmetric corroboration* benchmark:

* **R-65-COMPAT-LOOSE** (synthetic identity producer, magnitude-
  hinted prompt, ``T_decoder = None``; **W18-1 anchor**; n=8 ×
  bank_replicates=2): every closed-form salience scorer in the
  SDK — substrate FIFO, ``capsule_fifo``, ``capsule_priority``,
  ``capsule_coverage``, W7-2 cohort, W8 corroboration, W9 multi-
  service, W11 multi-round, W12 robust, W13 layered, W15
  ``AttentionAwareBundleDecoder``, ``capsule_layered_fifo_packed``
  — ties FIFO at ``accuracy_full = 0.000`` (W17-Λ-symmetric extends
  to R-65-COMPAT verbatim by W18-Λ-sym). The new W18
  ``RelationalCompatibilityDisambiguator`` achieves
  ``capsule_relational_compat = 1.000``. **+1.000 strict
  separation** vs every non-W18 capsule baseline.
* **R-65-COMPAT-TIGHT** (same regime under decoder-side budget
  pressure ``T_decoder = 24``): same headline as loose. The W18
  method composes cleanly with the W15 attention-aware pack;
  ``tokens_kept_sum`` is byte-for-byte identical to W15's
  (bounded-context honesty preserved).
* **5-seed stability** on R-65-COMPAT-LOOSE AND R-65-COMPAT-TIGHT:
  gap ``w18 − attention_aware = +1.000`` on every seed in
  ``{11, 17, 23, 29, 31}`` (saturated; well above the 0.50
  strong-bar threshold).
* **R-65-NO-COMPAT** (W18-Λ-no-compat falsifier): round-2
  disambiguator carries no service-tag mention; W18 abstains; ties
  FIFO at 0.000 on 8/8 cells.
* **R-65-CONFOUND** (W18-Λ-confound falsifier): round-2
  disambiguator mentions BOTH gold AND decoy; W18 abstains; ties
  FIFO at 0.000 on 8/8 cells.
* **R-65-DECEIVE** (W18-Λ-deceive falsifier): round-2 disambiguator
  mentions DECOY but NOT gold; W18 trusts its evidence and picks
  decoy; fails at 0.000 on 8/8 cells. Names the structural limit
  no closed-form bundle-relational scorer that trusts its evidence
  can escape (the named research move beyond it is W18-C-OUTSIDE,
  conjectural).
* **W18-3 backward-compat**: on R-58 default the W18 method ties
  W15 byte-for-byte on the answer field; on R-64-SYM the W18
  method partially recovers (only deadlock-flavored scenarios
  carry a relational mention). With ``enabled = False`` the W18
  method reduces to W15 byte-for-byte.
* **Audit T-1..T-7** OK on every capsule strategy of every cell of
  every regime.

The W18 surface is purely additive on top of the W15 surface (one
new dataclass + one tokeniser + one closed-form scorer + one
wrapping decoder). The SDK v3.18 runtime contract is byte-for-byte
unchanged. New tests cover the W18 unit semantics, the Phase-65
bench-property witnesses, the W18-1 strict-win anchor on loose
AND tight, the 5-seed stability, the three named falsifiers, the
backward-compat smoke, the token-budget honesty, and the
cross-regime synthetic summary.

See [`docs/RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md`](docs/RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md)
for the full milestone note.

## [3.18] — 2026-04-27 — SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem (first fresh-live end-to-end real-LLM strict +1.000 advance + first explicit symmetric-corroboration negative theorem) + W17 family

*Strictly additive on SDK v3.17. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The W17 surface adds
one new producer-prompt mode constant
(``PRODUCER_PROMPT_MAGNITUDE_HINTED``), one new dataclass
(``OperationalThreshold``), one new optional field on
``RoleExtractionSchema`` (``magnitude_thresholds``), one new
prompt-render helper (``_render_magnitude_hinted_prompt``), one
threshold-table data structure
(``INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS``), and an
optional ``magnitude_hinted=True`` parameter on
``incident_triage_role_schemas``. Re-exported via the SDK
``__init__``. The new
``vision_mvp/experiments/phase64_live_composition.py`` driver
ships as a research-slice addition with five synthetic + four
live + two symmetric-wall sub-banks.*

On the new Phase-64 *fresh-live end-to-end W14H + W15 composition
+ symmetric-corroboration wall* benchmark:

* **R-64-LIVE-MAGHINT** (fresh live ``qwen2.5:14b-32k`` Mac-1
  Ollama backend + magnitude-hinted prompt + ``T_decoder=14``;
  **W17-1 anchor**; n=8 × 24 producer calls; 0 endpoint
  failures; 128.2 s wall): pairing the W17 magnitude-hinted
  producer protocol with the W14 producer + W15 decoder pipeline
  achieves ``capsule_attention_aware = 1.000`` AND
  ``capsule_layered_fifo_packed = 0.000`` AND
  ``capsule_fifo = 0.000``. **+1.000 strict separation** on both
  axes. Bench property holds in **8/8** scenarios — closing the
  1/8 R-61-OLLAMA-A model-side judgment miss
  (``slow_query_archival``) that persisted under the SDK v3.15
  W14 anchor and the SDK v3.17 W16-Λ-real-replay anchor. The
  first programme result that beats the strongest non-composed
  baseline by ≥ 1.0 on a *fresh* live LLM probe.
* **R-64-LIVE-STRUCT** (fresh live 14B + legacy structured prompt;
  W17-Λ-no-hint anchor): bench property holds in 7/8;
  ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; +0.500 strict gain.
  Reproduces the W14-Λ-real / W16-Λ-real-replay envelope on a
  fresh probe; the magnitude-hint extension, not a re-run, is the
  load-bearing improvement.
* **R-64-LIVE-NAIVE** (fresh live 14B + naive prompt;
  W17-Λ-naive falsifier): bench property holds in 0/8; every
  capsule strategy ties FIFO at 0.000. Live counterpart of
  W14-Λ-prompt + W15-Λ-budget joint failure.
* **R-64-LIVE-XMODEL** (fresh live ``qwen3.5:35b`` MoE backend +
  magnitude-hinted prompt + ``think=False``; W17-C-XMODEL probe;
  n=8 × 24 producer calls; 0 failures; 92.0 s wall): bench
  property holds in **8/8** (the W17 magnitude-hint extension
  transfers byte-for-byte across the 14B → 36B-MoE jump on the
  bench-property axis); ``capsule_attention_aware = 0.750``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.750 strict gain**
  (well above the 0.50 strong-bar threshold). The 0.250 gap to
  1.000 is on ``accuracy_root_cause`` — a 35B-specific specific-
  tier kind judgment artifact, not a producer-protocol failure.
  Proved-conditional + empirical-research; partially discharges
  W16-C-CROSS-MODEL.
* **R-64-SYM** (synthetic ``build_phase64_sym_bank``;
  **W17-Λ-symmetric anchor**; n=8 × {None, 24} budget): every
  capsule strategy in the SDK ties FIFO at ``accuracy_full =
  0.000`` by construction. **The first explicit symmetric-
  corroboration limit theorem in the programme.** Discharges
  W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative theorem.

W17 theorem family: **W17-1** (proved-conditional + empirical-
research), **W17-Λ-no-hint** (empirical-research),
**W17-Λ-naive** (empirical-research), **W17-Λ-symmetric**
(proved-empirical + structural sketch), **W17-2** (proved by
inspection + mechanically-checked), **W17-3** (proved-empirical
full programme regression), **W17-C-XMODEL** (proved-conditional
+ empirical-research). The W17-C family (W17-C-DISAMBIGUATOR,
W17-C-LEARNED-HINT, W17-C-CROSS-BENCH) makes the next research
frontier explicit.

Discharged conjectures from prior SDKs:

* **W16-C-LIVE-OLLAMA** → DISCHARGED (W17-1 closes the 1/8 miss
  on a fresh live probe).
* **W16-C-CROSS-MODEL** → PARTIALLY DISCHARGED (W17-C-XMODEL on
  Ollama; MLX-distributed clause remains conjectural).
* **W15-C-SYMMETRIC** / **W16-C-SYMMETRIC** → DISCHARGED-NEGATIVE
  (W17-Λ-symmetric).

Backward-compat (W17-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``mode='naive'`` or ``mode='structured'`` AND
``magnitude_hinted_schema=False``, the W17 surface reduces to the
SDK v3.15 W14 anchor byte-for-byte. The CoordPy single-run product
runtime contract is byte-for-byte unchanged.

Added: ``vision_mvp/experiments/phase64_live_composition.py``,
``vision_mvp/tests/test_coordpy_phase64.py``, the W17 protocol
surface in ``vision_mvp/coordpy/team_coord.py``,
``docs/RESULTS_COORDPY_LIVE_COMPOSITION.md``, R-64 anchor +
bar 14 in ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
W17 family in ``docs/THEOREM_REGISTRY.md``, current-frontier
update in ``docs/RESEARCH_STATUS.md``, § 4.35 in
``docs/context_zero_master_plan.md``, latest-milestone pointer
in ``docs/START_HERE.md``, synthesis-after-v3.18 reading in
``papers/context_as_objects.md``, and the milestone artefacts
in ``docs/data/``
(``phase64_cross_regime_synthetic.json``,
``phase64_live_maghint_qwen2_5_14b_n8.json``,
``phase64_live_struct_qwen2_5_14b_n8.json``,
``phase64_live_naive_qwen2_5_14b_n8.json``,
``phase64_live_maghint_qwen3_5_35b_n8.json``).

Changed: ``vision_mvp/coordpy/__init__.py`` (SDK version bumped to
``coordpy.sdk.v3.18``; W17 surface re-exports);
``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``
(``CapturingOllamaExtractor.extract_round`` now treats
``PRODUCER_PROMPT_MAGNITUDE_HINTED`` as a structured-prompt
variant on the parser-dispatch axis — load-bearing fix because
the per-event mandate would otherwise lose its ``(kind, payload)``
dedup);
``vision_mvp/tests/test_coordpy_producer_ambiguity.py``
(``test_all_modes_listed`` updated for the additive third mode).

## [3.17] — 2026-04-27 — SDK v3.17 — end-to-end W14 + W15 composition (first end-to-end real-LLM strict advance over the strongest non-composed baseline) + W16 family

*Strictly additive on SDK v3.16. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. **No new SDK class
ships** — W16 is the demonstration that the existing W14
``StructuredProducerProtocol`` and W15
``AttentionAwareBundleDecoder`` compose on a single regime where
both are independently load-bearing. The new
``vision_mvp/experiments/phase63_composed_real_llm.py`` driver
plus the ``OllamaReplayExtractor`` ship as research-slice
additions; the runtime contract is unchanged.

On the new Phase-63 *composed end-to-end W14 + W15* benchmark:

* **R-63-COMPOSED-TIGHT** (mag-filter structured, ``T_decoder=24``;
  **W16-1 anchor**, n=8 × 5 seeds saturated): pairing the W14
  ``StructuredProducerProtocol`` + W15
  ``AttentionAwareBundleDecoder`` simultaneously achieves
  ``accuracy_full = 1.000`` while every non-composed baseline
  collapses to 0.000 (W14-only-budgeted, W15-only-without-W14,
  substrate FIFO). **+1.000 strict separation, stable across 5/5
  alternate ``bank_seed`` values**.
* **R-63-naive-tight** (mag-filter naive, ``T_decoder=24``;
  **W16-Λ-compose anchor**): joint failure of W14-Λ-prompt and
  W15-Λ-budget on the same regime; every capsule strategy ties
  FIFO at 0.000.
* **R-63-degen-budget** (mag-filter structured, ``T_decoder=2``;
  **W16-Λ-degenerate falsifier**): both packers collapse to
  0.000; the W16-1 win is conditional on a budget that admits
  *some* of the union.
* **R-63-OLLAMA-REPLAY-COMPOSED-TIGHT** (replay over recorded
  Phase-61 ``qwen2.5:14b-32k`` bytes at ``T_decoder=14,
  K_auditor=8``; **W16-Λ-real-replay anchor**):
  ``capsule_attention_aware = 0.500`` while
  ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict
  gain** over the FIFO-packed-W14-only baseline on a *real-LLM
  stream*. The first end-to-end real-LLM strict advance over the
  strongest non-composed baseline in the programme.

The W16 layer is the eighth structural move in the CoordPy programme:

| Layer                            | SDK   | Theorem | Anchor regime                                |
|----------------------------------|-------|---------|----------------------------------------------|
| Admission (cohort coherence)     | v3.8  | W7-2    | R-54                                         |
| Admission (cross-role corrob.)   | v3.9  | W8-1    | R-55                                         |
| Admission (multi-service)        | v3.10 | W9-1    | R-56                                         |
| Decoding (intra-round bundle)    | v3.11 | W10-1   | R-57                                         |
| Decoding (cross-round bundle)    | v3.12 | W11-1   | R-58                                         |
| Normalisation (fixed-vocabulary) | v3.13 | W12-1   | R-59                                         |
| Normalisation (open-world)       | v3.14 | W13-1   | R-60-wide                                    |
| Producer protocol                | v3.15 | W14-1   | R-61 + R-61-OLLAMA-A                         |
| Decoder context packing          | v3.16 | W15-1   | R-62-tightbudget                             |
| **End-to-end composition**       | v3.17 | **W16-1** | **R-63-COMPOSED-TIGHT + W16-Λ-real-replay** |

W16 family theorems minted by this milestone:

* **W16-Λ-compose** (proved-empirical + structural sketch via
  composition of W14-Λ-prompt and W15-Λ-budget).
* **W16-1** (proved-conditional + proved-empirical synthetic n=40 ×
  5 seeds).
* **W16-2** (proved-empirical, multiplicative composition).
* **W16-3** (proved-empirical, full programme regression: **442/442
  prior tests pass byte-for-byte**; 22 new tests cover the W16
  surface).
* **W16-Λ-degenerate** (proved-empirical falsifier).
* **W16-Λ-real-replay** (empirical-research, recorded real-LLM
  bytes).

W15-C-COMPOSE-W14 (SDK v3.16 conjecture) is **PARTIALLY DISCHARGED**
by W16-Λ-real-replay: the composition delivers a strict +0.500 gain
on recorded ``qwen2.5:14b-32k`` bytes (recovering the W14-only
loose-budget accuracy under tight budget pressure) but does NOT
close the 1/8 model-side judgment failure that W14 alone leaves
on the same recorded capture.

Honest scope: the Mac-1 endpoint at 192.168.12.191:11434 was
offline at milestone capture time (``HTTP=000``). A fresh live
LLM probe (W16-C-LIVE-OLLAMA) is conjectural; cross-model
transfer (W16-C-CROSS-MODEL) and cross-bench transfer (W16-C1)
are conjectural. See
[`docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md`](docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md)
for the full milestone note.

## [3.16] — 2026-04-27 — SDK v3.16 — attention-aware capsule context packing (first decoder-side context-packing move + first joint-correctness-and-context-efficiency strict-gain anchor) + W15 family

*Strictly additive on SDK v3.15. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``AttentionAwareBundleDecoder``, ``CapsuleContextPacker``, and
``FifoContextPacker`` ship in ``vision_mvp.coordpy.team_coord`` as
research-slice additions to the multi-agent coordination layer, not
part of the run-boundary product runtime. **First SDK milestone
where joint correctness AND decoder-side context efficiency is the
strict-gain axis** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (the new bar 12).

On the new Phase-62 *attention-aware capsule context packing* benchmark:

* **R-62-default** (synthetic, ``T_decoder=None``): bench property
  (multi-hypothesis comparable-magnitude + asymmetric corroboration:
  every decoy ≥ 2 distinct roles, every gold = 1 distinct role)
  holds 8/8; W11/W12/W13/W15 cross-round decoders all achieve
  ``accuracy_full = 1.000``. The W15 decoder ties W13 byte-for-byte
  on the answer field (W15-3 backward-compat anchor).
* **R-62-tightbudget** (synthetic, ``T_decoder=24`` strict; W15-1
  anchor): under FIFO packing the round-2 specific-tier
  disambiguator falls past the budget by construction
  (``position_of_first_causal_claim_avg = -1.0`` in 8/8 cells);
  ``capsule_layered_fifo_packed`` ties FIFO at
  ``accuracy_full = 0.000``. The W15 ``AttentionAwareBundleDecoder``
  salience-packs the union with hypothesis preservation and puts
  the round-2 specific claim at rank 0 in 8/8 cells; achieves
  ``accuracy_full = 1.000``. **+1.000 strict separation vs FIFO-
  packed-W13, stable across 5/5 alternate ``bank_seed`` values**.
* **R-62-saturation** (synthetic, ``T_decoder=9999`` effectively
  unbounded; W15-Λ-budget falsifier): under no decoder-side budget
  pressure both packers tie at 1.000. The W15-1 win is *conditional*
  on budget pressure — this regime makes the conditionality sharp.

The W15 layer is the seventh structural axis of the CoordPy programme:

| Layer                                | SDK   | Theorem | Anchor regime    |
|--------------------------------------|-------|---------|------------------|
| Admission (cohort coherence)         | v3.8  | W7-2    | R-54             |
| Admission (cross-role corrob.)       | v3.9  | W8-1    | R-55             |
| Admission (multi-service)            | v3.10 | W9-1    | R-56             |
| Decoding (intra-round bundle)        | v3.11 | W10-1   | R-57             |
| Decoding (cross-round bundle)        | v3.12 | W11-1   | R-58             |
| Normalisation (fixed-vocabulary)     | v3.13 | W12-1   | R-59             |
| Normalisation (open-world)           | v3.14 | W13-1   | R-60-wide        |
| Producer protocol                    | v3.15 | W14-1   | R-61 + R-61-OLLAMA-A |
| **Decoder context packing**          | v3.16 | **W15-1** | **R-62-tightbudget** |

The W14 layer (SDK v3.15) closed the producer-side gap; SDK v3.16
attacks the symmetric *downstream* gap directly. The W15 packer's
salience score is closed-form deterministic (tier + CCK +
corroboration + magnitude + round) with pre-committed weight
defaults; per-(tag, role, tier) hypothesis preservation guarantees
multi-hypothesis multi-role evidence survives the pack so the W11
contradiction-aware drop fires correctly.

**Token / context / attention measurement (Part E of the milestone
brief).** Pack-stats expose ``position_of_first_causal_claim`` (the
proxy attention metric — rank 0 in 8/8 W15 cells, −1 in 8/8 FIFO-
pack cells), ``tokens_kept_sum`` / ``tokens_input_sum`` (84.6 % vs
87.3 %), ``hypothesis_count_kept`` (4/4 in both packers), and
``n_dropped_budget`` for direct audit. Token reduction is not the
goal — *causal-evidence concentration in early prompt positions* is.
The proxy attention metric is auditable; we do NOT claim transformer
attention manipulation. The W15-1 win is conditional on (a) the
bench property holding, (b) ``T_decoder`` < admitted-union token
sum, AND (c) round-2 carrying a specific-tier disambiguator with
no ``service=`` token; W15-Λ-degenerate makes the conditionality
sharp.

Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 / R-59 /
R-60 / R-61 (default + falsifier). 393/393 prior coordpy tests pass
byte-for-byte; 37 new tests cover the W15 surface, hypothesis
preservation, FIFO packer, backward-compat with W13, Phase-62 bank
shape, default config (W15-1 anchor), 5-seed stability, and cross-
regime separation. The coordpy suite totals 430/430 passing.

Honest scope: SDK v3.16 is a *synthetic* milestone — the producer
is the deterministic ``IdentityExtractor``, not a real LLM. Real-LLM
transfer of W15 is W15-C-real, conjectural; it requires Mac 1 or
Mac 2 to be online and the bundle to be re-decoded by an LLM agent
under a real context window. SDK v3.16 does not run this probe.
"Attention-aware" uses an *honest proxy* metric — the
``position_of_first_causal_claim`` rank in the salience-ordered
pack — not transformer attention manipulation. Composition with
W14 on a real-Ollama stream (W15-C-COMPOSE-W14, conjectural) is
the natural next probe.

Public surface (additive):

* :class:`vision_mvp.coordpy.team_coord.AttentionAwareBundleDecoder` —
  two-stage decoder: first-pass priority decode → salience-aware
  repack → final W13 layered decode.
* :class:`vision_mvp.coordpy.team_coord.CapsuleContextPacker` — closed-
  form salience pack with hypothesis preservation.
* :class:`vision_mvp.coordpy.team_coord.FifoContextPacker` — load-
  bearing baseline (FIFO truncation under the same ``T_decoder``).
* :class:`vision_mvp.coordpy.team_coord.W15PackResult`,
  :class:`vision_mvp.coordpy.team_coord.W15PackedHandoff` — pack-stats
  surface.
* ``W15_DEFAULT_TIER_WEIGHT``, ``W15_DEFAULT_CCK_WEIGHT``,
  ``W15_DEFAULT_CORROBORATION_WEIGHT``,
  ``W15_DEFAULT_MAGNITUDE_WEIGHT``,
  ``W15_DEFAULT_ROUND_WEIGHT`` — pre-committed salience weights.

New experiment driver:
``vision_mvp.experiments.phase62_attention_aware_packing``.

Cross-references:
* Bench: ``vision_mvp/experiments/phase62_attention_aware_packing.py``
* Method: ``vision_mvp/coordpy/team_coord.py``
* Tests: ``vision_mvp/tests/test_coordpy_attention_aware.py`` (37 tests)
* Milestone note: ``docs/RESULTS_COORDPY_ATTENTION_AWARE.md``
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-62 anchor + bar 12 + § 2.11)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W15 family)
* Master plan: ``docs/context_zero_master_plan.md`` § 4.33
* Data: ``docs/data/phase62_*.json``

## [3.15] — 2026-04-27 — SDK v3.15 — structured producer protocol (first producer-protocol move + first real-LLM strict gain ≥ 0.50 over substrate FIFO) + W14 family

*Strictly additive on SDK v3.14. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``StructuredProducerProtocol`` + ``RoleExtractionSchema`` ship in
``vision_mvp.coordpy.team_coord`` as research-slice additions to the
multi-agent coordination layer, not part of the run-boundary
product runtime. **First SDK milestone to clear the R-61-OLLAMA-A
tier of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.5** —
the **first capsule-native multi-agent coordination method that
strictly improves over substrate FIFO on a real-LLM stream by ≥
0.50 pooled accuracy_full in the programme**.

On the new Phase-61 *producer-side ambiguity-preservation* benchmark:

* **R-61-default** (synthetic, identity extractor, naive prompt):
  bench property holds 8/8; W11/W12/W13 cross-round decoders all
  achieve ``accuracy_full = 1.000``. Sanity anchor.
* **R-61-naive-prompt** (synthetic, magnitude-filter, naive prompt;
  W14-Λ-prompt anchor): the magnitude-filter extractor calibrated
  against the W13-Λ-real real-Ollama observation collapses round-1
  by top-N-per-(role, kind) by magnitude; bench property holds 0/8;
  every capsule strategy ties FIFO at 0.000. The synthetic
  counterpart of W13-Λ-real, mechanically tractable in CI.
* **R-61-structured-prompt** (synthetic, magnitude-filter,
  structured prompt; W14-1 anchor): the structured prompt's per-
  event mandate disables the compression; bench property holds 8/8;
  W11/W12/W13 achieve ``accuracy_full = 1.000``; +1.000 strict
  separation vs naive-prompt counterpart, stable across **5/5**
  alternate ``bank_seed`` values.
* **R-61-ollama-naive** (real Mac-1 ``qwen2.5:14b-32k`` at
  ``temperature=0`` on the redesigned events under the *naive*
  prompt; W14-Λ-real-naive falsifier): bench property holds 0/8;
  every method ties FIFO at 0.000 — the W14-Λ-prompt prediction
  *empirically confirmed* on real Ollama.
* **R-61-ollama-structured** (real Mac-1 ``qwen2.5:14b-32k`` at
  ``temperature=0`` on the redesigned events under the *structured*
  prompt; W14-Λ-real anchor at the R-61-OLLAMA-A tier): bench
  property holds **7/8**; W11/W12/W13 cross-round decoders all
  achieve ``accuracy_full = 0.500``; ``layered − fifo = +0.500`` at
  exactly the 0.50 threshold; audit T-1..T-7 preserved on every
  cell. n_eval=8 × 24 producer calls, 0 endpoint failures, 111.4 s
  wall on Mac 1.

The W13 closure-widening (SDK v3.14) is structurally invisible on
R-61-ollama because the real LLM emits canonical kinds (zero kind
drift); the load-bearing layer on this regime is the W14 producer
protocol, not the W13 normaliser. The W13 layer is dormant on this
regime, not refuted; it remains the load-bearing layer on
R-60-wide.

Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 / R-59 /
R-60 (default + falsifier). 393/393 prior coordpy tests pass byte-for-
byte. Named falsifier (W14-4: real Ollama + comparable-magnitude
events + naive prompt) ties FIFO at 0.000 on 8/8 — *both* the
event redesign AND the structured prompt are required for W14-1.

**Files added.**

* ``vision_mvp/coordpy/team_coord.py`` — adds
  ``RoleExtractionSchema``, ``ProducerPromptResult``,
  ``StructuredProducerProtocol``, ``PRODUCER_PROMPT_NAIVE``,
  ``PRODUCER_PROMPT_STRUCTURED``, ``ALL_PRODUCER_PROMPT_MODES``,
  ``INCIDENT_TRIAGE_OBSERVATION_KINDS``,
  ``incident_triage_role_schemas``.
* ``vision_mvp/coordpy/__init__.py`` — re-exports the W14 surface;
  bumps ``SDK_VERSION = "coordpy.sdk.v3.15"``.
* ``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``
  — new benchmark.
* ``vision_mvp/tests/test_coordpy_producer_ambiguity.py`` — 27 new
  tests across schema soundness, protocol determinism, magnitude-
  filter calibration, Phase-61 default config (W14-Λ-prompt +
  W14-1), 5-seed stability, and cross-regime separation.
* ``docs/data/phase61_default_K8_n8.json``,
  ``docs/data/phase61_naive_prompt_K8_n8.json``,
  ``docs/data/phase61_structured_prompt_K8_n8.json``,
  ``docs/data/phase61_seed_sweep_naive_K8_n8.json``,
  ``docs/data/phase61_seed_sweep_structured_K8_n8.json``,
  ``docs/data/phase61_cross_regime.json``,
  ``docs/data/phase61_cross_regime_full.json``,
  ``docs/data/phase61_real_ollama_naive_qwen2_5_14b_n4.json``,
  ``docs/data/phase61_real_ollama_naive_qwen2_5_14b_n8.json``,
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n4.json``,
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``.
* ``docs/RESULTS_COORDPY_PRODUCER_AMBIGUITY.md`` — milestone results
  note (theory-forward, six-layer thesis, W14 theorem family, W14-C
  conjectures).

**Files updated.**

* ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` — R-61 anchor +
  bar 11 (producer-side ambiguity-preservation split) + § 1.5
  R-61-OLLAMA 4-tier grading + § 2.10 R-61 ingredients.
* ``docs/RESEARCH_STATUS.md`` — eleven-axis status, W14 active
  moves / observations / conjectures.
* ``docs/THEOREM_REGISTRY.md`` — W14 family (Λ-prompt, 1, 2, 3, 4,
  Λ-real) + W14-C conjectures + W13-C3 partial-discharge update.
* ``docs/HOW_NOT_TO_OVERSTATE.md`` — W14 framing rules ("solved
  real-LLM transfer" forbidden, "W14 makes W13 obsolete"
  forbidden).
* ``docs/context_zero_master_plan.md`` — § 4.32 SDK v3.15 + post-
  v3.15 master-plan reading + six-layer transfer story.
* ``docs/START_HERE.md`` — current-milestone pointer + SDK v3.15
  TL;DR paragraph.

## [3.13] — 2026-04-26 — SDK v3.13 — real-LLM-robust multi-round bundle decoder (first synthetic→real-LLM transfer move) + W12 family

*Strictly additive on SDK v3.12. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``RobustMultiRoundBundleDecoder`` and the closed-vocabulary
``CLAIM_KIND_SYNONYMS`` table are research-slice additions to the
multi-agent coordination layer (``vision_mvp.coordpy.team_coord``),
not part of the run-boundary product runtime. **First SDK milestone
to clear the strong success bar of
`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 (R-59 anchor
+ new bar 9 — synthetic→real-LLM transfer split)** — the **first
capsule-native multi-agent coordination method that survives a
calibrated bounded-producer-noise channel**. On the new Phase-59
*real-LLM-driven multi-round delayed-disambiguation* regime
(``synonym_prob=0.50, svc_token_alt_prob=0.30`` synthetic-noisy-LLM
extractor), every un-normalised capsule strategy in the SDK —
substrate, FIFO, priority, coverage, W7-2 cohort, W8 corroboration,
W9 multi-service, W10 single-round bundle decoder, AND **SDK v3.12
W11 ``MultiRoundBundleDecoder``** — ties FIFO at
``accuracy_full = 0.000`` (the W12-Λ un-normalised structural
limit at the real-LLM axis); the new
``RobustMultiRoundBundleDecoder`` (closed-vocabulary
``CLAIM_KIND_SYNONYMS`` + payload-rewrite layer ahead of W11)
achieves ``accuracy_full = 1.000`` (W12-1 sufficiency under
bounded LLM noise). Headline gap = +1.000 vs every un-normalised
method including W11; ``robust = 1.000`` on **5/5** alternate
(bank_seed, llm_seed) values, ``robust − w11`` min = 0.750
(seed 23), max = 1.000 (seeds 11, 29), well above the strong-bar
0.20 threshold on every seed.
Backward-compatible on R-54 / R-55 / R-56 / R-57 / R-58 and on
R-59 with ``llm_mode='synthetic_clean_llm'`` (rewrite counters =
0, ties W11 byte-for-byte). Named falsifier (W12-4: out-of-
vocabulary kinds outside the synonym closure) ties FIFO at 0.000.
Audit T-1..T-7 OK on every cell of every R-59 capsule strategy.

**Files added.**

* ``vision_mvp/coordpy/team_coord.py`` — adds
  ``RobustMultiRoundBundleDecoder``, ``CLAIM_KIND_SYNONYMS``,
  ``_SERVICE_TAG_REWRITES``, ``normalize_claim_kind``,
  ``normalize_payload``, ``normalize_handoff``.
* ``vision_mvp/coordpy/__init__.py`` — re-exports the W12 surface;
  bumps ``SDK_VERSION = "coordpy.sdk.v3.13"``.
* ``vision_mvp/experiments/phase59_real_llm_multi_round.py`` — new
  benchmark.
* ``vision_mvp/tests/test_coordpy_real_llm_multi_round.py`` — 24 new
  tests across normalisation, decoder semantics, bench property,
  default config (W12-Λ + W12-1), falsifier (W12-4), backward-compat
  (W12-3), and 5-seed stability.
* ``docs/data/phase59_default_K8_n12.json``,
  ``docs/data/phase59_falsifier_K8_n8.json``,
  ``docs/data/phase59_clean_K8_n8.json``,
  ``docs/data/phase59_seed_sweep_K8_n12.json``,
  ``docs/data/phase59_cross_regime.json``.
* ``docs/RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md`` — milestone note.
* ``docs/RESEARCH_STATUS.md``, ``docs/THEOREM_REGISTRY.md``,
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``,
  ``docs/HOW_NOT_TO_OVERSTATE.md``,
  ``docs/context_zero_master_plan.md``,
  ``docs/START_HERE.md``,
  ``papers/coordpy_capsule_native_runtime.md``,
  ``CHANGELOG.md`` updated for the W12 family.

**Honest scope.** The win is *conditional* on (a) the R-58
delayed-causal-evidence bench shape, (b) the producer-noise
channel being bounded by the closed-vocabulary closure (every
variant in ``NOISY_KIND_VARIANTS`` is in ``CLAIM_KIND_SYNONYMS``),
AND (c) round-N admission not being budget-starved (inherits
W11-4). The synthetic-noisy-LLM extractor is calibrated against
Phase-53 14B/35B empirical kind-drift distributions; the
``--llm-mode ollama`` opt-in mode is the W12-C2 next data point
and the natural Mac-2-returns probe.

## [3.11] — 2026-04-26 — SDK v3.11 — bundle-aware team decoder (first decoder-side coordination move) + W10 family

*Strictly additive on SDK v3.10. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``BundleAwareTeamDecoder`` and the closed-vocabulary
``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` table are research-slice
additions to the multi-agent coordination layer
(``vision_mvp.coordpy.team_coord``), not part of the run-boundary
product runtime. **Third consecutive SDK milestone to clear the
strong success bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (R-57 anchor + new bar 7 — admission/decoding split)** — the
**first capsule-native multi-agent coordination method that
crosses the admission/decoding split**. On the new Phase-57
*decoder-forcing* regime, every service-blind admission policy in
the SDK (FIFO, priority, coverage, W7-2, W8, W9) ties FIFO at
``accuracy_full = 0.000`` (the W10-Λ admission-only structural
limit); pairing W9 admission with the new bundle decoder achieves
``accuracy_full = 1.000`` (W10-1 sufficiency). Headline gap =
**+1.000** vs every admission baseline, stable across **5/5**
alternate bank seeds. Backward-compatible at 1.000 on R-54 / R-55
/ R-56 (W10-3 via the trust-admission fallback). Named falsifier
regime W10-4 (decoy CCK-promotion) correctly ties FIFO at 0.000.*

### Added

- **Phase-57 multi-service-gold + corroborated-decoy decoder-forcing
  benchmark** (new):
  `vision_mvp/experiments/phase57_decoder_forcing.py`. Eight
  strategies including the new ``capsule_bundle_decoder`` (W10-AD:
  W9 admission + bundle decoder) on a deterministic 12-scenario
  bank where every scenario has multi-service gold corroborated via
  causal claim_kinds AND a decoy corroborated via *non-causal*
  generic-noise claim_kinds. 10/10 default scenarios satisfy the
  bench property; mechanically verified by ``Phase57BankShapeTests``.
- **`BundleAwareTeamDecoder` (new)**:
  `vision_mvp/coordpy/team_coord.py`. Three knobs (``cck_filter``,
  ``role_corroboration_floor``,
  ``fallback_admitted_size_threshold``). Re-exported as
  ``BundleAwareTeamDecoder``, ``decode_admitted_role_view``, and
  ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE``.
- **Theorem family W10**: W10-Λ, W10-1, W10-2, W10-3, W10-4
  (proved or proved-empirical); W10-C1/C2/C3 (conjectural).
- **R-57 anchor + bar 7** added to
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`.
- **`docs/RESULTS_COORDPY_BUNDLE_DECODER.md`** (new): theory-forward
  results note.
- 21 new tests in `vision_mvp/tests/test_coordpy_bundle_decoder.py`.

### Changed

- ``SDK_VERSION = "coordpy.sdk.v3.11"``.
- ``docs/RESEARCH_STATUS.md``, ``docs/THEOREM_REGISTRY.md``,
  ``docs/context_zero_master_plan.md`` § 4.28 updated for SDK v3.11
  / W10 family.

### Discharged conjectures

- **W9-C1** (SDK v3.10): bundle-aware decoder companion.
  **DISCHARGED-empirical** by W10-1 on the Phase-57 decoder-forcing
  regime.

## [3.10] — 2026-04-26 — SDK v3.10 — multi-service top-K cross-role corroboration multi-agent coordination + W9 family

*Strictly additive on SDK v3.9. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``MultiServiceCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.coordpy.team_coord``), not part of the run-boundary
product runtime. **Second consecutive SDK milestone to clear the
strong success bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (R-56 anchor)** — strict separation from W8 on Phase 56
(+1.000 multi_service − corroboration on accuracy_full,
+1.000 vs FIFO and W7-2 too), backward-compat on Phase 55
(W9 ties W8 at 1.000 via the argmax-by-role-count gate),
backward-compat on Phase 54 (W9 ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800), cross-bank stability
across 5/5 seeds, named falsifier regime (W9-4) correctly ties
FIFO at 0.000.  **First programme result whose strict-gain regime
is not solvable by the previous SDK's strongest method.***

### Added

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase56_multi_service_corroboration.py`.
  Smallest deterministic regime where (a) every scenario has
  `gold_services` of size 2 (multi-service incident), (b) both gold
  services are corroborated by ≥ 2 distinct producer roles, (c) at
  least one decoy service has raw plurality but is corroborated by
  exactly 1 producer role. 5 base scenario builders × 2 replicates
  → 10-scenario default bank; named falsifier bank promotes a
  decoy to ≥ 2 distinct producer roles (W9-4 anchor).
- **`MultiServiceCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/coordpy/team_coord.py`. Deterministic, training-free
  admission rule that admits the **top-K cross-role-corroborated
  tier** (default `top_k=2, min_corroborated_roles=2`) via the
  argmax-by-role-count gate. Strictly generalises the SDK v3.9
  W8 single-tag corroboration policy (W9-3 backward-compat).
  Buffered factory `from_candidate_stream` is the W9-1 anchor.
  Re-exported as `TeamMultiServiceCorroborationAdmissionPolicy`.
- **`_dominant_tag_set`** helper (new): pure function with three
  structural properties (W9-2): single-role exclusion;
  argmax-tier collapse; argmax-tier multi-tag admission within
  `top_k` cap.
- **W9 theorem family** (new): W9-1 strict separation, W9-2
  argmax-tier strict-ordering, W9-3 backward-compat with W8
  + W7-2, W9-4 decoy-corroboration falsifier — all proved or
  proved-empirical on the pre-committed Phase-56 default. W9-C1
  / W9-C2 / W9-C3 conjectures (bundle-aware decoder, |gold|≥3,
  real-LLM transfer).
- **36 contract tests** in `test_coordpy_multi_service_corroboration.py`:
  policy unit tests, bank shape, default config win, seed stability,
  falsifier behaviour, W9-3 backward-compat with Phase 55, audit
  invariance, cross-regime contract, public-API contract.
- **`docs/RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md`** (new):
  milestone results note with W9 family theorem statements.
- **Frozen artefacts** in `docs/data/`: `phase56_multi_service_K4_n10.json`,
  `phase56_falsifier_K4_n10.json`, `phase56_seed_sweep.json`,
  `phase56_cross_regime.json`,
  `phase53_synthetic_w9_regression_check.json`.

### Changed

- **`SDK_VERSION`** bumped from `coordpy.sdk.v3.9` to `coordpy.sdk.v3.10`.
- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** — bar anchor
  advanced to R-56; R-56 named regime added with mechanical-witness
  ingredient list; falsifying-failure list extended to gate W8-1
  contract test; canonical phrasing for SDK v3.10 added.
- **`docs/THEOREM_REGISTRY.md`** — W9-1/W9-2/W9-3/W9-4 + W9-C1/C2/C3
  added; W8-C1 marked DISCHARGED; date stamp v3.10.
- **`docs/RESEARCH_STATUS.md`** — ninth research axis (multi-service
  top-K corroboration) added; SDK v3.10 frontier section.
- **`docs/HOW_NOT_TO_OVERSTATE.md`** — W9 overstatement guards
  added (W9-1 conditionality, W8 multi-service-gold falsifier
  named, "we solved multi-agent context" still forbidden).
- **`docs/context_zero_master_plan.md`** — § 4.27 added (SDK v3.10
  milestone summary + post-v3.10 reading).
- **`docs/START_HERE.md`** — SDK v3.10 paragraph + W9 family summary;
  links to milestone result + success bar updated.

### Preserved

- **CoordPy single-run product runtime contract.** Byte-for-byte
  unchanged from SDK v3.9. The Phase-45 product report schema
  (`PRODUCT_REPORT_SCHEMA = "phase45.product_report.v2"`) is
  unchanged.
- **SDK v3.5–v3.9 multi-agent surface.** Every fixed admission
  policy from previous SDKs (FIFO, priority, coverage,
  cohort_coherence, cross_role_corroboration) is unchanged; W7-2
  and W8-1 contract tests still pass byte-for-byte. The new W9
  policy is purely additive.
- **Lifecycle audit (T-1..T-7).** Holds on every cell of every
  regime (R-53 / R-54 / R-55 / R-56 default / R-56 falsifier).

## [3.9] — 2026-04-26 — SDK v3.9 — cross-role corroboration multi-agent coordination + W8 family

*Strictly additive on SDK v3.8. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``CrossRoleCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.coordpy.team_coord``), not part of the run-boundary
product runtime. **First SDK milestone to clear the strong success
bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1** —
strict separation from W7-2 on Phase 55 (+1.000 corroboration −
cohort_buffered, +1.000 corroboration − fifo on accuracy_full),
backward-compat on Phase 54 (corroboration ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800) or 14B real-LLM
(0.800), cross-bank stability across 5/5 seeds, named falsifier
regime correctly ties FIFO at 0.000.*

### Added

- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** (new): pre-
  committed strong / partial / null success bars with named
  regimes (R-53 / R-54 / R-55).
- **Phase-55 decoy-plurality + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase55_decoy_plurality.py`.
  Smallest deterministic regime where (a) decoy raw plurality
  breaks W7-2 single-tag plurality cohort coherence AND (b) gold
  cross-role corroboration provides a relational signal a (role,
  tag)-aggregating policy can exploit. Bench properties named and
  mechanically verified.
- **`CrossRoleCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/coordpy/team_coord.py`. Deterministic, training-free
  admission rule with score function `role_weight·|distinct_roles|
  + |raw_mentions|`. Buffered factory `from_candidate_stream` is
  the W8-1 anchor. Re-exported as
  `TeamCrossRoleCorroborationAdmissionPolicy`.
- **W8 theorem family**: W8-1 (strict separation, proved-empirical
  n=50), W8-2 (score-function strict ordering, proved structural),
  W8-3 (backward-compat with W7-2 on Phase 54, proved-empirical),
  W8-4 (decoy-corroboration falsifier, proved-empirical n=10).
  W8-C1 / W8-C2 / W8-C3 conjectures.
- **34 contract tests**:
  `vision_mvp/tests/test_coordpy_cross_role_corroboration.py`.
- **Frozen reproducibility artefacts**:
  `docs/data/phase55_decoy_plurality_K4_n10.json` (default),
  `docs/data/phase55_falsifier_K4_n10.json` (W8-4),
  `docs/data/phase55_budget_sweep.json`,
  `docs/data/phase55_seed_sweep.json`,
  `docs/data/phase55_cross_regime.json`,
  `docs/data/phase53_real_llm_corroboration_check.json`.

### Changed

- `vision_mvp/coordpy/__init__.py`: re-exports
  `TeamCrossRoleCorroborationAdmissionPolicy`; `SDK_VERSION`
  bumped to `"coordpy.sdk.v3.9"`.
- `vision_mvp/tests/test_coordpy_public_api.py`: SDK version test
  updated to v3.9; new corroboration export test.
- `docs/THEOREM_REGISTRY.md`: W8 family rows added; date stamp
  v3.9.
- `docs/RESEARCH_STATUS.md`: eighth research axis added.
- `docs/HOW_NOT_TO_OVERSTATE.md`: W8 overstatement guards added
  (W8-1 conditionality; "we solved multi-agent context" forbidden
  without naming the strong success bar; Phase-54/55 conflation
  forbidden; Phase-53/55 conflation forbidden).
- `docs/context_zero_master_plan.md`: § 4.26 SDK v3.9 added.
- `docs/START_HERE.md`: SDK v3.9 paragraph + canonical-reading
  pointer to the success-criterion doc.
- `docs/RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md`: new milestone
  results note.

### Honest scope

- The W8-1 win is **conditional** on the named bench property
  (decoy-plurality + cross-role-corroborated gold). The W8-4
  falsifier regime is the explicit named counterexample.
- Three named regimes is a stronger cross-regime result than two,
  but not "all regimes." Real production multi-agent teams have
  additional axes the W8 family does not test (heterogeneous
  producers, time-varying budgets, multi-round handoffs,
  multi-service gold). W8-C1 / W8-C2 / W8-C3 are conjectural;
  none yet shipped.
- The CoordPy single-run product runtime contract is byte-for-byte
  unchanged from SDK v3.8. The new admission policy is a
  research-slice addition.

## [3.8] — 2026-04-26 — SDK v3.8 — cross-role cohort-coherence multi-agent coordination + W7 family

*Strictly additive on SDK v3.7. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new
``CohortCoherenceAdmissionPolicy`` is a research-slice addition
to the multi-agent coordination layer
(``vision_mvp.coordpy.team_coord``), not part of the run-boundary
product runtime.*

### Added

- **Phase-54 cross-role cohort-coherence benchmark** (new):
  ``vision_mvp/experiments/phase54_cross_role_coherence.py``.
  Smallest deterministic multi-agent benchmark where cross-role
  coordination provides a strict structural advantage over
  substrate FIFO. Bench properties (gold-plurality, cross-role,
  budget-bound, decoder-pollution) are *named and mechanically
  verified* by the contract tests. Runs end-to-end without any
  LLM in the loop.
- **``CohortCoherenceAdmissionPolicy``** (new): in
  ``vision_mvp/coordpy/team_coord.py``. Deterministic, training-free,
  interpretable cross-role admission rule. Two sub-modes:
  *streaming* (running cohort over already-admitted; arrival-
  order-sensitive) and *buffered* (pre-fitted plurality from the
  full candidate stream's payloads via
  ``from_candidate_payloads``; arrival-order-stable). Re-exported
  as ``TeamCohortCoherenceAdmissionPolicy``.
- **W7 theorem family**: W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort_buffered structural win under
  gold-plurality, proved-empirical, n=50 saturated, 5/5 stable
  across bank seeds), W7-2-conditional (K-sweep window,
  proved-empirical), W7-3 (extraction floor, proved-negative,
  Capsule Contract C5 corollary). W7-C1/C2/C3 conjectures cover
  multi-service-gold, decoder-side coordination, and real-LLM
  transfer extensions.
- **21 contract tests** for the new policy + bench:
  ``vision_mvp/tests/test_coordpy_cross_role_coherence.py``.
- **Frozen reproducibility artefacts**:
  ``docs/data/phase54_cross_role_coherence_K4_n10.json`` (default
  config),
  ``docs/data/phase54_cross_role_coherence_budget_sweep.json``
  (K-sweep).
- **Milestone results note**:
  ``docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md``.

### Changed

- ``SDK_VERSION`` bumped to ``"coordpy.sdk.v3.8"``.
- ``vision_mvp/coordpy/__init__.py``: re-exports
  ``TeamCohortCoherenceAdmissionPolicy``.
- ``vision_mvp/coordpy/team_coord.py``:
  ``ALL_FIXED_POLICY_NAMES`` extended with
  ``"cohort_coherence"``; new helper ``_candidate_service_tag``.
- ``docs/THEOREM_REGISTRY.md``: W7-1 / W7-1-aux / W7-2 /
  W7-2-conditional / W7-3 / W7-C1 / W7-C2 / W7-C3 rows added.
- ``docs/RESEARCH_STATUS.md``: seventh research axis added (W7
  family); now lists 7 coupled axes.
- ``docs/HOW_NOT_TO_OVERSTATE.md``: W7-overstatement guards
  added (cohort-coherence wins are *conditional* on bench
  properties; SDK v3.7 and SDK v3.8 results are both true,
  conditioned on different bench properties; *buffered* vs
  *streaming* distinction must be specified).
- ``docs/context_zero_master_plan.md``: § 4.25 added.
- ``docs/START_HERE.md``: SDK v3.8 paragraph added.

### Honest scope

- **The W7-2 win is conditional.** It depends on the bench having
  gold-plurality + foreign-service decoys + ``|candidates| >
  K_auditor``. The Phase-53 (real-LLM) reading is preserved
  exactly: substrate FIFO ties every fixed strategy at
  ``accuracy_full = 0.800`` because the bench has no surplus
  (W7-1).
- **The CoordPy single-run product runtime contract is unchanged.**
  ``RunSpec`` / ``run`` / ``SweepSpec`` / ``run_sweep`` /
  report v2 schema: byte-for-byte identical from SDK v3.7.
- **The capsule layer's audit contribution is preserved.**
  T-1..T-7 hold on every Phase-54 cell unchanged.
- **Mac 2 still offline.** No two-Mac sharded inference happened
  in SDK v3.8; the ``MLXDistributedBackend`` integration boundary
  is byte-for-byte unchanged from SDK v3.6.

## [docs] — 2026-04-26 — documentation consolidation (no SDK change)

*Repo-cleanup only. No code change. SDK contract byte-for-byte
unchanged. Strictly additive on SDK v3.7.*

### Changed

- **Top-level Markdown clutter consolidated.** The repo root and
  `docs/` are reduced to a small canonical set; everything else is
  preserved under `docs/archive/`. The active scientific position is
  now obviously the live entry point and stale milestone notes can no
  longer read like current claims.
- **Canonical kept set** (top level): `README.md`, `ARCHITECTURE.md`,
  `CHANGELOG.md`, `LICENSE`, `CLAUDE.md`. **Canonical kept set**
  (`docs/`): `START_HERE.md`, `RESEARCH_STATUS.md`,
  `THEOREM_REGISTRY.md`, `HOW_NOT_TO_OVERSTATE.md`,
  `CAPSULE_FORMALISM.md`, `CAPSULE_TEAM_FORMALISM.md`,
  `context_zero_master_plan.md`, `MLX_DISTRIBUTED_RUNBOOK.md`,
  `RESULTS_COORDPY_SCALE_VS_STRUCTURE.md` (latest milestone, kept live).
- **Archive layout** (`docs/archive/`):
  - `capsule-research/` — `RESULTS_CAPSULE_LEARNING.md` +
    `RESULTS_CAPSULE_RESEARCH_MILESTONE[1-6].md`.
  - `coordpy-milestones/` — older CoordPy milestone notes
    `RESULTS_COORDPY_{CAPSULE, CAPSULE_NATIVE, INTRA_CELL,
    DEEP_INTRA_CELL, INNER_LOOP, TEAM_COORD, DISTRIBUTED}.md`
    (SDK v3.0 → v3.6).
  - `pre-coordpy-theory/` — pre-CoordPy Context Zero theory volumes:
    `PROOFS.md`, `EXTENDED_MATH[_1-7].md`, `OPEN_QUESTIONS.md`,
    `FRAMEWORK.md`, `EVALUATION.md`, `MVP.md`, `ROADMAP.md`,
    `VISION_MILLIONS.md`, `MATH_AUDIT.md`,
    `HIERARCHICAL_DECOMPOSITION.md`, `WAVES.md`.
  - `legacy-progress-notes/` — sprint prompts, paradigm-shift
    summaries, the pre-CoordPy benchmark-reproduction guide, the
    auto-generated theorem index.
- **`docs/archive/README.md`** *(new)* — archive index. Names every
  archived doc, points to the canonical replacement, and explains the
  read-only contract: the active scientific position is in `docs/`,
  the archive is historical record only.
- **Internal links updated.** Every cross-link inside the canonical
  docs (`README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/*.md`,
  `papers/*.md`) now resolves to the new file paths. Validated
  programmatically — zero broken Markdown links across the 14
  canonical docs.
- **`docs/START_HERE.md`** — adds a *Current canonical reading* table
  at the very top of the file. Mental-model diagram updated to show
  active vs archived theory paths.
- **`vision_mvp/scripts/generate_theorem_docs.py`** — auto-generated
  `THEOREMS_AUTO.md` now writes into
  `docs/archive/legacy-progress-notes/THEOREMS_AUTO.md` (was
  `docs/THEOREMS_AUTO.md`); the file was always a generated artefact,
  not a canonical claim source.

### Preserved

- All historical research material is intact under `docs/archive/`.
  No file deleted. No claim retracted. No theorem renumbered.
- `vision_mvp/RESULTS_PHASE*.md` (the per-phase research diary) is
  untouched — it lives with the code, not under `docs/`.
- The CoordPy SDK public contract, the Capsule Contract C1..C6, and
  the W3 / W4 / W5 / W6 theorem families are unchanged.

## [SDK v3.7] — 2026-04-26 — model-scale vs capsule-structure on multi-agent coordination (Phase-53 + W6 family)

*Strictly additive on SDK v3.6. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new surface is
the Phase-53 stronger-model multi-agent benchmark + W6 theorem
family. Mac 2 is still offline; no two-Mac sharded inference
happened in this milestone — the ``MLXDistributedBackend``
integration boundary is byte-for-byte unchanged from SDK v3.6
and waits for the runbook.*

### Added

- **`vision_mvp/experiments/phase53_scale_vs_structure.py`** *(new)* —
  Phase-53 stronger-model multi-agent benchmark. Drives the team
  coordinator with a real-LLM producer-role extractor across
  three model regimes (synthetic / qwen2.5:14b-32k / qwen3.5:35b)
  × five admission strategies (substrate, capsule_fifo,
  capsule_priority, capsule_coverage, capsule_learned) on the
  same candidate-handoff stream. Reports a clean ``model regime ×
  admission strategy`` decomposition with cross-regime
  candidate-kind TVD.
- **`vision_mvp/tests/test_coordpy_scale_vs_structure.py`** *(new)*
  — 19 contract tests: parser robustness on the closed-vocabulary
  claim grammar (16 cases), backend duck-typing, audit_ok grid
  end-to-end with a deterministic stub backend, schema lock.
- **`docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`** *(new)* — full
  milestone results note. Theorem-forward; declares W6-1..W6-4
  proved-or-empirical and W6-C1..W6-C5 conjectures (W6-C1 / C2
  drafted-then-falsified, W6-C3 positive, W6-C4 / C5 new).
- **`docs/data/phase53_scale_vs_structure_K4_n5.json`** *(new
  artefact)* — frozen benchmark output for reproducibility.

### Changed

- **`vision_mvp/coordpy/__init__.py`** — `SDK_VERSION` bumped to
  `coordpy.sdk.v3.7`. No public API change.
- **`docs/THEOREM_REGISTRY.md`** — W6-1 / W6-2 / W6-3 / W6-4 +
  W6-C1 / W6-C2 / W6-C3 / W6-C4 / W6-C5 rows added. The W4-C1
  row (SDK v3.5 conjecture) is now annotated as **conditional**:
  empirical-positive on its anchor distribution; falsified
  out-of-distribution on the Phase-53 real-LLM regime
  (capsule_learned 0.4 vs fixed 0.8 on synthetic and qwen2.5:14b;
  ties at qwen3.5:35b at 0.8/0.8).
- **`docs/RESEARCH_STATUS.md`** — sixth research axis added;
  active-conjectures section refreshed with W6-C family.
- **`docs/context_zero_master_plan.md`** — § 4.24 added: full
  Phase-53 narrative, W6 / W6-C summary, W4-C1 conditional
  reading, honest scope (Mac 2 offline, single-Mac qwen3.5:35b
  is the strongest model class actually exercised).
- **`docs/START_HERE.md`** — headline paragraph updated to
  reference the SDK v3.7 result and the *audit-axis* tightening
  of the original Context-Zero thesis.

### Headline empirical result

(n=5 saturated, K_auditor=4, T_auditor=128, three model regimes,
deterministic seeds (31, 32, 33))

| regime           | substrate | fixed capsule | learned |
| ---------------- | --------- | ------------- | ------- |
| synthetic        | 0.800     | 0.800         | 0.400   |
| qwen2.5:14b-32k  | 0.800     | 0.800         | 0.400   |
| qwen3.5:35b      | 0.800     | 0.800         | 0.800   |

* `structure_gain[regime]` = -0.4 / -0.4 / 0.0 (non-positive
  everywhere; scale narrows a *deficit*, not a *surplus*).
* `scale_gain[capsule_learned]` = +0.4; `scale_gain[fixed]` = 0.0.
* Cross-(14B, 35B) candidate-kind TVD = 0.167.
* Capsule-team lifecycle audit ``audit_team_lifecycle.is_ok()``
  = 60/60 across (regime × capsule strategy × scenario).

### Theorem registry deltas

- **W6-1 (proved + mechanically-checked).** Lifecycle audit
  T-1..T-7 holds 60/60 across the Phase-53 grid.
- **W6-2 (proved).** Phase-53 driver accepts duck-typed
  ``LLMBackend``.
- **W6-3 (proved + mechanically-checked).** Parser robustness
  on the closed-vocabulary claim grammar.
- **W6-4 (proved-empirical, real LLM, n=5 saturated).** The
  ``accuracy_full`` / ``structure_gain`` / ``scale_gain``
  decomposition is what is reported above.
- **W6-C1, W6-C2 (drafted, FALSIFIED-empirical).** Structure-
  preservation under scale (W6-C1) and synthetic→real-LLM
  transfer of the learned admission scorer (W6-C2) are both
  falsified on Phase-53 default; honest revised reading is in
  `docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md` § 4.3.
- **W6-C3 (empirical-positive).** Cross-(14B, 35B) candidate-
  kind TVD = 0.167 > 0.10 falsifier.
- **W6-C4, W6-C5 (new conjectures).** Substrate-FIFO competitive-
  ness at sufficient K, and scale-narrows-the-OOD-gap of the
  per-role admission scorer.

### Honest scope

* Mac 2 (192.168.12.248) is offline at the time of this
  milestone (ARP "incomplete"). **No two-Mac sharded inference
  ran.** No 70 B-class model ran. The strongest model class
  exercised is **single-Mac** qwen3.5:35b (36 B-MoE Q4) via
  Mac 1 Ollama.
* The MLX-distributed integration boundary
  (``MLXDistributedBackend``) is byte-for-byte unchanged from
  SDK v3.6 and remains correct against the in-process stub
  (W5-3). The runbook (`docs/MLX_DISTRIBUTED_RUNBOOK.md`) is the
  operator path when Mac 2 returns.
* Phase-53 is **incident-triage-bench-internal**. External
  validity to other multi-agent benchmarks is open
  (`task_scale_swe.py`, `phase33_security_escalation.py` are
  obvious next targets).
* The W4-C1 (SDK v3.5) reading on its anchor config (Phase-52
  default, K=8, spurious=0.30) is unchanged. The new SDK v3.7
  reading is OOD.

### Tests + validation

* `python3 -m unittest -v vision_mvp.tests.test_coordpy_scale_vs_structure`
  → **19 tests pass in 0.069 s**.
* `python3 -m unittest vision_mvp.tests.test_coordpy_team_coord
  vision_mvp.tests.test_coordpy_llm_backend
  vision_mvp.tests.test_coordpy_capsule_native_inner_loop
  vision_mvp.tests.test_coordpy_capsule_native
  vision_mvp.tests.test_coordpy_capsule_native_intra_cell
  vision_mvp.tests.test_coordpy_capsule_native_deeper
  vision_mvp.tests.test_coordpy_scale_vs_structure`
  → **116 tests pass in 3.207 s** (SDK v3.6 invariants intact).
* `python3 -m vision_mvp.experiments.phase53_scale_vs_structure
  --endpoint http://192.168.12.191:11434
  --models synthetic,qwen2.5:14b-32k,qwen3.5:35b
  --n-eval 5 --K-auditor 4 --T-auditor 128
  --out /tmp/coordpy-distributed/phase53_scale_vs_structure_K4.json`
  → 14B LLM wall 92.6 s; 35B LLM wall 152.0 s; n_results = 75.
  Frozen at `docs/data/phase53_scale_vs_structure_K4_n5.json`.

## [SDK v3.5] — 2026-04-26 — capsule-native multi-agent team coordination (research slice)

*Strictly additive on SDK v3.4. The CoordPy single-run product
runtime contract is byte-for-byte unchanged. The new surface is a
capsule-native multi-agent coordination research slice that
runs side-by-side with the CoordPy SDK.*

### Added

- **Three new closed-vocabulary `CapsuleKind` values** — `TEAM_HANDOFF`
  (capsule-native multi-agent handoff; distinct from `HANDOFF`
  which adapts a substrate `TypedHandoff`), `ROLE_VIEW` (per-role
  admitted view of one coordination round; `max_parents = K_role`,
  `max_tokens = T_role`), `TEAM_DECISION` (team-level decision).
- **`vision_mvp.coordpy.team_coord`** — `RoleBudget`,
  `DEFAULT_ROLE_BUDGETS`, `capsule_team_handoff`,
  `capsule_role_view`, `capsule_team_decision`, three fixed
  admission policies (`FifoAdmissionPolicy`,
  `ClaimPriorityAdmissionPolicy`, `CoverageGuidedAdmissionPolicy`),
  `TeamCoordinator`, `audit_team_lifecycle` over invariants
  `T-1..T-7` (Theorem **W4-1**, *proved + mechanically-checked*).
- **`vision_mvp.coordpy.team_policy`** —
  `LearnedTeamAdmissionPolicy` (per-role logistic-regression
  scorer over six capsule features), `TrainSample`, `TrainStats`,
  `train_team_admission_policy`. Numpy-only; deterministic given
  seed.
- **`vision_mvp/experiments/phase52_team_coord.py`** — reference
  benchmark over a noisy-extraction expansion of the Phase-31
  incident-triage bank. Cross-seed result on default config
  ($K_\text{auditor}=8$, $T_\text{auditor}=256$,
  $n_\text{eval}=31$, ``train_seed ∈ {0, …, 11}``,
  ``PYTHONHASHSEED=0``): **learned policy** admits **strictly
  fewer handoffs** than the strongest fixed baseline
  (coverage-guided) on every train seed (12/12), with mean
  savings ≈ 1.26 handoffs per scenario. The learned policy also
  improves pooled team-decision accuracy on most train seeds
  (gap on `accuracy_full` > 0 in 11/12 seeds, mean **+0.054**;
  gap on `accuracy_root_cause` > 0 in 8/12 seeds, mean
  **+0.032**) — but the accuracy advantage **reverses at higher
  noise** (`spurious_prob = 0.50`). `audit_ok_rate = 1.000` for
  every capsule strategy on every seed. Conjecture **W4-C1**:
  budget-efficiency dominance is robust per-seed; accuracy
  advantage is mean-positive on the default noise config but
  not strict per-seed; advantage does not survive heavier
  noise. (See ``docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md`` § Cross-seed
  result for the canonical reading; ``docs/HOW_NOT_TO_OVERSTATE.md``
  forbids reporting single-seed numbers without the cross-seed
  distribution.)
- **Theorems** — W4-1 (proved + mechanically-checked); W4-2
  (proved-conditional: coverage-implies-correctness); W4-3
  (proved-negative: per-role budget below the role's causal-
  share floor cannot be rescued by *any* admission policy).
- **Conjectures** — W4-C1, W4-C2 (cohort-lifted role view closes
  W4-3 sub-class), W4-C3 (capsule admission rule subsumes
  Phase-36 adaptive-sub).
- **`docs/CAPSULE_TEAM_FORMALISM.md`** — formal model.
- **`docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md`** — milestone note.
- **`vision_mvp/tests/test_coordpy_team_coord.py`** — 22 contract
  tests.
- **README**, **START_HERE**, **RESEARCH_STATUS**,
  **THEOREM_REGISTRY**, **HOW_NOT_TO_OVERSTATE**, **master plan
  §4.22** — all updated.

### Compatibility

- All 85 capsule-native run-boundary tests (v3.1..v3.4) +
  Phase-31 typed-handoff tests continue to pass byte-for-byte.
  Team-layer tests are 22 additional contracts.
- The CoordPy `coordpy` console scripts are unchanged. The team layer
  ships as `vision_mvp.coordpy.team_coord` /
  `vision_mvp.coordpy.team_policy` and is also re-exported from
  the top-level `vision_mvp.coordpy` namespace as
  `TeamCoordinator`, `audit_team_lifecycle`,
  `LearnedTeamAdmissionPolicy`, etc.

### Honest scope

The Phase-52 benchmark is synthetic; the result *direction* is
robust under deterministic noise; cross-bench transfer is open.
"We solved multi-agent context" is **forbidden** by
`docs/HOW_NOT_TO_OVERSTATE.md`; the defensible reading is
W4-1 / W4-2 / W4-3 / W4-C1 above.

## [SDK v3.4] — 2026-04-26 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + synthetic mode + cross-model parser-boundary research

*Strictly additive on SDK v3.3. Every v3.3 contract test (18) still
passes byte-for-byte; capsule view schema name unchanged
(`coordpy.capsule_view.v1` — PROMPT / LLM_RESPONSE payloads are
additive). Full CoordPy + capsule test suite green (199 tests).*

### Added
- **PROMPT capsule kind** (parent: SWEEP_SPEC; Theorem W3-42).
  Records prompt SHA-256 + byte length + bounded text snippet
  (≤ 4 KiB) + model_tag + prompt_style + coordinates.
  Idempotent on content (Capsule Contract C1) — byte-identical
  prompts collapse to one capsule.
- **LLM_RESPONSE capsule kind** (parent: PROMPT; Theorem W3-43).
  Records response SHA-256 + byte length + bounded snippet +
  elapsed milliseconds + coordinates. Admission rejects if
  prompt CID is not yet sealed (Capsule Contract C5).
- **`CapsuleNativeRunContext.seal_prompt`** /
  **`seal_llm_response`** runtime methods, plus
  **`seal_parse_outcome(llm_response_cid=...)`** optional
  argument. The end-to-end inner-loop chain is now five typed
  capsules: `PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
  PATCH_PROPOSAL → TEST_VERDICT`.
- **`capsule_from_prompt`**, **`capsule_from_llm_response`**
  adapters; `PROMPT_TEXT_CAP` / `LLM_RESPONSE_TEXT_CAP` constants.
- **Lifecycle audit invariants L-9 / L-10 / L-11** (Theorems
  W3-44 / W3-45):
  - L-9: PROMPT.parents == (SWEEP_SPEC,).
  - L-10: LLM_RESPONSE has exactly one parent, a sealed PROMPT.
  - L-11: PARSE_OUTCOME / LLM_RESPONSE coordinate consistency
    (instance_id / parser_mode / apply_mode / n_distractors;
    strategy may differ).
- **Synthetic-LLM mode**: `SweepSpec(mode="synthetic",
  synthetic_model_tag=<tag>)`. Uses a deterministic in-process
  `SyntheticLLMClient` instead of an Ollama endpoint. Seven
  calibrated distributions ship in
  `vision_mvp.coordpy.synthetic_llm.SYNTHETIC_MODEL_PROFILES`:
  `clean`, `unclosed`, `prose`, `empty`, `fenced`,
  `multi_block`, `mixed`. The full PROMPT / LLM_RESPONSE /
  PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT chain seals
  end-to-end without network access.
- **Cross-model parser-boundary experiment** (Conjecture W3-C6,
  empirical):
  `vision_mvp.experiments.parser_boundary_cross_model`. Sweeps
  `(model_tag, parser_mode)` across the synthetic distribution
  library; reports cross-distribution PARSE_OUTCOME failure-kind
  TVD up to 1.000 and parser-mode (strict→robust) shift up to
  1.000 on `synthetic.unclosed`. Reproducible from CLI:
  `python3 -m vision_mvp.experiments.parser_boundary_cross_model`.
- **16 new contract tests** in
  `vision_mvp/tests/test_coordpy_capsule_native_inner_loop.py`
  covering W3-42 / W3-43 / W3-44 / W3-45 / W3-C6.

### Changed
- **`SDK_VERSION`** bumped to `coordpy.sdk.v3.4`.
- **`CapsuleKind.ALL`** now includes `PROMPT` and `LLM_RESPONSE`.
- **`render_view.payload_kinds_always`** extended to include
  PROMPT and LLM_RESPONSE (so on-disk audits can navigate the
  full inner-loop chain from `capsule_view.json` alone).
- **`CapsuleLifecycleAudit.RULES`** extended from 8 rules to 11.
- **W3-13** (DAG height ≤ 4 on canonical run pattern) is updated
  to ≤ 5 on canonical SDK v3.4 runs (the inner-loop chain adds
  one structural layer). Documented in
  `docs/CAPSULE_FORMALISM.md` § 4.J.
- **Conjecture W3-C5 (legacy SDK v3.3)** is **DISCHARGED** by
  Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **Conjecture W3-C4 (legacy SDK v3.3)** is **superseded** by the
  sharper synthetic reading W3-C6.

### Documentation
- New milestone note: **`docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md`**.
- `docs/CAPSULE_FORMALISM.md` § 4.J added (W3-42 / W3-43 / W3-44 /
  W3-45 / W3-C6 + W3-C5-discharged).
- `docs/THEOREM_REGISTRY.md`, `docs/RESEARCH_STATUS.md`,
  `docs/HOW_NOT_TO_OVERSTATE.md` updated for SDK v3.4.
- `docs/START_HERE.md` adds "What changed in SDK v3.4" section.
- `docs/context_zero_master_plan.md` § 4.21 added.
- `papers/coordpy_capsule_native_runtime.md` strengthened —
  capsule-native execution is now its real centre, with strict
  claim taxonomy covering PROMPT / LLM_RESPONSE chain and the
  W3-C6 empirical anchor.
- README headline + stability matrix updated.

## [0.5.1] — 2026-04-22 — CoordPy identity & clarity pass

*Documentation / exemplar milestone. No SDK-contract change; all 1349
Slice-2 tests still pass.*

### Added
- **`docs/START_HERE.md`** — canonical one-pass orientation for new
  readers. Classifies every top-level surface (CoordPy SDK, CLI,
  extension protocols, unified runtime, legacy product path, core
  substrate, research shards, boundary). Meant to be the answer to
  "what is this repo?" without duplicating the README or the master
  plan.
- **`examples/out_of_tree_plugin/coordpy-markdown-sink/`** — first
  in-repo exemplar of a standalone pip-installable CoordPy plugin
  package. Declares `[project.entry-points."coordpy.report_sinks"]`,
  registers a Markdown `ReportSink` via
  `importlib.metadata.entry_points`, and requires zero edit under
  `vision_mvp/`. Closes master-plan § 10.5 ledger item 2 at the
  machinery-plus-artifact level (only the "published by a third
  party" condition remains future).
- **`vision_mvp/RESULTS_COORDPY_IDENTITY.md`** — theory-forward results
  note with theorem-style claims (W-IDN-1 identity projection,
  W-IDN-2 orientation sufficiency, W-IDN-3 extension-surface
  reality) and three conjectures (W-IDN-C1 cold-agent
  classification, W-IDN-C2 stable-identity robustness, W-IDN-C3
  distinctiveness via composition rather than primitive novelty).

### Changed
- **README headline** now leads with **CoordPy** (the shipped product)
  and positions CASR as original-substrate research; the scaling
  claims are preserved and re-anchored to Theorem 3 in `docs/archive/pre-coordpy-theory/PROOFS.md`.
- **ARCHITECTURE.md headline** re-anchored to CoordPy + Context Zero;
  a framing callout was added before the Phase 26–44 block so
  readers know that block is a historical incremental record and
  the durable architecture is the layered substrate diagram + § 3
  of the master plan.
- **`vision_mvp/__init__.py`** top-level docstring: CoordPy is the
  shipped product; `CASRRouter` is explicitly research-grade code
  used by the SDK under the hood.
- **`vision_mvp/api.py`** `CASRRouter` docstring no longer says
  "Phase-3 hierarchical protocol" or "CASR-theoretic optimum" in
  places where a user would read them as current product contract;
  the O(log N) bound is now anchored to Theorem 3.
- **`vision_mvp/product/__init__.py`** retitled from "Phase-45
  product-grade orchestration surface" to "Legacy product modules
  (pre-CoordPy import path)" — same code, correct framing.
- **`pyproject.toml`** — clearer comment on the `casr` legacy
  script; public CLI stays `coordpy` / `coordpy-import` / `coordpy-ci`.
- **Master plan § 10** — short "Programme vs Product" callout near
  the top; § 10.1 stability matrix row for out-of-tree plugins
  updated from "boundary / next-slice" to "exemplar landed";
  § 10.3 B.6 note and § 10.5 ledger item 2 updated.

### Not changed (deliberately)
- The CoordPy SDK contract (every Slice 2 public symbol remains).
- Any test; suite is green at 1349/1349.
- Docker-first-by-default flip for untrusted JSONLs (still Slice 3).
- GitHub Actions release-on-real-tag firing (workflow still declared,
  not yet exercised on a real tag).

## [0.5.0] — 2026-04-22 — CoordPy SDK Slice 2

### Added
- **Extension system** (`vision_mvp/coordpy/extensions/`). Three
  runtime-checkable Protocols — `SandboxBackend`, `TaskBankLoader`,
  `ReportSink` — each with an in-process registry and discovery via
  `importlib.metadata.entry_points` under groups
  `coordpy.sandboxes`, `coordpy.task_banks`, `coordpy.report_sinks`.
  One worked example (`JsonlWithMetaSink`) and a contract test
  suite that exercises the full register→resolve→emit path.
- **Unified mock/real runtime** (`vision_mvp/coordpy/runtime.py`).
  New `SweepSpec` dataclass; single `run_sweep(spec)` entry point
  dispatches mock and real runs through the same substrate
  primitives. Real runs execute in-process when
  `RunSpec.acknowledge_heavy=True`; otherwise the SDK refuses to
  start the heavy run and emits the resolved launch command.
- **`RunSpec.acknowledge_heavy`** and **`RunSpec.report_sinks`** —
  first-class cost gate and plugin hook on the top-level SDK spec.
- **`HeavyRunNotAcknowledged`** exception — strict cost-gate signal.
- **Env-driven endpoints**: `COORDPY_OLLAMA_URL_MAC1`,
  `COORDPY_OLLAMA_URL_MAC2`, `COORDPY_OLLAMA_URL` override profile-
  declared URLs at runtime. No hard-coded cluster IP is baked into
  code paths that a third-party consumer has to edit.
- **`--acknowledge-heavy` / `--report-sink`** flags on `coordpy`.
- **Report schema bump**: `phase45.product_report.v2`. v1 remains
  accepted by `coordpy-ci`; both listed in `EXPECTED_REPORT_SCHEMAS`.
- **GitHub Actions workflow** (`.github/workflows/coordpy-ci.yml`):
  SDK contract tests on 3.10/3.11/3.12, console-script smoke,
  `python -m build` sdist+wheel, release on tag.
- **Cluster-backed validation artifact** under
  `vision_mvp/artifacts/coordpy_slice2_g1/` — real ASPEN `mac1`
  `qwen2.5-coder:14b` run launched via `coordpy.run(RunSpec(...,
  acknowledge_heavy=True))`, with provenance manifest and
  `coordpy-ci` verdict.
- **Theory note**: `vision_mvp/RESULTS_COORDPY_SLICE2.md` —
  theorem-style claims W2-1 … W2-4.

### Changed
- `SDK_VERSION` bumped to `coordpy.sdk.v2`. The bump is additive;
  every Slice 1 public symbol remains available.
- `CI gate` accepts v1 and v2 report schemas.
- `product/runner.py` now routes all sweeps through
  `coordpy.runtime.run_sweep` instead of the legacy
  `_real_sweep_stub`.

### Deprecated
- `_real_sweep_stub` / `_mock_sweep` in `vision_mvp/product/runner.py`
  are private and will be removed in a future release; external code
  should use `coordpy.run_sweep(SweepSpec(...))`.

### Next-slice (deferred, still honest)
- Docker-first sandbox as the default for public/untrusted JSONLs
  (backend exists; default-flip is Slice 3).
- Public SWE-bench-Lite JSONL on local disk (🧱 external).
- Resident ≥70B coder-finetuned model (🧱 external).

## [0.4.0] — 2026-04-21 — CoordPy SDK Slice 1

See `docs/context_zero_master_plan.md` § 10.2.

- Introduced `vision_mvp/coordpy/` stable SDK boundary.
- `RunSpec` / `run`, `CoordPyConfig`, `build_manifest`, schema
  constants, profile/report/ci_gate/import_data re-exports.
- Provenance manifest (`coordpy.provenance.v1`) on every run.
- Console scripts: `coordpy`, `coordpy-import`, `coordpy-ci`.
- Package renamed to `coordpy` on PyPI; `SDK_VERSION = coordpy.sdk.v1`.
- `sys.path.insert` hacks removed from product modules.
- Contract tests: `test_coordpy_public_api.py`, `test_coordpy_provenance.py`.

---

## [0.1.0] — 2026-04-16

Initial alpha release. One continuous research session.

### Added — Core library (`vision_mvp/`)

- **`CASRRouter`** — black-box public API. `step(observations) -> estimates`.
- Core primitives: `Bus`, `Agent`, `Manifold` (given basis),
  `StreamingPCA` (learned basis), `Stigmergy` (CRDT register),
  `Workspace` (top-k admission), `NeuralPredictor` and `PredictorBank`
  (vectorized across agents).
- Phase-6 additions: `MarketWorkspace` (VCG pricing),
  `SharedRNG`/`DeltaChannel` (pre-shared randomness), `AdaptiveScale` and
  `ContinuousScaleProjector` (continuous-scale projection).
- Six coordination protocols: `naive`, `gossip`, `manifold_only`,
  `full_stack`, `adaptive`, `hierarchical`, `holographic`, `swarm`, and
  `llm_protocols` (real LLM agents via Ollama).
- Two coordination tasks: `consensus` (static) and
  `drifting_consensus` (non-stationary with optional shock).

### Added — Experiments & results

- Phase 1 through Phase 5 runnable experiment harnesses under
  `vision_mvp/experiments/`.
- Measured scaling law: peak per-agent context = ⌈log₂ N⌉ exactly at
  every N ∈ {10, 50, 200, 1 000, 5 000, 10 000, 20 000, 50 000, 100 000}.
- Real LLM demonstration at N = 10 (local qwen2.5:0.5b via Ollama) showing
  34 % token savings with 100 % accuracy.

### Added — Theory

- **`docs/archive/pre-coordpy-theory/PROOFS.md`** — twelve formal theorems, each with a proof and a
  machine-checkable empirical counterpart in `tests/`.
- **`EXTENDED_MATH_[1–7].md`** — 72-framework survey converging on the
  O(log N) bound from Information Bottleneck through Geometric Langlands.
- **`docs/archive/pre-coordpy-theory/VISION_MILLIONS.md`** — the 10-idea paradigm shift for million-agent
  systems. 6 of 10 ideas implemented.

### Added — Tests

- **94 tests**, all passing (0.45 s total wall time):
  - 55 core-module unit tests.
  - 15 protocol integration & regression tests (including the scaling-law
    assertion `test_full_stack_peak_context_is_log_n`).
  - 13 Phase-6 tests (market, shared randomness, continuous scale).
  - 11 public-API (`CASRRouter`) tests.

### Added — Developer UX

- `pyproject.toml` (installable as `context-zero`).
- `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`, top-level `README.md`.
- `casr` CLI entry-point (`python -m vision_mvp demo|scale|phase|test|info`).
- Four runnable `examples/`:
  1. basic consensus
  2. drift tracking
  3. scaling demo
  4. local LLM coordination

### Not yet

- Real LLM tests at N > 10 (need bigger compute budget).
- Async variants (current protocol is synchronous).
- A formal peer-review cycle for the math.
- PyPI upload.

All the mathematics says O(log N). The code and the test suite confirm it.
The next step is to run it in anger on harder tasks and let skeptical
reviewers tear it apart.
