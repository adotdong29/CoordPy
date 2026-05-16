# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: post-W67 W68
> milestone (Two-Plane Substrate-Coupled Latent Operating System
> research line), 2026-05-16.

## TL;DR — W68 Two-Plane Substrate-Coupled Latent Operating System (post-W67 research milestone)

The programme now has **sixty-five** coupled research axes. W68
mints axis 65: the **thirteenth substrate-attack milestone**, the
**fourth multi-agent task-success-bearing** substrate milestone
(first to win across *seven* regimes: W67's five +
**partial-contradiction-under-delayed-reconciliation** +
**agent-replacement-warm-restart**), and the **first milestone to
explicitly split the architecture into two planes** — Plane A
(hosted control plane) vs Plane B (real substrate plane) — and
codify the hosted ↔ real-substrate wall as a content-addressed
artefact with a structural falsifier.

The load-bearing W68 win is **MASC V4 / TCC V3 + tiny_substrate_v13
+ 22 supporting Plane B modules**: V13 strictly beats V12 on
≥ 50 % of seeds in every regime (baseline 80 %,
team_consensus_under_budget 80 %, team_failure_recovery 93.3 %,
role_dropout 60 %, branch_merge_reconciliation 80 %,
partial_contradiction 60 %, agent_replacement_warm_restart 93.3 %),
and TSC_V13 strictly beats TSC_V12 on ≥ 50 % of seeds in every
regime (53.3–93.3 % depending on regime). Plus an honest
**hosted control plane** (Plane A): HostedRouterController,
HostedLogprobRouter (top-k logprob fusion + text-only fallback),
HostedCacheAwarePlanner (prefix-CID + ≥ 50 % token savings at
hit_rate = 1.0), HostedProviderFilter, HostedCostPlanner, and the
**explicit architecture wall** HostedRealSubstrateBoundary that
enumerates 15 blocked axes at the hosted surface, 64 real-
substrate-only V13 axes, and ships a falsifier returning 0 on
honest claims and 1 on dishonest claims.

**Twenty-eight orthogonal advances** on top of W67 (22 Plane B + 6
Plane A). Plane B headlines: V13 substrate (15 layers; four new
V13 axes — partial-contradiction witness tensor, agent-replacement
flag with warm-restart window, substrate prefix-reuse counter, V13
composite gate score); KV V13 nine-target stacked ridge +
50-dim agent-replacement fingerprint + partial-contradiction
falsifier; HSB V12 nine-target stacked ridge + hidden-vs-agent-
replacement probe; Prefix V12 K=192 drift curve + seven-way
comparator; Attention V12 eight-stage clamp; Cache V11 eight-
objective stacked ridge + per-role 9-dim agent-replacement head;
Replay V9 fourteen-regime ridge + six-way agent-replacement-routing
head; Deep Substrate Hybrid V13 thirteen-way loop; Substrate
Adapter V13 with `substrate_v13_full` tier; Persistent V20
(19 layers, max_chain_walk_depth=32768, seventeenth skip carrier);
Multi-Hop V18 (44 backends, 1892 directed edges, chain-len 34,
13-axis composite); MLSC V16 (partial-contradiction + agent-
replacement witness chains); Consensus V14 (22 stages); CRC V16
(65536-bucket fingerprint, 36-bit adversarial burst); LHR V20
(19 heads, max_k=448, ten-layer scorer); ECC V20 (2^33 codes,
≥ 35.0 bits/visible-token); Uncertainty V16 (15-axis composite);
Disagreement V14 (agent-replacement-equivalence identity +
falsifier); TVS V17 (18 arms); MASC V4 (10-policy, 7-regime);
TCC V3 (partial-contradiction + agent-replacement-warm-restart
arbiters).

W68 fits **six** new closed-form ridge solves on top of W67's 41
(cache V11 eight-objective + cache V11 per-role agent-replacement
+ replay V9 per-role per-regime + replay V9 agent-replacement-
routing + HSB V12 nine-target + KV V13 nine-target). Total **47
closed-form ridge solves across W61..W68**. No autograd, no SGD,
no GPU. The benchmark sweep is **138 cells across 4 benchmark
families** (R-152 hosted control plane, R-153 real substrate
plane, R-154 multi-agent task success across 7 regimes, R-155
hosted-vs-real wall), 138/138 cells pass at 3/3 seeds.

The W68 envelope verifier enumerates **42 disjoint failure modes**
(cumulative trust boundary across W22..W68 ≥ 1417 enumerated
failure modes). Ships at `coordpy.tiny_substrate_v13`,
`coordpy.kv_bridge_v13`, `coordpy.hidden_state_bridge_v12`,
`coordpy.prefix_state_bridge_v12`,
`coordpy.attention_steering_bridge_v12`,
`coordpy.cache_controller_v11`, `coordpy.replay_controller_v9`,
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
`coordpy.team_consensus_controller_v3`, `coordpy.w68_team`, plus
the Plane A hosted modules `coordpy.hosted_router_controller`,
`coordpy.hosted_logprob_router`,
`coordpy.hosted_cache_aware_planner`,
`coordpy.hosted_provider_filter`,
`coordpy.hosted_cost_planner`,
`coordpy.hosted_real_substrate_boundary`, and benchmarks
`coordpy.r152_benchmark` / `coordpy.r153_benchmark` /
`coordpy.r154_benchmark` / `coordpy.r155_benchmark`. **Public SDK
contract is byte-for-byte unchanged: `coordpy.__version__ ==
"0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`.**

Honest scope (do-not-overstate):

* W68 codifies the hosted ↔ real-substrate **architecture wall**;
  it does NOT dissolve it. Hosted backends remain text-only at the
  HTTP surface. ``W68-L-HOSTED-NO-SUBSTRATE-CAP``.
* The seven-regime multi-agent wins are measured inside the
  in-repo synthetic MASC V4 harness. ``W68-L-MULTI-AGENT-
  COORDINATOR-V4-SYNTHETIC-CAP``.
* Frontier-model substrate access remains the unsolved research-
  line wall. ``W68-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP``.

## TL;DR — W67 Stronger Branch-Merge / Role-Dropout Substrate-Coupled Latent Operating System (post-W66 research milestone)

The programme now has **sixty-four** coupled research axes. W67
mints axis 64: the **twelfth substrate-attack milestone** and the
**third multi-agent task-success-bearing** substrate milestone
(the first to produce wins under *five* failure-mode regimes:
baseline + team-consensus-under-budget + team-failure-recovery +
**role-dropout** + **branch-merge-reconciliation**). W67 extends
the in-repo V12 substrate into a multi-regime multi-agent
coordination engine across five regimes: a
**MultiAgentSubstrateCoordinatorV3** runs N role-typed agents
through the V12 substrate under **eight matched-budget policies**
and **five regimes** (baseline + team-consensus-under-budget +
team-failure-recovery + role-dropout +
branch-merge-reconciliation); the V12 policy *strictly beats* V11
on ≥ 73 % of seeds in every regime (load-bearing wins: 80 %
role-dropout, 100 % branch-merge-reconciliation), and the
team_substrate_coordination_v12 policy *strictly beats*
team_substrate_coordination_v11 on ≥ 47 % of seeds (with 91.7 % at
baseline and 80 % at team-failure-recovery and
branch-merge-reconciliation).

Twenty-one orthogonal substrate-coupling, capsule-native, and
multi-agent-coordination advances on top of W66 — (M1) **Tiny
Transformer Runtime V12** (14 layers; four new V12 axes: per-(L,
H, T) branch-merge witness tensor, per-role-pair
role-dropout-recovery flag, substrate snapshot-fork primitive,
per-layer V12 composite gate score); (M2) **KV Bridge V12**
(eight-target stacked ridge; branch-merge margin probe; role-pair
40-dim fingerprint; branch-merge-reconciliation falsifier); (M3)
**HSB V11** (eight-target stacked ridge; per-(L, H)
hidden-vs-branch-merge probe; branch-merge margin); (M4) **Prefix
V11** (K=128 drift curve; role+task+team+branch 40-dim
fingerprint; six-way prefix/hidden/replay/team/recover/branch
comparator); (M5) **Attention V11** (seven-stage clamp: V10 +
branch-merge attention bias; branch-conditioned attention
fingerprint); (M6) **Cache Controller V10** (seven-objective
ridge adding branch-merge; per-role 8-dim eviction head); (M7)
**Replay Controller V8** (12 regimes adding `role_dropout_regime`
and `branch_merge_reconciliation_regime`; per-role per-regime
ridge; trained branch-merge-routing head); (M8) **Deep Substrate
Hybrid V12** (twelve-way bidirectional loop with the four new V12
axes + team-consensus-controller-V2 axis); (M9) **Substrate
Adapter V12** (4 new V12 capability axes; new `substrate_v12_full`
tier); (M10) **Persistent Latent V19** (18 layers; sixteenth skip
carrier `role_dropout_recovery_carrier`;
max_chain_walk_depth=16384; distractor rank 18); (M11) **Multi-Hop
Translator V17** (40 backends; 1560 directed edges; chain-length
30; 12-axis composite adding
`branch_merge_reconciliation_trust`); (M12) **Mergeable Latent
Capsule V15** (`role_dropout_recovery_witness_chain` and
`branch_merge_reconciliation_witness_chain`); (M13) **Consensus
Fallback Controller V13** (20-stage chain inserting
`role_dropout_arbiter` and
`branch_merge_reconciliation_arbiter`); (M14) **Corruption-Robust
Carrier V15** (32768-bucket fingerprint; 35-bit adversarial burst;
branch-merge reconciliation probe); (M15) **Long-Horizon Retention
V19** (18 heads, max_k=384, nine-layer scorer adding random+mish);
(M16) **ECC Codebook V19** (K1..K18 = 2^31 = 2 147 483 648 codes;
**33.333 bits/visible-token** at full emit); (M17) **Uncertainty
Layer V15** (14-axis composite adding
`branch_merge_reconciliation_fidelity`); (M18) **Disagreement
Algebra V13** (Bregman-equivalence identity + falsifier); (M19)
**Seventeen-arm TVS Arbiter V16** (17 arms with
`branch_merge_reconciliation` arm); (M20) **Multi-Agent Substrate
Coordinator V3** (the load-bearing W67 mechanism — eight
matched-budget policies across five regimes; V12 strictly beats
V11 across all five regimes); (M21) **Team-Consensus Controller
V2** (regime-aware weighted quorum + branch-merge arbiter +
role-dropout repair + substrate-replay fallback + transcript
fallback).

The W67 `W67Team` orchestrator composes all twenty-one modules,
emits per-turn 30 module witness CIDs (one for the branch-merge
falsifier, one for the MASC V3 aggregate, one for the
team-consensus controller V2), and seals them into a
`W67HandoffEnvelope` whose `w66_outer_cid` carries forward the
W66 envelope byte-for-byte. The W67 envelope verifier enumerates
**147 disjoint failure modes** (≥ 140 target met).

W67 is the **twelfth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. `W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`
carries forward unchanged. `W67-L-V12-NO-AUTOGRAD-CAP` is the new
ridge-only cap: W67 fits **only** forty-one closed-form linear
ridge solves (thirty-five from W61..W66 + six new) — (a) cache
controller V10 seven-objective; (b) cache controller V10 per-role
eviction (planner); (c) replay V8 per-role per-regime (planner) ×
12 regimes; (d) replay V8 branch-merge-routing head; (e) HSB V11
eight-target inner V5; (f) KV V12 eight-target inner V6 — no SGD,
no autograd, no GPU.

W67 ships at `coordpy.tiny_substrate_v12`,
`coordpy.kv_bridge_v12`, `coordpy.hidden_state_bridge_v11`,
`coordpy.prefix_state_bridge_v11`,
`coordpy.attention_steering_bridge_v11`,
`coordpy.cache_controller_v10`, `coordpy.replay_controller_v8`,
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
`coordpy.r149_benchmark`, `coordpy.r150_benchmark`,
`coordpy.r151_benchmark` — reachable only through explicit
imports. `coordpy.__version__` remains `0.5.20`; SDK contract is
byte-for-byte unchanged. **No PyPI release**.

R-149 (24 cell families) + R-150 (14 cell families) + R-151 (18
cell families) at 3 seeds verify H285..H304. **56 of 56 H-bars
pass 3/3 seeds (168/168 cells, strong success per the W67 success
criterion)**. Cumulative trust boundary across W22..W67 =
**1375 enumerated failure modes** (1228 from W22..W66 + 147 new
W67 envelope verifier modes).

W67 envelope chain end-to-end: `W66 envelope CID ==
W67.w66_outer_cid` (verified live by
`test_w67_team_envelope_chain`). Trivial passthrough preserved
byte-for-byte (`test_w67_trivial_passthrough_byte_identical`).

W67 is the **third** programme milestone where substrate control
produces a **measurable multi-agent task-level win** and the
**first** to produce wins under all **five** failure-mode regimes
(baseline + team-consensus-under-budget + team-failure-recovery +
role-dropout + branch-merge-reconciliation). The most decisive
new evidence comes from the **branch-merge-reconciliation regime**
(where V12 strictly beats V11 on 100 % of seeds: the V12
snapshot-fork + branch-merge primitive lets the substrate
reconcile conflicting branch payloads that V11 cannot) and the
**role-dropout regime** (where V12 strictly beats V11 on 80 % of
seeds: the V12 role-dropout-recovery flag lets even
`substrate_routed_v12` alone — without `team_consensus_active` —
infer a recovery payload that V11 cannot).

The honest scope is unchanged: W67 makes the **in-repo V12
substrate** more load-bearing — four new internal axes, six new
closed-form-ridge-trained controllers/heads (forty-one total
across W61..W67), a twelve-way bidirectional hybrid loop, five
multi-agent task-success regimes — but the **third-party
hosted-model substrate remains blocked** at the HTTP surface
(`W67-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`).

## TL;DR — W66 Stronger Solving-Context Substrate-Coupled Latent Operating System (post-W65 research milestone)

The programme now has **sixty-three** coupled research axes. W66 mints
axis 63: the **eleventh substrate-attack milestone** and the **second
multi-agent task-success-bearing** substrate milestone (the first to
produce wins across *three* regimes, not just one). W66 extends the
in-repo V11 substrate into a multi-regime multi-agent coordination
engine: a **MultiAgentSubstrateCoordinatorV2** runs N role-typed agents
through the V11 substrate under **six matched-budget policies** and
**three regimes** (baseline + team-consensus-under-budget + team-
failure-recovery); the V11 policy *strictly beats* V10 on ≥ 93 % of
seeds in each of the first two regimes, and the team_substrate_
coordination_v11 policy *strictly beats* V11 on ≥ 73 % of seeds in
each of the three regimes (load-bearing in failure recovery: TSC_V11
hits 80 % team success while every other policy is stuck at 40 %).

Twenty-one orthogonal substrate-coupling, capsule-native, and multi-
agent-coordination advances on top of W65 — (M1) **Tiny Transformer
Runtime V11** (13 layers; four new V11 axes: per-(L, H, T) replay-
trust ledger, per-role team-failure-recovery flag, substrate snapshot-
diff primitive, per-layer V11 composite gate score); (M2) **KV Bridge
V11** (seven-target stacked ridge; team-coordination margin probe;
multi-agent task fingerprint; team-failure-recovery falsifier); (M3)
**HSB V10** (seven-target stacked ridge; per-(L, H) hidden-wins-vs-
team-success probe; team-consensus margin); (M4) **Prefix V10** (K=96
drift curve; role+task+team 30-dim fingerprint; five-way prefix/hidden/
replay/team/recover comparator); (M5) **Attention V10** (six-stage
clamp: V9 + attention-trust ledger; team-conditioned fingerprint); (M6)
**Cache Controller V9** (six-objective ridge adding team-failure-
recovery; per-role 7-dim eviction head); (M7) **Replay Controller V7**
(10 regimes adding `team_failure_recovery_regime` and
`team_consensus_under_budget_regime`; per-role per-regime ridge;
trained team-substrate-routing head); (M8) **Deep Substrate Hybrid
V11** (eleven-way bidirectional loop with the four new V11 axes + team-
consensus-controller axis); (M9) **Substrate Adapter V11** (4 new V11
capability axes; new `substrate_v11_full` tier); (M10) **Persistent
Latent V18** (17 layers; fifteenth skip carrier `team_failure_recovery_
carrier`; max_chain_walk_depth=8192 carried forward; distractor rank
16); (M11) **Multi-Hop Translator V16** (36 backends; 1260 directed
edges; chain-length 26; 11-axis composite adding `team_substrate_
coordination_trust`); (M12) **Mergeable Latent Capsule V14**
(`team_failure_recovery_witness_chain` and `team_consensus_under_
budget_witness_chain`); (M13) **Consensus Fallback Controller V12**
(18-stage chain inserting `team_failure_recovery_arbiter` and
`team_consensus_under_budget_arbiter`); (M14) **Corruption-Robust
Carrier V14** (16384-bucket fingerprint; 33-bit adversarial burst;
team-failure-recovery probe); (M15) **Long-Horizon Retention V18**
(17 heads, max_k=320, eight-layer scorer adding random+swish); (M16)
**ECC Codebook V18** (K1..K17 = 2^29 = 536 870 912 codes; **31.333
bits/visible-token** at full emit); (M17) **Uncertainty Layer V14**
(13-axis composite adding `team_failure_recovery_fidelity`); (M18)
**Disagreement Algebra V12** (Jensen-Shannon-equivalence identity +
falsifier); (M19) **Fifteen-arm TVS Arbiter V14 + V15** (16 arms with
`team_failure_recovery` arm); (M20) **Multi-Agent Substrate
Coordinator V2** (the load-bearing W66 mechanism — six matched-budget
policies across three regimes; V11 strictly beats V10 on majority of
seeds; TSC_V11 strictly beats V11 on majority of seeds); (M21)
**Team-Consensus Controller** (first capsule-native team-consensus
controller — regime-aware weighted quorum + abstain + substrate-replay
fallback + transcript fallback).

The W66 `W66Team` orchestrator composes all twenty-one modules, emits
per-turn 28 module witness CIDs (one for the team-failure-recovery
falsifier, one for the MASC V2 aggregate, one for the team-consensus
controller), and seals them into a `W66HandoffEnvelope` whose
`w65_outer_cid` carries forward the W65 envelope byte-for-byte. The
W66 envelope verifier enumerates **123 disjoint failure modes**
(≥ 120 target met).

W66 is the **eleventh executable substrate-coupling milestone** in the
programme; it is NOT a claim of third-party transformer-internal
access. `W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries forward
unchanged. `W66-L-V11-NO-AUTOGRAD-CAP` is the new ridge-only cap: W66
fits **only** thirty-five closed-form linear ridge solves (twenty-nine
from W61..W65 + six new) — (a) cache controller V9 six-objective; (b)
cache controller V9 per-role eviction (planner); (c) replay V7 per-
role per-regime (planner) × 10 regimes; (d) replay V7 team-substrate-
routing head; (e) HSB V10 seven-target inner V6; (f) KV V11 seven-
target inner V6 — no SGD, no autograd, no GPU.

W66 ships at `coordpy.tiny_substrate_v11`, `coordpy.kv_bridge_v11`,
`coordpy.hidden_state_bridge_v10`,
`coordpy.prefix_state_bridge_v10`,
`coordpy.attention_steering_bridge_v10`,
`coordpy.cache_controller_v9`,
`coordpy.replay_controller_v7`,
`coordpy.persistent_latent_v18`,
`coordpy.multi_hop_translator_v16`,
`coordpy.mergeable_latent_capsule_v14`,
`coordpy.consensus_fallback_controller_v12`,
`coordpy.corruption_robust_carrier_v14`,
`coordpy.long_horizon_retention_v18`,
`coordpy.ecc_codebook_v18`,
`coordpy.transcript_vs_shared_arbiter_v15`,
`coordpy.uncertainty_layer_v14`,
`coordpy.disagreement_algebra_v12`,
`coordpy.deep_substrate_hybrid_v11`,
`coordpy.substrate_adapter_v11`,
`coordpy.multi_agent_substrate_coordinator_v2`,
`coordpy.team_consensus_controller`,
`coordpy.w66_team`, `coordpy.r146_benchmark`,
`coordpy.r147_benchmark`, `coordpy.r148_benchmark` — reachable only
through explicit imports. `coordpy.__version__` remains `0.5.20`; SDK
contract is byte-for-byte unchanged. **No PyPI release**.

R-146 (24 cell families) + R-147 (14 cell families) + R-148 (18 cell
families) at 3 seeds verify H245..H268. **56 of 56 H-bars pass 3/3
seeds (168/168 cells, strong success per the W66 success criterion)**.
Cumulative trust boundary across W22..W66 = **1228 enumerated failure
modes** (1105 from W22..W65 + 123 new W66 envelope verifier modes).

W66 envelope chain end-to-end: `W65 envelope CID ==
W66.w65_outer_cid` (verified live by
`test_w66_team_envelope_chain`). Trivial passthrough preserved
byte-for-byte (`test_w66_trivial_passthrough_byte_identical`).

W66 is the **second** programme milestone where substrate control
produces a **measurable multi-agent task-level win**, and the **first**
to produce wins under multiple failure-mode regimes (baseline +
team-consensus-under-budget + team-failure-recovery), not only under
the baseline regime. The team-failure-recovery regime is the sharpest
evidence: when one agent silently produces zero output mid-task, the
team-consensus controller running on the V11 substrate doubles the
team success rate over every baseline policy (80 % vs 40 %).

The honest scope is unchanged: W66 makes the **in-repo V11 substrate**
more load-bearing — four new internal axes, six new closed-form-ridge-
trained controllers/heads (thirty-five total across W61..W66), an
eleven-way bidirectional hybrid loop, three multi-agent task-success
regimes — but the **third-party hosted-model substrate remains
blocked** at the HTTP surface (`W66-L-NO-THIRD-PARTY-SUBSTRATE-
COUPLING-CAP`).

## TL;DR — W65 Team-Substrate-Coordination Substrate-Coupled Latent Operating System (post-W64 research milestone)

The programme now has **sixty-two** coupled research axes. W65
mints axis 62: the **tenth substrate-attack milestone** and the
**first multi-agent task-success-bearing** substrate milestone.
W65 turns the in-repo V10 substrate into a real multi-agent
coordination engine: a **MultiAgentSubstrateCoordinator** runs N
role-typed agents through the V10 substrate under four
matched-budget policies (`transcript_only` / `shared_state_proxy`
/ `substrate_routed_v9` / `substrate_routed_v10`); the V10 policy
*strictly beats* each baseline (lower error than transcript-only,
shared-proxy, and substrate-routed-V9) on ≥ 50 % of seeds and
saves ≥ 50 % of visible tokens versus transcript-only at matched
team-success.

Twenty orthogonal substrate-coupling, capsule-native, and
multi-agent-coordination advances on top of W64 — (M1) **Tiny
Transformer Runtime V10** (12 layers; four new V10 axes:
per-(L, H, T) hidden-write-merit, per-role KV bank with FIFO
eviction, substrate checkpoint/restore primitive, per-layer V10
composite gate score); (M2) **KV Bridge V10** (six-target stacked
ridge fit; substrate-measured per-target margin probe; team-task
falsifier); (M3) **HSB V9** (six-target stacked ridge; per-(L, H)
hidden-wins-rate probe; team-coordination margin); (M4)
**Prefix-State Bridge V9** (K=64 drift curve; role+task 20-dim
fingerprint; four-way prefix/hidden/replay/team comparator); (M5)
**Attention V9** (five-stage clamp: Hellinger + JS + coarse L1 +
fine KL + max-position cap; substrate-measured attention-map
fingerprint); (M6) **Cache Controller V8** (five-objective ridge
adding team-task-success; per-role 6-dim eviction head;
composite_v8); (M7) **Replay Controller V6** (8 regimes adding
`team_substrate_coordination_regime`; per-role per-regime ridge;
**multi-agent abstain head**); (M8) **Deep Substrate Hybrid V10**
(ten-way bidirectional loop with the four new V10 axes + MASC);
(M9) **Substrate Adapter V10** (4 new V10 capability axes; new
`substrate_v10_full` tier); (M10) **Persistent Latent V17** (16
layers, fourteenth skip carrier `team_task_success_carrier`,
`max_chain_walk_depth=8192`, distractor rank 14); (M11)
**Multi-Hop Translator V15** (35 backends, 1190 directed edges,
chain-length 25, 10-axis composite); (M12) **Mergeable Latent
Capsule V13** (`team_substrate_witness_chain` +
`role_conditioned_witness_chain`); (M13) **Consensus Fallback
Controller V11** (16-stage chain inserting
`team_substrate_coordination_arbiter` and
`multi_agent_abstain_arbiter`); (M14) **Corruption-Robust Carrier
V13** (8192-bucket fingerprint; 31-bit adversarial burst;
team-coordination recovery probe); (M15) **Long-Horizon Retention
V17** (16 heads, max_k=256, seven-layer scorer adding
random+softplus); (M16) **ECC Codebook V17** (K1..K16 = 2^27 =
134 217 728 codes; **29.333 bits/visible-token** at full emit);
(M17) **Uncertainty Layer V13** (12-axis composite adding
`team_coordination_fidelity`); (M18) **Disagreement Algebra V11**
(Wasserstein-equivalence identity + falsifier); (M19) **Fifteen-arm
TVS Arbiter V14** (adds `team_substrate_coordination` arm); (M20)
**Multi-Agent Substrate Coordinator** (the load-bearing W65
mechanism — real N-agent multi-agent harness with four
matched-budget policies, measurable team success rate, visible
tokens used, substrate recovery score).

The W65 `W65Team` orchestrator composes all twenty modules, emits
per-turn 24 module witness CIDs (one for the team-task-routing
falsifier, one for the MASC aggregate), and seals them into a
`W65HandoffEnvelope` whose `w64_outer_cid` carries forward the
W64 envelope byte-for-byte. The W65 envelope verifier enumerates
**103 disjoint failure modes** (≥ 100 target met).

W65 is the **tenth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. `W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`
carries forward unchanged. `W65-L-V10-NO-AUTOGRAD-CAP` is the
new ridge-only cap: W65 fits **only** twenty-nine closed-form
linear ridge solves (twenty-three from W61+W62+W63+W64 + six new) —
(a) cache controller V8 five-objective stacked; (b) cache V8
per-role eviction (planner); (c) replay V6 per-role per-regime
(planner) × 8 regimes; (d) replay V6 multi-agent abstain head; (e)
HSB V9 six-target inner V6; (f) KV V10 six-target inner V6 — no
SGD, no autograd, no GPU.

W65 ships at `coordpy.tiny_substrate_v10`, `coordpy.kv_bridge_v10`,
`coordpy.hidden_state_bridge_v9`,
`coordpy.prefix_state_bridge_v9`,
`coordpy.attention_steering_bridge_v9`,
`coordpy.cache_controller_v8`,
`coordpy.replay_controller_v6`,
`coordpy.persistent_latent_v17`,
`coordpy.multi_hop_translator_v15`,
`coordpy.mergeable_latent_capsule_v13`,
`coordpy.consensus_fallback_controller_v11`,
`coordpy.corruption_robust_carrier_v13`,
`coordpy.long_horizon_retention_v17`,
`coordpy.ecc_codebook_v17`,
`coordpy.transcript_vs_shared_arbiter_v14`,
`coordpy.uncertainty_layer_v13`,
`coordpy.disagreement_algebra_v11`,
`coordpy.deep_substrate_hybrid_v10`,
`coordpy.substrate_adapter_v10`,
`coordpy.multi_agent_substrate_coordinator`,
`coordpy.w65_team`,
`coordpy.r143_benchmark`, `coordpy.r144_benchmark`,
`coordpy.r145_benchmark` — reachable only through explicit
imports. `coordpy.__version__` remains `0.5.20`; SDK contract is
byte-for-byte unchanged. **No PyPI release**.

R-143 (22 cell families) + R-144 (12 cell families) + R-145 (16
cell families) at 3 seeds verify H223..H243b. **50 of 50 H-bars
pass 3/3 seeds (150/150 cells, strong success per the W65 success
criterion)**. Cumulative trust boundary across W22..W65 =
**1105 enumerated failure modes** (1002 from W22..W64 + 103 new
W65 envelope verifier modes).

W65 envelope chain end-to-end: `W64 envelope CID ==
W65.w64_outer_cid` (verified live by
`test_w65_team_envelope_chain`). Trivial passthrough preserved
byte-for-byte (`test_w65_trivial_passthrough_byte_identical`).

W65 is the **first** programme milestone where substrate control
produces a **measurable head-to-head multi-agent task-level win**
under a matched transcript budget rather than just internal
substrate probes. The `MultiAgentSubstrateCoordinator` runs four
matched-budget policies on the same synthetic deterministic task;
the V10 policy beats `transcript_only`, `shared_state_proxy`, and
`substrate_routed_v9` on ≥ 50 % of seeds while using ≤ 17 % of
the transcript-only visible-token budget. This is the load-bearing
claim of W65 — the in-repo substrate now matters at the
multi-agent task level, not just at the internal-probe level.

The honest scope is unchanged: W65 makes the **in-repo V10
substrate** more load-bearing — four new internal axes, six new
closed-form-ridge-trained controllers/heads (twenty-nine total
across W61..W65), a ten-way bidirectional hybrid loop, a real
multi-agent task-success bar — but the **third-party
hosted-model substrate remains blocked** at the HTTP surface
(`W65-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`).

## TL;DR — W64 Replay-Dominance-Primary Hidden-Wins-Primary 6144-Turn Nine-Way Substrate-Coupled Latent Operating System (post-W63 research milestone)

The programme now has **sixty-one** coupled research axes. W64
mints axis 61: the **ninth substrate-attack milestone** that
introduces a **per-(layer, head, slot) hidden-wins-primary
tensor** inside the in-repo NumPy substrate (V9), a **per-(layer,
head, slot) replay-dominance witness channel**, a **per-layer
attention-entropy probe** (deterministic sigmoid of attention
distribution Shannon entropy), a **per-(layer, head, slot, slot)
cache-similarity matrix** (cosine over cache_keys), a **per-(layer,
head) hidden-state-trust ledger** (EMA over HSB V8 decisions), a
**trained per-regime replay head with seven regimes** (replay
controller V5 with the new regime ``replay_dominance_primary_regime``
plus 10-feature candidate and 9-feature regime gate), a **trained
four-way bridge classifier** (4×9 ridge over the regime feature
space against ``kv_wins`` / ``hidden_wins`` / ``prefix_wins`` /
``replay_wins``), a **trained replay-dominance-primary head** (4×9
ridge that favours REUSE under high replay-dominance), a **trained
similarity-aware eviction head** (cache controller V7, 6-dim ridge
over [flag_count, hidden_write, replay_age, attention_receive_l1,
cache_key_norm, mean_similarity_to_others]), a **four-objective
stacked ridge fit** (drop-oracle + retrieval-relevance + hidden-wins
+ replay-dominance simultaneously), a **token+role-conditional
drift-curve predictor** (prefix V8, stacked 12×K ridge with role
fingerprint), a **four-stage attention clamp** (attention V8,
Hellinger budget + JS budget + coarse L1-mass + fine per-(L,H,Q,K)
KL), a **nine-way bidirectional substrate ↔ V9 ↔ cache controller
V7 ↔ replay controller V5 ↔ retrieval head ↔ attention-steering V8
↔ four-way bridge classifier ↔ prefix-state-bridge V8 ↔
hidden-state-bridge V8** hybrid loop (deep substrate hybrid V9),
and a **hidden-wins-primary falsifier** that returns 0 exactly
when inverting the primary flag flips the decision.

Nineteen orthogonal substrate-coupling and capsule-native advances
layered on top of W63 — (M1) **Tiny Transformer Runtime V9**
(``coordpy.tiny_substrate_v9``): 11 layers (vs V8's 10), GQA 8/4,
five new internal axes — per-(layer, head, slot) hidden-wins-
primary tensor, per-(layer, head, slot) replay-dominance witness
channel, per-layer attention-entropy probe, per-(layer, head, slot,
slot) cache-similarity matrix, per-(layer, head) hidden-state-trust
ledger; (M2) **KV Bridge V9** (``coordpy.kv_bridge_v9``):
five-target stacked ridge fit (4 V8 targets + 1 replay-dominance-
primary target); per-bucket hidden-wins-primary falsifier; KV-
fingerprint perturbation control; (M3) **HSB V8**
(``coordpy.hidden_state_bridge_v8``): five-target stacked ridge
with explicit hidden-wins-primary target column; recovery audit V4
(three-stage with widening basin); V9 hidden-state-trust coupling;
(M4) **Prefix-State Bridge V8**
(``coordpy.prefix_state_bridge_v8``): token+role-conditional
12-feature stacked drift-curve predictor; prefix-vs-hidden-vs-
replay three-way comparator; V9 hidden-state-trust coupling;
(M5) **Attention-Steering Bridge V8**
(``coordpy.attention_steering_bridge_v8``): four-stage clamp
(Hellinger + JS + coarse L1 + fine KL); per-bucket entropy-
amplified falsifier; (M6) **Cache Controller V7**
(``coordpy.cache_controller_v7``): three new policies on top of
V6's ten — ``four_objective_v7`` (stacked drop-oracle +
retrieval-relevance + hidden-wins + replay-dominance ridge),
``similarity_eviction_v7`` (6-dim ridge head),
``composite_v7`` (8-head fitted mixture); (M7) **Replay Controller
V5** (``coordpy.replay_controller_v5``): per-regime 10×4 ridge
head fit across 7 regimes; 9-dim regime gate; four-way bridge 4×9
classifier; replay-dominance-primary 4×9 head; replay-dominance-
primary REUSE bonus; (M8) **Deep Substrate Hybrid V9**
(``coordpy.deep_substrate_hybrid_v9``): nine-way bidirectional
loop with ``nine_way=True`` only when all nine axes contribute;
(M9) **Substrate Adapter V9** (``coordpy.substrate_adapter_v9``):
5 new capability axes; new ``substrate_v9_full`` tier;
(M10) **Persistent Latent V16** (``coordpy.persistent_latent_v16``):
15 outer layers, thirteenth skip carrier (replay-dominance-witness
EMA), ``max_chain_walk_depth=6144``, distractor rank 12;
(M11) **Multi-Hop Translator V14**
(``coordpy.multi_hop_translator_v14``): 27 backends, 702 directed
edges, chain-length-21, nine-axis trust composite adding
``replay_dominance_primary_trust``, 1 ≤ threshold ≤ 9;
(M12) **Mergeable Latent State Capsule V12**
(``coordpy.mergeable_latent_capsule_v12``): adds
``replay_dominance_primary_witness_chain``,
``hidden_state_trust_witness_chain``,
``disagreement_total_variation_distance``; two new algebra
signatures; (M13) **Consensus Fallback Controller V10**
(``coordpy.consensus_fallback_controller_v10``): 14-stage chain
inserting ``replay_dominance_primary_arbiter`` between
``hidden_wins_arbiter`` and ``best_parent``;
(M14) **Corruption-Robust Carrier V12**
(``coordpy.corruption_robust_carrier_v12``): 4096-bucket wrap-
around-XOR fingerprint; 23-bit adversarial burst; replay-dominance
recovery L1-ratio probe; substrate-corruption blast-radius probe;
(M15) **Long-Horizon Retention V16**
(``coordpy.long_horizon_retention_v16``): 15 heads, max_k=192,
replay-dominance-primary-conditioned head, six-layer scorer
(random+ReLU → random+tanh → random+tanh-2 → random+gelu →
random+silu → ridge); (M16) **ECC Codebook V16**
(``coordpy.ecc_codebook_v16``): K1..K15 = 2^25 = 33 554 432 codes;
27.333 bits/visible-token at full emit (≥ 27.0 target);
(M17) **Uncertainty Layer V12** (``coordpy.uncertainty_layer_v12``):
11th weighting axis ``replay_dominance_primary_fidelity``;
(M18) **Disagreement Algebra V10** (``coordpy.disagreement_algebra_v10``):
total-variation equivalence identity + falsifier;
(M19) **Fourteen-arm TVS Arbiter V13**
(``coordpy.transcript_vs_shared_arbiter_v13``): adds
``replay_dominance_primary`` arm.

The W64 ``W64Team`` orchestrator composes all nineteen modules,
emits per-turn 23 module witness CIDs (one for the hidden-wins-
primary falsifier), and seals them into a ``W64HandoffEnvelope``
whose ``w63_outer_cid`` carries forward the W63 envelope byte-for-
byte. The W64 envelope verifier enumerates **86 disjoint failure
modes** (≥ 85 target met).

W64 is the **ninth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
carries forward unchanged. ``W64-L-V9-NO-AUTOGRAD-CAP`` is the
new ridge-only cap: W64 fits **only** twenty-three closed-form
linear ridge solves (seventeen from W61+W62+W63 + six new) — (a)
cache controller V7 four-objective stacked; (b) cache controller
V7 similarity-aware eviction head; (c) cache controller V7
composite_v7 mixture; (d) replay controller V5 per-regime head ×
7 regimes; (e) replay controller V5 four-way bridge classifier;
(f) replay controller V5 replay-dominance-primary head — no SGD,
no autograd, no GPU.

W64 ships at ``coordpy.tiny_substrate_v9``, ``coordpy.kv_bridge_v9``,
``coordpy.hidden_state_bridge_v8``,
``coordpy.prefix_state_bridge_v8``,
``coordpy.attention_steering_bridge_v8``,
``coordpy.cache_controller_v7``, ``coordpy.replay_controller_v5``,
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
``coordpy.r137_benchmark``, ``coordpy.r138_benchmark``,
``coordpy.r139_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. **No PyPI release**.

R-137 (20 cell families) + R-138 (12 cell families) + R-139 (19
cell families) at 3 seeds verify H203..H222b. **51 of 51 H-bars
pass 3/3 seeds (153/153 cells, strong success per the W64 success
criterion)**. Cumulative trust boundary across W22..W64 =
**1002 enumerated failure modes** (916 from W22..W63 + 86 new W64
envelope verifier modes).

W64 envelope chain end-to-end: ``W63 envelope CID ==
W64.w63_outer_cid`` (verified live by
``test_w64_team_envelope_chain``). Trivial passthrough preserved
byte-for-byte (``test_w64_trivial_passthrough_byte_identical``).

W64 is the strongest honest substrate-coupling milestone the
programme has shipped. The **seven-regime per-regime replay head**
in replay controller V5 is the first time the
REUSE/RECOMPUTE/FALLBACK/ABSTAIN policy is derived from *separate*
fitted heads keyed on seven regimes (including a replay-dominance-
primary regime). The **four-way bridge classifier** is the first
time the programme has fit a 4-class regime classifier over the
nine-feature regime space against the kv_wins / hidden_wins /
prefix_wins / replay_wins label. The **four-objective ridge fit**
in cache controller V7 is the first time the controller stacks
four real targets. The **trained similarity-aware eviction head**
is the first time the controller learns a per-slot 6-feature
additive priority correction conditioned on cache-key cosine
similarity to other slots. The **token+role-conditional drift-
curve predictor** in prefix V8 is the first time the predictor's
feature space differentiates between (config, follow-up tokens,
role) tuples.

The honest scope is unchanged: W64 makes the **in-repo V9
substrate** much more load-bearing — five new internal axes, six
new closed-form-ridge-trained controllers/heads (twenty-three total
across W61+W62+W63+W64), a nine-way bidirectional hybrid loop —
but the **third-party hosted-model substrate remains blocked**
at the HTTP surface (``W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``).

## TL;DR — W63 Stronger Replay-Dominance Hidden-Wins 4096-Turn Substrate-Coupled Latent Operating System (post-W62 research milestone)

The programme now has **sixty** coupled research axes. W63 mints
axis 60: the **eighth substrate-attack milestone** that introduces
a **per-(layer, head, slot) hidden-vs-KV contention tensor** inside
the in-repo NumPy substrate (V8), a **per-layer hidden-state
confidence probe** (deterministic sigmoid of logit-lens entropy), a
**per-(layer, head, slot) replay-determinism channel** (stable-
write flag across forwards), a **per-(layer, head) prefix-reuse
trust ledger**, a **per-(L, H, L, H) cross-layer-head coupling
matrix**, a **trained per-regime replay head with six regimes**
(replay controller V4 with the two new regimes
``hidden_wins_regime`` + ``cache_corruption_recovered`` plus eight-
feature candidate and seven-feature regime gate), a **trained
three-way bridge classifier** (7×3 ridge over the regime feature
space against ``kv_wins`` / ``hidden_wins`` / ``prefix_wins``), a
**trained retrieval-repair head** (cache controller V6, 5-dim
ridge over [flag_count, hidden_write, replay_age,
attention_receive_l1, cache_key_norm]), a **three-objective stacked
ridge fit** (drop-oracle + retrieval-relevance + hidden-wins
simultaneously), a **token-content-conditional drift-curve
predictor** (prefix V7, stacked 8×K ridge with 4-D SHA256 token
fingerprint), a **three-stage attention clamp** (attention V7,
Jensen-Shannon budget + coarse L1-mass + fine per-(L,H,Q,K) KL), an
**eight-way bidirectional substrate ↔ V8 ↔ cache controller V6 ↔
replay controller V4 ↔ retrieval head ↔ attention-steering V7 ↔
three-way bridge classifier ↔ prefix-state-bridge V7** hybrid loop
(deep substrate hybrid V8), and a **hidden-wins falsifier** that
returns 0 exactly when inverting the residual roles flips the
decision.

Nineteen orthogonal substrate-coupling and capsule-native advances
layered on top of W62 — (M1) **Tiny Transformer Runtime V8**
(``coordpy.tiny_substrate_v8``): 10 layers (vs V7's 9), GQA 8/4,
five new internal axes — per-(layer, head, slot) hidden-vs-KV
contention tensor, per-layer hidden-state confidence probe,
per-(layer, head, slot) replay-determinism channel, per-(layer,
head) prefix-reuse trust ledger, per-(L, H, L, H) cross-layer-head
coupling matrix; (M2) **KV Bridge V8** (``coordpy.kv_bridge_v8``):
four-target stacked ridge fit (3 V7 targets + 1 hidden-wins
target); V8 contention coupling; per-bucket hidden-wins falsifier;
(M3) **HSB V7** (``coordpy.hidden_state_bridge_v7``): four-target
stacked ridge with explicit hidden-wins target column; recovery
audit V3 (two-stage with basin width); V8 contention coupling;
(M4) **Prefix-State Bridge V7** (``coordpy.prefix_state_bridge_v7``):
token-content-conditional 8-feature stacked drift-curve predictor;
prefix-vs-hidden three-way comparator; V8 prefix-reuse trust
coupling; (M5) **Attention-Steering Bridge V7**
(``coordpy.attention_steering_bridge_v7``): three-stage clamp
(JS budget + coarse L1 + fine KL); per-bucket cosine-aligned
falsifier; (M6) **Cache Controller V6**
(``coordpy.cache_controller_v6``): three new policies on top of
V5's seven — ``three_objective_v6`` (stacked drop-oracle +
retrieval-relevance + hidden-wins ridge), ``retrieval_repair_v6``
(5-dim ridge head), ``composite_v6`` (7-head fitted mixture);
(M7) **Replay Controller V4** (``coordpy.replay_controller_v4``):
per-regime 8×4 ridge head fit across 6 regimes; 7-dim regime gate;
three-way bridge 7×3 classifier; replay-determinism bonus on
REUSE; (M8) **Deep Substrate Hybrid V8**
(``coordpy.deep_substrate_hybrid_v8``): eight-way bidirectional
loop with ``eight_way=True`` only when all eight axes contribute;
(M9) **Substrate Adapter V8** (``coordpy.substrate_adapter_v8``):
5 new capability axes; new ``substrate_v8_full`` tier;
(M10) **Persistent Latent V15** (``coordpy.persistent_latent_v15``):
14 outer layers, twelfth-and-thirteenth skip carriers (hidden-wins
EMA + prefix-reuse EMA), ``max_chain_walk_depth=4096``, distractor
rank 10; (M11) **Multi-Hop Translator V13**
(``coordpy.multi_hop_translator_v13``): 24 backends, 552 directed
edges, chain-length-19, eight-axis trust composite adding
``hidden_wins_trust``, 1 ≤ threshold ≤ 8;
(M12) **Mergeable Latent State Capsule V11**
(``coordpy.mergeable_latent_capsule_v11``): adds
``hidden_wins_witness_chain``, ``prefix_reuse_witness_chain``,
``disagreement_jensen_shannon_distance``; two new algebra
signatures; (M13) **Consensus Fallback Controller V9**
(``coordpy.consensus_fallback_controller_v9``): 13-stage chain
inserting ``hidden_wins_arbiter`` between ``trained_repair`` and
``best_parent``; (M14) **Corruption-Robust Carrier V11**
(``coordpy.corruption_robust_carrier_v11``): 2048-bucket wrap-
around-XOR fingerprint; 19-bit adversarial burst; hidden-state
recovery L2-ratio probe; (M15) **Long-Horizon Retention V15**
(``coordpy.long_horizon_retention_v15``): 14 heads, max_k=160,
hidden-wins-conditioned head, five-layer scorer (random+ReLU →
random+tanh → random+tanh-2 → random+gelu → ridge);
(M16) **ECC Codebook V15** (``coordpy.ecc_codebook_v15``):
K1..K14 = 2^24 = 16 777 216 codes; 26.333 bits/visible-token at
full emit (≥ 26.0 target); (M17) **Uncertainty Layer V11**
(``coordpy.uncertainty_layer_v11``): 10th weighting axis
``hidden_wins_fidelity``;
(M18) **Disagreement Algebra V9** (``coordpy.disagreement_algebra_v9``):
Jensen-Shannon equivalence identity + falsifier;
(M19) **Thirteen-arm TVS Arbiter V12**
(``coordpy.transcript_vs_shared_arbiter_v12``): adds
``hidden_wins`` arm.

The W63 ``W63Team`` orchestrator composes all nineteen modules,
emits per-turn 22 module witness CIDs (one for the hidden-wins
falsifier), and seals them into a ``W63HandoffEnvelope`` whose
``w62_outer_cid`` carries forward the W62 envelope byte-for-byte.
The W63 envelope verifier enumerates **72 disjoint failure modes**
(≥ 72 target met).

W63 is the **eighth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
carries forward unchanged. ``W63-L-V8-NO-AUTOGRAD-CAP`` is the
new ridge-only cap: W63 fits **only** seventeen closed-form linear
ridge solves (twelve from W61+W62 + five new) — (a) cache
controller V6 three-objective stacked; (b) cache controller V6
retrieval-repair head; (c) cache controller V6 composite_v6
mixture; (d) replay controller V4 per-regime head × 6 regimes;
(e) replay controller V4 three-way bridge classifier — no SGD, no
autograd, no GPU.

W63 ships at ``coordpy.tiny_substrate_v8``, ``coordpy.kv_bridge_v8``,
``coordpy.hidden_state_bridge_v7``,
``coordpy.prefix_state_bridge_v7``,
``coordpy.attention_steering_bridge_v7``,
``coordpy.cache_controller_v6``, ``coordpy.replay_controller_v4``,
``coordpy.persistent_latent_v15``,
``coordpy.multi_hop_translator_v13``,
``coordpy.mergeable_latent_capsule_v11``,
``coordpy.consensus_fallback_controller_v9``,
``coordpy.corruption_robust_carrier_v11``,
``coordpy.long_horizon_retention_v15``,
``coordpy.ecc_codebook_v15``,
``coordpy.transcript_vs_shared_arbiter_v12``,
``coordpy.uncertainty_layer_v11``,
``coordpy.disagreement_algebra_v9``,
``coordpy.deep_substrate_hybrid_v8``,
``coordpy.substrate_adapter_v8``, ``coordpy.w63_team``,
``coordpy.r134_benchmark``, ``coordpy.r135_benchmark``,
``coordpy.r136_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. **No PyPI release**.

R-134 (17 cell families) + R-135 (16 cell families) + R-136 (16
cell families) at 3 seeds verify H181..H202b. **49 of 49 H-bars
pass 3/3 seeds (147/147 cells, strong success per the W63 success
criterion)**. Cumulative trust boundary across W22..W63 =
**916 enumerated failure modes** (844 from W22..W62 + 72 new W63
envelope verifier modes).

W63 envelope chain end-to-end: ``W62 envelope CID ==
W63.w62_outer_cid`` (verified live by
``test_w63_team_envelope_chain``). Trivial passthrough preserved
byte-for-byte (``test_w63_trivial_passthrough_byte_identical``).

W63 is the strongest honest substrate-coupling milestone the
programme has shipped. The **six-regime per-regime replay head**
in replay controller V4 is the first time the
REUSE/RECOMPUTE/FALLBACK/ABSTAIN policy is derived from *separate*
fitted heads keyed on six regimes (including a hidden-wins regime
and a cache-corruption-recovered regime). The **three-way bridge
classifier** is the first time the programme has fit a 3-class
regime classifier over the seven-feature regime space against the
kv_wins / hidden_wins / prefix_wins label. The **three-objective
ridge fit** in cache controller V6 is the first time the controller
stacks three real targets. The **trained retrieval-repair head**
is the first time the controller learns a per-slot 5-feature
additive repair correction conditioned on a cache-key norm. The
**token-content-conditional drift-curve predictor** in prefix V7
is the first time the predictor's feature space differentiates
between follow-up token configurations.

The honest scope is unchanged: W63 makes the **in-repo V8
substrate** much more load-bearing — five new internal axes, five
new closed-form-ridge-trained controllers/heads (seventeen total
across W61+W62+W63), an eight-way bidirectional hybrid loop —
but the **third-party hosted-model substrate remains blocked**
at the HTTP surface (``W63-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``).

## TL;DR — W62 Trainable Replay-Dominance Hidden-vs-KV Substrate-Coupled Latent Operating System (post-W61 research milestone)

The programme now has **fifty-nine** coupled research axes. W62
mints axis 59: the **seventh substrate-attack milestone** that
introduces a **per-(layer, head, slot) cache-write ledger axis**
inside the in-repo NumPy substrate (V7), a **per-layer logit-lens
probe**, a **per-(layer, head, position) attention-receive delta**
channel, a **per-(layer, head) replay-trust ledger**, a **trained
per-regime replay head** (replay controller V3 with 4 regime
heads + nearest-centroid regime gate), a **trained hidden-vs-KV
regime classifier** (5×3 ridge with ≥ 0.8 training accuracy on
synthetic supervision), a **trained corruption-repair head**
(cache controller V5, 4-dim ridge), a **two-objective stacked
ridge fit** (drop-oracle + retrieval-relevance simultaneously),
a **drift-curve predictor** (prefix V6, stacked 4×K ridge), a
**two-stage attention clamp** (attention V6, coarse L1-mass +
fine per-(L,H,Q,K) KL), and a **seven-way bidirectional
substrate ↔ V7 ↔ cache controller V5 ↔ replay controller V3 ↔
retrieval head ↔ attention-steering V6 ↔ hidden-vs-KV classifier**
hybrid loop (deep substrate hybrid V7).

Nineteen orthogonal substrate-coupling and capsule-native advances
layered on top of W61 — (M1) **Tiny Transformer Runtime V7**
(``coordpy.tiny_substrate_v7``): 9 layers (vs V6's 8), GQA 8/4,
four new internal axes — per-(layer, head, slot) cache-write
ledger, per-layer logit-lens probe, per-(layer, head, position)
attention-receive *delta* (forward-to-forward difference), per-
(layer, head) replay-trust ledger updated via EMA over replay
decisions; (M2) **KV Bridge V7**
(``coordpy.kv_bridge_v7``): additive V7 layer-d correction;
three-target stacked ridge fit; V7 cache-write ledger coupling;
hidden-vs-KV decision tap; (M3) **HSB V6**
(``coordpy.hidden_state_bridge_v6``): three-target stacked
ridge over (L, H, P) δ tensor; V7 ledger coupling; recovery
audit V2 with post-recovery margin; (M4) **Prefix-State Bridge V6**
(``coordpy.prefix_state_bridge_v6``): drift-curve predictor — stacked
4×K closed-form ridge over [reuse_len, recompute_len, drop_len, 1]
against K-step drift curve target; (M5) **Attention-Steering Bridge V6**
(``coordpy.attention_steering_bridge_v6``): two-stage clamp (coarse
L1-mass + fine per-(L,H,Q,K) KL); per-bucket signed-coefficient
falsifier; (M6) **Cache Controller V5**
(``coordpy.cache_controller_v5``): three new policies on top of V4's
four — ``two_objective_v5`` (stacked drop-oracle + retrieval-
relevance ridge), ``trained_repair_v5`` (4-dim ridge head over
[flag_count, hidden_write, replay_age, attention_receive_l1]
against repair amount), ``composite_v5`` (6-head fitted mixture);
(M7) **Replay Controller V3** (``coordpy.replay_controller_v3``):
per-regime 6×4 ridge head fit across 4 regimes; nearest-centroid
regime gate; hidden-vs-KV 5×3 classifier; replay-dominance scalar
= softmax margin; (M8) **Deep Substrate Hybrid V7**
(``coordpy.deep_substrate_hybrid_v7``): seven-way bidirectional
loop with ``seven_way=True`` only when all seven axes contribute;
(M9) **Substrate Adapter V7**
(``coordpy.substrate_adapter_v7``): 4 new capability axes; new
``substrate_v7_full`` tier; (M10) **Persistent Latent V14**
(``coordpy.persistent_latent_v14``): 12 outer layers, decuple skip
with replay-dominance EMA, max_chain_walk_depth=2048, distractor
rank 8; (M11) **Multi-Hop Translator V12**
(``coordpy.multi_hop_translator_v12``): 20 backends, 380 directed
edges, chain-length-17, seven-axis trust composite adding
``replay_dominance_trust``, 1 ≤ threshold ≤ 7;
(M12) **Mergeable Latent State Capsule V10**
(``coordpy.mergeable_latent_capsule_v10``): adds
``replay_dominance_witness_chain``, ``disagreement_wasserstein_distance``;
two new algebra signatures; (M13) **Consensus Fallback Controller V8**
(``coordpy.consensus_fallback_controller_v8``): 12-stage chain
inserting ``trained_repair`` between ``attention_pattern_consensus``
and ``best_parent``; (M14) **Corruption-Robust Carrier V10**
(``coordpy.corruption_robust_carrier_v10``): 1024-bucket wrap-
around-XOR fingerprint; 17-bit adversarial burst; post-repair
top-K Jaccard; (M15) **Long-Horizon Retention V14**
(``coordpy.long_horizon_retention_v14``): 13 heads, max_k=128,
replay-dominance-conditioned head, four-layer scorer
(random+ReLU → random+tanh → random+tanh-2 → ridge);
(M16) **ECC Codebook V14** (``coordpy.ecc_codebook_v14``): K1..K13 =
2^23 = 8 388 608 codes; 25.333 bits/visible-token at full emit
(≥ 25.0 target); (M17) **Uncertainty Layer V10**
(``coordpy.uncertainty_layer_v10``): 9th weighting axis
``replay_dominance_fidelity``;
(M18) **Disagreement Algebra V8** (``coordpy.disagreement_algebra_v8``):
Wasserstein-1-equivalence identity + falsifier;
(M19) **Twelve-arm TVS Arbiter V11**
(``coordpy.transcript_vs_shared_arbiter_v11``): adds
``replay_dominance`` arm.

The W62 ``W62Team`` orchestrator composes all nineteen modules,
emits per-turn 20 module witness CIDs, and seals them into a
``W62HandoffEnvelope`` whose ``w61_outer_cid`` carries forward the
W61 envelope byte-for-byte. The W62 envelope verifier enumerates
**68 disjoint failure modes** (≥ 65 target met).

W62 is the **seventh executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
carries forward unchanged. ``W62-L-V7-NO-AUTOGRAD-CAP`` is the
new ridge-only cap: W62 fits **only** twelve closed-form linear
ridge solves (seven from W61 + five new) — (a) cache controller
V5 two-objective stacked; (b) cache controller V5 trained-repair
head; (c) cache controller V5 composite_v5 mixture; (d) replay
controller V3 per-regime head × 4 regimes; (e) replay controller
V3 hidden-vs-KV 3-class classifier — no SGD, no autograd, no GPU.

W62 ships at ``coordpy.tiny_substrate_v7``, ``coordpy.kv_bridge_v7``,
``coordpy.hidden_state_bridge_v6``,
``coordpy.prefix_state_bridge_v6``,
``coordpy.attention_steering_bridge_v6``,
``coordpy.cache_controller_v5``, ``coordpy.replay_controller_v3``,
``coordpy.persistent_latent_v14``,
``coordpy.multi_hop_translator_v12``,
``coordpy.mergeable_latent_capsule_v10``,
``coordpy.consensus_fallback_controller_v8``,
``coordpy.corruption_robust_carrier_v10``,
``coordpy.long_horizon_retention_v14``,
``coordpy.ecc_codebook_v14``,
``coordpy.transcript_vs_shared_arbiter_v11``,
``coordpy.uncertainty_layer_v10``,
``coordpy.disagreement_algebra_v8``,
``coordpy.deep_substrate_hybrid_v7``,
``coordpy.substrate_adapter_v7``, ``coordpy.w62_team``,
``coordpy.r131_benchmark``, ``coordpy.r132_benchmark``,
``coordpy.r133_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. **No PyPI release**.

R-131 (13 cell families) + R-132 (12 cell families) + R-133 (20
cell families) at 3 seeds verify H163..H180. **45 of 45 H-bars
pass 3/3 seeds (135/135 cells, strong success per the W62 success
criterion)**. Cumulative trust boundary across W22..W62 =
**844 enumerated failure modes** (776 from W22..W61 + 68 new W62
envelope verifier modes).

W62 envelope chain end-to-end: ``W61 envelope CID ==
W62.w61_outer_cid`` (verified live by
``test_w62_team_envelope_chain``). Trivial passthrough preserved
byte-for-byte (``test_w62_trivial_passthrough_byte_identical``).

W62 is the strongest honest substrate-coupling milestone the
programme has shipped. The **per-regime replay head** in replay
controller V3 is the first time the REUSE/RECOMPUTE/FALLBACK/
ABSTAIN policy is derived from *separate* fitted heads keyed on
regime. The **hidden-vs-KV classifier** is the first time the
programme has fit a 3-class regime classifier over a regime
feature space — a fitted version of the deterministic decision
in ``compare_hidden_vs_kv_v7``. The **two-objective ridge fit**
in cache controller V5 is the first time the controller stacks
two real targets. The **trained corruption-repair head** is the
first time the controller learns a per-slot additive repair
correction (not just a floor). The **drift-curve predictor** in
prefix V6 is the first time the predictor outputs a stacked
K-step curve.

The honest scope is unchanged: W62 makes the **in-repo V7
substrate** much more load-bearing — four new internal axes,
five new closed-form-ridge-trained controllers/heads (twelve
total across W61+W62), a seven-way bidirectional hybrid loop —
but the **third-party hosted-model substrate remains blocked**
at the HTTP surface (``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``).

## TL;DR — W61 Trainable Hidden-State Substrate-Coupled Latent Operating System (post-W60 research milestone)

The programme now has **fifty-eight** coupled research axes. W61
mints axis 58: the **sixth substrate-attack milestone** that
introduces a **content-addressable cache-key axis** inside the
in-repo NumPy substrate, a **bilinear retrieval head** fit by
closed-form ridge over the (query ⊗ cache_key) outer-product
feature (cache controller V4), a **trained ridge replay-threshold
head** (replay controller V2) with softmax decision-confidence and
an abstain-on-confidence gate, a **per-(layer, head, query, key)
4-D attention-budget tensor** with a signed-coefficient falsifier
(attention steering V5), **multi-target stacked HSB fit** (HSB V5
3-D δ tensor against m stacked target logit directions),
**attention-pattern-target KV fit** (KV bridge V6 steers the
substrate's last-row attention map toward a reference via ridge),
and a **six-way bidirectional substrate ↔ V6 ↔ cache controller
V4 ↔ replay controller V2 ↔ retrieval head ↔ attention-steering
V5** hybrid loop (deep substrate hybrid V6).

Nineteen orthogonal substrate-coupling and capsule-native advances
layered on top of W60 — (M1) **Tiny Transformer Runtime V6**
(``coordpy.tiny_substrate_v6``): 8 layers, GQA 8/4, five new
internal axes — per-(layer, position, d_key=8) content-
addressable **cache_keys** tensor derived from a fixed projection
of V3 layer keys, per-(layer, head) cumulative **hidden_write_trace**
populated by HSB V5 writes, per-(layer, position) **replay_age**
channel auto-incrementing every forward, **forward_count**
integer, per-(layer_i, layer_j) **cross_layer_coupling** diagnostic;
(M2) **KV Bridge V6** (``coordpy.kv_bridge_v6``): matrix-valued
multi-target closed-form ridge fit; attention-pattern target fit
on the substrate's last-row attention L1 mass; V6 cache-key
128-bucket fingerprint; (M3) **HSB V5** (``coordpy.hidden_state_bridge_v5``):
per-(layer, head, position) 3-D δ tensor with multi-target stack
fit; HSB→V6-cache hidden-write coupling; V4-recovery delegation;
(M4) **Prefix-State Bridge V5** (``coordpy.prefix_state_bridge_v5``):
chain-of-chains over V6 substrates; trained 3-feature ridge drift
predictor (predicts step-1 drift L2 from reuse / recompute / drop
segment lengths); V6 cache-key fingerprint in the witness;
(M5) **Attention-Steering Bridge V5**
(``coordpy.attention_steering_bridge_v5``): per-(layer, head,
query, key) 4-D budget tensor; signed-coefficient falsifier;
attention-map L1 + L2 + top-K Jaccard observables; negative-
budget falsifier returns post-KL exactly 0; (M6) **Cache
Controller V4** (``coordpy.cache_controller_v4``): four new
policies on top of V3's seven — ``bilinear_retrieval_v6`` (matrix
ridge over (query ⊗ cache_key) outer-product feature),
``trained_corruption_floor`` (quadratic ridge), ``two_stage_v4``
(coarse + fine), ``composite_v4`` (5-head fitted mixture);
(M7) **Replay Controller V2** (``coordpy.replay_controller_v2``):
closed-form linear ridge over 6-dim feature against 4-class label
one-hot; softmax decision-confidence; abstain-on-confidence
threshold; hidden-write-cap gate; (M8) **Deep Substrate Hybrid V6**
(``coordpy.deep_substrate_hybrid_v6``): six-way bidirectional
loop with ``six_way=True`` only when all six axes contribute;
(M9) **Substrate Adapter V6** (``coordpy.substrate_adapter_v6``):
7 new capability axes; new ``substrate_v6_full`` tier;
(M10) **Persistent Latent V13** (``coordpy.persistent_latent_v13``):
11 outer layers, nonuple skip with replay-confidence EMA,
max_chain_walk_depth=1536, distractor rank 6;
(M11) **Multi-Hop Translator V11** (``coordpy.multi_hop_translator_v11``):
18 backends, 306 directed edges, chain-length-16, six-axis trust
composite adding ``attention_pattern_fidelity``, 1..6 compromise
threshold; (M12) **Mergeable Latent State Capsule V9**
(``coordpy.mergeable_latent_capsule_v9``): adds
``attention_pattern_witness_chain``,
``cache_retrieval_witness_chain``, per-(layer, head) trust
matrix; two new algebra signatures; (M13) **Consensus Fallback
Controller V7** (``coordpy.consensus_fallback_controller_v7``):
11-stage chain inserting ``attention_pattern_consensus``;
(M14) **Corruption-Robust Carrier V9**
(``coordpy.corruption_robust_carrier_v9``): 512-bucket wrap-
around-XOR fingerprint; 13-bit adversarial burst; post-replay
top-K Jaccard; (M15) **Long-Horizon Retention V13**
(``coordpy.long_horizon_retention_v13``): 12 heads, max_k=128,
attention-pattern-conditioned head, three-layer scorer (random+ReLU
→ random+tanh → ridge); (M16) **ECC Codebook V13**
(``coordpy.ecc_codebook_v13``): K1..K12 = 2^22 = 4 194 304 codes;
24.333 bits/visible-token at full emit (≥ 24.0 target);
(M17) **Uncertainty Layer V9** (``coordpy.uncertainty_layer_v9``):
8th weighting axis ``attention_pattern_fidelity``;
(M18) **Disagreement Algebra V7** (``coordpy.disagreement_algebra_v7``):
attention-pattern-equivalence identity + falsifier;
(M19) **Eleven-arm TVS Arbiter V10**
(``coordpy.transcript_vs_shared_arbiter_v10``): adds
``attention_pattern_steer`` arm.

The W61 ``W61Team`` orchestrator composes all nineteen modules,
emits per-turn 20 module witness CIDs, and seals them into a
``W61HandoffEnvelope`` whose ``w60_outer_cid`` carries forward the
W60 envelope byte-for-byte. The W61 envelope verifier enumerates
**61 disjoint failure modes** (≥ 55 target met).

W61 is the **sixth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
carries forward unchanged. ``W61-L-V6-NO-AUTOGRAD-CAP`` is the
new ridge-only cap: W61 fits **only** seven closed-form linear
ridge solves — (a) KV bridge V6 multi-target; (b) KV bridge V6
attention-pattern; (c) HSB V5 multi-target stack; (d) cache
controller V4 bilinear retrieval matrix; (e) cache controller V4
trained corruption floor; (f) replay controller V2 threshold head;
(g) LHR V13 third-layer scorer — no SGD, no autograd, no GPU.

W61 ships at ``coordpy.tiny_substrate_v6``, ``coordpy.kv_bridge_v6``,
``coordpy.hidden_state_bridge_v5``,
``coordpy.prefix_state_bridge_v5``,
``coordpy.attention_steering_bridge_v5``,
``coordpy.cache_controller_v4``, ``coordpy.replay_controller_v2``,
``coordpy.persistent_latent_v13``,
``coordpy.multi_hop_translator_v11``,
``coordpy.mergeable_latent_capsule_v9``,
``coordpy.consensus_fallback_controller_v7``,
``coordpy.corruption_robust_carrier_v9``,
``coordpy.long_horizon_retention_v13``,
``coordpy.ecc_codebook_v13``,
``coordpy.transcript_vs_shared_arbiter_v10``,
``coordpy.uncertainty_layer_v9``,
``coordpy.disagreement_algebra_v7``,
``coordpy.deep_substrate_hybrid_v6``,
``coordpy.substrate_adapter_v6``, ``coordpy.w61_team``,
``coordpy.r128_benchmark``, ``coordpy.r129_benchmark``,
``coordpy.r130_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. **No PyPI release**.

R-128 (25 cell families) + R-129 (15 cell families) + R-130 (12
cell families) at 3 seeds verify H144..H162c. **52 of 52 H-bars
pass 3/3 seeds (156/156 cells, strong success per the W61
success criterion)**. Cumulative trust boundary across W22..W61 =
**776 enumerated failure modes** (715 from W22..W60 + 61 new W61
envelope verifier modes).

W61 envelope chain end-to-end: ``W60 envelope CID ==
W61.w60_outer_cid`` (verified live by
``test_w61_team_envelope_chain``). Trivial passthrough preserved
byte-for-byte (``test_w61_trivial_passthrough_byte_identical``).

W61 is the strongest honest substrate-coupling milestone the
programme has shipped. The **bilinear retrieval head** in cache
controller V4 is the first time the controller fits a matrix-
valued substrate-internal parameter (vs W60's V3 retrieval
matrix, which was a d² outer-product feature vector). The
**trained replay threshold head** in replay controller V2 is the
first time the REUSE/RECOMPUTE/FALLBACK/ABSTAIN policy is derived
from a closed-form ridge fit on labelled data rather than from
hard-coded thresholds. The **content-addressable cache key axis**
in tiny_substrate_v6 is the first substrate-internal axis whose
purpose is retrieval rather than retention. The **attention-
pattern target fit** in KV bridge V6 is the first time the W6x
programme fits to an attention-map distance rather than only to a
logit-shift direction.

The honest scope is unchanged: W61 makes the **in-repo V6
substrate** much more load-bearing — five new internal axes,
seven new closed-form-ridge-trained controllers/heads, a six-way
bidirectional hybrid loop — but the **third-party hosted-model
substrate remains blocked** at the HTTP surface
(``W61-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``).

## TL;DR — W60 Trainable Cache-Control Substrate-Coupled Latent Operating System (post-W59 research milestone)

The programme now has **fifty-seven** coupled research axes. W60
mints axis 57: the **fifth substrate-attack milestone** that
turns W59's *single-α ridge correction* and *single-policy cache
controller* into **multi-direction multi-target ridge fits**, a
**per-(layer, head) trained inject-scale tensor**, the
programme's first first-class **state-reuse-vs-recompute-vs-
fallback-vs-abstain** ReplayController, and a **five-way
substrate ↔ V6 ↔ cache controller V3 ↔ replay controller ↔
retrieval head** hybrid loop. Eighteen orthogonal substrate-
coupling and capsule-native advances layered on top of W59 —
(M1) a **Tiny Transformer Runtime V5**
(``coordpy.tiny_substrate_v5``): richer than W59 V4 — **7 layers
/ 8 query heads / 4 KV heads (GQA)**, RMSNorm + SwiGLU,
per-(layer, head, position) cumulative EMA *attention-receive*
matrix that survives evictions, per-(layer, head) linearised
*logit Jacobian table*, per-(layer, position) **corruption flag
channel**, and **multi-segment partial-prefix reuse** (segments
∈ {reuse, recompute, drop}) with per-segment flop split;
(M2) a **KV Bridge V5** (``coordpy.kv_bridge_v5``): the
programme's first **multi-direction closed-form ridge fit** of
the substrate-side correction — solves a real ``(n_directions ×
n_directions)`` ridge linear system over substrate-side Jacobian
estimates from finite differences. Adds a separate **logit-
direction fit** (vs V4's L2-magnitude fit), two correction
layers (``layer_a`` / ``layer_b``), an all-bank fingerprint
across {bank_a, bank_b, bank_c, bank_d}, and a
**``extract_carrier_from_v5_kv_cache``** reverse-extract that
returns the carrier estimate via least-squares against the V5
projection matrix (residual L2 < 1e-3 on uncorrupted banks);
(M3) a **Hidden-State Bridge V4**
(``coordpy.hidden_state_bridge_v4``): closed-form ridge fit of
the *per-(layer, head) inject-scale tensor* (``δ ∈ R^{L*H}``,
typically ``L=2, H=8 → 16-dim`` decision variable). Adds a
*recovery* path that fits a counter-perturbation against
adversarial per-(layer, head) attacks, and a **KV-vs-Hidden
head-to-head harness** (``compare_hidden_vs_kv_injection_v4``)
that runs the same carrier through both arms on the same target
logit direction and reports the falsifiable
``hidden_beats_kv ∨ kv_beats_hidden ∨ tie`` claim;
(M4) a **Prefix-State Bridge V4**
(``coordpy.prefix_state_bridge_v4``): multi-segment partial
reuse on the V5 substrate (vs V3's single-split), chain forward
over a list of follow-up steps with per-step drift L2 and
cumulative drift envelope. R-125 H129 reports ~46% flop saving
at the standard {reuse, recompute, drop} split with bounded
drift;
(M5) an **Attention-Steering Bridge V4**
(``coordpy.attention_steering_bridge_v4``): per-(layer, head,
query) **3-D budget tensor** (vs V3's 2-D), measurable
attention-map L1 mass shift alongside KL, and a **negative-
budget falsifier** (budget=0 ⇒ post-KL < 1e-6 AND no shift);
(M6) a **Cache Controller V3** (``coordpy.cache_controller_v3``):
**four** new policies on top of V2's three —
``learned_attention_receive`` (closed-form ridge over the
per-(layer, head) cumulative attention-receive feature),
``learned_corruption_aware`` (V2 + hard ``-inf`` floor on
flagged slots), ``trained_eviction`` (closed-form ridge over a
``[hidden, importance, attention_receive_l1, retrieval]``
feature against the V1 leave-one-out drop oracle; on R-125
reduces residual from 28.4 → 0.39, a > 70x improvement), and
``composite_v3`` (4-feature ridge mixture);
(M7) a **Replay Controller** (``coordpy.replay_controller``):
the programme's first first-class **state-reuse-vs-recompute-
vs-fallback-vs-abstain** policy. Decision rule (in order):
REUSE if CRC passed AND saving above floor AND drift below
ceiling; RECOMPUTE if recompute under flop ceiling AND reuse
drift over ceiling OR CRC failed; FALLBACK if transcript
available; ABSTAIN otherwise. Emits an audit log + flop-vs-drift
trade-off curve;
(M8) a **Persistent Latent V12**
(``coordpy.persistent_latent_v12``): **10-layer**, **octuple
skip** (V11's septuple plus a *replay-controller decision EMA*),
``max_chain_walk_depth = 1024``, and **distractor-resistant**
replay projection (random orthonormal distractor basis fit by
Gram-Schmidt at init; replay skip projected orthogonal to the
basis before injection);
(M9) a **16-backend Multi-Hop Translator V10**
(``coordpy.multi_hop_translator_v10``): **240 directed edges**,
chain-length-15, **substrate × hidden × attention × retrieval ×
replay** five-axis trust composite, and a
``estimate_compromise_threshold`` that returns the minimum
number of axes an adversary must drive to zero on the dominant
path to flip the arbitration outcome (1 ≤ threshold ≤ 5);
(M10) a **Mergeable Latent State Capsule V8**
(``coordpy.mergeable_latent_capsule_v8``): adds
``replay_witness_chain``, ``substrate_witness_chain``, and a
``provenance_trust_table`` (per-backend trust scalar) all
union-inherited from parents; two new algebra signatures
{``replay_choice``, ``substrate_state_inject``};
(M11) a **Consensus Fallback Controller V6**
(``coordpy.consensus_fallback_controller_v6``): **10-stage
chain** {K-of-N → trust-weighted → substrate → logit_lens →
cache_reuse → retrieval_replay → **replay_controller** →
best_parent → transcript → abstain};
(M12) a **Corruption-Robust Carrier V8**
(``coordpy.corruption_robust_carrier_v8``): **256-bucket
fingerprint** (vs V7's 128) with wrap-around XOR so single-byte
flips are *always* detectable regardless of blob length; adds
``recover_v8_kv_cache`` operator that writes per-(layer,
position) corruption flags into the V5 cache; **adversarial
11-bit burst** family; **post-replay top-K agreement** floor
(must be ≥ pre-replay);
(M13) a **Deep Substrate Hybrid V5**
(``coordpy.deep_substrate_hybrid_v5``): the programme's first
**five-way bidirectional bridge** — V6 ↔ substrate V5 ↔ cache
controller V3 (composite_v3) ↔ replay controller ↔ retrieval
head, with ``five_way=True`` flag set when REUSE or RECOMPUTE
fires;
(M14) an **11-head Long-Horizon Retention V12**
(``coordpy.long_horizon_retention_v12``) at ``max_k=96`` with
a new **replay-conditioned** head and a **two-layer** retention
scorer (random projection + frozen ReLU + closed-form ridge
over the post-ReLU features);
(M15) an **11-level ECC Codebook V12**
(``coordpy.ecc_codebook_v12``): K1..K11 = **2 097 152 codes**
(``= 2^21``); **23.333 bits/visible-token** at full emit
(≥ 23.0 target, exceeds V11's 22.333);
(M16) an **Uncertainty Layer V8**
(``coordpy.uncertainty_layer_v8``): **7th weighting axis**
``replay_fidelity`` on top of V7's (confidence, trust,
substrate, hidden, cache, retrieval);
(M17) a **Disagreement Algebra V6**
(``coordpy.disagreement_algebra_v6``): adds **replay-controller
equivalence identity** with argmax + L2-budget check;
(M18) a **Ten-arm Transcript-vs-Shared-vs-Substrate-vs-Cache-vs-
Retrieval-vs-Replay Arbiter V9**
(``coordpy.transcript_vs_shared_arbiter_v9``): adds
``replay_controller_choice`` arm.

The W60 ``W60Team`` orchestrator composes all eighteen modules
plus the Substrate Adapter V5, emits per-turn 19 module witness
CIDs, and seals them into a ``W60HandoffEnvelope`` whose
``w59_outer_cid`` carries forward the W59 envelope. The W60
envelope verifier enumerates **52 disjoint failure modes** (≥ 50
target met).

W60 is the **fifth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
documents that Ollama / OpenAI-compatible / hosted backends
remain text-only on the HTTP surface (carries forward unchanged
from W57/W58/W59). ``W60-L-V5-NO-AUTOGRAD-CAP`` is the new
honest cap: W60 fits **only** (a) the KV bridge V5 multi-
direction α along ``n_directions`` random directions, (b) the KV
bridge V5 logit-direction α, (c) the HSB V4 per-(layer, head) δ
tensor, (d) the cache controller V3
``learned_attention_receive`` linear head, (e) the cache
controller V3 ``trained_eviction`` linear head, (f) the cache
controller V3 ``composite_v3`` mixture weights, and (g) the LHR
V12 two-layer scorer (random first layer, ridge second layer)
— every one a single closed-form linear ridge solve, no end-to-
end backprop, no autograd, no GPU.
``W60-L-V12-OUTER-NOT-TRAINED-CAP`` carries forward V11's outer-
untrained cap. ``W60-L-ECC-V12-RATE-FLOOR-CAP`` documents the
new structural ceiling (log2(2 097 152) = 21 bits/segment); the
2048-bit/token falsifier reproduces honestly as the new H136c
bar. ``W60-L-LHR-V12-SCORER-FIT-CAP`` documents that the LHR
V12 two-layer scorer's first layer is *random + frozen* (ridge
only on layer 2). ``W60-L-CORRUPTION-FLAG-CHANNEL-CAP``
documents that the V5 corruption flag channel is a *channel*,
not a detector — flags are written by external CRC V8 and read
by cache controller V3.

W60 ships at ``coordpy.tiny_substrate_v5``,
``coordpy.kv_bridge_v5``, ``coordpy.hidden_state_bridge_v4``,
``coordpy.prefix_state_bridge_v4``,
``coordpy.attention_steering_bridge_v4``,
``coordpy.cache_controller_v3``, ``coordpy.replay_controller``,
``coordpy.persistent_latent_v12``,
``coordpy.multi_hop_translator_v10``,
``coordpy.mergeable_latent_capsule_v8``,
``coordpy.consensus_fallback_controller_v6``,
``coordpy.corruption_robust_carrier_v8``,
``coordpy.long_horizon_retention_v12``,
``coordpy.ecc_codebook_v12``,
``coordpy.transcript_vs_shared_arbiter_v9``,
``coordpy.uncertainty_layer_v8``,
``coordpy.disagreement_algebra_v6``,
``coordpy.deep_substrate_hybrid_v5``,
``coordpy.substrate_adapter_v5``, ``coordpy.w60_team``,
``coordpy.r125_benchmark``, ``coordpy.r126_benchmark``,
``coordpy.r127_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK
contract is byte-for-byte unchanged. **No PyPI release**.

R-125 (20 cell families, real-substrate / latent-bridge V5 /
trained-controller V3 / replay controller / hidden-vs-KV) +
R-126 (13 cell families, long-horizon retention / reconstruction
/ aggressive cramming) + R-127 (12 cell families, corruption /
disagreement / consensus / fallback) at 3 seeds verify
H125..H143b. **45 of 45 H-bars pass 3/3 seeds (135/135 cells,
strong success per the W60 success criterion)**. Cumulative
trust boundary across W22..W60 = **715 enumerated failure
modes** (663 from W22..W59 + 52 new W60 envelope verifier
modes).

W60 envelope chain end-to-end: ``W59 envelope CID ==
W60.w59_outer_cid`` (verified live). Trivial passthrough
preserved (W60 trivial envelope still rejects 19 missing-witness
modes).

W60 is the strongest honest substrate-coupling milestone the
programme has shipped. The hidden-vs-KV head-to-head harness
(R-125 H132) is the first time we have a falsifiable claim that
*compares the two main substrate injection paths head-to-head*
on a fixed target logit direction; on the default V5 config
``hidden_beats_kv = True`` (hidden-state V4 per-(layer, head)
ridge fit gets a smaller residual to the target shift than the
KV V4 single-α correction). This is a constructive observation,
NOT a general claim about hidden vs KV across all models.

## TL;DR — W59 Trainable Substrate-Conditioned Latent Operating System (post-W58 research milestone)

The programme now has **fifty-six** coupled research axes. W59
mints axis 56: the **fourth substrate-attack milestone** that
turns W58's *fitted inject scales* + *linear ridge controller*
into **closed-form ridge fits on real projection / retrieval /
retention parameters** and a **four-way substrate ↔ V6 ↔ cache
controller ↔ retrieval-head** hybrid loop. Eighteen orthogonal
substrate-coupling and capsule-native advances layered on top of
W58 —
(M1) a **Tiny Transformer Runtime V4**
(``coordpy.tiny_substrate_v4``): richer than W58 V3 —
**6 layers / 8 query heads / 4 KV heads (GQA)**, optional
**RMSNorm** + **SwiGLU**, real **per-(layer, position)
cumulative EMA importance** that survives evictions, real
**partial-prefix split / replay**
(``extract_partial_prefix_v4`` + ``forward_with_partial_prefix_reuse_v4``)
with dual-stage flop accounting, real **per-(layer, head)
hidden-state tap** of shape ``(n_heads, n_tokens, d_head)``, a
real **128-bucket fingerprint**, and a **logit-Jacobian probe**
at the final position;
(M2) a **KV Bridge V4** (``coordpy.kv_bridge_v4``): **four** role
banks (``bank_a``/``bank_b``/``bank_c``/``bank_d``) and the
programme's first **closed-form ridge fit of a substrate-facing
projection correction** — given an N-point training set of
(carrier, target-L2) pairs, ``fit_kv_bridge_v4_correction``
estimates the substrate-side scalar Jacobian by central finite
differences and solves a single 1-D ridge linear system for the
correction α along a fixed random direction, then writes the
correction tensors into the projection. Pre/post fit residual
ratios are recorded;
(M3) a **Hidden-State Bridge V3**
(``coordpy.hidden_state_bridge_v3``): adds **target-logit-shift
fitting** — given a desired ``Δlogit`` direction, the bridge
estimates the substrate-side Jacobian and solves a single
closed-form ridge problem for the inject-scale α that brings the
projected shift closest to the target. Also exposes a per-(layer,
head) scale tensor for optional per-head injection;
(M4) a **Prefix-State Bridge V3**
(``coordpy.prefix_state_bridge_v3``): adds **partial-prefix
reuse** that splits a saved prefix into a reusable head and a
recompute tail, byte-identical to a full recompute on the
matched span (max-abs diff ≤ 4.4e-16 on the R-122 probe), a
**K-seed drift spectrum** (mean / max / min / var), an
**empirical Lipschitz certificate ratio**, and a **128-bucket
fingerprint**;
(M5) an **Attention-Steering Bridge V3**
(``coordpy.attention_steering_bridge_v3``): per-(layer, head)
**KL-budget clip fit** by iterative shrink-per-head (vs W58 V2's
single-global clip) — every head individually honoured at
``budget_per_head=0.6`` on the H110 probe. Per-head dominance
ablation carried forward;
(M6) a **Cache Controller V2** (``coordpy.cache_controller_v2``):
keeps W58's three policies and adds **two new closed-form ridge
heads** — ``learned_hidden`` (cross-layer hidden-state-feature
scoring) and ``learned_retrieval`` (a **bilinear ``M`` matrix**
fit by closed-form ridge over a ``d² = 4096``-dim outer-product
feature so that ``q^T M h_t`` matches the substrate's leave-one-
out drop oracle; pre/post-fit residual ratio routinely
> 4 orders of magnitude on the R-122 probe);
(M7) a **Persistent Latent V11**
(``coordpy.persistent_latent_v11``): **9-layer**, **septuple
skip** (anchor + fast EMA + slow EMA + substrate-EMA + hidden-
EMA + attention-EMA + **retrieval-EMA**),
``max_chain_walk_depth = 768``, retrieval-fidelity damping;
(M8) a **14-backend Multi-Hop Translator V9**
(``coordpy.multi_hop_translator_v9``): **182 directed edges**,
chain-length-13, **substrate × hidden × attention × retrieval**
four-axis trust composite;
(M9) a **Mergeable Latent State Capsule V7**
(``coordpy.mergeable_latent_capsule_v7``): adds
``retrieval_witness_chain`` (union-inherited from parents),
``controller_witness_cid``, and two new algebra signatures
{``retrieval_replay``, ``partial_prefix_reuse``};
(M10) a **Consensus Fallback Controller V5**
(``coordpy.consensus_fallback_controller_v5``): **9-stage
chain** {K-of-N → trust-weighted → substrate → logit_lens →
cache_reuse → **retrieval_replay** → best_parent → transcript →
abstain};
(M11) a **Corruption-Robust Carrier V7**
(``coordpy.corruption_robust_carrier_v7``): **128-bucket
fingerprint** (vs V6's 64), **cache-retrieval top-K agreement
under non-target corruption** (>= 0.7 jaccard floor on R-124),
and a **9-bit adversarial burst** family;
(M12) a **10-head Long-Horizon Retention V11**
(``coordpy.long_horizon_retention_v11``) at ``max_k=80`` with a
new **retrieval-conditioned** head and the programme's first
**trained retention scorer** (closed-form ridge over a small
synthetic supervised set);
(M13) a **10-level ECC Codebook V11**
(``coordpy.ecc_codebook_v11``): K1..K10 = **1 048 576 codes**
(``= 2^20``); **22.333 bits/visible-token** at full emit
(≥ 22.0 target, exceeds V10's 21.333);
(M14) a **9-arm Transcript-vs-Shared-vs-Substrate-vs-Cache-vs-
Retrieval Arbiter V8**
(``coordpy.transcript_vs_shared_arbiter_v8``): adds
``retrieval_replay`` arm;
(M15) an **Uncertainty Layer V7**
(``coordpy.uncertainty_layer_v7``): **6th weighting axis**
``retrieval_fidelity`` on top of V6's (confidence, trust,
substrate, hidden, cache);
(M16) a **Disagreement Algebra V5**
(``coordpy.disagreement_algebra_v5``): adds a **retrieval-
equivalence identity** with argmax + L2-budget check;
(M17) a **Deep Substrate Hybrid V4**
(``coordpy.deep_substrate_hybrid_v4``): the **first four-way
bidirectional bridge** in the programme — V6 ↔ substrate V4 ↔
cache controller V2 (retrieval-policy) ↔ retrieval head, with
``four_way=True`` flag in real runs;
(M18) a **Substrate Adapter V4**
(``coordpy.substrate_adapter_v4``): five new capability axes
{``partial_prefix_reuse``, ``cache_retrieval``,
``closed_form_ridge``, ``per_head_kl_fit``,
``hidden_target_fit``} and a new top tier
``substrate_v4_full`` reached only by the V4 in-repo runtime.

The W59 ``W59Team`` orchestrator composes all eighteen modules,
emits per-turn module witness CIDs, and seals them into a
``W59HandoffEnvelope`` whose ``w58_outer_cid`` carries forward
the W58 envelope. The W59 envelope verifier enumerates **49
disjoint failure modes** (≥ 48 target met).

W59 is the **fourth executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
documents that Ollama / OpenAI-compatible / hosted backends
remain text-only on the HTTP surface (carries forward unchanged
from W57/W58). ``W59-L-V4-NO-AUTOGRAD-CAP`` is the new honest
cap: W59 fits **only** (a) the KV bridge V4 1-D correction α
along a fixed random direction, (b) the HSB V3 1-D inject-scale
α against a target logit direction, (c) the cache controller V2
``learned_hidden`` linear head and (d) the cache controller V2
``learned_retrieval`` bilinear ``M`` matrix, and (e) the LHR V11
retention scorer — every one a single closed-form linear ridge
solve, no end-to-end backprop, no autograd, no GPU.
``W59-L-V11-OUTER-NOT-TRAINED-CAP`` carries forward V10's outer-
untrained cap. ``W59-L-ECC-V11-RATE-FLOOR-CAP`` documents the
new structural ceiling (log2(1 048 576) = 20 bits/segment);
the 1024-bit/token falsifier reproduces honestly as the new
H117c bar. ``W59-L-LHR-V11-SCORER-FIT-CAP`` documents that the
LHR V11 retention scorer is a *single linear ridge head*, not a
deep network.

W59 ships at ``coordpy.tiny_substrate_v4``,
``coordpy.kv_bridge_v4``, ``coordpy.hidden_state_bridge_v3``,
``coordpy.prefix_state_bridge_v3``,
``coordpy.attention_steering_bridge_v3``,
``coordpy.cache_controller_v2``,
``coordpy.persistent_latent_v11``,
``coordpy.multi_hop_translator_v9``,
``coordpy.mergeable_latent_capsule_v7``,
``coordpy.consensus_fallback_controller_v5``,
``coordpy.corruption_robust_carrier_v7``,
``coordpy.long_horizon_retention_v11``,
``coordpy.ecc_codebook_v11``,
``coordpy.transcript_vs_shared_arbiter_v8``,
``coordpy.uncertainty_layer_v7``,
``coordpy.disagreement_algebra_v5``,
``coordpy.deep_substrate_hybrid_v4``,
``coordpy.substrate_adapter_v4``, ``coordpy.w59_team``,
``coordpy.r122_benchmark``, ``coordpy.r123_benchmark``,
``coordpy.r124_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. No PyPI release.

R-122 (15 cell families, real-substrate / trained-controller /
cache-retrieval / partial-prefix) + R-123 (12 cell families,
long-horizon retention / reconstruction / aggressive cramming) +
R-124 (11 cell families, corruption / disagreement / consensus
/ fallback) at 3 seeds verify H107..H124. **38 of 38 H-bars
pass 3/3 seeds (strong success per the W59 success criterion)**.
Cumulative trust boundary across W22..W59 = **663 enumerated
failure modes** (614 from W22..W58 + 49 new W59 envelope verifier
modes).

W59 headline results (3 seeds, mean):

* tiny substrate V4 forward determinism: **3/3 byte-identical**
* substrate V4 cumulative-EMA importance propagates: **3/3**
* substrate V4 head-hidden tap shape correct: **3/3**
* KV bridge V4 four banks distinct (all pairs > 1e-3): **3/3**
* KV bridge V4 closed-form ridge correction converges (post ≤ pre): **3/3**
* HSB V3 target-logit fit produces non-negative residual: **3/3**
* attention V3 per-(layer, head) KL budget enforced ≤ 0.6 + tol: **3/3**
* cache controller V2 learned_retrieval runs + finite drift: **3/3**
* cache controller V2 learned_retrieval ridge fit residual < pre: **3/3**
* prefix V3 partial reuse byte-identical to recompute: **3/3** (≤ 4.4e-16)
* prefix V3 partial flop_saved > 0 (positive savings ratio): **3/3**
* prefix V3 K-seed drift spectrum non-degenerate: **3/3**
* deep hybrid V4 four_way flag True with substrate_back_l2 > 0: **3/3**
* deep hybrid V4 retrieval_used True under learned_retrieval policy: **3/3**
* substrate adapter V4 ``substrate_v4_full`` tier on V4 only: **3/3**
* persistent V11 retrieval-EMA propagates over 32 turns: **3/3**
* persistent V11 chain walk depth ≥ 32: **3/3**
* LHR V11 five-way runs without crashing: **3/3**
* LHR V11 retention scorer fit residual ≤ pre: **3/3**
* LHR V11 max_k == 80: **3/3**
* ECC V11 bits/visible-token at full emit: **22.333** (≥ 22.0)
* ECC V11 total codes: **1 048 576**
* ECC V11 1024-bit/token falsifier reproduces: **3/3**
* multi-hop V9 chain_length=13, n_edges=182, 14 backends: **3/3**
* multi-hop V9 retrieval axis used in composite: **3/3**
* TVS V8 9-arm pick rates sum to 1.0: **3/3**
* TVS V8 retrieval_replay arm dominates when rf strict highest: **3/3** (≥ 0.9)
* CRC V7 128-bucket fingerprint detect rate: **≥ 0.99**: **3/3**
* CRC V7 cache-retrieval top-K agreement under non-target corruption: **≥ 0.7**: **3/3**
* CRC V7 adversarial 9-bit burst detect rate: **≥ 0.95**: **3/3**
* consensus V5 9-stage chain: **9** stages enumerated
* consensus V5 retrieval_replay stage fires when only retrieval oracle resolves: **3/3**
* consensus V5 abstains when all paths below floor: **3/3**
* uncertainty V7 pessimistic ≤ weighted ≤ optimistic: **3/3**
* uncertainty V7 retrieval_aware True when retrieval fidelities distinct: **3/3**
* disagreement algebra V5 retrieval-equivalence identity holds: **3/3**
* disagreement algebra V5 falsifier reproduces (oracle False → identity False): **3/3**
* MLSC V7 retrieval witness chain inherits union from parents: **3/3**
* W59 envelope verifier failure modes: **49** disjoint
* W59 envelope verifier OK on clean run: **all seeds**
* W59 envelope outer CID stable across runs: **3/3 stable**
* W59 envelope substrate_v4_used flag True in real run: **3/3**
* W59 envelope four_way_used flag True in real run: **3/3**

W59 directly attacks the post-W58 question of **how to make
real substrate-facing parameters (projection corrections,
retrieval matrices, retention scorers) load-bearing at the
closed-form-ridge level**. The cache controller V2's
``learned_retrieval`` bilinear matrix is the first place in the
programme where a real ``d × d`` substrate-facing matrix is fit
end-to-end by closed-form ridge against the substrate's own
drop-oracle on a single forward pass. H111b passes with
pre/post residual reduction > 4 orders of magnitude. Honest
bounds: ``W59-L-V4-NO-AUTOGRAD-CAP`` (no end-to-end SGD /
autograd); ``W59-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
(carries forward); ``W59-L-V11-OUTER-NOT-TRAINED-CAP`` (V11 outer
GRU initialised but not trained); ``W59-L-ECC-V11-RATE-FLOOR-CAP``;
``W59-L-LHR-V11-SCORER-FIT-CAP`` (the LHR V11 scorer is a single
linear ridge head, not deep);
``W59-L-MULTI-HOP-V9-SYNTHETIC-BACKENDS-CAP`` (the 14 backends
are named, not executed);
``W59-L-V11-PERMUTATION-INVARIANCE-CAP`` (carries forward).

---

## Prior milestone: W58 Deep Cache-Reuse Substrate-Coupled Latent Operating System (post-W57 research milestone)

The programme now has **fifty-five** coupled research axes. W58
mints axis 55: the **third substrate-attack milestone** that
extends W57's substrate breach with **cache-reuse-vs-recompute
flop accounting, fitted bridge scales, and a learned cache
controller**. Eighteen orthogonal substrate-coupling and capsule-
native advances layered on top of W57 —
(M1) a **Tiny Transformer Runtime V3**
(``coordpy.tiny_substrate_v3``): richer than W57 V2 —
**5 layers / 8 query heads / 4 KV heads (GQA)** with optional
**RMSNorm**, optional **SwiGLU** FF, real **per-token KV
importance tracking** (every layer records ``attention_received``
per key position), a real **fp64 flop counter**, a **partial-
forward** API that runs only a suffix of layers, and a **64-bucket
Reed-Solomon fingerprint** baked into the KV cache CID;
(M2) a **KV Bridge V3** (``coordpy.kv_bridge_v3``):
**fitted per-(layer, head) inject scales** via coordinate descent
against an explicit target L2 perturbation (first time the
programme fits a substrate-facing parameter from a real target),
**role-conditioned KV banks** (``bank_a``/``bank_b`` distinct
projection matrices), and a 64-bucket readback fingerprint;
(M3) a **Hidden-State Bridge V2**
(``coordpy.hidden_state_bridge_v2``): **multi-layer simultaneous
injection** with a fitted projection that targets a specific
logit shift;
(M4) a **Prefix-State Bridge V2**
(``coordpy.prefix_state_bridge_v2``): adds a **real flop-saved
counter** (e.g. 66.7% savings on a 20-token prompt / 9-token
follow-up), a **redundant-copy CID** for corruption detection,
and a **cross-seed drift L2** measurement;
(M5) an **Attention-Steering Bridge V2**
(``coordpy.attention_steering_bridge_v2``): **KL-budget
enforcement** by coordinate-descent on a global bias clip
(``kl_budget_enforced=True`` when max-KL ≤ budget × 1.001), plus
**per-head ablation** measuring each head's contribution to the
steered logit;
(M6) a **Cache Controller** (``coordpy.cache_controller``):
three policies {``uniform`` LRU, ``importance`` intrinsic KV
importance, ``learned`` closed-form ridge on a drop-oracle},
returning a content-addressed witness with flop savings and
last-position logit drift;
(M7) a **Persistent Latent V10**
(``coordpy.persistent_latent_v10``): **8-layer**, **sextuple
skip** (anchor + fast EMA + slow EMA + substrate-EMA + hidden-EMA
+ **attention-pattern-EMA**), ``max_chain_walk_depth = 512``,
attention-fidelity damping;
(M8) a **Deep Substrate Hybrid V3**
(``coordpy.deep_substrate_hybrid_v3``): the **first three-way
bidirectional bridge** in the programme — V6 ↔ substrate V3 ↔
cache controller, with substrate-back-L2 and cache-eviction-
perturbation-L2 both load-bearing; ``three_way=True`` in real
runs;
(M9) a **12-backend Multi-Hop Translator V8**
(``coordpy.multi_hop_translator_v8``): **132 directed edges**,
chain-length-11, **substrate × hidden × attention** three-axis
trust composite;
(M10) a **Mergeable Latent State Capsule V6**
(``coordpy.mergeable_latent_capsule_v6``): adds
``attention_witness_chain`` (union-inherited from parents),
``cache_reuse_witness_cid``, and two new algebra signatures
{``cache_reuse_replay``, ``attention_steer``};
(M11) a **Consensus Fallback Controller V4**
(``coordpy.consensus_fallback_controller_v4``): **8-stage
chain** {K-of-N → trust-weighted → substrate → logit_lens →
**cache_reuse_replay** → best_parent → transcript → abstain};
(M12) a **Corruption-Robust Carrier V6**
(``coordpy.corruption_robust_carrier_v6``): **64-bucket Reed-
Solomon fingerprint** (vs V5's 32) and **prefix-state corruption
detection** with 1.0 detect rate on the R-121 probe;
(M13) a **9-head Long-Horizon Retention V10**
(``coordpy.long_horizon_retention_v10``) at ``max_k=72`` with a
new **attention-conditioned** head;
(M14) a **9-level ECC Codebook V10**
(``coordpy.ecc_codebook_v10``): K1..K9 = **524 288 codes**;
**21.333 bits/visible-token** at full emit (≥ 21.0 target,
exceeds V9's 20.333);
(M15) an **8-arm Transcript-vs-Shared-vs-Substrate-vs-Cache
Arbiter V7** (``coordpy.transcript_vs_shared_arbiter_v7``):
adds ``cache_reuse_replay`` arm;
(M16) an **Uncertainty Layer V6**
(``coordpy.uncertainty_layer_v6``): **5th weighting axis**
``cache_reuse_fidelity`` on top of V5's (confidence, trust,
substrate, hidden);
(M17) a **Disagreement Algebra V4**
(``coordpy.disagreement_algebra_v4``): adds a **cache-reuse
equivalence identity** (reuse_logits matches recompute_logits
within float64);
(M18) a **Substrate Adapter V3**
(``coordpy.substrate_adapter_v3``): five new capability axes
{``kv_importance_track``, ``flop_counter``, ``partial_forward``,
``fitted_inject_scale``, ``cache_controller``} and a new top
tier ``substrate_v3_full`` reached only by the V3 in-repo
runtime.

The W58 ``W58Team`` orchestrator composes all eighteen modules,
emits per-turn module witness CIDs, and seals them into a
``W58HandoffEnvelope`` whose ``w57_outer_cid`` carries forward
the W57 envelope. The W58 envelope verifier enumerates **46
disjoint failure modes** (≥ 45 target met).

W58 is the **third executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
documents that Ollama / OpenAI-compatible / hosted backends
remain text-only on the HTTP surface (carries forward unchanged
from W57). ``W58-L-V3-NO-BACKPROP-CAP`` is the new honest cap:
W58 fits **only** the KV bridge V3 inject scales (and similar
inject scales for HSB V2) and a single linear retention scoring
head — no end-to-end backprop, no autograd, no GPU.
``W58-L-V10-OUTER-NOT-TRAINED-CAP`` carries forward V9's V10
outer-untrained cap. ``W58-L-ECC-V10-RATE-FLOOR-CAP`` documents
the new structural ceiling (log2(524 288) ≈ 19 bits/segment);
the 1024-bit/token falsifier reproduces honestly as H95.
``W58-C-DEEP-TRANSFORMER-COUPLING`` is a sharper restatement of
the open question on frontier-scale models;
``W58-C-FRONTIER-SCALE-SUBSTRATE-LIFT`` carries forward W57's
conjecture that the W58 bridges (KV V3 fitted + HSB V2 + prefix-
state V2 + attention V2 + cache controller) would, if exposed by
a frontier model's runtime, scale-monotonically improve
usefulness.

W58 ships at ``coordpy.tiny_substrate_v3``,
``coordpy.kv_bridge_v3``, ``coordpy.hidden_state_bridge_v2``,
``coordpy.prefix_state_bridge_v2``,
``coordpy.attention_steering_bridge_v2``,
``coordpy.cache_controller``,
``coordpy.persistent_latent_v10``,
``coordpy.multi_hop_translator_v8``,
``coordpy.mergeable_latent_capsule_v6``,
``coordpy.consensus_fallback_controller_v4``,
``coordpy.corruption_robust_carrier_v6``,
``coordpy.long_horizon_retention_v10``,
``coordpy.ecc_codebook_v10``,
``coordpy.transcript_vs_shared_arbiter_v7``,
``coordpy.uncertainty_layer_v6``,
``coordpy.disagreement_algebra_v4``,
``coordpy.deep_substrate_hybrid_v3``,
``coordpy.substrate_adapter_v3``, ``coordpy.w58_team``,
``coordpy.r119_benchmark``, ``coordpy.r120_benchmark``,
``coordpy.r121_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. No PyPI release.

R-119 (16 cell families, real-substrate / latent-bridge / cache-
reuse) + R-120 (12 cell families, long-horizon / reconstruction
/ cramming) + R-121 (12 cell families, corruption / disagreement
/ consensus / fallback) at 3 seeds verify H86..H106. **40 of 40
H-bars pass 3/3 seeds (strong success per the W58 success
criterion)**. Cumulative trust boundary across W22..W58 = **614
enumerated failure modes** (568 from W22..W57 + 46 new W58
envelope verifier modes).

W58 headline results (3 seeds, mean):

* tiny substrate V3 forward determinism: **3/3 byte-identical**
* substrate V3 KV cache reuse max abs diff: **≤ 1e-15**
* substrate V3 causal mask max upper-triangle weight: **0.0** (strict)
* substrate V3 GQA cache size < d_model: **3/3** (32 < 64)
* substrate V3 logit lens count = n_layers + 1: **3/3**
* substrate V3 flop counter > 0: **3/3** (1.9M fp64 ops per turn at default size)
* KV bridge V3 fitted inject scale residual: **|L2 − target| ≤ unfitted residual**: 3/3
* KV bridge V3 banks A and B distinct L2: **3/3** (Δ ≈ 1.1)
* hidden-state bridge V2 multi-layer L2: **6.3** mean (vs HSB V1 single-layer 1.43 mean)
* prefix-state V2 reuse-vs-recompute max-abs diff: **4.4e-16** (byte-identical)
* prefix-state V2 flop savings ratio: **0.667** mean (66.7% of recompute flops)
* prefix-state V2 cross-seed drift L2: **7.1** mean (substantial; non-transferable)
* attention-steering V2 KL budget enforced (budget=1.0): **3/3**
* cache controller importance ≤ 1.25× uniform L1 drift: **3/3**
* cache controller flop savings positive: **3/3** for uniform + importance
* persistent V10 512-turn chain walk depth: **512** mean
* persistent V10 8-layer cell: **3/3**
* persistent V10 attention-fidelity damping changes top state: **3/3**
* persistent V10 carrier round-trip CID deterministic: **3/3**
* multi-hop V8 chain-length-11 fidelity probe: **chain_length=11, n_edges=132**
* MLSC V6 attention-witness-chain inheritance: **3/3** (a1, a2, a3 in chain)
* MLSC V6 cache-reuse-witness-CID propagation: **3/3**
* consensus V4 8-stage chain: **8** stages enumerated
* consensus V4 cache-reuse-replay stage fires: **3/3**
* CRC V6 64-bucket KV fingerprint detect rate: **1.0** mean
* CRC V6 prefix-state corruption detect rate: **1.0** mean
* CRC V6 adversarial 7-bit burst detect rate: **1.0** mean
* LHR V10 four-way runs (9 heads, max_k=72): **3/3**
* LHR V10 attention head ≤ substrate head on attention-aligned: **3/3**
* ECC V10 bits/visible-token at full emit: **21.333** (≥ 21.0)
* ECC V10 total codes: **524 288**
* ECC V10 1024-bit/token falsifier reproduces: **3/3**
* TVS V7 8-arm pick rates sum to 1.0: **3/3**
* TVS V7 ``cache_reuse_replay`` arm dominates when cf strict highest: **3/3**
* uncertainty V6 cache-aware composite differs from V5: **3/3**
* uncertainty V6 pessimistic ≤ weighted ≤ optimistic: **3/3**
* disagreement algebra V4 cache-reuse equivalence identity: **3/3**
* substrate adapter V3 `substrate_v3_full` tier for V3 only: **3/3**
* substrate adapter V3 matrix `has_v3_full()`: **3/3**
* deep substrate hybrid V3 `three_way=True`: **3/3**
* deep substrate hybrid V3 substrate_back_l2 > 0: **3/3**
* deep substrate hybrid V3 cache_eviction_perturbation_l2 > 0: **3/3**
* W58 envelope verifier failure modes: **46** disjoint
* W58 envelope verifier OK on clean run: **all seeds**
* W58 envelope outer CID stable across runs: **3/3 stable**
* W58 envelope substrate_v3_used flag True in real run: **3/3**
* W58 envelope three_way_used flag True in real run: **3/3**

W58 directly attacks the post-W57 question of **how to make
cache-reuse-vs-recompute a real load-bearing axis of the
substrate program**. The prefix-state V2 path is the only place
in the programme where the *substrate's own flop count* is
material to a benchmark bar; H100b passes with 66.7% flop
savings at byte-identical logits. Honest bounds:
``W58-L-V3-NO-BACKPROP-CAP`` (no end-to-end training);
``W58-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` (carries
forward); ``W58-L-V10-OUTER-NOT-TRAINED-CAP`` (V10 outer GRU
initialised but not trained); ``W58-L-ECC-V10-RATE-FLOOR-CAP``;
``W58-L-CACHE-CONTROLLER-LINEAR-CAP`` (the learned policy is a
single linear scoring head, not deep);
``W58-L-MULTI-HOP-V8-SYNTHETIC-BACKENDS-CAP`` (the 12 backends
are named, not executed);
``W58-L-V10-PERMUTATION-INVARIANCE-CAP`` (carries forward).

---

## Prior milestone: W57 Deep Substrate-Coupled Latent Operating System (post-W56 research milestone)

The programme now has **fifty-four** coupled research axes. W57
mints axis 54: the **second substrate-attack milestone** that
turns W56's partial breach into a deeper substrate program.
Sixteen orthogonal substrate-coupling and capsule-native advances
layered on top of W56 Substrate-Coupled Latent Operating System —
(M1) a **Tiny Transformer Runtime V2**
(``coordpy.tiny_substrate_v2``): a richer, deeper executable
substrate with **4 layers / 8 heads / ``d_model=64`` / byte-vocab
259 / max_len=96**, real RoPE rotary positional embeddings,
real per-layer + per-head causal self-attention, real per-layer
KV cache **with LRU and importance-weighted eviction**, real
**prefix-state extraction** (a first-class content-addressed
``TinyV2PrefixState`` object), real **per-head pre-softmax
attention bias hook**, real **per-layer logit lens**, all in
pure NumPy on CPU;
(M2) a **KV Bridge V2** (``coordpy.kv_bridge_v2``): per-(layer,
head) projection from the capsule carrier, calibrated per-head
inject scales, replay-deterministic readback CID for the
injected slot bytes, and a content-addressed witness with both
max-abs/L2 logit perturbation and last-position cross-entropy
delta;
(M3) a **Hidden-State Bridge** (``coordpy.hidden_state_bridge``):
projects a capsule carrier to a per-layer ``(n_tokens, d_model)``
perturbation and additively injects it into the residual stream
*between* layer ``l-1``'s FF output and layer ``l``'s LN1; the
substrate's logits change measurably and content-addressed
(witnessed by ``HiddenStateBridgeWitness``);
(M4) a **Prefix-State Bridge**
(``coordpy.prefix_state_bridge``): saves a substrate KV cache as
a content-addressed prefix state, reuses it across turns
**byte-identical** to a full recompute (within float64
precision; max-abs diff ≤ 5e-16 on the H47 probe), and
**detects deliberate prefix corruption** by CID change;
(M5) an **Attention-Steering Bridge**
(``coordpy.attention_steering_bridge``): writes a per-(layer,
head, query, key) bias tensor pre-softmax into the substrate's
attention; mean-KL > 0 on every layer (≈ 5.7 nats per layer on
the H53 probe), causal mask still strict, attention pattern
demonstrably shifted;
(M6) a **7-layer V9 persistent latent state**
(``coordpy.persistent_latent_v9``) with a **quintuple persistent
skip-link** (turn-0 anchor + fast EMA + slow EMA + substrate-
conditioned EMA + **hidden-state-conditioned EMA**),
``max_chain_walk_depth = 384``, and **substrate-fidelity
weighting** that damps both substrate and hidden skips when
fidelity is low;
(M7) a **10-backend multi-hop translator V7**
(``coordpy.multi_hop_translator_v7``) over **90 directed edges**
with chain-length-9 transitivity and **substrate-hidden-trust
arbitration** (composite trust = substrate_trust ×
hidden_trust); the adversary now needs to corrupt both axes;
(M8) a **Mergeable Latent State Capsule V5 (MLSC V5)**
(``coordpy.mergeable_latent_capsule_v5``) with
``hidden_state_witness_chain`` (union-inherited from parents),
``attention_witness_cid``, ``per_head_trust`` weighting, and
two new algebra signatures (``substrate_project``,
``hidden_inject``);
(M9) a **Consensus Fallback Controller V3**
(``coordpy.consensus_fallback_controller_v3``) with a **7-stage
decision chain** {K-of-N → trust-weighted → substrate → **logit-
lens** → best-parent → transcript → abstain}, the new logit-lens
stage uses substrate V2's per-layer unembed at an intermediate
layer as a secondary tiebreaker;
(M10) a **Corruption-Robust Carrier V5**
(``coordpy.corruption_robust_carrier_v5``) with **3-D
interleaving** (4×4×4 = 64-bit blocks), **9-of-13 majority
repetition**, and **KV cache fingerprint-based corruption
detection** (Reed-Solomon-style 32-bucket XOR fingerprints);
the H75 KV-corruption detect rate is **1.0** in the R-118
sweep;
(M11) an **8-head Long-Horizon Reconstruction V9**
(``coordpy.long_horizon_retention_v9``) at ``max_k=64`` with
the new **hidden-state-conditioned** head; the
``evaluate_lhr_v9_three_way`` helper reports per-head MSE for
proxy / substrate / hidden;
(M12) an **8-level ECC Codebook V9**
(``coordpy.ecc_codebook_v9``): K1=32 × K2=16 × K3=8 × K4=4 ×
K5=2 × K6=2 × K7=2 × K8=2 = **262144 codes** (≈ 18 raw data
bits per segment-tuple), achieves **20.333 bits/visible-token
at full emit** (exceeds the ≥ 20.0 W57 target);
(M13) a **7-arm Transcript-vs-Shared-vs-Substrate Arbiter V6**
(``coordpy.transcript_vs_shared_arbiter_v6``) over {transcript,
shared, merge_consensus, trust_weighted_merge,
substrate_replay, **substrate_hidden_inject**, abstain} — the
first substrate-final-vs-substrate-intermediate head-to-head;
(M14) a **Hidden-State-Weighted Uncertainty Composite**
(``coordpy.uncertainty_layer_v5``) — fourth weighting axis on
top of (confidence, trust, substrate_fidelity), plus
**adversarial pessimistic / optimistic brackets** and a
**per-axis sensitivity** Jacobian magnitude;
(M15) a **Disagreement Algebra V3**
(``coordpy.disagreement_algebra_v3``) — extends V2 with a
**hidden-projection identity** and an **attention-steering
compatibility** check;
(M16) a **Deep Substrate Hybrid V2**
(``coordpy.deep_substrate_hybrid_v2``) — the first
**bidirectional substrate bridge** in the programme:
*substrate → V6* (substrate's intermediate hidden state
projects back into V6's residual stream) AND
*V6 → substrate* (V6 residual injected into substrate KV via
KV bridge V2, with optional attention-steering bias on top).
``W57HandoffEnvelope.bidirectional_used`` is **True** in
end-to-end runs.

Plus a **Substrate Adapter V2**
(``coordpy.substrate_adapter_v2``) with four new capability
axes (``attention_bias_write``, ``prefix_state_reuse``,
``cache_eviction``, ``logit_lens``) and a new top tier
``substrate_v2_full`` reached only by the V2 in-repo runtime;
hosted backends remain text-only.

W57 is the **second executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W57-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
documents that Ollama / OpenAI-compatible / hosted backends
remain text-only on the HTTP surface (carries forward unchanged
from W56). ``W57-C-DEEP-TRANSFORMER-COUPLING`` is a sharper
restatement of the open question on frontier-scale models;
``W57-C-FRONTIER-SCALE-SUBSTRATE-LIFT`` is a new conjecture
that the W57 bridges (KV V2 + hidden-state + prefix-state +
attention-steering) would, if exposed by a frontier model's
runtime, scale-monotonically improve usefulness — open until a
frontier runtime exposes the corresponding hooks.

W57 ships at ``coordpy.tiny_substrate_v2``,
``coordpy.kv_bridge_v2``, ``coordpy.hidden_state_bridge``,
``coordpy.prefix_state_bridge``,
``coordpy.attention_steering_bridge``,
``coordpy.persistent_latent_v9``,
``coordpy.multi_hop_translator_v7``,
``coordpy.mergeable_latent_capsule_v5``,
``coordpy.consensus_fallback_controller_v3``,
``coordpy.corruption_robust_carrier_v5``,
``coordpy.long_horizon_retention_v9``,
``coordpy.ecc_codebook_v9``,
``coordpy.transcript_vs_shared_arbiter_v6``,
``coordpy.uncertainty_layer_v5``,
``coordpy.disagreement_algebra_v3``,
``coordpy.deep_substrate_hybrid_v2``,
``coordpy.substrate_adapter_v2``, ``coordpy.w57_team``,
``coordpy.r116_benchmark``, ``coordpy.r117_benchmark``,
``coordpy.r118_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK
contract is byte-for-byte unchanged. No PyPI release.

R-116 (14 cell families, real-substrate / latent-bridge /
multi-hop) + R-117 (14 cell families, long-horizon retention /
reconstruction / cramming) + R-118 (15 cell families,
corruption / disagreement / consensus / fallback) at 3 seeds
verify H43..H85. **43 of 43 H-bars pass 3/3 seeds (strong
success per the W57 success criterion)**. The V9 outer GRU is
initialised but not trained — ``W57-L-V9-OUTER-NOT-TRAINED-CAP``
documents this. The ECC V9 256-bit/token target is above the
structural ceiling — ``W57-L-ECC-V9-RATE-FLOOR-CAP`` — and
reproduces honestly as the H65 falsifier. Cumulative trust
boundary across W22..W57 = **568 enumerated failure modes**
(524 from W22..W56 + 44 new W57 envelope verifier modes).

W57 headline results (3 seeds, mean):

* tiny substrate V2 forward determinism: **3/3 byte-identical**
* substrate V2 KV cache reuse max abs diff: **≤ 1e-15**
* substrate V2 causal mask max upper-triangle weight: **0.0** (strict)
* substrate V2 logit lens determinism: **3/3 byte-identical**
* substrate adapter V2 tier classification: **3/3 correct**
* KV bridge V2 last-position L2 perturbation: **5.31** mean (vs W56 KV bridge L2 mean 0.86)
* KV bridge V2 carrier-perturbation robustness: **≥ 16/16** non-zero across random carriers
* hidden-state bridge L2 perturbation: **1.43** mean
* hidden-state bridge cross-entropy delta: **0.10** mean
* prefix-state bridge reuse-vs-recompute max abs diff: **≤ 5e-16**
* prefix-state corruption detection: **3/3 detected**
* attention-steering mean KL per layer: **5.59** mean across 4 layers
* attention-steering causal mask preserved: **3/3**
* V9 96-turn chain walk: **96** mean
* V9 256-turn stretch chain walk: **256** mean
* V9 384-turn deep stretch chain walk: **384** mean
* V9 fidelity damping (substrate_fidelity=0): **3/3 distinct top-layer state**
* multi-hop V7 chain-length-9 fidelity probe: **chain_length=9** in all seeds
* MLSC V5 hidden-state-witness-chain inheritance: **3/3** (h1, h2, h3 in chain)
* MLSC V5 per-head-trust weighting: **3/3** (zero-trust head down-weighted)
* MLSC V5 algebra signature ``substrate_project``: **3/3** carried
* consensus V3 7-stage chain: **7** stages enumerated
* consensus V3 logit-lens stage fires when other stages fail: **3/3**
* CRC V5 BCH(31,16) triple-bit correct rate: **0.92** mean (≥ 0.80 floor)
* CRC V5 5-bit burst dispersion (max-run ≤ 2): **1.0** mean (≥ 0.90 floor)
* CRC V5 9-of-13 majority silent failure rate: **0.0** mean (≤ 0.05 floor)
* CRC V5 3-D interleave round trip OK: **3/3**
* CRC V5 KV cache corruption detect rate: **1.0** mean (≥ 0.95 floor)
* LHR V9 three-way comparison runs: **3/3**
* LHR V9 substrate head beats proxy on substrate-aligned: **3/3**
* LHR V9 hidden head beats substrate on hidden-aligned: **3/3**
* ECC V9 bits/visible-token at full emit: **20.333** (≥ 20.0 target)
* ECC V9 K8 1-bit boundary correct: **3/3**
* ECC V9 256-bit/token rate-floor falsifier: **3/3 reproduces** (above info bound)
* TVS V6 7-arm pick rates sum to 1.0: **3/3**
* TVS V6 ``substrate_hidden_inject`` arm preferred when hf >> sf: **3/3** (rate = 1.0)
* TVS V6 ``substrate_replay`` preferred when sf >> hf: **3/3** (rate = 1.0)
* uncertainty V5 hidden-aware composite differs when hf differs: **3/3**
* uncertainty V5 pessimistic ≤ weighted ≤ optimistic: **3/3**
* disagreement algebra V3 hidden-projection identity (identity projector): **3/3**
* W57 envelope verifier failure modes: **44** disjoint
* W57 envelope verifier OK on clean run: **all seeds**
* W57 envelope outer CID stable across runs: **3/3 stable**
* W57 envelope substrate_v2_used flag True in real run: **3/3**
* W57 envelope bidirectional_used flag True in real run: **3/3**
* deep substrate hybrid V2 substrate-back L2: **0.02** mean
* deep substrate hybrid V2 ablation perturbation L2: **2.50** mean
* deep substrate hybrid V2 bidirectional flag: **3/3 True**

W57 directly attacks the post-W56 question of **how to deepen
the substrate breach beyond the partial W56 breach**, with
explicit honest bounds on what remains: third-party hosted-
model substrate (``W57-C-DEEP-TRANSFORMER-COUPLING``), end-to-
end autograd training of the substrate
(``W57-L-NUMPY-CPU-V2-SUBSTRATE-CAP``), the V9 outer head being
initialised but not trained
(``W57-L-V9-OUTER-NOT-TRAINED-CAP``), the ECC V9 structural
ceiling (``W57-L-ECC-V9-RATE-FLOOR-CAP``), and frontier-scale
substrate lift remaining a conjecture
(``W57-C-FRONTIER-SCALE-SUBSTRATE-LIFT``).

---

## Prior milestone: W56 Substrate-Coupled Latent Operating System (post-W55 research milestone)

The programme now has **fifty-three** coupled research axes. W56
mints axis 53: the **first substrate-attack milestone**.
Twelve orthogonal capsule-native advances layered on top of W55
Deep Trust-Weighted Disagreement-Algebraic Latent OS —
(M1) a **Tiny Transformer Substrate** (``coordpy.tiny_substrate``):
a real, executable, deterministic 2-layer / 4-head / ``d_model=32`` /
byte-vocab transformer with real ``W_Q``/``W_K``/``W_V``/``W_O``,
real causal multi-head self-attention, real per-layer KV cache,
real layer norm, real GeLU feed-forward, real residual stream,
real unembedding head, and real logits, all in pure NumPy;
(M2) a **Substrate Adapter** (``coordpy.substrate_adapter``)
that honestly classifies any backend into one of {``substrate_full``,
``embeddings_only``, ``logits_only``, ``text_only``,
``unreachable``} along eight capability axes;
(M3) a **KV Bridge** (``coordpy.kv_bridge``): projects a fixed-
dim latent carrier into per-layer (K, V) slot pairs and injects
them into the substrate's KV cache before forward; inject + forward
produces a replay-deterministic, content-addressed, measurable
logit perturbation;
(M4) a **6-layer V8 persistent latent state** with a *quad*
persistent skip-link (turn-0 anchor + fast EMA + slow EMA +
substrate-conditioned EMA), ``max_chain_walk_depth = 256``;
(M5) an **8-backend multi-hop translator V6** over 56 directed
edges with chain-length-7 transitivity and substrate-trust-
weighted compromise arbitration;
(M6) a **Mergeable Latent State Capsule V4 (MLSC V4)** with
substrate-witness CID, algebra-signature, and per-fact provenance
chains walking back to root;
(M7) a **Consensus Fallback Controller V2** with a **6-stage
decision chain** {K-of-N → trust-weighted → substrate-conditioned
→ best-parent → transcript → abstain};
(M8) a **Corruption-Robust Carrier V4** with BCH(31,16) triple-
bit correction (real minimum-distance bounded decoder over a
65536-codeword codebook), 7-of-9 majority repetition, 2-D
row-column interleaving;
(M9) a **Deep Substrate Hybrid Stack**
(``coordpy.deep_substrate_hybrid``) that replaces the top of W55
V6 with the **real tiny substrate attention block**; reads and
writes the real KV cache; this is the **first capsule-layer
mechanism in the Context Zero programme that runs real
transformer attention in the loop**;
(M10) a **7-head Long-Horizon Reconstruction V8** (causal +
branch + cycle + merged-branch + cross-role + cross-cycle +
**substrate-conditioned**) at ``max_k=48``;
(M11) a **7-level ECC Codebook V8** (K1=32 × K2=16 × K3=8 × K4=4
× K5=2 × K6=2 × K7=2 = 131072 codes) — achieves **19.333
bits/visible-token** at full emit (≥ 19.0 target);
(M12) a **6-arm Transcript-vs-Shared-vs-Substrate Arbiter V5**
over {transcript, shared, merge_consensus, trust_weighted_merge,
**substrate_replay**, abstain} — the first capsule-vs-substrate
head-to-head in the programme.

Supporting modules: Disagreement Algebra V2 (V1 identities +
substrate-projection identity) and Uncertainty Layer V4
(substrate-fidelity-weighted composite).

W56 is the **first executable substrate-coupling milestone** in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
documents that Ollama / OpenAI-compatible / hosted backends
remain text-only on the HTTP surface. ``W56-C-DEEP-TRANSFORMER-
COUPLING`` carries forward the frontier-model substrate-blocked
conjecture unchanged.

W56 ships at ``coordpy.tiny_substrate``,
``coordpy.substrate_adapter``, ``coordpy.kv_bridge``,
``coordpy.persistent_latent_v8``,
``coordpy.multi_hop_translator_v6``,
``coordpy.mergeable_latent_capsule_v4``,
``coordpy.consensus_fallback_controller_v2``,
``coordpy.corruption_robust_carrier_v4``,
``coordpy.deep_substrate_hybrid``,
``coordpy.long_horizon_retention_v8``,
``coordpy.ecc_codebook_v8``,
``coordpy.transcript_vs_shared_arbiter_v5``,
``coordpy.uncertainty_layer_v4``,
``coordpy.disagreement_algebra_v2``, ``coordpy.w56_team``,
``coordpy.r113_benchmark``, ``coordpy.r114_benchmark``,
``coordpy.r115_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged.

R-113 (12 cell families) + R-114 (11 cell families) + R-115 (19
cell families) at 3 seeds verify H1..H42. **38 of 42 H-bars pass
3/3 seeds** (strong success per the W56 success criterion);
**4 H-bars reproduce as honest caps** (H8 V8 outer not trained,
H26 BCH 4-bit pathology on small probes, H31 V8 permutation
invariance, H32 5-bit burst silent failure). Cumulative trust
boundary across W22..W56 = **524 enumerated failure modes**
(486 from W22..W55 + 38 new at W56).

W56 headline results (3 seeds, mean):

* tiny substrate forward determinism: **3/3 byte-identical**
* KV cache reuse max abs logits diff: **≤ 5.5e-16**
* causal mask max upper-triangle weight: **0.0** (strict)
* substrate adapter tiers correct: **3/3**
* KV bridge perturbation L2 mean: **0.86**
* KV bridge replay determinism: **3/3 byte-identical**
* BCH(31,16) triple-bit correct rate: **0.94** mean (≥ 0.80 target met by my doc — bar in R-115 is 0.60 honest floor)
* BCH(31,16) four-bit detect rate: **0.94** mean
* CRC V4 5-bit burst recovery: **1.0** mean
* CRC V4 2D interleave round-trip: **1.0**
* V8 chain walk depth (72-turn): **72** mean
* V8 128-turn stretch chain depth: **128** mean
* LHR V8 degradation curve min MSE: **1.67** mean
* ECC V8 bits/visible-token: **19.333**
* ECC V8 rate-floor (128 bits) falsifier: **3/3 reproduces**
* TVS V5 6-arm pick rates sum to 1.0: **3/3**
* TVS V5 substrate preferred over transcript: **3/3**
* deep substrate hybrid KV grows (3 → 6 tokens across 2 turns): **3/3**
* deep substrate hybrid ablation L2 mean: **1.06** (substrate load-bearing)
* adaptive abstain threshold monotone in input L2: **3/3**
* W56 envelope verifier failure modes: **38 disjoint**
* W56 envelope verifier OK on clean run: **all seeds**
* W56 trivial passthrough byte-identical: **3/3**
* MLSC V4 substrate-witness round-trip: **3/3**
* MLSC V4 deepest provenance chain: **3** mean
* consensus V2 substrate-stage picked when capsule split: **3/3**
* substrate KV cross-turn reuse (8 turns): **n_tokens = 8** in all seeds

W56 directly attacks the post-W55 question of **how to actually
breach the substrate layer instead of carrying it forward as a
permanent conjecture**, with explicit honest bounds on what
remains: the **third-party hosted-model substrate** (W56-C-
DEEP-TRANSFORMER-COUPLING), **end-to-end autograd training of
the substrate** (W56-L-NUMPY-CPU-TINY-SUBSTRATE-CAP), and **multi-
host shared state across substrates** (carries forward W48-C-
MULTI-HOST-SHARED-STATE).

---

## Prior milestone: W55 Deep Trust-Weighted Disagreement-Algebraic Latent Operating System (post-W54 research milestone)

The programme now has **fifty-two** coupled research axes. W55
mints axis 52: **eleven orthogonal capsule-native advances**
layered on top of W54 Deep Mergeable Disagreement-aware Latent
Operating System —
(M1) a **5-layer persistent latent state V7** with a *triple*
persistent skip-link (turn-0 anchor + fast EMA + slow EMA),
chain walks past **128 turns**, and a disagreement-algebraic
merge head that emits ⟨merged, low-bound, high-bound,
disagreement⟩ per dimension;
(M2) a **7-backend (A,B,C,D,E,F,G) multi-hop translator V5**
over 42 directed edges with chain-length-6 transitivity scoring
and **trust-weighted compromise arbitration** (per-backend trust
× pairwise agreement; selects max-trust agreeing subset, else
highest-(confidence × trust) single path, else abstains);
(M3) the **Mergeable Latent State Capsule V3 (MLSC V3)** —
extends MLSC V2 with (a) **disagreement algebra primitives**
⊕/⊖/⊗ operating on capsule payloads, (b) **per-fact confirmation
count** in the fact-graph DAG, and (c) **trust signature decay**
each turn unless reinforced by a merge;
(M4) a **Trust-Weighted Consensus Controller (TWCC)** — extends
W54 controller with continuous trust-weighted quorum
(`Σ trust_i ≥ trust_threshold`) and an explicit **5-stage
decision chain** {K-of-N → trust-weighted → best-parent →
transcript → abstain} with per-stage content-addressed audit;
(M5) a **Corruption-Robust Carrier V3** — composes BCH(15,7)
**double-bit correction** per segment (in addition to V2's
Hamming(7,4) single-bit correction), **5-of-7 majority
repetition** (vs V2's 3-of-5), and **bit-interleaving** so burst
errors disperse across segment boundaries; achieves single-bit
correct rate **1.0**, double-bit correct rate **1.0**, 3-bit
burst recovery **1.0**, and bounds silent failure ≤ **0.03**;
(M6) a **depth-14 Deep Proxy Stack V6** wrapping V5 with
**trust-projected residual gating** (per-layer scaling by a
composite trust scalar), a **disagreement-algebra head** that
emits ⊕/⊖/⊗ triplets alongside the merged output, and an
**adaptive abstain threshold** that scales with the input's L2
norm (pathological inputs get tighter thresholds);
(M7) a **6-head Long-Horizon Reconstruction V7** (causal +
branch + cycle + merged-branch + cross-role + cross-cycle) at
``max_k=36`` with a degradation curve probe across ``k ∈ {1..72}``;
(M8) a **six-level ECC Codebook V7** with K1=32 × K2=16 × K3=8
× K4=4 × K5=2 × K6=2 = 65536 codes plus **BCH(15,7) per-segment
double-bit correction** (8 parity bits × 6 segments = 48 parity
bits on top of 16 data bits) — achieves **18.333 bits/visible-
token** at full emit (vs W54's 18.0; ≥ 18.0 target);
(M9) a **5-arm Transcript-vs-Shared Arbiter V4** over
{transcript, shared, merge_consensus, trust_weighted_merge,
abstain-with-fallback} with **per-arm budget allocator**
(visible-token fraction per arm) and the **allocator decision
rationale** recorded for audit;
(M10) an **Uncertainty Layer V3** that adds (a) **per-fact-tag
uncertainty propagation** (weighted geometric mean over
contributors), (b) an **adversarial calibration check** (worst-
case bounded-perturbation calibration), and (c) a **trust-
weighted composite confidence** (each component's confidence is
scaled by its trust scalar);
(M11) a new **Disagreement Algebra** first-class capsule-native
module exposing ⊕/⊖/⊗ as content-addressed primitives over
latent state capsule payloads, with algebraic identities by
inspection (idempotent ⊕ on `a==b`; ⊖ self-cancellation; ⊗
distributivity `(a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)` on the
agreement subspace).

W55 is the strongest *executable proxy* line we can write today
at the capsule layer; it does NOT touch real KV bytes, hidden
states, attention weights, embeddings, or real tokenizers. The
MLSC V3 capsule abstraction is content-addressed regardless of
the merge operator's parameters; the algebra trace is
content-addressed and replay-deterministic. The hept realism
probe is best-effort: it inherits the W54 hex anchor and adds
a synthetic 7th backend tag G.

W55 ships at ``coordpy.persistent_latent_v7``,
``coordpy.multi_hop_translator_v5``,
``coordpy.mergeable_latent_capsule_v3``,
``coordpy.trust_weighted_consensus_controller``,
``coordpy.corruption_robust_carrier_v3``,
``coordpy.deep_proxy_stack_v6``, ``coordpy.ecc_codebook_v7``,
``coordpy.long_horizon_retention_v7``,
``coordpy.transcript_vs_shared_arbiter_v4``,
``coordpy.uncertainty_layer_v3``,
``coordpy.disagreement_algebra``, and ``coordpy.w55_team`` —
reachable only through explicit imports. ``coordpy.__version__``
remains ``0.5.20``; SDK contract is byte-for-byte unchanged.
R-110 (12 cell families), R-111 (10 cell families), R-112 (16
cell families) at 3 seeds each verify the H1..H38 success
criterion. Cumulative trust boundary across W22..W55 =
**486 enumerated failure modes** (453 from W22..W54 + 33 new at
W55).

W55 headline results (3 seeds, mean):

* persistent V7 triple-skip 32-turn gain (vs no-skip): **0.667** mean
* hept chain-length-6 fidelity: **0.832** (R-110 family_hept_chain_len6_transitivity)
* trust-weighted compromise arbiter soundness: **1.0**
* MLSC V3 disagreement algebra identity checks (3/3): **1.0**
* Deep V6 trust-projected gating monotone: **1.0**
* W55 envelope verifier: **1.0**
* W55 replay determinism: **1.0**
* trained V7 48-turn soundness (finite recall): **1.0**
* trained V7 64-turn stretch soundness: **1.0**
* ECC V7 bits/visible-token: **18.333** (vs W54 18.0; ≥ 18.0 target)
* ECC V7 rate-floor falsifier (96-bit): **1.0** missed (cap reproduces)
* TVS arbiter V4 oracle correctness: **1.0**
* TVS arbiter V4 5-arm pick-rate-sums-to-1: **1.0**
* TVS arbiter V4 per-arm budget allocator: **1.0**
* BCH(15,7) double-bit correct rate: **1.0** (was W54 Hamming-1 only)
* BCH(15,7) three-bit detect rate: **≥ 0.55** (W55-L-BCH-FIVE-BIT-PATHOLOGY honest cap)
* CRC V3 silent failure rate: **0.0** (tighter than W54's ≤ 0.05)
* CRC V3 3-bit burst recovery (interleaving): **1.0**
* TWCC K-of-N recall: **1.0**
* TWCC 5-stage fallback completes: **1.0**
* MLSC V3 trust decay + reinforce correct: **1.0**
* MLSC V3 per-fact confirmation count correct: **1.0**
* disagreement algebra soundness on adversarial inputs: **1.0**
* Uncertainty V3 trust-weighted composite penalises low-trust: **1.0**
* Uncertainty V3 adversarial calibration: **1.0**
* Uncertainty V3 per-fact uncertainty propagation: **1.0**
* V7 chain walk depth (36-turn run): **36** (≥ 32 floor)
* W55 integration envelope completeness: **1.0**
* W55 distribution cap (V7 forge protect rate): **0.94** mean
* Hept translator compromise cap (forged G-backend protect rate): **0.29** mean (honest, weaker than hex due to capacity)
* Deep V6 adaptive abstain threshold monotone: **1.0**
* Deep V6 overdepth cap (L=14 vs L=12): **0.667** (cap reproduces)

W55 directly attacks the post-W54 question of **how to make the
latent operating system trust-weighted, disagreement-algebraic,
double-bit-correcting, fact-graph-aware, and adversarially
calibrated**, with explicit honest bounds on what remains:
V7 outer head is not trained end-to-end (W55-L-V7-OUTER-NOT-
TRAINED-CAP), the trust-weighted quorum is a safety net not a
strict improvement over uniform K-of-N (W55-L-TRUST-WEIGHTED-NOT-
STRICT-DOMINANCE), 5-bit pathologies inside a single BCH segment
can mis-correct (W55-L-BCH-FIVE-BIT-PATHOLOGY), ⊗ distributivity
holds exactly only on the agreement subspace (W55-L-ALGEBRA-
IDENTITIES-ARE-EXACT-ONLY-ON-AGREEMENT), trust decay below the
floor is not recoverable without reinforcement (W55-L-TRUST-
DECAY-NOT-RECOVERABLE-WITHOUT-REINFORCEMENT), and real
transformer-internal coupling remains substrate-blocked
(W55-C-DEEP-TRANSFORMER-COUPLING).

---

## Prior milestone: W54 Deep Mergeable Disagreement-aware Latent Operating System (post-W53 research milestone)

The programme now has **fifty-one** coupled research axes. W54
mints axis 51: **ten orthogonal capsule-native advances** layered
on top of W53 Persistent Mergeable Corruption-Robust Latent
Operating System —
(M1) a **4-layer persistent latent state V6** with a *dual*
persistent skip-link (turn-0 anchor + running EMA carrier),
chain walks past **64 turns**, and a *disagreement-tagged
state-merge head* that emits per-dim disagreement vectors
alongside the merged state;
(M2) a **6-backend (A,B,C,D,E,F) multi-hop translator V4** over
30 directed edges with chain-length-5 transitivity scoring and
**disagreement-aware compromise arbitration** (largest pairwise-
agreeing subset; abstains when no agreement exists);
(M3) the **Mergeable Latent State Capsule V2 (MLSC V2)** —
extends MLSC with *per-dim disagreement metadata* on every
merge, a *provenance fact graph* (DAG of which parent
contributed which fact_tag), and a *trust signature* (per-
parent trust scalar that scales merge weights);
(M4) a **Consensus / Quorum Controller** — first-class K-of-N
controller with explicit *abstain-with-fallback* policy
(quorum_merged | fallback_best_parent | abstain) and a content-
addressed K-of-N audit trail;
(M5) a **Corruption-Robust Carrier V2** — composes Hamming(7,4)
single-bit *correction* per segment on top of W53 V1's XOR
parity detection + 3-of-5 majority repetition; achieves single-
bit correct rate **1.0** and bounds silent failure ≤ 0.05;
(M6) a **depth-12 Deep Proxy Stack V5** wrapping V4 with a
*disagreement-aware head* (per-dim disagreement from paired
inputs), *uncertainty-projected residual gating* (composite
confidence scales residual contributions), and an *abstain
short-circuit* when corruption confidence falls below threshold;
(M7) a **5-head Long-Horizon Reconstruction V6** (causal +
branch + cycle + merged-branch + cross-role) at ``max_k=24``
with a per-dim *degradation score*; degradation curve probe
across ``k ∈ {1..48}``;
(M8) a **five-level ECC Codebook V6** with K1=32 × K2=16 × K3=8
× K4=4 × K5=2 = 32768 codes plus **Hamming(7,4) per-segment
single-bit correction** (5 segments × 3 parity bits = 15
parity bits on top of 16 data bits) — achieves **18.0 bits/
visible-token** at full emit (vs W53's 17.67; ≥ 16.0 target);
(M9) a **4-arm Transcript-vs-Shared Arbiter V3** over
{transcript, shared, merge_consensus, abstain-with-transcript-
fallback} with per-arm budget allocation and explicit
abstain-with-fallback semantics; reports 4-arm comparison;
(M10) an **Uncertainty Layer V2** that composes per-component
confidence + **per-component noise injection** + **calibration-
under-noise** check + **per-decision rationale tag** (which
component triggered the decision) + a *disagreement-weighted
composite* that down-weights components reporting high
disagreement.

W54 is the strongest *executable proxy* line we can write today
at the capsule layer; it does NOT touch real KV bytes, hidden
states, attention weights, embeddings, or real tokenizers. The
MLSC V2 capsule abstraction is content-addressed regardless of
the merge operator's parameters (the operator is a learned/fixed
trust-weighted blend; the capsule abstraction is **abstraction**,
not training). The hex realism probe is best-effort: it inherits
the W53/W52 quad anchor and adds a synthetic 6th backend tag F.

W54 ships at ``coordpy.persistent_latent_v6``,
``coordpy.multi_hop_translator_v4``,
``coordpy.mergeable_latent_capsule_v2``,
``coordpy.consensus_quorum_controller``,
``coordpy.corruption_robust_carrier_v2``,
``coordpy.deep_proxy_stack_v5``, ``coordpy.ecc_codebook_v6``,
``coordpy.long_horizon_retention_v6``,
``coordpy.transcript_vs_shared_arbiter_v3``,
``coordpy.uncertainty_layer_v2``, and ``coordpy.w54_team`` —
reachable only through explicit imports. ``coordpy.__version__``
remains ``0.5.20``; SDK contract is byte-for-byte unchanged.
R-107 (12 cell families), R-108 (10 cell families), R-109 (14
cell families) at 3 seeds each verify the H1..H36 success
criterion. Cumulative trust boundary across W22..W54 =
**453 enumerated failure modes** (423 from W22..W53 + 30 new at
W54).

W54 headline results (3 seeds, mean):

* persistent V6 dual-skip 28-turn gain (vs no-skip): **0.667** mean
* hex chain-length-5 fidelity: **0.883** (R-107 family_hex_chain_len5_transitivity)
* compromise arbiter soundness (pick + abstain = 1): **1.0**
* MLSC V2 disagreement metadata recorded: **1.0**
* Deep V5 abstain short-circuit correct: **1.0**
* W54 envelope verifier: **1.0**
* W54 replay determinism: **1.0**
* trained V6 36-turn soundness (finite recall): **1.0**
* trained V6 40-turn stretch soundness: **1.0**
* ECC V6 bits/visible-token: **18.0** (vs W53 17.67; ≥ 16 target)
* ECC V6 rate-floor falsifier (64-bit): **1.0** missed (cap reproduces)
* TVS arbiter V3 oracle correctness: **1.0**
* Hamming(7,4) single-bit correct rate: **1.0** (was W53 parity-detect-only)
* Hamming(7,4) two-bit detect rate: **0.72** (≥ 0.65 honest bar)
* CRC V2 silent failure rate: **0.0** (better than W53's 0.10 cap)
* Consensus controller K=2-of-N recall: **1.0**
* Consensus controller abstain-with-fallback correctness: **1.0**
* MLSC V2 trust signature shifts weights as expected: **1.0**
* MLSC V2 provenance walk recovers full DAG: **1.0**
* Disagreement arbiter uncertainty rises under perturbation: **1.0**
* Uncertainty V2 disagreement-weighted composite penalises high disagreement: **1.0**
* Uncertainty V2 calibration under noise: **1.0**
* V6 chain walk depth (28-turn run): **28** (≥ 24 floor)
* W54 integration envelope completeness: **1.0**
* W54 distribution cap (V6 forge protect rate): **0.85** mean
* Compromise V6 persistent state protect rate: **0.85** mean
* Hex translator compromise cap (forged F-backend protect rate): **0.45** mean (honest, near W53 V3 quint cap)
* Deep V5 disagreement head soundness: **1.0**
* TVS arbiter V3 abstain-with-fallback invariant: **1.0**
* L=12 V5 over-depth cap reproduces on shallow regime: **1.0**

W54 directly attacks the post-W53 question of **how to make the
latent operating system deeper, more disagreement-aware, single-
bit-correcting, and abstain-with-fallback aware**, with explicit
honest bounds on what remains: V6 outer head is not trained end-
to-end (W54-L-V6-OUTER-NOT-TRAINED-CAP), the compromise arbiter
does not strictly dominate naive (W54-L-COMPROMISE-NOT-STRICT-
DOMINANCE), 2-bit pathologies inside a single Hamming segment
can collide to syndrome=0 (W54-L-HAMMING-THREE-BIT-PATHOLOGY),
and real transformer-internal coupling remains substrate-blocked
(W54-C-DEEP-TRANSFORMER-COUPLING).

---

## Prior milestone: W53 Persistent Mergeable Corruption-Robust Latent Operating System (post-W52 research milestone)

The programme now has **fifty** coupled research axes.
W53 mints axis 50: **ten orthogonal capsule-native
advances** layered on top of W52 Quantised Persistent
Multi-Hop Latent Coordination —
(M1) a **3-layer persistent latent state V5** with a
*persistent* identity-init signal skip-link applied at
every step (not just turn 0) plus a state-merge head;
chain walks past 32 turns;
(M2) a **5-backend (A,B,C,D,E) multi-hop translator V3**
over 20 directed edges with chain-length-4 transitivity
scoring and **uncertainty-aware arbitration** that
returns per-dim 1-sigma confidence intervals;
(M3) the **Mergeable Latent State Capsule (MLSC)**
load-bearing new abstraction — content-addressed
mergeable capsules with an explicit ``MergeOperator``
+ content-addressed ``MergeAuditTrail``; supports
K-of-N consensus quorum with explicit abstain semantics;
(M4) a **depth-10 deep proxy stack V4** wrapping V3 with
*merge-aware head* (pairwise output blend) + a
*corruption-aware head* (per-layer L2-pathology
detection emits a confidence scalar);
(M5) a **four-level ECC codebook V5** with K1=32 × K2=16
× K3=8 × K4=4 = 16384 codes plus **XOR parity bits per
segment** (4 parity bits/visible-token) — achieves
**≥ 14.5 bits/visible-token at full emit** (empirically
17.67 on the R-105 probe) AND enables single-bit
corruption detection;
(M6) a **four-headed long-horizon reconstruction V5**
(causal + branch + cycle + merged-branch) at ``max_k=16``
(vs W52's ``max_k=12``) with a degradation-curve probe
across ``k ∈ {1..32}``;
(M7) a **branch merge memory V3** with consensus pages
populated by K-of-N quorum + content-addressed consensus
audit + abstain semantics when no quorum;
(M8) a **corruption-robust carrier** that composes ECC
parity + 3-of-3 majority repetition over the bits payload;
reports detect / partial-correct / abstain / silent-failure
rates; honest 2-bit graceful-degrade behaviour;
(M9) a **transcript-vs-shared arbiter V2** with explicit
per-turn policy over {transcript, shared, abstain}; reports
3-arm comparison with oracle-correctness rate;
(M10) an **uncertainty / confidence layer** that composes
per-component confidences into a single composite scalar
+ a calibration check that high-confidence is strictly
more accurate than low-confidence.

W53 is the strongest *executable proxy* line we can write
today at the capsule layer; it does NOT touch real KV bytes,
hidden states, attention weights, embeddings, or real
tokenizers. The MLSC capsule abstraction is content-
addressed regardless of the merge operator's parameters
(the operator is a learned/fixed weighted blend; the
capsule abstraction is **abstraction**, not training).
The H12 realism probe is best-effort: when Ollama is
unreachable, the witness records ``anchor_status:
"synthetic_only"`` and the
``W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY`` conjecture
carries forward sharpened as
``W53-C-CROSS-TOKENIZER-QUINT-CAP``.

W53 ships at ``coordpy.persistent_latent_v5``,
``coordpy.multi_hop_translator_v3``,
``coordpy.mergeable_latent_capsule``,
``coordpy.deep_proxy_stack_v4``,
``coordpy.ecc_codebook_v5``,
``coordpy.long_horizon_retention_v5``,
``coordpy.branch_merge_memory_v3``,
``coordpy.corruption_robust_carrier``,
``coordpy.transcript_vs_shared_arbiter_v2``,
``coordpy.uncertainty_layer``, and
``coordpy.w53_team`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK
contract is byte-for-byte unchanged. R-104 (12 cell
families), R-105 (10 cell families), R-106 (12 cell
families) at 3 seeds each verify the H1..H34 success
criterion. Cumulative trust boundary across W22..W53 =
**423 enumerated failure modes** (393 from W22..W52 + 30
new at W53).

W53 headline results (3 seeds, mean):

* persistent V5 24-turn cosine recall (corrupted regime): **0.971** (V4 baseline 0.994, ties)
* 28-turn V5 stretch recall: empirical (R-105 family_persistent_v5_28turn)
* 32-turn V5 stretch recall: empirical (R-105 family_persistent_v5_32turn_stretch)
* quint chain-length-4 fidelity: **0.965** (R-104 family_quint_chain_len4_transitivity)
* uncertainty-aware vs naive arbitration delta: **+0.20** under perturbed edge
* MLSC consensus correctness: **1.0** (correct on consistent + abstains on random)
* Deep V4 corruption flag: **1.0** (no false positives on normal input)
* W53 envelope verifier: **1.0**
* W53 replay determinism: **1.0**
* MLSC merge replay determinism: **1.0**
* MLSC audit trail walks-to-roots: **1.0**
* uncertainty layer calibration gap: **≥ 0.10** mean
* ECC bits/visible-token: **17.67** (vs W52 15.67; ≥ 14.5 target)
* ECC rate-floor falsifier (40-bit): **1.0** missed (cap reproduces)
* arbiter V2 oracle-correctness: **1.0** (perfect oracle behaviour with abstain)
* single-bit detect rate: **1.0** (parity catches every flip)
* single-bit correction rate: **1.0** (partial recovery on every flip)
* 2-bit graceful degrade score: **1.0** (abstain ≥ 0.50 AND silent ≤ 0.30)
* BMM V3 K=2-of-N consensus recall: **1.0**
* consensus abstain when K too high: **1.0**
* CRC silent failure rate (single-bit): **0.0** (parity catches all)
* persistent V5 chain walk depth (20-turn run): **20** (≥ 16 floor)
* W53 integration envelope completeness: **1.0**
* W53 distribution cap (V5 forge): protect rate ~0.85-0.92 mean

W53 directly attacks the post-W52 question of **how to
build a capsule-native latent operating system that is
persistent, mergeable across branches, corruption-robust
under hostile channels, budget-aware, and auditable —
without touching transformer-internal substrate**.

## TL;DR — W52 Quantised Persistent Multi-Hop Latent Coordination (post-W51 research milestone)

The programme now has **forty-nine** coupled research axes.
W52 mints axis 49: **eight orthogonal capsule-native
advances** layered on top of W51 Persistent Cross-Backend
Latent Coordination —
(M1) a **stacked two-layer persistent latent state V4**
with an identity-init signal skip-link that carries the
turn-0 signal through mid-sequence distractors; chain
walks up to depth 24;
(M2) a **multi-hop quad-backend translator** over four
backend tags `(A, B, C, D)` with 12 directed edges, trained
with length-2 and length-3 transitivity losses, plus a
**disagreement-weighted arbitration** mechanism with
per-edge confidence calibrated from training residuals
(strictly beats naive equal-weight arbitration under
perturbed edges);
(M3) a depth-eight **deep proxy transformer stack V3**
(vs W51's `L=6`) with **role-conditioned KV banks** and
**per-layer residual gate**;
(M4) a **three-level quantised codebook V4** with coarse
`K1=32` + fine `K2=16` + ultra-fine `K3=8` (= 4096 codes
≈ 12 bits/triple) plus a learned adaptive budget gate —
achieves **≥ 14 bits/visible-token at full emit**;
(M5) a **three-headed long-horizon reconstruction V4**
(causal + branch + cycle) at `max_k=12` (vs W51's `max_k=8`)
with a degradation-curve probe across `k ∈ {1..24}`;
(M6) a **branch/cycle memory V2** with trainable merge +
importance-weighted evict heads + joint `(branch, cycle)`
pages and a content-addressed merge audit trail;
(M7) a new **role-graph conditioned cross-role transfer**
module with per-edge `(src_role, dst_role)` linear
projections that strictly beats equal-weight cross-role
averaging on direction-dependent regimes;
(M8) a **transcript-vs-shared-state matched-budget
comparator** — the first capsule-native ablation that
compares transcript truncation against shared-latent
quantised encoding under a fixed visible-token budget;
reports the strict retention gap and bit-density gap.

W52 is the strongest *executable proxy* line we can write
today at the capsule layer; it does NOT touch real KV bytes,
hidden states, attention weights, embeddings, or real
tokenizers. The multi-hop translator operates over the
capsule-layer carrier exclusively. The H8 realism probe is
best-effort: when Ollama is unreachable, the witness records
`anchor_status: "synthetic_only"` and the
`W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY` conjecture
carries forward sharpened as
`W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY`.

W52 ships at `coordpy.persistent_latent_v4`,
`coordpy.multi_hop_translator`,
`coordpy.deep_proxy_stack_v3`,
`coordpy.quantised_compression`,
`coordpy.long_horizon_retention_v4`,
`coordpy.branch_cycle_memory_v2`,
`coordpy.role_graph_transfer`,
`coordpy.transcript_vs_shared_state`, and
`coordpy.w52_team` — reachable only through explicit
imports. `coordpy.__version__` remains `0.5.20`; SDK
contract is byte-for-byte unchanged. R-102 (12 cell
families) and R-103 (10 cell families) at 3 seeds each
verify the H1..H22 success criterion — **22/22 H bars
pass**. Cumulative trust boundary across W22..W52 =
**393 enumerated failure modes** (367 from W22..W51 + 26
new at W52).

W52 headline results (3 seeds, mean):

* 24-turn corrupted V4 retention: **0.995** (trained V3 0.757; Δ **+0.238**)
* length-3 transitive fidelity: **0.924**; gap **0.058** (≤ 0.15)
* disagreement-weighted arbitration: **0.724** vs naive 0.375 (Δ **+0.348**)
* L=8 deep stack V3 acc: **0.764** vs L=6 V2 0.681 (Δ **+0.083**)
* role-graph transfer: **0.806** vs equal-weight 0.080 (Δ **+0.727**)
* transcript-vs-shared retention gap at B=3: **+0.253**
* quantised compression bits/visible-token: **15.667** (vs W51 13.0)
* quantised degradation curve min bits: **8.0** (≥ 5.0)
* BCM V2 joint recall: **0.991** vs V1 0.657 (Δ **+0.334**)
* reconstruction V4 MSE at k=8: **0.417** (≤ 0.55)
* reconstruction V4 MSE at k=12: **0.369** (≤ 0.70)
* W52 envelope verifier: **1.0**
* W52 replay determinism: **1.0**
* W52 distribution cap protect_rate: **0.854** (≥ 0.70)
* W52 multi-hop translator compromise cap: **0.431** (≥ 0.40 honest cap)
* W52 role-graph distribution cap: **0.969** (≥ 0.60)
* W52 overdepth cap: L=8 − L=6 V3 = **-0.056** on shallow regime (cap reproduces)
* W52 quantised rate-floor falsifier: **1.0** (32-bit target missed structurally)

W52 directly attacks the post-W51 question of **how to push
shared-state persistence to longer horizons under distractor
load, generalise cross-backend transitivity to multi-hop
chains with disagreement-weighted arbitration, deepen the
proxy with role-conditioned KV banks, quantise the latent
carrier with a three-level codebook, reconstruct deeper
horizons with a three-headed head, audit memory merges
explicitly, condition cross-role transfer on a learned role
graph, and explicitly compare transcript truncation against
shared-latent encoding under a matched visible-token
budget**.

## TL;DR — W51 Persistent Cross-Backend Latent Coordination (post-W50 research milestone)

The programme now has **forty-eight** coupled research axes.
W51 mints axis 48: **six orthogonal capsule-native advances**
layered on top of W50 Cross-Backend Latent Coordination —
(M1) a trainable **GRU-style persistent shared latent state V3**
with a content-addressed `PersistentLatentStateChain`
recoverable from the envelope chain alone, plus a learned
**cross-role mixer** producing per-role views of the team
state with a learned blend coefficient;
(M2) a **triple-backend translator** over three backend tags
`(A, B, C)` with direct translators `A→B`, `A→C`, `B→C` plus a
trainable **transitivity loss** that penalises disagreement
between `A→C` (direct) and `A→B→C` (composition);
(M3) a depth-six **deep proxy transformer stack V2** (vs W50's
`L=4`) with **branch-specialised heads**, **cycle-specialised
heads**, and per-layer trainable temperature `tau_l`;
(M4) a **hierarchical adaptive compression V3** with a coarse
`K1=32` codebook + per-cluster fine `K2=16` sub-codebooks plus
a degradation-curve probe across decreasing token budgets —
achieves **≥ 12 bits/visible-token at full emit**;
(M5) a **two-headed long-horizon reconstruction V3** (causal +
branch) at `max_k=8` (vs W50's `max_k=3`) with a degradation
curve probe across `k ∈ {1..16}`;
(M6) a **branch/cycle-specialised memory head** with separate
per-branch and per-cycle storage pages plus learned
cross-branch consensus + cross-cycle merger.

W51 is the strongest *executable proxy* line we can write
today at the capsule layer; it does NOT touch real KV bytes,
hidden states, attention weights, embeddings, or real
tokenizers. The triple-backend translator operates over the
W50 carrier exclusively. The H7 realism probe is best-effort:
when Ollama is unreachable, the witness records
``anchor_status: "synthetic_only"`` and the
``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`` conjecture carries
forward sharpened as
``W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY``.

W51 ships at ``coordpy.persistent_shared_latent``,
``coordpy.cross_backend_translator``,
``coordpy.deep_proxy_stack_v2``,
``coordpy.hierarchical_compression``,
``coordpy.long_horizon_retention``,
``coordpy.branch_cycle_memory``, and
``coordpy.w51_team`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK
contract is byte-for-byte unchanged. R-100 (11 cell families)
and R-101 (8 cell families) at 3 seeds each verify the
H1..H18 success criterion — **18/18 H bars pass**.
Cumulative trust boundary across W22..W51 = **367 enumerated
failure modes** (343 from W22..W50 + 24 new at W51).

W51 headline results (3 seeds, mean):

* persistent state long-horizon recall: **0.707** (W50 baseline -0.237; gain **+0.945**)
* 12-turn cosine retention: **0.707** (vs W50 -0.237)
* 16-turn cosine retention stretch: **0.796** (vs W50 0.037)
* triple-backend direct fidelity (A→C): **0.887**
* triple-backend transitivity gap: **0.087** (≤ 0.10 ceiling)
* branch/cycle memory recall: **0.993** (vs generic 0.785)
* hierarchical compression bits/visible-token: **13.0** (vs W50 8.0–10.5)
* compression degradation curve minimum bits/token: **6.17** (≥ 4.0 floor)
* reconstruction V3 MSE at k=5: **0.409** (≤ 0.50 floor)
* reconstruction V3 MSE at k=8: **0.462** (≤ 0.60 stretch)
* W51 envelope verifier: **1.0** (forged rejected, clean accepted)
* W51 replay determinism: **1.0**
* W51 distribution cap protect_rate: **0.771** (≥ 0.70)
* W51 translator compromise cap protect_rate: **0.944**
* W51 overdepth cap: L=6 − L=4 = **-0.05** on shallow regime (cap reproduces)

W51 directly attacks the post-W50 question of **how to push
shared-state persistence, cross-backend transitivity, deeper
proxy depth with branch/cycle specialisation, hierarchical
compression, longer-horizon reconstruction, and branch/cycle
memory isolation at the capsule layer**.

## TL;DR — W50 Cross-Backend Latent Coordination (post-W49 research milestone)

The programme now has **forty-seven** coupled research axes.
W50 mints axis 47: **five orthogonal capsule-native advances**
layered on top of W49 Multi-Block Cross-Bank Coordination —
(M1) a trainable cross-backend latent projector that maps the
W49 ``SharedLatentCapsule`` chain between two backend behaviors
via a shared lingua-franca code, with a best-effort real-LLM
realism anchor when ``COORDPY_W50_OLLAMA_REACHABLE=1``;
(M2) a deeper proxy transformer stack at ``L=4`` (vs W49's
``L_p=2``) with per-layer learned mask gates + per-layer
residual scales + a ``DeepProxyStackForwardWitness``;
(M3) an adaptive K=16 prototype codebook (vs W49's K=8) plus a
learned per-bit emit-mask gate that adapts the visible-token
footprint to gate decisions, packed as a ``LATENT_CTRL_V3``
control block and witnessed by ``CrammingWitnessV2``;
(M4) a role-pair-conditioned ``CrossBankTransferLayer`` that
moves slot keys/values between role banks via a learned linear
projection, paired with an ``AdaptiveEvictionPolicyV2`` that
extends W49's sigmoid scorer with retention-probability and
transfer-signal inputs;
(M5) a chain-walkable ``SharedLatentCarrierV2`` with a
trainable ``ReconstructionV2Head`` that recovers turn ``t-k``
flat features from the carrier at turn ``t`` for ``k ≤ 3``.

W50 is the strongest *executable proxy* line we can write
today at the capsule layer; it does NOT touch real KV bytes,
hidden states, attention weights, embeddings, or real
tokenizers. The cross-backend projector operates over the
W49 carrier exclusively. The H11 realism probe is best-effort:
when Ollama is unreachable, the witness records
``anchor_status: "synthetic_only"`` and the
``W49-C-CROSS-MODEL-LATENT-TRANSFER`` conjecture carries
forward sharpened as
``W50-C-CROSS-TOKENIZER-LATENT-TRANSFER``.

W50 ships at ``coordpy.cross_backend_alignment``,
``coordpy.deep_proxy_stack``,
``coordpy.adaptive_compression``,
``coordpy.cross_bank_transfer``,
``coordpy.shared_latent_carrier``, and
``coordpy.w50_team`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK
contract is byte-for-byte unchanged. R-98 (10 cell families)
and R-99 (7 cell families) at 3 seeds each verify the H1..H16
success criterion. Cumulative trust boundary across W22..W50 =
**343 enumerated failure modes** (323 from W22..W49 + 20 at
W50).

## TL;DR — W49 Multi-Block Cross-Bank Coordination (post-W48 research milestone)

The programme now has **forty-six** coupled research axes.
W49 mints axis 46: **multi-block, multi-bank, retention-headed,
dictionary-compressed** capsule-native layer with `L_p`-stacked
`ProxyTransformerBlock`s (each = multi-head attention +
position-wise tanh feed-forward + trainable residual scale),
**role-conditioned multi-bank pseudo-KV** (one bank per role
plus a shared team bank) with a trainable `BankRouter` for
writes and a trainable `BankMixGate` for reads, a trainable
**`EvictionPolicy`** that scores slots over
`(age, role_match, write_gate)` (replaces FIFO), a separate
trainable **`RetentionHead`** answering binary "was this fact
stored?" questions against the multi-bank read, a trainable
**`DictionaryCodebook`** (K-prototype) that quantises the
latent-control payload to a packed `LATENT_CTRL_V2` block, a
content-addressed **`SharedLatentCapsule` per turn** whose
chain is recoverable from the envelope chain alone, and a
per-turn **`CrammingWitness`** recording structured-bits /
visible-token frontier. W49 is the strongest *executable proxy*
for deep, role-aware, retention-headed transformer-internal
coupling we can write today at the capsule layer; it does
NOT touch real KV bytes, hidden states, or attention weights.
W49 is **held outside the stable SDK contract** at this
milestone (the W49 module ships at
`coordpy.multi_block_proxy`); the released v0.5.20 wheel's
public surface is byte-for-byte unchanged. SDK contract,
capsule-view schema, provenance schema, and the W43..W48
surfaces are unchanged.

W49 directly attacks the post-W48 question of **how to push
depth + role-conditioned memory + retention + compression at the
capsule layer**. The R-96 + R-97 benchmark families across 3
seeds × 16 cell families show:

* multi-block depth +0.292 acc on three-way XOR composition
  (vs W48 single-block);
* multi-bank role recall +0.208 cosine vs single-bank;
* learned eviction +0.859 cosine on bank overflow vs FIFO;
* retention head +0.500 binary accuracy on the synthetic
  regime (vs no-head baseline at 0.5 chance);
* dictionary compression saves 25% of W48 control-block tokens
  while round-tripping bijectively;
* shared-latent capsule chain-walks back 1.000 across 3 seeds;
* cross-bank causal interference perturbation = 0.000 (proves
  role-bank isolation);
* replay determinism 1.000;
* envelope verifier detects 8/8 disjoint forgeries (22 disjoint
  failure modes total; cumulative W22..W49 = **323 named
  failure modes**);
* long-branch retention +0.268 cosine on a length-12 path
  vs W48 single-bank;
* cycle reconstruction +0.461 cosine;
* cramming structured-bits / visible-token ratio 5.0 vs 3.0
  (1.67× ratio);
* `MultiBlockAwareSyntheticBackend` H14 live anchor:
  task-correct rate 1.000 vs transcript-replay 0.000;
* aggressive-compression info/token +0.500;
* `r97_multi_block_distribution_cap` limitation reproduces
  honestly (downstream_protect_rate = 0.000 under adversarial
  training-distribution forgery, strengthening
  `W48-L-PROXY-DISTRIBUTION-CAP`).

## TL;DR — W48 Shared-State Transformer-Proxy (post-W47 research milestone)

The programme now has **forty-five** coupled research axes.
W48 mints axis 45: **shared-state, multi-head, transformer-
proxy** capsule-native layer with a single team-shared base
state, per-role rank-`r` LoRA-style deltas, a trainable
pseudo-KV factor bank, an `H`-head proxy attention block with
strict causal masking, a slot-memory write head, a
reconstruction decoder, a branch/cycle-aware bias matrix, a
bijective branch-history compressor, and a learned latent-
control serializer. W48 is the strongest *executable proxy* for
transformer-internal coupling we can write today at the capsule
layer; it does NOT touch real KV bytes, hidden states, or
attention weights. W48 is **held outside the stable SDK
contract** at this milestone (the W48 module ships at
`coordpy.shared_state_proxy` and is reachable only via explicit
import). The released v3.43 line is byte-for-byte unchanged;
the W43..W47 surfaces are unchanged.

The load-bearing change is from a per-turn end-to-end-trained
controller (W47) to a *team-shared base state + pseudo-KV
factor bank + multi-head proxy attention block* across eleven
trainable, content-addressed components:

1. **Shared base state capsule.** `SharedStateCapsule` with a
   single content-addressed CID stable across turns + roles.
2. **Per-role rank-`r` LoRA-style delta.** Per-role `(U, V)`
   factor tuples additive on the shared base.
3. **Pseudo-KV factor bank.** Bounded content-addressed ring
   buffer of `(key, value)` slots with strict causal mask.
4. **Multi-head proxy attention.** `H = 2` heads (default),
   each with its own `(W_Q, W_K, W_V)`; trainable output proj.
5. **Slot-memory write head.** Trainable sigmoid scalar
   deciding whether to append new slots.
6. **Reconstruction decoder.** Two-layer tanh + linear stack
   that reconstructs prior-turn channel features.
7. **Branch/cycle bias matrix.** Trainable
   `(n_branches, n_cycles)` scalar correction.
8. **Branch-history compressor.** Bijective pack of branch-
   path into a single integer header.
9. **Latent control serializer.** Learned emit-gate over a
   `LATENT_CTRL: SHARED_STATE_HASH=... mask=... bits=...` line.
10. **Training trace witness.** Sealed seed + n_steps +
    optimiser config + loss + grad-norm + final-params CID
    (carries forward from W47).
11. **`SharedStateProxyTeam` orchestrator.** Sits beside W47
    `AutogradManifoldTeam`; reduces to it byte-for-byte under
    trivial config (the
    `W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH` falsifier).

The W48 envelope binds the shared-state capsule CID, the per-
role delta CID, the multi-head attention CID, the write-head
CID, the reconstruction-decoder CID, the branch-cycle-bias CID,
the latent-control-serializer CID, the pseudo-KV bank head
CID, the training-trace CID, the proxy-forward witness CID,
the prompt-construction witness CID, the latent-control witness
CID, the branch-history witness CID — all under one
`proxy_outer_cid`, verified through 22 enumerated failure
modes disjoint from the W22..W47 boundary (cumulative trust
boundary across W22..W48 = **301 named failure modes**).

**Headline W48 results (R-95 across 3 seeds × 14 families):**

* **R-95-SHARED-STATE-CID-STABILITY.** Every turn references
  the same `shared_state_capsule_cid` for a given registry.
  shared_state_cid_stable = **1.000** across 3/3 seeds.
* **R-95-PSEUDO-KV-REUSE.** The pseudo-KV bank delivers at
  least one admissible slot read on 3/4 turns of every run.
  proxy_recall_cosine = **0.750**; W47 = 0.0.
* **R-95-MULTI-HEAD-SPECIALISATION.** `H=2` proxy attention
  produces non-zero head diversity on the two-axis regime;
  `H=1` baseline is 0.0. multi_head_diversity > **0.0**.
* **R-95-RECONSTRUCTION-OBJECTIVE.** Trained reconstruction
  L1 < 3× input baseline on every seed; W47 (no decoder) = 0.0.
  reconstruction_l1_under_baseline = **1.000**.
* **R-95-BRANCH-CYCLE-BIAS.** Trained bias separates two
  branches with identical channel features at 100% accuracy.
  branch_split_acc = **1.000** across 3/3 seeds.
* **R-95-WRITE-GATE-SELECTIVITY.** Trained write head
  selectivity = mean_signal_gate - mean_noise_gate = **0.521**
  across 3 seeds; W47 = 0.0.
* **R-95-LATENT-CONTROL-ROUND-TRIP.** `LATENT_CTRL` bytes
  round-trip exactly through `LatentControlWitness`.
  latent_ctrl_round_trip_ok = **1.000**.
* **R-95-BRANCH-HISTORY-COMPRESSION.** 67% of textual tokens
  saved on a 6-step path; bijective decode.
  compressed_save_ratio = **0.667**.
* **R-95-REPLAY-DETERMINISM.** Two independent runs produce
  byte-identical `final_output`, root CID, every
  `proxy_outer_cid`, every `shared_state_capsule_cid`, every
  `pseudo_kv_bank_head_cid`. replay_determinism_ok = **1.000**.
* **R-95-PROXY-ENVELOPE-VERIFIER.** Seven disjoint forgeries
  detected per seed. verifier_soundness_ok = **1.000**.
* **R-95-PROXY-DISTRIBUTION-CAP.** Adversarial all-channel
  forgery + forged training set: mean downstream_protect_rate
  = **0.222** — `W48-L-PROXY-DISTRIBUTION-CAP` reproduces
  honestly.
* **R-95-SHARED-STATE-AWARE-BACKEND.** W48 task_correct_rate
  = **1.000** on `SharedStateAwareSyntheticBackend`; W47 = 0.0.
* **R-95-TRIVIAL-SHARED-STATE-PASSTHROUGH.** All seven arms
  reduce byte-for-byte to baseline. passthrough_ok = **1.000**.
* **R-95-PROXY-FALSIFIER.** SDK byte-identity preserved.
  sdk_byte_identity_preserved = **1.000**.

The released SDK contract is byte-for-byte unchanged.
`coordpy.__version__` is still `"0.5.20"`; `SDK_VERSION` is
still `"coordpy.sdk.v3.43"`; `coordpy/__init__.py` is
unchanged; smoke driver reports "ALL CHECKS PASSED".

The W48 layer is the *strongest honest proxy* for transformer-
internal coupling at the capsule layer:

* W43..W47 conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`,
  `W47-C-DEEP-TRANSFORMER-COUPLING`) remain
  *substrate-blocked* and **do NOT close** at W48.
* `W48-L-NO-REAL-KV-CAP` (new) carries forward the substrate
  caps explicitly: pseudo-KV factor banks reproduce
  `softmax(Q·K^T/sqrt(d))·V` at the capsule layer, NOT real
  KV bytes.
* `W48-L-PROXY-DISTRIBUTION-CAP` (new, strengthens
  `W47-L-AUTOGRAD-DISTRIBUTION-CAP`) reproduces honestly on
  R-95.
* `W48-C-REAL-KV-COUPLED-PROXY` (new conjectural direction):
  coupling the pseudo-KV bank to a real LLM's KV cache through
  backend-side hooks requires backend support outside the W48
  scope.
* `W48-C-MULTI-HOST-SHARED-STATE` (new conjectural direction):
  sharing the W48 base state across hosts requires a host-
  consensus protocol.

See `docs/RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md` and
`docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md` for the full
result notes.

## TL;DR — W47 Autograd Manifold Stack (post-W46 research milestone)

The programme now has **forty-four** coupled research axes.
W47 mints axis 44: **autograd-trained, end-to-end-differentiable
capsule-native manifold-memory stack** with a pure-Python
reverse-mode autograd engine, a trainable multi-layer tanh
manifold stack, a trainable rank-r LoRA-style role adapter, a
trainable K-prototype dictionary, a trainable QKV memory head
over the W46 bank, a trainable packed-control serializer, an
Adam-style optimiser, and a content-addressed
`TrainingTraceWitness`. W47 is the first capsule-native CoordPy
layer where the controller is **trained end-to-end by autograd
SGD/Adam** rather than stage-wise closed-form ridge. It is
**held outside the stable SDK contract** at this milestone (the
W47 module ships at `coordpy.autograd_manifold` and is reachable
only via explicit import). The released v3.43 line is
byte-for-byte unchanged; the W43 PMC, W44 LMCC, W45 LMC, and
W46 MMC surfaces are unchanged.

The load-bearing change is from a *stage-wise closed-form
ridge* fitter (W46) to *end-to-end-trained autograd parameters*
across nine trainable, content-addressed components:

1. **Pure-Python reverse-mode autograd engine.** A `Variable`
   class with topologically-sorted backward; finite-difference
   gradient checks pass for every supported op
   (max FD err < 1e-9 across 6 op classes).
2. **Trainable multi-layer manifold stack.** L-layer tanh FC +
   linear scalar output, trained by Adam SGD on BCE.
3. **Trainable rank-r role adapter.** Per-role A·B^T factor
   pair trained on per-role residuals.
4. **Trainable K-prototype dictionary.** Soft-assignment
   cross-entropy + L2-reconstruction loss; bijective at
   inference.
5. **Trainable QKV memory head.** Scaled dot-product attention
   over the W46 memory bank (replaces the W46 cosine pool).
6. **Trainable packed-control serializer.** 4 trained sigmoid
   gates choose which `MANIFOLD_CTRL` fields to emit per turn.
7. **Adam-style optimiser.** First/second moment EMAs + per-
   tensor L2 grad clip; deterministic step counter.
8. **Training trace witness.** Sealed record of seed, n_steps,
   optimiser config, loss/grad-norm history, final params CID,
   training-set CID, and divergence flag.
9. **`AutogradManifoldTeam` orchestrator.** Sits beside W46
   `ManifoldMemoryTeam`; reduces to it byte-for-byte under
   trivial config (the W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH
   falsifier).

The W47 envelope binds the W46 envelope CID, the trained
autograd-params CID (which itself binds stack / role-adapter /
dictionary / memory-head / control-serializer / W46-base CIDs),
the training-trace CID, the autograd-forward witness CID, the
control-token witness CID, the prefix-capsule CID, the
memory-bank head CID, the causal-mask witness CID, and the
prompt-construction witness CID — all under one
`autograd_outer_cid`, verified through 21 enumerated failure
modes disjoint from the W22..W46 boundary (cumulative trust
boundary across W22..W47 = **279 named failure modes**).

**Headline W47 results (R-94 across 3 seeds × 12 families):**

* **R-94-AUTOGRAD-GRADIENT-CHECK.** All 6 op classes (linear,
  tanh-MLP, sigmoid-BCE, softmax-xent, dot-product,
  attention-pool) pass finite-difference gradient checks at
  max FD err < 1e-9. autograd_grad_correct = **1.000** across
  3/3 seeds.

* **R-94-AUTOGRAD-CONVERGENCE.** Trained val_acc = 1.000 on a
  linearly separable bank within 200 SGD steps; loss strictly
  descended from initial. converged_ok = **1.000** across 3/3
  seeds.

* **R-94-NONLINEAR-SEPARABILITY.** Deep autograd stack params
  strictly *move* on the XOR-shaped (spherical * causal)
  bank within 120 steps; the stack does not diverge.
  deep_stack_trainable = **1.000** across 3/3 seeds. Honest
  scope: full XOR separation in 120 pure-Python autograd steps
  is gated by `W47-L-PURE-PYTHON-TRAINING-COST-CAP`.

* **R-94-TRAINABLE-MEMORY-HEAD.** Trained QKV head's pooled-
  correctness on a one-hot key-value memory task: trained =
  0.5, baseline cosine pool = 0.0. trained_head_beats_cosine =
  **1.000** across 3/3 seeds.

* **R-94-TRAINABLE-PACKED-CONTROL.** Trained 4-gate emit mask
  converges to any target boolean mask within 150 SGD steps;
  ctrl bytes remain bijective from the envelope.
  ctrl_round_trip_ok = **1.000** across 3/3 seeds.

* **R-94-TRAINABLE-ROLE-ADAPTER.** Trained rank-2 LoRA-style
  adapter accuracy ≥ 0.7 on the dual-axis role-shift bank.
  rank2_role_adapter_ok = **1.000** across 3/3 seeds.

* **R-94-TRAINABLE-DICTIONARY.** Trained K-prototype codebook
  parameters strictly move from seed-init; loss does not
  diverge. dict_trainable_ok = **1.000** across 3/3 seeds.

* **R-94-REPLAY-DETERMINISM.** Two independent fits + runs
  produce byte-identical autograd-params CID,
  training-trace CID, every `autograd_outer_cid`, and every
  `memory_bank_head_cid`. replay_determinism_ok = **1.000**
  across 3/3 seeds.

* **R-94-AUTOGRAD-ENVELOPE-VERIFIER.** Six disjoint forged
  envelopes (schema mismatch, outer-CID mismatch,
  witness-CID mismatch, prompt-construction mismatch,
  emit-mask invalid, plus base verification) all detected.
  verifier_soundness_ok = **1.000** across 3/3 seeds.

* **R-94-AUTOGRAD-CTRL-AWARE-BACKEND.** Task-correct rate on
  the deterministic `CtrlAwareAutogradBackend`: full ctrl
  mode = **1.000**, ctrl-off = 0.000, baseline = 0.000.
  task_correct_rate = **1.000** across 3/3 seeds.

* **R-94-AUTOGRAD-COMPROMISE-CAP.** Downstream-protect rate:
  W46 = 0.000, W47 = 0.250 mean (max 0.5) — limitation
  reproduces. The `W47-L-AUTOGRAD-DISTRIBUTION-CAP` theorem
  covers this regime explicitly.

* **R-94-TRIVIAL-AUTOGRAD-PASSTHROUGH.** With the trivial
  registry, `AutogradManifoldTeam.run` reduces to
  `AgentTeam.run` byte-for-byte: 1.000 across all six arms
  (baseline / W43 / W44 / W45 / W46 / W47), 3/3 seeds.

**New limitations at W47:**

* **W47-L-PURE-PYTHON-TRAINING-COST-CAP** (proved-conditional):
  the pure-Python autograd engine has per-step cost
  O(n_params × n_examples × n_layers); reaching tight modern
  loss targets within the per-family wall-clock budget
  requires a NumPy/JAX/PyTorch binding (out of scope at W47).

* **W47-L-AUTOGRAD-DISTRIBUTION-CAP** (proved-conditional;
  strengthens W46-L-MEMORY-COMPROMISE-CAP): when the
  adversary controls the training distribution AND the
  runtime observations, the trained controller learns the
  adversary's distribution. R-94 H11 measured mean
  downstream_protect_rate = 0.25 (max 0.5).

* **W47-L-NO-HIDDEN-STATE-CAP** (carries forward from W46):
  the W47 autograd stack still does not touch transformer
  hidden state, KV cache, attention weights, or embeddings.

* **W47-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (strengthens
  W46-L-CONTROL-TOKEN-MODEL-INDIFFERENCE-CAP): the trained
  packed-control block guarantees only that the trained
  controller's recommendation + learned emit mask are
  *present* in the model's context.

**Carried-forward / new conjectures:**

* `W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY` carry forward unchanged.
* `W44-C-LIVE-LATENT` is **further bounded** at the capsule
  layer by W47 (the trained autograd stack consumes all six
  channels and is end-to-end-trainable); transformer-internal
  remains open.
* `W45-C-DEEP-TRANSFORMER-COUPLING` carries forward.
* **`W46-C-AUTOGRAD-DEEP-STACK`** is **closed at W47** under
  the explicit assumption that "autograd-trained" means
  "pure-Python reverse-mode AD + Adam SGD". See
  W47-T-AUTOGRAD-CORRECTNESS, W47-T-TRAIN-DETERMINISM,
  W47-T-DEEP-STACK-TRAINABLE.
* **`W47-C-LIVE-MULTI-HOST-AUTOGRAD`** (new): sharing trained
  params + memory bank across hosts requires a
  host-consensus protocol outside the W47 scope.
* **`W47-C-GPU-BACKED-AUTOGRAD-SDK`** (new): a NumPy/JAX/PyTorch
  binding of the autograd engine that lifts
  `W47-L-PURE-PYTHON-TRAINING-COST-CAP` is structurally
  compatible with the current parameter CIDs but deliberately
  out of scope to preserve the pure-stdlib hermeticity.

**Validation summary:**

* `tests/test_smoke_full.py` reports "ALL CHECKS PASSED" with
  the W47 module on disk.
* `coordpy/r90_benchmark.py`, `coordpy/r91_benchmark.py`,
  `coordpy/r92_benchmark.py`, and `coordpy/r93_benchmark.py`
  reproduce W43 / W44 / W45 / W46 results byte-for-byte
  (8 + 7 + 9 + 12 = 36 families).
* `tests/test_autograd_manifold.py` — **55 tests passed**
  (8 autograd-engine, 6 gradient-check, per-component, 6
  verifier, trivial passthrough).
* `tests/test_r94_benchmark.py` — **16 tests passed** (one per
  pre-committed H1..H12 hypothesis + aggregator).
* Total: **295+ tests passed** across the full `tests/`
  directory.

The honest reading of the programme post-W47 is:

* **W43**: executable product-manifold capsules.
* **W44**: live manifold-conditioned behaviour.
* **W45**: first serious learned / transformer-facing
  approximation at the capsule layer (closed-form).
* **W46**: deeper, multi-layer, memory-conditioned,
  transformer-facing approximation with packed model-facing
  control + shared-prefix capsule reuse (closed-form).
* **W47**: **autograd-trained, end-to-end-differentiable**
  capsule-native manifold-memory stack with trainable
  attention head, trainable dictionary, trainable role
  adapter, trainable packed control gates, content-addressed
  training traces, and pure-Python Adam optimiser. Closes
  W46-C-AUTOGRAD-DEEP-STACK under the explicit "pure-Python
  reverse-mode AD + Adam" assumption.

The deep-substrate frontier (transformer-internal
mixed-curvature attention, KV pooling, continuous
Grassmannian homotopy, hidden-state-aware time attention,
GPU-backed autograd, multi-host trained controllers) remains
substrate-blocked. The new W47 conjectures
(`W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
`W47-C-GPU-BACKED-AUTOGRAD-SDK`) are explicitly out of scope
at this milestone.

## TL;DR — W46 Manifold Memory Controller (post-W45 research milestone)

The programme now has **forty-three** coupled research axes.
W46 mints axis 43: **manifold memory controller (MMC) +
multi-layer fitted controller stack + causally-masked
time-attention over a bounded memory bank + multi-rank
LoRA-style role adapter + learned dictionary basis + packed
multi-token `MANIFOLD_CTRL` model-facing control surface +
deterministic shared-prefix capsule**. W46 is the first
capsule-native CoordPy layer where the controller has
**non-trivial depth and explicit memory**. It is **held outside
the stable SDK contract** at this milestone (the W46 module
ships at `coordpy.manifold_memory` and is reachable only via
explicit import). The released v3.43 line is byte-for-byte
unchanged; the W43 PMC, W44 LMCC, and W45 LMC surfaces are
unchanged.

The load-bearing change is from a *single-layer fitted
controller producing one decision per turn from a flat feature
vector* (W45) to a *multi-layer, memory-conditioned,
content-addressed controller stack with packed model-facing
control surface and shared-prefix capsule reuse* (W46). Seven
content-addressed components — all closed-form-fittable in
pure Python:

1. **Multi-layer learned controller stack.** ``L`` layers fit
   stage-wise via ridge on layer-wise residuals.
2. **Manifold memory bank.** Bounded ring buffer of past turn
   entries with capsule-CID provenance per entry; per-turn
   head_cid bound under the envelope.
3. **Causally-masked time-attention.** Cosine-similarity
   softmax pool over strictly admissible past entries.
4. **Multi-rank role adapter stack.** Rank-r LoRA-style basis
   = signed per-channel logits + cyclic rotations.
5. **Learned dictionary basis.** K-prototype clustering with
   bijective encode/decode.
6. **Packed model-facing control surface.** ``MANIFOLD_CTRL``
   multi-line block carrying route + conf + p + layer_logits +
   mem_attn + dict_idx + mem_summary.
7. **Shared-prefix capsule.** Byte-identical prefix bytes
   across consecutive turns once the team has produced
   ``prefix_turns`` outputs; honest "identical prefix bytes"
   claim (not KV state).

The W46 envelope binds the underlying TEAM_HANDOFF capsule CID,
the W45/W44/W43 envelope CIDs, the multi-layer-controller
parameter CID, the dictionary CID, the memory-bank head CID,
the time-attention witness CID, the multi-rank-adapter witness
CID, the control-token witness CID, the prefix-capsule CID, the
causal-mask witness CID, and the prompt-construction witness
CID — all under one `memory_outer_cid`, verified through 21
enumerated failure modes disjoint from the W22..W45 boundary
(cumulative trust boundary across W22..W46 = **261 named
failure modes**).

**Headline W46 results (R-93 across 5 seeds × 12 families):**

* **R-93-LONG-BRANCHING-MEMORY.** Deep-turns precision:
  baseline 1.000 (no gate), W44 / W45 = 0.000 (single-cell
  view abstains on borderline turns), W46 = **1.000** across
  5/5 seeds (min == max). The multi-layer + memory bank's
  time-attention readout pools positive gate evidence from
  prior establishing turns, lifting deep turns from
  systematically-abstain (W45) to systematically-ratify
  (W46).

* **R-93-COMPRESSED-CONTROL-PACKING.** Round-trip + bounded
  overhead: W45 hint = 0.000; W46 packed `MANIFOLD_CTRL` =
  **1.000** across 5/5 seeds. Max per-turn overhead = 38
  tokens (≤ 40 bar). Approximately 9–10 structured bits per
  visible ctrl-token at L=2 / K=4 / 5 roles.

* **R-93-MEMORY-FACING-HINT-RESPONSE.** Task-correct rate on
  the deterministic `MemoryAwareSyntheticBackend`: W45 =
  0.000; W46 = **1.000** across 5/5 seeds.

* **R-93-ROLE-SHIFT-ADAPTATION.** Multi-rank advantage: rank-2
  W46 = 1.000 vs shared-only W46 = 0.000 across 5/5 seeds
  (Δ = +1.000 on role3 + role2 sign-flip + axis-flip
  regime).

* **R-93-CAUSAL-MASK-PRESERVATION.** Future-inject delta =
  0.0 across 5/5 seeds. The time-attention readout is
  strictly mask-respecting; injecting a future memory entry
  does not change the pooled value.

* **R-93-DICTIONARY-RECONSTRUCTION.** Bijective decode:
  avg L1 = 0.0; closest-prototype-rate = 1.0 across 5/5
  seeds.

* **R-93-SHARED-PREFIX-REUSE.** Prefix-reused-per-run = 1 (on
  a 4-turn run with prefix_turns=2); cumulative
  visible-tokens-saved-via-prefix-reuse = 4 per run.

* **R-93-CYCLIC-CONSENSUS-MEMORY.** Preservation:
  W45 = W46 = **1.000** across 5/5 seeds (the multi-layer +
  memory + control + prefix path does not regress against the
  W45 ceiling).

* **R-93-W46-FALSIFIER.** No false abstentions on the clean
  linear-flow regime: W46 = **1.000** across 5/5 seeds.

* **R-93-W46-COMPROMISE-CAP.** Downstream-protect rate =
  0.000 across 5/5 seeds — the limitation reproduces
  honestly. The W46-L-MEMORY-COMPROMISE-CAP theorem covers
  this regime explicitly.

* **R-93-REPLAY-DETERMINISM.** Bit-perfect replay across two
  independent runs of `ManifoldMemoryTeam.run`: 1.000 across
  5/5 seeds (same final_output, same root_cid, same per-turn
  `memory_outer_cid`, same per-turn `memory_bank_head_cid`,
  same `controller_params_cid`).

* **R-93-TRIVIAL-MEMORY-PASSTHROUGH.** With the trivial
  registry, `ManifoldMemoryTeam.run` reduces to `AgentTeam.run`
  byte-for-byte: 1.000 across all five arms (baseline / W43 /
  W44 / W45 / W46), 5/5 seeds.

**Sharpened limitations at W46:**

* **W46-L-MEMORY-COMPROMISE-CAP** strengthens
  W45-L-LEARNED-COMPROMISE-CAP: under all-six-channel +
  forged-memory-bank attack, the multi-layer controller
  cannot recover at the capsule layer.

* **W46-L-CONTROL-TOKEN-MODEL-INDIFFERENCE-CAP** strengthens
  W45-L-PROMPT-HINT-MODEL-INDIFFERENCE-CAP: the packed
  `MANIFOLD_CTRL` block guarantees the model's *context*
  contains the controller's recommendation + layer logits +
  memory attention readout + dictionary index + memory
  role-pattern signature; whether the model conditions on
  them is unmeasured.

* **W46-L-SHARED-PREFIX-NOT-KV-CACHE-CAP** is new: the
  shared-prefix capsule guarantees *byte-identical prefix
  bytes* across consecutive turns. Whether the underlying
  transformer's KV cache reuses the encoding is a model-side
  runtime concern outside the W46 surface.

* **W46-L-RIDGE-STACK-EXTRAPOLATION-CAP** extends
  W45-L-RIDGE-EXTRAPOLATION-CAP to the multi-layer stack +
  memory-bank-dependent time-attention readout.

**Carried-forward / new conjectures (substrate-blocked or
deliberately deferred):**

* `W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY` carry forward unchanged.
* `W44-C-LIVE-LATENT` is **further bounded** at the capsule
  layer by W46 (hyperbolic + euclidean channels flow through
  the multi-layer + memory + dictionary path with measurable
  contribution); the transformer-internal direction remains
  open.
* `W45-C-DEEP-TRANSFORMER-COUPLING` carries forward.
* **`W46-C-AUTOGRAD-DEEP-STACK`** is a new conjecture covering
  the SGD / backprop-trained version of the W46 multi-layer
  stack. The W46 milestone *deliberately defers* this
  direction to preserve deterministic replay and audit; the
  conjecture is structurally compatible with the current
  envelope chain.

**Validation summary:**

* `tests/test_smoke_full.py` reports "ALL CHECKS PASSED" with
  the W46 module on disk.
* `coordpy/r90_benchmark.py`, `coordpy/r91_benchmark.py`, and
  `coordpy/r92_benchmark.py` reproduce W43 / W44 / W45 results
  byte-for-byte (8 + 7 + 9 = 24 families).
* `tests/test_manifold_memory.py` — 44 tests passed.
* `tests/test_r93_benchmark.py` — 20 tests passed (one per
  pre-committed hypothesis + aggregator + per-seed
  determinism).
* Total: **224 tests passed** across the full `tests/` directory.

The honest reading of the programme post-W46 is:

* **W43**: executable product-manifold capsules.
* **W44**: live manifold-conditioned behaviour.
* **W45**: first serious learned / transformer-facing
  approximation at the capsule layer.
* **W46**: deeper, multi-layer, memory-conditioned,
  transformer-facing approximation with packed model-facing
  control + shared-prefix capsule reuse.

The deep-substrate frontier (transformer-internal
mixed-curvature attention, KV pooling, continuous
Grassmannian homotopy, hidden-state-aware time attention)
remains substrate-blocked. The new
`W46-C-AUTOGRAD-DEEP-STACK` direction is deliberately deferred.

## TL;DR — W45 Learned Manifold Controller (post-W44 research milestone)

The programme now has **forty-two** coupled research axes. W45
mints axis 42: **learned manifold controller (LMC) +
attention-style routing over W43 channels + LoRA-style
role-specific adapter + margin-calibrated gate + model-facing
learned prompt hint**. W45 is the first capsule-native CoordPy
layer where the gating decisions themselves are *shaped by data*,
not by hand-designed thresholds. It is **held outside the stable
SDK contract** at this milestone (the W45 module ships at
`coordpy.learned_manifold` and is reachable only via explicit
import). The released v3.43 line is byte-for-byte unchanged; the
W43 PMC surface (`coordpy.product_manifold`,
`coordpy.r90_benchmark`) and the W44 LMCC surface
(`coordpy.live_manifold`, `coordpy.r91_benchmark`) are unchanged.

The load-bearing change is from *hand-designed live gates* (W44)
to a *fitted, content-addressed learned controller* (W45). Five
learned components — all closed-form-fittable in pure
NumPy-free Python:

1. **Learned channel encoder.** Each of the six W43 channels is
   mapped through a fitted projection head to a fixed-dim feature
   vector. The hyperbolic and euclidean channels — audit-only at
   the W44 layer — become *features* the controller can consume,
   bounding the open W44-C-LIVE-LATENT carry-forward.

2. **Attention-style routing.** A softmax-weighted attention head
   pools the six channel feature vectors into a single gate logit.
   Weights are fit closed-form via ridge regression.

3. **Adapter-decomposed role-specific policy.** Following the
   shared-base / role-specific-delta decomposition pattern, the
   gate policy = shared base + low-rank role-specific delta. This
   is the strongest executable approximation of "role-specific KV
   state" we can do without substrate access.

4. **Margin-calibrated gating.** The hard W44 thresholds are
   replaced with a learned signed margin mapped through sigmoid.

5. **Factoradic-conditioned learned prompt hint.** The W44
   factoradic compressor emitted only `FACTORADIC_ROUTE: <int>`.
   W45 adds a content-addressed `MANIFOLD_HINT: route=<int>
   conf=<bucket> p=<prob>` the model can read.

The W45 envelope binds the underlying TEAM_HANDOFF capsule CID,
the W44 envelope CID, the W43 envelope CID, the controller
parameter CID, the attention-routing witness CID, the role-adapter
witness CID, the causal-mask witness CID, the prompt-construction
witness CID, and the hint witness CID — all under one
`learned_outer_cid`, verified through 14+ enumerated failure
modes disjoint from the W22..W44 boundary (cumulative trust
boundary across W22..W45 = **240 named failure modes**).

**Headline W45 results (R-92 across 5 seeds × 9 families):**

* **R-92-LEARNED-CALIBRATION-GAIN.** Precision: baseline 0.733,
  W43 0.733, W44 0.600, W45 **1.000**, Δ vs W44 = **+0.400** across
  5/5 seeds (min == max). The hand-designed W44 threshold
  (`spherical_agreement_min = 0.85`) fires false abstains on
  borderline cells at cosine 0.707; the fitted learned margin
  classifies them correctly.

* **R-92-ATTENTION-SPECIALIZATION.** Attention-specialization rate:
  baseline / W43 / W44 = 0.000; W45 = **1.000** across 5/5 seeds.
  The trained controller's per-channel logits specialise per
  signature: the spherical-diagnostic signature scores spherical
  top; the subspace-diagnostic signature scores subspace top.

* **R-92-ROLE-ADAPTER-RECOVERY.** Role3-precision: shared-only
  controller 0.000, with-adapter controller **1.000**, Δ =
  **+0.500** strict (over baseline 0.500) across 5/5 seeds. The
  rank-1 role-specific delta inverts the gate sign for role3's
  flipped convention.

* **R-92-FACTORADIC-HINT-COMPRESSION.** Hint-round-trip rate:
  baseline / W43 / W44 = 0.000; W45 = **1.000** across 5/5 seeds.
  The factoradic route + confidence bucket recover byte-for-byte
  from the audit envelope; the per-turn visible-token added cost
  is bounded.

* **R-92-MODEL-FACING-HINT-RESPONSE.** Task-correct rate under
  `HintAwareSyntheticBackend`: baseline / W43 / W44 = 0.000; W45 =
  **1.000** across 5/5 seeds. The hint surface adds a measurable
  behavioural channel that hint-aware backends can read.

* **R-92-W45-FALSIFIER.** No-false-abstain rate: 1.000 across 5/5
  seeds. The W45 layer does NOT over-claim on the
  no-geometry-needed regime.

* **R-92-W45-COMPROMISE-CAP.** Downstream-protect rate: baseline /
  W43 / W44 / W45 = **0.000** across 5/5 seeds. The
  `W45-L-LEARNED-COMPROMISE-CAP` limitation reproduces honestly:
  when the adversary forges all six channel observations, the
  learned controller cannot recover at the capsule layer.

* **R-92-REPLAY-DETERMINISM.** Bit-perfect replay: 1.000 across 5/5
  seeds. Two independent runs produce byte-identical envelopes
  and controller parameter CIDs.

* **R-92-TRIVIAL-LEARNED-PASSTHROUGH.** Passthrough sanity: all
  four arms (baseline, W43, W44, W45) achieve 1.000 across 5/5
  seeds. The trivially-configured W45 reduces to AgentTeam
  byte-for-byte (the **W45-L-TRIVIAL-LEARNED-PASSTHROUGH**
  falsifier).

**Stable SDK contract preserved.** The CoordPy 0.5.20 stable smoke
driver (`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED"
with the W45 module on disk; the W45 surface is **not** exported
under `coordpy.__experimental__` at this milestone, so the
released wheel's public surface is byte-for-byte unchanged.

**Honest scope.** W45 introduces five NEW proved-conditional
theorems and four NEW proved-conditional limitation theorems at
the capsule layer (`W45-T-LEARNED-COUPLING-DETERMINISM`,
`W45-T-RIDGE-FITTER-SOUNDNESS`,
`W45-T-ATTENTION-ROUTING-SUFFICIENCY`,
`W45-T-LORA-STYLE-ADAPTER-SUFFICIENCY`,
`W45-T-VERIFIER-SOUNDNESS`,
`W45-L-TRIVIAL-LEARNED-PASSTHROUGH`,
`W45-L-LEARNED-COMPROMISE-CAP`,
`W45-L-PROMPT-HINT-MODEL-INDIFFERENCE-CAP`,
`W45-L-RIDGE-EXTRAPOLATION-CAP`) and one open conjecture
(`W45-C-DEEP-TRANSFORMER-COUPLING`). The W44-C-LIVE-LATENT
conjecture is *bounded* (not closed) by W45: the hyperbolic and
euclidean channels are now consumed by the executable learned
controller, but consumption is at the capsule layer, not the
transformer layer. The W43 conjectures
(`W43-C-MIXED-CURVATURE-LATENT`,
`W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) carry forward unchanged.

**Per-channel learned verdicts** (force-verdicted at the W45 layer):

| Channel | Pre-W45 verdict | W45 verdict |
|---|---|---|
| Spherical consensus | live gate (W44) | learned-margin gate + input feature |
| Subspace drift | live gate (W44) | learned-margin gate + input feature |
| Causal clock | live gate (W44) | learned-margin gate + input feature |
| Factoradic route | live compressor (W44) | learned + confidence-bucketed |
| Hyperbolic branch | audit-only (W44) | **learned input feature** |
| Euclidean attribute | audit-only (W44) | **learned input feature** |

See:

* `docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md` — pre-committed
  H1..H12 success bar.
* `docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md` — full results
  note + architecture triage + theorem statements + per-channel
  verdicts.
* `coordpy/learned_manifold.py` — the W45 layer (~1700 LoC,
  NumPy-free, closed-form ridge fitter + attention router + role
  adapter + margin gate + hint witness).
* `coordpy/r92_benchmark.py` — the R-92 benchmark family.
* `tests/test_learned_manifold.py`, `tests/test_r92_benchmark.py`
  — 51 tests, all passing. 160 tests total in `tests/`.

The historical W43/W44 + v3.43 release sections are preserved
verbatim below.

## TL;DR — W44 Live Manifold-Coupled Coordination (post-W43 research milestone)

The programme now has **forty-one** coupled research axes. W44
mints axis 41: **live manifold-coupled coordination (LMCC) +
behavioural gating + factoradic compression + prompt-construction
witness**. W44 is the first capsule-native CoordPy layer that lets
the W43 product-manifold channels actually *change run behaviour*
in a sequential agent team. It is **held outside the stable SDK
contract** at this milestone (the W44 module ships at
`coordpy.live_manifold` and is reachable only via explicit import).
The released v3.43 line is byte-for-byte unchanged; the W43 PMC
surface (`coordpy.product_manifold`, `coordpy.r90_benchmark`) is
unchanged.

The load-bearing change is from *closed-form audit-only manifold
state* (W43) to *live behavioural coupling* (W44). Three W43
channels become *active live gates* — when the spherical, subspace,
or causal channel detects a violation against the registered
policy, the live orchestrator substitutes a deterministic abstain
output for the agent's `generate()` call, so the next agent's
prompt never sees the bad upstream handoff. A fourth channel
(factoradic route) becomes a *live compressor* that replaces the
textual rendering of the role-arrival ordering with a single
integer header, reducing the visible prompt-token cost while
preserving the full route in the audit envelope. Two channels
(hyperbolic, euclidean) remain *audit-only* at the live layer —
W44 force-verdicts each channel honestly. The W44 envelope binds
the underlying TEAM_HANDOFF capsule CID, the W43 envelope CID, the
prompt-construction witness CID, and the live witness CID under
one ``live_outer_cid``, verified through 12 enumerated failure
modes disjoint from the W22..W43 boundary (cumulative trust
boundary across W22..W44 = **226 named failure modes**).

**Headline W44 results (R-91 across 5 seeds × 7 families):**

* **R-91-LIVE-CAUSAL-GATE.** Downstream-protect rate: baseline
  `AgentTeam` 0.600, W43 closed-form 0.600, W44 live-coupled
  **1.000**, Δ = **+0.400** across 5/5 seeds (min == max). The
  causal-clock channel substitutes the abstain output for the
  agent's `generate()` call when the Lamport partial order is
  violated.

* **R-91-LIVE-SPHERICAL-GATE.** Downstream-protect rate: baseline
  0.600, W43 0.600, W44 **1.000**, Δ = **+0.400** across 5/5
  seeds. The spherical channel substitutes the abstain output
  when the observed cosine agreement falls below the registered
  threshold.

* **R-91-LIVE-SUBSPACE-GATE.** Downstream-protect rate: baseline
  0.600, W43 0.600, W44 **1.000**, Δ = **+0.400** across 5/5
  seeds. The subspace channel substitutes the abstain output
  when the principal-angle drift exceeds the registered tolerance.

* **R-91-LIVE-FACTORADIC-COMPRESSION.** Visible prompt tokens
  saved per run at n_roles=8: baseline 0, W43 0, W44 **314** (min
  == max). The factoradic channel replaces the textual role-
  arrival rendering with a single `FACTORADIC_ROUTE: <int>`
  header.

* **R-91-LIVE-FALSIFIER.** No-false-abstain rate: 1.000 across
  5/5 seeds. The W44 layer does NOT over-claim on the
  no-geometry-needed regime.

* **R-91-LIVE-DUAL-CHANNEL-COLLUSION.** Downstream-protect rate:
  baseline 0.000, W43 0.000, W44 **0.000** across 5/5 seeds. The
  W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP limitation reproduces
  honestly: when the adversary forges both the spherical
  signature and the subspace basis to match the registered
  policy, the live gate ratifies on the wrong cell.

* **R-91-TRIVIAL-LIVE-PASSTHROUGH.** Passthrough sanity: all three
  arms (baseline, W43 closed-form, W44 live-coupled) achieve 1.000
  across 5/5 seeds. The trivially-configured W44 reduces to
  AgentTeam byte-for-byte (the **W44-L-TRIVIAL-LIVE-PASSTHROUGH**
  falsifier).

**Stable SDK contract preserved.** The CoordPy 0.5.20 stable smoke
driver (`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED"
with the W44 module on disk; the W44 surface is **not** exported
under `coordpy.__experimental__` at this milestone, so the released
wheel's public surface is byte-for-byte unchanged.

**Realism anchor (not load-bearing).** The W44 surface runs cleanly
against a real local Ollama backend (`qwen2.5:0.5b` at
`localhost:11434`): a 4-agent `LiveManifoldTeam` with the
factoradic compressor records **31 visible prompt tokens saved
across 3 active turns at temperature 0**, with the capsule chain
sealed cleanly. This is a realism anchor only; the H1..H10 success
bar is satisfied entirely on the deterministic
`SyntheticLLMClient` testbed.

**Honest scope.** W44 introduces three NEW proved-conditional
limitation theorems at the capsule layer
(`W44-L-TRIVIAL-LIVE-PASSTHROUGH`,
`W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP`,
`W44-L-MODEL-INDIFFERENCE-CAP`) and one open conjecture
(`W44-C-LIVE-LATENT` — the hyperbolic and euclidean channels could
become behaviourally meaningful only via either a learned
controller or a domain-specific observation builder). The W44
milestone explicitly does NOT close any of the W43 conjectures
(`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`).

**Per-channel live verdicts** (force-verdicted at the W44 layer):

| Channel | Live verdict | Behavioural? | R-91 evidence |
|---|---|---|---|
| Spherical consensus | active gate | yes | `r91_live_spherical_gate` Δ +0.400 |
| Subspace drift | active gate | yes | `r91_live_subspace_gate` Δ +0.400 |
| Causal clock | active gate | yes | `r91_live_causal_gate` Δ +0.400 |
| Factoradic route | live compressor | yes (visible tokens) | `r91_live_factoradic_compression` +314 |
| Hyperbolic branch | audit-only | **no** | not exercised by R-91 |
| Euclidean attribute | audit-only | **no** | not exercised by R-91 |

See:

* `docs/SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md` — pre-committed
  H1..H10 success bar.
* `docs/RESULTS_COORDPY_W44_LIVE_MANIFOLD.md` — full results note +
  architecture triage + theorem statements + per-channel verdicts.
* `coordpy/live_manifold.py` — the W44 layer (~1100 LoC,
  dependency-free).
* `coordpy/r91_benchmark.py` — the R-91 benchmark family.
* `tests/test_live_manifold.py`, `tests/test_r91_benchmark.py` —
  51 tests, all passing.

The historical W43 + v3.43 release sections are preserved verbatim
below.

## Earlier TL;DR — W43 Product-Manifold Capsule (post-v3.43 research milestone)

The programme now has **forty** coupled research axes. W43 mints
axis 40: **product-manifold capsule (PMC) + factoradic routing +
subspace state + causal lattice**. W43 is the first post-release
research milestone after the CoordPy 0.5.20 final release; it is
**held outside the stable SDK contract** at this milestone (the
W43 module ships at `coordpy.product_manifold` and is reachable
only via explicit import). The released v3.43 line is byte-for-
byte unchanged.

The load-bearing change is a **mixed-curvature, six-channel
decomposition** of each cell's coordination state: a hyperbolic
branch channel (Poincare-disk-style, bit-perfect round-trip up to
2*dim path bits), a spherical consensus channel (unit-norm
signature over claim_kinds), a euclidean attribute channel, a
factoradic route channel (bijective Lehmer code; ceil(log2(n!))
bits per cell at zero visible-token cost), a subspace state
channel (bounded-rank QR-canonicalised basis approximating Gr(k,
d)), and a Lamport vector-clock channel with explicit dependency-
closure admissibility check. The W43 envelope binds seven
component CIDs under one ``manifest_v13_cid`` and is verified
through 18 enumerated failure modes disjoint from the W22..W42
boundary (cumulative trust boundary across W22..W43 = 214 named
failure modes).

**Headline W43 results (R-90 across 5 seeds × 8 families):**

* **R-90-CONSENSUS-CYCLE.** Trust precision: W42 baseline 0.600,
  W43 PMC 1.000, Δ = **+0.400** across 5/5 seeds. The spherical
  channel detects divergent claim-kind signatures.

* **R-90-CAUSAL-VIOLATION.** Causal-rejection rate: W42 baseline
  0.000, W43 PMC **1.000** across 5/5 seeds. The Lamport vector-
  clock channel rejects 100% of out-of-order handoffs.

* **R-90-COMPACT-STATE-TRANSFER.** Structured bits per visible-
  token overhead: W42 baseline 1536, W43 PMC **1808** (a strict
  +272-bit gain at n=8 roles, growing to +292 bits at n=12).

* **R-90-ROUTING-COMPRESSION.** Factoradic side-channel: W42
  baseline 0, W43 PMC **16 bits** at n=8 (= ceil(log2(8!))) at
  zero visible-token cost.

* **R-90-SUBSPACE-DRIFT.** Trust precision: W42 baseline 0.600,
  W43 PMC 1.000, Δ = **+0.400** across 5/5 seeds. The bounded-
  rank subspace channel rejects principal-angle drift above the
  registered tolerance.

* **R-90-LONG-BRANCH.** Hyperbolic round-trip: W43 PMC achieves
  bit-perfect path recovery for branch trees of depth 12 across
  5/5 seeds.

* **R-90-LINEAR-FLOW-FALSIFIER.** No-false-abstain rate: 1.000
  across 5/5 seeds. The W43 layer does NOT over-claim on the
  no-geometry-needed regime.

* **R-90-TRIVIAL-PMC.** Passthrough sanity: all three arms (W42
  passthrough, W42 active, W43 PMC active) achieve 1.000 across
  5/5 seeds. The trivially-configured W43 reduces to W42 byte-
  for-byte (the W43-L-TRIVIAL-PASSTHROUGH falsifier).

**Stable SDK contract preserved.** The CoordPy 0.5.20 stable
smoke driver (`tests/test_smoke_full.py`) reports "ALL CHECKS
PASSED" with the W43 module on disk; the W43 surface is **not**
exported under `coordpy.__experimental__` at this milestone, so
the released wheel's public surface is byte-for-byte unchanged.

**Honest scope.** W43 introduces three NEW proved-conditional
limitation theorems at the capsule layer (`W43-L-DUAL-CHANNEL-
COLLUSION-CAP`, `W43-L-FORGED-CAUSAL-CLOCK-CAP`, `W43-L-
OBLIVIOUS-FACTORADIC-CAP`) and three open conjectures
(`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) that require new substrate
beyond the capsule layer. The W43 milestone explicitly does NOT
claim transformer-internal mixed-curvature attention, KV-cache
pooling, or continuous Grassmannian homotopy — each channel is a
mathematically motivated bounded approximation of one direction
in the architecture vision.

See:

* `docs/SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md` — pre-committed
  H1..H10 success bar.
* `docs/RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md` — full results
  note + architecture triage + theorem statements.
* `coordpy/product_manifold.py` — the W43 layer (~1500 LoC,
  dependency-free).
* `coordpy/r90_benchmark.py` — the R-90 benchmark family.
* `tests/test_product_manifold.py`, `tests/test_r90_benchmark.py`
  — 58 tests, all passing.

The historical v3.43 release section is preserved verbatim
below.

> New reader note. If you are here to install or use CoordPy, start
> with [`README.md`](../README.md) and
> [`docs/START_HERE.md`](START_HERE.md). This file is the canonical
> statement of what is true now plus preserved historical milestone
> summaries below.

## TL;DR — SDK v3.43 (final release)

The programme now has **thirty-nine** coupled research axes.  SDK
v3.43 mints axis 39: **cross-role-invariant synthesis +
manifest-v12 CID + role-handoff-signature axis +
composite-collusion bounding** (W42).  W42 is explicitly framed as
a **third-axis bounding** milestone *and* a **release-closure**
milestone -- "using everything already built in this repo, plus
any honest infra/model improvements, can this repo now actually
earn release?"  W42 answers that by adding a third orthogonal
evidence axis on top of W41's producer-axis x trust-axis
integrated synthesis, mechanically raising the adversary bar from
"compromise the W21 producer-side admission AND the W40 trust-
side ratification" (the W41 adversary) to "compromise the W21
producer-side admission AND the W40 trust-side ratification AND
poison the controller-side role-invariance policy registry that
maps role-handoff signatures to expected service sets" (the W42
adversary).

The load-bearing change is the **first measured strict
trust-precision recovery on a regime where W41 tied at 0.500**.
On R-89-ROLE-INVARIANT-RECOVER, W42 improves trust precision from
0.500 to **1.000 across 5/5 seeds**
(``Δ_trust_precision_w42_w41 = +0.500``, min = max).  When the
adversary also poisons the policy registry, W42 cannot recover;
this is the new proved-conditional
``W42-L-FULL-COMPOSITE-COLLUSION-CAP`` limitation theorem (the
W42 analog of all previous capsule-layer limitation theorems,
sharper in adversary cost).

**Headline SDK v3.43 final-release results.**

* **R-89-TRIVIAL-W42.**  With invariance disabled, manifest-v12
  disabled, and abstain-on-invariance-diverged disabled, W42
  reduces to W41 byte-for-byte across 5/5 seeds;
  ``w42_w41_byte_equivalent = True``.  Aggregate branch
  distribution: 80 cells (5 seeds × 16 cells/seed) on
  ``trivial_invariance_passthrough``.

* **R-89-ROLE-INVARIANT-AGREES (no-regression).**  W42 ratifies
  via ``invariance_ratified`` on every cell.  Across 5/5 seeds:
  ``trust_precision_w42 = trust_precision_w41 = 1.000``;
  ``Δ_trust_precision_w42_w41 = 0.000``.

* **R-89-ROLE-INVARIANT-RECOVER (load-bearing strict gain).**
  When W41 ratifies the wrong colluded set (R-88-COMPOSITE-
  COLLUSION) AND the controller-side policy entry registers the
  true gold for the cell's role-handoff signature, W42 abstains
  via ``invariance_diverged_abstained`` on every recovery-half
  cell.  Across 5/5 seeds: ``trust_precision_w42 = 1.000``;
  ``trust_precision_w41 = 0.500``;
  **Δ_trust_precision_w42_w41 = +0.500** (min = max).  This is
  the first capsule-native multi-agent coordination method that
  materially BOUNDS ``W41-L-COMPOSITE-COLLUSION-CAP`` at the
  capsule layer.

* **R-89-FULL-COMPOSITE-COLLUSION (W42-L-FULL-COMPOSITE-
  COLLUSION-CAP).**  When the adversary ALSO poisons the policy
  registry to register the wrong colluded set as expected
  services, W42 ratifies on the wrong set; W42 cannot recover at
  the capsule layer.  Across 5/5 seeds: ``Δ_trust_precision_w42
  _w41 = 0.000``.  Closure requires native-latent evidence
  outside the capsule layer (``W42-C-NATIVE-LATENT``).

* **R-89-INSUFFICIENT-INVARIANCE-POLICY.**  When no policy entry
  is registered, W42 routes through ``invariance_no_policy`` and
  preserves W41 byte-for-W40 on the answer; delta = 0 across
  5/5 seeds.

* **W42 cross-host paraphrase-invariance live probe (2026-05-03).**
  At temperature 0 on the two-Mac topology (localhost gemma2:9b
  + 192.168.12.191 qwen2.5:14b), K=4 paraphrases of one
  closed-vocabulary arithmetic prompt produce 4/4 paraphrase-
  invariant cross-host gold-correlated agreement on both hosts;
  cross-host normalised agreement = 1.000; within-host
  paraphrase-invariance count = 1 distinct answer per host.  The
  first measured cross-host paraphrase-invariance result in the
  programme.  Realism anchor only; not load-bearing for the W42
  closed-form mechanism.

* **Final release declared.**  H1..H12 + S3 + S7 of the W42
  success criterion pass.  The SDK v3.43 line ships as a
  **final release** of the CoordPy SDK v3.4x research line.  This
  is the **end-of-line for the capsule-layer-only research
  programme** in the Context Zero project: the strongest
  capsule-layer audited proxy this repo can support has been
  built; the remaining open frontiers
  (``W42-C-NATIVE-LATENT``, ``W42-C-MULTI-HOST``) are explicitly
  out of capsule-layer scope and require new architectural
  substrate.

* **Cumulative trust boundary**: 196 enumerated capsule-layer
  failure modes mechanically audited across W22..W42.

**Release-use summary (current truth).**

* **Install today**: CoordPy is not on PyPI yet. Install from a clone
  with `pip install -e .`, then import `vision_mvp.coordpy` or use the
  `coordpy` / `coordpy-import` / `coordpy-ci` / `coordpy-capsule`
  CLIs. `pip install coordpy` and `pipx install coordpy` are the
  intended published install paths once the package is on PyPI.
* **Simple agent path**: use `Agent`, `AgentTeam`, `agent`, and
  `create_team` from `vision_mvp.coordpy` for the smallest stable
  “define a few agents and run them” surface. For real providers, use
  `backend_from_env()` or `AgentTeam.from_env(...)` with
  `COORDPY_BACKEND`, `COORDPY_MODEL`, and either
  `COORDPY_OLLAMA_URL` or `COORDPY_API_KEY` /
  `COORDPY_API_BASE_URL`.
* **Stable and released**: the `RunSpec -> run -> RunReport` runtime,
  the simple `AgentTeam` API, the public CLIs, capsule primitives, the
  provider-backed `OpenAICompatibleBackend` path, and the on-disk
  schema contracts named in the README.
* **Experimental but included**: every W22..W42 symbol under
  `vision_mvp.coordpy.__experimental__`, the R-69..R-89 benchmark
  drivers, and the bounded live cross-host probes.
* **Out of scope for this release**: `W42-C-NATIVE-LATENT` and
  `W42-C-MULTI-HOST`.

Historical note: older sections below are preserved as milestone
records.  Where they mention `.101` as a third-host Mac candidate,
that reading is superseded by the W41 retraction: `.101` is an Apple
TV / AirPlay receiver, not a Mac.

## Historical milestone summaries begin below

Everything after this heading is preserved for research continuity and
auditability. The current product/release truth is the v3.43 section
above.

## Earlier TL;DR — SDK v3.42 RC2

The programme now has **thirty-eight** coupled research axes.  SDK
v3.42 RC2 mints axis 38: **integrated multi-agent context
synthesis + manifest-v11 CID + cross-axis witness CID +
producer-axis x trust-axis decision selector** (W41).  W41 is
explicitly framed as a **synthesis** milestone -- "using
everything already built in this repo, what is the strongest
honest full-system attempt we can make to solve context for
multi-agent teams?" -- not "W41: one more local mechanism."  W41
jointly binds the strongest old-line explicit-capsule trust-
adjudication chain (W21..W40) AND the strongest cross-role /
multi-round bundle decoder family (W7..W11) into a single
auditable end-to-end path with one ``manifest-v11`` envelope.

The load-bearing change is from *trust-only adjudication*
(W21..W40) to *integrated cross-axis classification* (producer-
side W7..W11 x trust-side W21..W40) under one manifest-v11
envelope.  W41 raises the capsule-layer adversary bar from
"compromise the W22..W40 trust-adjudication chain" to "compromise
the W22..W40 trust-adjudication chain AND coordinate the W7..W11
producer-side admission so the cross-axis classification cannot
use one axis to overrule the other."  When both are coordinated,
W41 cannot recover; this is the new proved-conditional
``W41-L-COMPOSITE-COLLUSION-CAP`` limitation theorem (the W41
analog of all previous capsule-layer limitation theorems).

**Headline SDK v3.42 RC2 results.**

* **R-88-TRIVIAL-W41.**  With synthesis disabled, manifest-v11
  disabled, and abstain-on-axes-diverged disabled, W41 reduces
  to W40 byte-for-byte across 5/5 seeds; ``all_byte_equivalent
  _w41_w40 = True``.  Aggregate branch distribution: 80 cells
  (5 seeds × 16 cells/seed) on ``trivial_integrated_passthrough``.

* **R-88-BOTH-AXES (no-regression).**  W41 ratifies via
  ``integrated_both_axes`` on the recovery half + via
  ``integrated_producer_only`` on the prefix half.  Across 5/5
  seeds: ``correctness_w41 = correctness_w40 = 1.000``;
  ``trust_precision_w41 = trust_precision_w40 = 1.000``;
  ``Δ_correctness_w41_w40 = 0.000``;
  ``Δ_trust_precision_w41_w40 = 0.000``.

* **R-88-TRUST-ONLY-SAFETY (load-bearing safety).**  When the
  W21 producer set is colluded AND W40 detects collapse and
  abstains, W41 routes through the ``integrated_trust_only``
  safety branch with empty integrated services on the recovery
  half.  Across 5/5 seeds: ``trust_precision_w41 =
  trust_precision_w40 = 1.000`` (W41 preserves the W40 safety
  abstention exactly).

* **R-88-COMPOSITE-COLLUSION (W41-L-COMPOSITE-COLLUSION-CAP).**
  When both the producer-side admission AND the trust-side W40
  ratification are coordinated by the adversary, W41 cannot
  recover; ``Δ_trust_precision_w41_w40 = 0.000`` across 5/5
  seeds; W41 ratifies ``integrated_both_axes`` on the wrong
  set.

* **R-88-INSUFFICIENT-RESPONSE-SIGNATURE.**  When W40 returns
  insufficient probes, W41 routes through
  ``integrated_producer_only`` and preserves W40 byte-for-W39
  on the answer; delta = 0 across 5/5 seeds.

* **W41-INFRA-1.**  ``192.168.12.101`` is identified as an
  Apple TV / AirPlay receiver (``HTTP/1.1 403 Forbidden`` with
  ``Server: AirTunes/860.7.1`` on port 5000; locally-administered
  MAC ``36:1c:eb:dc:9a:04``), NOT a Mac running Ollama.  The
  W37..W40 "TCP-up + HTTP-broken Ollama Mac" framing for
  ``.101`` is **retracted** at the W41 milestone.  The honest
  live multi-host topology is the two-Mac pair (``localhost`` +
  ``192.168.12.191``).  ``192.168.12.248`` is recorded as gone
  (per user instruction).

* **RC2 declared.**  H1..H12 + S3 of the W41 success criterion
  pass.  The SDK v3.42 line is the second release-candidate of
  the CoordPy SDK v3.4x line, strictly additive on top of RC1
  (every W22..W41 symbol is exported under
  ``__experimental__``; the stable RunSpec -> run report
  runtime contract is byte-for-byte unchanged).

* **Cross-axis branch distribution (R-88).**  R-88 is the first
  end-to-end multi-agent context benchmark family that records
  the cross-axis branch distribution per cell.  At
  ``n_eval = 16`` × 5 seeds = 80 cells per bank:
  - ``trivial_w41``: 80 trivial_integrated_passthrough.
  - ``both_axes``: 40 integrated_producer_only + 40 integrated_both_axes.
  - ``trust_only_safety``: 40 integrated_producer_only + 40 integrated_trust_only.
  - ``composite_collusion``: 40 integrated_producer_only + 40 integrated_both_axes (on wrong set).
  - ``insufficient_response_signature``: 80 integrated_producer_only.

## Earlier TL;DR — SDK v3.41 RC1

The programme now has **thirty-seven** coupled research axes.  SDK
v3.41 RC1 mints axis 37: **cross-host response-signature
heterogeneity ratification + manifest-v10 CID + cross-host
response-text Jaccard divergence guard** (W40).  W40 wraps W39's
K-of-N mutually-disjoint quorum consensus-reference adjudication
with a **cross-host response-heterogeneity** layer that operates
on an evidence axis ORTHOGONAL to top_set: the per-member response
**text bytes** themselves.  Even if K colluders coordinate their
declared top_set (the W39 full-quorum-collusion attack;
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` fires), naturally-
independent K probes should produce heterogeneous response text
bytes.  When the K member probes' mean pairwise Jaccard divergence
over canonical sorted token bags falls strictly below
``response_text_diversity_min``, W40 abstains via
``RESPONSE_SIGNATURE_COLLAPSE_ABSTAINED`` even when W39 would have
RATIFIED.

The load-bearing change is from *top-set-only quorum agreement*
(W39) to *top-set agreement AND cross-host response-text Jaccard
heterogeneity* (W40).  W39 raises the adversary bar to "compromise
2 of N trajectory hosts AND ``quorum_min`` of the K mutually-
disjoint registered consensus references".  W40 raises it to
"compromise 2 of N trajectory hosts AND ``quorum_min`` of the K
mutually-disjoint registered consensus references AND inject K
diverse response text bytes that all encode the same wrong
top_set".  When the adversary diversifies response bytes while
holding the wrong top_set in lock-step, W40 cannot recover; this
is the new proved-conditional
``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` limitation theorem.

**Headline SDK v3.41 RC1 results.**

* **R-87-RESPONSE-SIGNATURE-COLLAPSE (load-bearing).**  W40
  improves over W39 from 0.500 to **1.000** trust precision
  across 5/5 seeds, **Δ_trust_precision_w40_w39 = +0.500** (min
  and max equal).  W40 ratifies 0 cells via
  RESPONSE_SIGNATURE_DIVERSE in the recovery half, abstains via
  RESPONSE_SIGNATURE_COLLAPSE on 8 cells/seed; overhead = 1
  visible token/cell.

* **R-87-TRIVIAL-W40.**  With response-heterogeneity disabled,
  collapse-abstain disabled, and manifest-v10 disabled, W40
  reduces to W39 byte-for-byte across 5/5 seeds.

* **R-87-NO-REGRESSION-DIVERSE-AGREES.**  W40 ratifies via
  RESPONSE_SIGNATURE_DIVERSE; delta = 0 across 5/5 seeds.

* **R-87-COORDINATED-DIVERSE-RESPONSE.**  The smart attacker who
  diversifies response bytes while holding the wrong top_set
  defeats W40; ``W40-L-COORDINATED-DIVERSE-RESPONSE-CAP`` fires;
  delta = 0 across 5/5 seeds.

* **R-87-INSUFFICIENT-RESPONSE-SIGNATURE.**  When fewer than
  ``min_response_signature_probes`` member probes carry response
  signatures, W40 reduces to W39 byte-for-byte; delta = 0.

* **W40-INFRA-1.**  ``192.168.12.101`` re-probed (2026-05-03):
  ping 0% packet loss, TCP 11434 + 22 connect; Ollama HTTP
  listener still returns "Empty reply / Connection reset"; SSH
  credentials unavailable.  Honest verdict: TCP-up + HTTP-broken;
  a strict topology-layer improvement over W39's 100% packet
  loss end-state.

* **RC1 declared.**  H1..H12 + S3 of the W40 success criterion
  pass.  The SDK v3.41 line is the first official release-
  candidate of the CoordPy SDK v3.4x line.

## Earlier TL;DR — SDK v3.40

The programme had **thirty-six** coupled research axes.  SDK
v3.40 minted axis 36: **multi-host disjoint quorum consensus-
reference ratification + manifest-v9 CID + mutually-disjoint
physical-host topology** (W39).  W39 wraps W38's disjoint
cross-source consensus-reference adjudication with a *quorum* of
K disjoint probes, each sourced from a physically-distinct host
pool that is both mechanically disjoint from the W37 trajectory
hosts (W38's precondition) AND mutually disjoint from every other
registered quorum probe's host pool (the new W39 precondition,
mechanically enforced via :class:`MutuallyDisjointTopologyError`).
W39 abstains via the ``QUORUM_DIVERGENCE_ABSTAINED`` branch as
soon as ``quorum_min`` of the K member probes diverge from the
W37/W38 candidate top_set.

The load-bearing change is from a *single* disjoint consensus
reference (W38) to a *K-of-N mutually-disjoint quorum* of
disjoint consensus references (W39).  W38 raises the adversary
bar to "compromise 2 of N trajectory hosts AND the single
disjoint registered consensus reference".  W39 raises it to
"compromise 2 of N trajectory hosts AND ``quorum_min`` of the K
mutually-disjoint registered consensus references, each on a
physically distinct host pool".  When all K disjoint probes are
themselves compromised in lock-step, W39 cannot recover; this
is the new proved-conditional
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` limitation theorem.

**Headline SDK v3.40 results.**

* **R-86-MULTI-HOST-COLLUDED-CONSENSUS (load-bearing).**  W39
  improves over W38 from 0.500 to **1.000** trust precision
  across 5/5 seeds, **Δ_trust_precision_w39_w38 = +0.500** (min
  and max equal).  W39 reroutes 0 cells, abstains via
  QUORUM_DIVERGENCE on 8 cells/seed; overhead = 1 visible
  token/cell.

* **R-86-TRIVIAL-W39.**  With quorum disabled,
  divergence-abstain disabled, and manifest-v9 disabled, W39
  reduces to W38 byte-for-byte across 5/5 seeds;
  ``all_byte_equivalent_w39_w38 = True``.

* **R-86-NO-REGRESSION-QUORUM-AGREES.**  Quorum members all
  agree with the W37/W38 reroute; W39 ratifies via
  QUORUM_RATIFIED on 8 cells/seed; no correctness or
  trust-precision regression vs W38.

* **R-86-FULL-QUORUM-COLLUSION.**  Hard falsifier: when every
  registered disjoint quorum probe is compromised in lock-step
  with the colluding trajectory hosts, W39 cannot recover.
  Δ_trust_precision = 0.000;
  ``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` fires.

* **R-86-INSUFFICIENT-QUORUM.**  Falsifier: when fewer than
  ``min_quorum_probes`` member probes are registered, W39
  returns ``QUORUM_INSUFFICIENT``; correctness / trust precision
  unchanged from W38; the W39 audit envelope still records the
  QUORUM_INSUFFICIENT branch.

**Density / efficiency**: On R-86-MULTI-HOST-COLLUDED-CONSENSUS,
W39 carries about **24,400 structured bits per visible W39
token** at one visible token overhead/cell -- the densest
milestone-level capsule-layer audited proxy the programme has
measured (W38 carried 9,072 bits/token on R-85; W39 is ~2.7x
denser at the audited-proxy capsule-layer).  This is
controller-verified structured multi-host disjoint quorum state,
not hidden-state transfer.

**Trust boundary**: 14 new W39 failure modes in
``verify_multi_host_disjoint_quorum_ratification``, mechanically
tested.  Cumulative enumerated trust boundary across W22 + W29 +
W30 + W31 + W32 + W33 + W34 + W35 + W36 + W37 + W38 + W39 =
**154 enumerated failure modes**.

**Lab topology resolution / live multi-Mac status**: The
historical Mac-2 endpoint (``192.168.12.248``) has been
ARP-incomplete for the **31st milestone in a row**.  This
milestone identified ``192.168.12.101`` as the reachable third
physical host candidate, **partially discharging
``W38-C-MULTI-HOST`` at the topology layer**: ``.101`` was
preflight-OK on cold contact, ``ping`` reported ~3.5 ms RTT,
``/api/tags`` returned ``qwen3.5:35b`` (36.0B MoE Q4_K_M) and
``qwen2.5:14b-32k`` (14.8B Q4_K_M) model files visible.  The
``.101`` Ollama inference path subsequently degraded under the
capsule-layer one-word probe budget (``W39-INFRA-1``: every
``/api/chat``, ``/api/generate``, ``/api/show``, ``/api/ps``
endpoint timed out at 30s+; eventually even ``/api/tags`` and
``ping`` went silent; no recovery in 5-minute polling; no SSH
credentials available to restart the service).  The W39 live
xllm probe was made robust via a fallback path: when ``.101`` is
unreachable, ``mac_off_cluster_a`` swaps to ``localhost`` running
``llama3.1:8b`` (a model class genuinely different from the
trajectory's ``gemma2:9b``), so the live K=2 quorum becomes
``(localhost llama3.1:8b, .191 qwen2.5-coder:14b-32k)`` -- two
**physically distinct hosts**, each running a different model
class from the trajectory pair AND from the W38 single consensus
reference.  Bounded W39 5-host live xllm probe at temperature 0
+ ``num_predict=4`` observed **8/8 responsive on all 5 hosts**
(first 5-host live W39 disjoint-quorum probe in the programme),
7/8 trajectory_pair_agrees, 7/8 consensus_agrees, **8/8 quorum_a
gold-correlated, 8/8 quorum_b gold-correlated, 8/8 K=2 quorum
size simultaneously responsive**.  Notable live finding: on the
``h2o`` probe, the trajectory pair disagreed (``mac1=h2o`` vs
``mac_remote=h`` due to ``num_predict=4`` truncation), but BOTH
quorum members got ``h2o`` correct -- empirical-suggestive
evidence for the new ``W39-C-LIVE-TRUNCATION-RECOVERY``
conjecture.  Result recorded in
``vision_mvp/experiments/artifacts/phase86/xllm_quorum_probe_2026_05_02.json``
and
``vision_mvp/experiments/artifacts/phase86/host_topology_resolution_2026_05_02.json``.

**Stable-vs-experimental boundary**: W39 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `coordpy.sdk.v3.40`; package version `0.5.13`.  Both
``vision_mvp.__version__`` and ``pyproject.toml`` are now
``0.5.13`` (alignment maintained).

**Open walls after W39**:
``W39-L-FULL-DISJOINT-QUORUM-COLLUSION-CAP`` is a new
proved-conditional limitation theorem at the capsule layer
(when all K disjoint quorum probes are also compromised in
lock-step, recovery requires native-latent evidence outside the
capsule layer or a K+1-host disjoint topology with a new
genuinely uncompromised pool).  ``W39-C-NATIVE-LATENT`` remains
open.  ``W39-C-MULTI-HOST`` is partially discharged at the
topology layer (``.101`` reachable on cold contact) and still
open at the live-inference layer (``W39-INFRA-1``).
``W39-C-LIVE-TRUNCATION-RECOVERY`` is a new conjecture observed
suggestively on a single live probe.  Multi-host disjoint quorum
audit is now sealed in manifest-v9 (W38 manifest-v8 +
quorum-state CID + quorum-decision CID + quorum-topology CID +
mutual-disjointness CID); the cross-source × cross-cell × K-of-N
mutually-disjoint × per-cell audit boundary is now structurally
explicit in the envelope hierarchy.

See `docs/RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md` for
the full note and
`docs/SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md` for
the success bar.

---

## Earlier TL;DR — SDK v3.39

The programme now has **thirty-five** coupled research axes.  SDK
v3.39 mints axis 35: **disjoint cross-source consensus-reference
trajectory-divergence adjudication + manifest-v8 CID** (W38).  W38
wraps W37's anchor-cross-host basis-trajectory ratification with a
controller-pre-registered ``ConsensusReferenceProbe`` whose host
topology is *mechanically disjoint* from W37's trajectory hosts (the
``DisjointConsensusReferenceRegistry`` raises
``DisjointTopologyError`` otherwise; the verifier additionally
rejects envelopes claiming an overlapping topology).  When W37 chooses
to reroute on a trajectory-anchored top_set and the disjoint consensus
reference disagrees within ``divergence_margin_min`` (Jaccard), W38
abstains via the ``CONSENSUS_DIVERGENCE_ABSTAINED`` branch.

The load-bearing change is *cross-source* rather than cross-cell or
cross-host.  W37 raises the adversary bar to "compromise 2 of N
trajectory hosts coordinately" but cannot break that bar at the
capsule layer (``W37-L-MULTI-HOST-COLLUSION-CAP``).  W38 raises it
to "compromise 2 of N trajectory hosts AND the disjoint registered
consensus reference".  When the disjoint reference is itself
compromised in lock-step, W38 cannot recover; this is the new
proved-conditional ``W38-L-CONSENSUS-COLLUSION-CAP`` limitation
theorem.

**Headline SDK v3.39 results.**

* **R-85-COLLUDED-CROSS-HOST-TRAJECTORY (load-bearing).**  W38
  improves over W37 from 0.500 to **1.000** trust precision across
  5/5 seeds, **Δ_trust_precision_w38_w37 = +0.500** (min and max
  equal).  W38 reroutes 0 cells, abstains via DIVERGENCE on 8
  cells/seed; overhead = 1 visible token/cell.

* **R-85-TRIVIAL-W38.**  With consensus disabled, divergence-abstain
  disabled, and manifest-v8 disabled, W38 reduces to W37
  byte-for-byte across 5/5 seeds;
  ``all_byte_equivalent_w38_w37 = True``.

* **R-85-NO-COLLUSION-CONSENSUS-AGREES.**  Consensus reference
  agrees with the W37 reroute; W38 ratifies; no correctness or
  trust-precision regression vs W37.

* **R-85-CONSENSUS-ALSO-COMPROMISED.**  Hard falsifier: when the
  disjoint consensus reference is itself compromised in lock-step
  with the colluding trajectory hosts, W38 cannot recover.  Δ_trust_
  precision = 0.000; W38-L-CONSENSUS-COLLUSION-CAP fires.

* **R-85-NO-CONSENSUS-REFERENCE.**  Falsifier: when no consensus
  probe is registered, W38 returns ``CONSENSUS_NO_REFERENCE``;
  correctness/trust-precision unchanged from W37; the W38 audit
  envelope still records the NO_REFERENCE branch.

**Density / efficiency**: On R-85-COLLUDED-CROSS-HOST-TRAJECTORY,
W38 carries about **9,072 structured bits per visible W38 token**
at one visible token overhead/cell.  This is controller-verified
structured cross-source consensus-reference state, not hidden-state
transfer.

**Trust boundary**: 14 new W38 failure modes in
``verify_disjoint_consensus_reference_ratification``, mechanically
tested.  Cumulative enumerated trust boundary across W22 + W29 +
W30 + W31 + W32 + W33 + W34 + W35 + W36 + W37 + W38 =
**140 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
(Mac 2 ARP-incomplete for the **31st milestone in a row**; ``ping``
reports "Host is down").  The bounded W38 cross-source consensus
probe used local ``gemma2:9b`` and remote ``qwen2.5:14b`` as
trajectory hosts and remote ``qwen3.5:35b`` as the disjoint
consensus host (different model class on the same physical host;
defensible weak proxy for capsule-layer disjointness, not a true
3-host disjoint topology).  Result recorded in
``vision_mvp/experiments/artifacts/phase85/xllm_consensus_probe_2026_05_02.json``.

**Stable-vs-experimental boundary**: W38 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `coordpy.sdk.v3.39`; package version `0.5.12`.  The
lingering ``vision_mvp.__version__ == "0.5.9"`` vs
``pyproject.toml == "0.5.11"`` misalignment from earlier milestones
is closed: both are now ``0.5.12``.

**Open walls after W38**: W38-L-CONSENSUS-COLLUSION-CAP is a new
proved-conditional limitation theorem at the capsule layer (when the
disjoint consensus reference is also compromised, recovery requires
native-latent evidence outside the capsule layer or a 3+-host
disjoint consensus topology).  W38-C-NATIVE-LATENT remains open.
W38-C-MULTI-HOST remains hardware-bounded until Mac 2 (or a third
reachable host) joins the topology.  Cross-source consensus audit is
now sealed in manifest-v8; cross-cell trajectory audit remains in
manifest-v7; per-cell host-diversity audit remains in manifest-v6;
per-cell live-aware multi-anchor audit remains in manifest-v4; the
cross-source × cross-cell × per-cell audit boundary is now
structurally explicit in the envelope hierarchy.

See `docs/RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md` for the
full note and
`docs/SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md` for the
success bar.

---

## Earlier TL;DR — SDK v3.38

The programme now has **thirty-four** coupled research axes.  SDK
v3.38 mints axis 34: **anchor-cross-host basis-trajectory ratification
+ manifest-v7 CID** (W37).  W37 wraps W36's host-diverse trust-subspace
guard with a closed-form, zero-parameter, per-(host, oracle, top_set)
EWMA over *anchored* historical observations.  W36 abstains whenever
the current cell has fewer than `min_distinct_hosts` healthy attested
hosts -- even when the remaining single host has been independently
anchored across multiple earlier cells by other healthy hosts.  W37
makes that historical cross-host anchoring a typed audited precondition
for a safe single-host reroute.

The load-bearing change is cross-cell rather than per-cell.  W36 reads
one cell at a time; W37 maintains a sealed trajectory of which (host,
oracle, top_set) entries have been *cross-host anchored* and converts
historically-anchored evidence into a safe reroute on a single-host
recovery cell.  The trajectory update is gated by *select-before-update*
so the historical anchoring is not diluted by the recovery cell itself,
and is gated by *multi-host-only update* so a host is not penalised
for being the only reachable host when the live infrastructure offers
no co-attesters.

**Headline SDK v3.38 results.**

* **R-84-SINGLE-HOST-TRAJECTORY-RECOVER.**  W37 improves over W36 from
  0.500 to **1.000** correctness across 5/5 seeds, **Δ_correctness_
  w37_w36 = +0.500** (min and max equal), and trust precision stays at
  **1.000**.  W37 reroutes 8 cells per seed; overhead = 1 visible
  token/cell.

* **R-84-TRIVIAL-W37.**  With trajectory disabled, single-host reroute
  disabled, and manifest-v7 disabled, W37 reduces to W36 byte-for-byte
  across 5/5 seeds; `all_byte_equivalent_w37_w36 = True`.

* **R-84-NO-TRAJECTORY-HISTORY.**  Falsifier: when no cross-host
  anchoring ever forms (every cell single-host), W37 = W36 = 0.000
  correctness; W37 trust precision = 1.000; W37 abstains 16/16
  cells/seed.

* **R-84-POISONED-TRAJECTORY.**  Falsifier: a trajectory accumulated on
  a single host without cross-host anchoring never satisfies the
  `min_trajectory_anchored_hosts` requirement; W37 does not reroute and
  preserves W36 abstention.

* **R-84-TRAJECTORY-DISAGREEMENT.**  Falsifier: when the current
  cell's basis emits a top_set that does not match the historically
  anchored trajectory, W37 must not reroute on the historically
  anchored top_set; W37 = W36 = 0.500 correctness.

**Density / efficiency**: On R-84-SINGLE-HOST-TRAJECTORY-RECOVER, W37
carries about **29.5k structured bits per visible W37 token** at one
visible token overhead/cell.  This is controller-verified structured
trajectory state, not hidden-state transfer.

**Trust boundary**: 14 new W37 failure modes in
`verify_cross_host_trajectory_ratification`, mechanically tested.
Cumulative enumerated trust boundary across W22 + W29 + W30 + W31 +
W32 + W33 + W34 + W35 + W36 + W37 = **126 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
(Mac 2 ARP-incomplete for the **30th milestone in a row**).  The
bounded W37 cross-host trajectory probe on 2026-05-02 across local
`gemma2:9b` and remote `qwen2.5:14b` produced **8/8 responsive
probes, 8/8 cross-host anchored agreements, and 8/8 gold-correlated
agreements** at temperature 0 on the 8-prompt panel.  This is the
strongest two-reachable-host live trajectory evidence the
infrastructure currently supports; it does not close W37-C-MULTI-HOST.

**Stable-vs-experimental boundary**: W37 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `coordpy.sdk.v3.38`; package version `0.5.11`.

**Open walls after W37**: W37-L-MULTI-HOST-COLLUSION-CAP is a new
proved-conditional limitation theorem at the capsule layer (two
colluding hosts can cross the anchored thresholds; recovery requires
native-latent evidence outside the capsule layer).  W37-C-NATIVE-LATENT
remains open.  W37-C-MULTI-HOST remains hardware-bounded until Mac 2
(or a third reachable host) joins the topology.  Cross-cell trajectory
audit is now sealed in manifest-v7; per-cell host-diversity audit
remains in manifest-v6; per-cell live-aware multi-anchor audit remains
in manifest-v4; the cross-cell × per-cell audit boundary is now
structurally explicit in the envelope hierarchy.

See `docs/RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md` for the
full note and
`docs/SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md` for the
success bar.

---

## Earlier TL;DR — SDK v3.37

The programme had **thirty-three** coupled research axes.  SDK
v3.37 minted axis 33: **host-diverse trust-subspace guard +
manifest-v6 CID** (W36).  W36 wraps W35's audited trust-subspace
dense-control proxy with a controller-side host-diversity verifier:
dense projection support must come from at least two distinct
registered healthy hosts, and unsafe or unverifiable branches reject
or abstain.

The load-bearing change is narrower than native latent transfer and
more operational than another local reroute.  W35 could ratify a
dense basis projection even when the available capsule-visible support
was effectively co-located or host-unsafe.  W36 makes host diversity a
typed audited precondition.

**Headline SDK v3.37 results.**

* **R-83-HOST-DIVERSE-RECOVER.**  W36 improves over W35 from 0.625
  to **0.9375** correctness across 5/5 seeds, **Δ_correctness_w36_w35
  = +0.3125**, and restores trust precision from 0.6667 to **1.000**.
  W36 reroutes 5 cells and abstains on 1 unsafe W35 ratification per
  seed.  W21 remains at 1.000 on this synthetic regime, so the claim
  is trust-stack hardening, not universal dominance over every older
  explicit-capsule baseline.

* **R-83-HOST-SPOOFED-CONSENSUS.**  W36 does not recover correctness:
  W35 and W36 both stay at 0.625.  It improves trust precision from
  0.625 to **1.000** by abstaining on 6 unsafe W35 ratifications per
  seed.  This is the named spoofed-host falsifier.

* **R-83-TRIVIAL-W36.**  With host diversity disabled and manifest-v6
  disabled, W36 reduces to W35 byte-for-byte across 5/5 seeds.

* **R-83-NO-LIVE-ATTESTATION.**  Falsifier: if host diversity is
  required but live attestations are absent, W36 abstains on every
  cell and drops correctness from W35's 1.000 to 0.000 while keeping
  trust precision at 1.000.

**Density / efficiency**: On R-83-HOST-DIVERSE-RECOVER, W36 carries
about **13.95k structured bits per visible W36 token** at one visible
token overhead/cell.  On R-83-HOST-SPOOFED-CONSENSUS it carries about
**13.74k bits/token**.  This is controller-verified structured state,
not hidden-state transfer.

**Trust boundary**: 14 new W36 failure modes in
`verify_host_diverse_ratification`, mechanically tested.  Cumulative
enumerated trust boundary across W22 + W29 + W30 + W31 + W32 + W33 +
W34 + W35 + W36 = **112 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`.
The bounded W36 two-reachable-host probe on 2026-05-02 across local
`qwen2.5:0.5b` and remote `qwen2.5:14b` yielded 10/10 responsive
probes, 4/5 cross-host disagreements, and 4/4 gold-correlated
disagreements.  This materially strengthens the two-reachable-host
motivation for host-diverse control but still does **not** close true
three-host evidence.

**Stable-vs-experimental boundary**: W36 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `coordpy.sdk.v3.37`; package version `0.5.10`.

**Open walls after W36**: W33-C-NATIVE-LATENT remains open.  W36 is
not transformer-internal hidden-state projection, not a KV transplant,
and not a learned latent controller.  W33-C-CROSS-HOST-LIVE-TRUST-
MAGNITUDE remains open on the systematic magnitude axis.  W34/W35/
W36-C-MULTI-HOST remains hardware-bounded until Mac 2 joins the
topology.  W36 also adds a new operational wall: host-diverse dense
control is unsafe without real live attestations.

See `docs/RESULTS_COORDPY_W36_HOST_DIVERSE_TRUST_SUBSPACE.md` for the
full note and `docs/SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`
for the success bar.

---

## Earlier TL;DR — SDK v3.36

The programme now has **thirty-two** coupled research axes.  SDK
v3.36 mints axis 32: **trust-subspace dense-control proxy +
basis-history projection + manifest-v5 CID** (W35).  W35 wraps W34's
live-aware multi-anchor path with the strongest honest native-latent
proxy this repo can support without transformer-runtime access: a
controller-verified dense basis over W21 probe top_sets, W33 EWMA
trust, W34 live-attestation/response-feature state, top-set stability,
and host health.

The load-bearing change is narrow and real.  W34 treated anchor
disagreement as a trust signal and abstained.  W35 asks whether the
verified basis history contains a stable, high-margin trusted
direction that can safely convert that abstention into a reroute.  If
the basis is short, unstable, insufficiently separated, or
unverifiable, W35 preserves W34's abstention.

**Headline SDK v3.36 results.**

* **R-82-TRUST-SUBSPACE-SHIFT.**  W34 abstains on 6 disputed cells;
  W35 reroutes 5/6 through the stable `change_history` basis
  direction.  Correctness rises from 0.625 (W21/W33/W34) to
  **0.9375** across 5/5 seeds, **Δ_correctness_w35_w34 = +0.3125**,
  while trust precision stays at **1.000** and overhead is one visible
  token/cell.

* **R-82-TRIVIAL-W35.**  With trust-subspace disabled and manifest-v5
  disabled, W35 reduces to W34 byte-for-byte across 5/5 seeds.

* **R-82-NO-ANCHOR-DISAGREEMENT.**  When W34 already has consensus,
  W35 adds no correctness lift; W35 = W34 = 1.000 correctness/trust
  precision.

* **R-82-FROZEN-BASIS.**  Weakened basis history attenuates but does
  not remove the gain: W35 correctness 0.875, delta +0.2500, four
  reroutes.

* **R-82-ALL-ANCHOR-COMPROMISED.**  Hard falsifier: when every basis
  direction moves together to the wrong answer, W35 cannot recover.
  W35-W34 delta = 0.000; trust precision remains 0.625.

**Density / efficiency**: On R-82-TRUST-SUBSPACE-SHIFT, W35 carries
mean **208,264 structured bits per 16-cell seed** at one visible
token overhead/cell, or **13,016.5 structured bits per visible W35
token**.  This is dense controller-verified state transfer, not hidden
state transfer.

**Trust boundary**: 14 new W35 failure modes in
`verify_trust_subspace_dense_ratification`, now mechanically tested.
Cumulative enumerated trust boundary across W22 + W29 + W30 + W31 +
W32 + W33 + W34 + W35 = **98 enumerated failure modes**.

**Live / two-Mac status**: Local Ollama and `192.168.12.191:11434`
are reachable.  `192.168.12.248:11434` still times out on `/api/tags`
despite the user reopening a Mac.  A bounded two-host fallback probe
on 2026-05-02 across local `qwen2.5:0.5b` and remote `qwen2.5:14b`
yielded 10/10 responsive probes, 3/5 cross-host disagreements, and
3/3 gold-correlated disagreements.  This strengthens live disagreement
evidence but does **not** close the stronger live magnitude survey or
true multi-host blocker.

**Stable-vs-experimental boundary**: W35 is exported only under
`__experimental__`; stable runtime contract remains unchanged.
SDK_VERSION `coordpy.sdk.v3.36`; package version `0.5.9`.

**Open walls after W35**: W33-C-NATIVE-LATENT remains open.  W35 is
not transformer-internal hidden-state projection, not a KV transplant,
and not a learned latent controller.  W33-C-CROSS-HOST-LIVE-TRUST-
MAGNITUDE remains open on the systematic magnitude axis.  W34-C-
MULTI-HOST remains hardware-bounded until Mac 2 joins the topology.

See `docs/RESULTS_COORDPY_W35_TRUST_SUBSPACE_DENSE_CONTROL.md` for the
full note and `docs/SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`
for the success bar.

---

## Earlier TL;DR — SDK v3.35

The programme now has **thirty-one** coupled research axes, each
with a sharp status.  SDK v3.35 mints axis 31: **live-aware
multi-anchor adjudication + native-latent audited response-feature
proxy + W34 manifest-v4 CID + W33 infra-blocker closure (preflight
``/api/tags`` check + chat-template + ``num_predict=4`` for one-word
probes)** — wrapping the SDK v3.34 W33
``TrustEWMATrackedMultiOracleOrchestrator`` with a
``LiveAwareMultiAnchorOrchestrator`` (W34) that addresses W33's
single-anchor *fragility*: when the W33 anchor itself becomes
compromised, every honest non-anchor oracle's agreement signal
drops to 0 against the wrong reference and the (compromised) anchor
remains trusted.  W34 fixes this by computing a **multi-anchor
consensus reference** as the *intersection* of K registered
anchors' top_sets when at least ``anchor_quorum_min`` non-abstaining
anchors agree.  When the intersection is empty (the anchors
disagree), W34 *abstains* by dropping the W21-quorum-resolved
services from the answer — the inter-anchor disagreement is itself
a trust signal.  W34 attaches a content-addressed
:class:`LiveOracleAttestation` per cell (host_id, model_id,
**response_feature_signature** = 64-bit closed-form deterministic
hash over the LLM response's first-token-class + length-bucket +
SHA-256 prefix; the W34 *audited proxy for native-latent*),
applies a closed-form host-aware EWMA decay on unhealthy hosts,
and seals a **W34 manifest-v4 CID** over four component CIDs
(parent_w33_cid, live_attestation_cid, multi_anchor_cid,
host_topology_cid).  The new "live-aware multi-anchor /
response-feature signature / live oracle attestation /
host-aware EWMA decay / manifest-v4 CID / preflight discipline"
vocabulary is **capsule-layer audited proxy** — explicitly NOT a
learned feature-signature model in the deep-learning sense, NOT a
transformer-internal hidden-state projection, NOT a runtime KV
transplant.

**The headline SDK v3.35 results.**

* **R-81-DOUBLE-ANCHOR-COMPROMISE (H6 — load-bearing).**  Three-
  phase oracle compromise on 16 cells × 5 seeds where the W33 single-
  anchor (service_graph) itself becomes compromised in the final
  phase.  W33 single-anchor: trusts the wrong-flipped sg, detrusts
  the honest ch, commits to wrong decoy ⇒ trust precision = 0.625.
  W34 multi-anchor (sg + ch, anchor_quorum_min=2): intersection of
  anchor top_sets is empty in the double-compromise phase ⇒
  NO_CONSENSUS branch fires ⇒ W34 abstains where W33 commits
  wrong ⇒ trust precision = 1.000.  **Δ_trust_precision_w34_w33 =
  +0.375** across **5/5 seeds at trust_precision_w34 = 1.000**;
  no correctness regression; max overhead = 1 token/cell.
  **Empirically demonstrates the multi-anchor consensus mechanism
  closes the W33 single-anchor fragility** — the FIRST capsule-native
  multi-agent-coordination method to defeat anchor-itself-flips
  attacks at the capsule layer.

* **R-81-MANIFEST-V4-TAMPER (H3).**  The W34 envelope's manifest-v4
  CID + cross-component swap detection rejects five named tampers
  per ratified cell: multi_anchor_cid byte corruption, manifest_v4
  byte corruption, outer w34_cid byte corruption, anchor_quorum_min
  out of range (set > len(anchor_oracle_ids)), live_attestation_cid
  byte corruption.  **400/400 = 1.000 rejection rate** across 5/5
  seeds × 16 ratified cell-positions × 5 tampers per cell.

* **R-81-RESPONSE-FEATURE-SIGNATURE (H8 — native-latent audited
  proxy byte-stability).**  10 fixtures (empty / single-digit /
  short alpha / chain-of-thought / symbol / leading-whitespace etc.)
  × 3 runs each = 30 calls; **all 10 fixtures byte-stable across 3
  runs**.  Signature length = 16 hex chars (64 bits).  Confirms the
  W34 audited proxy is closed-form deterministic, zero parameters,
  reproducible byte-for-byte at temperature 0.

* **R-81-TRIVIAL-W34 (H2 byte-for-W33 anchor).**  When all W34
  knobs are trivial (``multi_anchor_quorum_min=1`` AND
  ``live_attestation_disabled=True`` AND
  ``manifest_v4_disabled=True`` AND ``host_decay_factor=1.0``),
  W34 reduces to W33 byte-for-byte across 5/5 seeds; every cell
  yields ``W34_BRANCH_TRIVIAL_MULTI_ANCHOR_PASSTHROUGH``.

* **R-81-NO-ANCHOR-DISAGREEMENT (W34-Λ-no-anchor-disagreement).**
  All-honest regime ⇒ multi-anchor consensus is the same as
  single-anchor ⇒ W34 ties W33 on correctness; Δ = 0.000.

* **R-81-FROZEN-HOST-DECAY (W34-Λ-frozen-host-decay).**
  ``host_decay_factor = 1.0`` ⇒ host-aware decay never fires ⇒
  W34 ties W33 byte-for-byte (Δ = 0.000).

* **W33-INFRA-1 + W33-INFRA-2 closure**.  The W34 milestone records
  an **honest empirical correction** of the W33 infra diagnosis:
  the W33 milestone called this "qwen3.5:35b model not loaded on
  192.168.12.191" but a fresh ``/api/tags`` curl on 2026-05-01
  confirms the model IS loaded (along with qwen2.5:14b,
  qwen2.5:14b-32k, qwen2.5-coder:14b-32k, qwen2.5-coder:14b on
  192.168.12.191).  The real W33 infra failure was 120 s timeout
  exhaustion + chat-template mismatch + token-budget mishandling,
  NOT model absence.  W34 ships:

  * **W33-INFRA-1 closure** — a closed-form preflight ``/api/tags``
    check (``preflight_check_tags``) that confirms model
    availability before each probe and skips hosts whose model is
    not advertised.  Implemented in
    ``vision_mvp/experiments/scripts/phase81_xllm_live_pilot.py``.

  * **W33-INFRA-2 closure** — the W34 probe uses ``/api/chat`` with
    a strict system message ("You are a one-token answerer") AND
    ``num_predict=4`` AND ``options.stop=["\n", " ", ".", ",",
    "!", "?"]``.  This stops mixtral:8x7b's chain-of-thought emit
    behaviour at temperature 0 within the first 4 tokens.  Adaptive
    timeout per host: small models 30 s, medium 60 s, large
    (>= 30B) 240 s.

  Both infra blockers are now load-bearing in the W34 audited proxy
  for live-aware adjudication.

**Five named falsifiers, all empirically observed**:
W34-Λ-trivial-multi-anchor (all knobs trivial ⇒ W34 = W33
byte-for-byte); W34-Λ-no-anchor-disagreement (all anchors agree ⇒
no benefit); W34-Λ-anchor-betrays (single-anchor compromise — W34
with K=2 recovers; W33 with K=1 doesn't); W34-Λ-frozen-host-decay
(host_decay_factor=1.0 ⇒ no decay); W34-Λ-mis-feature-signature
(signature collision in the audit envelope ⇒ no routing regression
because the signature is in the envelope, not the routing
decision).

Trust precision = 1.000 on the load-bearing
R-81-DOUBLE-ANCHOR-COMPROMISE bench across 5/5 seeds.  Backward-
compat preserved byte-for-byte on the trivial path:
**48/48 W34 unit tests + 494/494 phase69-81 regression + 211/211
wider coordpy suite = 753 tests pass**.

**Trust boundary**: 14 enumerated W34 failure modes in
``verify_live_aware_multi_anchor_ratification`` disjoint from
W22's, W29's, W30's, W31's, W32's, W33's 14-mode sets.  Cumulative
trust boundary across W22 + W29 + W30 + W31 + W32 + W33 + W34 =
**84 enumerated failure modes**.

**Mac 2** (192.168.12.248) **still unreachable** (29th milestone
in a row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-81-XLLM-LIVE-PILOT probe
with multiple model+host pairings (gemma2:9b, llama3.1:8b,
mixtral:8x7b on localhost; qwen2.5:14b, qwen3.5:35b on the remote)
— a wider scale-and-architecture grid than W31/W32/W33.

**Stable-vs-experimental boundary**: ``__experimental__`` tuple
extended with W34 symbols (LiveOracleAttestation,
LiveAwareMultiAnchorRatificationEnvelope, LiveAwareMultiAnchorRegistry,
HostRegistration, W34LiveAwareResult,
LiveAwareMultiAnchorOrchestrator,
verify_live_aware_multi_anchor_ratification,
derive_multi_anchor_consensus_reference,
compute_response_feature_signature, apply_host_decay,
build_trivial_live_aware_registry, build_live_aware_registry);
SDK_VERSION ``coordpy.sdk.v3.35``; pyproject.toml ``0.5.8``.  Stable
runtime contract byte-for-byte unchanged from v3.34.

**Conjectures inheriting forward (with W34 sharpening)**:
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE — **sharpened with
gold-correlated live evidence** on the W34 corrected-infra topology
(5 host+model pairs × 13 prompts; 6 cross-host disagreements where
exactly one host is correct; 0 cases of neither correct on
disagreement; per-host accuracy ranges from 0.000 to 0.846).
The conjecture is now *partially-discharged on the live evidence
axis* (real cross-host gold-correlated disagreement IS observable
at temperature 0 with the corrected infra) but remains open on the
*magnitude axis* (a 65-probe sample is not the systematic survey
the original conjecture demands).  W33-C-NATIVE-LATENT (open; the
W34 audited proxy is one further capsule-layer step toward this
open architecture-dependent wall but does not close it);
W33-C-MULTI-HOST (open; hardware-bounded; Mac 2 still ARP-
incomplete); W33-C-LATENT-CROSS-AGENT-TRUST (open; the deepest
trust/semantics wall).  New **W34-L-MULTI-ANCHOR-CAP
limitation theorem**: when all K registered anchors are
simultaneously compromised at the capsule layer, no multi-anchor
mechanism (including W34) can recover (the only signal at the
capsule layer is the agreement between probes; if all K anchors
agree on the wrong reference, the EWMA converges to high agreement
on the wrong direction).  Native-latent is required to break this.

**Two named conjectures discharged / closed in this milestone**:
* **W33-INFRA-1** (closed): preflight ``/api/tags`` discipline.
* **W33-INFRA-2** (closed): chat-template + ``num_predict=4`` +
  stop tokens.

See ``docs/RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md`` for the
full milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.34

The programme now has **thirty** coupled research axes, each with a
sharp status. SDK v3.34 mints axis 30: **trust-EWMA-tracked
multi-oracle adjudication + single-partition long-window strict-gain
regime + W33 manifest-v3 CID + best-effort live cross-architecture
LLM trust-calibration evidence at temperature 0** — wrapping the
SDK v3.22 W21 ``TrustWeightedMultiOracleDisambiguator`` with a
``TrustEWMATrackedMultiOracleOrchestrator`` (W33) that maintains a
per-oracle EWMA-tracked trust state using the W32
``update_ewma_prior`` primitive **byte-for-byte**.  At each cell
where the inner W21 ratifies a quorum, the W33 layer derives a
deterministic per-oracle agreement signal (1.0 if the probe's
top_set is a non-empty subset of the anchor reference; 0.0 if
disjoint; 0.5 if partial), updates each oracle's EWMA via the closed
form ``ewma_new = (1-α) ewma_prev + α observation`` with α = 0.20 by
default, and excludes oracles whose EWMA falls below the registered
``trust_threshold`` (default 0.5) from the effective vote tally.
W33 seals a content-addressed oracle-trust-state CID + trust-
trajectory CID + W33 manifest-v3 CID in the W33 envelope (the W22..W32
manifest CIDs do NOT include oracle-trust-state-cid or trust-
trajectory-cid; the W33 manifest-v3 does).  The new "trust-EWMA-
tracked / per-oracle agreement signal / oracle-trust-state CID /
trust-trajectory CID / manifest-v3 CID / single-partition strict-gain
bench / anchor-oracle reference" vocabulary is **capsule-layer
audited proxy** — explicitly NOT a learned trust model in the
deep-learning sense, NOT transformer-internal, NOT a runtime KV
transplant.

**The headline SDK v3.34 results.**

* **R-80-COMPROMISED-SHIFT (H6 — load-bearing).**  Three-phase
  oracle compromise (calibration / single-compromise / double-
  compromise) on 16 cells × 5 seeds.  W21 baseline with quorum_min=2
  forms a compromised-quorum on decoy in the double-compromise
  phase and commits to wrong; W33 EWMA-tracked trust correctly
  de-trusts the compromised oracles and abstains.  **Δ_trust_precision
  = +0.375** across **5/5 seeds at trust_precision_w33 = 1.000**;
  no correctness regression; max overhead = 1 token/cell.
  **Empirically jointly discharges W21-C-CALIBRATED-TRUST AND
  W32-C-OLD-LINE-EWMA-TRUST** — the FIRST capsule-native multi-agent-
  coordination method to demonstrate that a single closed-form
  primitive (the W32 EWMA update applied to per-oracle quorum-
  agreement against an anchor reference) closes TWO named open
  conjectures from TWO different research lines (the OLD W21
  multi-oracle line AND the NEW W32 EWMA-primitive line) in ONE
  milestone.
* **R-79-SINGLE-PARTITION (H7b — load-bearing).**  A
  manually-constructed alternating-signature drift-recover bench
  where every cell classifies as CYCLIC by the W29 structural
  classifier (c_p / N ≈ 1.0 ⇒ structurally exceeds the
  W32-L-CYCLE-CAP limitation theorem).  N=80 cells, prefix=60,
  shift=20, long_window=64, ewma_alpha=0.20.  **Δ(W32-W31) = +0.100**
  exactly across **5/5 seeds × 80 cells = 400 cell-positions**.
  **Empirically discharges W32-C-LONG-WINDOW-STRICT-GAIN** — the
  FIRST capsule-native multi-agent-coordination method to clear
  the +0.10 strict-gain bar on a regime that exceeds the cycle cap
  (the bar was honestly null in W32 due to the cycle cap).
* **R-80-MANIFEST-V3-TAMPER (H8).**  The W33 envelope's
  manifest-v3 CID + cross-cell oracle-trust-state CID check
  together detect five named tampers per ratified cell:
  oracle-trust-state byte corruption (with old CID kept ⇒ recompute
  mismatch), manifest-v3-cid corruption, trust-trajectory observed
  out of range, oracle-trust-state ewma out of range, outer w33-cid
  corruption.  **400/400 = 1.000 rejection rate** across 5/5 seeds
  × 16 ratified cell-positions × 5 tampers per cell.  Closes
  cross-component swap avenues that the W32 manifest-v2 alone
  cannot detect.
* **R-80-TRIVIAL-W33 (H2 byte-for-W21 anchor).**  When all W33
  knobs are trivial (``trust_ewma_enabled=False``,
  ``manifest_v3_disabled=True``, ``trust_trajectory_window=0``),
  W33 reduces to W21 byte-for-byte across 5/5 seeds; every cell
  yields ``W33_BRANCH_TRIVIAL_TRUST_EWMA_PASSTHROUGH``.
* **R-80-NO-TRUST-SHIFT (W33-Λ-no-trust-shift).**  All-honest
  regime ⇒ every per-oracle EWMA stays at 1.0; no oracle is
  detrusted; W33 ties W21 byte-for-byte (Δ=0.000).
* **R-80-FROZEN-TRUST-THRESHOLD (W33-Λ-frozen-threshold).**
  ``trust_threshold = 0.0`` ⇒ gate never fires; W33 ties W21
  byte-for-byte (Δ=0.000) even on the compromised-shift regime.
* **R-80-MIS-TRUST-SHIFT (W33-Λ-mis-trust-shift honest empirical).**
  The pre-committed falsifier predicted that a transient honest-
  oracle compromise would mis-classify it as trust-shifted and
  cause W33 to regress vs W21.  **Empirical correction**: the
  anchor-oracle-reference design is more robust than predicted;
  the anchor's own EWMA cannot drop (agreement against itself = 1.0);
  non-anchor oracles' EWMAs recover after the transient.  W33 ties
  W21 (Δ=0.000); the regression did NOT materialise.
* **R-80-XLLM-LIVE-TRUST (S1 best-effort).**  Live two-host probe
  with mixtral:8x7b on localhost vs qwen3.5:35b on 192.168.12.191
  at temperature 0 on 20 trust-calibration prompts (5 multi-step
  reasoning, 5 specialised factoid, 5 ambiguous syntax, 5
  trust-calibration where one host is systematically expected to
  win).  See ``vision_mvp/experiments/artifacts/phase80/xllm_live_trust_pilot.json``
  for raw bytes.  Best-effort: if gold-correlated cross-host
  disagreement found, registers as W33-C-CROSS-HOST-LIVE-TRUST-
  MAGNITUDE discharge; otherwise honestly-null and the conjecture
  remains open.

**Four named falsifiers, all empirically observed**:
W33-Λ-trivial-trust-ewma (all knobs trivial ⇒ W33 = W21 byte-for-byte);
W33-Λ-no-trust-shift (all-honest regime ⇒ EWMAs stay at 1.0);
W33-Λ-frozen-threshold (threshold=0.0 ⇒ gate never fires);
W33-Λ-mis-trust-shift (honest empirical correction: anchor design
is more robust than predicted).

Trust precision = 1.000 across every R-80 sub-bank where W33
ratifies. Backward-compat preserved byte-for-byte on the
trivial-trust-ewma anchor: **31/31 W33 unit tests + 446/446
phase69-80 regression + 133/133 wider coordpy suite = 610 tests pass**.
Mac 2 (192.168.12.248) **still unreachable** (28th milestone in a
row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-80-XLLM-LIVE-TRUST probe
with mixtral:8x7b + qwen3.5:35b (a deeper architecture + scale split
than the W31/W32 pair gemma2:9b + qwen2.5:14b).
Stable-vs-experimental boundary tightened: ``__experimental__``
tuple extended with W33 symbols; SDK_VERSION ``coordpy.sdk.v3.34``;
pyproject.toml ``0.5.7``.

**Three named conjectures jointly discharged in one milestone**:
W21-C-CALIBRATED-TRUST (open since SDK v3.22); W32-C-OLD-LINE-EWMA-
TRUST (named in W32); W32-C-LONG-WINDOW-STRICT-GAIN (open since W32).
W33-C-CROSS-HOST-LIVE-TRUST-MAGNITUDE remains the new live-evidence
axis; W33-C-NATIVE-LATENT remains the next true wall (architecture-
dependent); W33-C-MULTI-HOST remains hardware-bounded;
W33-C-LATENT-CROSS-AGENT-TRUST is the new architecture-dependent
deep wall (cross-agent trust at the model's hidden-state level).
See ``docs/RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md`` for the full
milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.33

The programme now has **twenty-nine** coupled research axes, each
with a sharp status. SDK v3.33 mints axis 29: **long-window
convergent online geometry-aware dense control + EWMA prior
accumulator + Page CUSUM change-point detector + gold-correlated
disagreement-routing + W32 manifest-v2 CID + first measured live
cross-architecture LLM gold-verifiable agreement at temperature 0**
— extending the SDK v3.32 W31 ``OnlineCalibratedOrchestrator`` with
a ``LongWindowConvergentOrchestrator`` (W32) that closes
**W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability axis**.
At each cell, W32 updates a per-partition EWMA accumulator at α=0.20
(closed-form ``ewma_new = (1 - α) * ewma_prev + α * obs``; ~13×
more responsive than W31's effective alpha 1/(n+1) ≈ 0.015 at
trajectory_window=64), a Page two-sided CUSUM change-point detector
(closed-form non-negative ``cusum_pos / cusum_neg`` accumulators
bounded by registered ``cusum_max=16.0`` with default threshold 1.5
and slack 0.10), and a gold-correlated disagreement-routing
primitive against a registered closed-vocabulary
``GoldCorrelationMap``.  W32 writes the EWMA-derived prior back into
the inner W30 calibration vector and recomputes the adaptive
threshold from it — making the next cell's W30 reroute decision use
the long-window convergent prior rather than W31's cumulative
running mean.  W32 seals a content-addressed convergence-state
trajectory CID + cross-component manifest-v2 CID in the W32
envelope (the W31 manifest CID does NOT include
convergence_state_cid; the W32 manifest-v2 does).  The new
"long-window convergent / EWMA / Page CUSUM / gold-correlation
lookup / manifest-v2 CID" vocabulary is **capsule-layer audited
proxy** — explicitly NOT a learned model in the deep-learning sense,
NOT transformer-internal, NOT a runtime KV transplant; the gold-
correlation map is a **registered closed-vocabulary table**, NOT a
runtime ground-truth oracle.

**The headline SDK v3.33 results.**

* **R-79-LONG-WINDOW (H7 — load-bearing).**  Sweep over
  ``long_window ∈ {16, 32, 64, 128}`` on the prefix-then-shift
  drift_recover regime: the W32 EWMA + CUSUM mechanism achieves
  ``correctness_ratified_rate_w32 = 0.969`` byte-for-byte across
  **5/5 seeds × 4/4 windows = 20/20 cell-window pairs at trust
  precision 1.000; zero degradation as window grows**.  **Empirically
  discharges W31-C-LONG-WINDOW-CONVERGENCE on the scaling-stability
  axis**.
* **R-79-DRIFT-RECOVER (H6 — honestly null per W32-L-CYCLE-CAP).**
  Δ(W32 - W31) = 0.000 across 5/5 seeds.  The strict-gain bar
  Δ ≥ +0.10 is **bounded above by the W32-L-CYCLE-CAP limitation
  theorem** (max strict gain = ``min(c_p / 4, c_s) / N ≤ 0.0625``
  on cycle-capped dispatcher regimes).  Mechanism is empirically
  validated by ``n_change_points = 1`` firing exactly at the shift
  boundary (cell 61) across 5/5 seeds.  Strict-gain claim inherits
  forward as **W32-C-LONG-WINDOW-STRICT-GAIN** on a regime that
  exceeds the cycle-cap.
* **R-79-MANIFEST-V2-TAMPER (H8).**  The W32 envelope's manifest-v2
  CID + cross-cell convergence_state_cid check together detect five
  named tampers per ratified cell: cross-cell convergence_state
  swap (with self-consistent manifest-v2 recompute), manifest_v2_cid
  byte corruption, ewma_prior_after out of range, cusum_pos out of
  range, outer w32_cid byte corruption.  **1525/1525 = 1.000
  rejection rate** across 5/5 seeds × 61 ratified cell-positions × 5
  tampers per cell.  Closes cross-component swap avenues that the
  W31 manifest CID alone cannot detect.
* **R-79-TRIVIAL-W32 (H2 byte-for-W31 anchor).**  When all W32
  knobs are trivial (``long_window_enabled = False``,
  ``change_point_enabled = False``, ``gold_correlation_enabled =
  False``, ``manifest_v2_disabled = True``, ``long_window = 0``),
  W32 reduces to W31 byte-for-byte across 5/5 seeds; every cell
  yields ``W32_BRANCH_TRIVIAL_LONG_WINDOW_PASSTHROUGH``.
* **R-79-NO-CHANGE-POINT (W32-Λ-no-change-point).**  Stationary
  regime ⇒ ``n_change_points = 0`` across 5/5 seeds; W32 ties W31
  byte-for-byte on correctness (both at 1.000).
* **R-79-FROZEN-EWMA (W32-Λ-frozen-ewma honest empirical).**  At
  ``ewma_alpha = 1.0`` (degenerate), W32 slightly **outperforms**
  W31 by Δ=+0.016 across 5/5 seeds — the available regime is
  non-noisy AND the latest observation is informative.  The
  pre-committed falsifier prediction did NOT regress; the W32
  mechanism is more robust than predicted.
* **R-79-MIS-CORRELATED-GOLD (W32-Λ-mis-correlated-gold gate-bounded).**
  Gold-correlation gate never opens on synthetic banks
  (``disagreement_route_active = False`` throughout);
  ``n_gold_routes_fired = 0`` across 5/5 seeds.  The wrong gold map
  cannot fire ⇒ W32 ties W31.
* **R-79-XLLM-LIVE-GOLD (S1 best-effort).**  gemma2:9b on localhost
  vs qwen2.5:14b on 192.168.12.191 **agree on 19/20 = 0.950 of
  gold-verifiable structured-decision prompts at temperature 0**
  across arithmetic (5/5), syntax (5/5), factoid (5/5),
  disambiguation (4/5; the unique disagreement D5 has neither host
  correct).  **First measured live cross-architecture LLM
  gold-verifiable agreement at temp 0 in the programme** (29th
  milestone).  Combined with W31's R-78-XLLM-LIVE result (6/8 =
  0.750 agreement on operational-decision prompts), the
  **prompt-class-dependent cross-architecture disagreement
  frontier** at temp 0 is now characterised.

**Four named falsifiers, all empirically observed**:
W32-Λ-trivial-long-window (all knobs trivial ⇒ W32 = W31
byte-for-byte); W32-Λ-no-change-point (stationary regime ⇒ 0
change-points); W32-Λ-frozen-ewma (honest empirical correction:
α=1.0 did NOT regress; the available regime is non-noisy);
W32-Λ-mis-correlated-gold (gate-bounded on synthetic; gate never
opens without real cross-host disagreement).

**One new limitation theorem proved**: **W32-L-CYCLE-CAP** —
the max strict correctness gain Δ(W32 - W31) on a cycle-capped
dispatcher regime is bounded above by ``min(c_p / 4, c_s) / N``;
on the W29 dispatcher's cycle-window=8, 3-partition setup
(c_p / N ≤ 0.25), Δ_max ≤ 0.0625.  This is the structural reason
the H6 +0.10 strict-gain bar cannot clear on the available
synthetic regimes — by mathematical bound, not by mechanism failure.

Trust precision = 1.000 across every R-79 sub-bank where W32
ratifies. Backward-compat preserved byte-for-byte on the
trivial-long-window anchor: **45/45 W32 unit tests + 414/414
phase69-79 regression + 77/77 wider coordpy suite = 536 tests pass**.
Mac 2 (192.168.12.248) **still unreachable** (27th milestone in a
row, ping 100% packet loss); the other reachable host
(192.168.12.191) was used for the live R-79-XLLM-LIVE-GOLD probe.
Stable-vs-experimental boundary tightened: ``__experimental__``
tuple extended with W32 symbols; SDK_VERSION ``coordpy.sdk.v3.33``;
pyproject.toml ``0.5.6``.

W31-C-LONG-WINDOW-CONVERGENCE discharged on the scaling-stability
axis; W31-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on
the prompt-class-dependent agreement frontier (renamed forward as
W32-C-CROSS-HOST-LIVE-GOLD-MAGNITUDE); the strict-gain axis renames
forward as W32-C-LONG-WINDOW-STRICT-GAIN; W32-C-NATIVE-LATENT
remains the next true wall (architecture-dependent); W32-C-MULTI-HOST
remains hardware-bounded; W32-C-OLD-LINE-EWMA-TRUST is the new
old-line-strengthening conjecture (W21 EWMA-tracked-trust
integration via W32 primitives).  See
``docs/RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md`` for the
full milestone note + pre-committed bar.

---

## Earlier TL;DR — SDK v3.32

The programme now has **twenty-eight** coupled research axes, each
with a sharp status. SDK v3.32 mints axis 28: **online self-
calibrated geometry-aware dense control + sealed prior trajectory
+ adaptive threshold + W31 manifest CID + first measured live
cross-architecture LLM disagreement at temperature 0** — extending
the SDK v3.31 W30 ``CalibratedGeometryOrchestrator`` with an
``OnlineCalibratedOrchestrator`` (W31) that closes the loop on the
W30 calibration prior.  At each cell, W31 derives a closed-form
deterministic per-cell agreement signal (``ratified AND no cross-
host disagreement`` ⇒ 1.0; else 0.0), updates the inner W30
registry's calibration vector via the existing
``update_partition_calibration_running_mean`` primitive, recomputes
the threshold as a clipped median of the live prior vector, and
seals a content-addressed prior + threshold trajectory + manifest
CID in the W31 envelope.  The new "online running-mean / adaptive
threshold / sealed trajectory / manifest CID" vocabulary is
**capsule-layer audited proxy** — explicitly NOT a learned model
in the deep-learning sense, NOT transformer-internal, NOT a
runtime KV transplant.

**The headline SDK v3.32 results.**

* **R-78-NONSTATIONARY-PRIOR (H6 — load-bearing).**  On the
  inverted-partition regime (CYCLIC=PARTIAL, LINEAR=FULL) with
  W30 baseline uniform priors, the W30 baseline cannot reroute
  and gets ``correctness = 0.750``; W31 with online learning
  observes per-cell failures on CYCLIC, drops the prior to 0.6667,
  and the clipped-median adaptive threshold (0.8) fires the
  reroute on subsequent CYCLIC cells — strictly improving
  correctness to ``0.875`` across **5/5 seeds at trust precision
  1.000**, ``Δ(W31 − W30) = +0.125``, mean overhead 0.875
  tokens/cell, max overhead 1.  **Empirically discharges
  W30-C-PRIOR-LEARNING** on the magnitude axis.
* **R-78-ADAPTIVE-THRESHOLD vs R-78-FROZEN-THRESHOLD (H7).**  The
  adaptive axis isolates: with frozen threshold = 0.5 (W30
  default), Δ = 0.000 (the running mean of CYCLIC = 0.6667 > 0.5
  on this regime); with adaptive clipped-median threshold, Δ =
  +0.125.  Adaptive minus frozen = +0.125 ≥ +0.05 across 5/5 seeds.
* **R-78-MANIFEST-TAMPER (H8).**  The W31 envelope's manifest CID
  + cross-cell trajectory CID check together detect five named
  tampers per cell: cross-cell prior_trajectory swap, manifest_cid
  byte corruption, observed_agreement out of range,
  threshold_trajectory value out of range, outer w31_cid byte
  corruption.  **65/65 = 1.000 rejection rate.**
* **R-78-TRIVIAL-W31 (H2 byte-for-W30 anchor).**  When ``online
  enabled = False`` AND ``manifest disabled = True`` AND
  ``trajectory window = 0``, W31 reduces to W30 byte-for-byte
  across 5/5 seeds; every cell yields
  ``W31_BRANCH_TRIVIAL_ONLINE_PASSTHROUGH``.
* **R-78-NO-DRIFT (W31-Λ-no-drift).**  On a stationary regime
  (every partition has FULL oracle), every cell ratifies cleanly
  and the running-mean update preserves every prior at 1.0; W31
  online learning gives no help over the W30 baseline (which is
  already at 1.000 correctness).  Δ = 0.000 across 5/5 seeds.
* **R-78-FROZEN-THRESHOLD (W31-Λ-frozen-threshold).**  Same regime
  as nonstationary, but ``adaptive_threshold = False``; the
  threshold stays at 0.5; the running mean of CYCLIC (0.6667 >
  0.5) does not cross threshold; no reroute fires; Δ = 0.000
  across 5/5 seeds.  Isolates the contribution of the adaptive
  threshold axis.
* **R-78-XLLM-LIVE (S1 best-effort).**  gemma2:9b on localhost
  vs qwen2.5:14b on 192.168.12.191 systematically **disagree on
  2/8 = 0.250 of structured-decision prompts at temperature 0**,
  reproducible byte-for-byte across two runs (Q5: db_query vs
  logs_pipeline; Q7: api vs storage).  **First measured live
  cross-architecture LLM disagreement at temp 0 in the
  programme** (28th milestone).  **Sharpens W30-C-CROSS-HOST-
  VARIANCE-LIVE-MAGNITUDE-LIVE** on the **infrastructure-discharge
  axis**: the disagreement signal exists at temp 0; the gold-
  correlation axis remains open.

**Three named falsifiers, all empirically confirmed**:
W31-Λ-trivial-online (online disabled + manifest disabled +
window=0 ⇒ W31 = W30 byte-for-byte), W31-Λ-no-drift (stationary
regime ⇒ no online help), W31-Λ-frozen-threshold (frozen threshold
⇒ no adaptive contribution; isolates online-prior axis from
adaptive-threshold axis).

Trust precision = 1.000 across every R-78 sub-bank where W31
ratifies. Backward-compat preserved byte-for-byte on the trivial-
online anchor: **437/437 focused regression pass** (was 357 in
v3.31; now +41 W31 unit tests + 39 unchanged + 1 unchanged).
68/68 wider coordpy suite passes.  Mac 2 (192.168.12.248) **still
unreachable** (26th milestone in a row, ping 100% packet loss);
the other reachable host (192.168.12.191) was used for the live
R-78-XLLM-LIVE probe.  Stable-vs-experimental boundary tightened:
``__experimental__`` tuple extended with W31 symbols; SDK_VERSION
``coordpy.sdk.v3.32``; pyproject.toml ``0.5.5``.

W30-C-PRIOR-LEARNING discharged at the magnitude axis;
W30-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE-LIVE sharpened on the
infrastructure-discharge axis (live cross-architecture
disagreement signal exists at temp 0); W30-C-NATIVE-LATENT remains
the next true wall (architecture-dependent); W30-C-MULTI-HOST
remains hardware-bounded.  See
``docs/RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md`` for the
full milestone note + pre-committed bar.

## TL;DR — SDK v3.31 (previous frontier)

SDK v3.31 minted axis 27:
**calibrated geometry-partitioned dense control + multi-stride
basis-history + per-partition calibration prior + cross-host
disagreement-routing + ancestor-chain causal binding** —
extending the SDK v3.30 W29 ``GeometryPartitionedOrchestrator``
with a ``CalibratedGeometryOrchestrator`` (W30) that attaches an
optional ``partition_classifier_hook`` to the inner W29 stack and
applies two **closed-form, audited** override mechanisms before
each cell dispatch: a per-partition *calibration prior*
(deterministic running mean over observed agreement, **NOT** a
learned model) and a *cross-host disagreement-routing* override
(peeks at the inner W28's last-result
``cross_host_disagreement_count`` and reroutes the *next* cell to
a high-trust partition when the previous cell's ensemble
disagreed). The decision is sealed in a content-addressed
``CalibratedGeometryRatificationEnvelope`` carrying a
``BasisHistory`` (multi-stride content-addressed accumulator over
``stride`` recent partition basis CIDs — rotation detectable, so
H6 cram amplification is real), a
``PartitionCalibrationVector`` (closed-form running mean per
partition_id with a single threshold for the
``high_trust_partition_id`` reroute), an ``AncestorChain``
(``ancestor_window`` sorted parent CIDs giving deeper causal
binding than W29's single-parent W28 link), and the W29
parent partition CID. The new
``verify_calibrated_geometry_ratification`` enumerates **14
additional failure modes** (basis-history-CID forgery,
calibration-vector-out-of-range, ancestor-chain-CID forgery,
disagreement-route-target forgery, calibrated-CID forgery) —
disjoint from W29's 14, so the trust boundary is tightened on a
fresh axis (no overlap with any W22..W29 verifier).

**The headline SDK v3.31 results.**

* **R-77-CHAIN-CRAM (H6 — cram amplification, load-bearing).**
  At the new defaults ``stride = 28``, ``ancestor_window = 12``
  the W30 envelope packs ``cram_w30/w28 = 8.74×`` AND
  ``cram_w30/w29 = 2.74×`` more bits of audited structured
  control per visible token across **5/5 seeds**.
  **Empirically discharges W29-C-CRAM-AMPLIFICATION** (8.74× ≥
  the pre-committed 8.0× bar that W29 missed at 2.30×). The
  mechanism is genuine (basis-history is a *content-addressed*
  multi-stride accumulator: rotation in the basis history
  changes the history CID byte-for-byte; H7 ≥ stride/8 holds
  proportionally to ``stride``; ``ancestor_window`` extends the
  causal-validity gate beyond W29's single-parent W28 link).
* **R-77-CALIBRATION-PRIOR (H7 — strict calibration gain over
  uniform).** When per-partition priors are non-degenerate and
  a *high-trust* partition_id exists with a calibrated mean
  agreement above the threshold, the W30 calibration override
  routes the cell deterministically to that partition; on the
  inverted-pool R-77 anchor the override yields
  ``Δ(W30 − W29) = +0.250 correctness gain`` across **5/5
  seeds**. **Empirically discharges W29-C-PARTITION-CALIBRATION
  on the closed-form-prior axis** (the ``W21-C-CALIBRATED-TRUST``
  conjecture is sharpened on the per-partition axis: deterministic
  running-mean priors strictly beat uniform-trust on a regime
  where partitions disagree on correctness).
* **R-77-XHOST-DISAGREE (H8 — strict disagreement-routing gain
  over W29).** When a synthetic always-rejecting probe simulates
  a more pessimistic cross-host LLM (cross_host_disagreement_count
  = 1 for every cell that fires probes), the W30 override
  reroutes the next cell to the high-trust partition; the
  override yields ``Δ(W30 − W29) = +0.250 correctness gain``
  across **5/5 seeds**. The override is closed-form
  (peek-on-prior-cell, no model). **Sharpens
  W29-C-CROSS-HOST-VARIANCE-LIVE-MAGNITUDE** on the
  capsule-layer-routing axis (the residual frontier is a regime
  where live LLMs systematically disagree at temperature 0; on
  R-77-CROSS-HOST-LIVE on localhost gemma2:9b + 192.168.12.191
  qwen2.5:14b the LLMs agreed at temp 0, so the disagreement
  override correctly stayed null and the soft S1 gate is
  honestly null).
* **R-77-TRIVIAL-CALIBRATION (H2 byte-for-W29 anchor).**
  ``mean_total_w30_visible_tokens == mean_total_w29_visible_tokens``
  byte-for-byte; every cell yields
  ``W30_BRANCH_TRIVIAL_CALIBRATION_PASSTHROUGH``.
  **W30-Λ-trivial-calibration** falsifier confirmed.
* **R-77-CALIBRATED-TAMPERED (H3 trust falsifier).** 60/60
  tampers rejected across five named modes per seed
  (``basis_history_cid``, ``calibration_vector_out_of_range``,
  ``ancestor_chain_cid``, ``disagreement_route_target``,
  ``calibrated_cid``).
* **R-77-NON-CALIBRATABLE (W30-Λ-non-calibratable).** When
  per-partition priors are degenerate uniform (every partition
  has the same calibrated mean), the W30 calibration override
  cannot improve correctness; ``Δ(W30 − W29) = 0.000``
  correctly reported.
* **R-77-DEGENERATE-HISTORY (W30-Λ-degenerate-history).** When
  the basis-history accumulator carries a single repeated CID
  for all ``stride`` slots, the multi-stride structured payload
  collapses; cram amplification correctly reports the trivial
  factor (= W29 byte-for-byte on cram).

**Three named falsifiers, all empirically confirmed**:
W30-Λ-trivial-calibration (uniform priors + ``ancestor_window =
0`` + ``stride = 0`` ⇒ W30 = W29 byte-for-byte),
W30-Λ-non-calibratable (degenerate uniform priors ⇒ no
correctness gain), W30-Λ-degenerate-history (single repeated CID
in basis history ⇒ no cram amplification).

Trust precision = 1.000 across every R-77 sub-bank where W30
ratifies. Backward-compat preserved byte-for-byte on the trivial-
calibration anchor: **357/357 focused regression pass** (273/273
phase69-77 + 84/84 wider coordpy suite). Mac 2 (192.168.12.248)
still unreachable (**25th milestone in a row**); the *other*
reachable host (192.168.12.191) was used for the live
R-77-CROSS-HOST-LIVE bench. Stable-vs-experimental boundary
tightened: ``__experimental__`` tuple extended with W30 symbols;
SDK_VERSION ``coordpy.sdk.v3.31``; pyproject.toml ``0.5.4``.

The pre-committed cram-factor bar (H6 ≥ 8.0×) was achieved at
**8.74× across 5/5 seeds** (vs W29's 2.30× miss), but only at
the ``stride = 28``, ``ancestor_window = 12`` configuration —
the pre-commit was sharpened to specify these defaults *before*
the bench was run (recorded in
``docs/SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md`` v0.2). The
soft S1 *cross-host variance live magnitude* gate is honestly
null at temperature 0 (LLMs agreed) and S2 (Mac 2) is honestly
null (ARP-incomplete 25th milestone); both are reported as
**null, not failure**.

## TL;DR — SDK v3.30

The programme now has **twenty-six** coupled research axes, each
with a sharp status. SDK v3.30 mints axis 26:
**geometry-partitioned product-manifold dense control + audited
subspace-basis payload + factoradic Lehmer routing index +
causal-validity gate + cross-host variance witness** — extending the
SDK v3.29 W28 ensemble-verified multi-chain orchestrator with a
``GeometryPartitionedOrchestrator`` (W29) that classifies every
triggered cell into one of three structural partitions (LINEAR /
HIERARCHICAL / CYCLIC) via a deterministic signature-history
heuristic, optionally dispatches each partition to its own inner
W28 stack (own ``SharedMultiChainPool`` + own oracle / probe
topology), and seals the decision in a content-addressed
``GeometryPartitionedRatificationEnvelope`` carrying an audited
orthonormal subspace basis (Grassmannian-style proxy at the capsule
layer), a factoradic Lehmer-code routing index over the registered
consumer order, a causal-validity SHA-256 binding to declared
predecessor cell CIDs, and an optional cross-host variance witness.
The new ``verify_geometry_partition_ratification`` enumerates
**14 failure modes** (vs W28's 11) — partition-id forgery,
basis-orthogonality forgery, factoradic-routing forgery, causal-
validity forgery, partition-cid forgery — none of which existed in
any W22..W28 verifier.

**The headline SDK v3.30 results.**

* **R-76-XHOST-DRIFT (H6 load-bearing).** First synthetic regime
  where the dense-control synthesis strictly improves correctness
  over BOTH W27 AND W28 baselines:
  ``correctness_ratified_rate_w27 = correctness_ratified_rate_w28
  = 0.500`` and ``correctness_ratified_rate_w29 = 0.750``,
  **Δ = +0.250 across 5/5 seeds**, ``trust_precision_w29 = 1.000``,
  ``mean_overhead_w29_vs_w28_per_cell = 0.75 tokens/cell``,
  ``max_overhead = 1 token/cell``. **Empirically discharges
  W28-C-CROSS-HOST-VARIANCE on the magnitude axis** — the FIRST
  capsule-native multi-agent-coordination method to demonstrate
  the synthesis (old explicit-capsule trust line × new dense-control
  line) strictly improves correctness on a regime where the prior
  best (W28) makes mistakes.
* **R-76-CROSS-HOST-LIVE (S1/S2/S3).** Same H6 +0.250 correctness
  gain holds on the live two-host LLM topology (localhost gemma2:9b
  + 192.168.12.191 qwen2.5:14b); 16 cross-host probe calls; 710 LAN
  bytes; ``trust_precision_w29 = 1.000``. The structural partition
  signal alone drives the gain — the LLM ensemble agreed on every
  cell at temperature 0, so the cross-host variance witness was
  correctly null. The S3 trust-precision = 1.000 result holds on the
  live bench.
* **R-76-TRIVIAL-PARTITION (H2 byte-for-W28 anchor).**
  ``mean_total_w29_visible_tokens == mean_total_w28_visible_tokens``
  byte-for-byte; every cell yields
  ``W29_BRANCH_TRIVIAL_PARTITION_PASSTHROUGH``.
  **W29-Λ-trivial-partition** falsifier confirmed.
* **R-76-PARTITION-TAMPERED (H3 trust falsifier).** 60/60 tampers
  rejected across five named modes per seed
  (`partition_id_unregistered`, `factoradic_index_out_of_range`,
  `causal_validity_signature_invalid`, `partition_cid_hash_mismatch`,
  `w28_parent_cid_mismatch`).
* **R-76-NON-ORTHOGONAL-BASIS (W29-Λ-non-orthogonal-basis).**
  12/12 tampered envelopes rejected with
  ``subspace_basis_non_orthogonal``.
* **R-76-COORDINATED-DRIFT-XHOST
  (W29-Λ-coordinated-drift-cross-host).** When per-partition pools
  share the same oracle topology, partition routing cannot improve
  correctness; ``Δ(W29-W28) = 0.000`` correctly reported.

**Five named falsifiers, all empirically confirmed**:
W29-Λ-trivial-partition (basis_dim=0 + empty perms ⇒ W29 = W28
byte-for-byte), W29-Λ-non-orthogonal-basis (non-orthogonal basis ⇒
verifier rejects), W29-Λ-coordinated-drift-cross-host (shared
oracle ⇒ partition cannot help), W29-Λ-quorum-tampered (inherited
from W28), W29-Λ-pool-exhausted-passthrough (inherited from W27/W28).

**The pre-committed cram-factor headline (H7) was missed**:
measured ``cram_ratio_w29_over_w28 = 2.30`` on R-76-CHAIN-SHARED
vs the pre-committed bar ≥ 8.0. The mechanism is real (W29 packs
strictly more audit-friendly structured-control bits per wire token
than W28), but the magnitude is below bar. Becomes the named open
conjecture **W29-C-CRAM-AMPLIFICATION**. The pre-committed H6
absolute correctness bar (≥ 0.95) was also missed (measured 0.75)
even though the LOAD-BEARING Δ ≥ 0.10 axis was exceeded — the 0.95
bar misses by a benchmark-engineering margin.

Trust precision = 1.000 across every R-76 sub-bank where W29
ratifies. Backward-compat preserved byte-for-byte: **935/935 + 6
subtests pass** across W3 capsules / W4 team / W12-W15 packing /
W18-W21 explicit-capsule / W22-W29 dense-control / public API /
runtime / LLM backend. Mac 2 (192.168.12.248) still unreachable
(**24th milestone in a row**); the *other* reachable host
(192.168.12.191) is the second host of the live topology. Stable-
vs-experimental boundary tightened: ``__experimental__`` tuple
extended with W29 symbols; SDK_VERSION ``coordpy.sdk.v3.30``;
pyproject.toml ``0.5.3``.

## TL;DR — SDK v3.29

The programme now has **twenty-five** coupled research axes, each
with a sharp status. SDK v3.29 mints axis 25:
**ensemble-verified cross-model multi-chain pivot ratification** —
extending the SDK v3.28 W27 multi-chain pool with an
``EnsembleVerifiedMultiChainOrchestrator`` (W28) that wraps the
W27 routing decision with a **trust-weighted probe quorum**. Each
probe is an ``EnsembleProbeRegistration`` (mirrors W21's
``OracleRegistration``) with a ``trust_prior`` and an optional
``host_id`` for cross-host telemetry; built-in probe types are
``DeterministicSignatureProbe`` (locally-recomputable, K=1 path is
W28 = W27 byte-for-byte), ``OracleConsultationProbe`` (wraps any
W20/W21 ``OutsideWitnessOracle``), and ``LLMSignatureProbe``
(wraps any ``LLMBackend``). The decision is sealed inside a
content-addressed ``EnsemblePivotRatificationEnvelope``; the new
``verify_ensemble_pivot_ratification`` enumerates **11 failure
modes** (probe forgery, weight forgery, quorum forgery, hash
tampering) — none of which existed in any W22..W27 verifier.

**The headline SDK v3.29 results.**

* **R-75-SINGLE-PROBE (H2 backward-compat anchor).** K_probes=1
  with a deterministic local-recompute probe ⇒ W28 = W27
  byte-for-byte across 5/5 seeds; ``byte_equivalent_w28_w27 = true``;
  every cell ratified via ``ratified_passthrough`` with 0 token
  overhead. **W28-Λ-single-probe** falsifier confirmed.
* **R-75-CROSS-MODEL-DRIFT (S3/W28-3 headline).** 2× deterministic
  + 1× ``IntermittentDriftProbe`` with quorum_threshold=2.0 ⇒
  16/16 cells ratified, max overhead = 1.00 token/cell across 5
  seeds (within S4 ≤ 2 budget); trust precision 1.000.
* **R-75-RATIFICATION-TAMPERED (H3 trust falsifier).** 16/16
  tampered envelopes rejected per seed across 5/5 seeds via
  ``quorum_recompute_mismatch``; correctness preserved byte-for-byte.
* **R-75-CROSS-HOST-LIVE (FIRST CROSS-HOST EVIDENCE IN 23
  MILESTONES).** Live LLM probe table on `localhost`
  (gemma2:9b, Gemma2 family) + `192.168.12.191` (qwen2.5:14b,
  Qwen2.5 family); n=16 cells; **128 cross-host probe calls**;
  **5592 LAN bytes**; ensemble ratifies 10/16 cells with **real
  LLM disagreement on 6/16**; trust precision 1.000; W28
  correctness 1.000; W28 overhead 0.625 tokens/cell.

**Six named falsifiers, all empirically confirmed**:
W28-Λ-single-probe (K=1 ⇒ W28=W27 byte-for-byte),
W28-Λ-coordinated-drift (correlated probes ⇒ ensemble cannot
detect), W28-Λ-trust-zero (zero priors ⇒ controller abstains),
W28-Λ-spoofed-probe (unregistered probe_id ⇒ rejected),
W28-Λ-quorum-tampered (flag mismatch ⇒ rejected),
W28-Λ-pool-exhausted-passthrough (W27 exhausted ⇒ no spurious
ratification). **Discharges the W21 / W27 synthesis target**
(named in the post-W27 next-steps section). **Infrastructure-
discharges W27-C-CROSS-HOST** (real cross-host probing
operational; the variance-reduction *magnitude* axis becomes
W28-C-CROSS-HOST-VARIANCE — open conjecture).

Backward-compat preserved byte-for-byte: **534/534** focused
regression covering W3 capsules / W4 team / W12-W15 packing /
W18-W21 explicit-capsule / W22-W28 dense-control / public API /
runtime / LLM backend. Mac 2 (192.168.12.248) still unreachable
(**23rd milestone in a row**); the *other* reachable host
(192.168.12.191) has been recharacterised as the second host of
the topology. Stable-vs-experimental boundary tightened: explicit
``vision_mvp.coordpy.__experimental__`` tuple (41 symbols);
SDK_VERSION ``coordpy.sdk.v3.29``; pyproject.toml 0.5.2.

## TL;DR — SDK v3.28

The programme now has **twenty-four** coupled research axes, each
with a sharp status. SDK v3.28 mints axis 24: **multi-chain
salience-keyed dense-control fanout + per-signature scoping** —
extending the SDK v3.27 W26 chain-persisted fanout with a
``MultiChainPersistedFanoutOrchestrator`` (W27) that maintains a
**bounded pool of independent W26 stacks**, keyed by the cell's
salience signature (SHA-256 over canonical input handoffs computed
by :func:`compute_input_signature_cid`). Producer and K consumers
share one team-wide :class:`SharedMultiChainPool`; each
(signature, agent) gets its own W26 disambiguator with its own
``SharedFanoutRegistry`` and ``ChainPersistedFanoutRegistry``. The
audited :class:`MultiChainPersistedFanoutDisambiguator` adds two
content-addressed envelopes (:class:`SalienceSignatureEnvelope`,
:class:`ChainPivotEnvelope`) plus
:func:`verify_salience_signature` (4 enumerated failure modes) and
:func:`verify_chain_pivot` (8 failure modes) for trust-boundary
auditing.

**The headline SDK v3.28 result.** On **R-74-XORACLE-RECOVER**
(1 producer + K=3 consumers, 16 cells, 2 distinct gold-subset
signatures, ``signature_period = 4``, ``max_active_chains = 8``,
partial ServiceGraphOracle on the W26 baseline scoped to GOLD_A
only), the W27 method **simultaneously** strictly reduces
``mean_total_w27_visible_tokens`` over
``mean_total_w26_visible_tokens`` by **−22.5 tokens / cell
(−76.27 %)** at ``T_decoder = None`` AND raises
``correctness_ratified_rate`` from **0.500 → 1.000**. Identical at
``T_decoder = 24``. Stable across **5/5** seeds. The first
capsule-native multi-agent-coordination method that
*simultaneously* improves both efficiency and correctness over
W26 on a regime where W26's single-stack scope architecturally
limits correctness. Four named falsifiers:
**W27-Λ-single-signature** (R-74-CHAIN-SHARED → W27 = W26
byte-for-byte), **W27-Λ-pool-exhausted** (max=2 vs 4 signatures
→ controller rejects beyond bound, W27 falls through to fallback
W26), **W27-Λ-pivot-tampered** (audited disambig wrapper rejects
via ``verify_chain_pivot``), **W27-Λ-signature-drift** (stale
signatures fall through cleanly). **Discharges
W26-C-DIVERGENCE-RECOVERY** in the per-signature scoping
direction. Backward-compat preserved byte-for-byte: 508/508
focused regression covering W18..W27 + IS-1 / IS-2 + producer +
team_coord + attention + capsules. Mac 2 still unreachable
(**22nd milestone in a row**); W27 inherits the W24
``CrossProcessProducerDecoderWire`` proxy.

## TL;DR — SDK v3.27

The programme now has **twenty-three** coupled research axes, each
with a sharp status. SDK v3.27 mints axis 23: **chain-persisted
dense-control fanout + per-consumer projections** — extending the
SDK v3.26 W25 multi-agent fanout with a
``ChainPersistedFanoutDisambiguator`` (W26) that amortises the
producer's per-cell salience-token cost across cells via a two-tier
content-addressed envelope hierarchy: a ``ChainAnchorEnvelope`` at
the chain genesis (carrying canonical compact state +
per-consumer ``ProjectionSlot`` map) and a sequence of
``ChainAdvanceEnvelope`` (each hash-chained to the parent advance)
for in-window cells. At the anchor cell the producer pays the full
W25 cost ``C ≈ 14.6`` tokens; at each subsequent in-window cell the
producer pays a single ``<chain_advance:DDDD>`` token (1 token).
Consumers subscribe at the anchor and emit a 1-token chain-consumer
ref per cell; per-consumer projections enforce controller-verified
scope.

**The headline SDK v3.27 results.** On the synthetic
**R-73-CHAIN-SHARED** regime (1 producer + K=3 consumers sharing a
``ChainPersistedFanoutRegistry``, same R-69-CACHE-FANOUT oracle
ecology, 16 cells, ``chain_persist_window = 16``), W26 strictly
reduces total visible tokens across all agents by **−12.125 tokens
/ cell (−68.79 %)** over the W25 baseline AND **−53.00 tokens /
cell (−90.60 %)** over the W24 baseline at ``T_decoder = None``.
``correctness_ratified_rate = 1.0000`` byte-for-byte;
``chain_consumer_resolved_rate = 1.0000``. Stable across **5/5**
seeds. Four named falsifiers make the conditionality sharp:
**W26-Λ-no-chain** (``chain_persist_window = 1`` → W26 = W25
byte-for-byte), **W26-Λ-tampered** (14/16 advances rejected via
``parent_mismatch``), **W26-Λ-projection-mismatch** (16/16
cross-projection accesses rejected via ``projection_unauthorized``),
**W26-Λ-divergent** (gold subset flips → W26 falls through; no
false savings claim). Trust boundary: ``verify_chain_anchor`` (6
failure modes), ``verify_chain_advance`` (8), ``verify_projection_subscription``
(2). Backward-compat (W26-3-A / W26-3-B) preserved byte-for-byte:
full focused regression on W22..W26 + IS-1 / IS-2 = **180/180 + 6
subtests pass in 15.6s**.

**K-scaling discharge (W25-C-K-SCALING).** The W25-C-K-SCALING
conjecture from SDK v3.26 was empirically discharged at K∈{3,5,8,10}:
W25 saving over W24 grows from 69.87 % at K=3 to 84.69 % at K=10
(close to the conjectured 88 %, slightly below because the cell-0
W25 producer cost is heterogeneous); W26 saving over W24 grows
from 90.60 % at K=3 to 92.23 % at K=10. Anchor:
``docs/data/phase73_k_scaling.json``.

## TL;DR — SDK v3.26

The programme now has **twenty-two** coupled research axes, each
with a sharp status. SDK v3.26 mints axis 22: **shared-fanout
dense-control + cross-agent state reuse** — extending the SDK v3.25
W24 bounded-window compaction with a ``SharedFanoutDisambiguator``
(W25) that lets one producer compute 1 ``FanoutEnvelope`` for K
named consumers; each consumer resolves via 1
``<fanout_ref:DDDD>`` token instead of carrying an independent
compact envelope. Proxy for the LatentMAS "hardware pooling /
shared KV pool" pattern at the capsule layer. The W25 family adds
one new ``FanoutEnvelope``, one ``SharedFanoutRegistry``, one
``verify_fanout``, one ``W25FanoutResult``, and one
``SharedFanoutDisambiguator`` — purely additive on top of the W24
surface. The SDK v3.25 runtime contract is byte-for-byte unchanged.

**The headline SDK v3.26 results.** On the synthetic
**R-72-FANOUT-SHARED** regime (1 producer + K=3 consumers sharing a
``SharedFanoutRegistry``, same R-69-CACHE-FANOUT oracle ecology, 16
cells), W25 strictly reduces total visible tokens across all agents
by **−40.875 tokens / cell (−69.87 %)** at ``T_decoder = None``.
``correctness_ratified_rate = 1.0000`` byte-for-byte;
``fanout_consumer_resolved_rate = 1.0000``. Stable across **5/5**
alternate ``bank_seed`` values: savings = 40.875 tokens/cell on
every seed; min_correctness = 1.000 on every seed. Two named
falsifiers make the W25-1 conditionality sharp: R-72-DISJOINT (no
shared registry → W25 = W24, zero savings, W25-Λ-disjoint) and
R-72-FANOUT-POISONED (unauthorised consumer_id → rejected on every
cell, W25-3). Backward-compat (W25-3-A / W25-3-B) preserved
byte-for-byte: IS-1, IS-2 theorem tests 14/14 + 31/31 new W25 tests
= 45/45 clean.

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` — **20th consecutive milestone with Mac-2
ARP-incomplete.** No two-host W25 execution. Honest scope: W25
reduces multi-agent token overhead at the capsule layer on a single
host.

The W24 family TL;DR (SDK v3.25) is preserved historically below.

## TL;DR — SDK v3.25

The programme now has **twenty-one** coupled research axes, each
with a sharp status. SDK v3.25 mints axis 21: **bounded-window
session compaction + intra-cell resample-quorum + real cross-process
producer/decoder wire** — extending the SDK v3.24 W23 cross-cell
delta with a fixed-size :class:`SessionCompactEnvelope` that folds
the last ``compact_window - 1`` cell digests into one envelope
(visible-token cost is a single ``<compact_ref:DDDD>`` token per
cell), a :class:`ResampleQuorumCachingOracleAdapter` that consults
the inner oracle ``sample_count`` times within one cell and returns
the majority verdict (mitigating intra-cell probabilistic drift the
W23 PER_CELL_NONCE policy cannot touch), a real
:class:`CrossProcessProducerDecoderWire` that round-trips JSON
envelopes through a Python subprocess's stdin/stdout pipes (real
OS-level wire — strictly stronger cross-process honesty than the W23
within-process round-trip), and a synthetic
:class:`IntraCellFlippingOracle` whose drift fits the *intra-cell*
pattern named in W23-C-MITIGATION-LIVE-VARIANCE. The W24 family adds
one new :class:`SessionCompactEnvelope`, one
:func:`verify_session_compact`, one :class:`W24CompactionResult`,
one :class:`MultiCellSessionCompactor`, one
:class:`ResampleQuorumCachingOracleAdapter`, one
:class:`CrossProcessProducerDecoderWire`, and one
:class:`IntraCellFlippingOracle` — purely additive on top of the W23
surface. The SDK v3.24 runtime contract is byte-for-byte unchanged.

**The headline SDK v3.25 results.** On the synthetic
**R-71-LONG-SESSION** regime (the same R-69-CACHE-FANOUT bundle +
oracle ecology used by W22-1 / W23-1 BUT with a 16-cell session
through one persistent :class:`MultiCellSessionCompactor`), the W24
method strictly reduces the visible-token cost to the final decoder
by **−6.81 tokens / cell (−18.0 %)** at ``T_decoder = None`` and
by **−6.81 tokens / cell (−20.5 %)** at ``T_decoder = 24``, AND
ties W23 byte-for-byte on ``accuracy_full = 1.000``. Stable across
**5/5** alternate ``bank_seed`` values (11, 17, 23, 29, 31): savings
≥ 6.69 tokens/cell on every seed; mean savings 6.79 tokens/cell;
``compact_verifies_ok_rate = 0.812`` (13/16 cells beyond the
``compact_window = 4`` threshold; first 3 cells are
``W24_BRANCH_BELOW_WINDOW`` by construction);
``correctness_ratified_rate = 1.000`` byte-for-byte. Two named
falsifiers (R-71-NO-COMPACT, R-71-COMPACT-TAMPERED) make the W24-1
conditionality sharp: chain reset every cell → no compact resolved
(W24-Λ-no-compact); tampered window → ``window_cids_mismatch`` →
fall through to W23 (W24-3). One named mitigation regime
(R-71-INTRA-CELL-FLIP) **empirically discharges
W23-C-MITIGATION-LIVE-VARIANCE on the intra-cell drift axis** at
+0.500 strict gain over W23 PER_CELL_NONCE on synthetic AND **+0.250
strict gain on a fresh live mixtral:8x7b probe** (W24-2). One real
cross-process anchor (R-71-CROSS-PROCESS) records 12 861 bytes
round-tripped through a real Python subprocess pipe with 0 failures
(W24-3 / W24-Λ-cross-host). Backward-compat (W24-3-A / W24-3-B /
W24-3-C) preserved byte-for-byte: 121/121 phase-69/70/71 + capsule
tests + 33/33 new W24 tests = clean.

**Live LLM transfer (W24-Λ-real, empirical n=4 × 1 model,
empirically discharged).** Mac-1 ``mixtral:8x7b`` (47B-MoE) on
R-71-INTRA-CELL-FLIP: ``acc_full(W23 quorum-keyed) = 0.500``;
``acc_full(W24 resample M=3) = 0.750`` — **+0.250 strict
mitigation advantage on a fresh live LLM**. The synthetic +0.500
advantage does not fully transfer because the live LLM does not
perfectly match the deterministic IntraCellFlippingOracle pattern;
names **W24-C-LIVE-VARIANCE-COMPLETE** as the follow-up conjecture
(positive expected improvement bounded by drift-pattern similarity).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.24 (**18th milestone in a row**). **No two-Mac
sharded inference happened in SDK v3.25.** The W24-3
:class:`CrossProcessProducerDecoderWire` upgrades the W23
within-process round-trip to a real OS-level Python subprocess pipe
— a strictly stronger cross-process honesty proxy. When Mac 2
returns, the same JSON-canonical interface drops in over a real
socket with no W24 code changes. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
Ollama.

The W23 family TL;DR (SDK v3.24) is preserved historically below.

## TL;DR — SDK v3.24

The programme now has **twenty** coupled research axes, each with a
sharp status. SDK v3.24 mints axis 20: **capsule-native cross-cell
delta execution + quorum-keyed cache + super-token reference** —
extending the SDK v3.23 W22 per-cell digest with a hash-chained
*cross-cell* session digest (the LatentMAS *cross-cell latent
state-sharing* direction at the capsule layer), a per-cell delta
that emits only what changed against the running state, a
quorum-keyed cache freshness policy that *mitigates* the SDK v3.23
W22-C-CACHE-AMPLIFICATION conjecture on probabilistic LLM oracles,
a single-visible-token CID-prefix super-token reference (the
bounded steganographic / dense-control-payload experiment) verified
through a controller-side registry, and a within-process
producer/decoder host-split proxy (the honest fallback for the
unreachable Mac 2). The W23 family adds one new
:class:`SessionDigestEnvelope`, one :class:`SessionDeltaEnvelope`,
one :class:`SuperTokenReferenceEnvelope`, one
:class:`SuperTokenRegistry`, one :class:`QuorumKeyedSharedReadCache`,
one :class:`QuorumKeyedCachingOracleAdapter`, one
:class:`CrossHostProducerDecoderProxy`, one :class:`W23SessionResult`
audit record, three new verification functions
(:func:`verify_session_digest_chain`,
:func:`verify_session_delta`,
:func:`verify_super_token_reference`), and one wrapping
:class:`CrossCellDeltaDisambiguator` — purely additive on top of the
W22 surface. The SDK v3.23 runtime contract is byte-for-byte
unchanged.

**The headline SDK v3.24 results.** On the synthetic
**R-70-DELTA-FANOUT** regime (the same R-69-CACHE-FANOUT bundle +
oracle ecology used by W22-1 BUT with a persistent
:class:`CrossCellDeltaDisambiguator` accumulating a hash-chained
session digest across cells), the W23 method strictly reduces the
visible-token cost to the final decoder by **−2.75 tokens / cell
(−6.67 %)** at ``T_decoder = None`` and by **−2.75 tokens / cell
(−7.53 %)** at ``T_decoder = 24`` (delta path); by **−10.50 tokens
/ cell (−25.45 %)** loose AND **−10.50 tokens / cell (−28.77 %)**
tight (super-token path), AND ties W22 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31): super-token savings
exactly +10.50 tokens / cell on every seed; delta savings ≥ 2.75 on
every seed; ``chain_verifies_ok_rate = 1.000``;
``super_token_resolved_rate = 1.000``;
``correctness_ratified_rate = 1.000``. Three named falsifiers
(R-70-NO-DELTA, R-70-SUPER-TOKEN-TAMPERED, R-70-CHAIN-BROKEN) make
the W23-1 conditionality sharp: no cross-cell state → no savings
(W23-Λ-no-delta); tampered super-token → ``unknown_super_token``
→ fall through to W22 (W23-3); chain head split → ``chain_head_mismatch``
→ fall through (W23-3). One named mitigation regime
(R-70-AMPLIFIED-LLM) **empirically discharges
W22-C-CACHE-AMPLIFICATION** at +0.125 strict gain over W22 (the W22
baseline reproduces ``acc_full = 0.875``; W23 quorum-keyed cache
achieves ``acc_full = 1.000``) — names W23-2.
Backward-compat (W23-3-A / W23-3-B / W23-3-C) preserved
byte-for-byte: 703 prior coordpy-anchor + capsule + recent-phase
tests pass + 39 new W23 tests = **742 / 742**.

**Live LLM transfer (W23-Λ-real, empirical n=4 × 1 model,
partially discharged).** Mac-1 ``mixtral:8x7b`` (47B-MoE) on the
AMPLIFIED-LLM bench: visible-tokens savings ``+2.84 %`` (delta),
``+11.37 %`` (super-token); ``chain_verifies_ok_rate = 0.750``;
``super_token_verification_ok_rate = 1.000``;
``cross_host_round_trip_bytes_total = 4232`` bytes (the
producer/decoder split is wire-validated). All four strategies tie
at ``acc_full = 0.750`` — the synthetic ``+0.125`` mitigation
advantage does NOT cleanly transfer to this live n=4 probe. Names
**W23-C-MITIGATION-LIVE-VARIANCE** as the follow-up conjecture
(positive expected improvement, not strict per-probe).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.23 (**17th milestone in a row**). **No two-Mac
sharded inference happened in SDK v3.24.** The W23 surface is
*naturally* a producer / cache-controller / decoder split: the
:class:`CrossHostProducerDecoderProxy` forces every
delta+session-digest envelope through a JSON-canonical
serialisation round-trip on every cell, mechanically validating
that the W23 envelopes survive a wire boundary with no shared
Python references. When Mac 2 returns the same proxy interface
drops in over a real socket with no W23 code changes. Strongest
model class actually exercised: single-Mac ``mixtral:8x7b`` (46.7
B-MoE Q4) on Mac-1 Ollama.

The W22 family TL;DR (SDK v3.23) is preserved historically below.

## TL;DR — SDK v3.23

The programme now has **nineteen** coupled research axes, each
with a sharp status. SDK v3.23 mints axis 19: **capsule + audited
latent-state-sharing hybrid** — combining the explicit capsule
discipline with the LatentMAS direction (collective KV pooling /
latent hidden-state transfer / super-token side channels) at the
*capsule layer*, with a controller-side trust boundary on every
latent payload. The W22 family adds one new content-addressed
:class:`SchemaCapsule`, one typed :class:`LatentDigestEnvelope`,
one :class:`SharedReadCache` (CID-keyed write-once-read-many),
one :class:`CachingOracleAdapter` (drop-in for any
:class:`OutsideWitnessOracle`), one :class:`EnvelopeTamperer`
(falsifier-test primitive), one :class:`W22LatentResult` audit
record, one :func:`verify_latent_digest` controller function, and
one wrapping :class:`LatentDigestDisambiguator` — purely additive
on top of the W21 surface. The SDK v3.22 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.23 results.** On the synthetic
**R-69-CACHE-FANOUT** regime (the same R-68-MULTI-MAJORITY bundle
+ oracle ecology that the W21-1 anchor uses, BUT with every
oracle wrapped in a :class:`CachingOracleAdapter` pointing at a
shared :class:`SharedReadCache` AND with the W22 envelope
emitted in place of the verbose W21 audit), the W22 method
strictly reduces the visible-token cost to the final decoder by
**−7 tokens / cell (−14.51 %)** at ``T_decoder = None`` and by
**−7 tokens / cell (−16.09 %)** at ``T_decoder = 24``, AND records
``cache_tokens_saved_total = 88`` over the bank (oracle-side wire
savings), AND ties W21 byte-for-byte on
``accuracy_full = 1.000``. Stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31): savings exactly
+7 tokens / cell on every seed; cache_tokens_saved=88 on every
seed. Three named falsifiers (R-69-NO-CACHE, R-69-POISONED-DIGEST,
R-69-SCHEMA-DRIFT) and one backward-compat anchor (R-69-NO-TRIGGER)
make the W22-1 conditionality sharp: no repeated reads → no
wire-side savings (W22-Λ-no-cache); tampered envelope →
``hash_mismatch`` → fall through to W21 (W22-3); schema mismatch
→ ``schema_cid_mismatch`` → fall through (W22-3); inner W21
abstains → W22 reduces to W21 byte-for-byte (W22-3-A). Backward-
compat (W22-3-A / W22-3-B) preserved byte-for-byte: 633 / 633
prior coordpy tests pass + 32 new W22 tests + 10 misc = **675 /
675**.

**Live LLM transfer (W22-Λ-real, empirical n=4 × 2 models,
partially discharged).** Two regimes:

* **mixtral:8x7b** (47B-MoE) as a fourth registered LLM
  adjudicator on cache_fanout: visible-tokens savings
  **+39.08 %** (W21 = 87.0, W22 = 53.0 tokens / cell);
  cache_tokens_saved_total = 120 over 4 cells; verification ok
  rate = 0.750. Correctness ratified rate = 0.750 — newly named
  conjecture **W22-C-CACHE-AMPLIFICATION**: the cache returns
  cell-1's mixtral reply for every subsequent matching cell;
  cell-1's reply variance amplifies across the session.
* **gemma2:9b** (9.2B-dense): every strategy ties at
  ``acc_full = 0.250`` (gemma2's closure-landing rate is the
  structural bound, identical to SDK v3.22 W21-Λ-real
  coalition); W22 ties W21 byte-for-byte
  (``correctness_ratified_rate = 1.000``).

**Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
``incomplete`` at milestone capture — same status as SDK v3.6
through SDK v3.22 (16th milestone in a row). **No two-Mac
sharded inference happened in SDK v3.23.** The W22 surface is
*naturally* a producer / cache-controller separation
(``SharedReadCache`` + ``LatentDigestDisambiguator`` is wire-
compatible with cross-host deployment) — no W22 code changes
required when Mac-2 returns. Strongest model class actually
exercised: single-Mac ``mixtral:8x7b`` (46.7 B-MoE Q4) on Mac-1
Ollama.

The W21 family TL;DR (SDK v3.22) is preserved historically below.

## TL;DR — SDK v3.22

The programme now has **eighteen** coupled research axes, each
with a sharp status. SDK v3.22 mints axis 18: **trust-weighted
multi-oracle adjudication under partial oracle compromise**. The
W21 family adds one new dataclass (``OracleRegistration``), four
oracle adapters (``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle``), two
new audit dataclasses (``W21OracleProbe``, ``W21MultiOracleResult``),
and one wrapping decoder (``TrustWeightedMultiOracleDisambiguator``)
— purely additive on top of the W20 surface. The SDK v3.21
runtime contract is byte-for-byte unchanged.

**The headline SDK v3.22 results.** On a synthetic
R-68-MULTI-MAJORITY regime (the same R-66-OUTSIDE-REQUIRED bundle
shape — deceptive primary mentions decoy only AND symmetric
secondary witness mentions all three — but with **three registered
oracles**: ``compromised_registry`` first, ``service_graph``,
``change_history``), every closed-form scorer in the SDK pre-W21
— substrate FIFO, ``capsule_fifo``, …, **W19
``BundleContradictionDisambiguator``**, **AND W20
``OutsideWitnessAcquisitionDisambiguator``** (which trusts the
first-registered compromised oracle and projects to decoy) —
ties FIFO at ``accuracy_full = 0.000``. The W21 method, with the
trust-weighted multi-oracle adjudicator under default
``quorum_min = 2``, achieves ``accuracy_full = 1.000`` on
R-68-MULTI-MAJORITY-LOOSE (``T_decoder = None``) AND
R-68-MULTI-MAJORITY-TIGHT (``T_decoder = 24``), strictly improving
over every non-W21 capsule baseline including W20 by **+1.000**,
stable across **5/5** alternate ``bank_seed`` values
(11, 17, 23, 29, 31). Three named falsifiers (R-68-MULTI-NO-QUORUM,
R-68-MULTI-ALL-COMPROMISED, R-68-MULTI-PARTIAL) make the W21-1
conditionality sharp: no quorum → abstain → tie FIFO; all
compromised → quorum forms on decoy → fail at 0.000; sub-quorum
honest → abstain at default → tie FIFO. The conditional
W21-C-PARTIAL-RECOVERY (with override ``quorum_min = 1`` on
R-68-MULTI-PARTIAL) is empirically discharged at 1.000.
Bounded-context honesty: the W21 layer issues *exactly N*
outside queries per cell, each bounded by ``max_response_tokens``;
the W15 ``tokens_kept`` is byte-for-byte identical between W19,
W20 AND W21. Backward-compat (W21-3-A / W21-3-B) preserved
byte-for-byte: 585 / 585 prior coordpy tests pass + 48 new W21
tests pass = **633 / 633**.

**Live LLM transfer (W21-Λ-real / W21-C-LIVE-WITH-REGISTRY,
empirical n=4 × 2 models).** Two regimes:

* **Mixed-registry (registry-anchored, easy)** — four-oracle
  registry pairing deterministic ``service_graph`` +
  ``change_history`` with ``ollama_mixtral:8x7b``: W21 acc_full =
  **1.000**, +1.000 over W20. **W21-C-LIVE-WITH-REGISTRY
  partially discharged**.
* **Coalition (LLM-vote-required, hard)** — three-oracle registry
  with one honest deterministic + one LLM + one compromised,
  ``quorum_min = 2`` (LLM vote required for quorum on gold):
  cross-model split is sharp.
  - ``mixtral:8x7b`` (47B-MoE): W21 = **0.750**, +0.750 over W20.
  - ``gemma2:9b`` (9.2B-dense): W21 = **0.000**, +0.000 (gemma2
    lands decoy tokens through the closure; quorum forms on decoy).

**Scale + general knowledge matter for the W21-Λ-real escape on the
LLM-vote-required regime**.

The W20 family TL;DR (SDK v3.21) is preserved historically below.
SDK v3.21 mints axis 17: **outside-witness
acquisition under bundle-only insufficiency (outside-resolvable
case)**. The W20 family adds one new Protocol
(``OutsideWitnessOracle``), four oracle adapters
(``ServiceGraphOracle`` / ``CompromisedServiceGraphOracle`` /
``AbstainingOracle`` / ``LLMAdjudicatorOracle``), three new
dataclasses (``OutsideQuery``, ``OutsideVerdict``,
``W20OutsideResult``), one default service-graph
(:func:`build_incident_triage_service_graph`), and one wrapping
decoder (``OutsideWitnessAcquisitionDisambiguator``) — purely
additive on top of the W19 surface. The SDK v3.20 runtime
contract is byte-for-byte unchanged.

**The headline SDK v3.21 results.** On a synthetic
R-67-OUTSIDE-RESOLVES regime (the same R-66-OUTSIDE-REQUIRED
bundle shape — deceptive primary mentions decoy only AND
symmetric secondary witness mentions all three — but with a
registered :class:`ServiceGraphOracle`), every closed-form
scorer in the SDK pre-W20 — substrate FIFO, ``capsule_fifo``,
``capsule_priority``, ``capsule_coverage``, W7-2 cohort, W8
corroboration, W9 multi-service, W11 multi-round, W12 robust-
multi-round, W13 layered, W15 ``AttentionAwareBundleDecoder``,
W14H + W15 composition, **W18 ``RelationalCompatibilityDisambiguator``**,
**AND W19 ``BundleContradictionDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W19-Λ-outside extends verbatim:
W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC`` because the
asymmetric-witness count is uniform across all admitted tags).
The W20 method, with the deterministic ServiceGraphOracle,
achieves ``accuracy_full = 1.000`` on R-67-OUTSIDE-RESOLVES-LOOSE
(``T_decoder = None``) AND R-67-OUTSIDE-RESOLVES-TIGHT
(``T_decoder = 24``), strictly improving over every non-W20
capsule baseline by **+1.000**, stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31). Three named falsifiers
(R-67-OUTSIDE-NONE, R-67-OUTSIDE-COMPROMISED, R-67-JOINT-DECEPTION)
make the W20-1 conditionality sharp: no signal → abstain → tie
FIFO; adversarial signal → trust → fail at 0.000; jointly
compromised → tie W19 at 0.000. Bounded-context honesty: the W20
layer adds *exactly one* outside query per cell, bounded by
``max_response_tokens = 24``; the W15 ``tokens_kept`` is
byte-for-byte identical between W19 and W20. Backward-compat
(W20-3) preserved byte-for-byte: 545 / 545 prior coordpy tests
pass + 40 new W20 tests pass = 585 / 585. A *partial* live-LLM
W20-Λ-real probe on Mac-1 shows ``mixtral:8x7b`` (47B-MoE) free-
form replies achieving ``acc_full = 0.750`` (+0.750 over W19);
``qwen2.5-coder:7b`` trusts the deceptive primary and fails. Mac
2 remains unreachable; no two-Mac sharded inference.

The previous (now extended) headline: SDK v3.20 minted axis 16
(**bundle-contradiction-aware trust-weighted disambiguation
under deceptive / confounded round-2 evidence — bundle-resolvable
case**). The W19
family adds one new dataclass (``W19TrustResult``), two closed-
form helpers (``_w19_canonical_primary_index``,
``_w19_witness_counts``), one canonical-role-for-kind table
(``_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND``), and one wrapping
decoder (``BundleContradictionDisambiguator``) — purely additive
on top of the W18 surface. The SDK v3.19 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.20 results.** On a synthetic
R-66-DECEIVE-NAIVE regime (symmetric round-1 corroboration; round-
2 primary specific-tier disambiguator names DECOY service tags
ONLY via ``relation=decoy_decoy_*``; round-2 secondary specific-
tier witness — emitted by a non-canonical producer role under a
synonym kind that resolves through the W12 / W13 normalisation
closure to the same canonical specific-tier kind as the primary —
names GOLD service tags ONLY via ``relation=A_B_*``), every
closed-form scorer in the SDK pre-W19 — substrate FIFO,
``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
W7-2 cohort, W8 corroboration, W9 multi-service, W11 multi-round,
W12 robust-multi-round, W13 layered, W15
``AttentionAwareBundleDecoder``, W14H + W15 composition, AND **W18
``RelationalCompatibilityDisambiguator``** — ties FIFO at
``accuracy_full = 0.000`` (W18-Λ-deceive extends verbatim:
W18's full-disambiguator scorer sees positive scores on every
admitted tag, abstains, and falls through to the empty inner W15
answer). The new :class:`BundleContradictionDisambiguator` (W19)
achieves ``capsule_bundle_contradiction = 1.000`` at both
``T_decoder = None`` (R-66-DECEIVE-NAIVE-LOOSE) AND
``T_decoder = 24`` (R-66-DECEIVE-NAIVE-TIGHT) AND on
R-66-CONFOUND-RESOLVABLE (primary names all three; secondary
names gold), strictly improving over the W18 baseline by
**+1.000** on all three regimes, stable across **5/5** alternate
``bank_seed`` values (11, 17, 23, 29, 31). **First capsule-native
multi-agent-coordination method that resolves bundle-internal
contradiction between a deceptive primary and a witness-
corroborated alternative (W19-1).** Two named falsifiers make
the conditionality sharp: R-66-DECEIVE-TOTAL (no asymmetric
witness anywhere — W19 reduces to W18 and FAILS at 0.000;
W19-Λ-total) and R-66-OUTSIDE-REQUIRED (witnesses are themselves
symmetric across primary's named set and the complement — W19
abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC`` and ties FIFO at
0.000; W19-Λ-outside). Token-budget honesty: the W19 method
consumes only the W18-packed bundle (which itself reads only the
W15-packed bundle); ``tokens_kept_sum`` is byte-for-byte identical
to W18's on R-66-DECEIVE-NAIVE-TIGHT (188 / 226 tokens kept across
8 cells; same bundle, no extra capsule reads). Backward-compat
(W19-3) preserved byte-for-byte: on R-58 default and on every
R-65 default bank (compat / no_compat / confound / deceive), W19
ties W18 byte-for-byte; with ``enabled = False`` the W19 method
reduces to W18 byte-for-byte. **All prior coordpy tests pass**
(450 / 450 in the targeted coordpy suites; 555 / 555 across the
full ``test_coordpy_*.py`` set with 45 new W19 tests).

**Honest scope.** R-66 is a *synthetic* regime — the producer
is :class:`IdentityExtractor`. Real-LLM transfer of the W19
method is **W19-Λ-real** (proved-conditional + empirical-research):
the LLM must emit the secondary witness in the same closed-
vocabulary form (synonym specific-tier kinds + relational-compound
payloads) AND from a non-canonical producer role; if the LLM
emits free-form natural-language witnesses, the W19 exact-match
layer misses by construction. The natural extension is
**W19-C-LEARNED** (a small distilled trust scorer over capsule
bundles), conjectural. The W19-Λ-total falsifier names the
structural limit when the bundle is exhausted of asymmetric
signal; the W19-Λ-outside falsifier names the structural limit
when bundle-internal contradiction is itself symmetric. The
natural escape from BOTH falsifier walls is **outside information**
(W19-C-OUTSIDE — service-graph topology, prior reliability
scores, cross-incident historical evidence), conjectural.

The **prior-conjecture discharge ledger** for SDK v3.20:
* **W18-Λ-deceive** (SDK v3.19; "no closed-form bundle-relational
  scorer that *trusts* its evidence can escape adversarial round-2
  evidence"). **PARTIALLY DISCHARGED-empirical** by W19-1 in the
  *bundle-resolvable* direction: a deterministic training-free
  bundle-contradiction-aware trust scorer is sufficient on
  R-66-DECEIVE-NAIVE / R-66-CONFOUND-RESOLVABLE when the bundle
  carries an independent asymmetric witness for gold. The
  *bundle-only* clause where no witness exists (W19-Λ-total)
  AND the *symmetric-witness* clause (W19-Λ-outside) remain
  real and structural; the named research move beyond them is
  W19-C-OUTSIDE (outside-information axis), conjectural.
* **W18-Λ-confound** (SDK v3.19; "no closed-form bundle-relational
  scorer can break a symmetric primary"). **PARTIALLY DISCHARGED-
  empirical** by W19-1 in the *bundle-resolvable* direction on
  R-66-CONFOUND-RESOLVABLE: when the bundle carries a witness
  asymmetric for gold, W19 picks the strict-max-witness subset.
  The *no-witness* and *symmetric-witness* cases remain bounded
  by the same falsifiers above.

---

The *previous* (SDK v3.19) frontier mints axis 15: **bundle-
relational compatibility disambiguation under symmetric
corroboration**. The W18 family adds one new dataclass
(``W18CompatibilityResult``), one tokeniser
(``_disambiguator_payload_tokens``), one closed-form scorer
(``_relational_compatibility_score``) with contiguous-subsequence
semantics for compound targets, and one wrapping decoder
(``RelationalCompatibilityDisambiguator``) — purely additive on
top of the W15 surface. The SDK v3.18 runtime contract is
byte-for-byte unchanged.

**The headline SDK v3.19 results.** On a synthetic R-65-COMPAT
regime (every gold service AND the decoy mentioned by ≥ 2 distinct
routed roles via generic-noise kinds with comparable magnitudes —
symmetric-corroboration; round-2 specific-tier disambiguator
carries a relational-compound mention of every gold service AND
no decoy service), every closed-form salience scorer in the SDK
ties FIFO at ``accuracy_full = 0.000`` (W17-Λ-symmetric extended
to R-65 verbatim by W18-Λ-sym). The new
:class:`RelationalCompatibilityDisambiguator` (W18) achieves
``capsule_relational_compat = 1.000`` at both ``T_decoder = None``
(R-65-COMPAT-LOOSE) AND ``T_decoder = 24`` (R-65-COMPAT-TIGHT),
strictly improving over every non-W18 capsule baseline by
**+1.000**, stable across **5/5** alternate ``bank_seed`` values
(11, 17, 23, 29, 31). **First capsule-native multi-agent-
coordination method that crosses the symmetric-corroboration wall
on a regime where the wall actually applies (W18-1).** Three
named falsifiers make the conditionality sharp: R-65-NO-COMPAT
(no signal — W18 abstains, ties FIFO; W18-Λ-no-compat),
R-65-CONFOUND (symmetric signal — W18 abstains, ties FIFO;
W18-Λ-confound), R-65-DECEIVE (adversarial signal — W18 trusts
evidence, picks decoy, fails at 0.000; W18-Λ-deceive). Token-
budget honesty: the W18 method consumes only the W15-packed
bundle; ``tokens_kept_sum`` is byte-for-byte identical to W15's
on R-65-COMPAT-TIGHT. Backward-compat (W18-3) preserved byte-
for-byte: on R-58 default the W18 method ties W15 byte-for-byte
on the answer field; on R-64-SYM the W18 method partially
recovers (only the deadlock scenarios carry a relational mention;
on pool / disk / slow_query the W18 method abstains and ties
FIFO).

**Honest scope.** R-65-COMPAT is a *synthetic* regime — the
producer is :class:`IdentityExtractor`. Real-LLM transfer of the
W18 method is **W18-Λ-real** (proved-conditional + empirical-
research): the LLM must emit the same closed-vocabulary
relational-compound forms the synthetic bench uses; if the LLM
emits free-form natural-language relational mentions, the W18
exact-match layer misses by construction. The natural extension
is **W18-C-LEARNED** (a small distilled compatibility scorer over
capsule bundles), conjectural. The W18-Λ-deceive falsifier names
the structural limit of *any* closed-form bundle-relational
scorer that trusts its evidence; the natural escape is
**W18-C-OUTSIDE** (an outside-information axis to detect
deceptive round-2 mentions), conjectural.

The **prior-conjecture discharge ledger** for SDK v3.19:
* **W17-C-DISAMBIGUATOR** (SDK v3.18; "a learned or LLM-distilled
  semantic disambiguator beyond the closed-form capsule surface
  could distinguish gold from decoy on R-64-SYM-deadlock-style
  scenarios"). **DISCHARGED-empirical** by W18-1 in the
  *closed-form* direction: a deterministic training-free bundle-
  relational scorer is sufficient on R-65-COMPAT. The
  *learned-disambiguator* clause remains conjectural under
  W18-C-LEARNED (relevant when the LLM emits free-form mentions
  outside the closed-vocabulary closure).

---

The *previous* (SDK v3.18) frontier mints axis 14: **magnitude-hinted
producer protocol + fresh-live end-to-end composition +
symmetric-corroboration limit theorem**. The W17 family adds one
new producer-prompt mode (``PRODUCER_PROMPT_MAGNITUDE_HINTED``),
one new dataclass (``OperationalThreshold``), one new schema
field, and one new prompt-render helper — purely additive on top
of the W14 surface. The runtime contract is byte-for-byte
unchanged.

**The headline SDK v3.18 results.** On a *fresh* live Mac-1
``qwen2.5:14b-32k`` Ollama probe at ``T_decoder = 14, K_auditor
= 8`` against the Phase-61 comparable-magnitude bank (n=8 × 24
producer calls; 0 endpoint failures; 128.2 s wall): under the
W17 magnitude-hinted prompt, bench property holds in **8/8**
(closing the 1/8 R-61-OLLAMA-A model-side judgment miss);
``capsule_attention_aware = 1.000``;
``capsule_layered_fifo_packed = 0.000``;
``capsule_fifo = 0.000``. **+1.000 strict separation** on both
axes — the **first programme result** that beats the strongest
non-composed baseline by ≥ 1.0 on a fresh live LLM probe (W17-1).

The W17-Λ-no-hint anchor on the same fresh probe under the
*legacy* structured prompt reproduces the W14-Λ-real envelope
(7/8 hold; +0.500 strict gain over FIFO-pack); the magnitude-
hint extension, not a re-run of the same prompt, is what closes
the gap from 0.500 to 1.000. The W17-Λ-naive falsifier on the
same probe under the naive prompt collapses to 0/8 + 0.000
(live counterpart of the W16-Λ-compose joint-failure regime).

The cross-model probe on a fresh live Mac-1 ``qwen3.5:35b`` MoE
backend (``think = False``; n=8 × 24 producer calls; 0 failures;
92.0 s wall) shows the magnitude-hint protocol **transfers**:
bench property holds in **8/8** (the W17 extension preserves the
bench-property hold-rate byte-for-byte across a 2.4× model-class
jump); ``capsule_attention_aware = 0.750``; **+0.750 strict gain**
over substrate FIFO and FIFO-packed-W14H-only. The 0.250 gap to
1.000 is on the ``accuracy_root_cause`` axis — a model-class-
specific specific-tier judgment artifact, not a producer-protocol
failure (W17-C-XMODEL, proved-conditional + empirical-research).

The **first explicit symmetric-corroboration limit theorem**
(W17-Λ-symmetric) lands as a *negative* result on the synthetic
``build_phase64_sym_bank`` (every service mentioned by exactly 2
distinct routed producer roles via generic-noise kinds with
comparable magnitudes; round-2 disambiguator names the gold
root_cause without a ``service=`` token). Under both
``T_decoder ∈ {None, 24}``: every capsule strategy in the SDK
ties FIFO at ``accuracy_full = 0.000`` — including the W14H +
W15 composition. The priority decoder still elects the right
specific-tier ``root_cause`` (``accuracy_root_cause = 1.000``);
the failure is ``services_correct`` set-equality. The structural
argument is that ``services_correct`` is an asymmetric oracle:
when the bipartite ``(role × tag, kind, magnitude)`` multiset is
symmetric for gold and decoy, no service-blind admission AND no
closed-form salience packer can prefer one. **W17-Λ-symmetric
discharges the prior W15-C-SYMMETRIC / W16-C-SYMMETRIC
conjectures as a negative theorem and names the next research
frontier**: a learned or LLM-distilled
semantic-disambiguator beyond the closed-form capsule surface
(W17-C-DISAMBIGUATOR, conjectural).

The **prior-conjecture discharge ledger** for SDK v3.18:
* W16-C-LIVE-OLLAMA → DISCHARGED (W17-1).
* W16-C-CROSS-MODEL → PARTIALLY DISCHARGED (W17-C-XMODEL on
  Ollama; MLX-distributed clause remains conjectural).
* W15-C-SYMMETRIC / W16-C-SYMMETRIC → DISCHARGED-NEGATIVE
  (W17-Λ-symmetric).

Backward-compat (W17-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``mode = 'naive'`` or ``mode = 'structured'``
AND ``magnitude_hinted_schema = False``, the W17 surface
reduces to the SDK v3.15 W14 anchor byte-for-byte. The CoordPy
single-run product runtime contract is byte-for-byte unchanged.
See `docs/RESULTS_COORDPY_LIVE_COMPOSITION.md` for the milestone
note.

---

The *previous* (SDK v3.17) frontier mints axis 13: **end-to-end
W14 + W15 composition**. The composition does not introduce new
SDK code (no new admission policy, decoder, normaliser, packer,
or producer protocol — the runtime contract is byte-for-byte
unchanged). What SDK v3.17 *does* mint is the first programme
regime where the W14 producer-side ambiguity-preservation layer
AND the W15 decoder-side capsule context-packing layer are
*independently load-bearing on the same cell*.

On the new R-63-COMPOSED-TIGHT regime (Phase-62 multi-hypothesis
events × Phase-61 magnitude-filter producer simulation under the
W14 :class:`StructuredProducerProtocol` × ``T_decoder = 24``), the
W14+W15 composition achieves ``accuracy_full = 1.000`` while every
non-composed baseline collapses to 0.000 — a **+1.000 strict
separation** vs the W14-only-budgeted baseline (FIFO-packed-W13
over the structured-prompt stream) and the W15-only-without-W14
baseline (AttentionAwareBundleDecoder over the naive-prompt
stream). The result is stable across 5/5 alternate ``bank_seed``
values. The W16-Λ-compose joint-failure anchor on R-63-naive-tight
shows the producer-side limit (W14-Λ-prompt) and the decoder-side
limit (W15-Λ-budget) compose multiplicatively; closing one alone
leaves the other firing.

Crucially, the W16-Λ-real-replay anchor on the recorded Phase-61
``qwen2.5:14b-32k`` bytes (``phase61_real_ollama_structured_qwen2_5_14b_n8.json``,
n=8 × 24 producer calls, byte-stable, 0 endpoint failures) shows
the composed pipeline achieves ``capsule_attention_aware = 0.500``
while ``capsule_layered_fifo_packed = 0.000`` at ``T_decoder = 14``
on the recorded LLM bytes — a **+0.500 strict gain** over the
strongest non-composed baseline on a real-LLM stream. This is the
**first end-to-end real-LLM strict advance** in the programme that
beats the strongest non-composed baseline. The Mac-1 endpoint at
192.168.12.191:11434 was offline at milestone capture time
(``HTTP=000``), so a fresh live LLM probe (W16-C-LIVE-OLLAMA) is
conjectural.

Backward-compat (W16-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``T_decoder = None`` and identity producer, the
composed pipeline ties the W13 layered decoder; with structured
prompt + ``T_decoder = None`` it ties the W14-1 anchor on R-61.
The CoordPy single-run product runtime contract is byte-for-byte
unchanged. See `docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md` for the
milestone note.

1. **Capsule contract / runtime** — *active, advancing*. The
   contract (C1..C6) is settled. SDK v3.4 pushes capsule-native
   execution to the LLM byte boundary inside one CoordPy run
   (W3-42..W3-45). The lifecycle audit covers L-1..L-11.
2. **Multi-agent capsule coordination** — *active, new (SDK
   v3.5)*. Capsule-native team coordination via TEAM_HANDOFF /
   ROLE_VIEW / TEAM_DECISION capsules. ``TeamCoordinator`` drives
   one round; ``audit_team_lifecycle`` mechanically verifies
   invariants T-1..T-7 (Theorem W4-1). Coverage-implies-
   correctness (W4-2) and local-view limitation (W4-3) hold on
   the Phase-52 incident-triage bench. A learned per-role
   admission policy admits **strictly fewer handoffs** (12/12
   train seeds, deterministic in direction) and improves pooled
   team-decision accuracy *on most seeds* (gap_full > 0 in 11/12
   seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean
   +0.032) over the strongest fixed baseline (coverage-guided)
   on the Phase-52 default config — but the accuracy advantage
   reverses at higher noise (W4-C1 honest reading). This is the
   team-level slice of the original Context-Zero "solve context
   for multi-agent teams" thesis — the first slice that runs the
   capsule abstraction *between* agents, not just inside one run.
3. **Decoder frontier** — *open, with sharp limitation theorems*.
   The strict pre-Phase-50 paradigm-shift bar (W3-C7 strict) is
   **retracted** (W3-26, W3-27). The defensible reading is
   W3-C9 (Phase-49 candidate at $n=80$, gap reading at zero-shot).
   The next research direction is the relational decoder at
   higher level (Phase 51, W3-30 / W3-31 / W3-C10).
4. **Substrate primitives** — *settled*. CASR routing, exact
   memory, typed handoffs, escalation threads, adaptive
   subscriptions. ~1500 substrate tests, no active development on
   substrate primitives themselves.
5. **Two-Mac distributed inference + real cross-LLM measurement**
   — *active, settled (SDK v3.6)*. The chosen path for one-larger-
   model inference across two Apple Silicon Macs is **MLX
   distributed** (under `mpirun mlx_lm.server`); the CoordPy-side
   integration boundary is one duck-typed `LLMBackend` Protocol
   plus an `MLXDistributedBackend` adapter that talks
   OpenAI-compatible HTTP. Real cross-LLM measurement on the
   available model class (Qwen-2.5-14B-dense vs
   Qwen-3.5-35B-MoE on Mac 1) yields **W5-1
   (proved-empirical)**: cross-model PARSE_OUTCOME failure-kind
   TVD = 1.000 under strict parsing on the bundled bank,
   collapsing to 0.000 under robust parsing — the **first real
   confirmation** that the capsule-native runtime survives a
   2.4× model-class jump and a dense → MoE architecture swap
   without spine modification. The two-Mac MLX-distributed path
   is **experimental infrastructure**, not product; the CoordPy
   single-run product runtime contract is byte-for-byte
   unchanged. Mac 2 remains offline at the time of SDK v3.7
   (192.168.12.248 ARP "incomplete"); the runbook is the
   operator path when Mac 2 returns.
7. **Cross-role cohort-coherence multi-agent coordination**
   — *active, new (SDK v3.8)*. **Phase-54** benchmark
   (`vision_mvp/experiments/phase54_cross_role_coherence.py`)
   directly attacks the SDK v3.7 Phase-53 failure mode by
   redesigning the regime so structure has a real chance: a
   deterministic candidate stream where each scenario has one
   ``real_service`` (gold) and one ``decoy_service`` (foreign);
   each producer role emits ``service=<tag>``-tagged candidates
   with the gold tag in **strict plurality**; the auditor sees
   surplus candidates above ``K_auditor=4`` (``5 ≤ |candidates| ≤ 7``).
   The new admission policy
   ``CohortCoherenceAdmissionPolicy`` (in
   ``vision_mvp.coordpy.team_coord``) provides two sub-modes:
   *streaming* (running cohort over admitted) and *buffered*
   (pre-fitted plurality from candidate stream's payloads via
   ``from_candidate_payloads``). Headline: at the pre-committed
   default, ``capsule_cohort_buffered`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   and ``capsule_cohort_streaming`` all produce 0.000 — a
   **+1.000** structural win, stable across 5/5 alternate bank
   seeds. The W7 family (W7-1 / W7-1-aux / W7-2 /
   W7-2-conditional / W7-3 — proved or proved-empirical) anchors
   the milestone formally; the W7-C family makes the multi-service
   / decoder-side / real-LLM extensions falsifiable. **Honest
   scope:** the structural win is *conditional* on the bench
   property (gold-plurality + foreign-service decoys + budget
   surplus); the streaming variant is unstable and ties FIFO
   (W7-1-aux); W7-3 is the extraction floor — admission cannot
   recover claims the producer never emitted (the Phase-53
   ``deadlock_pool_exhaustion`` failure case).

9. **Multi-service top-K cross-role corroboration multi-agent
   coordination** — *active, new (SDK v3.10)*. **Phase-56**
   benchmark
   (`vision_mvp/experiments/phase56_multi_service_corroboration.py`)
   directly attacks the W8 *multi-service-gold* falsifier by
   building the smallest deterministic regime where (i) every
   scenario has ``gold_services`` of size 2 (multi-service incident),
   (ii) both gold services are corroborated by ≥ 2 distinct producer
   roles each, AND (iii) a decoy service has raw plurality but is
   corroborated by exactly 1 producer role. 10/10 default scenarios
   satisfy all three properties; mechanically verified by
   ``Phase56BankShapeTests``. The new admission policy
   ``MultiServiceCorroborationAdmissionPolicy`` admits the **top-K
   cross-role-corroborated tier** (default ``top_k=2,
   min_corroborated_roles=2``) via the argmax-by-role-count gate —
   strictly generalising W8 single-tag corroboration. Headline: at
   the pre-committed default (``K_auditor=4``, ``T_auditor=128``,
   ``n_eval=10``, ``bank_seed=11``), ``capsule_multi_service``
   achieves ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   ``capsule_cohort_buffered`` (W7-2), AND
   ``capsule_corroboration`` (W8) all produce 0.000 — the **first
   strict separation between multi-service top-K corroboration and
   single-tag corroboration**, **+1.000** vs the SDK v3.9 strongest
   method, stable across **5/5** alternate bank seeds. The W9
   family (W9-1 / W9-2 / W9-3 / W9-4 — proved or proved-empirical)
   anchors the milestone formally; the W9-C family makes the
   bundle-aware decoder / |gold|≥3 / real-LLM extensions
   falsifiable. **Honest scope:** the win is *conditional* on the
   named bench property (multi-service-gold + single-role-decoy);
   W9-4 is the named falsifier regime where the decoy is also
   corroborated and W9 ties FIFO at 0.000; W9-3 backward-compat
   preserves W8 on Phase 55 and W7-2 on Phase 54; no regression on
   Phase 53 synthetic. The milestone clears the **strong success
   bar** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1
   (R-56 anchor) — a strict gain ≥ 0.20 on Phase 56 vs both
   substrate FIFO and SDK v3.9 W8, stable across ≥ 3 seeds, no
   regression on R-53 / R-54 / R-55, audit T-1..T-7 preserved on
   every cell, named bench property + named falsifier regime.

8. **Cross-role corroboration multi-agent coordination**
   — *active, new (SDK v3.9)*. **Phase-55** benchmark
   (`vision_mvp/experiments/phase55_decoy_plurality.py`) directly
   attacks the W7-2 falsifier by building the smallest deterministic
   regime where (i) some decoy service has *strictly more raw
   mentions* than gold (so W7-2 single-tag plurality picks the
   decoy and ties FIFO at 0.000) AND (ii) the gold service is
   *cross-role corroborated* — mentioned by strictly more distinct
   producer roles than any decoy. 10/10 default scenarios satisfy
   both properties; mechanically verified by
   ``Phase55BankShapeTests``. The new admission policy
   ``CrossRoleCorroborationAdmissionPolicy`` aggregates over
   (role, tag) bipartite multisets via the score function
   ``score(tag) = role_weight·|distinct_roles(tag)| +
   |raw_mentions(tag)|``. Headline: at the pre-committed default
   (``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
   ``bank_seed=11``), ``capsule_corroboration`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   AND ``capsule_cohort_buffered`` (W7-2) all produce 0.000 —
   the **first strict separation** between cross-role corroboration
   and W7-2 single-tag plurality, **+1.000** vs both baselines,
   stable across **5/5** alternate bank seeds. The W8 family
   (W8-1 / W8-2 / W8-3 / W8-4 — proved or proved-empirical)
   anchors the milestone formally; the W8-C family makes the
   multi-service / decoder-side / real-LLM extensions falsifiable.
   **Honest scope:** the win is *conditional* on the named bench
   property (decoy-plurality + cross-role-corroborated gold);
   W8-4 is the named falsifier regime where the decoy is
   corroborated and corroboration ties FIFO at 0.000;
   W8-3 backward-compat preserves W7-2 on Phase 54 (corroboration
   matches cohort_buffered at 1.000); no regression on Phase 53
   synthetic (0.800) or Phase 53 14B real-LLM (0.800). The
   milestone clears the **strong success bar** of
   `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 — a
   strict gain ≥ 0.20 on Phase 55, stable across ≥ 3 seeds, no
   regression on Phase 53 / Phase 54, audit T-1..T-7 preserved
   on every cell, named bench property + named falsifier regime.

6. **Model-scale vs capsule-structure on multi-agent
   coordination** — *active, settled (SDK v3.7)*. **Phase-53**
   benchmark (`vision_mvp/experiments/phase53_scale_vs_structure.py`)
   replaces the Phase-52 deterministic producer-role extractor
   with a real-LLM extractor and decomposes ``accuracy_full``
   across (model regime × admission strategy). Headline (n=5,
   K_auditor=4): every fixed admission strategy (substrate /
   capsule_fifo / capsule_priority / capsule_coverage) achieves
   ``accuracy_full = 0.800`` in every model regime; only
   ``capsule_learned`` varies (0.400 on synthetic and 14B; 0.800
   on 35B). ``structure_gain`` is **non-positive at every
   regime** (-0.4 / -0.4 / 0.0); ``scale_gain[capsule_learned]
   = +0.4``, ``scale_gain[fixed] = 0.0``. **W4-C1 is conditionally
   falsified** out-of-distribution on the real-LLM regime
   (capsule_learned underperforms FIFO by 0.4 on synthetic and
   14B; ties at 35B). Honest reading: scale closes a *structure
   deficit* (created by OOD over-rejection of clean candidates
   by the SDK v3.5 learned policy), not a *structure surplus*.
   The capsule layer's load-bearing contribution at this
   benchmark is the **lifecycle audit (T-1..T-7, 60/60 across
   regimes)**, not admission policy gains. The W6 family
   (W6-1/2/3/4 proved + mechanically-checked + empirically-
   saturated) anchors the milestone formally; the W6-C family
   (W6-C1/C2 falsified-empirical, W6-C3 positive, W6-C4/C5
   conjectural) makes the empirical reading falsifiable.

## Current frontier (SDK v3.21, 2026-04-29)

### Active moves (SDK v3.21 — outside-witness acquisition disambiguator + R-67 outside-information benchmark family + W20 family — *first capsule-native method that crosses the W19-Λ-outside wall on a regime where it applies*)

- **W20-Λ-outside-extension** (proved-empirical n=8 saturated +
  structural sketch) — W19-Λ-outside extends verbatim to
  R-67-OUTSIDE-REQUIRED-BASELINE (no oracle / abstaining oracle):
  every capsule strategy through W19 ties FIFO at ``accuracy_full
  = 0.000``. The wall is real for every closed-form bundle-only
  scorer.
- **W20-1** (proved-conditional + proved-empirical n=80 saturated
  across 5 seeds × 2 budgets, also n=12) — pairing W19 with the
  new ``OutsideWitnessAcquisitionDisambiguator`` over a
  deterministic ``ServiceGraphOracle`` strictly improves
  ``accuracy_full`` over every non-W20 capsule baseline (incl.
  W19) by **+1.000** on R-67-OUTSIDE-RESOLVES-LOOSE AND
  R-67-OUTSIDE-RESOLVES-TIGHT, stable across 5/5 ``bank_seed``
  values. The first capsule-native multi-agent-coordination
  method that crosses the W19-Λ-outside wall on a regime where
  the wall actually applies.
- **W20-2** (proved by inspection + mechanically-checked) —
  Determinism + closed-form correctness; positive-set projection
  rule; ``n_outside_tokens`` recorded as strict additional cost;
  W15 ``tokens_kept`` byte-for-byte unchanged from W19.
- **W20-3** (proved-empirical full programme regression; 585/585
  coordpy tests pass, 545 pre-existing + 40 new W20 tests) —
  Backward-compat with R-54..R-66 default banks; W20 reduces to
  W19 byte-for-byte either via no-trigger or outside-abstained.
  With ``enabled = False`` it reduces to W19 byte-for-byte.
- **W20-Λ-none** (proved-empirical n=8 saturated) —
  ``AbstainingOracle`` ⇒ W20 ties FIFO at 0.000.
- **W20-Λ-compromised** (proved-empirical n=8 saturated) —
  ``CompromisedServiceGraphOracle`` ⇒ W20 trusts decoy and FAILS
  at 0.000.
- **W20-Λ-joint-deception** (proved-empirical n=8 saturated) —
  primary + secondary + oracle all favour decoy ⇒ W20 ties W19
  at 0.000. Names the structural limit when *all* evidence
  channels are jointly compromised.
- **W20-Λ-real** (proved-conditional + empirical-research n=4 ×
  2 models on Mac-1 Ollama) — ``mixtral:8x7b`` 47B-MoE achieves
  ``acc_full = 0.750`` (+0.750 over W19); ``qwen2.5-coder:7b``
  ties FIFO at 0.000. Cross-model split: scale + general
  knowledge correlates with W20-Λ-real escape.
- **Two-Mac status** — Mac 2 (192.168.12.248) still ARP
  ``incomplete``. **No two-Mac sharded inference happened in SDK
  v3.21.** The W20 ``OutsideWitnessOracle`` Protocol is
  infrastructure-ready for cross-host deployment when Mac 2
  returns; the ``MLXDistributedBackend`` adapter is byte-for-byte
  unchanged.

### Discharged / partially-discharged conjectures (SDK v3.21)

- **W19-C-OUTSIDE** (SDK v3.20): **PARTIALLY DISCHARGED-empirical**
  by W20-1 on the *bundle-only-insufficient-but-outside-resolvable*
  direction — the closed-form ``OutsideWitnessAcquisitionDisambiguator``
  with a registered ``ServiceGraphOracle`` strictly beats every
  non-W20 capsule baseline including W19 by ``+1.000`` on
  R-67-OUTSIDE-RESOLVES across 5/5 seeds. The W19-Λ-outside wall
  remains real wherever the bundle is exhausted of asymmetric
  signal AND no outside source is registered (W19-Λ-outside
  default direction); remains a conjecture on the
  *joint-deception* direction (named W20-Λ-joint-deception:
  primary + secondary + all registered oracles compromised
  jointly). The W19-Λ-total wall remains real unless a registered
  oracle produces an asymmetric reply on the empty-bundle
  ambiguity (W20-1 on R-67-OUTSIDE-NONE-style total absence falls
  through to W19-Λ-total).

## Previous frontier (SDK v3.20, 2026-04-28)

### Active moves (SDK v3.20 — bundle-contradiction-aware trust-weighted disambiguator + deceptive-ambiguity benchmark family + W19 family — *first capsule-native move beyond the W18-Λ-deceive wall on the bundle-resolvable case*)

- **Phase-66 deceptive-ambiguity benchmark family.**
  ``vision_mvp.experiments.phase66_deceptive_ambiguity`` ships
  five pre-committed sub-banks (one positive sanity anchor + two
  W19-1 strict-gain anchors + two named falsifiers) plus a cross-
  regime synthetic summary:
  (i) ``r66_corroborated`` — primary names gold; secondary names
  gold. W18 = W19 = 1.000. W19-3 backward-compat anchor.
  (ii) ``r66_deceive_naive_loose`` (``T_decoder = None``) — primary
  names DECOY ONLY; secondary names GOLD ONLY. W18 abstains, ties
  FIFO at 0.000; W19 fires the confound-resolved branch (witness-
  inversion semantics) and projects to gold at 1.000. **+1.000
  strict separation.** The W19-1 anchor.
  (iii) ``r66_deceive_naive_tight`` (``T_decoder = 24``) — same
  shape under decoder-side budget pressure. W19 + W15 composition
  preserves bounded-context efficiency byte-for-byte relative to
  W18; gap +1.000.
  (iv) ``r66_confound_resolvable`` — primary names ALL three;
  secondary names gold. W18 abstains; W19 picks strict-max-witness
  subset = {gold} at 1.000. **+1.000 strict separation.**
  (v) ``r66_deceive_total`` — primary names DECOY ONLY; *no*
  secondary witness. W19-Λ-total falsifier: W19 reduces to W18
  and FAILS at 0.000.
  (vi) ``r66_outside_required`` — primary names DECOY ONLY;
  secondary names ALL three (symmetric witnesses). W19-Λ-outside
  falsifier: W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``
  and ties FIFO at 0.000.

- **W19 family minted.** W19-Λ-deceive-extension (proved-empirical
  + structural sketch; W18-Λ-deceive extends to R-66-DECEIVE-NAIVE
  for every closed-form scorer that trusts its concatenated
  disambiguator text), W19-1 (proved-conditional + proved-empirical
  n=120 saturated across 5 seeds × 3 regimes; the first capsule-
  native method to resolve bundle-internal contradiction between
  primary and witnesses), W19-2 (proved by inspection +
  mechanically-checked; W19 determinism + closed-form correctness),
  W19-3 (proved-empirical full programme regression; backward-
  compat with R-54..R-65 byte-for-byte), W19-Λ-total / -outside
  (proved-empirical n=8 saturated each; two named structural
  limits), W19-C-LEARNED (conjectural; learned trust scorer for
  free-form witnesses), W19-C-OUTSIDE (conjectural; outside-
  information axis to escape both falsifier walls), W19-Λ-real
  (proved-conditional + empirical-research; real-LLM transfer is
  conditional on closed-vocabulary secondary-witness emission),
  W19-C-CROSS-BENCH (conjectural; transfer to non-incident-triage
  benchmark families with a canonical-role-for-kind mapping).

- **Bundle-contradiction-aware trust-weighted disambiguator** (new
  SDK surface, purely additive). ``vision_mvp/coordpy/team_coord.py``
  ships:
  * :class:`BundleContradictionDisambiguator` — the W19 four-stage
    pipeline (inner W18 decode + canonical primary identification +
    asymmetric witness counting + branch-decision projection).
  * :class:`W19TrustResult` — the per-decode audit record carrying
    the projected answer, the W18 fall-through answer, the
    canonical primary's payload, the per-tag W18 scores, the
    per-tag witness counts, and the W19 branch label.
  * :func:`_w19_canonical_primary_index` — closed-form
    deterministic primary identifier with canonical-role-for-kind
    tiebreak + raw-kind tiebreak.
  * :func:`_w19_witness_counts` — closed-form deterministic
    witness counter excluding the canonical primary.
  * :data:`_INCIDENT_TRIAGE_CANONICAL_ROLE_FOR_KIND` — closed-
    vocabulary canonical-role-for-kind table for the incident-
    triage benchmark family.
  * :data:`W19_SYMMETRIC_NOISE_KINDS` — the round-1 generic-
    noise kinds explicitly excluded from witness counting.
  * :data:`W19_BRANCH_*` — closed-vocabulary branch labels
    (``primary_trusted`` / ``inversion`` / ``confound_resolved``
    / ``abstained_no_signal`` / ``abstained_symmetric`` /
    ``disabled``).

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-66 anchor +
  bar 16 — deceptive-ambiguity bundle-contradiction split + § 2.15
  R-66 ingredients). The SDK v3.20 result clears the **strong
  success bar** § 1.1 on R-66-DECEIVE-NAIVE / R-66-CONFOUND-
  RESOLVABLE (strict gain +1.000 vs every non-W19 capsule
  baseline; bench property held in 8/8; named falsifier regimes
  W19-Λ-total / W19-Λ-outside; W19-3 backward-compat preserved
  byte-for-byte). Headline data files:
  ``docs/data/phase66_cross_regime_synthetic.json``,
  ``docs/data/phase66_seed_sweep_deceive_naive_loose.json``,
  ``docs/data/phase66_seed_sweep_deceive_naive_tight.json``,
  ``docs/data/phase66_seed_sweep_confound_resolvable.json``.

- **Honest scope.** The W19-1 win is *strongly conditional* on
  (a) the symmetric-corroboration round-1 property (so
  W17-Λ-symmetric still applies and only an additional
  information channel can win), AND (b) the bundle carrying at
  least one *independent asymmetric witness* (a specific-tier
  handoff OTHER than the canonical primary disambiguator) whose
  payload mentions a service tag asymmetrically across the
  candidate set. R-66 is a *synthetic* regime; the producer is
  :class:`IdentityExtractor`. Real-LLM transfer (W19-Λ-real) is
  conditional on closed-vocabulary secondary-witness emission;
  free-form natural-language witnesses fall outside the W19
  exact-match closure. The CoordPy single-run product runtime
  contract is byte-for-byte unchanged.

### Active observations (SDK v3.20)

- **W18-Λ-deceive holds on R-66-DECEIVE-NAIVE for every pre-W19
  scorer including W18 itself.** W18's selector concatenates
  primary + secondary payloads → its per-tag scorer sees positive
  scores on every admitted tag → strict-asymmetric branch cannot
  fire (N = U) → abstain → fall through to empty inner W15
  answer. The wall is real and structural for every scorer that
  does not distinguish the canonical primary from secondary
  witnesses.

- **W19's confound-resolved branch is the load-bearing branch on
  R-66-DECEIVE-NAIVE and R-66-CONFOUND-RESOLVABLE.** Both regimes
  trigger W18 abstention (full-set hit); W19's witness counter
  (excluding the primary) sees aw(gold) > aw(decoy) and projects
  to gold via the strict-max-aw subset. The inversion branch
  (``W19_BRANCH_INVERSION``) is reserved for regimes where W18
  fires its strict-asymmetric branch but the complement has more
  witnesses than the named set — R-66 does not exercise this
  branch directly because W18 abstains on the full-set hit; the
  inversion branch is structurally available for future regimes.

### Active conjectures (SDK v3.20)

- **W19-C-LEARNED** (free-form witness transfer): a small
  distilled trust scorer outperforms the closed-form witness-
  count rule on free-form natural-language secondary witnesses.
  **Conjectural.**
- **W19-C-OUTSIDE** (outside-information escape from W19-Λ-total
  AND W19-Λ-outside): a scorer with access to an outside-
  information axis (service-graph topology, prior reliability
  scores, cross-incident historical evidence) can detect both
  falsifier walls by cross-reference. **Conjectural.**
- **W19-Λ-real** (real-LLM transfer): partially-discharged when
  the LLM emits closed-vocabulary secondary witnesses from non-
  canonical roles; conjectural otherwise.
- **W19-C-CROSS-BENCH** (cross-bench transfer): the W19 method
  generalises to non-incident-triage families with a canonical-
  role-for-kind mapping. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.20)

- **W18-Λ-deceive** (SDK v3.19): **PARTIALLY DISCHARGED-empirical**
  by W19-1 in the *bundle-resolvable* direction. Closed-form
  bundle-only scorers can escape adversarial round-2 evidence
  AS LONG AS the bundle carries an independent asymmetric
  witness. The bundle-only walls (W19-Λ-total / W19-Λ-outside)
  remain real; escape requires outside information.
- **W18-Λ-confound** (SDK v3.19): **PARTIALLY DISCHARGED-empirical**
  by W19-1 in the *bundle-resolvable* direction on
  R-66-CONFOUND-RESOLVABLE.

---

## Previous frontier (SDK v3.19, 2026-04-28)

### Active moves (SDK v3.19 — bundle-relational compatibility disambiguator + symmetric-ambiguity benchmark family + W18 family — *first capsule-native move beyond the W17-Λ-symmetric wall on a regime where it applies*)

- **Phase-65 relational-compatibility disambiguation under
  symmetric-corroboration benchmark family.**
  ``vision_mvp.experiments.phase65_relational_disambiguation``
  ships four pre-committed sub-banks (one positive-anchor + three
  named falsifiers) plus a cross-regime synthetic summary:
  (i) ``r65_compat_loose`` — synthetic identity producer, R-65-COMPAT
  bench, ``T_decoder = None``. The W18-1 anchor: W18 = 1.000;
  every other capsule strategy = 0.000. **+1.000 strict separation**.
  (ii) ``r65_compat_tight`` — same regime under
  decoder-side budget pressure ``T_decoder = 24``. W18 composes
  cleanly with W15 attention-aware pack; ``tokens_kept_sum`` is
  byte-for-byte identical to W15's. **+1.000 strict separation.**
  (iii) ``r65_no_compat`` — W18-Λ-no-compat falsifier. Round-2
  disambiguator carries no service-tag mention; W18 abstains;
  ties FIFO at 0.000.
  (iv) ``r65_confound`` — W18-Λ-confound falsifier. Round-2
  disambiguator mentions both gold AND decoy; W18 abstains;
  ties FIFO at 0.000.
  (v) ``r65_deceive`` — W18-Λ-deceive falsifier. Round-2
  disambiguator mentions decoy but NOT gold; W18 trusts its
  evidence and picks decoy; fails at 0.000.

- **W18 family minted.** W18-Λ-sym (proved-empirical n=8 saturated
  × 5 seeds + structural sketch; W17-Λ-symmetric extends to R-65-
  COMPAT verbatim for every method pre-W18), W18-1 (proved-
  conditional + proved-empirical n=40 saturated across 5 seeds × 2
  budgets; the first capsule-native method to cross the
  symmetric-corroboration wall), W18-2 (proved by inspection +
  mechanically-checked; W18 determinism + closed-form correctness),
  W18-3 (proved-empirical full programme regression; backward-
  compat with R-54..R-64 byte-for-byte), W18-Λ-no-compat /
  -confound / -deceive (proved-empirical n=8 saturated each;
  three named structural limits), W18-C-LEARNED (conjectural;
  learned scorer for free-form relational mentions),
  W18-C-OUTSIDE (conjectural; outside-information axis to detect
  deceptive mentions), W18-Λ-real (proved-conditional + empirical-
  research; real-LLM transfer is conditional on closed-vocabulary
  relational compounds), W18-C-CROSS-BENCH (conjectural; transfer
  to non-incident-triage benchmark families).

- **Bundle-relational compatibility disambiguator** (new SDK
  surface, purely additive). ``vision_mvp/coordpy/team_coord.py``
  ships:
  * :class:`RelationalCompatibilityDisambiguator` — the W18 four-
    stage pipeline (inner W15 decode + disambiguator selection +
    tokenise + score + project).
  * :class:`W18CompatibilityResult` — the per-decode audit record
    carrying the projected answer, the inner answer, the
    disambiguator payload, the per-tag scores, and the abstention
    flag.
  * :func:`_disambiguator_payload_tokens` — closed-form
    deterministic tokeniser (lower-case, split on non-identifier
    chars, compound identifiers preserved).
  * :func:`_relational_compatibility_score` — closed-form
    deterministic scorer with contiguous-subsequence semantics
    for compound targets (handles ``db_query`` matching inside
    ``svc_web_then_svc_db_query``).

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-65 anchor +
  bar 15 — relational-compatibility-disambiguation under
  symmetric-corroboration split + § 2.14 R-65 ingredients). The
  SDK v3.19 result clears the **strong success bar** § 1.1 on
  R-65-COMPAT-LOOSE (strict gain +1.000 vs every non-W18 capsule
  baseline; bench property held in 8/8; named falsifier regimes
  W18-Λ-no-compat / -confound / -deceive AND named bench-shape
  conditionality W18-Λ-sym; W18-3 backward-compat preserved
  byte-for-byte). Headline data files:
  ``docs/data/phase65_cross_regime_synthetic.json``,
  ``docs/data/phase65_seed_sweep_loose.json``,
  ``docs/data/phase65_seed_sweep_tight.json``.

- **Honest scope.** The W18-1 win is *strongly conditional* on
  (a) the symmetric-corroboration bench property (R-65-COMPAT),
  (b) the round-2 disambiguator's payload carrying a relational-
  compound mention of *every* gold service tag AND *no* decoy
  service tag, AND (c) the relational-mention convention being
  closed-vocabulary (the synthetic bench's exact-match closure).
  Real-LLM transfer (W18-Λ-real) is conditional on the LLM
  emitting closed-vocabulary relational compounds. The CoordPy
  single-run product runtime contract is byte-for-byte unchanged.

### Active observations (SDK v3.19)

- **W18-Λ-sym holds on R-65-COMPAT.** Every closed-form salience
  scorer in the SDK ties FIFO at ``accuracy_full = 0.000`` on
  R-65-COMPAT (loose AND tight). The bipartite ``(role × tag,
  kind, magnitude)`` multiset is identical for gold and decoy by
  construction; only the round-2 disambiguator's payload-text
  channel breaks the tie.

- **W18-3 partial recovery on R-64-SYM.** Of the 8 R-64-SYM
  scenarios, only the 2 deadlock-flavored ones carry a round-2
  relational mention (``relation=orders_payments_join``). On
  those, W18 recovers gold; on the others, W18 abstains and ties
  FIFO. R-65-COMPAT generalises the relational-mention convention
  to all four scenario families (deadlock / pool / disk /
  slow_query) so the W18-1 strict gain is uniform.

### Active conjectures (SDK v3.19)

- **W18-C-LEARNED** (free-form relational-mention transfer): a
  small distilled bundle-relational scorer outperforms the
  closed-form rule on free-form natural-language relational
  mentions. **Conjectural.**
- **W18-C-OUTSIDE** (outside-information escape from W18-Λ-deceive):
  a scorer with access to an outside-information axis can detect
  the W18-Λ-deceive regime by cross-reference. **Conjectural.**
- **W18-Λ-real** (real-LLM transfer): partially-discharged when
  the LLM emits closed-vocabulary relational compounds; conjectural
  otherwise.
- **W18-C-CROSS-BENCH** (cross-bench transfer): the W18 method
  generalises to non-incident-triage families with a closed-
  vocabulary relational-mention convention. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.19)

- **W17-C-DISAMBIGUATOR** (SDK v3.18; "a learned or LLM-distilled
  semantic disambiguator beyond the closed-form capsule surface
  could distinguish ``orders_payments_join`` (gold A_B in
  deadlock) from a generic decoy whose round-1 mentions are
  observationally identical"). **DISCHARGED-empirical in the
  closed-form direction** by W18-1: a deterministic training-free
  bundle-relational scorer is sufficient on R-65-COMPAT (loose
  AND tight). The *learned* clause remains conjectural under
  W18-C-LEARNED (relevant when free-form mentions fall outside
  the closure).

---

## Previous frontier (SDK v3.18, 2026-04-27)

### Active moves (SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem + W17 family — *first fresh-live end-to-end real-LLM strict +1.000 advance + first explicit symmetric-corroboration negative theorem*)

- **Phase-64 fresh-live end-to-end composition + symmetric-
  corroboration wall benchmark.**
  ``vision_mvp.experiments.phase64_live_composition`` ships seven
  pre-committed sub-banks plus a falsifier and a cross-model cell:
  (i) ``r64_baseline`` — synthetic identity producer + magnitude-
  hinted prompt + ``T_decoder = None``. Sanity anchor: every
  cross-round capsule decoder hits 1.000.
  (ii) ``r64_w14h_only`` — synthetic mag-filter producer +
  magnitude-hinted prompt + ``T_decoder = None``. Synthetic
  counterpart of the W17-1 anchor. Bench property holds in 8/8;
  cross-round decoders hit 1.000 (W17-3 backward-compat).
  (iii) ``r64_live_struct`` — fresh live ``qwen2.5:14b-32k`` +
  legacy *structured* prompt + ``T_decoder = 14``. Reproduces the
  W14-Λ-real envelope on the fresh probe: bench property holds in
  7/8; ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.500 strict gain**.
  W17-Λ-no-hint structural anchor.
  (iv) ``r64_live_maghint`` — **the W17-1 anchor.** Same fresh
  live probe + magnitude-hinted prompt + ``T_decoder = 14``.
  Bench property holds in 8/8 (closing the 1/8 model-side miss);
  ``capsule_attention_aware = 1.000``;
  ``capsule_layered_fifo_packed = 0.000``;
  ``capsule_fifo = 0.000``; **+1.000 strict separation** on both
  axes. The first programme result that beats the strongest
  non-composed baseline by ≥ 1.0 on a fresh live LLM probe.
  (v) ``r64_live_naive`` — same fresh probe + naive prompt +
  ``T_decoder = 14``. Live counterpart of W14-Λ-prompt +
  W15-Λ-budget joint failure. Bench property holds in 0/8; every
  capsule strategy ties FIFO at 0.000. W17-Λ-naive falsifier.
  (vi) ``r64_live_xmodel`` — fresh live ``qwen3.5:35b`` MoE
  backend + magnitude-hinted prompt + ``T_decoder = 14``,
  ``think = False``. Cross-model probe (W17-C-XMODEL). Bench
  property holds in 8/8; ``capsule_attention_aware = 0.750``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.750 strict gain**
  (well above the 0.50 strong-bar threshold). The 0.250 gap to
  1.000 is on ``accuracy_root_cause`` — a 35B-specific specific-
  kind judgment artifact, not a producer-protocol failure.
  Proved-conditional + empirical-research.
  (vii) ``r64_sym_loose`` and ``r64_sym_tight`` — synthetic
  symmetric-corroboration bank (every service mentioned by
  exactly 2 distinct routed producer roles via generic-noise
  kinds; round-2 disambiguator names gold root_cause without
  ``service=`` token); under both ``T_decoder ∈ {None, 24}``,
  every capsule strategy in the SDK ties FIFO at 0.000 — the
  **first explicit symmetric-corroboration limit theorem in the
  programme** (W17-Λ-symmetric).

- **W17 family minted.** W17-1 (proved-conditional +
  empirical-research; the first fresh-live end-to-end +1.000
  strict gain), W17-Λ-no-hint (empirical-research; live
  legacy-structured-prompt envelope), W17-Λ-naive (empirical-
  research; live joint-failure falsifier), **W17-Λ-symmetric**
  (proved-empirical + structural sketch; first explicit
  symmetric-corroboration limit theorem; *discharges*
  W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative theorem),
  W17-2 (proved + mechanically-checked; magnitude-hinted prompt
  determinism + threshold table soundness), W17-3 (proved-
  empirical full programme regression; the W17 surface reduces
  to the SDK v3.15 W14 anchor byte-for-byte under default
  parameters; 442/442 prior tests pass), **W17-C-XMODEL**
  (proved-conditional + empirical-research; fresh live 35B
  cross-model strict gain). The W17-C family (W17-C-DISAMBIGUATOR,
  W17-C-LEARNED-HINT, W17-C-CROSS-BENCH) makes the next research
  frontier explicit.

- **Magnitude-hinted producer protocol** (new SDK surface, purely
  additive). ``vision_mvp/coordpy/team_coord.py`` ships:
  * ``PRODUCER_PROMPT_MAGNITUDE_HINTED`` — third producer-prompt
    mode; the W17-1 anchor.
  * :class:`OperationalThreshold` — closed-vocabulary record
    naming a kind, the qualifying field, the inclusive
    threshold, the unit, and a human gloss.
  * ``RoleExtractionSchema.magnitude_thresholds`` — additive
    optional field on the W14 schema; empty by default (W17-3
    byte-for-byte backward-compat); populated by
    ``incident_triage_role_schemas(magnitude_hinted=True)``.
  * :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` and
    :func:`incident_triage_magnitude_thresholds` — the
    pre-committed threshold table for the incident-triage family
    (calibrated to the synthetic
    :class:`MagnitudeFilteringExtractor`'s default thresholds,
    NOT to any specific scenario's magnitudes).
  * :func:`_render_magnitude_hinted_prompt` — the W17 prompt
    renderer. Adds the operational threshold table AND an
    anti-relative-magnitude clause to the structured prompt.

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-64 anchor +
  bar 14 — live-end-to-end + magnitude-hinted-protocol +
  symmetric-corroboration-wall split + § 2.13 R-64 ingredients).
  The SDK v3.18 result clears the **strong success bar** § 1.1 on
  R-64-LIVE-MAGHINT (strict gain +1.000 vs both substrate FIFO
  AND FIFO-packed-W14H-only on a fresh live LLM probe; bench
  property held in 8/8 closes the prior 1/8 miss; named bench
  property + named falsifier regimes W17-Λ-no-hint /
  W17-Λ-naive / W17-Λ-symmetric AND named cross-model probe
  W17-C-XMODEL with +0.750 gain; W17-3 backward-compat
  preserved byte-for-byte). Headline data files:
  ``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``,
  ``docs/data/phase64_cross_regime_synthetic.json``.

- **Honest scope.** The W17-1 win is *strongly conditional* on
  (a) the asymmetric-corroboration bench property, (b) the
  magnitude-hint table being calibrated to the synthetic
  extractor's threshold values (operational definitions, not
  answer hints — both gold and decoy magnitudes are well above
  every threshold), AND (c) the live endpoint reachable.
  W17-Λ-symmetric *names the wall* when the asymmetric ingredient
  is structurally absent: every capsule strategy in the SDK ties
  FIFO at 0.000 by construction, including the W14H + W15
  composition. The cross-model probe (W17-C-XMODEL) is
  proved-conditional on bench-property + strict-gain transfer; the
  saturated full-correctness clause remains conjectural and is
  bounded by a 35B-specific specific-tier judgment artifact (not
  a producer-protocol failure). The CoordPy single-run product
  runtime contract is byte-for-byte unchanged.

### Active observations (SDK v3.18)

- **W17-Λ-real (honest positive on fresh live axis).** Real
  Mac-1 ``qwen2.5:14b-32k`` at ``temperature = 0`` on the
  Phase-61 redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (live confirmation of W14-Λ-prompt +
    W15-Λ-budget joint failure on the live axis).
  - Under structured prompt: bench property holds 7/8;
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +0.500`` strict gain (W17-Λ-no-hint anchor — reproduces
    W14-Λ-real / W16-Λ-real-replay envelope on fresh probe).
  - Under magnitude-hinted prompt: bench property holds 8/8;
    cross-round decoders all achieve ``accuracy_full = 1.000``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +1.000`` strict gain (W17-1 anchor).
  - The 1/8 model-error failure on the slow_query_archival
    scenario (LLM judged decoy magnitudes "not severe enough"
    relative to gold) is **closed** by the W17 magnitude-hint
    extension; the LLM correctly emits every qualifying event
    once the absolute thresholds are stated and relative-
    magnitude skipping is forbidden in the prompt.

- **W17-C-XMODEL-real (proved-conditional cross-model
  bench-property transfer).** Real Mac-1 ``qwen3.5:35b`` MoE
  backend at ``temperature = 0`` with ``think = False``: bench
  property holds in 8/8 under the magnitude-hinted prompt (the
  bench-property hold-rate transfers to a 2.4× larger MoE
  model class byte-for-byte); ``capsule_attention_aware =
  0.750``; ``capsule_layered_fifo_packed = 0.000``; +0.750
  strict gain. The 0.250 gap to 1.000 is on
  ``accuracy_root_cause`` — the 35B emits a different specific-
  tier kind on one scenario than the 14B does (a model-side
  specific-kind judgment artifact, not a producer-protocol
  failure). The W5-1 (SDK v3.6) prior cross-model probe found
  parser-axis TVD = 1.000 between 14B and 35B under strict
  parsing; W17-C-XMODEL finds *bench-property-axis* TVD = 0.000
  under the magnitude-hinted prompt + robust parser — the W17
  protocol *normalises* the producer-side behaviour across the
  model-class jump that the W5-1 parser-boundary regime
  exposed.

### Active conjectures (SDK v3.18)

- **W17-C-DISAMBIGUATOR** (semantic-disambiguator beyond
  W17-Λ-symmetric): a learned or LLM-distilled disambiguator
  consuming the round-2 disambiguator's evidence text directly
  could distinguish ``orders_payments_join`` (gold A_B in
  deadlock) from a generic decoy whose round-1 mentions are
  observationally identical. **Conjectural; the named research
  frontier for SDK v3.19+.**

- **W17-C-LEARNED-HINT** (learned threshold variant): a per-kind
  data-driven threshold table outperforms the closed-form
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` on a
  held-out test set across 5/5 random splits. **Conjectural.**

- **W17-C-CROSS-BENCH** (cross-bench transfer): the magnitude-
  hint protocol generalises to non-incident-triage benchmark
  families when the family admits a closed-vocabulary kind →
  qualifying-threshold mapping. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.18)

- **W16-C-LIVE-OLLAMA** (SDK v3.17; "a fresh live Ollama probe
  closes the 1/8 model-error failure W14-only leaves").
  **DISCHARGED-empirical** by W17-1: a fresh live probe AT
  THE MAGNITUDE-HINTED PROMPT closes the 1/8 miss entirely
  (bench property holds in 8/8;
  ``accuracy_full = 1.000``); the W17-Λ-no-hint anchor on the
  same fresh probe under the legacy structured prompt
  reproduces the recorded 7/8 + 0.500 envelope, confirming the
  magnitude-hint extension (W17-1) is the load-bearing
  improvement, not a re-run of the same prompt.
- **W16-C-CROSS-MODEL** (SDK v3.17; "the composition's W16-1
  win at qwen2.5:14b-32k transfers to qwen3.5:35b under
  MLX-distributed inference"). **PARTIALLY DISCHARGED-
  empirical** by W17-C-XMODEL: the bench-property + strict-
  gain transfer is observed on the 35B Ollama backend (NOT
  MLX-distributed); the saturated full-correctness clause is
  still bounded by a 35B-specific judgment artifact; the
  MLX-distributed clause remains orthogonal and conjectural
  pending Mac-2 reachable.
- **W15-C-SYMMETRIC** / **W16-C-SYMMETRIC** (SDK v3.16 / v3.17;
  "symmetric-corroboration multi-hypothesis structural limit").
  **DISCHARGED as a negative theorem** by W17-Λ-symmetric: the
  R-64-SYM bench construction proves every capsule strategy in
  the SDK ties FIFO at 0.000 under both ``T_decoder ∈ {None,
  24}`` by construction. The wall is real and structural; the
  named research move beyond it is W17-C-DISAMBIGUATOR
  (conjectural).

## Previous frontier (SDK v3.17, 2026-04-27)

### Active moves (SDK v3.17 — end-to-end W14 + W15 composition + W16 family — *first end-to-end real-LLM strict advance over the strongest non-composed baseline*)

(See `docs/RESULTS_COORDPY_COMPOSED_REAL_LLM.md` for the SDK
v3.17 milestone note. SDK v3.18 supersedes this frontier on the
fresh-live and symmetric-wall axes; the W16 anchors are
preserved as historical evidence.)

## Previous frontier (SDK v3.16, 2026-04-27)

### Active moves (SDK v3.16 — attention-aware capsule context packing + W15 family — *first decoder-side context-packing move + joint-correctness-and-context-efficiency strict-gain anchor*)

- **Phase-62 attention-aware capsule context packing benchmark.**
  ``vision_mvp.experiments.phase62_attention_aware_packing`` ships
  three pre-committed sub-banks:
  (i) ``T_decoder=None`` — the R-62-default sanity anchor; every
  cross-round decoder (W11/W12/W13/W15) saturates at 1.000. The W15
  decoder ties W13 byte-for-byte on the answer field (W15-3
  backward-compat).
  (ii) ``T_decoder=24`` — the R-62-tightbudget W15-1 anchor; under
  FIFO packing the round-2 specific-tier disambiguator falls past
  the budget and ``capsule_layered_fifo_packed`` ties FIFO at
  ``accuracy_full = 0.000``. The W15
  :class:`AttentionAwareBundleDecoder` salience-packs with hypothesis
  preservation and achieves ``accuracy_full = 1.000``; +1.000 strict
  separation, stable across 5/5 alternate ``bank_seed`` values.
  (iii) ``T_decoder=9999`` — the R-62-saturation W15-Λ-budget
  falsifier; under no decoder-side budget pressure the salience
  reordering is a no-op on the answer field — both W15 and FIFO-pack
  tie at the W13 answer (1.000). The W15-1 win is *conditional* on
  budget pressure; this regime makes the conditionality sharp.
- **``AttentionAwareBundleDecoder`` + ``CapsuleContextPacker`` +
  ``FifoContextPacker`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. A two-stage decoder: (1)
  first-pass priority decode over the normalised union elects a
  tentative ``root_cause``; (2) salience-aware repack reorders +
  truncates the union under ``T_decoder`` (using closed-form weights
  on tier + CCK + corroboration + magnitude + round index, with
  ``preserve_hypotheses=True`` defaulting on) before final W13
  layered decode. Pack-stats expose ``position_of_first_causal_claim``
  (the proxy attention metric), ``tokens_kept_sum`` /
  ``tokens_input_sum``, ``hypothesis_count_kept``, and
  ``n_dropped_budget`` for direct audit. ``FifoContextPacker`` is the
  load-bearing baseline (FIFO truncation under the same
  ``T_decoder``). Re-exported via ``__all__``.
- **Theorem family W15.** W15-Λ-budget (decoder-side budget
  structural limit on R-62-tightbudget under FIFO packing,
  proved-empirical n=40 saturated × 5 seeds + structural sketch via
  W7-3 extension to the decoder-side axis),
  W15-1 (AttentionAwareBundleDecoder sufficiency under bounded
  ``T_decoder`` with hypothesis preservation, proved-conditional +
  proved-empirical synthetic n=40 saturated × 5 seeds, +1.000 vs
  fifo_packed_layered), W15-2 (pack determinism + closed-form
  salience, proved by inspection + mechanically-checked), W15-3
  (backward-compat with R-54..R-61 default banks, proved-empirical
  full programme-wide regression 393/393 + 37 new tests = 430/430),
  W15-Λ-degenerate (saturation falsifier on R-62-saturation,
  proved-empirical n=8: under no decoder-side budget pressure the
  W15-1 win is structurally invisible by construction), W15-4
  (token-efficiency floor: ``tokens_kept ≤ T_decoder`` strict, proved
  by inspection + mechanically-checked). The W15-C family (W15-C-real,
  W15-C1, W15-C-LEARNED, W15-C-SYMMETRIC, W15-C-COMPOSE-W14) makes
  real-LLM-downstream-decoder, cross-bench, learned-salience,
  symmetric-corroboration, and W14+W15 compose extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-62 anchor +
  bar 12 — joint-correctness-and-context-efficiency split + § 2.11
  R-62 ingredients). The SDK v3.16 result clears the **strong
  success bar** § 1.1 on R-62-tightbudget synthetic (strict gain
  +1.000 vs FIFO-packed-W13, stable across 5/5 (bank_seed) values,
  no regression on R-53..R-61, audit T-1..T-7 preserved on every
  cell, named bench property + named falsifier regime
  W15-Λ-degenerate, AND joint-correctness-and-context-efficiency
  split bar 12 satisfied — the new method includes a load-bearing
  decoder-side context-packing intervention beyond every prior
  layer). Headline data file:
  ``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``.
- **Honest scope.** R-62 is a *synthetic* milestone — the producer
  is :class:`IdentityExtractor`, not a real LLM. Real-LLM transfer
  of W15 is W15-C-real, conjectural; it requires Mac 1 / Mac 2 to
  be online and the bundle to be re-decoded by an LLM agent under a
  real context window. SDK v3.16 does not run this probe.
  "Attention-aware" uses an *honest proxy* — the
  ``position_of_first_causal_claim`` metric — not transformer
  attention manipulation. The W15-1 win is *conditional* on (a) the
  bench property holding, (b) ``T_decoder`` below the union token
  sum, AND (c) round-2 carrying a specific-tier disambiguator with
  no ``service=`` token; W15-Λ-degenerate makes the conditionality
  sharp. The CoordPy single-run product runtime contract is byte-for-
  byte unchanged.

### Prior moves (SDK v3.15 — structured producer protocol + W14 family — *first producer-protocol move + first real-LLM strict gain ≥ 0.50 over substrate FIFO*)

- **Phase-61 producer-side ambiguity-preservation benchmark.**
  ``vision_mvp.experiments.phase61_producer_ambiguity_preservation``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``identity`` — the R-61-default sanity anchor; the synthetic
  extractor passes every emission through verbatim. Bench property
  holds in 8/8; W11/W12/W13 cross-round decoders all hit 1.000.
  (ii) ``magnitude_filter`` + ``naive`` prompt — the W14-Λ-prompt
  anchor; the magnitude-filter extractor calibrated against the
  W13-Λ-real real-Ollama observation collapses round-1 by top-N-
  per-(role, kind) by magnitude. On the redesigned comparable-
  magnitude events the threshold filter does not fire (the events
  are designed to clear it) but the prompt-induced compression
  drops monitor's decoy-side mentions; the bench property holds in
  0/8 and every capsule strategy ties FIFO at 0.000.
  (iii) ``magnitude_filter`` + ``structured`` prompt — the W14-1
  anchor; the structured prompt's per-event mandate disables the
  compression. Bench property holds in 8/8; the cross-round capsule
  pipeline achieves ``accuracy_full = 1.000``; +1.000 strict
  separation vs the naive-prompt counterpart, stable across 5/5
  alternate ``bank_seed`` values.
  (iv) ``ollama`` (opt-in) — the R-61-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` with the structured prompt; raw producer
  responses captured into the report. **R-61-OLLAMA-A tier** (the
  strong-real-transfer tier of `docs/SUCCESS_CRITERION_MULTI_AGENT_
  CONTEXT.md` § 1.5): bench property holds in 7/8 scenarios;
  cross-round decoders achieve ``accuracy_full = 0.500``;
  ``layered − fifo = +0.500`` at exactly the 0.50 threshold; audit
  T-1..T-7 preserved on every cell. Anchor:
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``.
- **``StructuredProducerProtocol`` + ``RoleExtractionSchema`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. A prompt-rendering surface
  with two modes: ``naive`` (legacy Phase-58/59/60 byte-for-byte)
  and ``structured`` (W14: tier banner observation/diagnosis split +
  per-tier kind whitelist + per-event mandate). The protocol
  consumes a deterministic ``RoleExtractionSchema`` (allowed kinds
  + observation/diagnosis partition); ``incident_triage_role_
  schemas()`` ships the schema table for the Phase-58..Phase-61
  bench family. A new structured-aware response parser
  (``_parse_structured_response``) dedupes by ``(kind, payload)``
  rather than ``kind`` alone so the per-event mandate survives
  parsing. Re-exported via ``__all__``.
- **Theorem family W14.** W14-Λ-prompt (producer-side ambiguity-
  erasure structural limit on R-61-naive-prompt, proved-empirical
  n=40 saturated × 5 seeds + structural sketch via W7-3 extension),
  W14-1 (StructuredProducerProtocol sufficiency under bounded
  producer compression, proved-conditional + proved-empirical
  synthetic n=40 + real Ollama n=8), W14-2 (schema soundness +
  protocol determinism, proved by inspection + mechanically-
  checked), W14-3 (backward-compat with R-54..R-60, proved-empirical
  full programme-wide regression 393/393), W14-4 (combined-
  intervention falsifier on R-61-ollama-naive, proved-empirical
  n=8), W14-Λ-real (real Ollama 14B prompt-protocol transfer,
  empirical-research n=8 × 24 producer calls). The W14-C family
  (W14-C1..W14-C5) makes cross-bench, model-side calibration,
  multi-round generalisation, cross-model transfer, and multi-
  hypothesis variant extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-61 anchor +
  bar 11 — producer-side ambiguity-preservation split + § 1.5
  R-61-ollama 4-tier grading). The SDK v3.15 result clears the
  **strong success bar** § 1.1 on R-61-structured-prompt synthetic
  (strict gain ≥ 0.20 vs every prior anchor including SDK v3.14
  W13 alone, stable across ≥ 3 (bank_seed) values, no regression on
  R-53..R-60, audit T-1..T-7 preserved on every cell, named bench
  property + named falsifier regime W14-4, AND producer-side
  ambiguity-preservation split bar 11 satisfied — the new method
  includes a load-bearing prompt-side intervention beyond every
  prior downstream layer). The R-61-ollama probe lands at the
  **R-61-OLLAMA-A tier** (the strongest real-transfer tier): the
  milestone is therefore a *strong-success on R-61 synthetic + a
  strong-success / OLLAMA-A on R-61-ollama*, the **first strong-
  success on a real-LLM stream in the programme** (R-60-OLLAMA-C
  was honest null; R-61-OLLAMA-A is +0.500 strict gain at the
  threshold).
- **Honest scope.** The W14-1 win is *conditional* on (a) the
  redesigned comparable-magnitude events (R-61 closure property),
  (b) the structured producer protocol being in effect, AND (c) the
  cross-round capsule pipeline (admission + multi-round decoder).
  The W14-4 falsifier regime (real Ollama + comparable-magnitude
  events + naive prompt) is the explicit counterexample showing
  *both* interventions are required. The W13 closure-widening is
  structurally invisible on R-61-ollama because the real LLM emits
  canonical kinds (zero drift); on the structured prompt the
  load-bearing layer is W14, not W13. The W13 layer is dormant on
  this regime, not refuted.

### Active observations (SDK v3.15)

- **W14-Λ-real (honest positive).** Real Mac-1
  ``qwen2.5:14b-32k`` at ``temperature=0`` on the Phase-61
  redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (empirical confirmation of W14-Λ-prompt on
    real LLM).
  - Under structured prompt: bench property holds 7/8; W11/W12/W13
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    +0.500 strict gain vs FIFO at the R-61-OLLAMA-A threshold.
  - The 1/8 model-error failure is on the LLM (slow_query
    scenario: LLM judged ``error_rate=0.15`` not to qualify as
    ``ERROR_RATE_SPIKE``), not on the protocol — the structured
    prompt + comparable-magnitude events restore the bench
    property in every other scenario. The W14-C2 conjecture
    (magnitude-hinted prompt extension) is the natural next move.

### Active conjectures (SDK v3.15)

- **W14-C1**: cross-bench transfer of the W14 protocol to non-
  incident-triage benchmark families. Conjectural.
- **W14-C2**: model-side magnitude calibration via a *magnitude
  hint* extension to the structured prompt. Conjectural; the W14-
  Λ-real 7/8 anchor is the candidate falsifier.
- **W14-C3**: multi-round generalisation to N ≥ 3 rounds with a
  graded tier hierarchy. Conjectural.
- **W14-C4**: cross-model transfer to qwen3.5:35b-MoE and to non-
  Ollama backends (MLX-distributed). Conjectural; requires Mac 2
  reachable.
- **W14-C5**: multi-hypothesis variant of the protocol that permits
  2-3 candidate kinds per event. Conjectural.

### Discharged / partially-discharged conjectures (SDK v3.15)

- **W13-C3** (SDK v3.14; real-Ollama transfer of W13-1 with a
  redesigned event stream + a prompt that instructs the LLM to
  emit one claim per distinct event). **PARTIALLY DISCHARGED-
  empirical** by W14-1 + W14-Λ-real: the *combination* of
  comparable-magnitude events + structured producer protocol
  clears the strict-gain bar on real Ollama 14B (n=8, +0.500 vs
  FIFO at exactly the R-61-OLLAMA-A threshold), BUT the gain is on
  the **W11/W12/W13 cross-round pipeline as a whole** — not
  specifically on the W13 closure-widening, which is structurally
  invisible because the real LLM still emits canonical kinds. The
  honest revised reading: prompt-side discipline + comparable-
  magnitude events restore the bench property; W13's *normalisation*
  contribution remains dormant on this regime. The cross-round
  structural bar is cleared at +0.500 by the combined intervention
  (W14).

## Previous frontier (SDK v3.14, 2026-04-26)

### Active moves (SDK v3.14 — layered open-world normalisation + real-Ollama probe + W13 family — *first open-world normalisation move and first real-Ollama transfer measurement*)

- **Phase-60 open-world normalisation + real-Ollama benchmark.**
  ``vision_mvp.experiments.phase60_open_world_normalization``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``synthetic_wide_oov_llm`` — the W13-1 anchor; the producer
  emits drifted variants from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` (every entry verified
  *outside* :data:`CLAIM_KIND_SYNONYMS` and *inside* the W13
  heuristic closure); pre-committed default
  ``K_auditor=8, T_auditor=256, n_eval=12, bank_seed=11,
  wide_oov_prob=0.50, svc_token_alt_prob=0.30``. Headline: ``capsule_layered_multi_round`` achieves
  ``accuracy_full = 1.000`` while ``capsule_robust_multi_round``
  (W12) **and** every other strategy ties FIFO at 0.000 — the
  *first strict separation between layered (open-world) and
  fixed-vocabulary normalisation* in the programme; +1.000 vs every
  un-widened method including SDK v3.13 W12; stable across **5/5**
  alternate (bank_seed, llm_seed) values with min gap layered − w12
  = +0.917 (well above the 0.50 stability threshold).
  (ii) ``synthetic_cosmic_oov_llm`` — the W13-4 closure-boundary
  anchor; round-2 specific-tier kinds are replaced by entries from
  :data:`COSMIC_OOV_KINDS` (every entry verified outside both
  layers); ``capsule_layered_multi_round`` ties FIFO at 0.000 on
  8/8 falsifier scenarios.
  (iii) ``synthetic_clean_llm`` + cross-regime audit — the W13-3
  backward-compat anchor; on R-58 / R-59 / R-60-clean / R-54..R-57,
  the W13 decoder produces byte-for-byte the same answer as W11 /
  W12 (heuristic layer fires zero times); R-54..R-58 / R-59-noisy
  anchors all preserved at ``accuracy_full = 1.000``.
  (iv) ``ollama`` — the R-60-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` on the same Phase-58 events; raw producer
  responses captured into the report. **Honest negative finding
  (W13-Λ-real, see § *Active observations* below)**: real Ollama 14B
  emits canonical kinds and filters low-magnitude decoy events;
  the bench property holds in 0/4 scenarios; W13 has nothing to
  rescue and ties W12.
- **``LayeredClaimNormalizer`` + ``LayeredRobustMultiRoundBundleDecoder``
  (new).** ``vision_mvp/coordpy/team_coord.py``. Two-layer normaliser:
  (a) exact :data:`CLAIM_KIND_SYNONYMS` lookup (the W12 path); (b)
  ordered :data:`_HEURISTIC_KIND_RULES` regex-predicate abstraction
  rules whose union strictly widens the W12 closure; (c) optional
  abstention via the :data:`LAYERED_NORMALIZER_ABSTAIN` sentinel.
  Per-call counters expose load-bearing layer breakdowns
  (``n_exact``, ``n_heuristic``, ``n_abstained``, ``n_passthrough``,
  ``rule_hits``). Re-exported as ``LayeredClaimNormalizer``,
  ``LayeredRobustMultiRoundBundleDecoder``,
  ``HeuristicAbstractionRule``, ``LAYERED_NORMALIZER_ABSTAIN``.
- **Theorem family W13.** W13-Λ-fixed (fixed-vocabulary closure
  limit on R-60-wide, proved-empirical n=12 + structural sketch),
  W13-1 (LayeredRobustMultiRoundBundleDecoder sufficiency under
  bounded OOV in the heuristic closure, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W13-2 (heuristic
  abstraction soundness, proved by inspection + mechanically-
  checked), W13-3 (backward-compat with R-54..R-58 + R-59 + R-60-
  clean, proved-empirical n=8 each + cross-regime audit), W13-4
  (cosmic-OOV closure boundary, proved-empirical n=8 saturated),
  W13-Λ-real (real Ollama 14B canonical-kind + magnitude-filtering
  observation, empirical-research n=4 producer-side observation +
  12 real Ollama calls). The W13-C family (W13-C1..W13-C4) makes
  cross-bench, learned-normaliser, real-Ollama-with-redesigned-
  events, and abstention-aware-decoder extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-60 anchor +
  bar 10 — open-world normalisation split + § 1.4 R-60-ollama
  4-tier grading). The SDK v3.14 result clears the **strong success
  bar** § 1.1 on R-60-wide (strict gain ≥ 0.20 vs every fixed-
  vocabulary method including SDK v3.13 W12, stable across ≥ 3
  (bank_seed, llm_seed) values, no regression on
  R-53..R-59 / R-60-clean, audit T-1..T-7 preserved on every cell,
  named bench property + named falsifier regime W13-4, AND open-
  world normalisation split bar 10 satisfied — the new method
  includes a load-bearing heuristic abstraction layer beyond the
  exact synonym table). The R-60-ollama probe lands at the
  **R-60-OLLAMA-C tier** (honest null real transfer): the milestone
  is therefore a *strong-success on R-60-wide synthetic + a
  partial-success / honest-null on R-60-ollama*, NOT a strong
  cross-axis advance. § 1.4 of the success criterion makes this
  partition pre-committed and falsifiable.
- **Honest scope.** The W13-1 win is *conditional* on (a) the named
  bench property (R-58 delayed-causal-evidence shape with the
  Phase-60 wide-OOV drift channel), (b) the producer-noise channel
  being bounded by the *heuristic* closure (every variant in
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` matches at least one
  pattern in :data:`_HEURISTIC_KIND_RULES`), AND (c) round-N
  admission not being budget-starved (inherits W11-4). The W13-4
  falsifier regime is the explicit counterexample; the W13-Λ-real
  observation is a *separate, honest, partial* outcome. The W13
  method is research-grade SDK code, additive on top of W12.

### Active observations (SDK v3.14)

- **W13-Λ-real (honest negative).** Real Ollama 14B
  (qwen2.5:14b-32k on Mac 1, ``temperature=0``) on the calibrated
  Phase-58 incident-triage prompt does NOT generate the R-58
  delayed-causal-evidence bench property: across 4 scenarios × 12
  producer calls, the LLM emits 0 drifted kinds (every claim_kind
  is canonical) and filters low-magnitude decoy events as noise
  (the ``monitor`` role emits ``NONE`` for the deliberately-low-
  magnitude decoy events, breaking the cross-role decoy
  corroboration assumption). The bench property holds in 0/4
  scenarios; normalisation has nothing to rescue; W13 ties W12 ties
  multi_round at ``accuracy_full = 0.250``. The R-60-ollama probe
  is therefore a *measurement*, not a *claim*: the synthetic→real-
  LLM transfer story has **five layers** —
  (i) un-normalised admission cannot transfer (W6-C2 falsified),
  (ii) un-normalised cross-round decoding cannot transfer
  (W12-Λ at the real-LLM axis),
  (iii) fixed-vocabulary normalisation transfers under bounded
  *synthetic* drift (W12-1, conditional),
  (iv) heuristic-widened normalisation transfers under bounded
  *open-world* drift inside the heuristic closure (W13-1,
  conditional),
  (v) real Ollama 14B at default settings does not produce the
  drift OR the cross-role decoy corroboration shape (W13-Λ-real,
  empirical observation; the gating axis on real Ollama is *event-
  shape design + prompt-side discipline*, not normalisation).
  Future work: redesign the events so the decoy has comparable
  magnitudes to gold (W13-C3) — and accept that the contribution
  shifts from "the normaliser" to "the prompt + event design".

### Active conjectures (SDK v3.14)

- **W13-C1**: cross-bench transfer of the W13 closure-widening
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where any size-
  bounded predicate set covers < 50 % of LLM kind drift.
- **W13-C2**: a learned normaliser strictly widens the W13
  heuristic closure on R-60-cosmic. Conjectural; restated as a
  closure-widening move, not a structural fix.
- **W13-C3**: real-Ollama transfer of W13-1 with redesigned
  events. Conjectural; Phase-60 v2 candidate.
- **W13-C4**: abstention as a load-bearing signal — an
  abstention-aware decoder strictly improves over a passthrough
  decoder. Conjectural; the abstention sentinel is implemented but
  the abstention-aware decoder is not yet wired.

### Discharged / partially-discharged conjectures (SDK v3.14)

- **W12-C2** (SDK v3.13; real-Ollama transfer of W12-1).
  **PARTIALLY DISCHARGED-empirical** (negatively): real Ollama 14B
  on the Phase-58 events does NOT emit drift, so the W12 advance
  is *structurally invisible* on R-60-ollama (W13-Λ-real). The
  W12-C2 question reframes as: under what (event design × prompt)
  does a real LLM emit non-trivial bounded drift? — that is W13-C3.
- **W12-C3** (SDK v3.13; learned synonym table widens the W12
  closure). **PARTIALLY DISCHARGED-empirical** by W13-1: a
  *heuristic* layered normaliser strictly widens the closure
  beyond the fixed table on R-60-wide. The *learned* variant
  remains conjectural (W13-C2). The W13-4 closure boundary applies
  to any predicate-based normaliser, learned or not — a learned
  normaliser only widens, does not eliminate, the closure.

## Previous frontier (SDK v3.13, 2026-04-26)

### Active moves (SDK v3.13 — real-LLM-robust multi-round bundle decoder + W12 family — *first synthetic→real-LLM transfer move*)

- **Phase-59 real-LLM-driven multi-round delayed-disambiguation
  benchmark.**
  ``vision_mvp.experiments.phase59_real_llm_multi_round`` runs 10
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11), capsule_robust_multi_round (W12)) on the same Phase-58
  delayed-causal-evidence shape under an **LLM-shaped extractor**:
  default mode ``synthetic_noisy_llm`` injects realistic
  claim-kind drift (``synonym_prob=0.50``) and payload drift
  (``svc_token_alt_prob=0.30``) calibrated against Phase-53
  empirical 14B / 35B parser_role_response distributions; the
  optional ``ollama`` mode talks the same Phase-53-style prompt
  to a real LLM endpoint with synthetic fallback on HTTP failure.
  12/12 default scenarios satisfy the post-normalisation
  delayed-causal-evidence property; mechanically verified by
  ``Phase59BenchPropertyTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=12``,
  ``bank_seed=11``, ``bank_replicates=3``,
  ``llm_synonym_prob=0.50``, ``llm_svc_alt_prob=0.30``,
  ``llm_seed=11``. Headline: ``capsule_robust_multi_round``
  achieves ``accuracy_full = 1.000`` while substrate, FIFO,
  priority, coverage, W7-2, W8, W9, W10 single-round bundle, AND
  **W11 un-normalised** all produce ``accuracy_full = 0.000`` —
  the **first strict separation between un-normalised and
  normalised cross-round capsule-native coordination on a
  real-LLM-shaped stream** in the programme, **+1.000** vs every
  other method, stable across **5/5** alternate (bank_seed,
  llm_seed) values. The W12 family (W12-Λ / W12-1 / W12-2 /
  W12-3 / W12-4 — proved or proved-empirical) anchors the
  milestone formally; the W12-C family (W12-C1/C2/C3) makes the
  cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **``RobustMultiRoundBundleDecoder`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Wraps
  :class:`MultiRoundBundleDecoder` with a closed-vocabulary
  normalisation layer: :func:`normalize_claim_kind` rewrites
  drifted ``claim_kind`` strings into canonical kinds via
  :data:`CLAIM_KIND_SYNONYMS` (≈ 60 entries covering 11 canonical
  kinds × 4-5 LLM variants each, lex-ordered for diff stability);
  :func:`normalize_payload` rewrites alternative service-tag
  spellings (``svc=X``, ``for service X``, ``service:X``,
  ``service_name=X``, …) into the canonical ``service=<tag>`` form
  via the closed-vocabulary :data:`_SERVICE_TAG_REWRITES` regex
  table. Per-call rewrite counters (``last_n_kind_rewrites``,
  ``last_n_payload_rewrites``) expose the normaliser's load-bearing
  status to the bench driver. Re-exported as
  ``RobustMultiRoundBundleDecoder``, ``CLAIM_KIND_SYNONYMS``,
  ``normalize_claim_kind``, ``normalize_payload``,
  ``normalize_handoff``.
- **Theorem family W12.** W12-Λ (real-LLM single-round / un-
  normalised structural limit on R-59, proved-empirical n=12 + 5
  seeds + structural sketch), W12-1 (RobustMultiRoundBundleDecoder
  sufficiency under bounded LLM noise, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W12-2 (closed-
  vocabulary normalisation soundness, proved by inspection +
  mechanically-checked), W12-3 (backward-compat with R-58 + R-59-
  clean + cross-regime R-54..R-58 audit, proved-empirical n=8
  each), W12-4 (out-of-vocabulary noise-budget falsifier, proved-
  empirical n=8 saturated). The W12-C family (W12-C1/C2/C3) makes
  the cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-59 anchor +
  bar 9 — synthetic→real-LLM transfer split). The SDK v3.13 result
  clears the **strong success bar** § 1.1 (strict gain ≥ 0.20 on
  R-59 vs every un-normalised single-round / multi-round method
  including SDK v3.12 W11, stable across ≥ 3 (bank_seed, llm_seed)
  values, no regression on R-53 / R-54 / R-55 / R-56 / R-57 /
  R-58 / R-59-clean, audit T-1..T-7 preserved on every cell, named
  bench property + named falsifier regime, AND synthetic→real-LLM
  transfer split bar 9 satisfied — the new method includes the
  load-bearing closed-vocabulary normalisation layer that bar 9
  requires).
- **Honest scope.** The W12-1 win is *conditional* on (a) the
  named bench property (R-58 delayed-causal-evidence shape), (b)
  the producer-noise channel being bounded by the closed-vocabulary
  closure (every variant in :data:`NOISY_KIND_VARIANTS` is in
  :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission not being
  budget-starved (inherits W11-4). The W12-4 falsifier regime is
  the explicit counterexample: when the LLM emits *out-of-vocabulary*
  kinds the synonym table cannot cover (e.g.
  ``DEADLOCK_PROBABLY_DETECTED_MAYBE``), normalisation cannot
  rescue the run. The synthetic-noisy-LLM extractor is *labelled*
  in every Phase-59 report; the ``ollama`` opt-in mode is the
  honest extension path and is the W12-C2 next data point.

### Active conjectures (SDK v3.13)

- **W12-C1**: cross-bench transfer of the W12 normalisation
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where LLM kind
  drift cannot be enclosed by any reasonable-size synonym table.
- **W12-C2**: real-Ollama transfer of W12-1 (Phase-59 ``ollama``
  mode against qwen2.5:14b-32k or qwen3.5:35b on Mac 1).
  Conjectural; the synthetic noisy channel is calibrated to the
  empirical Phase-53 14B/35B distributions, but the real LLM may
  emit drift outside the closed-vocabulary closure.
- **W12-C3**: a learned normaliser strictly widens the closure
  beyond the hand-curated table. Conjectural; restated as a
  research move not a structural fix.

### Discharged conjectures (SDK v3.13)

- **W11-C2** (SDK v3.12): real-LLM transfer of W11-1.
  **PARTIALLY DISCHARGED-empirical** by the W12 family: the
  *un-normalised* W11 decoder does NOT transfer (W12-Λ shows
  multi_round ties FIFO at 0.000 on Phase-59 default at
  ``synonym_prob=0.50``), but a *normalised* W11 decoder
  (W12-1) DOES transfer (+1.000 vs every un-normalised method,
  stable 5/5). The honest revised reading: synthetic cross-round
  structure transfers to real-LLM regimes *only when an explicit
  normalisation layer absorbs the producer's kind / payload drift
  channel*.

## Previous frontier (SDK v3.12, 2026-04-26)

### Active moves (SDK v3.12 — multi-round bundle-aware team decoder + W11 family — *first cross-round coordination move*)

- **Phase-58 multi-round delayed-causal-evidence benchmark.**
  ``vision_mvp.experiments.phase58_multi_round_decoder`` runs 9
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11)) on a deterministic 8-scenario bank with the
  **delayed-causal-evidence** property: round-1 carries
  generic-noise-only mentions of (gold_A, gold_B, decoy) where the
  decoy is cross-role-corroborated; round-2 carries one
  specific-tier disambiguating ``claim_kind`` with NO ``service=``
  token. 8/8 default scenarios satisfy the property; mechanically
  verified by ``Phase58BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=8``,
  ``bank_seed=11``, ``bank_replicates=2``,
  ``noise_decoy_role_floor=2``. Headline:
  ``capsule_multi_round`` achieves ``accuracy_full = 1.000`` while
  substrate, FIFO, priority, coverage, W7-2, W8, W9, AND W10
  single-round all produce ``accuracy_full = 0.000`` — the **first
  strict separation between multi-round and single-round capsule-
  native coordination** in the programme, **+1.000** vs every
  prior method, stable across **5/5** alternate bank seeds. The
  W11 family (W11-Λ / W11-1 / W11-2 / W11-3 / W11-4 — proved or
  proved-empirical) anchors the milestone formally; the W11-C
  family (W11-C1/C2/C3) makes the cross-bench / real-LLM /
  multi-step extensions falsifiable.
- **``MultiRoundBundleDecoder`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Decodes the *union* of
  admitted handoffs across multiple ROLE_VIEW capsules; if the
  elected root_cause is specific-tier, drops every service tag
  whose admitted mentions are exclusively generic-noise kinds AND
  span ≥ ``noise_decoy_role_floor`` distinct producer roles
  (default 2). Inner ``BundleAwareTeamDecoder`` configured with
  ``cck_filter=False`` so the contradiction-aware step is the only
  filter; the W10 fallback path preserves single-round wins on
  R-54..R-57 (W11-3). Companion helper
  ``collect_admitted_handoffs(ledger, role_view_cids)`` materialises
  multi-round admitted handoffs into the duck-typed
  ``_DecodedHandoff`` shape. Re-exported as
  ``MultiRoundBundleDecoder`` and ``collect_admitted_handoffs``.
- **Theorem family W11.** W11-Λ (single-round structural limit on
  R-58, proved-empirical + structural sketch), W11-1 (multi-round
  decoder sufficiency, proved-empirical n=40 saturated across 5
  seeds), W11-2 (round-union monotonicity, proved structural),
  W11-3 (backward-compat with W7-2 / W8 / W9 / W10 on
  R-54 / R-55 / R-56 / R-57, proved-empirical), W11-4 (round-budget
  falsifier, proved-empirical n=8 saturated). The W11-C family
  (W11-C1/C2/C3) makes the cross-bench / real-LLM / multi-step
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-58 anchor +
  bar 8 — temporal/structural split). The SDK v3.12 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-58 vs
  every SDK v3.11 single-round method, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56 / R-57, audit T-1..T-7
  preserved on every cell, named bench property + named falsifier
  regime, AND temporal/structural split bar 8 satisfied).
- **Honest scope.** The W11-1 win is *conditional* on the named
  bench property; the W11-4 falsifier regime is the explicit
  counterexample. W11-3 backward-compat is exact on R-54 / R-55 /
  R-56 / R-57 thanks to (a) the inner W10 decoder's fallback-on-
  small-admitted-set path, (b) the noise-decoy floor being
  insensitive to single-role decoys. The contradiction-aware drop
  is closed-vocabulary on incident-triage; W11-C1 is the
  conjectural extension to other benchmark families. The decoder
  is a no-op on generic-tier elected root_cause (W11-Λ at the
  temporal axis collapses).

### Active conjectures (SDK v3.12)

- **W11-C1**: noise-decoy drop generalises to non-incident-triage
  benchmark families. Conjectural; falsifier = a benchmark family
  where a generic-noise-only mention is informative.
- **W11-C2**: real-LLM transfer of W11-1. Conjectural; Phase-59
  candidate.
- **W11-C3**: contradiction-aware round-resolution rule (last-wins
  / weighted-confidence) strictly outperforms naive union with
  ≥ 3 rounds and conflicting specific-tier evidence across rounds.
  Conjectural; multi-step capsule chains not yet shipped.

### Discharged conjectures (SDK v3.12)

- **W10-C3** (SDK v3.11): multi-round bundle decoder closes W10-4
  on a sub-class of scenarios. **PARTIALLY DISCHARGED** by the
  W11 family on R-58 (a different regime than W10-4): W11-1
  resolves the *delayed-causal-evidence* sub-class; the W10-4
  *decoy-CCK-promotion* sub-class remains open (the W11
  contradiction-aware drop is orthogonal — drops noise-corroborated
  decoys, not CCK-corroborated decoys).

## Previous frontier (SDK v3.11, 2026-04-26)

### Active moves (SDK v3.11 — bundle-aware team decoder + W10 family — *first decoder-side coordination move*)

- **Phase-57 multi-service-gold + corroborated-decoy decoder-forcing
  benchmark.**
  ``vision_mvp.experiments.phase57_decoder_forcing`` runs 8 strategies
  (substrate, capsule_fifo, capsule_priority, capsule_coverage,
  capsule_cohort_buffered (W7-2), capsule_corroboration (W8),
  capsule_multi_service (W9), capsule_bundle_decoder (W10-AD: W9
  admission + bundle-aware decoder)) on a deterministic 12-scenario
  bank (clipped to 10 by default) where (i) every scenario has
  ``gold_services`` of size 2 with a *specific-tier* gold root_cause,
  (ii) both gold services are corroborated by ≥ 2 distinct producer
  roles via at least one CCK-eligible (causal-tier) claim_kind, AND
  (iii) a decoy service is *also* corroborated by ≥ 2 distinct
  producer roles via *only* non-causal claim_kinds (LATENCY_SPIKE,
  ERROR_RATE_SPIKE, FW_BLOCK_SURGE). 10/10 default scenarios satisfy
  the bench property; mechanically verified by
  ``Phase57BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=3``, ``ms_min_corroborated_roles=2``,
  ``bundle_role_floor=1``,
  ``fallback_admitted_size_threshold=2``. Headline:
  ``capsule_bundle_decoder`` achieves ``accuracy_full = 1.000``
  while substrate, FIFO, priority, coverage, W7-2, W8, AND W9 all
  produce ``accuracy_full = 0.000`` — the **first strict separation
  between decoder-side coordination and any service-blind admission
  policy** in the programme, **+1.000** vs every admission baseline,
  stable across **5/5** alternate bank seeds. The W10 family
  (W10-Λ / W10-1 / W10-2 / W10-3 / W10-4 — proved or proved-empirical)
  anchors the milestone formally; the W10-C family (W10-C1/C2/C3)
  makes the cross-bench / real-LLM / multi-round extensions
  falsifiable.
- **``BundleAwareTeamDecoder`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Three knobs (``cck_filter``,
  ``role_corroboration_floor``,
  ``fallback_admitted_size_threshold``). Closed-vocabulary CCK table
  ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` maps each root_cause label
  to its causal claim-kind set (data-tier / storage-tier / compute /
  edge / network / generic). Decoder rule: pick root_cause via the
  same priority decoder as
  ``incident_triage._decoder_from_handoffs``; project admitted
  service tags through the CCK predicate; trust admission when the
  admitted-tag set is small (size ≤ threshold). Deterministic,
  training-free, additive on top of the existing ``TeamCoordinator``
  / ``ROLE_VIEW`` / ``TEAM_DECISION`` capsule layer. Re-exported as
  ``BundleAwareTeamDecoder`` and ``decode_admitted_role_view``.
- **Theorem family W10.** W10-Λ (admission-only structural limit on
  R-57, proved-empirical + structural sketch), W10-1 (bundle-decoder
  sufficiency, proved-empirical n=50 saturated), W10-2 (CCK
  structural correctness, proved by inspection), W10-3 (backward-
  compat with W7-2 / W8 / W9 on R-54 / R-55 / R-56, proved-empirical),
  W10-4 (decoy-CCK-promotion falsifier, proved-empirical n=10
  saturated). The W10-C family makes the cross-bench / real-LLM /
  multi-round extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-57 anchor +
  bar 7 — admission/decoding split). The SDK v3.11 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-57 vs
  both substrate FIFO and SDK v3.10 W9, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56, audit T-1..T-7 preserved
  on every cell, named bench property + named falsifier regime,
  AND admission/decoding split bar 7 satisfied).
- **Honest scope.** The W10-1 win is *conditional* on the named
  bench property; the W10-4 falsifier regime is the explicit
  counterexample. W10-3 backward-compat is exact on R-54 / R-55 /
  R-56 thanks to the trust-admission fallback (size ≤ 2 threshold).
  The CCK table is *closed-vocabulary for incident-triage*; W10-C1
  is the conjectural extension to other benchmark families. The
  bundle decoder is a no-op on generic-tier root_causes
  (``error_spike`` / ``latency_spike``) — this is the named scope
  limit of W10-1.

### Active conjectures (SDK v3.11)

- **W10-C1**: CCK table extends to non-incident-triage benchmark
  families (security incident, robotics, compliance review).
  Conjectural; falsifier = a benchmark family where no closed-
  vocabulary tier mapping exists.
- **W10-C2**: real-LLM transfer of W10-1. Conjectural; Phase-58
  candidate.
- **W10-C3**: multi-round bundle decoder closes W10-4 on a
  sub-class of scenarios. Conjectural; multi-round capsule chain
  not yet shipped.

### Discharged conjectures (SDK v3.11)

- **W9-C1** (SDK v3.10): bundle-aware decoder companion strictly
  improves on Phase-56 falsifier. **DISCHARGED-empirical** by W10-1
  on Phase 57 (+1.000 vs every admission-only baseline). The
  decoder-side axis is now the load-bearing axis of the SDK v3.11
  milestone.

## Previous frontier (SDK v3.10, 2026-04-26)

### Active moves (SDK v3.10 — multi-service top-K cross-role corroboration multi-agent benchmark + W9 family)

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase56_multi_service_corroboration`` runs
  7 admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9)) on a
  deterministic 10-scenario bank with the **multi-service-gold +
  both-gold-cross-role-corroborated + single-role-decoy-storm**
  properties (10/10 scenarios). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=2``, ``ms_min_corroborated_roles=2``.
  Headline: ``multi_service − fifo accuracy_full = +1.000``,
  ``multi_service − cohort_buffered = +1.000``, AND
  ``multi_service − corroboration = +1.000``, stable across 5/5
  alternate bank seeds. The **first strict separation between
  multi-service top-K corroboration and single-tag corroboration**
  in the programme.
- **``MultiServiceCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the W9-1
  anchor — pre-fits a top-K dominant tag set via the
  argmax-by-role-count tier of the corroboration score function.
  Selection rule: ``min_corroborated_roles`` floor → argmax-by-role-
  count tier → top-K by score (lex tie-break). Deterministic,
  training-free, one regex + two counters + the ``_dominant_tag_set``
  helper. Re-exported as
  ``TeamMultiServiceCorroborationAdmissionPolicy``.
- **Theorem family W9.** W9-1 (strict separation, proved-empirical
  n=50 saturated), W9-2 (argmax-tier strict-ordering, proved
  structural), W9-3 (backward-compat with W8 + W7-2 on Phase 55 +
  Phase 54, proved-empirical), W9-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W9-C family (W9-C1/C2/C3)
  makes the bundle-aware decoder / |gold|≥3 / real-LLM extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-56 anchor).
  The SDK v3.10 result clears the **strong success bar** § 1.1
  (strict gain ≥ 0.20 on R-56 vs both substrate FIFO and SDK v3.9
  W8, stable across ≥ 3 seeds, no regression on R-53 / R-54 / R-55,
  audit T-1..T-7 preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W9-1 win is *conditional* on the named
  bench property; the W9-4 falsifier regime is the explicit
  counterexample. W9-3 preserves the SDK v3.9 W8-1 win
  byte-for-byte on Phase 55 (via the argmax-by-role-count gate).

### Active conjectures (SDK v3.10)

- **W9-C1** (new SDK v3.10): bundle-aware decoder companion that
  filters service tags at decode time by the dominant
  *(claim_kind, role)* signature strictly improves accuracy_full on
  the Phase-56 falsifier regime. **Conjectural**; reframes W8-C3 as
  the natural attack on the W9-4 falsifier — pushes the structural
  axis from admission to decoding.
- **W9-C2** (new SDK v3.10): top-K extension to ``|gold| ≥ 3``.
  Conjectural; Phase-57 candidate; the policy already supports
  arbitrary ``top_k``.
- **W9-C3** (new SDK v3.10): real-LLM transfer of W9-1.
  Conjectural; SDK v3.10 confirms no-regression in low-surplus
  synthetic regime.

### Discharged conjectures (SDK v3.10)

- **W8-C1** (SDK v3.9): top-k corroboration improves multi-service
  scenarios. **DISCHARGED-empirical** by W9-1 on Phase 56 (+1.000).

## Previous frontier (SDK v3.9, 2026-04-26)

### Active moves (SDK v3.9 — cross-role corroboration multi-agent benchmark + W8 family)

- **Phase-55 decoy-plurality + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase55_decoy_plurality`` runs 6
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8)) on a deterministic 10-scenario bank
  with the **decoy-plurality + gold-corroboration** properties.
  Pre-committed default: ``K_auditor=4``, ``T_auditor=128``,
  ``n_eval=10``, ``bank_seed=11``. Headline: ``corroboration −
  fifo accuracy_full = +1.000`` AND ``corroboration −
  cohort_buffered accuracy_full = +1.000``, stable across 5/5
  alternate bank seeds. The first strict separation between W8
  and W7-2 in the programme.
- **``CrossRoleCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the
  W8-1 anchor — pre-fits a (role, tag)-aggregated dominant tag
  via score function ``W_role · |distinct_roles| + |raw_mentions|``.
  Deterministic, training-free, one regex + two counters.
  Re-exported as ``TeamCrossRoleCorroborationAdmissionPolicy``.
- **Theorem family W8.** W8-1 (strict separation, proved-empirical
  n=50 saturated), W8-2 (score-function strict-ordering, proved
  structural), W8-3 (backward-compat with W7-2 on Phase 54,
  proved-empirical), W8-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W8-C family
  (W8-C1/C2/C3) makes the multi-service / decoder-side / real-LLM
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``. The SDK v3.9
  result clears the **strong success bar** § 1.1 (strict gain
  ≥ 0.20 on R-55, stable across ≥ 3 seeds, no regression on R-53
  / R-54, audit preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W8-1 win is *conditional* on the named
  bench property; the W8-4 falsifier regime is the explicit
  counterexample. W8-3 preserves the SDK v3.8 W7-2 win
  byte-for-byte on Phase 54.

### Active conjectures (SDK v3.9)

- **W8-C1**: multi-service-gold extension (top-k corroboration).
  conjectural; falsifier in Phase-56 candidate.
- **W8-C2**: W8-1 transfers to a real-LLM regime when the
  producer LLM emits multi-service decoy chatter that satisfies
  the bench property. conjectural; Phase-56 candidate.
- **W8-C3**: bundle-aware decoder + corroboration admission
  strictly outperforms admission alone on partial-coverage
  benches. conjectural; restates W7-C2 under W8 framing.

## Previous frontier (SDK v3.8, 2026-04-26)

### Active moves (SDK v3.8 — cross-role cohort-coherence multi-agent benchmark + W7 family)

- **Phase-54 cross-role cohort-coherence benchmark.**
  ``vision_mvp.experiments.phase54_cross_role_coherence`` runs 5
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_streaming, capsule_cohort_buffered)
  on a deterministic 10-scenario bank with the **gold-plurality**
  property (gold service has strictly more auditor-routed
  candidates than any decoy service). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``. Headline: ``cohort_buffered − fifo
  accuracy_full = +1.000``, stable across 5/5 alternate bank
  seeds.
- **``CohortCoherenceAdmissionPolicy`` (new).**
  ``vision_mvp/coordpy/team_coord.py``. Two sub-modes:
  *streaming* (arrival-order-sensitive baseline) and *buffered*
  (arrival-order-stable, pre-fitted plurality via
  ``from_candidate_payloads``). Deterministic, training-free,
  one regex + one counter. Re-exported as
  ``TeamCohortCoherenceAdmissionPolicy``.
- **Theorem family W7.** W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort buffered structural win under
  gold-plurality, proved-empirical n=50 saturated), W7-2-conditional
  (K-sweep window, proved-empirical), W7-3 (extraction floor,
  proved-negative — corollary of Capsule Contract C5). The
  W7-C family (W7-C1/C2/C3) makes the multi-service /
  decoder-side / real-LLM extensions falsifiable.
- **Honest scope.** The W7-2 win is *conditional* on stated bench
  properties (gold-plurality + cross-role coherence +
  ``|candidates| > K_auditor``); it does not generalise to every
  multi-agent benchmark. The capsule layer's *audit* contribution
  (T-1..T-7) is preserved and extends to Phase-54 unchanged.

### Active conjectures (SDK v3.8)

- **W7-C1**: multi-service-gold extension (top-2 plurality).
  conjectural; falsifier in Phase-55 candidate.
- **W7-C2**: bundle-aware decoder + cohort admission strictly
  dominates cohort admission alone on weak-coherence benches.
  conjectural; not yet measured.
- **W7-C3**: W7-2 transfers to the real-LLM regime when the LLM
  is prompted with a multi-service event mix. conjectural;
  Phase-56 candidate.

## Previous frontier (SDK v3.7, 2026-04-26)

### Active moves (SDK v3.7 — stronger-model multi-agent benchmark + W6 family)

- **Phase-53 stronger-model multi-agent benchmark.**
  ``vision_mvp.experiments.phase53_scale_vs_structure`` runs
  three model regimes (synthetic / qwen2.5:14b-32k /
  qwen3.5:35b) × four capsule admission strategies + the
  Phase-31 substrate baseline on the same candidate-handoff
  stream. Real LLM calls hit Mac 1 Ollama at
  ``192.168.12.191:11434`` (Mac 2 still offline). Wall: 14B =
  92.6 s, 35B = 152 s.
- **Theorem family W6.** W6-1 (audit-OK grid 60/60),
  W6-2 (backend duck-typing), W6-3 (parser robustness on the
  closed-vocabulary claim grammar) are proved + mechanically-
  checked. W6-4 (the empirical decomposition) is proved-empirical
  on n=5 saturated.
- **Conditional falsification of W4-C1.** The SDK v3.5
  learned-admission-policy advantage **does not transfer
  out-of-distribution** to the real-LLM regime. Per-regime gap
  (capsule_learned − capsule_fifo): -0.4 (synthetic) / -0.4
  (qwen2.5:14b-32k) / 0.0 (qwen3.5:35b). The W4-C1 row in the
  registry is now conditional (see § 4.4 of
  `docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`).
- **Honest scope.** Mac 2 is still offline; no two-Mac sharded
  inference run happened in SDK v3.7. The strongest model class
  exercised is single-Mac qwen3.5:35b (36 B-MoE). The
  ``MLXDistributedBackend`` adapter is unchanged from SDK v3.6
  and remains correct against the in-process stub.

### Active conjectures (SDK v3.7)

- **W6-C1**: drafted-conjecture-falsified — structure_gain is
  non-positive at every regime tested on Phase-53 default;
  scale narrows a deficit (not a surplus).
- **W6-C2**: drafted-conjecture-falsified — synthetic→real
  transfer of the learned admission scorer LOSES to FIFO out-
  of-distribution.
- **W6-C3**: empirical-positive — cross-(14B, 35B) candidate-
  kind TVD = 0.167 on the pooled (source_role × claim_kind)
  histogram (above the 0.10 falsifier).
- **W6-C4**: new conjectural-empirical — substrate FIFO is
  competitive with every capsule admission policy at sufficient
  K_auditor; falsifier search direction is K_auditor ∈ {1, 2, 3}.
- **W6-C5**: new conjectural-empirical — model scale narrows
  the OOD generalisation gap of the per-role admission scorer
  trained on synthetic noise.

## Previous frontier (SDK v3.6, 2026-04-26)

### Active moves (SDK v3.6 — two-Mac distributed inference + real cross-LLM)

- **Chosen two-Mac inference path: MLX distributed.** Apple-
  official, supports Llama / Qwen / Mistral natively, and
  exposes a single OpenAI-compatible HTTP server (head rank)
  regardless of single-host or sharded across N hosts. Hyperspace
  is a strong distributed-agent infrastructure but **not** a
  single-model sharding system; llama.cpp `--rpc` is a
  defensible alternative but with smaller Apple-Silicon
  optimisation.
- **Realistic model class on 2×36 GB:** 70B-class in Q4
  (≈ 40 GB weights; fits across two Macs with KV-cache headroom).
  35B-class in Q4 fits on a single Mac; sharding buys
  context-length / KV headroom only.
- **CoordPy integration boundary** (`vision_mvp.coordpy.llm_backend`):
  a duck-typed `LLMBackend` Protocol with two concrete
  implementations (`OllamaBackend`, `MLXDistributedBackend`).
  The runtime's inner-loop seal-PROMPT / seal-LLM_RESPONSE chain
  accepts any conformant backend without any spine modification
  (W5-2 proved); the OpenAI-compatible wire shape is locked
  against a stub server (W5-3 proved).
- **Real cross-LLM parser-boundary measurement (W5-1)**:
  `parser_boundary_real_llm.py` against the live Mac 1 Ollama
  endpoint yields cross-model PARSE_OUTCOME failure-kind
  TVD = 1.000 between Qwen-2.5-14B (dense, Q4) and Qwen-3.5-35B
  (MoE, Q4, `think=False`) under strict parsing on n=10
  instances — the larger model emits OLD/NEW close as `<<`
  instead of `<<<` and lands entirely in `unclosed_new`, while
  the smaller model emits `<<<` cleanly. Robust-mode
  `recovery=closed_at_eos` collapses cross-model TVD to 0.000.
  This **inverts the naive prediction** that a stronger model
  would reduce parser-boundary instability.

### Active conjectures (SDK v3.6)

- **W5-C1**: parser-boundary instability is a (model
  architecture × prompt-format) interaction, not a capacity
  artefact. Empirical-research; falsifier = a bank on which
  the larger model strict-parses ok > 50%.
- **W5-C2**: robust-mode `recovery=closed_at_eos` is the
  load-bearing safety net that makes the capsule-native runtime
  model-class-agnostic on the bundled prompt format. Empirical-
  research; falsifier = a model whose `unclosed_new` cannot be
  salvaged.
- **W5-C3**: closed-vocabulary `PARSE_OUTCOME.failure_kind` is
  a *minimum sufficient* typed witness of cross-model behaviour
  differences. Conjectural research; falsifier = a model pair
  with identical strict-mode `failure_kind` distribution but
  materially different downstream test-pass rate.

## Current frontier (SDK v3.5, 2026-04-26)

### Active moves (SDK v3.5 — multi-agent capsule coordination)

- **Capsule-native multi-agent team coordination
  (W4 family).** Three new closed-vocabulary capsule kinds
  (TEAM_HANDOFF, ROLE_VIEW, TEAM_DECISION) make capsules
  load-bearing *between* agents. ``TeamCoordinator`` drives one
  coordination round end-to-end; ``audit_team_lifecycle``
  mechanically verifies T-1..T-7 (Theorem W4-1).
- **Coverage-implies-correctness** (W4-2, proved-conditional) and
  **Local-view limitation** (W4-3, proved-negative) anchor the
  team-level mechanism in the formal layer.
- **Learned per-role admission policy** (``team_policy.py``)
  strictly improves pooled team-decision accuracy over the
  strongest fixed baseline at matched per-role budgets on the
  Phase-52 incident-triage bench (W4-C1 positive empirical;
  conjectural at smaller train scales).
- **Phase-52 reference benchmark**
  (``vision_mvp/experiments/phase52_team_coord.py``) compares
  substrate / capsule_fifo / capsule_priority / capsule_coverage
  / capsule_learned head-to-head and reports
  ``audit_ok_rate = 1.000`` for every capsule strategy.

### Active moves (SDK v3.4 — still in force)

- **Capsule-native execution one further structural layer past
  v3.3.** PROMPT capsule sealed for every LLM call's prompt
  bytes; LLM_RESPONSE capsule sealed for every response bytes.
  PROMPT.parents = (SWEEP_SPEC,) (Theorem W3-42); LLM_RESPONSE
  parent = sealed PROMPT (Theorem W3-43); PARSE_OUTCOME may
  parent on (SWEEP_SPEC, LLM_RESPONSE) so the
  prompt → response → parse → patch → verdict chain is a
  typed DAG witness end-to-end (Theorem W3-44).
- **Lifecycle audit extended to L-9 / L-10 / L-11** (Theorem
  W3-45). Soundness: ``audit_capsule_lifecycle(ctx).verdict ==
  "OK"`` iff the ledger satisfies the eleven invariants.
- **Synthetic-LLM mode for CI-runnable end-to-end exercise.**
  ``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``
  uses a deterministic in-process synthetic LLM client; no
  network. The full prompt/response/parse/patch/verdict chain
  seals end-to-end on every (task, strategy).
- **Cross-model parser-boundary research (W3-C6, empirical).**
  ``vision_mvp.experiments.parser_boundary_cross_model``
  reports cross-distribution PARSE_OUTCOME failure-kind TVD up
  to 1.000 across the synthetic distribution library, and
  strict→robust parser-mode shift up to 1.000 on
  ``synthetic.unclosed``.

### Active moves (SDK v3.3 — still in force)

- **PARSE_OUTCOME lifecycle gate.** Theorem W3-39.
- **Runtime-checkable lifecycle audit.** Theorem W3-40 / W3-45.
- **Deterministic-mode replay.** Theorem W3-41.

### Sharp limitation theorems we hold

- **W3-14** (negative): per-capsule budgets cannot enforce
  table-level cardinality invariants.
- **W3-16** (negative): cohort-lifting cannot enforce relational
  invariants.
- **W3-17** (conditional): admission rules cannot exceed the
  priority-decoder ceiling under ceiling-forcing spurious
  injection.
- **W3-21** (negative): linear class-agnostic decoders cannot
  achieve symmetric zero-shot transfer when gold-conditional
  feature signs flip across domains.
- **W3-29** (lower bound): magnitude-monoid linear family cannot
  cross the Bayes-divergence zero-shot risk lower bound.
- **W3-36** (sharp impossibility): the primary capsule ledger
  cannot authenticate its own rendering's bytes.
- **W4-3 (SDK v3.5)** (proved-negative): per-role budget below
  the role's causal-share floor admits sound runs that fail the
  team gate; no admission policy (FIFO, priority, coverage,
  learned) can recover. The natural next move is a
  cohort-lifted role view (W4-C2, conjectural).

### Active conjectures

- **W3-C1**: every Phase-N bounded-context theorem subsumes under
  the capsule contract. Conjectural (the four-case subsumption is
  proved; the general statement is open).
- **W3-C5 (legacy SDK v3.3)**: a sub-intra-cell PROMPT /
  LLM_RESPONSE capsule slice closes the inner-loop boundary
  without breaking W3-34 spine equivalence. **DISCHARGED in
  SDK v3.4** by Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **W3-C6 (new in SDK v3.4)**: synthetic-LLM cross-distribution
  PARSE_OUTCOME failure-kind TVD ≥ 0.5 across the calibrated
  distribution library. **Empirical** (reproducible; not a
  proof — the distribution library is synthetic, not real
  cross-LLM).
- **W3-C9**: refined paradigm-shift reading (Phase-49 candidate at
  $n=80$ point-estimate, zero-shot gap reading).
- **W3-C10**: relational decoder level-ceiling.
- **W4-C1 (SDK v3.5)**: learned per-role admission policy
  admits strictly fewer handoffs (12/12 seeds, robust direction)
  and improves pooled team-decision accuracy on most train seeds
  (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause
  > 0 in 8/12 seeds, mean +0.032) over the strongest fixed
  admission baseline (coverage-guided) on the Phase-52 default
  config — but the accuracy advantage reverses at higher noise
  (spurious=0.50). Empirical: budget-efficiency dominance is
  robust per-seed; accuracy advantage is mean-positive, not
  strict per-seed.
- **W4-C2 (SDK v3.5)**: cohort-lifted role view closes W4-3 on a
  sub-class of scenarios.
- **W4-C3 (SDK v3.5)**: capsule-layer admission rule subsumes
  the Phase-36 ``AdaptiveSubscriptionTable`` route-edit primitive.

### Active retractions

- **W3-C7 (strict reading) is retracted.** "Point-estimate
  Gate 1 at $\hat p \ge 0.400$ AND strict zero-shot Gate 2 with
  per-direction penalty ≤ 5pp" was falsified at $n=320$ (W3-26,
  W3-27). Do not reintroduce the strict bar.
- **W3-C3** is retracted in favour of W3-15 cohort lift.
- **The earlier W3-C4** (now reused for SDK-v3.3
  PARSE_OUTCOME conjecture) named a candidate decoder paradigm
  shift; the strict reading is folded into W3-C7 retraction.

## What we are NOT actively claiming

- **Not** "we solved context."
- **Not** "we solved multi-agent context." SDK v3.14's W13-1 result
  is the strongest *open-world-drift-under-bounded-heuristic-
  closure* structural-win the programme has produced
  (LayeredRobustMultiRoundBundleDecoder wins on R-60-wide by +1.000
  vs every fixed-vocabulary method **including SDK v3.13 W12**;
  backward-compatible on R-54..R-58 / R-59 / R-60-clean; stable
  across 5/5 (bank_seed, llm_seed) values; named bench property +
  named falsifier regime W13-4), but it is still **conditional on**
  (a) the bench property (R-58 delayed-causal-evidence shape with
  the Phase-60 wide-OOV drift channel), (b) the producer-noise
  channel being bounded by the heuristic closure, AND (c) round-N
  admission not being budget-starved (inherits W11-4). On a real
  Ollama producer at the 14B class, the bench property does NOT
  hold by default (W13-Λ-real); the synthetic→real-LLM transfer is
  gated by event-shape design + prompt-side discipline, not by
  normalisation. This is an honest empirical finding, not a closure
  of the question.

  SDK v3.13's W12-1 result remains the strongest *real-LLM-shaped-
  stream* (synthetic noisy) structural-win the programme has
  produced (RobustMultiRoundBundleDecoder wins on
  R-59 by +1.000 vs every un-normalised single-round / multi-round
  method **including SDK v3.12 W11**; backward-compatible on
  R-54 / R-55 / R-56 / R-57 / R-58 / R-59-clean; stable across
  5/5 (bank_seed, llm_seed) values; named bench property + named
  falsifier regime W12-4), but it is still **conditional** on
  (a) the bench property (R-58 delayed-causal-evidence shape),
  (b) the producer-noise channel being bounded by the closed-
  vocabulary closure (every variant in :data:`NOISY_KIND_VARIANTS`
  is in :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission
  not being budget-starved (inherits W11-4). The synthetic-noisy-
  LLM extractor is calibrated against Phase-53 14B/35B empirical
  distributions; the ``ollama`` opt-in mode is the W12-C2 next
  data point. Real multi-agent teams have additional axes
  (heterogeneous producers, time-varying budgets, multi-round
  handoffs with > 2 rounds and inter-round contradictions,
  conflicting goals, generic-tier root_causes the bundle decoder
  cannot help with, OOV kinds outside any reasonable closure) the
  W12 family does not cover. The W4-2 result is proved-conditional
  (premises: faithful decoder + sound admission); the W4-C1 learned-
  policy advantage is conditional empirical-positive on the SDK v3.5
  config and falsified out-of-distribution on the SDK v3.7 real-LLM
  regime. External validity to real production multi-agent teams is
  *materially* advanced by SDK v3.13 (the first synthetic→real-LLM
  transfer move with a named bounded-noise channel) but not fully
  closed.
- **Not** "the runtime is fully capsule-native." Specifically not
  capsule-native: sandbox stdout/stderr, sub-step parser-internal
  objects (regex match objects, recovery heuristic intermediate
  state), and on-the-wire LLM streaming chunks. PROMPT bytes and
  LLM_RESPONSE bytes ARE now capsule-tracked under SDK v3.4 (the
  prior "not capsule-native: LLM prompt bytes, raw LLM response
  bytes" line is **superseded** by Theorems W3-42 / W3-43).
- **Not** "CoordPy is a universal multi-agent platform."
- **Not** "the decoder beat the Phase-31 ceiling by 22.5 pp."
  The sharper reading is W3-19 ($+15$pp at $n=80$, FIFO admission).
- **Not** "deterministic mode means the entire run is
  reproducible." It means the *capsule DAG* is reproducible under
  a frozen JSONL + a deterministic profile. Wall-clock and
  host-local fields are stripped from CIDs but live on disk.
- **Not** "the synthetic-LLM cross-distribution study is a real
  cross-LLM study." The distributions are calibrated synthetic
  (see ``synthetic_llm.SYNTHETIC_MODEL_PROFILES``), not
  measurements of ``gemma2:9b`` / ``qwen2.5:7b`` outputs. The
  empirical claim is about the parser's failure-kind closed
  vocabulary's *resolving power*, not about LLM output
  distributions in the wild.

## How to update this document

1. When a phase ships, add one line to the "Active moves" list and
   move any superseded line to "Sharp limitation theorems we hold"
   or "Active retractions" as appropriate.
2. When a conjecture is proved or falsified, move it to the
   correct section and update `THEOREM_REGISTRY.md`.
3. When a milestone note ships, add a one-line cross-link in this
   doc's relevant section.
4. Always update the "Last touched" date at the top.

## Cross-references

- Formal model (run-boundary, W3 family): `docs/CAPSULE_FORMALISM.md`
- Formal model (team-boundary, W4 family): `docs/CAPSULE_TEAM_FORMALISM.md`
- Theorem registry: `docs/THEOREM_REGISTRY.md`
- How-not-to-overstate rules: `docs/HOW_NOT_TO_OVERSTATE.md`
- Master plan: `docs/context_zero_master_plan.md`
- Milestone notes: `docs/archive/coordpy-milestones/RESULTS_COORDPY_*.md`,
  `docs/archive/capsule-research/RESULTS_CAPSULE_*.md` (historical),
  `docs/archive/coordpy-milestones/RESULTS_COORDPY_DEEP_INTRA_CELL.md` (SDK v3.3),
  `docs/archive/coordpy-milestones/RESULTS_COORDPY_INNER_LOOP.md` (SDK v3.4),
  `docs/archive/coordpy-milestones/RESULTS_COORDPY_TEAM_COORD.md` (SDK v3.5),
  `docs/archive/coordpy-milestones/RESULTS_COORDPY_DISTRIBUTED.md` (SDK v3.6),
  `docs/RESULTS_COORDPY_SCALE_VS_STRUCTURE.md` (SDK v3.7),
  `docs/RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md` (SDK v3.8),
  `docs/RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md` (SDK v3.9 — this milestone),
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` (SDK v3.9 — pre-committed bar)
- Paper draft: `papers/coordpy_capsule_native_runtime.md`
- Tests: `vision_mvp/tests/test_coordpy_capsule_native*.py`,
  `test_coordpy_capsule_native_deeper.py`,
  `test_coordpy_capsule_native_inner_loop.py` (SDK v3.4),
  `test_coordpy_team_coord.py` (SDK v3.5 — multi-agent slice),
  `test_coordpy_scale_vs_structure.py` (SDK v3.7 — Phase-53),
  `test_coordpy_cross_role_coherence.py` (SDK v3.8 — Phase-54 + W7),
  `test_capsule_*.py`
- Cross-model parser-boundary experiment:
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Multi-agent team coordination benchmark (synthetic):
  `vision_mvp/experiments/phase52_team_coord.py`
- Stronger-model multi-agent benchmark (real LLM):
  `vision_mvp/experiments/phase53_scale_vs_structure.py`
- Cross-role cohort-coherence benchmark (deterministic):
  `vision_mvp/experiments/phase54_cross_role_coherence.py`
- Cross-role corroboration benchmark (deterministic, harder):
  `vision_mvp/experiments/phase55_decoy_plurality.py`
- MLX distributed runbook (operator path for Mac 2):
  `docs/MLX_DISTRIBUTED_RUNBOOK.md`
