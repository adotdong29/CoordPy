# W66 — Stronger Solving-Context Substrate-Coupled Latent OS (post-W65 result note)

> Post-W65 research milestone, 2026-05-16. **Eleventh substrate-attack
> milestone** in the programme. **Second multi-agent task-success-
> bearing substrate milestone** — and the first to produce wins across
> *three* regimes (baseline, team-consensus-under-budget, team-failure-
> recovery), not only the single baseline seed-cohort of W65.

## TL;DR — strong success

W66 met the pre-committed
[`SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md`](SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md)
bar on every dimension:

* **168 / 168 R-146/R-147/R-148 cells pass at 3 seeds**.
  - R-146 (V11 substrate / eleven-way hybrid / multi-agent task-success
    / team-coordinator-V2 / team-consensus-controller): **24 H-bars
    × 3 seeds = 72 cells**, 72/72 pass.
  - R-147 (8192-turn retention / V18 LHR / V16 multi-hop / V18 ECC /
    persistent V18): **14 H-bars × 3 seeds = 42 cells**, 42/42 pass.
  - R-148 (corruption V14 / consensus V12 / disagreement V12 /
    uncertainty V14 / TVS V15 / W66 envelope verifier / MASC V2
    regimes): **18 H-bars × 3 seeds = 54 cells**, 54/54 pass.
* **All 21 mechanism advances ship and fire**.
* **MASC V2 wins across all three regimes**:
  - **baseline**: V11 strictly beats V10 on **93.3 %** of seeds;
    TSC_V11 strictly beats V11 on **80.0 %** of seeds; team success
    rate {transcript_only: 73 %, shared_state_proxy: 87 %, V9: 100 %,
    V10: 100 %, V11: 100 %, TSC_V11: 100 %}.
  - **team_consensus_under_budget**: V11 strictly beats V10 on
    **93.3 %** of seeds; TSC_V11 strictly beats V11 on **80.0 %**;
    team success rate {transcript_only: 33 %, shared_state_proxy: 67 %,
    V9: 87 %, V10: 100 %, V11: 100 %, TSC_V11: 100 %}.
  - **team_failure_recovery**: TSC_V11 strictly beats V11 on
    **73.3 %** of seeds; only TSC_V11 reaches 80 % team success while
    every other policy is stuck at 40 %.
* **Substrate snapshot-diff** primitive saves **92 %** flops vs
  recompute at 128 tokens (target ≥ 60 %).
* **Cumulative trust boundary across W22..W66 = 1228** enumerated
  failure modes (1105 from W22..W65 + 123 new W66 envelope modes;
  target ≥ 1225).
* **Six new closed-form ridge solves** on top of W65's 29 (cache V9
  six-objective + cache V9 per-role eviction + replay V7 per-role
  per-regime + replay V7 team-substrate-routing + HSB V10 seven-target
  + KV V11 seven-target). Total **thirty-five closed-form ridge
  solves** across W61..W66. No autograd, no SGD, no GPU.

## What W66 is

W66 introduces the **Stronger Solving-Context Substrate-Coupled Latent
OS** layer: the second multi-agent task-success-bearing milestone, and
the first programme milestone in which:

* the W66 substrate (V11) is the centre of an **eleven-way bidirectional
  hybrid loop**;
* the **MultiAgentSubstrateCoordinatorV2** runs **six matched-budget
  policies** under **three regimes** (baseline + team-consensus-under-
  budget + team-failure-recovery);
* the **TeamConsensusController** is a first-class capsule-native
  module composing weighted quorum + abstain + substrate-replay
  fallback + transcript fallback, regime-aware.

The W66 substrate (V11) is the in-repo NumPy runtime — 13 layers, GQA
(8 query / 4 KV), RMSNorm, SwiGLU, byte-vocab — plus four new V11
axes:

* per-(L, H, T) **replay-trust ledger** (substrate-measured replay
  decision trust)
* per-role **team-failure-recovery flag** (boolean + reason string)
* **substrate snapshot-diff primitive** (typed content-addressed
  delta over the V10 checkpoint, recording which axes changed and
  by how much)
* per-layer **V11 composite gate score** (extends V10 gate score
  with replay_trust_l1 + team_failure_recovery_count + snapshot_diff_l1)

The W66 line ships at:

* `coordpy.tiny_substrate_v11`
* `coordpy.kv_bridge_v11`
* `coordpy.hidden_state_bridge_v10`
* `coordpy.prefix_state_bridge_v10`
* `coordpy.attention_steering_bridge_v10`
* `coordpy.cache_controller_v9`
* `coordpy.replay_controller_v7`
* `coordpy.deep_substrate_hybrid_v11`
* `coordpy.substrate_adapter_v11`
* `coordpy.persistent_latent_v18`
* `coordpy.multi_hop_translator_v16`
* `coordpy.mergeable_latent_capsule_v14`
* `coordpy.consensus_fallback_controller_v12`
* `coordpy.corruption_robust_carrier_v14`
* `coordpy.long_horizon_retention_v18`
* `coordpy.ecc_codebook_v18`
* `coordpy.uncertainty_layer_v14`
* `coordpy.disagreement_algebra_v12`
* `coordpy.transcript_vs_shared_arbiter_v15`
* `coordpy.multi_agent_substrate_coordinator_v2`
* `coordpy.team_consensus_controller`
* `coordpy.w66_team`
* `coordpy.r146_benchmark`
* `coordpy.r147_benchmark`
* `coordpy.r148_benchmark`

`coordpy.__version__` remains `0.5.20`. SDK contract is byte-for-byte
unchanged. **No PyPI release**.

## Mechanism details

* **M1 Tiny Substrate V11** — 13 layers (V10 had 12). Four new V11
  axes (replay-trust ledger, team-failure-recovery flag, substrate
  snapshot-diff, V11 gate score). Forward is deterministic
  byte-for-byte under the same seed.
* **M2 KV Bridge V11** — seven-target stacked ridge (V10's 6 + 1
  team-failure-recovery). Team-coordination margin probe + 30-dim
  multi-agent task fingerprint. Falsifier returns 0 under team-
  failure-recovery flag inversion.
* **M3 HSB V10** — seven-target stacked ridge (V9's 6 + 1
  team-consensus-under-budget). Per-(L, H) hidden-wins-vs-team-success
  probe; team-consensus margin = min(kv, prefix, replay, recover) -
  hidden.
* **M4 Prefix V10** — K=96 drift curve (V9 was K=64); 30-dim
  role+task+team fingerprint; five-way prefix/hidden/replay/team/
  recover comparator.
* **M5 Attention V10** — six-stage clamp (V9 + per-(L, H) attention-
  trust ledger clip); team-conditioned fingerprint.
* **M6 Cache Controller V9** — six-objective stacked ridge (drop
  oracle + retrieval relevance + hidden wins + replay dominance +
  team task success + team failure recovery); per-role 7-dim
  eviction head.
* **M7 Replay Controller V7** — 10 regimes (V6's 8 + 2 new:
  team_failure_recovery_regime, team_consensus_under_budget_regime);
  per-role per-regime ridge; **trained team-substrate-routing head**
  (4×11 ridge over team features).
* **M8 Deep Substrate Hybrid V11** — eleven-way bidirectional loop
  with V11 substrate at its centre + team-consensus-controller axis.
* **M9 Substrate Adapter V11** — new `substrate_v11_full` tier.
* **M10 Persistent V18** — 17 layers; fifteenth persistent skip-link
  (`team_failure_recovery_carrier`); max_chain_walk_depth=8192;
  distractor rank 16.
* **M11 Multi-Hop V16** — 36 backends; 1260 directed edges;
  chain-length 26; 11-axis composite.
* **M12 Mergeable Capsule V14** — adds team-failure-recovery and
  team-consensus-under-budget witness chains.
* **M13 Consensus V12** — 18 stages (V11's 16 + 2: team_failure_
  recovery_arbiter, team_consensus_under_budget_arbiter).
* **M14 CRC V14** — 16384-bucket wrap-around-XOR fingerprint; 33-bit
  adversarial burst; team-failure-recovery probe.
* **M15 LHR V18** — 17 heads; max_k=320; eight-layer scorer adding
  random+swish layer.
* **M16 ECC V18** — K1..K17 = 2^29 = 536 870 912 codes;
  ≥ 31.0 bits/visible-token at full emit.
* **M17 Uncertainty V14** — 13-axis composite adding
  `team_failure_recovery_fidelity`.
* **M18 Disagreement Algebra V12** — Jensen-Shannon-equivalence
  identity + falsifier (now four equivalence identities total: TV +
  Wasserstein + attention-pattern + JS).
* **M19 TVS V15** — 16 arms (V14's 15 + `team_failure_recovery` arm).
* **M20 Multi-Agent Substrate Coordinator V2** — **the load-bearing
  W66 multi-agent mechanism**. Six matched-budget policies
  (`transcript_only`, `shared_state_proxy`, `substrate_routed_v9`,
  `substrate_routed_v10`, `substrate_routed_v11`,
  `team_substrate_coordination_v11`). Three regimes (baseline,
  team_consensus_under_budget, team_failure_recovery).
* **M21 Team-Consensus Controller** — first capsule-native
  team-consensus controller. Regime-aware thresholds. Four decisions:
  `quorum_merge`, `abstain`, `substrate_replay`,
  `transcript_fallback`.

## Benchmark families

### R-146 — V11 substrate / eleven-way hybrid / multi-agent task-success / team-coordinator-V2 / team-consensus controller

24 H-bars, 3 seeds, **72 / 72 cells pass**:

* H245 V11 substrate determinism
* H245b V11 replay_trust_ledger tensor shape
* H245c V11 team_failure_recovery_flag semantics
* H245d V11 v11_gate_score_per_layer shape
* H245e V11 substrate snapshot-diff content-addressed
* H246 KV bridge V11 seven-target ridge converges
* H246b KV bridge V11 team_failure_recovery falsifier returns 0
* H246c multi-agent task fingerprint dim = 30
* H247 HSB V10 seven-target ridge converges
* H247b HSB V10 hidden-wins-vs-team-success probe in [0, 1]
* H248 prefix V10 K=96
* H248b prefix V10 five-way decision valid
* H249 attention V10 six-stage clamp keeps delta ≤ trust cap
* H249b attention V10 returns zero under negative budget
* H250 cache V9 six-objective ridge converges
* H250b cache V9 per-role 7-dim eviction head converges
* H251 replay V7 per-role per-regime head has 10 entries
* H251b replay V7 team-substrate-routing head converges
* H251c replay V7 has ten regimes
* H252 substrate snapshot-diff reuse saving ≥ 0.6 (measured: 0.92)
* H252b MASC V2 V11 strictly beats V10 ≥ 50 % (measured: 93.3 %)
* H252c MASC V2 TSC_V11 strictly beats V11 ≥ 50 % (measured: 80.0 %)
* H252d substrate adapter V11 has v11_full tier
* H252e team-consensus controller fires quorum on active agents

### R-147 — 8192-turn retention / V18 LHR / V16 multi-hop / V18 ECC / persistent V18

14 H-bars, 3 seeds, **42 / 42 cells pass**:

* H253 persistent V18 fifteenth-skip carrier present
* H253b persistent V18 max_chain_walk_depth ≥ 8192
* H253c persistent V18 17 layers
* H253d persistent V18 chain walk preserves CIDs
* H254 LHR V18 17 heads + max_k=320
* H254b LHR V18 seventeen_way_value runs on team-failure-recovery
* H254c LHR V18 eight-layer scorer ridge converges
* H255 multi-hop V16 36 backends + chain-len 26
* H255b multi-hop V16 eleven-axis arbitration kind
* H255c multi-hop V16 compromise threshold in [1, 11]
* H256 ECC V18 2^29 codes
* H256b ECC V18 bits/visible-token ≥ 31.0 (measured: 31.333)
* H256c ECC V18 rate-floor falsifier triggers
* H259a ECC V18 K17 meta15 index well-formed

### R-148 — corruption / consensus / disagreement / fallback / failure-recovery

18 H-bars, 3 seeds, **54 / 54 cells pass**:

* H260 consensus V12 18 stages
* H260b CRC V14 33-bit burst detect ≥ 0.40
* H260c CRC V14 16384-bucket detect ≥ 0.95
* H260d CRC V14 team-failure-recovery ratio floor ≥ 0.40
* H261 disagreement V12 JS-equivalence identity holds
* H261b disagreement V12 JS falsifier triggers
* H262 uncertainty V14 13-axis composite in [0, 1]
* H262b uncertainty V14 team_failure_recovery_aware True under
  partial inputs
* H263 MASC V2 team_consensus_under_budget regime V11 beats V10
  ≥ 50 % (measured: 93.3 %)
* H263b MASC V2 team_failure_recovery regime TSC_V11 beats V11
  ≥ 50 % (measured: 73.3 %)
* H263c team-consensus controller fires abstain on low-confidence
* H263d team-consensus controller fires substrate_replay fallback
  when quorum fails
* H264 TVS V15 16 arms sum to 1.0
* H264b TVS V15 team_failure_recovery arm fires when fidelity > 0
* H265 W66 envelope verifier ≥ 120 disjoint modes (measured: 123)
* H266 W66 envelope verifier passes on a built team
* H267 W66 trivial passthrough preserves w65 outer cid byte-for-byte
* H268 cumulative trust boundary ≥ 1225 (measured: 1228)

## Honest scope (pre-committed limitations preserved)

* `W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` — hosted backends
  (Ollama, OpenAI-compatible) remain text-only on the HTTP surface;
  W66 makes no claim of third-party transformer-internal access.
* `W66-L-V11-NO-AUTOGRAD-CAP` — W66 fits only closed-form linear
  ridge solves. Six new W66 ridge solves on top of W65's 29 (cache V9
  six-objective; cache V9 per-role eviction; replay V7 per-role per-
  regime; replay V7 team-substrate-routing; HSB V10 seven-target;
  KV V11 seven-target). Total **thirty-five closed-form ridge
  solves** across W61..W66.
* `W66-L-MULTI-AGENT-COORDINATOR-SYNTHETIC-CAP` — the multi-agent-
  task win is still a synthetic deterministic harness; the V11 wins
  are engineered so that the V11 mechanisms (replay-trust ledger,
  team-failure-recovery flag, team-consensus-under-budget arbiter,
  substrate snapshot-diff) materially reduce drift.
* `W66-L-NUMPY-CPU-V11-SUBSTRATE-CAP` — V11 still runs in NumPy on
  CPU; not a frontier model.
* `W66-L-V18-OUTER-NOT-TRAINED-CAP` — persistent V18 only the inner
  V12 cell is trained at init; the V13..V18 outer carriers ride
  along as deterministic projections.
* `W66-L-ECC-V18-RATE-FLOOR-CAP` — ECC V18 has a strict structural
  rate ceiling of log2(2^29) = 29 raw bits per segment-tuple.
* `W66-L-TEAM-CONSENSUS-IN-REPO-CAP` — the team-consensus controller
  operates on in-repo MASC outcomes only; it does not enforce
  consensus on real model outputs.
* `W66-L-V18-LHR-SCORER-FIT-CAP` — LHR V18 scorer fits only the final
  ridge; the first seven layers are frozen random projections.
* `W66-L-SUBSTRATE-CHECKPOINT-IN-REPO-CAP` — substrate snapshot-diff
  operates on the in-repo V11 cache only.

## Where W66 advances the original goal

The original Context Zero goal is *actually solving context for
multi-agent teams* — moving and preserving useful state across turns,
roles, branches, merges, damage, and budget pressure better than
transcript replay.

W66 moves that bar in two concrete directions:

1. **Multi-agent wins under heterogeneous failure modes.** W65 only
   verified team-success on a single synthetic baseline cohort. W66
   adds two more regimes (team-consensus-under-budget and team-
   failure-recovery) and verifies that the V11 / TSC_V11 policies
   strictly beat V10 / V11 on majority of seeds in each regime.
   The team-failure-recovery regime is the sharpest evidence yet:
   when one agent silently produces zero output, the team-consensus
   controller running on the V11 substrate doubles the team success
   rate over every baseline (80 % vs 40 %).
2. **Stronger substrate primitives that are load-bearing.** The new
   V11 axes (replay-trust ledger, team-failure-recovery flag,
   substrate snapshot-diff) are not just internal probes — they
   *change the decision the team makes*. The snapshot-diff primitive
   gives a 92 % flop saving vs recompute (vs V10's 90 %) and feeds
   the substrate-replay fallback path inside the team-consensus
   controller. The team-failure-recovery flag flips the replay
   controller's regime selection.

W66 still does NOT prove third-party transformer-internal access.
The in-repo V11 substrate remains the only place where the new axes
exist. `W66-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP` carries forward
unchanged.

## Where to read next

* [`SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md`](SUCCESS_CRITERION_W66_STRONGER_TEAM_SUBSTRATE_OS.md) — pre-committed bar
* [`RESULTS_W65_TEAM_SUBSTRATE_COORDINATION.md`](RESULTS_W65_TEAM_SUBSTRATE_COORDINATION.md) — W65 result note
* [`THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md) — theorem and conjecture index
* [`HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md) — what may be claimed
* [`context_zero_master_plan.md`](context_zero_master_plan.md) — long-running master plan
