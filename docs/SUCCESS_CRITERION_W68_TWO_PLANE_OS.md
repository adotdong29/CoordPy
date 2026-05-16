# W68 — Two-Plane Substrate-Coupled Latent Operating System (pre-committed success criterion)

> Post-W67 research milestone, 2026-05-16. **Thirteenth** substrate-
> attack milestone in the programme. **Fourth** multi-agent task-
> success-bearing milestone — and the first to **explicitly split
> the architecture into two planes** (hosted control plane and real
> substrate plane) so honest claims and the blocked frontier are
> separated cleanly. The bar is written BEFORE any code change so
> the result can be checked against this file. **No version bump;
> no PyPI release.**

## What W68 is supposed to do

After W67 produced multi-agent task wins under five failure-mode
regimes, the structural blocker is no longer "better coordination
ideas." It is **substrate access at deployment time**: hosted APIs
(OpenRouter, Groq, OpenAI-compat) do not expose hidden state / KV /
attention at the HTTP surface, so any substrate work that wants to
ship has to split into two planes:

1. **Plane A — Hosted control plane.** Runs over text /
   optional-logprobs / optional-prefix-cache. Useful for cheap eval
   coverage, provider routing, cost/latency planning, prefix-cache-
   aware planning. **Does not** pierce the hosted substrate
   boundary.
2. **Plane B — Real substrate plane.** Runs on runtimes we control
   (in-repo V13 substrate). Honestly exposes hidden state / KV /
   attention / branch-merge / role-dropout / partial-contradiction /
   agent-replacement / prefix-reuse axes. Load-bearing for real
   context preservation.
3. **Still-blocked frontier:** the *combination* of frontier-model
   text quality and full substrate access. W68 does not claim to
   solve this — it codifies the wall.

Two new multi-agent regimes are added on top of the W67 five so
that the V13 substrate has more work to do:

* **`partial_contradiction_under_delayed_reconciliation`** —
  agents produce conflicting payloads with delayed arbitration.
  The V13 substrate's partial-contradiction witness drives the
  reconciliation.
* **`agent_replacement_warm_restart`** — at a mid-task checkpoint
  a role's primary agent is replaced; the replacement warm-starts
  from a substrate snapshot. The V13 substrate's agent-replacement
  flag + warm-restart window drive the recovery.

W68's bar: V13 must produce **multi-agent task wins across seven
regimes** (W67's five + the two new), the hosted control plane
must ship at least six honest hosted-only mechanisms, **and** the
hosted-vs-real wall must be a first-class benchmark family.

## Pre-committed quantitative bars

### A. Mechanism advances over W67

W68 must ship at least **twelve** substantial mechanism advances on
top of W67. The pre-committed list is **twenty-eight**:

**Plane B — Real substrate (22 advances):**

* M1 `coordpy.tiny_substrate_v13` — 15 layers (vs V12's 14); four
  new V13 axes (partial-contradiction witness, agent-replacement
  flag w/ warm-restart, prefix-reuse counter, V13 composite gate).
* M2 `coordpy.kv_bridge_v13` — 9-target stacked ridge (8 V12 + 1
  partial-contradiction); substrate-measured partial-contradiction
  margin; **50-dim agent-replacement fingerprint**; **partial-
  contradiction falsifier**.
* M3 `coordpy.hidden_state_bridge_v12` — 9-target stacked ridge with
  agent-replacement target; per-(L, H) hidden-vs-agent-replacement
  probe.
* M4 `coordpy.prefix_state_bridge_v12` — **K=192** drift curve;
  role+task+team+branch+agent 50-dim fingerprint; **7-way**
  comparator adding `contradict`.
* M5 `coordpy.attention_steering_bridge_v12` — **eight-stage** clamp
  (V11 + partial-contradiction attention bias); agent-conditioned
  attention fingerprint.
* M6 `coordpy.cache_controller_v11` — **eight-objective** stacked
  ridge (adds partial-contradiction); per-role 9-dim agent-
  replacement priority head.
* M7 `coordpy.replay_controller_v9` — **14 regimes** (adds
  `partial_contradiction_under_delayed_reconciliation_regime`,
  `agent_replacement_warm_restart_regime`); per-role per-regime
  ridge; trained **agent-replacement-routing head** (6 labels × 11
  features ridge).
* M8 `coordpy.deep_substrate_hybrid_v13` — **thirteen-way**
  bidirectional loop adding partial-contradiction-witness axis.
* M9 `coordpy.substrate_adapter_v13` — new `substrate_v13_full`
  tier; V13 capability axes.
* M10 `coordpy.persistent_latent_v20` — **19 layers**; seventeenth
  skip carrier `partial_contradiction_carrier`;
  `max_chain_walk_depth=32768`.
* M11 `coordpy.multi_hop_translator_v18` — **44 backends**;
  chain-length 34; **13-axis** composite adding
  `partial_contradiction_reconciliation_trust`.
* M12 `coordpy.mergeable_latent_capsule_v16` — adds
  `partial_contradiction_witness_chain` and
  `agent_replacement_witness_chain`.
* M13 `coordpy.consensus_fallback_controller_v14` — **22-stage**
  chain inserting `partial_contradiction_arbiter` and
  `agent_replacement_warm_restart_arbiter`.
* M14 `coordpy.corruption_robust_carrier_v16` — **65536-bucket**
  fingerprint; **36-bit** adversarial burst; partial-contradiction
  recovery probe.
* M15 `coordpy.long_horizon_retention_v20` — **19 heads**,
  **max_k=448**; partial-contradiction-conditioned head.
* M16 `coordpy.ecc_codebook_v20` — K1..K19 = **2^33 = 8 589 934 592
  codes**; **≥ 35.0 bits/visible-token** at full emit.
* M17 `coordpy.uncertainty_layer_v16` — **15-axis** composite adding
  `partial_contradiction_resolution_fidelity`.
* M18 `coordpy.disagreement_algebra_v14` — **agent-replacement-
  equivalence identity** + falsifier.
* M19 `coordpy.transcript_vs_shared_arbiter_v17` — **18-arm**
  comparator adding `partial_contradiction_resolution`.
* M20 `coordpy.multi_agent_substrate_coordinator_v4` — ten-policy
  MASC V4 across **seven regimes**; V13 strictly beats V12;
  TSC_V13 strictly beats TSC_V12.
* M21 `coordpy.team_consensus_controller_v3` — regime-aware
  weighted quorum + partial-contradiction arbiter + agent-
  replacement-warm-restart arbiter.
* M22 `coordpy.w68_team` — orchestrator composing all 22 Plane B
  modules + all 6 Plane A modules into a `W68HandoffEnvelope` that
  carries `w67_outer_cid` byte-for-byte.

**Plane A — Hosted control plane (6 advances):**

* H1 `coordpy.hosted_router_controller` — content-addressed
  provider registry + capability-aware routing decision
  (cheapest-then-fastest with data-policy and tier filters).
* H2 `coordpy.hosted_logprob_router` — honest top-k logprob fusion
  over hosted providers that expose logprobs; text-only quorum
  fallback when none do.
* H3 `coordpy.hosted_cache_aware_planner` — per-turn prefix-CID
  planner that maximises hosted prefix-cache hit rates;
  cross-plane bridge to the V13 substrate's prefix-reuse counter.
* H4 `coordpy.hosted_provider_filter` — data-policy-aware filter
  (`no_log`, `no_train`) and tier filter; produces a derived
  content-addressed registry.
* H5 `coordpy.hosted_cost_planner` — cost/latency-aware provider
  selection under a matched-quality constraint.
* H6 `coordpy.hosted_real_substrate_boundary` — explicit
  architecture-wall assertion module; `HostedRealSubstrateBoundary`
  + falsifier; `HostedRealSubstrateWallReport` listing the
  hosted-solvable / real-substrate-only / blocked-frontier axes.

### B. Four benchmark families

W68 must ship at least **three** benchmark families. The pre-
committed list is **four**:

* **R-152 — Hosted control plane (Plane A)**: provider routing
  determinism, logprob fusion + text-only fallback, prefix-CID
  content-addressing, cache-aware savings, provider filter,
  cost planner, hosted/real wall (≥ 10 H-bars).
* **R-153 — Real substrate (Plane B)**: V13 substrate determinism +
  new V13 axes, KV V13 nine-target ridge + falsifier, HSB V12
  nine-target ridge, prefix V12 K=192, cache V11 eight-objective,
  replay V9 fourteen-regime + agent-replacement-routing, hidden-vs-
  KV pillar, transcript-vs-shared-state pillar, substrate adapter
  V13 (≥ 14 H-bars).
* **R-154 — Multi-agent task success (across all seven regimes)**:
  baseline / team_consensus_under_budget / team_failure_recovery /
  role_dropout / branch_merge_reconciliation /
  partial_contradiction / agent_replacement_warm_restart — V13
  strictly beats V12 on ≥ 50 % of seeds in every regime;
  TSC_V13 strictly beats TSC_V12 on ≥ 50 % of seeds in every
  regime; V13 saves visible tokens vs transcript-only (≥ 12
  H-bars).
* **R-155 — Hosted-vs-real wall**: boundary is content-addressed;
  ≥ 15 blocked axes at the hosted surface; ≥ 60 real-substrate-only
  V13 axes; hosted-cache-aware savings ≥ 50 % at hit_rate=1.0;
  real-substrate branch-merge ≥ 60 % flops saving; falsifier
  triggers on dishonest hosted hidden-state claim (≥ 4 H-bars).

### C. Seven-regime multi-agent wins

V13 (substrate_routed_v13) and TSC_V13 (team_substrate_
coordination_v13) must each strictly beat the V12 / TSC_V12
predecessors on ≥ 50 % of seeds in **every** regime:

* baseline
* team_consensus_under_budget
* team_failure_recovery
* role_dropout
* branch_merge_reconciliation
* **partial_contradiction_under_delayed_reconciliation** (new)
* **agent_replacement_warm_restart** (new)

### D. Long-horizon and compression bars

* Persistent V20 chain-walk depth ≥ 32 768.
* Persistent V20 has seventeenth skip carrier.
* Long-horizon retention V20 max_k = 448, 19 heads.
* ECC V20 ≥ 35.0 bits/visible-token at full emit, total codes
  ≥ 2^32.

### E. Corruption / disagreement / consensus bars

* CRC V16 65536-bucket fingerprint single-byte detect rate ≥ 0.9.
* CRC V16 36-bit adversarial burst detect rate ≥ 0.0 (stress test).
* Consensus V14 ≥ 22-stage chain.
* Consensus V14 `partial_contradiction_arbiter` fires under
  contradictions.
* Consensus V14 `agent_replacement_warm_restart_arbiter` fires
  when a parent has a replacement score above threshold.
* Disagreement Algebra V14 agent-replacement-equivalence identity
  holds iff argmax preserved AND symmetric-KL ≤ floor AND
  fingerprint matches.

### F. Hosted / wall bars

* HostedRouterController decision is deterministic on
  (registry CID, request CID).
* HostedLogprobRouter fuses on shared top-k when present; falls
  back to text-only quorum when no logprobs.
* HostedCacheAwarePlanner prefix CID is content-addressed across
  invocations.
* HostedCacheAwarePlanner ≥ 50 % token savings on 4-turn plans at
  hosted-cache-hit-rate = 1.0.
* HostedProviderFilter drops train-policy providers when
  `require_data_policy="no_log"`.
* HostedCostPlanner picks the cheapest eligible provider and
  abstains when no provider passes the quality floor.
* HostedRealSubstrateBoundary falsifier returns 0 on honest
  claims and 1 on dishonest hidden-state claims at the hosted
  surface.

### G. Falsifiers and limitation reproductions

* W68-F-V13-NO-AUTOGRAD — V13 ridge solves remain closed-form
  linear (no SGD, no autograd, no GPU).
* W68-F-V13-NO-THIRD-PARTY-SUBSTRATE — V13 substrate is in-repo
  NumPy; not bridging to third-party hosted models.
* W68-F-PARTIAL-CONTRADICTION-IN-REPO — partial-contradiction
  primitive operates on the in-repo V13 cache only.
* W68-F-AGENT-REPLACEMENT-IN-REPO — agent-replacement-warm-restart
  primitive operates on the in-repo V13 cache only.
* W68-F-HOSTED-NO-HIDDEN-STATE — hosted control plane cannot
  honestly access hidden state; falsifier triggers on any
  dishonest claim.
* W68-L-MULTI-AGENT-COORDINATOR-V4-SYNTHETIC — MASC V4 is a
  synthetic deterministic harness.
* W68-L-HOSTED-ESTIMATES-CALLER — cost / latency / quality scores
  are caller-declared; the router does not measure live hosted
  traffic.
* W68-L-FRONTIER-SUBSTRATE-STILL-BLOCKED — frontier-model
  substrate access remains a research-line wall.

### H. Trust boundary

* ≥ 40 disjoint failure modes enumerated in the W68 envelope
  verifier.
* ≥ 46 H-bars across R-152/R-153/R-154/R-155 (3 seeds → 138 cells
  pass).
* Cumulative trust boundary across W22..W68 ≥ 1410 enumerated
  failure modes.

### I. Stable-boundary preservation

* `coordpy.__version__ == "0.5.20"` (unchanged).
* SDK_VERSION unchanged.
* No PyPI release.
* Trivial passthrough preserved: W67 envelope CID identical
  byte-for-byte when `W68Params.build_trivial()` is used.

## Verdict labels

* **Strong success** — ≥ 12 mechanism advances + ≥ 3 benchmark
  families + ≥ 46 H-bars at 3 seeds pass + ≥ 7 regime wins fire +
  hosted control plane ships + hosted-vs-real wall benchmark passes.
* **Partial success** — ≥ 8 mechanism advances + ≥ 2 benchmark
  families + ≥ 24 H-bars pass + at least 5 regime wins fire +
  hosted control plane ships.
* **Failure** — fewer than 8 mechanism advances OR ≥ 2 benchmark
  families fail OR no new regime wins fire OR the architecture
  wall is not codified as a first-class artefact.

W68 is engineered so that strong success is the expected outcome
on the in-repo V13 substrate plus the new Plane A hosted control
plane. Failure to reach strong success would demonstrate that the
W67 substrate is the local optimum and the next move must be the
third-party transformer-internal bridge.
