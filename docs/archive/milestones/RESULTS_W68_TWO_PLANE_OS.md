# W68 — Two-Plane Substrate-Coupled Latent OS (post-W67 result note)

> Post-W67 research milestone, 2026-05-16. **Thirteenth substrate-
> attack milestone** in the programme. **Fourth multi-agent task-
> success-bearing substrate milestone** — and the first to make
> the **hosted ↔ real-substrate wall** a first-class architectural
> artefact instead of a sentence in limitations.

## TL;DR — strong success

W68 met the pre-committed
[`SUCCESS_CRITERION_W68_TWO_PLANE_OS.md`](SUCCESS_CRITERION_W68_TWO_PLANE_OS.md)
bar on every dimension:

* **138 / 138 R-152/R-153/R-154/R-155 cells pass at 3 seeds**.
  * R-152 (hosted control plane, Plane A): **10 H-bars × 3 seeds =
    30 cells**, 30/30 pass.
  * R-153 (real substrate, Plane B): **16 H-bars × 3 seeds = 48
    cells**, 48/48 pass.
  * R-154 (multi-agent task success across 7 regimes): **14
    H-bars × 3 seeds = 42 cells**, 42/42 pass.
  * R-155 (hosted-vs-real wall): **6 H-bars × 3 seeds = 18 cells**,
    18/18 pass.
* **All 28 mechanism advances ship and fire** (22 Plane B + 6
  Plane A).
* **MASC V4 wins across all seven regimes** (V13 strictly beats
  V12 on ≥ 50 % of seeds in every regime; TSC_V13 strictly beats
  TSC_V12 on ≥ 50 % of seeds in every regime):
  * **baseline**: V13 vs V12 = **80.0 %**; TSC_V13 vs TSC_V12 =
    **86.7 %**.
  * **team_consensus_under_budget**: V13 vs V12 = **80.0 %**;
    TSC_V13 vs TSC_V12 = **86.7 %**.
  * **team_failure_recovery**: V13 vs V12 = **93.3 %**; TSC_V13 vs
    TSC_V12 = **93.3 %**.
  * **role_dropout**: V13 vs V12 = **60.0 %**; TSC_V13 vs TSC_V12 =
    **80.0 %**.
  * **branch_merge_reconciliation**: V13 vs V12 = **80.0 %**;
    TSC_V13 vs TSC_V12 = **53.3 %**.
  * **partial_contradiction_under_delayed_reconciliation** (new):
    V13 vs V12 = **60.0 %**; TSC_V13 vs TSC_V12 = **80.0 %**.
  * **agent_replacement_warm_restart** (new): V13 vs V12 =
    **93.3 %**; TSC_V13 vs TSC_V12 = **93.3 %**.
* **Substrate prefix-reuse** primitive saves **94 %** flops vs
  recompute over a 4-reuse plan at 128 tokens (target ≥ 80 %).
* **Hosted cache-aware planning** saves **≥ 50 %** input tokens
  on multi-turn plans at hosted-cache-hit-rate = 1.0; cross-plane
  bridge to the V13 prefix-reuse counter ships.
* **Hosted ↔ real-substrate wall** is content-addressed,
  enumerates **15 blocked axes** at the hosted surface and **64
  real-substrate-only V13 axes**; the falsifier returns 0 on
  honest claims and 1 on dishonest hidden-state claims at the
  hosted surface.
* **Six new closed-form ridge solves** on top of W67's 41 (cache
  V11 eight-objective + cache V11 per-role agent-replacement +
  replay V9 per-role per-regime + replay V9 agent-replacement-
  routing + HSB V12 nine-target + KV V13 nine-target). **Total
  forty-seven closed-form ridge solves across W61..W68.** No
  autograd, no SGD, no GPU.

## What W68 is

W68 introduces the **Two-Plane Substrate-Coupled Latent Operating
System** — the thirteenth substrate-attack milestone. It is the
first milestone to make the **hosted ↔ real-substrate architecture
wall** a first-class artefact instead of a footnote. The two
planes:

### Plane A — Hosted control plane (Plane A, 6 advances)

Operates over OpenRouter / Groq / OpenAI-compatible HTTP surfaces
at the **text / optional-logprobs / optional-prefix-cache** layer.
Pure-Python and honest about its blast radius:

* `coordpy.hosted_router_controller` — content-addressed registry
  of providers + their honest capability tier (text-only,
  logprobs, prefix-cache, logprobs+prefix-cache); cheapest-then-
  fastest routing with data-policy and tier filters.
* `coordpy.hosted_logprob_router` — top-k logprob fusion over the
  shared top-k vocabulary across providers; text-only quorum
  fallback when no provider exposes logprobs.
* `coordpy.hosted_cache_aware_planner` — per-turn prefix-CID
  planner that maximises hosted prefix-cache hit rates;
  cross-plane bridge to the V13 substrate's prefix-reuse counter.
* `coordpy.hosted_provider_filter` — data-policy-aware filter
  (`no_log`, `no_train`) + tier filter; produces a derived
  content-addressed registry.
* `coordpy.hosted_cost_planner` — cost/latency-aware provider
  selection under a matched-quality constraint.
* `coordpy.hosted_real_substrate_boundary` — the explicit
  architecture wall; `HostedRealSubstrateBoundary` content-
  addressed boundary object, `HostedRealSubstrateWallReport`
  enumeration of hosted-solvable / real-substrate-only / blocked-
  frontier axes, and a falsifier that triggers on any dishonest
  hosted hidden-state claim.

### Plane B — Real substrate plane (22 advances)

Runs the in-repo V13 substrate plus its 21 bridges / controllers /
benchmarks. Strictly extends W67's twelve-axis stack with:

* `coordpy.tiny_substrate_v13` — 15-layer V13 substrate. Same GQA
  (8 query / 4 KV). Four new V13 axes (partial-contradiction
  witness tensor, agent-replacement flag w/ warm-restart window,
  substrate prefix-reuse counter, V13 composite gate score).
* Bridges: `kv_bridge_v13`, `hidden_state_bridge_v12`,
  `prefix_state_bridge_v12`, `attention_steering_bridge_v12`.
* Controllers: `cache_controller_v11`, `replay_controller_v9`,
  `consensus_fallback_controller_v14`,
  `team_consensus_controller_v3`,
  `multi_agent_substrate_coordinator_v4`.
* Carriers / retention / codes:
  `persistent_latent_v20`, `multi_hop_translator_v18`,
  `mergeable_latent_capsule_v16`, `corruption_robust_carrier_v16`,
  `long_horizon_retention_v20`, `ecc_codebook_v20`.
* Composites:
  `transcript_vs_shared_arbiter_v17`, `uncertainty_layer_v16`,
  `disagreement_algebra_v14`, `deep_substrate_hybrid_v13`,
  `substrate_adapter_v13`.

The load-bearing W68 win is the V13 substrate plus the V13
multi-agent stack (`MultiAgentSubstrateCoordinatorV4` + V13
substrate + Team-Consensus Controller V3 + 22 supporting modules).
V13 strictly beats V12 across all seven regimes; the **80 %
strict-beat in partial-contradiction** and the **93.3 % strict-
beat in agent-replacement-warm-restart** are the new evidence.

### Still-blocked frontier (not solved by W68)

Frontier-model **substrate access** at deployment remains the
research-line wall. W68 codifies it (`HostedRealSubstrateBoundary`)
and provides a falsifier so any future claim that hosted APIs
expose hidden state is mechanically falsifiable.

## What W68 is NOT

W68 is not a third-party substrate-coupling milestone. Hosted
backends remain text-only at the HTTP surface (with logprobs and
prefix-cache as optional declared capabilities); the V13 substrate
is the same in-repo NumPy runtime as the V8..V12 substrates, only
15 layers instead of 14. The wins are measured **inside the
synthetic MASC V4 harness** — they are not real hosted multi-
agent task-success wins.

W68 is also not a SGD / autograd / GPU milestone. All "training"
remains single-step closed-form linear ridge — 47 solves total
across W61..W68. No autograd anywhere.

W68 is also not a release. `coordpy.__version__ == "0.5.20"` is
byte-for-byte preserved. No PyPI publish. The smoke driver passes
unchanged.

## Files added

Plane B — real substrate:

- `coordpy/tiny_substrate_v13.py`
- `coordpy/kv_bridge_v13.py`
- `coordpy/hidden_state_bridge_v12.py`
- `coordpy/prefix_state_bridge_v12.py`
- `coordpy/attention_steering_bridge_v12.py`
- `coordpy/cache_controller_v11.py`
- `coordpy/replay_controller_v9.py`
- `coordpy/persistent_latent_v20.py`
- `coordpy/multi_hop_translator_v18.py`
- `coordpy/mergeable_latent_capsule_v16.py`
- `coordpy/consensus_fallback_controller_v14.py`
- `coordpy/corruption_robust_carrier_v16.py`
- `coordpy/long_horizon_retention_v20.py`
- `coordpy/ecc_codebook_v20.py`
- `coordpy/transcript_vs_shared_arbiter_v17.py`
- `coordpy/uncertainty_layer_v16.py`
- `coordpy/disagreement_algebra_v14.py`
- `coordpy/deep_substrate_hybrid_v13.py`
- `coordpy/substrate_adapter_v13.py`
- `coordpy/multi_agent_substrate_coordinator_v4.py`
- `coordpy/team_consensus_controller_v3.py`
- `coordpy/w68_team.py`

Plane A — hosted control plane:

- `coordpy/hosted_router_controller.py`
- `coordpy/hosted_logprob_router.py`
- `coordpy/hosted_cache_aware_planner.py`
- `coordpy/hosted_provider_filter.py`
- `coordpy/hosted_cost_planner.py`
- `coordpy/hosted_real_substrate_boundary.py`

Benchmarks:

- `coordpy/r152_benchmark.py`
- `coordpy/r153_benchmark.py`
- `coordpy/r154_benchmark.py`
- `coordpy/r155_benchmark.py`

Tests:

- `tests/test_w68_modules.py`
- `tests/test_w68_team_envelope_chain.py`
- `tests/test_w68_trivial_passthrough_byte_identical.py`
- `tests/test_r152_r153_r154_r155_w68.py`

Docs:

- `docs/SUCCESS_CRITERION_W68_TWO_PLANE_OS.md`
- `docs/RESULTS_W68_TWO_PLANE_OS.md` (this file)

## Docs updated

- `docs/RESEARCH_STATUS.md` — W68 TL;DR added.
- `docs/THEOREM_REGISTRY.md` — W68-T-* / W68-L-* claims added.
- `docs/context_zero_master_plan.md` — W68 milestone marker added.
- `papers/context_as_objects.md` — W68 paper-line update added.
- `CHANGELOG.md` — W68 entry added.
- `docs/HOW_NOT_TO_OVERSTATE.md` — W68 do-not-overstate rules
  added.
- `docs/START_HERE.md` — W68 milestone summary added.

## Empirical results (selected)

### R-154 — Seven-regime multi-agent task success

V13 (`substrate_routed_v13`) and TSC_V13 (`team_substrate_
coordination_v13`) strict-beat rates over 15 seeds at matched
budget:

| Regime | V13 vs V12 | TSC_V13 vs TSC_V12 |
|---|---|---|
| baseline | 80.0 % | 86.7 % |
| team_consensus_under_budget | 80.0 % | 86.7 % |
| team_failure_recovery | 93.3 % | 93.3 % |
| role_dropout | 60.0 % | 80.0 % |
| branch_merge_reconciliation | 80.0 % | 53.3 % |
| **partial_contradiction** (new) | 60.0 % | 80.0 % |
| **agent_replacement_warm_restart** (new) | 93.3 % | 93.3 % |

V13 saves > 50 % visible tokens vs `transcript_only` at baseline.

### R-153 — Real-substrate pillars

* Hidden-vs-KV pillar wins (per-seed engineered): hidden residual
  strictly < KV residual on matched targets.
* Transcript-vs-shared-state pillar wins: transcript-only residual
  > shared-state proxy residual on matched-fidelity probe.
* Substrate prefix-reuse flops saving: **94 %** at 128 tokens × 4
  reuses (target ≥ 80 %).
* KV V13 nine-target stacked ridge converges.
* HSB V12 nine-target stacked ridge converges; per-(L, H) hidden-
  vs-agent-replacement win-rate mean ≥ 0.99 on engineered grid.
* Cache V11 eight-objective stacked ridge converges; per-role
  9-dim agent-replacement priority head converges.
* Replay V9: fourteen regimes; six agent-replacement routing
  labels; routing head trained and converged.
* Substrate adapter V13: `substrate_v13_full` tier is exclusive
  to the in-repo V13 substrate.

### R-152 — Hosted control plane

* HostedRouterController decision is deterministic on (registry
  CID, request CID).
* HostedLogprobRouter top-k fusion fires on shared top-k; falls
  back to text-only quorum when no provider exposes logprobs.
* HostedCacheAwarePlanner prefix CID is content-addressed; ≥ 71 %
  token savings on 4-turn plans at hit_rate = 1.0.
* HostedProviderFilter drops train-policy and tier-incompatible
  providers; output registry CID is a deterministic function of
  (input registry CID, filter spec CID).
* HostedCostPlanner picks free providers when quality threshold
  passes; abstains when no provider passes.
* HostedRealSubstrateBoundary falsifier returns 0 on honest
  blocked-axis claims and 1 on dishonest claims.

### R-155 — Hosted ↔ real-substrate wall

* Wall boundary is content-addressed: same CID across
  invocations.
* Wall enumerates **15 blocked axes** at the hosted surface
  (hidden_state_read / hidden_state_write / kv_bytes_read /
  kv_bytes_write / attention_weights_read /
  attention_weights_write / per_layer_head_tensor_read /
  branch_merge_witness / role_dropout_recovery_flag /
  substrate_snapshot_fork / v12_gate_score /
  partial_contradiction_witness / agent_replacement_flag /
  prefix_reuse_counter / v13_gate_score).
* Wall enumerates **64 real-substrate-only V13 axes** (the union
  of V8..V13 capability axes minus the 6 hosted-available axes).
* Hosted cache-aware savings ≥ 50 % at hit_rate = 1.0.
* Real-substrate branch-merge ≥ 60 % flops saving.
* Falsifier returns 0 on honest claims, 1 on dishonest claims.

## Theory / limitations (selected)

See `docs/THEOREM_REGISTRY.md` for the full set. Headline claims:

* **W68-T-TWO-PLANE-ARCHITECTURE** — *code-backed*. The
  hosted-vs-real-substrate boundary is content-addressed and
  exposes a structural falsifier; any future claim that a hosted
  API gives hidden-state access is mechanically falsifiable.
* **W68-T-PARTIAL-CONTRADICTION-WITNESS** — *empirical / in-repo*.
  The V13 substrate's partial-contradiction witness tensor is a
  load-bearing signal in the MASC V4 partial-contradiction regime
  (60 % strict-beat at substrate_routed_v13; 80 % at TSC_V13).
* **W68-T-AGENT-REPLACEMENT-WARM-RESTART** — *empirical / in-repo*.
  V13's agent-replacement flag drives 93.3 % strict-beat in the
  agent-replacement-warm-restart regime.
* **W68-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP** — *code-backed*.
  Hosted backends remain text-only at the HTTP surface; the
  hosted control plane explicitly does NOT pierce the boundary.
* **W68-L-FRONTIER-SUBSTRATE-STILL-BLOCKED** — *limitation*. The
  combination of frontier-model text quality and full substrate
  access is unsolved; W68 codifies the wall, not its dissolution.
* **W68-L-HOSTED-ESTIMATES-CALLER** — *limitation*. Hosted cost /
  latency / quality scores are caller-declared; the router does
  not measure live hosted traffic.
* **W68-L-MASC-V4-SYNTHETIC** — *limitation*. MASC V4 is a
  synthetic deterministic harness; the seven-regime wins are
  measured inside the in-repo substrate, not on real model
  outputs.
* **W68-L-V13-NO-AUTOGRAD** — *code-backed*. All V13 ridge solves
  are closed-form linear; total 47 solves across W61..W68.

## Stable-boundary preservation

* `coordpy.__version__ == "0.5.20"` (unchanged).
* `SDK_VERSION` unchanged.
* No PyPI release.
* Trivial passthrough preserved: W67 outer CID identical
  byte-for-byte when `W68Params.build_trivial()` is used.
* The W66 smoke driver and W67 envelope chain remain byte-
  identical.

## Verdict

**Strong success.** W68 meets every pre-committed bar from
[`SUCCESS_CRITERION_W68_TWO_PLANE_OS.md`](SUCCESS_CRITERION_W68_TWO_PLANE_OS.md):

* 28 mechanism advances (target ≥ 12).
* 4 benchmark families (target ≥ 3).
* 138 / 138 R-152/R-153/R-154/R-155 cells pass at 3 seeds (target
  ≥ 138).
* 7 / 7 regimes show V13 strictly beating V12 on ≥ 50 % of seeds
  (target ≥ 7).
* hosted control plane ships (6 modules) with cost / latency /
  cache / filter / routing.
* hosted-vs-real wall is a first-class benchmark family with a
  structural falsifier.
* 47 closed-form ridge solves total (6 new on top of W67's 41).
* 42 W68 envelope failure modes (cumulative trust boundary across
  W22..W68 ≥ 1417 enumerated failure modes).

This materially advances Context Zero beyond W67 because it
**explicitly splits the architecture** along the hosted ↔ real-
substrate seam and codifies the wall in code, not docs. Future
substrate work has to attack one of the three named slots:

1. **Plane A wins** — pure hosted; no substrate. Bounded by the
   hosted surface.
2. **Plane B wins** — real substrate; only on runtimes we
   control. Bounded by what we can implement honestly.
3. **Frontier substrate wins** — still blocked.

W68 is the last milestone that can succeed without the third-
party transformer-internal bridge.
