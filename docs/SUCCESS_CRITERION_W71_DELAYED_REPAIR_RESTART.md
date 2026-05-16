# W71 — Pre-commit success criterion

**Stronger Delayed-Repair-After-Restart / Repair-Trajectory-Primary
Two-Plane Multi-Agent Substrate Programme.**

> Drafted before benchmarks ran. Authoritative bar for what counts
> as W71 success, partial success, and failure. *Not* a press
> release — every bar is mechanically checkable.

## Three coupled fronts

W71 advances exactly three fronts, all aimed at *team outcomes*, not
substrate probes alone:

1. **Front 1 — Repair / restart / rejoin robustness.** Delayed
   repair after restart under tight budget; restart-aware repair
   trajectory; silent-failure recovery that survives a member
   replacement *and* a contradiction in the same run.
2. **Front 2 — Replay / recompute / handoff economics.** Repair-
   trajectory-primary hosted-vs-real handoff; replay V12 nineteenth
   regime + restart-aware routing head; cache V14 eleven-objective
   ridge + per-role restart-priority head.
3. **Front 3 — Long-horizon state survival.** Retention V23 twenty-
   two heads + restart-aware reconstruction; persistent V23 chain-
   walk cap doubled again to ``262144`` + twentieth carrier
   (``restart_dominance_carrier``) + rank-22 distractor basis.

## Pre-committed bars

Each bar is a *single, falsifiable check* and is mirrored by an
H-cell in the W71 benchmark families R-165..R-168.

### Multi-agent task success (Front 1, F2, F3 coupled)

* **W71-T-MASC-V7-V16-STRICTLY-BEATS-V15-BASELINE** ≥ 50 % of
  seeds in baseline (15-seed batch).
* **W71-T-MASC-V7-V16-STRICTLY-BEATS-V15-ALL-W70-REGIMES** ≥ 50 %
  of seeds in every regime W70 already passes (baseline +
  team_consensus_under_budget + team_failure_recovery +
  role_dropout + branch_merge_reconciliation + partial_contradiction
  + agent_replacement_warm_restart + multi_branch_rejoin +
  silent_corruption_plus_member_replacement +
  contradiction_then_rejoin_under_budget).
* **W71-T-MASC-V7-NEW-REGIME-DELAYED-REPAIR-AFTER-RESTART** ≥
  50 % seeds for both ``substrate_routed_v16`` and
  ``team_substrate_coordination_v16``.
* **W71-T-MASC-V7-TSC-V16-STRICTLY-BEATS-TSC-V15-EVERY-REGIME**
  ≥ 50 % of seeds.

### Team success per visible token (explicit bar)

* **W71-T-MASC-V7-VISIBLE-TOKEN-SAVINGS** ≥ 65 % savings of
  ``team_substrate_coordination_v16`` vs ``transcript_only`` on
  baseline (W70: ≥ 60 %).
* **W71-T-MASC-V7-TEAM-SUCCESS-PER-VISIBLE-TOKEN-FLOOR** team-
  success-per-visible-token of V16 strictly greater than V15 on
  baseline.

### Team success per recompute flop (explicit bar)

* **W71-T-SUBSTRATE-V16-REPAIR-DOMINANCE-FLOPS-SAVING** ≥ 0.83
  saving ratio vs full recompute across the **seven** W71 repair
  primitives (W70's six + ``restart_dominance``) at 128 tokens.
* **W71-T-SUBSTRATE-V16-BUDGET-PRIMARY-THROTTLE** ≥ 0.5 saving
  ratio at ``visible_token_budget=64`` / ``baseline=512``.
* **W71-T-HANDOFF-V3-RECOMPUTE-FLOP-SAVING** restart-aware handoff
  V3 saving ratio ≥ 50 % vs forcing every turn through ``hosted_only``.

### Hosted control plane V4 (Plane A)

* **W71-T-HOSTED-ROUTER-V4-DETERMINISTIC** on
  ``(registry_cid, request_v4_cid, visible_token_budget,
  repair_dominance_label, restart_pressure)``.
* **W71-T-HOSTED-ROUTER-V4-RESTART-PRESSURE-SCORE** > 0 for any
  non-zero restart-pressure input.
* **W71-T-HOSTED-LOGPROB-V4-RESTART-AWARE-ABSTAIN** abstains *more
  aggressively* (lower entropy floor) under high restart pressure
  than under no restart pressure.
* **W71-T-HOSTED-CACHE-V4-72-PCT-SAVINGS** ≥ 72 % savings on
  10×8 at ``hit_rate=1.0`` (V3 was ≥ 65 % at 8×8).
* **W71-T-HOSTED-COST-V4-COST-PER-REPAIR-SUCCESS-UNDER-BUDGET**
  finite when within budget; ``+inf`` when budget violated.
* **W71-T-HOSTED-PROVIDER-FILTER-V3-RESTART-AWARE** restart-aware
  filter rejects ``noisy`` providers under high restart pressure.
* **W71-T-HOSTED-REAL-SUBSTRATE-BOUNDARY-V4-25-BLOCKED-AXES** wall
  V4 enumerates ≥ 25 blocked axes (W70's 22 + 3 new V16 axes:
  ``restart_dominance_carrier``, ``restart_aware_handoff_label``,
  ``delayed_repair_trajectory_cid``).
* **W71-T-HOSTED-REAL-SUBSTRATE-BOUNDARY-V4-FRONTIER-BLOCKED**
  carries the W70 frontier_blocked_axes set forward unchanged.

### Real substrate plane V16 (Plane B)

* **W71-T-SUBSTRATE-V16-FORWARD-DETERMINISM** — identical params +
  token_ids → byte-identical V16 trace CID.
* **W71-T-SUBSTRATE-V16-DELAYED-REPAIR-TRAJECTORY-CID** byte-stable
  on ``(params, token_ids, repair_events, restart_events)``.
* **W71-T-SUBSTRATE-V16-RESTART-DOMINANCE-PER-LAYER** label in
  ``[0..7]`` of shape ``(L,)`` with ``L=18`` (W70 had L=17).
* **W71-T-SUBSTRATE-V16-DELAYED-REPAIR-GATE** per-layer gate of
  shape ``(L,)`` calibrated by ``(visible_token_budget,
  baseline_cost, restart_count, repair_dominance_count, delay_turns)``.
* **W71-T-KV-V16-TWELVE-TARGET-RIDGE** twelve-target stacked ridge
  fit (W70's 11 + ``restart_dominance_routing``).
* **W71-T-KV-V16-DELAYED-REPAIR-FINGERPRINT** 84-dim SHA-256
  fingerprint of ``(role, repair_trajectory_cid,
  delayed_repair_trajectory_cid, dominant_repair_label,
  restart_count, visible_token_budget, baseline_cost, task_id,
  team_id, branch_id, delay_turns)``.
* **W71-T-KV-V16-RESTART-DOMINANCE-FALSIFIER** ``= 0`` iff
  inverting the restart-dominance flag flips the routing decision.
* **W71-T-CACHE-V14-ELEVEN-OBJECTIVE-RIDGE** converges (W70's 10 +
  ``restart_dominance``).
* **W71-T-CACHE-V14-PER-ROLE-RESTART-PRIORITY** per-role 12-dim
  ridge head (W70's 11 + ``restart_pressure``).
* **W71-T-REPLAY-V12-NINETEEN-REGIMES** V12 introduces
  ``delayed_repair_after_restart_regime`` on top of V11's eighteen.
* **W71-T-REPLAY-V12-RESTART-AWARE-ROUTING** 9×(F+1) ridge head
  over team features predicts routing label (W70's 8 + a new
  ``restart_route`` label).
* **W71-T-PERSISTENT-V23-CHAIN-WALK-262144** doubled vs W70.
* **W71-T-PERSISTENT-V23-TWENTIETH-SKIP** ``restart_dominance_carrier``
  populates and persists across chain steps; rank-22 distractor basis.
* **W71-T-LHR-V23-TWENTY-TWO-HEADS** ``max_k=640``.
* **W71-T-MLSC-V19-DELAYED-REPAIR-CHAIN** repair-trajectory + delayed-
  repair-trajectory chains both content-addressed across merges.
* **W71-T-CONSENSUS-V17-TWENTY-EIGHT-STAGES** V17 stage chain has
  ≥ 28 stages (W70's 26 + ``restart_aware_arbiter`` +
  ``delayed_repair_after_restart_arbiter``).
* **W71-T-DEEP-HYBRID-V16-SIXTEEN-WAY** sixteen-way loop fires when
  all sixteen axes fire on the same step.
* **W71-T-SUBSTRATE-ADAPTER-V16-V16-FULL-TIER** only the W71 V16
  in-repo runtime satisfies every axis; hosted backends remain
  text-only at the HTTP surface.

### Handoff V3 (Plane A ↔ Plane B)

* **W71-T-HANDOFF-V3-ENVELOPE-CONTENT-ADDRESSED** envelope CID
  is deterministic on ``(request V3 CID, hosted_decision_cid,
  substrate_self_checksum_cid, budget, restart_pressure,
  delayed_repair_trajectory_cid)``.
* **W71-T-HANDOFF-V3-RESTART-AWARE-PROMOTION** any request with
  ``restart_pressure ≥ 0.5`` is promoted to ``real_substrate_only``
  with ``restart_alignment = 1.0``.
* **W71-T-HANDOFF-V3-DELAYED-REPAIR-FALLBACK** new decision label
  ``delayed_repair_fallback`` fires when hosted is cheaper but the
  delayed-repair-trajectory CID requires Plane B.
* **W71-T-HANDOFF-V3-REPAIR-DOMINANCE-FALSIFIER** ``= 0`` on
  honest, ``= 1`` on dishonest "hosted satisfies delayed-repair"
  claim against a hosted_only envelope.
* **W71-T-HANDOFF-V3-CROSS-PLANE-SAVINGS** ≥ 70 % at default
  workload (W70 was ≥ 65 %).

### Falsifiers / limitation reproductions

* **W71-T-HANDOFF-V3-REPAIR-DOMINANCE-FALSIFIER** (above).
* **W71-T-BOUNDARY-V4-FALSIFIER** honest = 0, dishonest = 1.
* **W71-T-KV-V16-RESTART-DOMINANCE-FALSIFIER** honest = 0 on
  zero label, dishonest = 1 when caller asserts "hosted satisfies
  restart-dominance" with hosted-only routing.
* **W71-L-MASC-V7-SYNTHETIC-CAP** MASC V7 is a synthetic
  deterministic harness.
* **W71-L-NUMPY-CPU-V16-SUBSTRATE-CAP** V16 substrate is an
  18-layer NumPy byte-tokenised runtime, not a frontier model.
* **W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP** carries forward
  unchanged.
* **W71-L-HOSTED-V4-NO-SUBSTRATE-CAP** carries forward unchanged.

## Mechanism advance count

**Twelve** load-bearing advances minimum:

1. ``tiny_substrate_v16`` (18 layers; W71 axes:
   ``delayed_repair_trajectory_cid``,
   ``restart_dominance_per_layer``, ``delayed_repair_gate``).
2. ``kv_bridge_v16`` (12-target ridge + 84-dim fingerprint +
   restart-dominance falsifier).
3. ``cache_controller_v14`` (11-objective ridge + per-role 12-dim
   restart-priority head).
4. ``replay_controller_v12`` (19 regimes + 9-label restart-aware
   routing head).
5. ``persistent_latent_v23`` (22 layers, 20th carrier,
   ``max_chain_walk_depth=262144``, rank-22).
6. ``long_horizon_retention_v23`` (22 heads, max_k=640).
7. ``mergeable_latent_capsule_v19`` (delayed-repair chain).
8. ``consensus_fallback_controller_v17`` (28 stages — 2 new
   arbiters).
9. ``multi_agent_substrate_coordinator_v7`` (16-policy, 11-regime).
10. ``team_consensus_controller_v6`` (delayed-repair-after-restart
    arbiter + restart-dominance arbiter on top of W70).
11. ``deep_substrate_hybrid_v16`` (16-way bidirectional loop).
12. ``substrate_adapter_v16`` (substrate_v16_full tier; new axes).
13. ``hosted_router_controller_v4`` (restart-pressure weight +
    delayed-repair match).
14. ``hosted_logprob_router_v4`` (restart-aware abstain floor).
15. ``hosted_cache_aware_planner_v4`` (per-role staggered + rotated
    + two-layer rotated; ≥ 72 % savings on 10×8).
16. ``hosted_cost_planner_v4`` (cost-per-repair-success-under-
    budget).
17. ``hosted_real_substrate_boundary_v4`` (25 blocked axes).
18. ``hosted_real_handoff_coordinator_v3`` (restart-aware promotion
    + delayed-repair fallback decision).
19. ``hosted_provider_filter_v3`` (restart-aware filter).

(More than the required 12 — the bundle is internally coherent.)

## Benchmark families (≥ 3 required, four delivered)

* **R-165** Hosted control plane V4 (10 H-bars).
* **R-166** Real substrate plane V16 (16 H-bars).
* **R-167** Multi-agent task success across 11 regimes (24 H-bars).
* **R-168** Handoff V3 + falsifier + limitation reproductions
  (14 H-bars).

Total **64 H-bars × 3 seeds = 192 cells**.

## Strong vs partial vs failure

* **Strong success.** All 19 mechanism advances ship, R-165–R-168
  pass at 3 seeds, the eleventh regime wins on V16 substrate and
  TSC V16, ≥ 70 % handoff V3 saving, ≥ 65 % visible-token saving,
  ≥ 25 blocked hosted axes, all falsifiers pass.
* **Partial success.** Most mechanisms ship, the new regime
  partially wins (V16 OR TSC V16 ≥ 50 %; not both), handoff V3
  saving in [50 %, 70 %), 60–65 % visible-token saving.
* **Failure.** New regime loses for both V16 and TSC V16, OR
  ≥ 1 existing W70 regime regresses, OR ≥ 1 falsifier inverts,
  OR ≥ 1 blocked axis becomes "satisfied at hosted" without
  evidence.

## Stable-boundary preservation, no version bump, no release

W71 ships at ``coordpy.tiny_substrate_v16``, ``coordpy.kv_bridge_v16``,
``coordpy.cache_controller_v14``, ``coordpy.replay_controller_v12``,
``coordpy.persistent_latent_v23``,
``coordpy.long_horizon_retention_v23``,
``coordpy.mergeable_latent_capsule_v19``,
``coordpy.consensus_fallback_controller_v17``,
``coordpy.deep_substrate_hybrid_v16``,
``coordpy.substrate_adapter_v16``,
``coordpy.multi_agent_substrate_coordinator_v7``,
``coordpy.team_consensus_controller_v6``,
``coordpy.hosted_router_controller_v4``,
``coordpy.hosted_logprob_router_v4``,
``coordpy.hosted_cache_aware_planner_v4``,
``coordpy.hosted_cost_planner_v4``,
``coordpy.hosted_real_substrate_boundary_v4``,
``coordpy.hosted_real_handoff_coordinator_v3``,
``coordpy.hosted_provider_filter_v3``, ``coordpy.w71_team``, and
``coordpy.r165_benchmark`` / ``coordpy.r166_benchmark`` /
``coordpy.r167_benchmark`` / ``coordpy.r168_benchmark``. Reachable
only via explicit import. **No** ``coordpy.__version__`` bump
(stays at ``0.5.20``); **no** ``SDK_VERSION`` bump (stays at
``coordpy.sdk.v3.43``); **no** PyPI publish.
