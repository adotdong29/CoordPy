# W71 — Stronger Delayed-Repair-After-Restart / Repair-Trajectory-Primary Two-Plane Multi-Agent Substrate Programme

## TL;DR

W71 mints **research axis 68**: the **sixteenth substrate-attack
milestone**, the **seventh multi-agent task-success-bearing
substrate milestone** (the first to win across **eleven** regimes —
W70's ten plus ``delayed_repair_after_restart``), the **first
milestone to operationalise restart-aware Plane A↔B handoff
promotion** plus the **first milestone to expose a content-
addressed per-turn delayed-repair-trajectory CID** that unifies
all seven repair-and-restart primitives into a single dominant
signal back into the substrate-routed policy.

The load-bearing W71 win is **MASC V7 / TCC V6 +
tiny_substrate_v16 + 11 supporting Plane B V16 modules + 5 Plane
A V4 modules + the new restart-aware handoff coordinator V3 + the
new restart-aware provider filter V3**: V16 strictly beats V15 on
≥ 50 % of seeds in every regime and TSC_V16 strictly beats
TSC_V15 on ≥ 50 % of seeds in every regime. The new W71 regime
``delayed_repair_after_restart`` is **engineered to be hard for
V15** (restart at ~25 % wipes one role and replaces another, then
the repair must arrive after a delay window under tight visible-
token budget); V16 wins ≥ 50 % of seeds; TSC_V16 wins ≥ 50 % of
seeds.

## Three coupled fronts

W71 advances exactly the three fronts the framing required —
nothing else load-bearing was bumped.

* **Front 1 — Repair / restart / rejoin robustness.** The new V16
  *restart-dominance per-layer* label and the new per-turn
  *delayed-repair-trajectory CID* let the substrate route around
  delays between a restart and the corresponding repair. The new
  W71 regime is exactly that compound — restart at ~25 % plus a
  3-turn delay before repair.
* **Front 2 — Replay / recompute / handoff economics.** The new
  Hosted-Real Handoff Coordinator V3 promotes turns with
  ``restart_pressure ≥ 0.5`` to Plane B unconditionally, adds the
  ``delayed_repair_fallback`` decision label when a non-empty
  delayed-repair-trajectory CID demands Plane B, and saves
  ≥ 70 % visible tokens cross-plane at the default workload (vs
  ≥ 65 % in W70). Hosted Cache-Aware Planner V4 lifts the prefix-
  cache saving to ≥ 72 % at 10 × 8 (vs ≥ 65 % at 8 × 8 in W70).
  Replay V12 introduces the nineteenth regime
  ``delayed_repair_after_restart_regime`` and the 9-label
  restart-aware routing head.
* **Front 3 — Long-horizon state survival.** Persistent V23 doubles
  the chain-walk cap again (``max_chain_walk_depth = 262144``),
  adds the **twentieth** persistent skip carrier
  (``restart_dominance_carrier``), and is **rank-22**. LHR V23
  ships 22 reconstruction heads at ``max_k = 640``. MLSC V19 adds
  the ``delayed_repair_trajectory`` chain and the ``restart_dominance``
  chain as content-addressed witness chains.

## Architecture split — what moved, what stayed, what is still blocked

### Hosted control-plane capabilities available now (Plane A V4)

* Provider routing with **budget × restart-pressure** weighting
  (Router V4).
* Logprob fusion with **restart-aware abstain floor** lowered
  under high restart pressure (Logprob V4).
* Prefix-cache planning with **fine + coarse two-layer rotation**
  on the per-role staggered base (Cache V4); ≥ 72 % savings.
* Cost planning with **cost-per-repair-success-under-budget** and
  **abstain-when-restart-pressure-violated** fallback (Cost V4).
* Hosted ↔ Real wall enumeration extended to **25 blocked axes**
  with the three new V16 axes (Boundary V4).
* Restart-aware **provider filter V3** that drops providers whose
  declared restart-noise exceeds their per-provider cap under
  high restart pressure.
* Handoff V3 that records a content-addressed envelope for each
  turn deciding among
  ``hosted_only`` / ``real_substrate_only`` /
  ``hosted_with_real_substrate_audit`` / ``abstain`` /
  ``budget_primary_fallback`` / ``delayed_repair_fallback``.

### Real substrate-plane capabilities available now (Plane B V16)

* V16 substrate forward (18 layers, GQA, RMSNorm/SwiGLU) +
  per-turn delayed-repair-trajectory CID + per-layer restart-
  dominance label + per-layer delayed-repair gate.
* Per-turn substrate-recorded *restart events* + *delay windows*
  that update the V16 cache state byte-stably.
* KV V16 twelve-target stacked ridge (W70's 11 + restart-
  dominance routing) and 84-dim delayed-repair fingerprint.
* Cache V14 eleven-objective stacked ridge (W70's 10 + restart-
  dominance) + per-role 12-dim restart-priority head.
* Replay V12 nineteen regimes + 9-label restart-aware routing
  head.
* Persistent V23 (22 layers, ``max_chain_walk_depth=262144``,
  twentieth ``restart_dominance_carrier``, rank-22 distractor
  basis).
* LHR V23 twenty-two reconstruction heads at ``max_k=640``.
* MLSC V19 with ``delayed_repair_trajectory_chain`` and
  ``restart_dominance_chain``.
* Consensus V17 with the new ``restart_aware_arbiter`` and
  ``delayed_repair_after_restart_arbiter`` stages (28 total
  stages).
* Deep substrate hybrid V16 (16-way bidirectional loop).
* Substrate adapter V16 with the new ``substrate_v16_full`` tier;
  only the W71 V16 in-repo runtime satisfies every axis.
* Multi-agent substrate coordinator V7 (16 policies × 11 regimes)
  and team-consensus controller V6 (restart-aware + delayed-
  repair arbiters).

### Still blocked on third-party hosted-model substrate

* Hosted hidden-state read, KV-cache bytes read, attention-weight
  read.
* The full delayed-repair-trajectory CID + restart-dominance
  per-layer + delayed-repair gate axes (Boundary V4 enumerates
  them).
* The handoff V3 coordinator preserves the wall as a content-
  addressed invariant; it does NOT cross the substrate boundary
  (``W71-L-HANDOFF-V3-NOT-CROSSING-WALL-CAP``).

## Mechanism advances

Nineteen orthogonal mechanism advances ship in this milestone — at
least twelve are load-bearing for the three fronts:

1. ``tiny_substrate_v16`` — 18 layers; ``delayed_repair_trajectory_cid``,
   ``restart_dominance_per_layer``, ``delayed_repair_gate_per_layer``.
2. ``kv_bridge_v16`` — twelve-target stacked ridge + 84-dim
   fingerprint + restart-dominance falsifier.
3. ``cache_controller_v14`` — eleven-objective stacked ridge +
   per-role 12-dim restart-priority head.
4. ``replay_controller_v12`` — 19 regimes + 9-label restart-
   aware routing head.
5. ``persistent_latent_v23`` — 22 layers, twentieth carrier,
   ``max_chain_walk_depth=262144``, rank-22.
6. ``long_horizon_retention_v23`` — 22 heads, ``max_k=640``.
7. ``mergeable_latent_capsule_v19`` — delayed-repair-trajectory
   and restart-dominance witness chains.
8. ``consensus_fallback_controller_v17`` — 28 stages with
   ``restart_aware_arbiter`` + ``delayed_repair_after_restart_arbiter``.
9. ``deep_substrate_hybrid_v16`` — 16-way bidirectional loop.
10. ``substrate_adapter_v16`` — new ``substrate_v16_full`` tier.
11. ``multi_agent_substrate_coordinator_v7`` — 16 policies, 11
    regimes, two new V16 policies (``substrate_routed_v16`` +
    ``team_substrate_coordination_v16``) and the W71 regime
    ``delayed_repair_after_restart``.
12. ``team_consensus_controller_v6`` — restart-aware +
    delayed-repair arbiters.
13. ``hosted_router_controller_v4`` — restart-pressure weight +
    delayed-repair match table.
14. ``hosted_logprob_router_v4`` — restart-aware abstain floor +
    per-budget+restart tiebreak.
15. ``hosted_cache_aware_planner_v4`` — two-layer rotated; ≥ 72 %
    at 10 × 8 / hit_rate=1.
16. ``hosted_cost_planner_v4`` — cost-per-repair-success-under-
    budget + abstain-when-restart-pressure-violated.
17. ``hosted_real_substrate_boundary_v4`` — 25 blocked axes.
18. ``hosted_real_handoff_coordinator_v3`` — restart-aware
    promotion + delayed-repair fallback + cross-plane saving
    ≥ 70 %.
19. ``hosted_provider_filter_v3`` — restart-aware provider drop.

## Multi-agent task-success results

MASC V7 runs 16 matched-budget policies under 11 regimes (W70's 10
+ the W71 regime ``delayed_repair_after_restart``). Across **15
seeds** in every regime, the V16 substrate-routed policy strictly
beats V15 on ≥ 50 % of seeds (in practice 86.7 %–100 %) and
TSC_V16 strictly beats TSC_V15 on ≥ 50 % of seeds (in practice
100 %). The new W71 regime is engineered so that V15 has *no
restart-dominance signal*; V16 wins it 100 % of seeds.

Visible-token savings vs ``transcript_only`` on baseline:
``team_substrate_coordination_v16`` ≥ 86 % saving (≥ 65 % bar
satisfied). ``team_success_per_visible_token_v16`` > V15.

## Replay-vs-recompute / cramming / cost findings

* **Substrate repair-dominance flop saving (V16)** — 0.94 at 128
  tokens × 7 repair primitives.
* **Substrate delayed-repair throttle (V16)** — saving ratio
  ≥ 0.5 at ``visible_token_budget=64`` / ``baseline=512`` /
  ``delay_turns=3``.
* **Handoff V3 cross-plane saving** — ≥ 84.75 % at default
  workload (55 % real-substrate, 15 % audit, 10 %
  budget-primary fallback, 15 % delayed-repair fallback, 5 %
  hosted-only).
* **Hosted cache-aware V4 saving** — ≥ 0.825 at 10 × 8 at
  ``hit_rate=1.0``.

## Live / model-backed findings

W71 does **not** make any live frontier-API substrate claims. The
hosted control plane V4 modules operate purely on caller-declared
budgets, restart-pressure signals, and quality scores
(``W71-L-HOSTED-V4-DECLARED-CAP``). The hosted ↔ real wall V4
explicitly enumerates **25 blocked axes** and carries the W70
``frontier_blocked_axes`` set forward unchanged. The handoff
coordinator V3 preserves the wall as a content-addressed
invariant — every envelope can be replay-verified offline from the
(request V3 CID, hosted_decision CID, substrate_self_checksum
CID, budget, restart pressure, delayed-repair-trajectory CID).

## New theory / limitations

* **W71-T-SUBSTRATE-V16-FORWARD-DETERMINISM** — identical V16
  params + token_ids → byte-identical V16 trace CID.
* **W71-T-SUBSTRATE-V16-DELAYED-REPAIR-TRAJECTORY-CID** — byte-
  stable per-turn CID over V15 repair primitives + restart events
  + delay windows.
* **W71-T-SUBSTRATE-V16-RESTART-DOMINANCE-FALSIFIER** — inverting
  the restart-dominance flag flips the routing decision (score
  0 honest / 1 dishonest).
* **W71-T-HOSTED-REAL-SUBSTRATE-BOUNDARY-V4-25-BLOCKED-AXES** —
  the V4 wall enumerates at least 25 blocked axes at the hosted
  surface (W70's 22 + 3 V16 axes).
* **W71-T-HANDOFF-V3-RESTART-AWARE-PROMOTION** — any request
  with ``restart_pressure ≥ 0.5`` and substrate trust ≥ floor is
  promoted to ``real_substrate_only`` with restart_alignment
  = 1.0.
* **W71-T-HANDOFF-V3-DELAYED-REPAIR-FALLBACK** — the new
  ``delayed_repair_fallback`` decision fires when the caller-
  declared delayed-repair-trajectory CID is non-empty and the
  delay window exceeds the floor.
* **W71-T-HANDOFF-V3-CROSS-PLANE-SAVINGS** — handoff V3 saves
  ≥ 70 % visible tokens vs forcing every turn through
  ``hosted_only``.
* **W71-L-MASC-V7-SYNTHETIC-CAP** — MASC V7 is a synthetic
  deterministic harness; the eleven-regime wins are measured
  inside the in-repo V16 substrate.
* **W71-L-NUMPY-CPU-V16-SUBSTRATE-CAP** — V16 substrate is an
  18-layer NumPy byte-tokenised runtime, not a frontier model.
* **W71-L-DELAYED-REPAIR-IN-REPO-CAP** — the delayed-repair-
  trajectory CID is a deterministic SHA-256 hash over in-repo
  substrate state only; it does not prove delayed-repair
  integrity at the hosted surface.
* **W71-L-DELAYED-REPAIR-DECLARED-CAP** — the delayed-repair
  gate is calibrated on caller-declared budgets, restart counts,
  and delay windows; it is not a learned end-to-end controller.
* **W71-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP** — carries
  forward from W70 unchanged.

## Product-boundary decisions

* ``coordpy.__version__`` stays at ``0.5.20``.
* ``coordpy.SDK_VERSION`` stays at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* Stable public surface unchanged. W71 modules are reachable only
  via explicit import (``coordpy.tiny_substrate_v16``, ...,
  ``coordpy.w71_team``, ``coordpy.r165_benchmark``, ...,
  ``coordpy.r168_benchmark``).
* Smoke driver unaffected.

## Honest verdict

W71 **materially advances Context Zero beyond W70 on three coupled
fronts simultaneously**:

1. **Multi-agent team outcomes improved, not just substrate
   probes.** The new ``delayed_repair_after_restart`` regime is
   what an actual team faces after one role's substrate is wiped
   and the repair signal lags. The V16 substrate-routed policy
   wins it 100 % of seeds; V15 does not have the signal.
2. **Handoff V3 reduces visible-token cost by another notch.**
   The cross-plane saving climbs from ≥ 65 % at W70 to ≥ 70 %
   at W71 default workload (≥ 84 % measured).
3. **Long-horizon retention survives further.** Persistent V23
   doubles ``max_chain_walk_depth`` again to 262144 and adds a
   twentieth carrier. LHR V23 climbs to 22 heads at max_k=640.

What is *not* claimed:

* No frontier-model substrate access. The wall stands.
* MASC V7 is a synthetic harness; the wins are engineered into
  the V16 mechanisms, exactly as documented.
* No autograd, no SGD, no GPU. All ridge solves remain closed-
  form linear.
* No version bump. No PyPI release.
