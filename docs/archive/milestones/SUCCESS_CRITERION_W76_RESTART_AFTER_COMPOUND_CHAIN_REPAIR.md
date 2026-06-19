# Success criterion — W76 Stronger Restart-After-Compound-Chain-Repair / Compound-Chain-Then-Restart Budget-Primary Two-Plane Multi-Agent Substrate Programme

Pre-committed before implementation, on `main`, no version bump,
no PyPI release. Direct extension of W75
(``coordpy/tiny_substrate_v20`` line; MASC V11 / TCC V10; Plane A
V8 + handoff V7 + provider filter V7).

## Headline win the milestone must produce

The team must materially survive a *restart of a member* that
fires **immediately after** the compound-chain-repair arc (the
W75 ``replacement → delayed_repair → rejoin`` chain), without
losing the compound-chain-repair trajectory the team just earned.
That is a strictly harder failure modality than W75 because the
compound chain has just succeeded and the team is at its most
fragile.

The W76 substrate must:

1. Expose a content-addressed **per-turn compound-chain-then-
   restart trajectory CID** that unifies all W75 V20 primitives
   (compound_repair_trajectory_cid plus
   compound_chain_repair_trajectory_cid) with the post-restart
   event chain, and surface it as a per-turn signal back into the
   substrate-routed policy.
2. Expose a **per-layer compound-chain-then-restart length label**
   in [0..12] (W75's [0..11] plus the new ``restart_after_
   compound_chain_repair`` label).
3. Expose a **per-layer compound-chain-then-restart-pressure
   gate** that throttles substrate work as a function of:
   (visible-token budget, restart count, rejoin count, replacement
   count, contradiction count, delayed-repair count, compound-
   failure count, compound-chain-window, and the new
   post-compound-chain-restart-window).

Plane B (in-repo NumPy) and Plane A (hosted HTTP-text-only) must
both observe these new axes consistently with the existing W75 V8
wall and the W70 frontier-blocked set.

## Exactly one new regime — drop-in extension of MASC V11

``restart_after_compound_chain_repair_under_budget``:

* Same chain regime as W75 (``compound_repair_after_replacement_
  then_rejoin_under_budget``) up to turn ~75 % — the replacement /
  delayed-repair / rejoin chain fires unchanged.
* At ~75 % of turns, role 0 (the team member that *just* finished
  absorbing the chain) is **restarted** under a tight visible-
  token budget. The team must now survive the restart without
  losing the compound-chain repair trajectory it just built.
* The W76 substrate's compound-chain-then-restart-pressure gate +
  the new per-layer chain-then-restart label trigger a coordinated
  fresh-restart arbitrator on top of the W75 compound-chain
  arbitrator.

This is a single regime, strictly load-bearing, directly on the
three focus fronts (compound repair robustness; replay / recompute
/ handoff economics; long-horizon state survival under budget).

## Modules — coherent V21 / V9 bundle

**Plane B — Real substrate plane V21.** Twelve modules:

1. ``tiny_substrate_v21`` — 23 layers (V20's 22 + 1). Three new
   V21 axes: per-turn compound-chain-then-restart trajectory CID;
   per-layer compound-chain-then-restart length label in [0..12];
   per-layer compound-chain-then-restart-pressure gate.
2. ``kv_bridge_v21`` — 17-target stacked ridge (V20's 16 + the new
   compound-chain-then-restart target); 140-dim compound-chain-
   then-restart fingerprint; compound-chain-then-restart-pressure
   falsifier.
3. ``cache_controller_v19`` — 16-objective stacked ridge (V18's
   15 + a 16th compound-chain-then-restart target); per-role
   17-dim compound-chain-then-restart-pressure head.
4. ``replay_controller_v17`` — 24 regimes (V16's 23 + the new
   chain-then-restart regime); 14-label compound-chain-then-
   restart-aware routing head.
5. ``deep_substrate_hybrid_v21`` — 21-way bidirectional loop
   adding the new ``compound_chain_then_restart_axis``.
6. ``substrate_adapter_v21`` — ``substrate_v21_full`` tier.
7. ``persistent_latent_v28`` — 27 layers, 25th persistent skip
   carrier (compound-chain-then-restart-pressure EMA carrier),
   ``max_chain_walk_depth = 8388608`` (W75 doubled again).
8. ``long_horizon_retention_v28`` — 27 heads, eighteen-layer
   scorer, ``max_k = 960``.
9. ``mergeable_latent_capsule_v24`` — adds two new V24 chains
   (compound-chain-then-restart-trajectory chain; post-compound-
   chain-restart chain).
10. ``consensus_fallback_controller_v22`` — 38-stage chain (V21's
    36 + two new V22 stages: compound-chain-then-restart arbiter;
    post-compound-chain-restart-best-parent arbiter).
11. ``multi_agent_substrate_coordinator_v12`` — 26-policy, 16-
    regime MASC V12 (V11's 24 + ``substrate_routed_v21`` +
    ``team_substrate_coordination_v21``).
12. ``team_consensus_controller_v11`` — adds compound-chain-then-
    restart-pressure arbiter and post-compound-chain-restart-
    after-RTR arbiter.

**Plane A — Hosted control plane V9.** Seven modules:

1. ``hosted_router_controller_v9`` — adds
   ``weight_compound_chain_then_restart_pressure`` weighting and a
   per-(label, restart, rejoin, replacement, compound, chain,
   chain-then-restart) match table; per-budget+restart+rejoin+
   replacement+compound+chain+chain-then-restart routing CID.
2. ``hosted_logprob_router_v9`` — adds compound-chain-then-
   restart-aware abstain floor delta and a chain-then-restart
   tiebreak that further shrinks effective top-k under joint
   chain + restart pressure.
3. ``hosted_cache_aware_planner_v9`` — adds a *seventh* exa-
   coarse rotation cycle on top of V8's six; reports ≥ 88 %
   savings on 20 × 8 at hit_rate = 1.0.
4. ``hosted_cost_planner_v9`` — cost-per-compound-chain-then-
   restart-success-under-budget; abstain-when-compound-chain-
   then-restart-pressure-violated.
5. ``hosted_real_substrate_boundary_v9`` — extends the V8 wall
   with the three new V21 blocked axes (now ≥ 40 blocked axes).
6. ``hosted_real_handoff_coordinator_v8`` — adds compound-chain-
   then-restart-aware promotion + a new
   ``restart_after_compound_chain_repair_fallback`` decision label
   (eleventh on top of V7's ten); cross-plane savings ≥ 85 %.
7. ``hosted_provider_filter_v8`` — drops providers whose declared
   compound-chain-then-restart-noise exceeds per-provider cap.

Hard count: **19 substantial advances** beyond W75 (12 Plane B +
7 Plane A). All other W75 modules untouched (V20 substrate, V11
MASC, V10 TCC remain available; they are reused unchanged).

## Closed-form ridge solves added

W76 fits **three** new closed-form linear ridge solves on top of
W75's 73 (no autograd, no SGD, no GPU):

* Cache V19 sixteen-objective stacked ridge (n_features × 16) —
  1 new solve.
* Cache V19 per-role 17-dim compound-chain-then-restart-pressure
  head — re-uses the V18 family solver (counted as 0 new solves,
  same family).
* Replay V17 24-regime per-(role, regime) head — re-uses the V16
  family solver (0 new solves).
* Replay V17 14-way compound-chain-then-restart-aware routing
  head — 1 new solve.
* KV V21 seventeen-target stacked ridge — 1 new solve.

Total across W61..W76: **76 closed-form ridge solves**.

## Carried-forward W75 regimes (load-bearing, must keep)

All fifteen W75 regimes remain load-bearing:

1. ``baseline``
2. ``team_consensus_under_budget``
3. ``team_failure_recovery``
4. ``role_dropout``
5. ``branch_merge_reconciliation``
6. ``partial_contradiction_under_delayed_reconciliation``
7. ``agent_replacement_warm_restart``
8. ``multi_branch_rejoin_after_divergent_work``
9. ``silent_corruption_plus_member_replacement``
10. ``contradiction_then_rejoin_under_budget``
11. ``delayed_repair_after_restart``
12. ``delayed_rejoin_after_restart_under_budget``
13. ``replacement_after_contradiction_then_rejoin``
14. ``replacement_after_delayed_repair_under_budget``
15. ``compound_repair_after_replacement_then_rejoin_under_budget``

Plus the new W76 regime:

16. ``restart_after_compound_chain_repair_under_budget``

## Benchmark families — four families, 70+ H-bars

R-185 (Plane A V9), R-186 (Plane B V21), R-187 (multi-agent task
success across 16 regimes; **the primary scoreboard**), R-188
(handoff V8 + falsifier + limitation reproductions).

Total: **≥ 70 H-bars × 4 seed sets ≥ 280 cells**. Every cell
must pass.

## Quantitative bars

* MASC V12: V21 strictly beats V20 on ≥ 50 % of seeds in **every
  regime** (the 16 load-bearing ones). Practice: ≥ 80 %.
* TSC_V21 strictly beats TSC_V20 on ≥ 50 % of seeds in every
  regime. Practice: ≥ 80 %.
* V21 strictly beats V20 on ≥ 50 % of seeds on the new
  ``restart_after_compound_chain_repair_under_budget`` regime
  (V20 has no signal there).
* Cache planner V9 ≥ 88 % savings on 20 × 8 at hit_rate=1.0.
* Cost planner V9 finite cost-per-compound-chain-then-restart-
  success-under-budget at safe pressure; abstain at high pressure.
* Boundary V9 enumerates ≥ 40 blocked axes (W75's 37 + 3 V21).
* Provider filter V8 drops noisy provider under high
  compound-chain-then-restart pressure.
* Handoff V8 cross-plane savings ≥ 85 % (vs forcing every turn
  through hosted_only) at default V8 config; chain-then-restart-
  promotion sets ``compound_chain_then_restart_alignment = 1.0``.
* Substrate compound-chain-then-restart-repair-dominance flops
  saves ≥ 0.9 vs full recompute over twelve primitives.
* Substrate compound-chain-then-restart-pressure throttle saves
  > 0 visible tokens under tight budget.
* W75 envelope chaining preserved byte-for-byte (the new W76
  envelope's ``w75_outer_cid`` reproduces the input W75 envelope
  CID under trivial passthrough).

## Falsifiers (at least one)

* H-bar (R-188): KV V21 compound-chain-then-restart-pressure
  falsifier honest case scores 0; dishonest claim scores 1.
* H-bar (R-188): Handoff V8 compound-chain-then-restart
  falsifier honest=0, dishonest=1.
* H-bar (R-188): Boundary V9 falsifier honest=0 on the new V21
  axes, dishonest=1.

## Limitation reproductions

* H-bar: W76 substrate is still in-repo NumPy (CPU only). 22 →
  23 layers; default config still untrained. ``W76-L-NUMPY-CPU-
  V21-SUBSTRATE-CAP``.
* H-bar: Hosted plane V9 does **not** pierce the wall. ≥ 40
  blocked axes at the hosted surface. ``W76-L-HOSTED-V9-NO-
  SUBSTRATE-CAP``.
* H-bar: Frontier-model substrate access still blocked; W76
  carries the W70 ``frontier_blocked_axes`` set forward unchanged
  (``W76-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP``).
* H-bar: No SGD / autograd / GPU. ``W76-L-V21-NO-AUTOGRAD-CAP``.
* H-bar: ``coordpy.__version__ == "0.5.20"`` and
  ``SDK_VERSION == "coordpy.sdk.v3.43"`` byte-for-byte unchanged.

## Tests

* ``tests/test_w76_tiny_substrate_v21.py`` — V21 substrate
  determinism, chain-then-restart CID content-addressing, label
  fires under chain+restart, recompute-dominance flops saves,
  pressure-throttle saves.
* ``tests/test_w76_masc_v12.py`` — MASC V12 beats W74's V10
  bar and W75's V11 bar; V21 ≥ V20 ≥ V19 across 16 regimes.
* ``tests/test_w76_hosted_handoff_v8.py`` — handoff V8 envelope
  content-addressing; chain-then-restart promotion; chain-then-
  restart fallback fires; cross-plane savings ≥ 85 %.
* ``tests/test_w76_team.py`` — W76Team end-to-end smoke + W75
  envelope chaining preserved byte-for-byte under trivial
  passthrough.
* ``tests/test_w76_benchmarks.py`` — R-185..R-188 all_pass under
  three seed sets.
* ``tests/test_w76_modules.py`` — focused unit tests for each new
  module's witness CID + cid() stability.

## Theorem registry

Adds W76-T-* theorems for:

* Compound-chain-then-restart trajectory CID determinism.
* W21 V21 ≻ V20 strict-beat across all 16 regimes.
* W76 cache-V19 fifteen-objective ridge convergence (empirical).
* W76 KV V21 seventeen-target stacked ridge convergence
  (empirical).
* W76 handoff V8 wall preservation (code-backed).
* W76 frontier-substrate-still-blocked (limitation reproduction).

## Stable boundary preservation

* No coordpy.__version__ bump.
* No SDK_VERSION bump.
* No PyPI release.
* `tests/test_smoke_full.py` must still pass.
* Trivial-passthrough chain CID equal under
  ``W76Params.build_trivial()``.

## Strong success vs partial success vs failure

* **Strong success**: every quantitative bar passes; new regime
  load-bearing; four benchmark families pass; W75 regression
  passes; smoke driver passes; no instability across two
  consecutive runs.
* **Partial success**: ≥ 14 of 16 regimes hit V21 ≻ V20 ≥ 0.5;
  benchmark families pass but one bar is mildly under target;
  W75 regression and smoke driver pass.
* **Failure**: < 0.5 V21 ≻ V20 on more than 2 regimes, or
  benchmark family fails after two consecutive runs, or smoke
  driver regresses, or invariant byte-for-byte equality of
  W75 envelope under trivial-passthrough breaks.

## Storyline / paper line update

Adds the W76 paragraph at the top of
``papers/context_as_objects.md``:

> *W76* makes restart-after-compound-chain-repair load-bearing
> by minting a single content-addressed per-turn compound-chain-
> then-restart trajectory CID that absorbs the W75 V20 axes and
> the post-chain-repair restart-event chain into one substrate-
> routed signal. The new sixteenth regime ``restart_after_
> compound_chain_repair_under_budget`` measures whether the team
> can survive a fresh restart of the recovering member without
> losing the compound-chain-repair trajectory it just earned.

## Honest scope ("do not overstate")

* **Synthetic harness.** V21 beats V20 inside the deterministic
  MASC V12 harness only.
* **In-repo NumPy substrate.** V21 substrate is 23 layers, GQA
  8q/4kv, d_model=64, untrained.
* **No hidden-state / KV / attention access on hosted APIs.**
  V9 hosted modules read text + logprobs + prefix-cache hit only.
* **Frontier substrate access still blocked.** No third-party
  hidden-state, KV-byte, or attention-weight reads.
* **Compound-chain-then-restart pressure is caller-declared**;
  not an end-to-end learned controller.
