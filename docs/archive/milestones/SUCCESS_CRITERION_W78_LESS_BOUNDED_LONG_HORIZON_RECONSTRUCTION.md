# Success criterion — W78 Stronger Less-Bounded Long-Horizon Reconstruction / Bounded-Window-Falsifier Budget-Primary Two-Plane Multi-Agent Substrate Programme

Pre-committed before implementation, on `main`, no version bump,
no PyPI release. Direct extension of W77 (``coordpy/tiny_substrate_v22``
line; MASC V13 / TCC V12; Plane A V10 + handoff V9 + provider filter V9).

## Philosophical change the milestone must make load-bearing

W77 and predecessors compounded contradiction / replacement /
restart / chain mechanisms — but the *visible* state per agent
remained whatever the local turn observed. W78 deliberately moves
toward **genuinely less-bounded** state retention by:

1. Making the **bounded-window transcript baseline** an explicit
   first-class falsifier module (fixed-k windows at k ∈ {4, 8, 16,
   32}) the substrate is *expected to materially beat* on long-
   horizon reconstruction after compound-chain failure.
2. Introducing a **long-horizon reconstruction substrate**
   (``long_horizon_reconstruction_substrate_v1``) that can
   reconstruct the W77 compound-chain-then-restart-then-
   replacement trajectory from a persistent latent carrier even
   when the visible transcript has been truncated past any
   fixed-k window.
3. Making the new W78 regime
   ``long_delay_reconstruction_after_compound_chain_failure``
   the single new load-bearing regime: an arc where compound-
   chain failure occurs at ~30 % of turns, *then* a long
   visible-window-only blackout from ~30 % to ~85 %, then a
   reconstruction request at ~88 %. The bounded-window baselines
   are expected to fail this regime; the W78 substrate is
   expected to succeed.

## Headline win the milestone must produce

The team must reconstruct the W77 compound-chain-then-restart-
then-replacement trajectory after a **long visible-token
blackout** in which any fixed-k bounded-window transcript
baseline (k ∈ {4, 8, 16, 32}) has provably lost the original
event. That is a strictly harder failure modality than W77
because the substrate must survive without help from visible
context for >50 % of the run.

The W78 substrate must:

1. Expose a content-addressed **per-turn long-horizon
   reconstruction trajectory CID** that unifies all W77 V22
   primitives (post-restart-replacement trajectory CID plus the
   thirteen prior primitives) with the recorded
   **long-delay-then-reconstruction event chain**, and surface
   it as a per-turn signal back into the substrate-routed policy.
2. Expose a **per-layer long-horizon reconstruction length
   label** in [0..14]. V22's [0..13] are extended by 14 =
   ``long_delay_reconstruction_after_compound_chain_failure``
   (any layer on which a reconstruction event was observed after
   a long-delay blackout window).
3. Expose a **per-layer long-horizon-reconstruction-pressure
   gate** that throttles substrate work as a function of:
   (visible-token budget, blackout window length, prior-chain
   depth, reconstruction-request count).

Plane B (in-repo NumPy) and Plane A (hosted HTTP-text-only)
must both observe these new axes consistently with the W77 V10
wall and the W70 frontier-blocked set.

## Exactly one new regime — drop-in extension of MASC V13

``long_delay_reconstruction_after_compound_chain_failure``:

* Same chain regime as W77's
  ``replacement_after_restart_after_compound_chain_repair_under_
  budget`` up to turn ~30 % — the replacement / delayed-repair /
  rejoin / restart / replacement-after-restart chain fires
  unchanged.
* At ~30 % of turns, a compound-chain failure interrupts the
  arc; for turns 30 %..85 % the team operates under a *tight*
  visible-token budget that hides the compound-chain trajectory
  from any fixed-k bounded-window transcript.
* At ~88 % of turns, a **reconstruction request** is issued.
  Bounded-window baselines (k ∈ {4, 8, 16, 32}) cannot answer
  it because the relevant events are outside any fixed window.
  The W78 substrate's reconstruction module reads from the
  long-horizon persistent latent carrier and reconstructs the
  trajectory deterministically.

This is a single regime, strictly load-bearing, directly on the
three focus fronts (less-bounded state survival; compound-failure
robustness; replay / recompute / handoff economics).

## Modules — coherent V23 / V11 bundle (≥ 12 substantial advances)

**Plane B — Real substrate plane V23.** Thirteen modules:

1. ``tiny_substrate_v23`` — inherits V22's 23 physical layers +
   one new V23 substrate axis (long-horizon-reconstruction
   trajectory CID, length-per-layer in [0..14], reconstruction-
   pressure gate). New ``LongHorizonReconstructionWindow`` record.
2. ``kv_bridge_v23`` — 19-target ridge (V22's 18 + the new
   long-horizon-reconstruction target); 156-dim long-horizon-
   reconstruction fingerprint; long-horizon-reconstruction
   falsifier.
3. ``cache_controller_v21`` — 18-objective ridge (V20's 17 + the
   new long-horizon-reconstruction target); per-role 19-dim
   long-horizon-reconstruction head.
4. ``replay_controller_v19`` — 26 regimes (V18's 25 + the new
   long-delay-reconstruction regime); 16-label long-horizon-
   reconstruction-aware routing head.
5. ``persistent_latent_v30`` — 29 layers, 27th carrier (long-
   horizon-reconstruction-pressure EMA carrier),
   ``max_chain_walk_depth = 33554432``.
6. ``long_horizon_retention_v30`` — 29 heads, max_k = 2048,
   20-layer scorer (a real less-bounded retention bump).
7. ``mergeable_latent_capsule_v26`` — adds two new V26 chains
   (long-horizon-reconstruction-trajectory; reconstruction-
   request-window).
8. ``consensus_fallback_controller_v24`` — 42-stage chain (V23's
   40 + two new V24 stages: long-horizon-reconstruction arbiter;
   reconstruction-best-parent arbiter).
9. ``multi_agent_substrate_coordinator_v14`` — 30-policy, 18-
   regime MASC V14 (V13's 28 + ``substrate_routed_v23`` +
   ``team_substrate_coordination_v23``).
10. ``team_consensus_controller_v13`` — adds long-horizon-
    reconstruction-pressure arbiter and reconstruction-trajectory
    arbiter.
11. ``deep_substrate_hybrid_v23`` — 23-way bidirectional loop
    adding the new ``long_horizon_reconstruction_axis``.
12. ``substrate_adapter_v23`` — ``substrate_v23_full`` tier.
13. **``long_horizon_reconstruction_substrate_v1``** — NEW
    standalone reconstruction primitive. Reads the persistent
    latent V30 chain and reconstructs the W77 compound-chain-
    then-restart-then-replacement trajectory CID *without
    looking at the visible transcript at all*. This is the
    less-bounded state-retention win.

**Plane A — Hosted control plane V11.** Seven modules:

14. ``hosted_router_controller_v11`` — adds
    ``weight_long_horizon_reconstruction_pressure`` weighting +
    long-horizon-reconstruction-after-PCR match table.
15. ``hosted_logprob_router_v11`` — long-horizon-reconstruction-
    aware abstain floor delta + 9-pressure tiebreak.
16. ``hosted_cache_aware_planner_v11`` — *nine*-layer rotated
    prefix cycle on top of V10's eight; reports ≥ 89 % savings
    on 20 × 8 at hit_rate = 1.0 (matches the W77 V10 bar; the
    700-prefix/80-role-token mix structurally caps further
    savings).
17. ``hosted_cost_planner_v11`` — cost-per-long-horizon-
    reconstruction-success-under-budget; abstain-when-
    reconstruction-pressure-violated.
18. ``hosted_real_substrate_boundary_v11`` — wall V11 enumerates
    ≥ 46 blocked axes (V10's 43 + 3 new V23 axes).
19. ``hosted_real_handoff_coordinator_v10`` — long-horizon-
    reconstruction-aware promotion + new
    ``long_delay_reconstruction_after_compound_chain_failure_
    fallback`` decision label (twelfth on top of V9's eleven);
    cross-plane savings ≥ 87 %.
20. ``hosted_provider_filter_v10`` — drops providers whose
    declared long-horizon-reconstruction-noise exceeds the
    per-provider cap.

**Anti-goal mechanism — explicit bounded-window falsifier.**

21. **``bounded_window_baseline_v1``** — NEW load-bearing
    falsifier module. Implements four fixed-k transcript
    baselines (k ∈ {4, 8, 16, 32}) and a fixed rolling-summary
    baseline. **Each one is provably unable to answer a
    reconstruction request whose source events are >k turns ago**,
    by construction (the visible window literally does not
    contain the event). The W78 substrate is required to beat
    all four on the new regime. This is the bounded-context
    audit made load-bearing.

Hard count: **≥ 21 substantial advances** beyond W77 (13 Plane B
substrate + 7 Plane A hosted + 1 new bounded-window falsifier =
21). All other W77 modules untouched (V22 substrate, V13 MASC,
V12 TCC remain available; reused unchanged).

## Closed-form ridge solves added

W78 fits **two** new closed-form linear ridge solves on top of
W77's 77 (no autograd, no SGD, no GPU):

* Cache V21 eighteen-objective stacked ridge (n_features × 18) —
  1 new solve.
* KV V23 nineteen-target stacked ridge — 1 new solve.

Total across W61..W78: **79 closed-form ridge solves**.

## Carried-forward W77 regimes (load-bearing, must keep)

All seventeen W77 regimes remain load-bearing:

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
16. ``restart_after_compound_chain_repair_under_budget``
17. ``replacement_after_restart_after_compound_chain_repair_under_budget``

Plus the new W78 regime:

18. ``long_delay_reconstruction_after_compound_chain_failure``

## Benchmark families — four families, 80+ H-bars

* R-193 (Plane A V11) — hosted control plane
* R-194 (Plane B V23) — real substrate
* R-195 (multi-agent task success across 18 regimes; **the
  primary scoreboard**, includes bounded-window-baseline
  failure bar)
* R-196 (handoff V10 + bounded-window falsifier + long-horizon
  limitation reproductions)

Total: **≥ 80 H-bars × 3+ seed sets ≥ 250 cells**. Every cell
must pass.

## Quantitative bars

* MASC V14: V23 strictly beats V22 on ≥ 50 % of seeds in **every
  regime** (the 18 load-bearing ones). Practice: ≥ 80 %.
* TSC_V23 strictly beats TSC_V22 on ≥ 50 % of seeds in every
  regime. Practice: ≥ 80 %.
* **Bounded-window-baseline failure bar**: on the new regime,
  bounded-window baselines (k ∈ {4, 8, 16, 32}) achieve
  *zero* successful reconstructions; the W78 substrate achieves
  ≥ 50 % on the same regime, strictly beating each baseline on
  100 % of seeds.
* Cache planner V11 ≥ 89 % savings on 20 × 8 at hit_rate=1.0
  (matches the W77 V10 bar; the 700-prefix/80-role-token mix
  structurally caps further savings).
* Cost planner V11 finite cost-per-long-horizon-reconstruction-
  success-under-budget at safe pressure; abstain at high
  pressure.
* Boundary V11 enumerates ≥ 46 blocked axes (W77's 43 + 3 V23).
* Provider filter V10 drops noisy provider under high
  long-horizon-reconstruction pressure.
* Handoff V10 cross-plane savings ≥ 87 % (vs forcing every turn
  through hosted_only) at default V10 config; long-horizon-
  reconstruction-promotion sets
  ``long_horizon_reconstruction_alignment = 1.0``.
* Substrate long-horizon-reconstruction-dominance flops saves
  ≥ 0.92 vs full recompute over fourteen primitives.
* Substrate long-horizon-reconstruction-pressure throttle saves
  > 0 visible tokens under tight budget.
* **Long-horizon retention bar**: V30 retention head retains
  ≥ 95 % of compound-chain trajectory CIDs across a blackout of
  ≥ 200 turns; bounded-window baselines retain 0 % outside the
  fixed window.
* W77 envelope chaining preserved byte-for-byte (the new W78
  envelope's ``w77_outer_cid`` reproduces the input W77
  envelope CID under trivial passthrough).

## Falsifiers (≥ 1 mandatory, W78 has three)

* H-bar (R-196): KV V23 long-horizon-reconstruction falsifier
  honest case scores 0; dishonest claim scores 1.
* H-bar (R-196): Handoff V10 long-horizon-reconstruction
  falsifier honest=0, dishonest=1.
* H-bar (R-196): **Bounded-window falsifier** — for each
  k ∈ {4, 8, 16, 32}, prove that the bounded-window baseline
  cannot answer reconstruction queries with source >k turns
  ago: honest=0, dishonest=1.

## Limitation reproductions

* H-bar: W78 substrate is still in-repo NumPy (CPU only). V22 →
  V23 is one layer-axis bump. Default config still untrained.
  ``W78-L-NUMPY-CPU-V23-SUBSTRATE-CAP``.
* H-bar: Hosted plane V11 does **not** pierce the wall. ≥ 46
  blocked axes at the hosted surface. ``W78-L-HOSTED-V11-NO-
  SUBSTRATE-CAP``.
* H-bar: Frontier-model substrate access still blocked; W78
  carries the W70 ``frontier_blocked_axes`` set forward
  unchanged (``W78-L-FRONTIER-SUBSTRATE-STILL-BLOCKED-CAP``).
* H-bar: No SGD / autograd / GPU. ``W78-L-V23-NO-AUTOGRAD-CAP``.
* H-bar: **Bounded-window baselines fail by construction on
  reconstruction queries outside the window**. This is a
  fundamental insufficiency of bounded-context architectures
  on long-horizon multi-agent state retention.
  ``W78-L-BOUNDED-WINDOW-INSUFFICIENT-CAP``.
* H-bar: ``coordpy.__version__ == "0.5.20"`` and
  ``SDK_VERSION == "coordpy.sdk.v3.43"`` byte-for-byte unchanged.

## Tests

* ``tests/test_w78_modules.py`` — focused unit tests for each
  new module's witness CID + cid() stability.
* ``tests/test_w78_bounded_window_baseline.py`` — explicit
  bounded-window-failure-by-construction proof and W78-beats-
  bounded-window-baseline bar.
* ``tests/test_w78_long_horizon_reconstruction.py`` — W78
  reconstruction module deterministically reconstructs the W77
  compound-chain-then-restart-then-replacement trajectory CID
  from the persistent latent carrier alone.
* ``tests/test_w78_benchmarks.py`` — R-193..R-196 all_pass under
  three seed sets.
* ``tests/test_w78_team.py`` — W78Team end-to-end smoke + W77
  envelope chaining preserved byte-for-byte under trivial
  passthrough.

## Theorem registry

Adds W78-T-* theorems for:

* Long-horizon-reconstruction trajectory CID determinism.
* W78 V23 ≻ V22 strict-beat across all 18 regimes.
* W78 cache-V21 eighteen-objective ridge convergence (empirical).
* W78 KV V23 nineteen-target stacked ridge convergence
  (empirical).
* W78 handoff V10 wall preservation (code-backed).
* W78 frontier-substrate-still-blocked (limitation reproduction).
* **W78-T-BOUNDED-WINDOW-INSUFFICIENT** — bounded-window
  baselines provably cannot answer reconstruction queries whose
  source events lie outside their fixed window. *Proof
  sketch:* by construction, the visible-window has no
  information about events outside it; therefore any predictor
  that operates only on the visible window is uniform over
  events outside it and cannot strictly beat random on
  ≥ 1-bit-of-information reconstruction queries.

## Stable boundary preservation

* No coordpy.__version__ bump.
* No SDK_VERSION bump.
* No PyPI release.
* `tests/test_smoke_full.py` must still pass.
* Trivial-passthrough chain CID equal under
  ``W78Params.build_trivial()``.

## Strong success vs partial success vs failure

* **Strong success**: every quantitative bar passes; new regime
  load-bearing; four benchmark families pass; W77 regression
  passes; smoke driver passes; bounded-window-baseline failure
  bar passes; no instability across two consecutive runs.
* **Partial success**: ≥ 16 of 18 regimes hit V23 ≻ V22 ≥ 0.5;
  benchmark families pass but one bar is mildly under target;
  W77 regression and smoke driver pass.
* **Failure**: < 0.5 V23 ≻ V22 on more than 2 regimes, or
  benchmark family fails after two consecutive runs, or smoke
  driver regresses, or invariant byte-for-byte equality of
  W77 envelope under trivial-passthrough breaks, or bounded-
  window-baseline failure bar does not hold.

## Storyline / paper line update

Adds the W78 paragraph at the top of
``papers/context_as_objects.md``:

> *W78* moves the programme toward genuinely **less-bounded**
> multi-agent context by making the bounded-window transcript
> baseline an explicit first-class falsifier and by introducing
> a long-horizon reconstruction substrate that reconstructs the
> W77 compound-chain-then-restart-then-replacement trajectory
> CID from a persistent latent carrier across a long visible-
> token blackout. The new eighteenth regime
> ``long_delay_reconstruction_after_compound_chain_failure``
> measures whether the team can recover the compound-chain
> trajectory after a > 200-turn visible-token blackout — a
> task bounded-window baselines (k ∈ {4, 8, 16, 32}) cannot do
> by construction.

## Honest scope ("do not overstate")

* **Synthetic harness.** V23 beats V22 inside the deterministic
  MASC V14 harness only.
* **In-repo NumPy substrate.** V23 substrate is 23 layers (one
  V23 cache axis on top of V22), GQA 8q/4kv, d_model=64,
  untrained.
* **No hidden-state / KV / attention access on hosted APIs.**
  V11 hosted modules read text + logprobs + prefix-cache hit
  only.
* **Frontier substrate access still blocked.** No third-party
  hidden-state, KV-byte, or attention-weight reads.
* **Long-horizon-reconstruction pressure is caller-declared**;
  not an end-to-end learned controller.
* **Bounded-window failure is a structural claim**, not a
  scaled-up benchmark of real frontier transcripts; it holds
  by construction within the synthetic regime.
