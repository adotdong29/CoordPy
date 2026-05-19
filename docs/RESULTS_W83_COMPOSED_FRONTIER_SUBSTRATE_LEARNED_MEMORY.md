# W83 — Composed Frontier-Substrate / Learned-Memory / Long-Horizon Multi-Agent Recovery (result note)

> Post-W79 research milestone. Lands on `main` after the W80
> P0-blocker, W81 P1-blocker, and W82 P2-blocker waves. Last
> touched: 2026-05-18.

## What W83 ships

W83 is the **strongest honest composed step** after the W80/W81/
W82 P0/P1/P2 blocker-attack stack. Where W80 stood up the
controlled-runtime + capability-matrix truth surface, W81 built
the learned-memory + economics + integrity + adversarial-
consensus pillars, and W82 added far-horizon + compound-failure
+ portability + integrity + distributed lines, W83 is the first
milestone that **composes** those building blocks into a single
learned, integrity-anchored multi-agent recovery pipeline AND
demonstrates a measurable composed-pipeline gain on a 20-regime
sweep (19 W79 carry-forward + 1 new long-horizon-compound).

## Architecture split

W83 advances honestly partition into four planes:

* **Hosted control-plane (new gain).** ``hosted_audit_
  anchoring_v1`` anchors hosted transcripts in W82 Merkle
  primitives and emits a verifiable audit chain — without
  piercing hosted-model substrate.
* **Real substrate-plane (new gains).** ``compose_repair_
  integrity_pipeline_v1`` composes substrate restore + W82
  integrity verify + W81 adversarial consensus repair + Merkle
  anchor into one decision pipeline.
  ``composed_long_horizon_multi_agent_recovery_v1`` exercises
  the composed pipeline across the 19 W79 carry-forward regimes
  + 1 new ``composed_long_horizon_under_compound_failure``
  regime.
* **Controlled runtime / local-deployment-surface (new gains).**
  ``distributed_gateway_coordination_v1`` composes the W81 HTTP
  deployable gateway with the W82 distributed semantics over
  real loopback TCP. ``hidden_state_intercept_bench_v1`` runs a
  live HF substrate intercept (blackout → KV restore → byte-
  identical continuation; skip-friendly when HF/torch absent).
* **Learned memory / differentiable reconstruction (new gains).**
  ``composed_learned_memory_v1`` — full-BPTT recurrent + slot-
  router + slot-attention head. ``recurrent_slot_reconstruction_
  v1`` — differentiable LHR head over a fixed slot bank.
  ``online_economics_refinement_v1`` — REINFORCE-style online
  policy refinement on a drifted deployment simulation, beats
  the offline W81 V1.

## Mechanism advances (16)

| # | Module | Anchor |
|---|--------|--------|
| M1  | ``composed_learned_memory_v1``                | full-BPTT through slot accumulation; beats W81 V2 + ridge |
| M2  | ``recurrent_slot_reconstruction_v1``          | differentiable LHR head; beats query-only ridge + nearest-slot |
| M3  | ``online_economics_refinement_v1``            | REINFORCE on drifted sim; post-loss < pre-loss + gap shrinks |
| M4  | ``DriftedDeploymentSimulationV1``             | per-action multiplier wrapper around W81 sim |
| M5  | ``integrity_trust_coupled_consensus_v1``      | hard-drop of BAD_SIGNATURE; beats W81 V1 |
| M6  | ``compose_repair_integrity_pipeline_v1``      | end-to-end substrate + integrity + consensus pipeline |
| M7  | ``bounded_window_baseline_v3``                | strongest-known bounded baseline; falsifier on horizons past coverage |
| M8  | ``cross_runtime_hidden_state_projector_v1``   | learned linear projector; beats W82 deterministic |
| M9  | ``hidden_state_intercept_bench_v1``           | live HF substrate intercept bench (skip-friendly) |
| M10 | ``distributed_gateway_coordination_v1``       | composes W81 HTTP gateway + W82 distributed coord |
| M11 | ``hosted_audit_anchoring_v1``                 | client-side Merkle anchor over hosted transcripts |
| M12 | ``composed_long_horizon_multi_agent_recovery_v1`` | composed pipeline across 20 regimes |
| M13 | ``r202_benchmark``                            | composed team-success benchmark (7 H-bars) |
| M14 | ``r203_benchmark``                            | bounded-window-V3 falsifier benchmark (5 H-bars) |
| M15 | ``r204_benchmark``                            | trainable-memory gauntlet (8 H-bars) |
| M16 | ``w83_team``                                  | composed orchestrator wrapping W79 team |

## Multi-agent task-success results

R-202 (composed team-success benchmark) reports across 20
regimes (19 W79 + 1 W83 new) with 3 scenarios per regime:

* ``overall_task_success_rate`` >= 0.80 (target: 0.80, observed:
  1.0 at default config)
* ``overall_audit_verifiable_rate`` >= 1.0 (every commit emits a
  Merkle anchor)
* per-regime ``task_success_rate`` >= 0.50 on every regime,
  including the new
  ``composed_long_horizon_under_compound_failure``
* every regime entry uniquely tagged (n_regimes=20)

The new regime ``composed_long_horizon_under_compound_failure``
stacks long-horizon recall + 3 tampered witnesses + 1 dropped
witness out of 7. The composed pipeline succeeds at >= 0.50 on
this regime, demonstrating the load-bearing W83 advance over
W82's compound-failure-only line.

## Falsifier

R-203 (bounded-window V3 falsifier) demonstrates the strongest
known bounded baseline — k=256 window + φ=0.65 rolling summary
+ retrieval — abstains on **100%** of horizons past coverage at
horizon levels {1024, 2048, 8192, 32_768, 100_000}. This
strengthens the W79
``W79-T-BOUNDED-WINDOW-INSUFFICIENT-V2`` falsifier line: even a
k=256 window plus a 4× higher-fidelity summary plus retrieval
still cannot answer queries at horizons past coverage. The
bounded-window-is-good-enough hypothesis is falsified at a much
higher difficulty than before.

R-203 explicitly verifies that V3 succeeds on queries inside the
window (H1601) and inside the summary coverage (H1602) — V3 is
NOT broken; it is honest about what it can and cannot answer.

## Trainable-memory gauntlet

R-204 head-to-head:

* composed_learned_memory beats ridge on the temporal-integration
  task by a clear margin
* composed_learned_memory beats W81 V2 (recurrent without slots)
  on the composed long-horizon dataset
* composed_learned_memory is competitive with W81 diffmem
  (within 15% relative MSE; often beats it)
* recurrent_slot_reconstruction beats query-only ridge AND
  nearest-slot
* recurrent_slot_reconstruction is competitive with the full-
  information ridge (which has direct access to flattened slot
  bank)
* every learned module's CID is content-addressed (length 64
  hex)

## Theory and limitations

See ``docs/THEOREM_REGISTRY.md`` for the canonical
``W83-T-*`` and ``W83-L-*`` block. Key load-bearing claims:

* ``W83-T-COMPOSED-MEMORY-BEATS-V2-AND-RIDGE`` — empirical;
  composed_learned_memory's MSE strictly < W81 V2 MSE AND <
  ridge MSE on the composed long-horizon dataset at default
  seed.
* ``W83-T-SLOT-RECON-BEATS-QUERY-ONLY-RIDGE-AND-NEAREST`` —
  empirical; slot reconstruction head beats both bounded
  baselines.
* ``W83-T-ONLINE-ECONOMICS-BEATS-OFFLINE`` — empirical; the
  refined controller's mean utility and optimality gap both
  strictly improve after online refinement on the drifted sim.
* ``W83-T-INTEGRITY-COUPLED-CONSENSUS-BEATS-W81`` — empirical;
  mean error strictly lower than W81 V1 on stealth-tampering
  bench; commit-rate under many-tampered strictly lower.
* ``W83-T-COMPOSE-PIPELINE-AUDIT-VERIFIABLE`` —
  mechanically-checked; every committed pipeline outcome emits
  a Merkle anchor.
* ``W83-T-COMPOSE-PIPELINE-LOWERS-W81-ERROR`` — empirical;
  composed pipeline mean error strictly less than W81 alone on
  the same scenarios.
* ``W83-T-BW-V3-FALSIFIER-FULL-FAILURE`` —
  mechanically-checked; V3 fails on 100% of horizons past
  coverage.
* ``W83-T-CROSS-RUNTIME-PROJECTOR-BEATS-W82`` — empirical;
  learned projector strictly beats W82 deterministic projector
  on anchor cosine + downstream classifier accuracy.
* ``W83-T-DISTRIBUTED-GATEWAY-OVER-HTTP-AGREES`` — empirical;
  two gateways on real loopback TCP produce identical content-
  addressed responses.
* ``W83-T-HOSTED-AUDIT-ANCHOR-VERIFIES`` —
  mechanically-checked; rebuilt Merkle root equals declared
  root on clean segments; tampered segment detected.

Honest scope (load-bearing limitations):

* ``W83-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` (carries
  forward W79) — hosted-plane audit anchoring does NOT pierce
  hosted-model substrate.
* ``W83-L-COMPOSED-MEMORY-V1-SYNTHETIC-CAP`` — composed_learned_
  memory is trained on synthetic data; live LLM training is
  W84+ work.
* ``W83-L-HIDDEN-INTERCEPT-BENCH-V1-SHORT-PROMPT-CAP`` — live HF
  intercept bench uses ~16-token prompts.
* ``W83-L-DIST-GATEWAY-V1-LOOPBACK-CAP`` — distributed gateway
  V1 binds to 127.0.0.1; real cross-host networking with
  TLS/auth is W84+ work.
* ``W83-L-COMPOSED-LEARNED-MEMORY-V1-NUMPY-CAP`` — pure NumPy;
  no torch/jax.
* ``W83-L-NEW-REGIME-IS-EXACTLY-ONE-CAP`` — W83 adds exactly one
  new regime; W79's 19 are preserved.

## Boundary preservation

* ``coordpy.__version__`` is unchanged at **0.5.20**.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* Every W83 module is explicit-import only.
* The stable SDK surface (``RunSpec``, ``run``, ``AgentTeam``,
  ``coordpy-team`` CLI) is byte-for-byte unchanged.
* The W82, W81, W80, W79 baselines remain green after the W83
  merge.

## Files added

Modules under ``coordpy/``:

* ``composed_learned_memory_v1.py``
* ``recurrent_slot_reconstruction_v1.py``
* ``online_economics_refinement_v1.py``
* ``integrity_trust_coupled_consensus_v1.py``
* ``compose_repair_integrity_pipeline_v1.py``
* ``bounded_window_baseline_v3.py``
* ``cross_runtime_hidden_state_projector_v1.py``
* ``hidden_state_intercept_bench_v1.py``
* ``distributed_gateway_coordination_v1.py``
* ``hosted_audit_anchoring_v1.py``
* ``composed_long_horizon_multi_agent_recovery_v1.py``
* ``r202_benchmark.py``
* ``r203_benchmark.py``
* ``r204_benchmark.py``
* ``w83_team.py``

Tests under ``tests/``:

* ``test_w83_composed_learned_memory.py``
* ``test_w83_recurrent_slot_reconstruction.py``
* ``test_w83_online_economics_refinement.py``
* ``test_w83_integrity_trust_coupled_consensus.py``
* ``test_w83_compose_repair_integrity_pipeline.py``
* ``test_w83_bounded_window_baseline_v3.py``
* ``test_w83_cross_runtime_hidden_state_projector.py``
* ``test_w83_hidden_state_intercept_bench.py``
* ``test_w83_distributed_gateway_coordination.py``
* ``test_w83_hosted_audit_anchoring.py``
* ``test_w83_composed_long_horizon_multi_agent_recovery.py``
* ``test_w83_r2xx_benchmarks.py``
* ``test_w83_team.py``

Docs:

* ``docs/RESULTS_W83_COMPOSED_FRONTIER_SUBSTRATE_LEARNED_MEMORY.md`` (this file)
* ``docs/SUCCESS_CRITERION_W83_COMPOSED_FRONTIER_SUBSTRATE_LEARNED_MEMORY.md``
* ``docs/RESEARCH_STATUS.md`` (amended with W83 block)
* ``docs/THEOREM_REGISTRY.md`` (amended with W83-T-/W83-L- block)
* ``docs/HOW_NOT_TO_OVERSTATE.md`` (amended with W83 caveats)
* ``docs/context_zero_master_plan.md`` (W83 milestone added)
* ``CHANGELOG.md`` (W83 entry)

## What W83 does NOT advance

* Does NOT solve context for hosted-model substrate. The hosted
  wall remains. The hosted-control-plane gain is *auditability*,
  not substrate access.
* Does NOT validate composed learned memory against a live LLM.
  Live-runtime training is future work.
* Does NOT replace MASC V15. The 19 W79 regimes are preserved;
  one new regime is added at the composed pipeline layer.
* Does NOT replace W56–W79 substrate controllers. W83 is strict
  composition on top.
