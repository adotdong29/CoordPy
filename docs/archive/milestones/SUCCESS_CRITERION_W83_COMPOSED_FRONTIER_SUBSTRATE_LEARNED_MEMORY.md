# Success criterion — W83 Composed Frontier-Substrate / Learned-Memory / Long-Horizon Multi-Agent Recovery

> Pre-committed, falsifiable, code-backed bar for the W83
> milestone. Set on **2026-05-18** before the W83 line landed.
> Failing any of these bars retracts W83 to "partial".

W83 follows the W80/W81/W82 P0/P1/P2 blocker-attack stack on top
of the W79 stronger-direct-blocker-attack milestone. W83 does NOT
attempt to redo the blocker-attack waves — those are done. W83
attempts to make the **composed multi-agent recovery line
materially stronger** by fusing the pieces W80–W82 already
shipped into a single learned, integrity-anchored pipeline.

## Milestone framing

* Carry forward all 19 W79 MASC V15 regimes unchanged.
* Add exactly **one** new regime:
  ``composed_long_horizon_under_compound_failure``. This regime
  composes long-horizon recall + simultaneous compound failure
  (contradiction + corruption + replacement + restart +
  blackout) at one of the W82 32-mask compound failure factor
  budgets.
* Do NOT alter MASC V15, the W79 substrate stack, or any
  W80/W81/W82 module. Compose on top.

## Mechanism advances (>= 12 required)

1. ``composed_learned_memory_v1`` — full-BPTT recurrent state +
   slot-router + slot-attention learned memory.
2. ``recurrent_slot_reconstruction_v1`` — differentiable LHR
   head over a fixed slot bank.
3. ``online_economics_refinement_v1`` — REINFORCE-style online
   refinement of W81 learned economics on a drifted deployment
   simulation.
4. ``DriftedDeploymentSimulationV1`` — per-action multiplier
   wrapper around the W81 ``EconomicsSimulationV1``.
5. ``integrity_trust_coupled_consensus_v1`` — composes W82
   integrity verdicts into W81 trust prior, with hard-drop of
   BAD_SIGNATURE / CORRUPT witnesses.
6. ``compose_repair_integrity_pipeline_v1`` — end-to-end
   pipeline: substrate restore → integrity verify → adversarial
   consensus repair → Merkle-anchored audit.
7. ``bounded_window_baseline_v3`` — strongest-known bounded
   baseline (k=256 + rolling summary + retrieval); plus
   ``prove_bounded_window_v3_insufficient_v1`` falsifier.
8. ``cross_runtime_hidden_state_projector_v1`` — learned least-
   squares projector across runtime signatures; beats the W82
   deterministic orthonormal projector.
9. ``hidden_state_intercept_bench_v1`` — live HF transformers
   substrate intercept bench (skip-friendly when HF/torch
   absent).
10. ``distributed_gateway_coordination_v1`` — composes the W81
    HTTP deployable gateway with the W82 distributed
    coordination semantics over real loopback TCP.
11. ``hosted_audit_anchoring_v1`` — content-addressed Merkle
    anchor for hosted-plane transcripts. Hosted-control-plane
    advance: tamper-evident audit chain without piercing hosted
    substrate.
12. ``composed_long_horizon_multi_agent_recovery_v1`` — composed
    pipeline benchmark across the 19 W79 regimes + the new
    ``composed_long_horizon_under_compound_failure`` regime.
13. ``r202_benchmark`` — composed team-success benchmark family.
14. ``r203_benchmark`` — strongest-known bounded baseline V3
    falsifier benchmark.
15. ``r204_benchmark`` — trainable-memory gauntlet benchmark.
16. ``w83_team`` — composed orchestrator wrapping the W79 team.

## Benchmark families (>= 3 required)

* **Family 1 (multi-agent task-success):** R-202 composed team-
  success benchmark across 20 regimes; reports per-regime task
  success, visible-token spend, recompute-flop spend, abstain
  rate, replay-vs-recompute breakdown.
* **Family 2 (bounded-window falsifier):** R-203 strongest-known
  bounded baseline V3 falsifier; demonstrates V3 abstains on
  100% of horizons past coverage.
* **Family 3 (trainable-memory):** R-204 composed_learned_memory
  + recurrent_slot_reconstruction head-to-head against ridge,
  W81 V2, W81 diffmem, nearest-slot.

## Carry-forward regimes (mandatory)

All 19 W79 MASC V15 regimes must be preserved unchanged. The
W83 composed pipeline runs them as drop-ins; W83 reports a
per-regime task_success_rate. **Bar:** per-regime success rate
>= 0.50 on every carry-forward regime.

## New regime (at most one)

``composed_long_horizon_under_compound_failure`` — combines
long-horizon recall with simultaneous compound failure (3
tampered + 1 dropped witnesses out of 7). **Bar:** task success
rate >= 0.50 on this regime.

## Required gain bars

| Bar | Requirement | Where verified |
| --- | --- | --- |
| **Substrate-coupling bar** | hidden-state intercept bench V1 emits a content-addressed trace CID on live HF runtime; skip-friendly | `coordpy.hidden_state_intercept_bench_v1`, `tests/test_w83_hidden_state_intercept_bench.py` |
| **Hosted control-plane gain** | hosted audit anchor's rebuilt Merkle root equals declared root on a 12-segment synthetic hosted run | `coordpy.hosted_audit_anchoring_v1`, `tests/test_w83_hosted_audit_anchoring.py` |
| **Real-substrate-plane gain** | composed pipeline beats W81 alone on mean error AND emits Merkle-anchored audit on every committed outcome | `coordpy.compose_repair_integrity_pipeline_v1`, `tests/test_w83_compose_repair_integrity_pipeline.py` |
| **Controlled-runtime / direct-blocker-attack gain** | distributed gateway coordination V1 ships migration over real HTTP loopback transport; sender + receiver root CIDs match | `coordpy.distributed_gateway_coordination_v1`, `tests/test_w83_distributed_gateway_coordination.py` |
| **Learned-memory gain** | composed_learned_memory beats W81 V2 (recurrent without slots) AND ridge | `coordpy.composed_learned_memory_v1`, `tests/test_w83_composed_learned_memory.py` |
| **Merge / consensus / disagreement bar** | integrity-trust-coupled consensus V1 beats W81 V1 on mean error AND on commit-rate under many-tampered scenarios | `coordpy.integrity_trust_coupled_consensus_v1`, `tests/test_w83_integrity_trust_coupled_consensus.py` |
| **Hostile-channel / corruption bar** | integrity-trust-coupled consensus V1 hard-drops BAD_SIGNATURE witnesses (integrity_witnesses_dropped > 0) | same |
| **Transcript-vs-shared-state bar** | composed pipeline returns a fused-value commit at task-success rate >= 80%, beating bounded-window-baseline V3 (which fails at 100% on horizons past coverage) | `coordpy.r202_benchmark` + `coordpy.r203_benchmark` |
| **Abstain / fallback bar** | composed pipeline emits abstain or replay under tampered scenarios; many-tampered probe drives W83 commit-rate strictly below W81 commit-rate | `tests/test_w83_integrity_trust_coupled_consensus.py::test_w83_itc_consensus_refuses_to_commit_under_many_tampered` |
| **Latent-to-KV bar** | hidden-state intercept bench tests live HF replay-from-KV; reports byte-identical on fp32 CPU when HF available | `coordpy.hidden_state_intercept_bench_v1` |
| **Latent-to-hidden-state or prefix-state bar** | hidden-state intercept bench tests hidden-state injection trace CID moves on live HF runtime | same |
| **Cache-reuse-vs-recompute bar** | composed_long_horizon_multi_agent_recovery_v1 distinguishes replay-used vs recompute paths in the per-scenario outcome | `coordpy.composed_long_horizon_multi_agent_recovery_v1` |
| **Multi-agent task-success bar** | R-202 reports overall task_success_rate >= 0.80 across 20 regimes | `coordpy.r202_benchmark` H1503 |
| **Team-success-per-visible-token bar** | R-202 reports mean visible-token spend per regime; mean across regimes is bounded | `coordpy.r202_benchmark` |
| **Team-success-per-recompute-flop bar** | R-202 reports mean recompute-flop spend per regime; mean across regimes is bounded | same |
| **Bounded-window-baseline failure bar** | R-203 H1604: V3 failure rate at horizon 1024 == 1.0 | `coordpy.r203_benchmark` |
| **Direct blocker-attack result** | Hidden-state intercept bench V1 reports a content-addressed live-HF trace; this is the live-runtime evidence | `coordpy.hidden_state_intercept_bench_v1` |
| **Falsifier** | R-203 emits BoundedWindowV3FailureProofV1 with failure_rate_beyond_coverage == 1.0 | `coordpy.bounded_window_baseline_v3` |
| **Limitation reproduction** | R-203's H1601/H1602 reproduce V3's *internal-window* and *internal-summary* success modes; these are baseline successes, not falsified | same |
| **Stable-boundary preservation** | every W83 module is explicit-import only; `coordpy.__version__` unchanged at 0.5.20; no PyPI release; coordpy/__init__.py untouched | `coordpy/_version.py`, `coordpy/__init__.py` |

## Explicit non-claims

* W83 does NOT pierce third-party hosted-model substrate.
  Hosted audit anchoring is over the observed text/logprob/
  prefix-cache surface only. The W79
  ``W79-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged.
* W83 does NOT validate composed_learned_memory against a live
  LLM hidden-state trace. The training data is synthetic.
* W83 does NOT replace W56–W79 substrate controllers. It
  composes on top.
* W83 does NOT replace MASC V15. The 19 W79 regimes are
  preserved; only one new regime is added at the recovery
  pipeline layer (not at MASC).
* W83 distributed gateway coordination V1 is loopback-only.
  Real cross-host networking is W84+ work.
* W83 hidden-state intercept bench skips when transformers /
  torch are not installed; CI on lean environments stays green.

## Strong, partial, failure

* **Strong success:** every bar above passes; the W83 team's
  ``verify_w83_handoff`` returns ``(True, [])``.
* **Partial:** one or two bars partially pass (e.g. composed
  beats some baselines but ties another); the milestone ships
  with explicit caveats in the result note.
* **Failure:** any load-bearing bar fails (e.g. composed loses
  to a baseline; R-202 below 0.80; falsifier reaches < 1.0
  failure rate). The milestone is retracted and the underlying
  module is fixed before re-landing.

## No version bump / no PyPI release

* ``coordpy.__version__`` stays at 0.5.20.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* All W83 modules are explicit-import only.
* Stable SDK surface (``RunSpec``, ``run``, ``AgentTeam``,
  ``coordpy-team`` CLI, etc.) is byte-for-byte unchanged.
