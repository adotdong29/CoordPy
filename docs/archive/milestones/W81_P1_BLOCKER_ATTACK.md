# W81 P1-Blocker Attack — overview

> Closes the five P1 child blockers from the meta issue (#4):
> #7 (deployable substrate gateway), #9 (sequence-conditioned
> learned consolidation), #14 (learned multi-runtime economics),
> #19 (differentiable memory substrate), #20 (adversarial
> consensus & repair). Builds on top of the W80 P0 wave (PR #21).

## Sequencing of the P1 attack

The meta issue's recommended sequence places P1 in two phases:

* **Phase 2 — Honest evaluation bridge.** #7 (deployable
  substrate gateway).
* **Phase 3 — Learned memory and economics.** #9 → #14 → #19 →
  #20.

W81 executes the P1 wave in that order, taking the W80 truth-
surface foundation (controlled runtime instrumentation contract
+ HF transformers backend + parity matrix + living capability
matrix + R-201 live local-model benches) as given and building
the next-wave research line on top.

| # | Module | Module path | Tests |
|---|--------|-------------|-------|
| #7  | Deployable Substrate Gateway V1            | `coordpy.deployable_substrate_gateway_v1`        | 18 |
| #9  | Sequence-Conditioned Learned Consolidation V2 | `coordpy.learned_consolidation_v2`           | 10 |
| #14 | Learned Multi-Runtime Economics Controller V1 | `coordpy.learned_economics_controller_v1`    | 14 |
| #19 | Differentiable Memory Substrate V1           | `coordpy.differentiable_memory_substrate_v1`  | 11 |
| #20 | Adversarial Consensus & Repair V1            | `coordpy.adversarial_consensus_repair_v1`     | 12 |

Total: **65 new tests, all pass on Python 3.11 with NumPy.**

## What's load-bearing about each pillar

### #7 — Deployable substrate gateway V1

The W79 façade was honest about its limitation:
``W79-L-FACADE-IN-PROCESS-CAP`` recorded that the façade was
in-process only. W81 closes that gap with:

* **Real HTTP transport.** ``http.server``-backed (stdlib only;
  no third-party dependency). ``serve_forever(host, port)``
  binds a real TCP listener, ``stop()`` cleans up.
* **OpenAI-compatible routes.** ``/v1/chat/completions`` and
  ``/v1/completions`` produce OpenAI-shaped JSON. Clients with
  ``base_url`` pointed at the gateway get a valid response.
* **Explicit substrate endpoints.** ``/v1/substrate/forward``,
  ``/v1/substrate/replay``, ``/v1/substrate/conformance``,
  ``/v1/substrate/capabilities``. Substrate access is routed
  through dedicated paths, not buried in the chat completion
  envelope. The chat completion shape *can* return a side
  channel, but only when both the gateway config and the
  request opt in.
* **Content addressing everywhere.** Every request and response
  has a CID. The audit log is keyed on (request_cid,
  response_cid).
* **Auth shim.** Bearer-token check, configurable. ``None``
  means open for local dev; setting it locks the gateway down
  for research deployments.
* **End-to-end bench.**
  ``run_gateway_end_to_end_bench_v1()`` exercises chat
  completion + substrate forward + replay + conformance through
  the gateway, returns a content-addressed report.

The honest caveats are explicit in the module docstring and in
the ``/v1/substrate/capabilities`` response:

* gateway does NOT pierce hosted-model substrate
* V1 routes to ``controlled_runtime_substrate_v1`` (in-repo
  NumPy runtime, not a frontier-scale model)
* V1 does NOT implement streaming, TLS, or signed-JWT auth

### #9 — Sequence-conditioned learned consolidation V2

W79's V1 was a pointwise two-layer nonlinear head — important
proof-of-direction (a learned head beats closed-form ridge on
a nonlinear target) but not yet sequential. V2 closes that gap:

* **Recurrent state-space update.** ``h_t = tanh(W_h h_{t-1} +
  W_x x_t + b_h)`` carries information across timesteps.
* **Learned write and read heads.** ``m_t = M_W h_t + M_b``
  (write), ``y_t = M_R m_t + M_Rb`` (read).
* **End-to-end BPTT training.** Hand-rolled NumPy autograd
  with analytical backward through the tanh recurrent core
  and the linear heads. Deterministic on seed.
* **Three baselines.** V2 strictly beats closed-form
  pointwise ridge, V1's pointwise two-layer head, AND a k=2
  bounded-window ridge — on a synthetic temporal-integration
  dataset where the target at each timestep depends on the
  history of inputs (decay + delayed echo at lag 3).

### #14 — Learned multi-runtime economics controller V1

The W79 outer line uses closed-form ridge controllers and
hand-coded arbitration. V1 stands up a learned policy:

* **Five canonical actions.** ``replay``,
  ``runtime_recompute``, ``transcript_recompute``,
  ``promote_to_richer_substrate``, ``abstain`` — the issue's
  full menu.
* **Seven-dim feature schema.** horizon, budget pressure,
  evidence completeness, prior failure rate, cache freshness,
  task difficulty, controlled-runtime health. Deterministic
  schema, content-addressed.
* **2-layer softmax policy.** ``logits = W2 @ swish(W1 @ x +
  b1) + b2``. Trained supervised on a synthetic optimal-
  action dataset derived from a deterministic cost/quality
  simulation.
* **Heuristic baseline included.** ``heuristic_economics_choice``
  reproduces the existing pressure-gate / cache-fresh-gate
  arbitration logic so V1 has a real baseline to beat.
* **Two load-bearing wins.** On the held-out eval set:
  * V1 mean utility strictly exceeds heuristic mean utility.
  * V1 optimality gap (vs simulation's true optimum) is
    strictly smaller than heuristic optimality gap.
* **No collapse.** V1 uses at least 3 of the 5 actions on the
  eval set — i.e. the learned policy does not degenerate to
  always-replay or always-abstain.

### #19 — Differentiable memory substrate V1

The W79 outer line is ridge-heavy. P1 #19 asks for a frontier
differentiable-memory line. V1 stands up:

* **State-space recurrent core.** Linear ``A`` matrix + tanh
  residual gives state-space dynamics.
* **K learned addressable memory slots.** ``slot_keys`` are
  learned. Each timestep writes to all slots (gated by a
  per-slot sigmoid ``alpha_t``) and reads via softmax
  attention over the slot keys.
* **Compressed-snapshot CID.** Final K-slot state is content-
  addressed via ``compressed_snapshot_cid(X=...)``. Identical
  inputs -> identical CIDs; different inputs -> different CIDs.
* **End-to-end gradient training.** Analytical gradients
  through the output head, the read attention, the write head,
  and the recurrent core via BPTT (slot accumulation treated
  with per-step detachment for tractability).
* **Two head-to-head wins.** On a delayed-recall task where
  the target at each timestep equals ``tanh(x_{t-4} @ W_recall)``:
  * V1 (slot memory) strictly beats V2 (single recurrent state,
    hidden_dim 8 — strictly smaller memory capacity than V1's
    K*D_mem = 128 floats).
  * V1 strictly beats closed-form pointwise ridge.

### #20 — Adversarial consensus & repair V1

The W56..W79 consensus chain has 44 stages of hand-coded
arbitration. P1 #20 asks for a principled trust-weighted
fusion line. V1 stands up:

* **Four canonical decision kinds.** ``commit``, ``abstain``,
  ``escalate_to_richer_substrate``, ``replay_from_trusted``.
* **Delay-decay prior trust.** ``prior_t = self_confidence *
  2^(-delay / halflife)``. Late witnesses count less; the
  team doesn't need to wait for stragglers.
* **Corruption-suspicion penalty.** Witnesses far from the
  cluster get an exponential trust penalty. Adversarial
  witnesses drop to < 10% of typical honest trust.
* **Bootstrap CI on the fused estimate.** Trust-weighted
  re-sampling produces a confidence-interval half-width;
  wide CI -> abstain or replay routing.
* **Escalate when corruption looks systematic.** When the
  max-deviation-vs-median ratio crosses
  ``escalate_corruption_threshold``, the controller routes
  to ``escalate_to_richer_substrate`` rather than
  committing.
* **Replay when CI is wide but trust exists.** Wide CI +
  at-least-one above-floor witness -> replay rather than
  abstain.
* **Empirical robustness win on the V1 bench.** On n=7,
  f=2, 80 seeds: V1 strictly beats naive averaging on mean
  error AND wins on ≥ 80% of seeds. V1 is also competitive
  with the median (beats it on ≥ 40% of seeds) while adding
  explicit abstain/escalate/replay routing.
* **Content-addressed audit chain.** Every decision carries
  an ``audit_cid`` hashed over (config_cid, witness_cids,
  decision_kind, trust_cid, ci_hw, susp, fused_cid).

## What W81 does NOT claim

See ``docs/HOW_NOT_TO_OVERSTATE.md`` for the full list. The
key non-claims:

* W81 does not pierce the hosted wall. The gateway routes to a
  controlled local runtime; substrate axes are real only
  because the runtime is controlled.
* W81 does not displace the W56..W79 main scoreboard. The
  differentiable-memory and adversarial-consensus lines are
  new research lines, not drop-in replacements for the
  existing closed-form controllers.
* W81 does not provide analytical bounds. The empirical wins
  hold on the W81 benchmark families; analytical proofs are
  follow-on work.
* W81 does not validate against live local-model long-horizon
  reconstruction. The learned-memory and economics lines are
  trained on synthetic data; live-runtime coupling is W82+
  work.

## How to verify

```
python -m pytest tests/test_w81_deployable_substrate_gateway.py \
                 tests/test_w81_learned_consolidation_v2.py \
                 tests/test_w81_learned_economics_controller.py \
                 tests/test_w81_differentiable_memory_substrate.py \
                 tests/test_w81_adversarial_consensus_repair.py
```

Expected: 65 passed.

W79 baseline (28/28) and W80 baseline (12/12 instrumentation +
parity, plus substrate adapter V25) remain green after the
W81 merge.
