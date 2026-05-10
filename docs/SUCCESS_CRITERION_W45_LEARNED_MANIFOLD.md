# Pre-committed success criterion — W45 Learned Manifold Controller

> Programme step: post-W44. Mints axis 42 of the Context Zero
> programme. Strictly additive on top of W44 LMCC, W43 PMC, and the
> released v3.43 line. Honest scope: the W45 mechanism replaces the
> hand-designed thresholds and channel-equal weighting of the W44
> live gate with a **learned, closed-form-fittable, content-addressed
> manifold controller** that operates over the W43 product-manifold
> channels and emits richer model-facing prompt control. It does
> NOT close the W43 conjectures
> (`W43-C-MIXED-CURVATURE-LATENT`, `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) and does NOT promote the
> W44-C-LIVE-LATENT carry-forward to a closed claim — the learned
> controller is an executable proxy at the capsule layer, not
> transformer-internal access.

## Mechanism

W45 introduces the **Learned Manifold Controller (LMC)** — the
first capsule-native CoordPy layer that *learns* a manifold-
conditioned routing / gating policy from cell observations and
emits richer, content-addressed model-facing prompt control. The
mechanism is composed of five learned components, all
closed-form-fittable in pure NumPy/stdlib so they can be trained
deterministically from a tiny synthetic bank without an autograd
dependency:

* **Learned channel encoder.** Each of the six W43 channels
  (hyperbolic, spherical, euclidean, factoradic, subspace, causal)
  is mapped through a frozen-then-fitted projection head to a
  fixed-dim feature vector. The hyperbolic and euclidean channels —
  audit-only at the W44 layer — become *features* that the learned
  controller can consume, closing one direction of the open
  W44-C-LIVE-LATENT carry-forward at the capsule layer.

* **Attention-style routing over channels.** A small softmax-
  weighted attention head pools the six channel feature vectors
  into a single scalar gate logit. The attention weights are
  fit-on-policy (closed-form ridge regression on synthetic
  positive/negative cells); the per-channel weights are part of
  the audit envelope so an auditor can trace which channel drove
  any single gating decision.

* **Adapter-decomposed role-specific policy.** Following the
  shared-base / role-specific-delta decomposition pattern
  ("LoRA-style at the capsule layer"), the gate policy = shared
  base + low-rank role-specific delta. The shared base is fitted
  across the role universe; per-role deltas are rank-1 perturbations
  of the base. This is the strongest executable approximation we
  can do for "role-specific KV state" without substrate access.

* **Margin-calibrated gating.** The hard W44 thresholds
  (`spherical_agreement_min`, `subspace_drift_max`) are replaced
  with a learned **signed margin**: the gate logit + bias maps to
  a sigmoid-shaped ratify probability with a learned calibration
  point. The W44 abstain branches still fire when the margin
  crosses zero, but the crossing point is now per-signature and
  measurable.

* **Factoradic-conditioned learned prompt hint.** The W44 factoradic
  compressor emitted only `FACTORADIC_ROUTE: <int>`. W45 adds a
  learned **manifold hint** — `MANIFOLD_HINT: route=<int>
  conf=<integer>` — where `<conf>` is a quantized confidence bucket
  drawn from the gate logit. The hint is content-addressed and the
  full controller state is recoverable bijectively from the
  envelope, so an auditor can prove which prompt bytes the model
  actually saw.

The W45 layer is strictly additive on top of W44 and the released
v3.43 SDK. When the learned controller is configured with the
zero-fitted seed (`learned_enabled=False`, `prompt_hint_mode='off'`,
`use_attention_routing=False`, `role_adapter_disabled=True`), the
W45 orchestrator reduces to `LiveManifoldTeam.run` byte-for-byte —
the **W45-L-TRIVIAL-LEARNED-PASSTHROUGH** falsifier.

## Pre-committed hypotheses (H1..H12)

Each hypothesis is testable from the bundled R-92 benchmark family
(`coordpy.r92_benchmark`) and its accompanying pytest tests
(`tests/test_r92_benchmark.py`). Pre-committed seed set:
`(0, 1, 2, 3, 4)`. Backend: deterministic `SyntheticLLMClient`
with hint-conditioned response variants for the model-facing
families; a real Ollama backend is supported through the same
surface but is not load-bearing for the pre-committed bar.

* **H1 — Trivial learned passthrough is byte-for-W44.**
  Family `r92_trivial_learned_passthrough`: with the zero-fitted
  registry (no learned components active), `LearnedManifoldTeam.run`
  produces the same `final_output`, `turns`, and capsule chain head
  as `LiveManifoldTeam.run` for the same backend.
  **Pass if**: `passthrough_ok = 1.0` across 5/5 seeds.

* **H2 — Learned calibration strictly raises precision over W44 hand
  thresholds on hard-to-threshold regimes.**
  Family `r92_learned_calibration_gain`: a regime where the W44
  hand-designed `spherical_agreement_min = 0.85` produces false
  abstains on borderline-but-honest cells. The W45 controller is
  fit on a synthetic bank; the learned threshold rejects the bad
  cells while accepting the borderline-honest cells.
  **Pass if**: `learned.mean - w44.mean >= 0.20` precision gain
  AND `learned.min == max == 1.0` across 5/5 seeds.

* **H3 — Attention routing specializes per signature.**
  Family `r92_attention_specialization`: two distinct role-handoff
  signatures with different "diagnostic channel" structure (one
  needs the spherical channel; the other needs the subspace
  channel). The learned attention weights differ measurably by
  signature.
  **Pass if**: `||w_sig1 - w_sig2||_1 >= 0.5` AND each signature's
  top-attention channel matches the diagnostic channel for that
  signature, across 5/5 seeds.

* **H4 — Role-specific adapter strictly improves per-role precision.**
  Family `r92_role_adapter_recovery`: a 4-role team where one role
  has a systematically different handoff distribution. The learned
  role-specific delta lifts that role's precision over the shared-
  base-only baseline.
  **Pass if**: `with_adapter.mean - shared_only.mean >= 0.20` on the
  role-specific subset across 5/5 seeds, AND the shared base is
  unchanged byte-for-byte regardless of role-delta presence (audit
  invariant).

* **H5 — Factoradic-conditioned hint preserves full route + adds a
  measurable confidence channel.**
  Family `r92_factoradic_hint_compression`: with
  `prompt_hint_mode='factoradic_with_hint'`, the prompt's
  `MANIFOLD_HINT` is recoverable byte-for-byte from the envelope
  and adds a 4-bucket confidence integer drawn from the learned
  margin.
  **Pass if**: hint bytes recoverable across 5/5 seeds AND the
  confidence histogram has at least 2 distinct buckets across the
  bench AND visible-token cost is bounded (`live.max - w44.max <= 4`
  added tokens per turn).

* **H6 — Model-facing hint response: synthetic hint-aware backend
  improves task-correct rate.**
  Family `r92_model_facing_hint_response`: a deterministic
  `HintAwareSyntheticBackend` that returns one canonical answer
  when the prompt contains `MANIFOLD_HINT: route=` and a different
  answer otherwise. The W45 arm produces a strictly higher
  task-correct rate than the W44 arm because the W44 prompt does
  not carry the hint.
  **Pass if**: `learned.mean - w44.mean >= 0.40` across 5/5 seeds.

* **H7 — No false abstention on linear-flow falsifier.**
  Family `r92_w45_falsifier`: a clean linear-flow regime where
  the geometry adds nothing. The learned controller must not
  trigger spurious abstentions even after fitting on a
  representative bank that includes both clean and dirty cells.
  **Pass if**: `learned.min == max == 1.0` across 5/5 seeds.

* **H8 — W45 compromised-observation cap (limitation).**
  Family `r92_w45_compromise_cap`: when the adversary forges all
  six channel observations to look honest, the learned controller
  cannot recover. This is the W45 counterpart of
  `W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP` strengthened to all
  channels.
  **Pass if**: `learned.mean == w44.mean == 0.0` across 5/5 seeds
  (a *limitation* falsifier — passing means the limitation
  reproduces honestly).

* **H9 — Learned envelope verifier enumerates ≥ 14 disjoint failure
  modes.**
  `verify_learned_manifold_handoff` returns the empty-envelope
  failure as the first mode, recomputes every component CID, and
  detects tampering with any subfield through one of the disjoint
  named modes (parameter CID, attention witness CID, role adapter
  CID, hint witness CID, causal mask witness CID, ...).
  **Pass if**: a successful verification reports `n_checks >= 14`.

* **H10 — Replay determinism: bit-perfect replay of a learned-
  controller run.**
  For any deterministic backend, two independent
  `LearnedManifoldTeam.run` invocations over the same task with
  the same registry, agents, observation builder, and fitted
  controller params produce byte-identical `final_output`,
  byte-identical capsule chain root CIDs, byte-identical sequences
  of `LearnedManifoldHandoffEnvelope`, AND byte-identical
  controller-parameter CIDs.
  **Pass if**: `r92_replay_determinism` passes across 5/5 seeds.

* **H11 — Cumulative trust boundary across W22..W45.**
  W45 introduces ≥ 14 new disjoint enumerated failure modes. The
  cumulative trust-boundary count is therefore at least
  `196 + 18 (W43) + 12 (W44) + 14 (W45) = 240`.
  **Pass if**: `len(W45_ALL_FAILURE_MODES) >= 14` AND the modes
  are disjoint from W22..W44's named modes.

* **H12 — Stable SDK contract preserved.**
  CoordPy 0.5.20's stable smoke driver (`tests/test_smoke_full.py`)
  reports "ALL CHECKS PASSED" with the W45 module on disk. The
  W45 surface is held outside `coordpy.__experimental__`; it is
  reachable only through an explicit
  `from coordpy.learned_manifold import ...` import. The released
  v0.5.20 wheel's public surface is byte-for-byte unchanged.
  **Pass if**: smoke driver reports "ALL CHECKS PASSED" with the
  W45 module installed AND `coordpy.__version__ == "0.5.20"` AND
  `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`.

## Falsifiers + per-channel verdicts

Each W43 channel is force-verdicted at the *learned* layer:

| Channel             | Pre-W45 verdict   | W45 verdict                    |
|---------------------|-------------------|--------------------------------|
| Spherical consensus | live gate         | learned-margin gate            |
| Subspace drift      | live gate         | learned-margin gate            |
| Causal clock        | live gate         | learned-margin gate            |
| Factoradic route    | live compressor   | learned + confidence-bucketed  |
| Hyperbolic branch   | audit-only        | **learned feature** (consumed) |
| Euclidean attribute | audit-only        | **learned feature** (consumed) |

This is the structural lift over W44: the hyperbolic and euclidean
channels — audit-only at the W44 layer — become input features to
the learned controller and can therefore drive behaviour through
the learned attention weights. The **W44-C-LIVE-LATENT** carry-
forward is *bounded* (not closed) at the capsule layer by W45:
recovery is via fitted projections rather than substrate access.

A channel's W45 verdict fails if (a) its weight in the trained
controller is not measurable (always zero across all signatures)
or (b) the corresponding R-92 family does not register the
predicted gain. The hyperbolic and euclidean channels are
specifically required to register a non-zero learned attention
weight on at least one signature in the bench.

## Scope (do-not-overstate)

W45 does NOT:

* claim transformer-internal access. The learned controller
  operates strictly over W43 capsule-layer channel encodings; it
  does not read hidden states, transplant KV cache, inspect
  attention weights, or modify the model's attention computation.
  The W43 conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) carry forward unchanged.

* claim the LoRA-style adapter decomposition is true LoRA on a
  transformer. The W45 "adapter" is a low-rank perturbation of the
  capsule-layer policy parameters, not a low-rank perturbation of
  the model's weight matrices. It is a *capsule-layer
  approximation* of the same algebraic move.

* claim the attention routing is true transformer attention. The
  W45 attention is a softmax pool over six scalar channel features;
  the model's attention computation is unchanged.

* claim the learned hint guarantees a real-LLM behavioural change.
  The hint is a deterministic, content-addressed text fragment
  the model may attend to or ignore. The H6 hint-response gain is
  measured on a deterministic `HintAwareSyntheticBackend`; on
  real LLMs the saving is bounded to "the model has the hint
  available in its context" — a strict superset of the W44 prompt
  surface, but not a guarantee of behavioural lift.

* claim the controller is "trained on real LLM traces." Fitting
  is closed-form ridge / Newton on synthetic banks pre-committed
  in the R-92 sources. A future milestone may relax this; the W45
  bar is measured on the hermetic synthetic bench.

* claim multi-host learned coupling. The W45 surface is a
  same-process orchestrator over the released `AgentTeam`
  contract; cross-host learning requires substrate beyond the
  capsule layer.

## Strong / partial / failure

* **Strong success**: H1..H12 all pass; the four active learned
  channels each register their predicted gain or saving with
  min == max across 5/5 seeds; the previously-audit-only channels
  (hyperbolic, euclidean) register non-zero learned attention
  weight on at least one signature; the released SDK contract is
  preserved.

* **Partial success**: H1 + H10 + H11 + H12 pass plus at least
  three of H2..H7 pass; the rest are honestly downgraded with a
  per-channel retraction.

* **Failure**: H1 or H10 or H11 or H12 regresses; the learned
  mechanism is rolled back behind the trivial registry while the
  W43/W44 layers remain green.

## Stable boundary preservation (PART K)

The W45 surface ships at `coordpy.learned_manifold` and
`coordpy.r92_benchmark` and is reachable only through an explicit
import. The first-run UX (`coordpy-team run --preset quant_desk
...`) is unaffected. The W43 surface
(`coordpy.product_manifold`, `coordpy.r90_benchmark`) and the
W44 surface (`coordpy.live_manifold`, `coordpy.r91_benchmark`)
are unchanged. The released `AgentTeam.run` and
`LiveManifoldTeam.run` paths are byte-for-byte unchanged —
`LearnedManifoldTeam` is a new class that sits beside them. A
future milestone may promote a stable subset of the W45 surface
under `coordpy.__experimental__` once cross-host learned evidence
is acquired.

## Why W45 is the strongest honest next step

W44 made *gating* live. The next honest move is to let the gates
themselves be **shaped by data** — but only as far as the capsule
layer permits. The learned controller is the smallest move that
is genuinely deep-learning-shaped (closed-form fittable
projections, attention-style routing, adapter decomposition,
margin calibration) while staying inside the audit boundary. We
do not claim more than that. The deeper architectural directions
(transformer-internal mixed-curvature attention, KV pooling,
continuous Grassmannian homotopy, real shared-state across hosts)
remain explicitly substrate-blocked and out of scope.
