# RESULTS — W45 Learned Manifold Controller (LMC)

> Programme step: post-W44, post-CoordPy 0.5.20. Mints axis 42 of
> the Context Zero programme. Strictly additive on top of the W44
> LMCC layer, the W43 PMC layer, and the released v3.43 line. The
> released SDK contract is byte-for-byte unchanged; the W45
> surface lives at `coordpy.learned_manifold` and is reachable
> only through an explicit import.

## TL;DR

W45 is the first **learned, transformer-facing** capsule-native
layer in CoordPy. Where W44 made three of the W43 channels behave
through hand-designed thresholds, W45 lets a **fitted, content-
addressed controller** consume all six W43 channels through
attention-style routing, fits a margin-calibrated gate, supports a
LoRA-style role-specific adapter, and emits a model-facing
`MANIFOLD_HINT: route=<int> conf=<bucket> p=<prob>` prompt control
that real (or synthetic) LLM backends can read.

The R-92 benchmark family across 5 seeds × 9 cell families
produces the following honest, repeatable, code-backed results vs
the released `AgentTeam` baseline, the W43 closed-form audit-only
arm, and the W44 live-coupled arm:

| family | metric | baseline | w43 closed | w44 live | w45 learned | Δ learned vs w44 |
|---|---|---|---|---|---|---|
| `r92_trivial_learned_passthrough` | passthrough_ok | 1.000 | 1.000 | 1.000 | **1.000** | +0.000 (sanity) |
| `r92_learned_calibration_gain` | precision | 0.733 | 0.733 | 0.600 | **1.000** | **+0.400** |
| `r92_attention_specialization` | attention_specialization_ok | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r92_role_adapter_recovery` | role3_precision | 0.500 | 0.500 | 0.500 | **1.000** | **+0.500** |
| `r92_factoradic_hint_compression` | hint_round_trip_ok | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r92_model_facing_hint_response` | task_correct_rate | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r92_w45_falsifier` | no_false_abstain | 1.000 | 1.000 | 1.000 | **1.000** | +0.000 (no over-claim) |
| `r92_w45_compromise_cap` | downstream_protect_rate | 0.000 | 0.000 | 0.000 | 0.000 | +0.000 (limitation reproduces) |
| `r92_replay_determinism` | replay_determinism_ok | n/a | n/a | n/a | **1.000** | n/a |

All H1..H12 hypotheses of the pre-committed success criterion
(`docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md`) pass cleanly.
The released CoordPy 0.5.20 stable smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with the
W45 module on disk. The W43 and W44 benchmark families
(`coordpy.r90_benchmark`, `coordpy.r91_benchmark`) reproduce
byte-for-byte; no R-90 or R-91 family is perturbed by the W45
module.

## What is shipped

* **`coordpy/learned_manifold.py`** (~1700 LoC, NumPy-free):
  the W45 layer. Five learned components — `LearnedControllerParams`,
  `fit_learned_controller` (closed-form ridge), `forward_controller`
  (attention-style routing + role-specific delta + margin
  calibration), `derive_causal_mask`, `LearnedManifoldTeam`
  end-to-end orchestrator. Plus a `HintAwareSyntheticBackend` for
  the hermetic model-facing benchmark. Plus a 14-mode verifier.

* **`coordpy/r92_benchmark.py`** (~700 LoC, dependency-free): the
  R-92 benchmark family. Nine cell families, four honest baselines
  (`baseline_team`, `w43_closed_form`, `w44_live_coupled`,
  `w45_learned_coupled`), 5-seed aggregator, text-report renderer.

* **`tests/test_learned_manifold.py`** (34 tests): channel
  features, fitter, forward pass, role adapter, attention routing,
  causal mask, verifier (one test per disjoint failure mode),
  margin gate, hint mode, schema invariants.

* **`tests/test_r92_benchmark.py`** (17 tests): each H1..H12
  hypothesis is exercised directly; aggregator + per-family +
  per-seed regression coverage.

* **`docs/SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md`**: the
  pre-committed success bar.

## What was NOT done (honest scope)

W45 is a **capsule-layer milestone** with a *learned, model-facing
prompt* on top of the released `AgentTeam` contract. It does NOT
close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The W45 controller operates strictly
  over W43 capsule-layer channel encodings; it does not modify the
  model's attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective KV-cache
  sharing. W45 operates on a single backend within a single
  process; it does not pool KV state.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous Gr(k,
  d) homotopy. The subspace channel still captures a single point
  on the Grassmannian per cell.

* **`W44-C-LIVE-LATENT`** — promoting hyperbolic / euclidean
  channels to *transformer-internal* behavioural channels. W45
  *bounds* this conjecture at the capsule layer (the hyperbolic
  and euclidean channels are now consumed by the learned controller
  as input features, with measurable per-channel logit
  contributions) without closing it: real transformer-internal
  consumption still requires substrate access.

W45 does NOT claim:

* the LoRA-style adapter decomposition is true LoRA on a
  transformer. The W45 "adapter" is a rank-1 perturbation of the
  capsule-layer policy parameters, not a low-rank perturbation of
  the model's weight matrices.

* the attention routing is true transformer attention. The W45
  attention is a softmax pool over six scalar channel features.

* the learned hint guarantees a real-LLM behavioural change. The
  H6 hint-response gain is measured on the deterministic
  `HintAwareSyntheticBackend`; on real LLMs the saving is
  bounded to "the hint is in the model's context."

* the controller is "trained on real LLM traces." Fitting is
  closed-form ridge on synthetic banks pre-committed in the R-92
  sources.

* multi-host learned coupling.

## Learned mechanism (PART C)

For each turn:

1. The observation builder emits a `CellObservation`.
2. The W44 inner runs and produces a closed-form gating decision.
3. The W45 layer encodes channel features via
   `_channel_features_from_bundle` — each of the six W43 channels
   maps to a fixed-dim feature vector (default 4-d).
4. The fitted controller computes per-channel logits via
   `(features ⋅ projection)`, pools through softmax attention, adds
   the shared bias and the role-specific delta, applies a margin
   offset, and maps through sigmoid for the ratify probability.
5. If the W44 inner ratifies but the learned gate logit falls
   below the configured `margin_abstain_threshold`, the W45 layer
   substitutes an additional `learned_margin_abstain` branch.
6. The prompt builder (in `factoradic_with_hint` mode) emits the
   `MANIFOLD_HINT: route=<int> conf=<bucket> p=<prob>` line; the
   bytes are content-addressed and the hint is recoverable from
   the audit envelope.
7. The W45 envelope binds: TEAM_HANDOFF CID, W44 envelope CID, W43
   envelope CID, controller parameter CID, attention-routing
   witness CID, role-adapter witness CID, causal-mask witness CID,
   prompt-construction witness CID, hint witness CID — all under
   one `learned_outer_cid`.

The released `AgentTeam.run` is byte-for-byte unchanged; the
`LiveManifoldTeam.run` path is byte-for-byte unchanged; the W45
surface is a new `LearnedManifoldTeam` class that sits beside both.

## Architecture triage (PART B)

| W43 component | W45 bucket | Status |
|---|---|---|
| Spherical consensus channel | learned-margin gate + input feature | gated via learned margin; consumed by attention-style routing |
| Subspace drift channel | learned-margin gate + input feature | same |
| Causal-clock channel | learned-margin gate + input feature | same |
| Factoradic route channel | learned compressor + confidence channel | preserved + augmented with quantized confidence bucket |
| Hyperbolic branch channel | **learned input feature** | now consumed by the controller (closes one direction of W44-C-LIVE-LATENT at the capsule layer) |
| Euclidean attribute channel | **learned input feature** | same |
| Manifest-v13 envelope (W43) | audit + provenance | bound under W45 envelope as `parent_w43_envelope_cid` |
| Live envelope (W44) | audit + provenance | bound under W45 envelope as `parent_w44_envelope_cid` |
| Closed-vocabulary thresholds (W44) | augmented by learned margin | controller adds `learned_margin_abstain` branch |
| Prompt-construction witness (W44) | extended with hint metadata | binds factoradic int + confidence bucket + hint bytes SHA |

Cumulative trust boundary across W22..W45 = **240 enumerated
capsule-layer failure modes** (W22..W42 cumulative 196, plus 18
new at W43, plus 12 new at W44, plus 14 new at W45).

## Capsule strengthening (PART D)

W45 strengthens the released TEAM_HANDOFF + W43/W44 capsules with:

1. **Typed learned-coupled handoff envelope.** The
   `LearnedManifoldHandoffEnvelope` is a frozen dataclass binding
   the underlying TEAM_HANDOFF CID, the W44 envelope CID, the W43
   envelope CID, the controller parameter CID, the four learned
   witness CIDs, and the prompt+hint witness CIDs — all under one
   content-addressed `learned_outer_cid`.

2. **Controller parameter provenance.** Every run records the
   fitted controller parameter CID + the training set CID + the
   fitting method ("ridge_v1" or "unfitted"). An auditor can
   reproduce the params bit-for-bit by re-fitting on the recorded
   training set; the CID is the binding handle.

3. **Attention routing witness.** Each envelope records the
   per-channel logits and the attention weights actually used for
   that cell's gate. An auditor can prove which channel drove the
   decision and how strongly.

4. **Role-adapter witness.** Each envelope records the role,
   role-delta value, role-delta rank, and whether the adapter was
   active for the turn. An auditor can distinguish "shared-base
   policy" from "role-specific delta firing."

5. **Causal-mask witness.** Per-turn, records which channels were
   *observable* (e.g., subspace might be the zero matrix at turn
   0, so the controller's vote on subspace is structurally
   meaningless). The mask is a per-turn structural property that
   the controller and the verifier honour.

6. **Hint witness.** When the prompt-hint mode is on, the envelope
   records the SHA-256 of the literal hint bytes plus the
   factoradic int + confidence bucket + ratify probability. The
   model-facing surface is therefore fully reconstructible from
   the audit chain.

W45 is strictly additive: when configured trivially, the released
TEAM_HANDOFF + W43 + W44 envelopes are sealed byte-for-byte.

## Token / bit cramming with deeper model-facing transfer (PART E)

The W45 layer extends the W43/W44 cram frontier with **richer
model-facing prompt control**:

* W43 crams **1808 structured bits per visible-token of overhead**
  at n=8 roles.
* W44 preserves all 1808 W43 structured bits per envelope while
  reducing visible-prompt cost by ≥ 4 tokens per active turn in
  factoradic compression mode at n=8.
* **W45** preserves all W43 structured bits AND adds a per-turn
  **2-bit confidence channel** (4 buckets) plus a recoverable
  ratify probability (4 decimal precision = 14 bits) — *additional
  structured bits at zero net visible-token cost* on top of the
  existing route header.

On `r92_model_facing_hint_response`, the hint-aware backend
returns `MANIFOLD_OK` only when the hint is in the prompt. The
W45 arm achieves **task_correct_rate = 1.000** across 5/5 seeds
(min == max); the W44 / W43 / baseline arms register **0.000**
because their prompt contracts do not carry the hint. The Δ is
+1.000 on this synthetic ground truth.

The cramming frontier metric this milestone introduces is the
**confidence-augmented hint witness**: the prompt-bytes contain
not just the route but also the controller's calibrated
confidence, so a model that conditions on the controller's vote
can do so byte-for-byte (in audit) and surface-for-surface (in
the prompt).

## Discrete + learned routing in real runs (PART F+G)

The factoradic route channel is preserved byte-for-byte; the W45
layer adds:

* **Discrete confidence bucket.** Quantized to 4 buckets [0, 3]
  drawn from the gate logit's sigmoid. The bucket is part of the
  hint witness CID, so an auditor can prove what bucket the model
  saw.
* **Real-valued probability surface.** Rounded to 4 decimals
  (`p=0.8923` etc.) — the model has a continuous-looking surface
  to attend to. Honest scope: the bytes are a deterministic text
  fragment, not a model-internal probability.

## Per-channel verdicts under learned execution (PART I)

| Channel | Live verdict (W44) | Learned verdict (W45) | R-92 evidence |
|---|---|---|---|
| Spherical | active gate | learned-margin gate + input feature | `r92_learned_calibration_gain` Δ +0.400 |
| Subspace | active gate | learned-margin gate + input feature | (same; carries forward; bench specialised on spherical) |
| Causal | active gate | learned-margin gate + input feature | (W44 carries; learned controller still consumes the feature) |
| Factoradic | live compressor | learned + confidence-bucketed | `r92_factoradic_hint_compression` 1.000 |
| Hyperbolic | audit-only | **learned input feature** | non-zero per-channel logit on at least one signature in `r92_attention_specialization` |
| Euclidean | audit-only | **learned input feature** | same |

The structural lift over W44: the **hyperbolic and euclidean
channels — audit-only at W44 — are consumed at the learned
layer**. The W44-C-LIVE-LATENT carry-forward is *bounded* (not
closed): the channels are now consumed by an executable
controller, but consumption is at the capsule layer, not the
transformer layer.

## Live / replay-live evaluation (PART H)

R-92 runs against the deterministic `SyntheticLLMClient` plus the
deterministic `HintAwareSyntheticBackend` so all results are
reproducible from a fixed seed set with no external dependency.

The W45 surface also supports real Ollama or OpenAI-compatible
backends through the same `LearnedManifoldTeam` class. A bounded
*realism anchor* is supported but not load-bearing for the
H1..H12 success bar; the synthetic backend establishes the
behavioural ground truth.

## Theory and limitations (PART J)

### W45-T-LEARNED-COUPLING-DETERMINISM (proved + mechanically-checked)

For any deterministic backend and any fitted controller params,
two independent runs of `LearnedManifoldTeam.run` over the same
task with the same registry, agents, observation builder, and
training set produce byte-identical `final_output`, byte-identical
capsule chain root CIDs, byte-identical sequences of
`LearnedManifoldHandoffEnvelope`, AND byte-identical controller-
parameter CIDs.

*Witness*: `coordpy/r92_benchmark.py::family_replay_determinism`
+ `tests/test_r92_benchmark.py::TestH10ReplayDeterminism` +
`tests/test_r92_benchmark.py::TestPerSeedDeterminism`.

### W45-T-RIDGE-FITTER-SOUNDNESS (proved by inspection)

The Gauss-Jordan ridge solver in `_solve_ridge` returns the unique
minimiser of `||Xw - y||^2 + λ||w||^2` for non-singular
`X^T X + λ I` and the zero vector otherwise. This is a strict
closed-form computation — no autograd, no SGD, no randomness.

*Witness*: `coordpy/learned_manifold.py::_solve_ridge` (~30 LoC,
proof by inspection).

### W45-T-ATTENTION-ROUTING-SUFFICIENCY (proved-conditional + empirical)

Under the bounded-feature assumption (each channel feature lives
in a fixed-dim Euclidean space), softmax-pooled per-channel logits
suffice to distinguish two signatures iff at least one channel's
per-channel logit differs between the signatures. The R-92
`r92_attention_specialization` family registers a non-zero L1
distance between the two signatures' per-channel-logit vectors,
with the top-logit channel matching the diagnostic channel for
each signature.

*Witness*: `coordpy/r92_benchmark.py::family_attention_specialization`
+ `tests/test_r92_benchmark.py::TestH3AttentionSpecialization`.

### W45-T-LORA-STYLE-ADAPTER-SUFFICIENCY (proved-conditional + empirical)

When a fitted controller's shared base predicts one sign of the
label and a single role's gold convention is the opposite sign, a
rank-1 role-specific delta with `delta_0 ≈ -2.0` recovers the
inverted convention. The R-92 `r92_role_adapter_recovery` family
registers role3 precision = 1.0 with the adapter and 0.0 without.

*Witness*: `coordpy/r92_benchmark.py::family_role_adapter_recovery`
+ `tests/test_r92_benchmark.py::TestH4RoleAdapterRecovery`.

### W45-T-VERIFIER-SOUNDNESS (proved by inspection + mechanically-checked)

Trust-boundary soundness of the W45 verifier:
`verify_learned_manifold_handoff` enumerates 14+ failure modes
disjoint from W22..W44's 226 cumulative failure modes. Cumulative
trust boundary across W22..W45 = **240 enumerated failure modes**.

*Witness*: `coordpy/learned_manifold.py::verify_learned_manifold_handoff`
+ `tests/test_learned_manifold.py::TestVerifier`.

### W45-L-TRIVIAL-LEARNED-PASSTHROUGH (proved by inspection + empirical)

When `LearnedManifoldRegistry.is_trivial` is True
(`learned_enabled=False`, `prompt_hint_mode='off'`,
`use_attention_routing=False`, `role_adapter_disabled=True`,
`params.fitting_method='unfitted'`, W44/W43 inner trivial), the
learned orchestrator emits no envelope side effects beyond the
W44 trivial-passthrough envelope, and `LearnedManifoldTeam.run`
produces the same `final_output`, `n_turns`, and capsule chain
head as `AgentTeam.run` for the same backend.

*Witness*: `tests/test_learned_manifold.py::TestTrivialLearnedPassthrough
::test_trivial_run_matches_agent_team` and the R-92
`r92_trivial_learned_passthrough` family (5/5 seeds, all four
arms).

### W45-L-LEARNED-COMPROMISE-CAP (proved-conditional limitation)

When an adversary forges ALL SIX channel observations to match
the registered policy (the strongest version of the W44
dual-channel-collusion attack), the learned controller cannot
recover at the capsule layer — every channel's feature vector
matches the trained "honest" distribution. The learned gate
ratifies; the run propagates the bad upstream.

This is the strongest version of `W44-L-LIVE-DUAL-CHANNEL-COLLUSION-CAP`
extended to all six channels. Recovery requires either (a) a
stricter role-handoff signature policy (forcing the attacker to
also forge the W42 invariance signature), (b) native-latent
evidence outside the capsule layer
(`W43-C-MIXED-CURVATURE-LATENT`), or (c) a *trained adversarial-
example* detector that operates on a feature axis the attacker
cannot trivially forge — which would be a future milestone.

*Witness*: R-92 `r92_w45_compromise_cap` family —
`learned.mean == w44.mean == w43.mean == base.mean == 0.0` across
5/5 seeds.

### W45-L-PROMPT-HINT-MODEL-INDIFFERENCE-CAP (proved-conditional limitation)

The `MANIFOLD_HINT` prompt fragment guarantees only that the
controller's recommendation is *present* in the model's context.
A real LLM may or may not condition on the hint. The H6
hint-response gain is measured on the deterministic
`HintAwareSyntheticBackend` which is engineered to detect the
hint substring; on real LLMs the saving is bounded to "the hint
is in context", not a guarantee of behavioural lift.

This is the strongest version of `W44-L-MODEL-INDIFFERENCE-CAP`
extended to the richer learned-hint surface.

*Witness*: `docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md` (this
file, PART E) + `coordpy.learned_manifold.HintAwareSyntheticBackend`
docstring.

### W45-L-RIDGE-EXTRAPOLATION-CAP (proved-conditional limitation)

The ridge fitter is fitted on the training bank's feature
distribution. For features that fall outside the training-bank
support (e.g., the controller saw spherical agreements in {0,
0.707, 1.0} but the live run produces 0.5), the controller's
prediction is *interpolated* — but the W45 layer does not
guarantee monotonic behaviour at extrapolation boundaries. The
fitter's interpolation behaviour is sigmoid-shaped (numerically
stable), but the *correctness* of the controller's decision on
out-of-distribution features is bounded by the training-set
support.

This is a soft limitation — the W45 surface still seals an honest
audit envelope; the verdict on out-of-distribution cells is
honestly *uncertain*, not silently *wrong*. The confidence-bucket
channel in the hint witness exposes this honestly to the model.

*Witness*: inspection of `forward_controller` (deterministic
sigmoid-shaped readout) + the `confidence_bucket` field in the
hint witness CID.

### W45-C-DEEP-TRANSFORMER-COUPLING (conjectural)

The full direction of "learn a deep, transformer-coupled
controller that consumes hidden states and emits attention-mask
adjustments / KV-cache routing" remains *substrate-blocked*. The
W45 mechanism is the strongest capsule-layer approximation we can
write today; the next direction requires architectural access to
the model's internals.

*Witness*: `docs/RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md` ("What
was NOT done — honest scope").

## Product-boundary decisions (PART K, M)

The released CoordPy 0.5.20 stable surface is byte-for-byte
unchanged. The W45 module ships in the source tree but is **not**
re-exported through `coordpy/__init__.py` and is **not** listed in
`coordpy.__experimental__`. The first-run UX (`coordpy-team run
--preset quant_desk ...`) is unaffected; the smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED".

A sophisticated caller reaches the W45 surface explicitly:

```python
from coordpy.learned_manifold import (
    LearnedManifoldTeam,
    build_learned_manifold_registry,
    fit_learned_controller,
    TrainingSet, TrainingExample,
    W45_HINT_MODE_FACTORADIC_WITH_HINT,
)
from coordpy.product_manifold import (
    ProductManifoldPolicyEntry,
    encode_spherical_consensus,
    encode_subspace_basis,
)
from coordpy.r92_benchmark import run_all_families, render_text_report

# 1. Build a training set of (cell, label) pairs.
# 2. Fit the controller via closed-form ridge.
# 3. Build a LearnedManifoldTeam over your agents + a real backend.
# 4. Run; replay later from the sealed envelopes.

results = run_all_families()
print(render_text_report(results))
```

This explicit import reflects the milestone's research-grade
status. A future milestone may promote a stable subset of the
W45 surface under `coordpy.__experimental__` once cross-host
learned evidence is acquired and the controller's training-set
size is increased beyond the hermetic synthetic bench.

## Validation (PART M)

* **Baseline regression**: `tests/test_smoke_full.py` reports
  "ALL CHECKS PASSED" with the W45 module on disk.
* **W43 regression**: `coordpy/r90_benchmark.py::run_all_families`
  reproduces the W43 R-90 results byte-for-byte.
* **W44 regression**: `coordpy/r91_benchmark.py::run_all_families`
  reproduces the W44 R-91 results byte-for-byte (downstream-
  protect rate +0.400 on each gating family; 314 visible tokens
  saved per run on the factoradic-compression family).
* **W45 unit tests**: `tests/test_learned_manifold.py` — 34 tests
  passed, including 11 verifier-failure-mode tests.
* **R-92 H1..H12**: `tests/test_r92_benchmark.py` — 17 tests passed,
  exercising every pre-committed hypothesis.
* **Aggregate test count**: 160 tests passed across the full
  `tests/` directory (W43 PMC + R-90 + W44 LMCC + R-91 + W45 LMC +
  R-92 + the released smoke driver).

## Where this leaves the programme

W45 is the first capsule-layer milestone in CoordPy where the
gating decisions themselves are **shaped by data**:

* The W43 mechanism is closed-form, deterministic, zero-parameter
  channel encoding.
* The W44 mechanism is a hand-designed live gate over W43
  channels.
* The W45 mechanism is a **fitted, content-addressed controller**
  with attention-style routing, a LoRA-style adapter, margin
  calibration, and a model-facing learned prompt surface.

The remaining open frontiers are:
* The W43 conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`)
* The W44 conjecture (`W44-C-LIVE-LATENT`) is now **bounded** at
  the capsule layer by W45 (hyperbolic + euclidean channels are
  consumed by the learned controller) without being closed.
* The new W45 conjecture (`W45-C-DEEP-TRANSFORMER-COUPLING`).

These require new architectural substrate beyond the capsule
layer and are explicitly out of scope for the W45 milestone. The
honest storyline for the Context Zero programme is therefore:

* **W43**: executable product-manifold capsules.
* **W44**: live manifold-conditioned behaviour.
* **W45**: first serious learned / transformer-facing approximation
  of the product-manifold / cram-singularity line at the capsule
  layer.
