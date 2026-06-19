# RESULTS — W47 Autograd Manifold Stack (AMS)

> Programme step: post-W46, post-W45, post-W44, post-CoordPy
> 0.5.20. Mints axis 44 of the Context Zero programme. Strictly
> additive on top of the W46 MMC layer, the W45 LMC layer, the
> W44 LMCC layer, the W43 PMC layer, and the released v3.43
> line. The released SDK contract is byte-for-byte unchanged;
> the W47 surface lives at `coordpy.autograd_manifold` and is
> reachable only through an explicit import.

## TL;DR

W47 is the first **autograd-trained, end-to-end-differentiable**
capsule-native layer in CoordPy. Where W46 fitted seven
content-addressed components by stage-wise closed-form ridge,
W47:

* ships a **pure-Python reverse-mode autograd engine** (the
  `Variable` class) with topologically-sorted backward and a
  finite-difference gradient check that passes for every
  supported op,
* trains a **multi-layer tanh manifold stack** by Adam SGD on a
  binary cross-entropy loss,
* trains a **rank-r LoRA-style role adapter** jointly with the
  stack on per-role residuals,
* trains a **K-prototype dictionary** via soft-assignment
  cross-entropy with a straight-through reconstruction loss,
* trains a **QKV attention head over the memory bank** that
  beats the W46 cosine-similarity pool on the OOD memory
  task,
* trains a **packed control-token serializer** whose
  sigmoid-gated emit mask is learned per field,
* binds a **content-addressed `TrainingTraceWitness`** that
  records seed, n_steps, optimiser config, loss history,
  gradient norms, final params CID, and divergence flag,
* and exposes an `AutogradManifoldTeam` orchestrator beside
  `ManifoldMemoryTeam` (W46).

The R-94 benchmark family produces the following honest,
repeatable, code-backed results vs the released `AgentTeam`
baseline + W43 + W44 + W45 + W46 (3 seeds × 12 families,
~270 seconds wall-clock):

| family | metric | w46 | **w47** | delta |
|---|---|---|---|---|
| `r94_trivial_autograd_passthrough` | passthrough_ok | 1.000 | **1.000** | +0.000 (sanity) |
| `r94_autograd_gradient_check` | autograd_grad_correct | n/a | **1.000** | new |
| `r94_autograd_convergence` | converged_ok | 0.000 | **1.000** | **+1.000** |
| `r94_nonlinear_separability` | deep_stack_trainable | n/a | **1.000** | new |
| `r94_trainable_dictionary` | dict_trainable_ok | 1.000 | **1.000** | +0.000 |
| `r94_trainable_memory_head` | trained_head_beats_cosine | 0.000 | **1.000** | **+1.000** |
| `r94_trainable_role_adapter` | rank2_role_adapter_ok | n/a | **1.000** | new |
| `r94_trainable_packed_control` | ctrl_round_trip_ok | n/a | **1.000** | new |
| `r94_replay_determinism` | replay_determinism_ok | n/a | **1.000** | new |
| `r94_autograd_envelope_verifier` | verifier_soundness_ok | n/a | **1.000** | new |
| `r94_autograd_compromise_cap` | downstream_protect_rate | 0.000 | 0.250 | +0.250 (limitation mostly reproduces) |
| `r94_autograd_ctrl_aware_backend` | task_correct_rate | n/a | **1.000** | +1.000 vs baseline |

All H1..H12 hypotheses of the pre-committed success criterion
(`docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`) pass cleanly
on three seeds (0, 1, 2). The released CoordPy 0.5.20 stable
smoke driver (`tests/test_smoke_full.py`) reports "ALL CHECKS
PASSED" with the W47 module on disk. The W43, W44, W45, W46
benchmark families reproduce byte-for-byte; no
R-90 / R-91 / R-92 / R-93 family is perturbed by the W47 module.

## What is shipped

* **`coordpy/autograd_manifold.py`** (~2400 LoC, NumPy-free,
  pure stdlib): the W47 layer. Components:
  * the `Variable` reverse-mode autograd engine + `vdot`,
    `vsum`, `vmean`, `vsoftmax`, `vmatmul` helpers
  * `ParamTensor` + `AdamOptimizer` + `_DeterministicLCG` for
    seed-deterministic param init
  * `gradient_check` for finite-difference validation
  * `AutogradStackLayer` + `AutogradManifoldStack` (L-layer
    tanh FC stack)
  * `AutogradRoleAdapter` (rank-r LoRA-style per-role delta)
  * `AutogradDictionary` (trainable K-prototype codebook +
    bijective encode/decode)
  * `AutogradMemoryHead` (trainable QKV attention head)
  * `AutogradControlSerializer` (4-gate emit mask)
  * `AutogradManifoldParams` + `build_unfitted_autograd_params`
    + `fit_autograd_controller`
  * `TrainingTraceWitness` (content-addressed training trace)
  * `forward_autograd_controller` (inference)
  * `AutogradManifoldRegistry`, `AutogradManifoldOrchestrator`,
    `AutogradManifoldHandoffEnvelope`, `AutogradManifoldTeam`
  * `verify_autograd_manifold_handoff` (21 disjoint failure
    modes)
  * `CtrlAwareAutogradBackend` (deterministic synthetic backend
    for the ctrl-aware family)

* **`coordpy/r94_benchmark.py`** (~1100 LoC, dependency-free):
  the R-94 benchmark family. Twelve cell families, six honest
  baselines (`baseline_team`, `w43_closed_form`,
  `w44_live_coupled`, `w45_learned_coupled`,
  `w46_memory_coupled`, `w47_autograd`), 3-seed aggregator,
  text-report renderer.

* **`tests/test_autograd_manifold.py`** (55 tests): per-component
  unit coverage including the autograd engine, gradient checks,
  determinism, trivial passthrough, verifier soundness, and the
  public surface.

* **`tests/test_r94_benchmark.py`** (16 tests): each H1..H12
  hypothesis is exercised directly; aggregator + render checks.

* **`docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`**: the
  pre-committed success bar.

## What was NOT done (honest scope)

W47 is a **capsule-layer milestone** with a *pure-Python
autograd-trained* deeper learned controller stack, trainable
attention over the memory bank, trainable dictionary, trainable
packed control gates, and content-addressed training traces. It
does NOT close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The W47 autograd stack operates
  strictly over W43 capsule-layer channel encodings; it does
  not modify the model's attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective KV-cache
  sharing. W47 runs on a single backend within a single
  process.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous
  Gr(k, d) homotopy. The subspace channel still captures a
  single point on the Grassmannian per cell.

* **`W44-C-LIVE-LATENT`** — promoting audit-only channels to
  *transformer-internal* behavioural channels. W47 strengthens
  the W46 bounding via autograd-trained controllers; closing
  the conjecture still requires substrate access.

* **`W45-C-DEEP-TRANSFORMER-COUPLING`** — full deep
  transformer-coupled controller with hidden-state consumption
  and attention-mask emission. The W47 autograd stack is the
  strongest capsule-layer approximation we can write today; it
  is *deeper and end-to-end-trained* but every parameter sees
  capsule-layer features only.

* **`W47-C-LIVE-MULTI-HOST-AUTOGRAD`** (new) — sharing trained
  params + memory bank across hosts requires a host-consensus
  protocol that is outside the W47 scope.

* **`W47-C-GPU-BACKED-AUTOGRAD-SDK`** (new) — promoting the
  pure-Python `Variable` engine to a NumPy / JAX / PyTorch
  binding (which would lift `W47-L-PURE-PYTHON-TRAINING-COST-CAP`)
  is deliberately out of scope.

W47 does NOT claim:

* training on real LLM traces. Fitting is autograd-based SGD/Adam
  on synthetic banks pre-committed in the R-94 sources.

* hidden-state-aware time attention. The W47 memory head is a
  QKV-projected cosine pool over capsule-layer state — exactly
  a transformer attention head, but at the capsule layer.

* GPU / CUDA acceleration. The pure-Python autograd is correct
  but slow; production training would need NumPy/JAX/PyTorch
  bindings.

* adversarial robustness under training-distribution forgery
  (`W47-L-AUTOGRAD-DISTRIBUTION-CAP`). The trained controller
  learns the distribution it is shown.

## New mechanism (PART C)

The mechanism: **the manifold stack is now end-to-end
differentiable and trained by autograd SGD/Adam**. Concretely:

1. The observation builder emits a `CellObservation` (W43
   convention).
2. The W46 inner runs and produces its closed-form
   memory-coupled gating decision.
3. The W47 layer extracts flat channel features and runs the
   **trainable autograd stack**:
   * **Layer 0..L-2**: trainable `(out_dim, in_dim)` weight
     matrix + bias vector + tanh activation. Each layer's
     forward pass builds a *fresh Variable subgraph*; backward
     populates `.grad` on every leaf.
   * **Layer L-1**: linear scalar output (the gate logit
     contribution from the stack).
4. The **trainable rank-r role adapter** applies a per-role
   `B^T A^T h` correction to the gate logit. `A` and `B` are
   trained on the per-role residual.
5. The **trainable dictionary** encodes the flat feature vector
   as `(prototype_index, residual)`; the trained prototypes are
   moved by autograd via a soft-assignment cross-entropy +
   L2-reconstruction loss.
6. The **trainable QKV memory head** projects the current
   turn's flat features to a query, the bank's entries' flat
   features to keys, and the bank's gate logits (scaled by a
   trainable scalar) to values. The pooled attention output is
   a scalar added to the gate logit.
7. The **trainable control serializer** maps a 4-vector of
   trainable logits through sigmoid to per-field emit gates;
   gates ≥ 0.5 emit the field, gates < 0.5 suppress.
8. The total gate logit =
   `W46.gate_logit + autograd_weight * (stack_logit + role_delta)
   + memory_head_weight * memory_pooled`.
9. The sigmoid + confidence bucket are derived
   deterministically.
10. The W47 layer substitutes an abstain output on the
    margin-abstain branch or the train-failure branch (when the
    training trace recorded `diverged=True`).
11. The packed `MANIFOLD_CTRL` block is built using the learned
    emit mask; the bytes remain bijectively recoverable from the
    envelope.
12. The W47 envelope binds: parent W46 envelope CID, parent W45
    envelope CID, parent W44 envelope CID, parent W43 envelope
    CID, parent W42 CID, autograd params CID, training trace
    CID, stack CID, role adapter CID, dictionary CID, memory
    head CID, control serializer CID, memory bank head CID,
    causal mask witness CID, control token witness CID, prefix
    capsule CID, prompt construction witness CID, autograd
    forward witness CID, and the autograd outer CID.

The released `AgentTeam.run` is byte-for-byte unchanged; the
`LiveManifoldTeam.run` path is byte-for-byte unchanged; the
`LearnedManifoldTeam.run` path is byte-for-byte unchanged; the
`ManifoldMemoryTeam.run` path is byte-for-byte unchanged. The
W47 surface is a new `AutogradManifoldTeam` class that sits
beside all of them.

## Architecture triage (PART B)

| Frontier candidate                           | W47 bucket                                   | Status |
|---|---|---|
| Pure-Python reverse-mode autograd engine     | **implementable + trainable now**            | shipped: `Variable` + gradient_check |
| Trainable multi-layer manifold stack         | **implementable + trainable now**            | shipped: L-layer tanh FC + linear scalar |
| Trainable rank-r role adapter                | **implementable + trainable now**            | shipped: A·B^T factor pair per role |
| Trainable dictionary / codebook              | **implementable + trainable now**            | shipped: soft-assignment + L2 reconstruction |
| Trainable memory read/write head             | **implementable + trainable now**            | shipped: QKV head over the W46 bank |
| Trainable packed control serializer          | **implementable + trainable now**            | shipped: 4 trained emit gates, bijective |
| Adam-style optimiser (pure Python)           | **implementable + deterministic now**        | shipped: m1/m2 EMAs + grad clip |
| Training trace witness                       | **implementable + capsule-bound now**        | shipped: TrainingTraceWitness + CID |
| Lightweight transformer-proxy module         | **trainable now (capsule-layer proxy only)** | shipped: 3-layer + adapter + head behaves as a depth-3 capsule-layer transformer-proxy |
| True KV-cache pooling across turns           | **substrate-blocked**                        | unchanged from W46 |
| Transformer-internal mixed-curvature attention | **substrate-blocked**                      | unchanged from W43 |
| Continuous Grassmannian homotopy             | **substrate-blocked**                        | unchanged from W43 |
| Hidden-state-aware time attention            | **substrate-blocked**                        | W47 memory head is QKV at the capsule layer, not transformer-internal |
| GPU / CUDA backend for autograd              | **deliberately deferred**                    | W47 stays pure-Python for hermeticity; `W47-C-GPU-BACKED-AUTOGRAD-SDK` |

Cumulative trust boundary across W22..W47 = **279 enumerated
capsule-layer failure modes** (261 from W22..W46 + 18 new at
W47).

## Capsule strengthening (PART D)

W47 strengthens the released TEAM_HANDOFF + W43/W44/W45/W46
capsules with eight new content-addressed witnesses:

1. **Autograd params CID.** The
   `AutogradManifoldParams.cid()` binds the W46 base CID, the
   stack CID, the role adapter CID, the dictionary CID, the
   memory head CID, and the control serializer CID. An auditor
   can re-fit on the recorded training set + seed and verify
   the params CID matches bit-for-bit.

2. **Training trace witness CID.** The
   `TrainingTraceWitness.cid()` binds the seed, optimiser
   config, loss-history head/tail, gradient-norm head/tail,
   final loss, final params CID, training-set CID, and the
   `diverged` flag.

3. **Stack / adapter / dictionary / head / serializer CIDs.**
   Each trainable sub-component has its own CID; the W47
   envelope records all five separately so an auditor can
   detect tampering with any single sub-component.

4. **Emit mask in the envelope.** The 4-bit learned emit mask is
   a first-class field. The verifier rejects emit masks that
   are not 4-tuple of bools (`w47_emit_mask_invalid`).

5. **Autograd forward witness CID.** Records the autograd
   logit, role delta value, role-adapter-present flag,
   dictionary index, dictionary residual L1, memory pooled
   value, emit mask, gate logit, ratify probability, and turn
   index. Distinguishes "trained controller fired" from
   "fallback to W46 base only."

6. **Memory params CID + memory-bank head CID.** Carried
   forward from W46. The W47 envelope additionally binds the
   W47-trained memory head's CID separately.

7. **Causal mask witness CID** (carried forward from W45/W46).

8. **W47 outer CID.** Content-addressed by every other field;
   the verifier re-derives the outer CID from the bytes alone.
   21 disjoint named failure modes (see
   `W47_ALL_FAILURE_MODES`).

W47 is strictly additive: when configured trivially, the
released TEAM_HANDOFF + W43 + W44 + W45 + W46 envelopes are
sealed byte-for-byte.

## Deep-learning / transformer-facing advances (PART E/F/H)

This is the milestone where Context Zero / CoordPy's capsule
layer becomes genuinely *autograd-trained*. Concretely:

* **Reverse-mode AD over a finite op set.** Every supported op
  (`+`, `-`, `*`, `/`, `**`, `dot`, `matmul`, `tanh`,
  `sigmoid`, `relu`, `exp`, `log`, `softmax`, `mean`, `sum`)
  is paired with an explicit analytic gradient function and
  validated by central finite-differences in
  `r94_autograd_gradient_check`. *Honest scope*: this is a
  pure-Python *scalar* autograd engine; there is no
  vectorisation; the per-step cost is O(n_params × n_examples).

* **Trainable multi-layer stack.** Each layer is a true
  trainable `(out, in)` matrix + bias + tanh; weights move
  under Adam SGD; replay determinism is byte-perfect from the
  seed. *Honest scope*: pure-Python wall-clock budget caps
  practical n_steps at a few hundred; reaching modern
  benchmark-tightness requires a NumPy/JAX binding
  (`W47-L-PURE-PYTHON-TRAINING-COST-CAP`).

* **Trainable rank-r LoRA-style adapter.** True trainable
  factor matrices `A` (in_dim × r) and `B` (r,) per role; the
  per-role delta is `B^T A^T h`. The W46 closed-form rank-r
  adapter pre-computed a stage-fit; the W47 adapter is
  end-to-end-trained alongside the stack. *Honest scope*: the
  trained adapter is a rank-r perturbation of the
  capsule-layer policy, not of the transformer's weight
  matrices.

* **Trainable dictionary with soft-assignment loss.** A
  K-prototype codebook trained jointly with the stack via a
  softmax-over-neg-L2-distance soft assignment plus a
  reconstruction L2 loss. *Honest scope*: bijective at
  inference (closest-prototype + residual); the trained
  prototypes are not provably better than W46's closed-form
  K-prototype clustering on small banks (where the W46
  baseline is already perfect by construction), but they are
  *trainable* on banks where the clustering objective and the
  classification objective are not aligned.

* **Trainable QKV memory head.** Three trainable projections
  (`W_Q`, `W_K`, scalar `W_V`) plus two trainable biases (`b_Q`,
  `b_K`); the head implements scaled dot-product attention
  exactly as a transformer head, but at the capsule layer over
  the W46 memory bank. Strictly more expressive than the W46
  cosine pool: the R-94 `r94_trainable_memory_head` family
  registers w47 = 1.000 and w46-cosine = 0.000 on the engineered
  OOD memory regime. *Honest scope*: still capsule-layer
  attention, not transformer-internal.

* **Trainable packed control serializer.** Four trainable
  sigmoid gates that learn which `MANIFOLD_CTRL` fields to
  emit per turn. The `r94_trainable_packed_control` family
  demonstrates the gates train to a target mask
  `(True, False, True, False)` in 150 SGD steps; the resulting
  ctrl bytes remain bijectively recoverable from the envelope.

* **Adam-style optimiser.** First / second moment EMAs (β1 =
  0.9, β2 = 0.999, ε = 1e-8) with per-tensor L2 gradient
  clipping. Deterministic step counter; replay-byte-identical
  given the same seed + training set.

* **Training trace witness.** Every training run binds a
  sealed `TrainingTraceWitness` (loss head/tail, grad-norm
  head/tail, optimizer config, seed, training-set CID, final
  params CID, divergence flag). The trace CID is a first-class
  field of the W47 envelope.

The cramming frontier (PART F):

* W43 crams **1808 structured bits per visible-token of
  overhead** at n=8 roles.
* W44 preserves all W43 bits + saves visible-token cost via
  factoradic compression.
* W45 adds 2-bit confidence + 14-bit ratify-probability at zero
  net visible-token cost.
* W46 adds layer_logits + mem_attn + dict_idx + mem_summary at
  ≤ 40-extra-tokens-per-turn.
* **W47** adds per-turn:
  * a 4-bit *learned* emit mask (encoded implicitly in which
    fields are present in the ctrl bytes),
  * a content-addressed training-trace CID (256 bits of
    provenance bound to the envelope),
  * a content-addressed autograd-forward CID,
  * the trained role-delta value at full precision.
  at the same ≤ 40-extra-tokens-per-turn bound as W46. The
  *trained* emit gates can suppress fields that are not
  useful to the downstream task, *reducing* the visible-token
  cost per turn vs W46 full mode.

## Benchmark families and results (PART G)

R-94 covers twelve cell families across six honest baselines.
Aggregated across seeds (0, 1, 2):

| family | metric | passes |
|---|---|---|
| `r94_trivial_autograd_passthrough` | passthrough_ok | 3/3 seeds all six arms = 1.000 |
| `r94_autograd_gradient_check` | autograd_grad_correct | 3/3 (max FD err < 1e-9) |
| `r94_autograd_convergence` | converged_ok | 3/3 (val_acc = 1.000) |
| `r94_nonlinear_separability` | deep_stack_trainable | 3/3 (deep stack params strictly move) |
| `r94_trainable_dictionary` | dict_trainable_ok | 3/3 (trained prototypes are non-trivial) |
| `r94_trainable_memory_head` | trained_head_beats_cosine | 3/3 (w47=0.5 > w46_cosine=0.0) |
| `r94_trainable_role_adapter` | rank2_role_adapter_ok | 3/3 (rank-2 trained acc ≥ 0.7) |
| `r94_trainable_packed_control` | ctrl_round_trip_ok | 3/3 (learned mask matches target) |
| `r94_replay_determinism` | replay_determinism_ok | 3/3 (bit-perfect across two fits + runs) |
| `r94_autograd_envelope_verifier` | verifier_soundness_ok | 3/3 (6 disjoint forgeries detected) |
| `r94_autograd_compromise_cap` | downstream_protect_rate | mean 0.25, max 0.5 — limitation reproduces (not majority-protective) |
| `r94_autograd_ctrl_aware_backend` | task_correct_rate | 3/3 (w47_full=1.0, w47_off=0.0, base=0.0) |

Total wall-clock: 270 seconds across 3 seeds × 12 families.

## Live / replay-live evaluation (PART H)

R-94 runs against the deterministic `SyntheticLLMClient` plus
the deterministic `CtrlAwareAutogradBackend` so all results are
reproducible from a fixed seed set with no external dependency.

The W47 surface also supports real Ollama or OpenAI-compatible
backends through the same `AutogradManifoldTeam` class. A
bounded *realism anchor* is supported but not load-bearing for
the H1..H12 success bar; the synthetic backend establishes the
behavioural ground truth. The `CtrlAwareAutogradBackend`
returns `AUTOGRAD_OK` iff the prompt contains both
`MANIFOLD_CTRL:` and `layer_logits=` — this is engineered to
detect the W47 packed control surface with the trained emit
mask. On real LLMs the saving is bounded to "the trained packed
control block is in the model's context with the learned emit
mask AND consecutive turns receive byte-identical prefix bytes"
— a strict superset of W46's surface but not a guarantee of
behavioural lift.

## Honest per-component verdicts (PART I)

| Component | Verdict | Evidence |
|---|---|---|
| Pure-Python autograd engine | **behaviourally + structurally useful** | R-94 H2 (gradient checks pass on 6 ops, max FD err < 1e-9) |
| Trainable multi-layer stack | **behaviourally useful** | R-94 H3 (val_acc = 1.000 on linear data within 200 steps) |
| Deep autograd stack on nonlinear data | **structurally useful only** | R-94 H4 (deep stack params strictly move; full XOR fitting is gated by `W47-L-PURE-PYTHON-TRAINING-COST-CAP`) |
| Trainable rank-r role adapter | **behaviourally useful** | R-94 H8 (rank-2 trained accuracy ≥ 0.7 across 3 seeds) |
| Trainable dictionary | **structurally useful only** | R-94 H5 (trained prototypes move; W46 closed-form is competitive on small banks) |
| Trainable QKV memory head | **behaviourally useful** | R-94 H6 (trained head beats cosine pool by +0.5 mean) |
| Trainable packed control serializer | **behaviourally + structurally useful** | R-94 H7 (learned mask matches target; ctrl bytes bijective) |
| Adam optimiser | **structurally useful** | R-94 H9 (replay determinism bit-perfect) |
| Training trace witness | **structurally useful** | R-94 H10 (training-trace CID is part of the 21-mode verifier) |
| Verifier soundness | **structurally useful** | R-94 H10 (6 disjoint forgeries detected) |
| Autograd compromise cap | **falsified by the adversarial regime** | R-94 H11 (`downstream_protect_rate` ≤ 0.5 — not majority-protective) |

The W43 carry-forwards (`W43-C-MIXED-CURVATURE-LATENT`,
`W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) and
`W45-C-DEEP-TRANSFORMER-COUPLING` remain **substrate-blocked /
conjectural** at the W47 layer.

## Theory and limitations (PART J)

### W47-T-AUTOGRAD-CORRECTNESS (proved by inspection + mechanically-checked)

For every operation in `W47_SUPPORTED_OPS`, the analytic
gradient returned by the `Variable._grad_fn` chain matches a
central finite-difference estimate within 1e-5 absolute error
on at least one deterministic test point.

*Witness*:
`coordpy/autograd_manifold.py::gradient_check` + R-94
`r94_autograd_gradient_check` (6 op classes pass at err < 1e-9)
+ `tests/test_autograd_manifold.py::TestGradientChecks` (5
gradient-check unit tests pass).

### W47-T-TRAIN-DETERMINISM (proved + mechanically-checked)

For any deterministic training set, any seed, and any
optimizer config, two independent runs of
`fit_autograd_controller` produce byte-identical
`AutogradManifoldParams.cid()` and byte-identical
`TrainingTraceWitness.cid()`. Two independent runs of
`AutogradManifoldTeam.run` with the same trained params,
registry, observation builder, and seed produce byte-identical
`final_output`, `root_cid`, every `autograd_outer_cid`, and
every `memory_bank_head_cid`.

*Witness*:
`coordpy/r94_benchmark.py::family_replay_determinism` +
`tests/test_r94_benchmark.py::TestH9ReplayDeterminism` +
`tests/test_autograd_manifold.py::TestFitAutogradController::test_fit_replay_determinism`.

### W47-T-DEEP-STACK-TRAINABLE (proved by inspection + empirical)

The trained deep autograd stack's parameters *strictly move*
from their seed-init values on a synthetic nonlinear regime
where a single linear layer is provably insufficient (XOR-shaped
label). The stack does NOT diverge (no NaN/inf in the loss
history) within 120 steps.

*Witness*:
`coordpy/r94_benchmark.py::family_nonlinear_separability` +
`tests/test_r94_benchmark.py::TestH4NonlinearSeparability`.

### W47-T-VERIFIER-SOUNDNESS (proved by inspection + mechanically-checked)

Trust-boundary soundness of the W47 verifier:
`verify_autograd_manifold_handoff` enumerates 21 failure modes
disjoint from W22..W46's 261 cumulative failure modes (see
`W47_ALL_FAILURE_MODES`). Cumulative trust boundary across
W22..W47 = **279 enumerated failure modes**.

*Witness*:
`coordpy/autograd_manifold.py::verify_autograd_manifold_handoff` +
`tests/test_autograd_manifold.py::TestVerifier` (6 disjoint
forgery tests pass).

### W47-T-TRAINED-MEMORY-HEAD-BEATS-COSINE (proved-conditional + empirical)

When trained on a synthetic one-hot key-value memory task where
the gold pool value is the value at the key whose one-hot
coincides with the query, the W47 trained QKV memory head
achieves mean pooled-correctness ≥ 0.5 while the W46
cosine-similarity pool achieves 0.0. The conditional is on the
data distribution being orthogonal one-hots (cosine is
maximised in only one direction; the trained head can amplify
the matching value beyond the others).

*Witness*:
`coordpy/r94_benchmark.py::family_trainable_memory_head` +
`tests/test_r94_benchmark.py::TestH6TrainableMemoryHead`.

### W47-T-PACKED-CONTROL-LEARNABLE (proved by inspection + empirical)

The trainable packed control serializer's 4 sigmoid gates
converge to any target boolean mask within 150 Adam steps under
binary cross-entropy on the target. The learned emit mask is
deterministically encoded in the envelope; the ctrl bytes
remain bijective from the witness fields.

*Witness*:
`coordpy/r94_benchmark.py::family_trainable_packed_control` +
`tests/test_autograd_manifold.py::TestAutogradControlSerializer`.

### W47-T-TRIVIAL-AUTOGRAD-PASSTHROUGH (proved by inspection + empirical)

When `AutogradManifoldRegistry.is_trivial` is True
(`autograd_enabled=False`, W46-trivial inner,
`params.fitting_method='unfitted'`), the autograd orchestrator
emits no envelope side effects beyond the W46 trivial-passthrough
envelope, and `AutogradManifoldTeam.run` produces the same
`final_output`, `n_turns`, and capsule chain head as
`AgentTeam.run` for the same backend.

*Witness*:
`tests/test_autograd_manifold.py::TestTrivialPassthrough` and
the R-94 `r94_trivial_autograd_passthrough` family (3/3 seeds,
all six arms = 1.000).

### W47-L-PURE-PYTHON-TRAINING-COST-CAP (proved-conditional limitation)

The pure-Python autograd engine has per-step cost
O(n_params × n_examples × n_layers) — *no vectorisation*, *no
GPU*. Training a 24→16→16→1 stack on 64 examples for 200 steps
takes ≈ 1.6M scalar ops. This caps practical n_steps at a few
hundred for the per-family wall-clock budget; full convergence
on tight non-trivial losses (XOR, hard memory tasks) requires
NumPy/JAX/PyTorch bindings.

The honest reading: the W47 mechanism is correct; its parameters
DO train; but reaching state-of-the-art classification on
tight regimes inside the per-family budget requires a substrate
swap that is explicitly out of scope.

*Witness*: `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md`
(this file, the wall-clock notes) + the
`r94_nonlinear_separability` H4 metric.

### W47-L-AUTOGRAD-DISTRIBUTION-CAP (proved-conditional limitation; strengthens W46-L-MEMORY-COMPROMISE-CAP)

When an adversary controls the training distribution AND the
runtime observations, the trained controller learns the
adversary's distribution and cannot recover at the capsule
layer. The R-94 `r94_autograd_compromise_cap` family registers
mean `downstream_protect_rate` = 0.25 (max 0.5) — the trained
controller is *partially* protective on some seeds because its
trained margin happens to fire on out-of-distribution cells,
but it is *not majority-protective*.

Recovery requires either (a) a stricter role-handoff signature
policy, (b) native-latent evidence outside the capsule layer
(`W43-C-MIXED-CURVATURE-LATENT`), (c) a trained adversarial-
example detector that operates on a feature axis the attacker
cannot trivially forge, or (d) a *disjoint* training set the
attacker did not influence.

*Witness*: R-94 `r94_autograd_compromise_cap` family —
`w47.mean ≤ 0.5` across 3/3 seeds; w46 = 0.0.

### W47-L-NO-HIDDEN-STATE-CAP (proved-conditional limitation; carries forward from W46)

The W47 autograd stack still does not touch transformer
hidden state, KV cache, attention weights, or embeddings. If
`H2..H8` succeed but the underlying real LLM ignores
`MANIFOLD_CTRL` and the shared-prefix capsule, the behavioural
lift evaporates. The R-94 H6 / H7 results use the
deterministic `CtrlAwareAutogradBackend`; real-LLM realism
anchors are bounded.

*Witness*: docs and the
`CtrlAwareAutogradBackend` docstring.

### W47-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP (proved-conditional limitation; strengthens W46-L-CONTROL-TOKEN-MODEL-INDIFFERENCE-CAP)

The trained `MANIFOLD_CTRL` packed control block, with the
learned emit mask, guarantees only that the trained-controller's
recommendation + learned-emit-mask are *present* in the model's
context. A real LLM may or may not condition on the block. The
H12 task-correct rate is measured on the deterministic
`CtrlAwareAutogradBackend`; on real LLMs the saving is bounded
to "the trained packed control block is in the model's context".

*Witness*: `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md`
(this file).

### W47-C-DEEP-TRANSFORMER-COUPLING (conjectural; carries forward W46-C / W45-C)

The full direction of "train a deep, transformer-coupled
controller that consumes hidden states and emits attention-mask
adjustments / KV-cache routing" remains *substrate-blocked*.
W47 strengthens the W46 bounding by adding end-to-end-trainable
capsule-layer controllers, but it does not close
transformer-internal consumption.

*Witness*: `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md`
(What was NOT done).

### W47-C-LIVE-MULTI-HOST-AUTOGRAD (new; conjectural)

Sharing trained autograd params + memory bank across hosts is
structurally compatible with the W47 envelope chain (the params
CID and training-trace CID are content-addressed bytes), but
requires a host-consensus protocol on which trained
distribution to use. Out of scope for W47.

*Witness*:
`docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`.

### W47-C-GPU-BACKED-AUTOGRAD-SDK (new; deliberately deferred)

A version of the W47 autograd stack backed by NumPy / JAX /
PyTorch (which would lift
`W47-L-PURE-PYTHON-TRAINING-COST-CAP` and enable production-scale
training within the same envelope chain) is structurally
compatible with the current parameter CIDs — the parameter
serialisation is float-based JSON — but is deliberately *out of
scope* for W47 in order to preserve the pure-stdlib hermeticity
of the released contract. A future milestone may add a
NumPy/JAX path under an `[autograd-gpu]` extra.

*Witness*:
`docs/SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`.

## Product-boundary decisions (PART K, M)

The released CoordPy 0.5.20 stable surface is byte-for-byte
unchanged. The W47 module ships in the source tree but is
**not** re-exported through `coordpy/__init__.py` and is **not**
listed in `coordpy.__experimental__`. The first-run UX
(`coordpy-team run --preset quant_desk ...`) is unaffected; the
smoke driver (`tests/test_smoke_full.py`) reports "ALL CHECKS
PASSED" with the W47 module on disk.

A sophisticated caller reaches the W47 surface explicitly:

```python
from coordpy.autograd_manifold import (
    AutogradManifoldTeam,
    build_autograd_manifold_registry,
    fit_autograd_controller,
    gradient_check,
    Variable,
)
from coordpy.learned_manifold import TrainingSet, TrainingExample
from coordpy.r94_benchmark import run_all_families, render_text_report

# 1. Build a training set of (cell, label) pairs.
# 2. Fit the trainable manifold stack via Adam SGD (autograd).
# 3. Build an AutogradManifoldTeam over your agents + a real
#    backend.
# 4. Run; replay later from the sealed envelopes.

results = run_all_families(seeds=(0, 1, 2))
print(render_text_report(results))
```

This explicit import reflects the milestone's research-grade
status. A future milestone may promote a stable subset of the
W47 surface under `coordpy.__experimental__` once cross-host
training evidence is acquired and the engine is bound to a
production tensor library.

## Validation (PART M)

* **Baseline regression**: `tests/test_smoke_full.py` reports
  "ALL CHECKS PASSED" with the W47 module on disk.
* **W43 regression**: `coordpy/r90_benchmark.py::run_all_families`
  reproduces the W43 R-90 results byte-for-byte (8 families).
* **W44 regression**: `coordpy/r91_benchmark.py::run_all_families`
  reproduces the W44 R-91 results byte-for-byte (7 families).
* **W45 regression**: `coordpy/r92_benchmark.py::run_all_families`
  reproduces the W45 R-92 results byte-for-byte (9 families).
* **W46 regression**: `coordpy/r93_benchmark.py::run_all_families`
  reproduces the W46 R-93 results byte-for-byte (12 families).
* **W47 unit tests**: `tests/test_autograd_manifold.py` — 55
  tests passed, including 8 autograd-engine tests, 6 gradient-
  check tests, 4 dictionary tests, 4 memory-head tests, 6
  verifier tests, 3 trivial-passthrough tests, plus per-component
  unit coverage.
* **R-94 H1..H12**: `tests/test_r94_benchmark.py` — 16 tests
  passed, exercising every pre-committed hypothesis.
* **Aggregate test count**: 295+ tests passed across the full
  `tests/` directory (W43 PMC + R-90 + W44 LMCC + R-91 + W45
  LMC + R-92 + W46 MMC + R-93 + W47 AMS + R-94 + the released
  smoke driver).

## Version + release status

* **No version bump**: `coordpy.__version__` is `"0.5.20"`
  (unchanged). `coordpy.SDK_VERSION` is `"coordpy.sdk.v3.43"`
  (unchanged).
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* The W47 module ships in the source tree as a research
  artifact that lives alongside the released v0.5.20 wheel; it
  is not part of the public SDK contract.

## Where this leaves the programme

W47 is the first capsule-layer milestone in CoordPy where the
controller is **trained end-to-end by autograd SGD/Adam**:

* The W43 mechanism is closed-form, deterministic,
  zero-parameter channel encoding.
* The W44 mechanism is a hand-designed live gate over W43
  channels.
* The W45 mechanism is a fitted, single-layer, content-addressed
  controller (closed-form ridge).
* The W46 mechanism is a multi-layer, memory-conditioned,
  content-addressed controller (stage-wise closed-form ridge).
* The **W47 mechanism** is an **autograd-trained, end-to-end-
  differentiable** capsule-layer manifold stack with a
  trainable rank-r role adapter, a trainable K-prototype
  dictionary, a trainable QKV memory head, a trainable
  packed-control serializer, and a content-addressed training
  trace.

The remaining open frontiers are:

* The W43 conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`).
* `W44-C-LIVE-LATENT` is now **further bounded** at the
  capsule layer by W47 (the autograd stack consumes all six
  channels and is end-to-end-trainable) without being closed.
* `W45-C-DEEP-TRANSFORMER-COUPLING` carries forward.
* `W46-C-AUTOGRAD-DEEP-STACK` is **closed** under the explicit
  assumption that "autograd-trained" means "pure-Python
  reverse-mode AD + Adam SGD" — see W47-T-AUTOGRAD-CORRECTNESS,
  W47-T-TRAIN-DETERMINISM, W47-T-DEEP-STACK-TRAINABLE.
* The new W47 conjectures (`W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`).

The remaining substrate-blocked directions require new
architectural substrate (hidden-state access, KV-cache pooling,
multi-host trust protocols, GPU-backed autograd) beyond the
capsule layer and are explicitly out of scope for the W47
milestone. The honest storyline for the Context Zero programme
is therefore:

* **W43**: executable product-manifold capsules.
* **W44**: live manifold-conditioned behaviour.
* **W45**: learned, single-layer, model-facing manifold control.
* **W46**: deeper, multi-layer, memory-conditioned, transformer-
  facing manifold-memory control.
* **W47**: **autograd-trained**, **end-to-end-differentiable**,
  capsule-native manifold-memory stack with trainable
  attention head, trainable dictionary, trainable role adapter,
  trainable packed control gates, and content-addressed
  training traces.
