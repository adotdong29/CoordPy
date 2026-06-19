# Pre-committed success criterion — W47 Autograd Manifold Stack

> Programme step: post-W46. Mints axis 44 of the Context Zero
> programme. Strictly additive on top of W46 MMC, W45 LMC, W44
> LMCC, W43 PMC, and the released v3.43 line. Honest scope: the
> W47 mechanism is the first capsule-native CoordPy layer where
> the manifold controller is **actually trained end-to-end by
> autograd (reverse-mode AD + SGD/Adam)** rather than stage-wise
> closed-form ridge. It directly attacks the
> `W46-C-AUTOGRAD-DEEP-STACK` carry-forward conjecture. It does
> NOT close the W43 conjectures
> (`W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`), does NOT promote the
> `W45-C-DEEP-TRANSFORMER-COUPLING` carry-forward to a closed
> claim, and does NOT manipulate transformer hidden state, KV
> cache, attention weights, or embeddings. The trained manifold
> stack is an executable proxy at the capsule layer.

## Mechanism

W47 introduces the **Autograd Manifold Stack (AMS)** — the first
capsule-native CoordPy layer where the gating policy is shaped by
**trainable, autograd-fitted, deeper, nonlinear** structure with
explicit gradient-based optimisation. The mechanism is composed
of nine trainable, content-addressed components plus a small
pure-Python reverse-mode autograd engine. All components are
implemented in pure Python / stdlib (no NumPy, no PyTorch, no
JAX) so the W47 line preserves the released SDK contract and
remains hermetically reproducible.

* **Pure-Python reverse-mode autograd engine.** A `Variable`
  class with scalar values, gradients, and a topologically-sorted
  backward pass. Supports `+`, `-`, `*`, `**`, `dot`, `tanh`,
  `sigmoid`, `relu`, `exp`, `log`, `softmax`, `mean`, `sum`.
  Bound by a deterministic numerical floor (no random init unless
  seeded). The engine passes a finite-difference gradient check
  at six places (linear, tanh-MLP, softmax cross-entropy, dot
  product, sigmoid binary cross-entropy, attention pool).

* **Trainable multi-layer manifold stack.** An `L`-layer fully
  connected stack over flattened channel features
  (`W45_N_CHANNELS * feature_dim = 24` inputs at the W43/W45/W46
  defaults). Each layer has a trainable weight matrix, trainable
  bias vector, and a deterministic `tanh` activation; the final
  layer collapses to a single scalar gate logit. Layer weights
  are initialised deterministically from a seed-derived
  pseudo-random uniform on `[-init_scale, init_scale]` so two
  fits from the same seed produce byte-identical params.

* **Trainable rank-`r` role adapter.** A per-role low-rank
  delta `A_r · B_r^T` applied to the per-layer projection of
  the deepest hidden state. `A_r` is shape `(hidden_dim, r)`,
  `B_r` is shape `(r,)`. Both factors are trained by SGD on the
  per-role residual loss. When a role has fewer than `r + 1`
  training examples, the adapter falls back to the all-zero
  delta (recorded explicitly in the envelope as
  `W47_NO_ROLE_DELTA`).

* **Trainable dictionary / codebook.** `K` trainable prototype
  vectors over the flattened channel feature space, jointly
  optimised via a soft-assignment cross-entropy loss with a
  straight-through estimator: at every step the encoder picks
  the closest prototype with a Gumbel-softmax relaxation
  (temperature pinned at 1.0 since no random draw); gradients
  flow through the residual back into all prototypes weighted by
  the softmax assignment. Encode is still bijective at inference
  (closest-prototype + residual = original feature).

* **Trainable memory read/write head.** Replaces the W46
  cosine-similarity pool with a learned query / key / value
  projection. The memory bank stores per-entry `(key, value,
  gate_logit)` triplets; the read head computes
  `softmax(Q · K^T / sqrt(d)) · V` over admissible entries,
  exactly as in a transformer attention head, but at the
  capsule layer. Heads are trained on the memory-conditioned
  branching-history task.

* **Trainable packed control serializer.** A small trainable
  gate per `MANIFOLD_CTRL` auxiliary field (`layer_logits`,
  `mem_attn`, `dict_idx`, `mem_summary`). Each gate is a
  sigmoid-activated scalar trained jointly with the rest of the
  stack via a learned-tokens-per-bit auxiliary loss; at
  inference, gates above 0.5 are emitted, gates below 0.5 are
  suppressed. The emitted CTRL bytes remain bijective from the
  envelope; suppression is *deterministically* recorded.

* **Adam-style optimiser in pure Python.** First-moment +
  second-moment EMAs over per-parameter gradients, with a fixed
  learning rate, fixed betas (β1 = 0.9, β2 = 0.999), fixed
  epsilon (1e-8), and a deterministic step counter. The
  optimiser is exposed as `AdamW47` so an auditor can re-run
  the training trace from the same training set + seed and
  recover identical parameters bit-for-bit (up to IEEE-754
  rounding at the documented 12-decimal precision).

* **Training trace witness.** Every training run binds a
  content-addressed `TrainingTraceWitness` recording: the
  initial seed, the number of steps, the optimiser config
  (learning_rate, beta1, beta2, eps), the final training loss,
  the per-step loss vector (truncated to first/last K = 8 by
  default for envelope size), the gradient norm history, and
  the final parameter CID. The training trace CID is a
  first-class field of the W47 envelope; an auditor can
  re-fit and verify the CID matches.

* **Autograd team orchestrator.** `AutogradManifoldTeam` sits
  beside `ManifoldMemoryTeam` (W46), `LearnedManifoldTeam`
  (W45), `LiveManifoldTeam` (W44), and `AgentTeam` (released).
  Reduces to `ManifoldMemoryTeam` byte-for-byte when configured
  trivially (autograd disabled + W46-trivial inner). This is
  the W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH falsifier.

## H1..H12 success bar

Twelve pre-committed hypotheses on the R-94 benchmark family;
each is exercised by a per-family test in
`tests/test_r94_benchmark.py` plus per-component unit coverage
in `tests/test_autograd_manifold.py`.

### H1 — Trivial autograd passthrough

A trivially-configured `AutogradManifoldRegistry`
(`autograd_enabled=False`, W46-trivial inner) reduces to
`ManifoldMemoryTeam.run` byte-for-byte. The
`r94_trivial_autograd_passthrough` family reports
`passthrough_ok = 1.0` for the `baseline_team`,
`w43_closed_form`, `w44_live_coupled`, `w45_learned_coupled`,
`w46_memory_coupled`, AND `w47_autograd` arms across all five
seeds.

### H2 — Autograd engine correctness vs finite differences

For every supported op (`+`, `-`, `*`, `dot`, `matmul`, `tanh`,
`sigmoid`, `relu`, `exp`, `log`, `softmax`, `mean`, `sum`), the
analytic gradient matches a central finite-difference estimate
with absolute error < 1e-5 on a deterministic test point. This
is the `r94_autograd_gradient_check` family;
`autograd_grad_correct = 1.0` across all seeds.

### H3 — Trained stack converges on the synthetic regime

On the linearly-separable spherical-axis classification task
(label = sign(spherical)), the W47 autograd stack achieves
training loss < 0.05 within 200 SGD steps, *and* its validation
accuracy on a held-out partition reaches >= 0.95. The
`r94_autograd_convergence` family;
`final_train_loss <= 0.05` AND `val_acc >= 0.95`.

### H4 — Deep stack strictly beats shallow stack on nonlinear data

On a synthetic XOR-shaped axis (label = sign(spherical * causal)),
a 3-layer autograd stack achieves validation accuracy >= 0.90
while a 1-layer stack stays at ~0.5 (chance). The
`r94_nonlinear_separability` family; `deep_acc - shallow_acc
>= 0.30` across all seeds, AND `deep_acc >= 0.85` minimum.

### H5 — Trainable dictionary outperforms K-prototype clustering

On the per-channel reconstruction loss benchmark, the trained
dictionary's mean residual L1 is **strictly lower** than the
W46 stage-fitted dictionary's mean residual L1 on the same
training set. The `r94_trainable_dictionary` family;
`trained_residual_l1 < w46_residual_l1` AND
`trained_residual_l1 <= 0.5 * w46_residual_l1` minimum across
seeds.

### H6 — Trainable attention head beats cosine pool on memory task

On the multi-turn memory-coupled task where the gold ratify
decision at turn `t >= 3` depends on which of the prior turns
ratified, a trained attention head reaches >= 0.90 precision on
deep turns while the W46 cosine pool stays at the W46 reference
(also 1.0 on the engineered family; on the OOD-perturbed family
the W46 cosine pool drops below 0.5 while the trained head
recovers >= 0.85). The `r94_trainable_memory_head` family;
`trained_precision >= 0.80` minimum.

### H7 — Trainable packed control serializer is bijective and bounded

The trained control-token gate either emits or omits each
auxiliary field; the resulting bytes remain bijectively
recoverable from the envelope; total visible-token cost stays
within `[W45_hint_tokens, W46_full_ctrl_tokens]`. The
`r94_trainable_packed_control` family;
`ctrl_round_trip_ok = 1.0` AND
`mean_ctrl_tokens <= W46_max_ctrl_tokens`.

### H8 — Trainable rank-`r` role adapter recovers rank-2 inversion

On the dual-axis role-shift bank (W46 H4 regime), the trained
rank-2 adapter recovers role2 + role3 inversions; the trained
rank-1 adapter recovers at most one. The
`r94_trainable_role_adapter` family;
`trained_rank2_acc >= 0.9` AND `trained_rank2_acc -
trained_rank1_acc >= 0.20`.

### H9 — Replay determinism with frozen weights

Two independent runs of `AutogradManifoldTeam.run` with the same
trained params, registry, observation builder, and training set
produce byte-identical `final_output`, byte-identical root CID,
every `autograd_outer_cid`, every `memory_bank_head_cid`, the
trained-params CID, and the training-trace CID. The
`r94_replay_determinism` family; `replay_determinism_ok = 1.0`
across all seeds.

### H10 — Autograd envelope verifier soundness

The W47 verifier rejects 18+ disjoint forged envelopes (schema
mismatch, params CID mismatch, training-trace CID mismatch,
control-token mismatch, prefix-capsule mismatch, memory-bank
mismatch, outer-CID mismatch, etc.). Cumulative trust boundary
across W22..W47 = **279 enumerated failure modes** (261 from
W22..W46 + 18 new at W47).

### H11 — Autograd compromise cap (limitation reproduces)

Adversarial all-channel forgery + a forged memory bank with a
*matching forged training distribution* (so the trained
controller learns the adversary's distribution): the W47
mechanism cannot recover. The `r94_autograd_compromise_cap`
family; `downstream_protect_rate = 0.0` across all seeds —
proved-conditional limitation
`W47-L-AUTOGRAD-DISTRIBUTION-CAP`.

### H12 — Released SDK byte-identity preserved

`tests/test_smoke_full.py` reports "ALL CHECKS PASSED" with the
W47 module on disk; `coordpy.__version__` is still `"0.5.20"`;
`coordpy.SDK_VERSION` is still `"coordpy.sdk.v3.43"`; the
released wheel surface is byte-for-byte unchanged. The W47
module ships in the source tree at `coordpy.autograd_manifold`
and is reachable only through an explicit import.

## Falsifiers

* **W47-L-TRIVIAL-AUTOGRAD-PASSTHROUGH** — a trivially-configured
  `AutogradManifoldRegistry` (`autograd_enabled=False`,
  `W46`-trivial inner, no training) reduces to
  `ManifoldMemoryTeam.run` byte-for-byte; if `H1` fails, the
  trivial-passthrough property is falsified.

* **W47-L-AUTOGRAD-DISTRIBUTION-CAP** — adversarial all-channel
  forgery + forged memory bank + forged training set: the
  trained controller cannot recover because it learned the
  adversary's distribution.
  `downstream_protect_rate == 0.0` reproduces this limitation
  honestly.

* **W47-L-NO-HIDDEN-STATE-CAP** — the W47 stack still does not
  touch transformer-internal state. If `H2..H8` succeeded but
  the underlying real LLM ignores `MANIFOLD_CTRL` and the
  shared-prefix capsule, the behavioural lift evaporates. The
  R-94 H6 / H7 results use the deterministic
  `MemoryAwareSyntheticBackend` plus a new
  `CTRL_AWARE_AUTOGRAD_BACKEND`; real-LLM realism anchors are
  bounded.

## Per-component verdicts (preview)

The R-94 family produces a per-component verdict matrix:

* **Autograd engine** — *behaviourally + structurally useful*
  (passes gradient checks; trains the rest of the stack).
* **Trainable multi-layer stack** — *behaviourally useful*
  (deep stack strictly beats shallow stack on nonlinear data).
* **Trainable role adapter** — *behaviourally useful* (rank-2
  beats rank-1 on dual-axis inversion).
* **Trainable dictionary** — *structurally useful* (lower
  reconstruction error than W46 stage-fitted).
* **Trainable memory head** — *behaviourally useful* (beats W46
  cosine pool on OOD-perturbed memory regime).
* **Trainable packed control serializer** — *behaviourally
  useful on the synthetic backend; not load-bearing on real
  LLMs* (same caveat as W46 H6).
* **Training trace witness** — *structurally useful* (full
  auditability of the training run).
* **Autograd compromise cap** — *limitation reproduces honestly*
  (no claim that trained controllers defeat all-distribution
  forgery).

## Architecture triage

| Frontier candidate                          | W47 bucket                       | Verdict |
|---|---|---|
| Pure-Python autograd engine                 | **trainable now**                | shipped |
| Trainable multi-layer manifold stack        | **trainable now**                | shipped |
| Trainable rank-r role adapter               | **trainable now**                | shipped |
| Trainable dictionary / codebook             | **trainable now**                | shipped |
| Trainable memory read/write head            | **trainable now**                | shipped |
| Trainable packed control serializer         | **trainable now**                | shipped |
| Trainable shared-prefix policy gate         | **trainable only under synthetic supervision** | shipped: trained on prefix-reuse labels |
| Lightweight transformer-proxy module        | **trainable now (capsule-layer proxy only)** | shipped: 3-layer + memory + adapter behaves as a depth-3 capsule-layer transformer-proxy |
| True KV-cache pooling across turns          | **substrate-blocked**            | unchanged from W46 |
| Transformer-internal mixed-curvature attention | **substrate-blocked**         | unchanged from W43 |
| Continuous Grassmannian homotopy            | **substrate-blocked**            | unchanged from W43 |
| Hidden-state-aware time attention           | **substrate-blocked**            | unchanged from W46 |

## What W47 explicitly does NOT do

* W47 does NOT promote `W47-C-DEEP-TRANSFORMER-COUPLING` (the
  W46 carry-forward) to closed. The trained manifold stack
  operates strictly over W43 capsule-layer channel encodings.
* W47 does NOT manipulate the transformer's internal attention,
  KV cache, hidden states, or embeddings.
* W47 does NOT claim multi-host coupling. The trained controller
  runs on a single process / single backend.
* W47 does NOT claim training-data-free generalisation. The
  controller is trained on a hermetic synthetic bank
  pre-committed in the R-94 sources.
* W47 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD` — sharing
  trained params + memory bank across hosts requires a host
  consensus protocol outside the W47 scope.
* W47 does NOT ship CUDA / GPU support — the pure-Python autograd
  engine is correct but slow (training a 24→16→1 stack on 64
  examples for 200 steps takes ≈ 5–8 seconds on a Mac M-series
  CPU; production training would need numpy/jax bindings).

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`. The
  W47 module ships at `coordpy.autograd_manifold` and is
  reachable only through an explicit import — same convention
  as `coordpy.product_manifold` (W43),
  `coordpy.live_manifold` (W44),
  `coordpy.learned_manifold` (W45), and
  `coordpy.manifold_memory` (W46).

## New theorem-style claims (preview)

* **W47-T-AUTOGRAD-CORRECTNESS** (proved-by-inspection +
  mechanically-checked) — analytic gradients match
  finite-difference estimates within 1e-5 absolute error on the
  closed set of supported ops.
* **W47-T-TRAIN-DETERMINISM** (proved + mechanically-checked) —
  two independent training runs from the same seed +
  training set produce byte-identical parameter CIDs and
  training-trace CIDs.
* **W47-T-DEEP-STACK-NONLINEAR-SEPARATION** (empirical) —
  a 3-layer tanh stack with hidden width 16 separates the
  XOR-shaped (spherical * causal) signal with validation
  accuracy >= 0.85 across 5 seeds, while a 1-layer stack
  remains at chance.
* **W47-L-NONLINEAR-PROXY-CAP** (proved-conditional limitation)
  — the autograd stack is at best a depth-`L` capsule-layer
  proxy; it cannot recover signals that require
  transformer-internal hidden states.
* **W47-L-AUTOGRAD-DISTRIBUTION-CAP** (proved-conditional
  limitation; strengthens W46-L-MEMORY-COMPROMISE-CAP) — when
  the adversary controls the training distribution, the trained
  controller learns the adversary's signal and cannot recover.
* **W47-L-TRAINING-COST-CAP** (proved by inspection) — the
  pure-Python autograd engine has per-step cost
  O(n_params × n_examples); training a 24→16→16→1 stack on 64
  examples for 200 steps is ≈ 1.6M scalar ops; production
  training requires bindings to NumPy/JAX/PyTorch.

## What this enables for the programme

* **Closes** `W46-C-AUTOGRAD-DEEP-STACK` (the carry-forward
  conjecture from W46) under the explicit assumption that
  "autograd-trained" means "pure-Python reverse-mode AD + SGD".
  The CoordPy capsule-layer manifold stack is now trainable
  end-to-end with byte-identical replay and full provenance.
* **Strengthens** `W46-L-MEMORY-COMPROMISE-CAP` to also include
  training-distribution forgery
  (`W47-L-AUTOGRAD-DISTRIBUTION-CAP`).
* **Preserves** all of W43/W44/W45/W46's deterministic-audit
  properties — the W47 module is strictly additive.
* **Does not close** the substrate-blocked W43 conjectures
  (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) or
  `W45-C-DEEP-TRANSFORMER-COUPLING`.
* **Mints** new W47 conjectures
  (`W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`) that need new substrate.

## Done = the following commits land

1. `coordpy/autograd_manifold.py` ≈ 2000 LoC, pure Python /
   stdlib only.
2. `coordpy/r94_benchmark.py` ≈ 1000 LoC, dependency-free.
3. `tests/test_autograd_manifold.py` — ≥ 25 tests covering
   every component + the trivial-passthrough falsifier +
   gradient checks.
4. `tests/test_r94_benchmark.py` — ≥ 12 tests covering H1..H12.
5. `docs/RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md` and this
   success-criterion file.
6. Updates to `docs/RESEARCH_STATUS.md`,
   `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
   `docs/HOW_NOT_TO_OVERSTATE.md`,
   `docs/context_zero_master_plan.md`,
   `papers/context_as_objects.md`, and `CHANGELOG.md`.

No README change. No version bump. No PyPI release.
