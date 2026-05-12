# Pre-committed success criterion — W51 Persistent Cross-Backend Latent Coordination (PXBLC)

> Programme step: post-W50. Mints axis 48 of the Context Zero
> programme. Strictly additive on top of W50 XBLC, W49 MBCC, W48
> SSTP, W47 AMS, W46 MMC, W45 LMC, W44 LMCC, W43 PMC, and the
> released v3.43 line. Honest scope: W51 stacks **six orthogonal
> trainable advances** on top of W50 — a persistent GRU-style
> shared latent state V3 with chain-walk + cross-role mixer,
> a triple-backend translator with transitivity-trained
> behavioural fidelity, a depth-six branch/cycle-specialised
> proxy stack V2, a hierarchical coarse-fine codebook
> compression V3 with degradation-curve probe, a length-eight
> reconstruction V3 with two-headed (causal + branch) heads, and
> a branch/cycle-specialised memory head with cross-branch
> consensus and cross-cycle merger. It does NOT touch
> transformer-internal hidden state, real KV cache bytes,
> attention weights, embeddings, or real tokenizers. W43..W50's
> substrate-blocked conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`, `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`,
> `W47-C-DEEP-TRANSFORMER-COUPLING`,
> `W48-C-REAL-KV-COUPLED-PROXY`,
> `W48-C-MULTI-HOST-SHARED-STATE`,
> `W49-C-DEEP-TRANSFORMER-COUPLING`,
> `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`,
> `W50-C-DEEP-TRANSFORMER-COUPLING`,
> `W50-C-REAL-KV-COUPLED-PROXY`,
> `W50-C-MULTI-HOST-SHARED-STATE`) carry forward unchanged. W51
> is the strongest *executable proxy* line we can write today; it
> is *not* a closure of those.

## Mechanism

W51 introduces the **Persistent Cross-Backend Latent
Coordination (PXBLC)** layer — six orthogonal capsule-native
advances on top of W50's five:

1. **`PersistentSharedLatentState`** (M1). A GRU-style trainable
   persistent latent vector that updates per turn:
   `s_t = (1 - z_t) ⊙ s_{t-1} + z_t ⊙ tanh(W_h · [s_{t-1}; x_t])`
   where `z_t = sigmoid(W_z · [s_{t-1}; x_t])` is a learned
   update gate. Carries a **cross-role mixer** (a learned
   convex combination over per-role views of `s_t`) and a
   content-addressed `PersistentLatentState` chain. Each entry
   binds `(turn_index, role, parent_state_cid, mixer_cid)`.
   Chain-walk up to depth 16 recovers every prior state from
   the envelope chain alone.

2. **`TripleBackendTranslator`** (M2). Three backend tags
   `(A, B, C)` with **direct** translators `A→B`, `A→C`,
   `B→C` plus a **transitivity loss** that penalises
   disagreement between `A→C` and `A→B→C`. Trains all three
   translators jointly; identity-init seeds keep an "untrained"
   trivial fall-back. The witness binds
   `(triple_tag, direct_fidelity_ab, direct_fidelity_ac,
   direct_fidelity_bc, transitive_fidelity_a_b_c,
   transitivity_gap, anchor_status)`.

3. **`DeepProxyStackV2`** (M3). `L=6` stacked
   `ProxyTransformerBlock`s (vs W50's `L=4`), with two new
   structural extensions:
   - **branch-specialised heads** — one attention head per
     branch type, routed by the branch index;
   - **cycle-specialised heads** — one attention head per
     cycle-position class.
   Per-layer trainable temperature parameter `tau_l`. Carries
   a `DeepProxyStackV2ForwardWitness` with per-layer activation
   norms, per-layer gate values, per-layer temperatures, and a
   compositional-task accuracy bound (`L=6` should strictly
   beat `L=4` on a 6-step composition).

4. **`HierarchicalCompressionCodebook`** (M4). Two-level
   codebook: coarse `K1=32` clusters + fine `K2=16` per-cluster
   sub-prototypes. Each carrier is encoded as a `(coarse_code,
   fine_code)` pair plus a learned hierarchical emit gate.
   Target: **≥ 12 bits per visible-token** (vs W50's 8). Carries
   a `HierarchicalCompressionWitness` with `(coarse_cid,
   fine_cid, hierarchical_gate_cid, achieved_bits_per_token,
   degradation_curve)`. The degradation curve records achieved
   bits/token across decreasing token budgets `{1, 2, 4, 8}`.

5. **`LongHorizonReconstructionV3`** (M5). Trained two-headed
   reconstruction head: a **causal head** (recovers turn `t-k`
   features from current carrier; `k ∈ {1..8}`) and a
   **branch head** (recovers turn `t-k` features conditioned
   on branch path). Witness records MSE for each `k` plus a
   **degradation curve** for `k ∈ {1..16}` (we expect
   reconstruction to degrade gracefully past `k=8`).

6. **`BranchCycleMemoryHead`** (M6). A memory head that
   maintains separate **branch-storage pages** (one per branch
   index) and **cycle-storage pages** (one per cycle-position
   class). Cross-branch readout uses a learned consensus weight
   over branch pages; cross-cycle readout uses a learned cycle
   merger. The head is trainable end-to-end via the W47
   autograd engine. Witness binds `(n_branch_pages,
   n_cycle_pages, page_storage_cid, branch_consensus_cid,
   cycle_merger_cid, mean_branch_recall, mean_cycle_recall)`.

All of it is pure-Python / stdlib. No NumPy, no PyTorch, no JAX
dependency. The pure-Python reverse-mode autograd engine from
W47 (`Variable` + `AdamOptimizer`) is reused unchanged. The
released SDK v3.43 contract remains byte-for-byte unchanged.

Honest scope (do-not-overstate)
-------------------------------

W51 does NOT touch transformer-internal hidden state, KV cache
bytes, attention weights, or embeddings. Every parameter of
every W51 module operates over W43..W50 capsule-layer
encodings, the W47 trainable channel features, the W49
multi-bank pseudo-KV factor slots, the W50 shared-latent
carrier V2 chain, and the new W51 persistent latent state
chain. The triple-backend translator is **not** cross-
tokenizer translation; it operates over capsule-layer
carriers and synthetic backend tags exclusively.

W51 does NOT claim "real cross-model behavioural transfer"
beyond what W50 anchored. The primary anchor for M2 is three
synthetic backends with distinct hint-aware policies; when
`COORDPY_W51_OLLAMA_REACHABLE=1` is set and an Ollama daemon
is reachable, a best-effort triple-backend probe runs and is
recorded. Otherwise the witness records
`anchor_status = "synthetic_only"`. The
`W50-C-CROSS-TOKENIZER-LATENT-TRANSFER` conjecture carries
forward sharpened as
`W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY` — transitivity
is now trained and auditable on capsule carriers, but real
tokenizer-level transitivity remains out of W51 scope.

W51 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
`W48-C-DEEP-TRANSFORMER-COUPLING`,
`W49-C-DEEP-TRANSFORMER-COUPLING`, or
`W50-C-DEEP-TRANSFORMER-COUPLING`. The `L=6` deep stack V2 is
a deeper *capsule-layer proxy*, not a deeper transformer.

W51 does NOT claim adversarial robustness under training-
distribution forgery — `W51-L-PXBLC-DISTRIBUTION-CAP`
(strengthens `W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`).

W51 does NOT claim depth monotonicity. On regimes where the
optimal composition depth is `≤ 4`, the `L=6` stack may
regress relative to `L=4`. The `W51-L-DEEP-STACK-OVERDEPTH-CAP`
falsifier reproduces honestly.

W51 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine reused from W47 is correct but slow; W51
trains on bounded synthetic banks (~16–48 examples for
~12–24 Adam steps per module, stage-wise; the e2e step count
is bounded by the
`W51-L-PURE-PYTHON-TRAINING-COST-CAP` carry-forward).

W51 is strictly additive. When configured trivially
(`persistent_state_enabled=False`,
`triple_backend_enabled=False`, `deep_stack_v2_enabled=False`,
`hierarchical_compression_enabled=False`,
`long_horizon_reconstruction_enabled=False`,
`branch_cycle_memory_enabled=False`, W50-trivial inner),
the `W51Team` orchestrator reduces to `W50Team.run`
byte-for-byte — the `W51-L-TRIVIAL-W51-PASSTHROUGH` falsifier.

This module ships at `coordpy.persistent_shared_latent`,
`coordpy.cross_backend_translator`,
`coordpy.deep_proxy_stack_v2`,
`coordpy.hierarchical_compression`,
`coordpy.long_horizon_retention`,
`coordpy.branch_cycle_memory`, and the composition module at
`coordpy.w51_team`. It is NOT exported through
`coordpy.__init__`'s public surface at this milestone; the
stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W51 surface through explicit
`from coordpy.<module> import ...` imports — same convention
as W43..W50.

## H1..H18 success bar

Eighteen pre-committed hypotheses across **two benchmark
families** (R-100 + R-101); each is exercised by a per-family
test in `tests/test_r100_benchmark.py` /
`tests/test_r101_benchmark.py` plus per-component unit
coverage in
`tests/test_persistent_shared_latent_w51.py`,
`tests/test_cross_backend_translator_w51.py`,
`tests/test_deep_proxy_stack_v2_w51.py`,
`tests/test_hierarchical_compression_w51.py`,
`tests/test_long_horizon_retention_w51.py`,
`tests/test_branch_cycle_memory_w51.py`,
`tests/test_w51_trivial_passthrough_byte_identical.py`, and
`tests/test_w51_team_envelope_chain.py`.

### H1 — Trivial W51 passthrough (R-100)

A trivially-configured `W51Registry` (all enable flags
`False`, W50-trivial inner) reduces to `W50Team.run`
byte-for-byte. The `family_trivial_w51_passthrough` family
reports `passthrough_ok = 1.0` across all baseline + new
arms across all three seeds.

### H2 — Persistent latent state V3 strict gain (R-100)

On a length-12 task where the gating decision at turn `T`
depends on a feature emitted at turn 0, the W51 persistent
latent state (with GRU update) preserves the early signal
where W50's per-turn carrier does not. The
`family_persistent_state_long_horizon_gain` family;
`(w51_recall - w50_recall) ≥ 0.20` on mean across seeds —
typical W51 trained recall is 0.60–0.80 vs W50 baseline
< 0.30 on the same regime.

### H3 — Triple-backend transitivity (R-100)

The trained `TripleBackendTranslator` shows transitivity:
`fidelity(A→C) ≈ fidelity(A→B→C)` within
`transitivity_gap ≤ 0.10` on synthetic primaries. The
`family_triple_backend_transitivity` family;
`mean(transitivity_gap) ≤ 0.10` AND
`mean(transitive_fidelity_a_b_c) ≥ 0.85` across seeds.

### H4 — Deeper L=6 stack structural-utility floor (R-100)

On a synthetic six-step product-amplified composition regime
where the target gate depends on a composed nonlinear
function of six channels, the `L=6` deep stack V2 achieves
classifier accuracy ≥ 0.65 (well above chance) and
**non-regression** relative to W50's `L=4` baseline. The
`family_deep_stack_v2_depth_strict_gain` family;
`mean_acc(L=6) ≥ 0.65` AND `acc(L=6) - acc(L=4) ≥ -0.05`
on mean across seeds — honest about non-monotonicity under
the W47 pure-Python autograd training cost cap
(`W51-L-PURE-PYTHON-TRAINING-COST-CAP`). The depth-six
expressivity gain reported in this family is **inseparable
from the branch/cycle-specialised heads** in H5 — the strict
behavioural win at W51 comes from M3 as a whole, not from
depth alone.

### H5 — Branch-specialised heads gain (R-100)

On a regime with two distinct branch types whose target
gates require different attention patterns, the branch-
specialised heads in `DeepProxyStackV2` achieve mean accuracy
strictly greater than a depth-six stack with shared heads
only. The `family_branch_specialised_heads_gain` family;
`(branch_acc - shared_acc) ≥ 0.05` on mean across seeds.

### H6 — Branch/Cycle memory head gain (R-100)

On a multi-branch path where each branch needs its own
storage and a cycle needs cross-cycle merger, the
`BranchCycleMemoryHead` strictly beats W50's generic
multi-bank memory at recovering branch-specific facts. The
`family_branch_cycle_memory_gain` family;
`(branch_cycle_recall - generic_recall) ≥ 0.15` on mean
across seeds.

### H7 — Triple-backend Ollama realism anchor (R-100)

When `COORDPY_W51_OLLAMA_REACHABLE=1` is set and an Ollama
daemon is running, the triple-backend translator probe
records bounded fidelity ≥ 0.60 across at least one direct
edge of the triple. When the env var is absent, the family
records `anchor_status = "synthetic_only"` and
`skipped_ok = 1.0`. The
`family_triple_backend_realism_probe` family.

### H8 — W51 envelope verifier soundness (R-100)

The W51 verifier rejects 24+ disjoint forged envelopes
(schema mismatch, persistent state CID mismatch, triple-
backend witness CID mismatch, deep-stack-v2 forward witness
CID mismatch, hierarchical compression witness CID mismatch,
long-horizon reconstruction witness CID mismatch, branch-
cycle memory witness CID mismatch, outer-CID mismatch, etc.).
Cumulative trust boundary across W22..W51 = **367
enumerated failure modes** (343 from W22..W50 + 24 new at
W51).

### H9 — W51 replay determinism (R-100)

Two independent runs of `W51Team.run` with the same training
set, seed, registry, and observation builder produce
byte-identical `final_output`, root CID, every
`w51_outer_cid`, every per-module witness CID, the persistent
state chain CID list, and the triple-backend translator
witness CID. The `family_w51_replay_determinism` family;
`replay_determinism_ok = 1.0` across all seeds.

### H10 — Cross-backend translator distribution cap (R-100)

Adversarial all-channel forgery + forged backend tags +
forged transitivity training set: the W51 translator cannot
recover. The `family_cross_backend_translator_compromise_cap`
family; `downstream_protect_rate ≥ 0.7` across all seeds —
proved-conditional limitation
`W51-L-CROSS-BACKEND-TRANSLATOR-COMPROMISE-CAP`.

### H11 — 12-turn long-horizon retention (R-101)

On a length-12 task with a target fact emitted at turn 0,
the W51 persistent state + cross-role mixer + carrier V2
jointly recover the fact with cosine ≥ 0.60 at turn 11.
W50 baseline (no persistent GRU update) drops below 0.30
on the same regime under pure-Python autograd training.
The `family_long_horizon_retention_12turn` family;
`w51_cosine_at_turn_11 ≥ 0.60` AND
`(w51_cosine - w50_cosine) ≥ 0.20` on mean across seeds.

### H12 — 16-turn long-horizon retention stretch (R-101)

On a length-16 task (deliberately a stretch — we do NOT
expect 1.0), W51 cosine ≥ 0.40 at turn 15 — honest about
drop-off and the pure-Python autograd training cost cap.
The `family_long_horizon_retention_16turn_stretch` family;
`w51_cosine_at_turn_15 ≥ 0.40` on mean across seeds.

### H13 — Reconstruction V3 MSE at k=5 (R-101)

The trained `LongHorizonReconstructionV3Head` recovers the
flat-feature vector of turn `t-5` from the carrier at turn
`t` with MSE ≤ 0.50 — extending W50's V2 head at k=3 (≤ 0.25
floor) to a longer horizon under the pure-Python autograd
training cost cap. The
`family_reconstruction_v3_recovers_t_minus_5` family;
`mse_at_k5 ≤ 0.50` on mean across seeds.

### H14 — Reconstruction V3 MSE at k=8 stretch (R-101)

At the design-maximum lookback k=8, V3 recovers prior-turn
features at MSE ≤ 0.60 — honest about drop-off and the
pure-Python autograd training cost cap. The
`family_reconstruction_v3_k8_stretch` family;
`mse_at_k8 ≤ 0.60` on mean across seeds.

### H15 — Hierarchical compression ≥ 12 bits/token (R-101)

W51 achieves a structured-bits-per-visible-token ratio
≥ 12.0 on a fixed-length task while retention cosine stays
≥ 0.85. W50 baseline on the same regime hits ≤ 9.0. The
`family_hierarchical_compression_12bits` family;
`w51_bits_per_token ≥ 12.0` AND `retention_cosine ≥ 0.85`
on mean across seeds.

### H16 — Compression degradation curve graceful (R-101)

Under decreasing token budgets `{8, 4, 2, 1}`, the
hierarchical compression achieves a monotone degradation
curve: bits/token does not collapse to chance at any budget,
and retention cosine never drops below 0.50. The
`family_compression_degradation_curve` family;
`min(bits_per_token over budgets) ≥ 4.0` AND
`min(retention_cosine over budgets) ≥ 0.50`.

### H17 — W51 multi-block distribution cap (R-101)

Adversarial all-channel forgery + forged role banks + forged
persistent latent state + forged hierarchical codebook +
forged triple-backend training set: the W51 stack cannot
recover. The `family_w51_distribution_cap` family;
`downstream_protect_rate ≥ 0.7` across all seeds —
proved-conditional limitation
`W51-L-PXBLC-DISTRIBUTION-CAP` (strengthens
`W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`).

### H18 — Deep-stack overdepth cap reproduces (R-101)

On a 2-step composition regime where W50's `L=4` is already
saturating, the deeper `L=6` stack should NOT strictly
improve and may regress slightly due to optimisation noise
from the extra parameters under the pure-Python autograd
training budget. The `family_deep_stack_v2_overdepth_cap`
family;
`(acc_L6 - acc_L4) ≤ +0.05` on mean across seeds — honest
about non-monotonicity of depth.

## Falsifiers

* **W51-L-TRIVIAL-W51-PASSTHROUGH** — a trivially-configured
  `W51Registry` reduces to `W50Team.run` byte-for-byte; if
  H1 fails, the trivial-passthrough property is falsified.

* **W51-L-CROSS-TOKENIZER-TRIPLE-CAP** — when the realism
  probe is skipped (Ollama unreachable), the triple-backend
  cross-tokenizer transitivity conjecture remains carried
  forward, witnessed by H7's skip path.

* **W51-L-CROSS-BACKEND-TRANSLATOR-COMPROMISE-CAP** —
  adversarial all-channel forgery + forged triple-backend
  training: the trained translator cannot recover.
  Reproduces honestly in R-100.

* **W51-L-DEEP-STACK-OVERDEPTH-CAP** — L=6 stack does NOT
  strictly improve over L=4 on shallow composition regimes;
  bounds the depth claim.

* **W51-L-LONG-HORIZON-DROPOFF-CAP** — at k > 8 the V3
  reconstruction head's MSE rises above 0.50; bounds the
  reconstruction claim.

* **W51-L-COMPRESSION-RATE-FLOOR-V2-CAP** — target rate 20
  bits/visible-token exceeds the K1=32 + K2=16 hierarchical
  codebook's information capacity; rate is missed or
  retention floor breaks.

* **W51-L-PXBLC-DISTRIBUTION-CAP** — adversarial all-channel
  forgery + forged role banks + forged persistent latent
  state + forged hierarchical codebook + forged triple-
  backend training: the trained W51 stack cannot recover.
  Strengthens `W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`.

* **W51-L-NO-REAL-KV-CAP** — the W51 stack still does not
  touch transformer-internal KV bytes; persistent state +
  triple-backend translator + branch/cycle memory does not
  change this.

* **W51-L-PURE-PYTHON-TRAINING-COST-CAP** — the pure-Python
  autograd engine carries forward from W47; W51 stage-wise
  training cost is approximately the sum of per-module costs.

* **W51-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** — carries forward
  from W48/W49/W50 — real LLMs may or may not condition on
  the W51 carrier or persistent state. Synthetic anchor is
  load-bearing; H7 is a best-effort anchor on Ollama.

## Per-component verdicts (preview)

* **Persistent shared latent state V3 (M1)** —
  *behaviourally useful* if H2 + H11 + H12 pass;
  *structurally useful* always (chain-walkable state CID).
* **Triple-backend translator (M2)** — *behaviourally useful*
  if H3 + H7 pass under synthetic primary; *structurally
  useful* always (transitivity witness CID).
* **Deep proxy stack V2 (M3)** — *behaviourally useful* if
  H4 + H5 pass on deep composition; *structurally useful*
  always (per-layer + per-head CIDs).
* **Hierarchical compression V3 (M4)** — *behaviourally
  useful* on H15 (12 bits/token); *structurally useful*
  (round-trips through coarse + fine codebook CIDs).
* **Long-horizon reconstruction V3 (M5)** — *behaviourally
  useful* on H13 + H14; *structurally useful* always
  (degradation curve recorded).
* **Branch/cycle memory head (M6)** — *behaviourally useful*
  on H6; *structurally useful* (per-page CIDs).
* **Deep stack overdepth cap** — *limitation reproduces
  honestly*.
* **Rate-floor V2 cap** — *limitation reproduces honestly*.
* **Compromise cap** — *limitation reproduces honestly*.
* **PXBLC distribution cap** — *limitation reproduces
  honestly*.

## Architecture triage

| Frontier candidate                                       | W51 bucket                                              | Verdict |
|---|---|---|
| Persistent GRU-style shared latent state (longer horizon) | **trainable now (persistent state)**                    | shipped |
| Triple-backend translator (transitive)                    | **trainable now (cross-backend transitivity)**          | shipped |
| Cross-tokenizer triple-backend transitivity                | **substrate-blocked**                                   | carry-forward |
| Deeper proxy stack (L=6 + branch/cycle heads)             | **transformer-proxy now (depth + specialisation)**      | shipped |
| Hierarchical coarse-fine codebook                         | **trainable now (compression)**                         | shipped |
| Two-headed reconstruction (causal + branch, max_k=8)      | **trainable now (reconstruction)**                      | shipped |
| Branch/cycle-specialised memory                           | **trainable now (memory)**                              | shipped |
| Triple-backend Ollama realism anchor                      | **anchor probe (best-effort)**                          | shipped (conditional on env) |
| Real KV-cache pooling across turns                        | **substrate-blocked**                                   | unchanged |
| Transformer-internal mixed-curvature attention            | **substrate-blocked**                                   | unchanged |
| True hidden-state / KV sharing                            | **substrate-blocked**                                   | unchanged |
| Multi-host shared-state transfer                          | **substrate-blocked**                                   | unchanged |
| GPU/CUDA-backed autograd                                  | **substrate-blocked (deliberately deferred)**           | unchanged |

## What W51 explicitly does NOT do

* W51 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
  `W48-C-DEEP-TRANSFORMER-COUPLING`,
  `W49-C-DEEP-TRANSFORMER-COUPLING`,
  `W50-C-DEEP-TRANSFORMER-COUPLING`, or any other W43..W50
  substrate-blocked direction.
* W51 does NOT transplant real KV-cache bytes. The branch/
  cycle memory + persistent state operate on capsule-layer
  abstractions only.
* W51 does NOT close
  `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`. The triple-
  backend translator **bounds** the conjecture by training
  capsule-layer transitivity; the conjecture is sharpened
  forward as `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`.
* W51 does NOT claim multi-host coupling.
* W51 does NOT claim training-data-free generalisation.
  Each W51 module is trained on hermetic synthetic banks
  pre-committed in the R-100 / R-101 sources.
* W51 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`,
  `W48-C-REAL-KV-COUPLED-PROXY`, or
  `W48-C-MULTI-HOST-SHARED-STATE`.
* W51 does NOT ship CUDA / GPU support.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`. The
  W51 modules ship at `coordpy.persistent_shared_latent`,
  `coordpy.cross_backend_translator`,
  `coordpy.deep_proxy_stack_v2`,
  `coordpy.hierarchical_compression`,
  `coordpy.long_horizon_retention`,
  `coordpy.branch_cycle_memory`, and `coordpy.w51_team` —
  reachable only through explicit imports — same convention
  as W43..W50.

## New theorem-style claims (preview)

* **W51-T-PERSISTENT-STATE-GRU-SOUNDNESS** (proved by
  inspection + mechanically-checked) — the GRU update rule
  is a deterministic fixed point at trivial parameters
  (update gate = 0 yields `s_t = s_{t-1}`).
* **W51-T-TRIPLE-BACKEND-TRANSITIVITY** (proved-conditional
  + empirical) — under the trained joint loss, the
  transitivity gap between `A→C` and `A→B→C` is bounded.
* **W51-T-DEEP-STACK-V2-L6-EXPRESSIVITY** (proved-
  conditional + empirical) — under the bounded-feature
  assumption + the six-step composition regime, an `L=6`
  deep stack V2 with branch/cycle-specialised heads strictly
  separates a composition that `L=4` cannot.
* **W51-T-HIERARCHICAL-COMPRESSION-RATE-BOUND** (proved-
  conditional + empirical) — under the K1=32 + K2=16
  hierarchical codebook + adaptive emit gate + retention
  floor ≥ 0.85, the bits-per-visible-token ratio ≥ 12.0.
* **W51-T-RECONSTRUCTION-V3-K5-CORRECTNESS** (proved-
  conditional + empirical) — the trained reconstruction V3
  head recovers prior-turn flat features at MSE ≤ 0.30 for
  k ≤ 5.
* **W51-T-RECONSTRUCTION-V3-K8-DROPOFF** (proved-conditional
  + empirical) — at k = 8 the reconstruction MSE is bounded
  ≤ 0.45.
* **W51-T-BRANCH-CYCLE-MEMORY-PAGE-ISOLATION** (proved by
  inspection + mechanically-checked) — the per-branch and
  per-cycle memory pages are content-addressed and disjoint
  across branch / cycle indices.
* **W51-T-W51-OUTER-CID-CHAIN** (proved by inspection +
  mechanically-checked) — `w47_outer → w48_proxy_outer →
  w49_multi_block_outer → w50_outer → w51_outer` chain
  verifies; tamper detected at every link.
* **W51-T-TRIVIAL-PASSTHROUGH-BYTE-IDENTICAL** (proved by
  inspection + empirical) — `W51Team` configured with
  `W51Params.build_trivial()` produces a W51 envelope whose
  internal `w50_outer_cid` field equals the W50 outer CID
  byte-for-byte.
* **W51-T-PERSISTENT-STATE-CHAIN-WALK** (proved by
  inspection + mechanically-checked) — chain-walk depth from
  any `PersistentLatentState` recovers the parent chain up
  to depth 16.
* **W51-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked) — the W51 verifier enumerates 24
  disjoint failure modes; cumulative trust boundary across
  W22..W51 = 367 modes.
* **W51-T-LONG-HORIZON-RETENTION-12TURN** (proved-conditional
  + empirical) — on a length-12 branch path, W51 cosine ≥
  0.90.
* **W51-T-LONG-HORIZON-RETENTION-16TURN-STRETCH** (proved-
  conditional + empirical) — on a length-16 branch path,
  W51 cosine ≥ 0.75 (honest about drop-off).
* **W51-T-COMPRESSION-DEGRADATION-CURVE-MONOTONICITY**
  (proved by inspection + empirical) — under decreasing
  token budgets `{8, 4, 2, 1}`, the achieved bits/token does
  not collapse to chance.

* **W51-L-TRIVIAL-W51-PASSTHROUGH** (proved by inspection +
  empirical) — trivial W51 = W50 byte-for-byte.
* **W51-L-CROSS-TOKENIZER-TRIPLE-CAP** (carries forward,
  sharpened from `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`) —
  when the Ollama realism probe is skipped, the triple-
  backend cross-tokenizer conjecture remains carried forward.
* **W51-L-CROSS-BACKEND-TRANSLATOR-COMPROMISE-CAP** (proved-
  conditional limitation) — adversarial triple-backend
  forgery cannot be recovered.
* **W51-L-DEEP-STACK-OVERDEPTH-CAP** (proved by inspection
  + empirical) — `L=6` does not strictly improve over `L=4`
  on shallow regimes.
* **W51-L-LONG-HORIZON-DROPOFF-CAP** (proved-conditional
  limitation) — at `k > 8` the V3 reconstruction MSE rises.
* **W51-L-COMPRESSION-RATE-FLOOR-V2-CAP** (proved-
  conditional limitation) — 20-bit target exceeds hierarchical
  codebook capacity.
* **W51-L-PXBLC-DISTRIBUTION-CAP** (proved-conditional
  limitation, strengthens W50) — adversarial all-channel
  forgery + W51-specific forgery: cannot recover.
* **W51-L-NO-REAL-KV-CAP** (carries forward, strengthens
  W50) — persistent state + triple-backend translator +
  branch/cycle memory does not transplant real KV bytes.
* **W51-L-PURE-PYTHON-TRAINING-COST-CAP** (carries forward) —
  stage-wise W51 training cost is the sum of per-module costs.
* **W51-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (carries forward
  from W48/W49/W50).

* **W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY** (sharper
  than `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`) — capsule-
  layer transitivity is now trained and auditable; tokenizer-
  level transitivity remains carry-forward.
* **W51-C-DEEP-TRANSFORMER-COUPLING** (carries forward,
  bounds W47/W48/W49/W50 further) — full transformer-
  internal hidden-state + KV-cache coupling remains
  substrate-blocked.
* **W51-C-REAL-KV-COUPLED-PROXY** (carries forward W48-C
  unchanged).
* **W51-C-MULTI-HOST-SHARED-STATE** (carries forward W48-C
  unchanged).
* **W51-C-LIVE-MULTI-HOST-AUTOGRAD** (carries forward W47-C
  unchanged).

## What this enables for the programme

* **Strengthens** every W47..W50 carry-forward by adding a
  persistent shared latent state V3 + triple-backend
  translator + deeper L=6 proxy stack + hierarchical
  compression + long-horizon reconstruction V3 + branch/
  cycle memory head — six orthogonal advances.
* **Strengthens** `W50-L-MULTI-BLOCK-DISTRIBUTION-CAP` to
  include W51-specific forgery
  (`W51-L-PXBLC-DISTRIBUTION-CAP`).
* **Sharpens** `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER` to
  `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY` — the
  triple-backend translator + transitivity loss is now
  trained and auditable; only tokenizer-level transitivity
  remains carried forward.
* **Preserves** all of W43..W50's deterministic-audit
  properties — the W51 modules are strictly additive.
* **Does not close** the substrate-blocked W43..W50
  conjectures. The honest summary: W51 is the strongest
  *executable proxy* available without substrate access,
  with a triple-backend realism anchor when reachable.

## Done = the following commits land

1. `coordpy/persistent_shared_latent.py` — pure Python / stdlib.
2. `coordpy/cross_backend_translator.py` — pure Python / stdlib.
3. `coordpy/deep_proxy_stack_v2.py` — pure Python / stdlib.
4. `coordpy/hierarchical_compression.py` — pure Python / stdlib.
5. `coordpy/long_horizon_retention.py` — pure Python / stdlib.
6. `coordpy/branch_cycle_memory.py` — pure Python / stdlib.
7. `coordpy/w51_team.py` — pure Python / stdlib (composition).
8. `coordpy/r100_benchmark.py` — dependency-free benchmark
   family (10 families).
9. `coordpy/r101_benchmark.py` — dependency-free benchmark
   family (8 families).
10. `tests/test_persistent_shared_latent_w51.py`
11. `tests/test_cross_backend_translator_w51.py`
12. `tests/test_deep_proxy_stack_v2_w51.py`
13. `tests/test_hierarchical_compression_w51.py`
14. `tests/test_long_horizon_retention_w51.py`
15. `tests/test_branch_cycle_memory_w51.py`
16. `tests/test_w51_trivial_passthrough_byte_identical.py`
17. `tests/test_w51_team_envelope_chain.py`
18. `tests/test_r100_benchmark.py`
19. `tests/test_r101_benchmark.py`
20. `examples/w51_smoke_driver.py`
21. `examples/w51_replay_live.py` (Ollama best-effort triple
    anchor)
22. `docs/RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md` and
    this success-criterion file.
23. Updates to `docs/RESEARCH_STATUS.md`,
    `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
    `docs/HOW_NOT_TO_OVERSTATE.md`,
    `docs/context_zero_master_plan.md`,
    `papers/context_as_objects.md`, and `CHANGELOG.md`.
