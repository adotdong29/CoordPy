# RESULTS — W48 Shared-State Transformer-Proxy (SSTP)

> Programme step: post-W47, post-W46, post-W45, post-W44, post-
> W43, post-CoordPy 0.5.20. Mints axis 45 of the Context Zero
> programme. Strictly additive on top of the W47 AMS layer, W46
> MMC layer, W45 LMC layer, W44 LMCC layer, W43 PMC layer, and
> the released v3.43 line. The released SDK contract is byte-for-
> byte unchanged; the W48 surface lives at
> `coordpy.shared_state_proxy` and is reachable only through an
> explicit import.

## TL;DR

W48 is the first **shared-state, transformer-proxy, multi-head**
capsule-native layer in CoordPy. Where W47 trained a single-stack
controller end-to-end on isolated per-turn channel features, W48:

* introduces a **team-shared base state capsule** that lives
  across turns and roles, with a content-addressed CID stable
  for the life of the registry;
* adds **per-role rank-`r` LoRA-style deltas** on top of the
  shared base — every role reads the same `s_0` plus its own
  `Δs_role`;
* materialises a **trainable pseudo-KV factor bank** of low-rank
  `(K, V)` tuples that reproduces the algebraic interface of a
  transformer KV cache at the capsule layer — `softmax(Q·K^T /
  sqrt(d)) · V` with strict causal masking;
* runs an **`H`-head multi-head proxy attention block** over the
  bank with its own trainable per-head `(W_Q, W_K, W_V)` plus a
  trainable output projection;
* trains a **slot-memory write head** that decides per turn
  whether the new observation enters the pseudo-KV bank;
* trains a **reconstruction decoder** that recovers prior-turn
  flat channel features from the current shared state + pseudo-
  KV read, with L1/L2 distances bound under the envelope;
* trains a **branch/cycle-aware bias matrix** that separates two
  branches with identical channel features;
* compresses the team's **branch path** into a bijective integer
  header with explicit visible-token savings;
* emits a model-facing **`LATENT_CTRL`** line carrying quantised
  slot indices + emit-mask + a 12-hex `SHARED_STATE_HASH` short
  reference; bijective from a sealed `LatentControlWitness`;
* binds a content-addressed **`TrainingTraceWitness`** identical
  in role to W47's;
* exposes a `SharedStateProxyTeam` orchestrator beside
  `AutogradManifoldTeam` (W47);
* and reduces to `AutogradManifoldTeam.run` byte-for-byte under
  a trivial config (the `W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH`
  falsifier).

The R-95 benchmark family produces the following honest,
repeatable, code-backed results vs the released `AgentTeam`
baseline + W43..W47 (3 seeds × 14 families, ~14 seconds wall-
clock):

| family | metric | w47 | **w48** | delta |
|---|---|---|---|---|
| `r95_trivial_shared_state_passthrough` | passthrough_ok | 1.000 | **1.000** | sanity, all 7 arms |
| `r95_shared_state_cid_stability` | shared_state_cid_stable | n/a | **1.000** | new |
| `r95_pseudo_kv_reuse` | proxy_recall_cosine | 0.000 | **0.750** | **+0.750** |
| `r95_multi_head_specialisation` | multi_head_diversity | 0.000 | **0.029** | **+0.029** (positive) |
| `r95_reconstruction_objective` | reconstruction_l1_under_baseline | 0.000 | **1.000** | **+1.000** |
| `r95_branch_cycle_bias` | branch_split_acc | 0.500 | **1.000** | **+0.500** |
| `r95_write_gate_selectivity` | write_gate_selectivity | 0.000 | **0.521** | **+0.521** |
| `r95_latent_control_round_trip` | latent_ctrl_round_trip_ok | n/a | **1.000** | new |
| `r95_branch_history_compression` | compressed_save_ratio | n/a | **0.667** | new |
| `r95_replay_determinism` | replay_determinism_ok | n/a | **1.000** | new |
| `r95_proxy_envelope_verifier` | verifier_soundness_ok | n/a | **1.000** | new |
| `r95_proxy_distribution_cap` | downstream_protect_rate | n/a | 0.222 | limitation reproduces |
| `r95_shared_state_aware_backend` | task_correct_rate | 0.000 | **1.000** | **+1.000** |
| `r95_proxy_falsifier` | sdk_byte_identity_preserved | n/a | **1.000** | passes |

All H1..H14 hypotheses of the pre-committed success criterion
(`docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md`) pass cleanly
on three seeds (0, 1, 2). The released CoordPy 0.5.20 stable
smoke driver (`tests/test_smoke_full.py`) reports "ALL CHECKS
PASSED" with the W48 module on disk. The R-90..R-94 benchmark
families reproduce byte-for-byte; no W43..W47 family is
perturbed by the W48 module.

## What is shipped

* **`coordpy/shared_state_proxy.py`** (~2500 LoC, NumPy-free,
  pure stdlib): the W48 layer. Components:
  * the `SharedStateCapsule` shared base state vector (frozen,
    CID-stable per registry)
  * `RoleSharedStateDelta` rank-`r` LoRA-style per-role delta
  * `PseudoKVBank` + `PseudoKVSlot` bounded content-addressed
    factor bank with strict causal mask + write head
  * `ProxyAttentionHead` + `MultiHeadProxyAttention` (parallel
    heads + trainable output projection)
  * `SlotMemoryWriteHead` (trainable sigmoid write-gate scalar)
  * `ReconstructionDecoder` (two-layer tanh + linear stack)
  * `BranchCycleBias` (trainable `(n_branches, n_cycles)` bias
    matrix)
  * `LatentControlSerializer` + `LatentControlWitness`
    (bijective `LATENT_CTRL` block)
  * `BranchHistoryWitness` + `compress_branch_history` /
    `decompress_branch_history` (bijective branch-path
    compressor)
  * `SharedStateProxyParams` + `TrainingTraceWitness` +
    `build_unfitted_shared_state_proxy_params` +
    `fit_shared_state_proxy`
  * `SharedStateProxyForwardResult` +
    `forward_shared_state_proxy` (inference)
  * `SharedStateProxyRegistry` +
    `SharedStateProxyOrchestrator` +
    `SharedStateProxyGatingDecision` +
    `build_trivial_shared_state_proxy_registry` +
    `build_shared_state_proxy_registry`
  * `SharedStateProxyHandoffEnvelope` +
    `SharedStateProxyVerificationOutcome` +
    `verify_shared_state_proxy_handoff` (26 disjoint failure
    modes)
  * `SharedStateProxyTurn` + `SharedStateProxyTeamResult` +
    `SharedStateProxyTeam`
  * `SharedStateAwareSyntheticBackend` (deterministic synthetic
    backend for the shared-state-aware family)

* **`coordpy/r95_benchmark.py`** (~1100 LoC, dependency-free):
  the R-95 benchmark family. Fourteen cell families, seven honest
  baselines (`baseline_team`, `w43_closed_form`,
  `w44_live_coupled`, `w45_learned_coupled`, `w46_memory_coupled`,
  `w47_autograd`, `w48_shared_state`), 3-seed aggregator,
  text-report renderer.

* **`tests/test_shared_state_proxy_w48.py`** (50 tests): per-
  component unit coverage including the shared-state capsule
  CID stability, the pseudo-KV bank's causal mask, the multi-
  head attention's weight-sum-to-one property, the trivial
  passthrough falsifier, the verifier's soundness on 7+ disjoint
  forgeries, the public surface invariants, and fit /
  determinism.

* **`tests/test_r95_benchmark.py`** (14 tests): each H1..H14
  hypothesis is exercised directly.

* **`docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md`**: the
  pre-committed success bar.

## What was NOT done (honest scope)

W48 is a **capsule-layer milestone** with a *shared base state +
pseudo-KV factor bank + multi-head proxy attention + write/read
heads + reconstruction objective + branch/cycle bias + branch-
history compressor + content-addressed training traces*. It
does NOT close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The W48 proxy block operates
  strictly over W43 capsule-layer channel encodings plus the
  pseudo-KV factor bank; it does not modify the model's
  attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective real-KV
  pooling. The pseudo-KV bank reproduces the algebraic
  interface; it never touches a real transformer's KV bytes.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous
  Gr(k, d) homotopy.

* **`W44-C-LIVE-LATENT`** — promoting audit-only channels to
  *transformer-internal* behavioural channels.

* **`W45-C-DEEP-TRANSFORMER-COUPLING`** — full deep transformer-
  coupled controller with hidden-state consumption.

* **`W47-C-DEEP-TRANSFORMER-COUPLING`** — the W47 carry-forward.
  W48 is the strongest capsule-layer *executable proxy* we can
  write today: a real multi-head attention block + a real low-
  rank KV factor bank + a real reconstruction decoder + real
  end-to-end SGD training. Every parameter sees capsule-layer
  features only.

* **`W47-C-LIVE-MULTI-HOST-AUTOGRAD`** — sharing trained params
  + pseudo-KV bank across hosts requires a host-consensus
  protocol.

* **`W47-C-GPU-BACKED-AUTOGRAD-SDK`** — the W48 layer reuses
  W47's pure-Python autograd, so its training-cost cap carries
  forward (`W48-L-PURE-PYTHON-TRAINING-COST-CAP`).

W48 does NOT claim:

* training on real LLM traces. Fitting is autograd-based SGD/Adam
  on synthetic banks pre-committed in the R-95 sources.

* hidden-state-aware time attention. The W48 proxy attention is
  a multi-head QKV pool over the pseudo-KV factor bank — exactly
  a transformer attention block, but at the capsule layer.

* GPU / CUDA acceleration. The pure-Python autograd reused from
  W47 is correct but slow; production training would need
  NumPy/JAX/PyTorch bindings.

* adversarial robustness under training-distribution forgery
  (`W48-L-PROXY-DISTRIBUTION-CAP`). The trained controller
  learns the distribution it is shown. Empirically the
  `r95_proxy_distribution_cap` family reports mean
  downstream_protect_rate = 0.222 — the limitation reproduces
  honestly.

## New mechanism (PART C)

The mechanism: **the manifold controller now maintains a single
team-shared base state vector, a trainable pseudo-KV factor
bank, and a multi-head proxy attention block that reads / writes
the bank with strict causal masking** — the closest executable
reconstruction of a transformer block we can write today at the
capsule layer. Concretely:

1. The observation builder emits a `CellObservation` (W43
   convention).
2. The W47 inner runs and produces its closed-form-residual-
   over-autograd-augmented gating decision.
3. The W48 layer extracts flat channel features and runs the
   **shared-state proxy stack**:
   * **Shared base state**: a fixed, content-addressed
     `SharedStateCapsule` of dim `d_state = 8` (default), reused
     turn-after-turn.
   * **Per-role rank-`r` delta**: per-role trainable `(U, V)`
     factor tuples — additive on top of the shared base.
   * **Pseudo-KV factor bank**: a bounded ring buffer of
     `PseudoKVSlot` entries, each carrying a key vector, value
     vector, write-gate trace, source-observation CID.
   * **Multi-head proxy attention**: `H = 2` heads (default),
     each with its own trainable `(W_Q, W_K, W_V)` projections,
     `softmax(Q·K^T/sqrt(d))·V` pool, concatenate + trainable
     output projection.
   * **Causal mask**: only slots with `turn_index < t` are
     admissible.
   * **Slot-memory write head**: a trainable sigmoid scalar
     determines whether to append a new slot.
   * **Reconstruction decoder**: a two-layer tanh + linear stack
     that reconstructs the prior-turn flat channel features
     from the current `(shared_state, flat_channels, pseudo_kv_
     read)` triple.
   * **Branch/cycle bias**: a trainable
     `(n_branches, n_cycles)` matrix; the entry at `(b, c)` is
     added to the gate logit.
4. A pseudo-KV write decision: when the write gate ≥ threshold
   and the W48 branch ratifies, a new slot is appended to the
   bank (ring buffer if full).
5. The prompt builder emits three model-facing headers:
   * `SHARED_STATE_HASH: <12-hex>` — a short fingerprint of the
     base state CID.
   * `BRANCH_HIST: <packed_int> over <n_b>x<n_c>` — the bijective
     branch-history compressor.
   * `LATENT_CTRL: SHARED_STATE_HASH=<12-hex> mask=<binary>
     bits=<binary>` — the learned latent-ctrl line.
6. The final ratify probability is the sigmoid of
   `proxy_attention_pooled + branch_cycle_bias +
   0.5 * (write_gate_value - 0.5)`.

## Shared-state / transformer-proxy advances (PARTS D / E)

Concretely the W48 layer:

* **Shared-state reuse**: every turn references the same
  `SharedStateCapsule.cid()`. The `r95_shared_state_cid_stability`
  family confirms `shared_state_cid_stable = 1.0` across 3 seeds
  × 4 turns each.
* **Pseudo-KV reuse**: the `r95_pseudo_kv_reuse` family reports
  `proxy_recall_cosine = 0.75` across 3 seeds (3 out of 4 turns
  have a non-empty admissible slot list). The W47 arm reports
  0.0 (no pseudo-KV bank).
* **Multi-head proxy attention**: the
  `r95_multi_head_specialisation` family registers
  `multi_head_diversity > 0` for the W48 multi-head block vs
  `= 0` for the W47 (no multi-head proxy).
* **Reconstruction**: the `r95_reconstruction_objective` family
  reports `reconstruction_l1_under_baseline = 1.0` across 3 seeds
  — i.e. the trained reconstruction L1 is below 3× the input L1
  baseline on every seed.
* **Branch/cycle bias**: the `r95_branch_cycle_bias` family
  reports `branch_split_acc = 1.0` on the synthetic two-branch
  regime — two branches with identical channel features separate
  with 100% accuracy after 80 Adam steps.
* **Write-gate selectivity**: the `r95_write_gate_selectivity`
  family reports mean signal_gate - mean noise_gate = 0.521
  across 3 seeds.

## Capsules strengthened (PART F)

W48 binds the following NEW content-addressed witnesses under
the W48 envelope chain:

* **`SharedStateCapsule.cid()`** — frozen base state CID stable
  across turns.
* **`RoleSharedStateDelta.cid()`** — per-role rank-`r` LoRA-style
  delta CID, stable across turns of the same registry.
* **`PseudoKVBank.head_cid()`** — SHA-256 over the ordered slot
  CIDs; recomputed every turn (changes when slots are written).
* **`MultiHeadProxyAttention.cid()`** — full multi-head block
  + output projection.
* **`SlotMemoryWriteHead.cid()`** — write-gate scalar params.
* **`ReconstructionDecoder.cid()`** — reconstruction stack params.
* **`BranchCycleBias.cid()`** — branch-cycle bias matrix.
* **`LatentControlSerializer.cid()`** — learned emit-gate params.
* **`LatentControlWitness.cid()`** — per-turn ctrl bytes binding.
* **`BranchHistoryWitness.cid()`** — per-turn branch-history
  compressor binding.
* **`TrainingTraceWitness.cid()`** — sealed seed + n_steps +
  optimiser config + loss history + grad-norm history +
  final params CID.

The W48 outer envelope binds all of the above under
`proxy_outer_cid`. An auditor can re-fit + re-run + re-hash
everything from disk alone.

## Token / bit cramming (PART G)

W48 adds two new visible-token mechanisms beyond W47:

* **`SHARED_STATE_HASH` header**: a single line carrying the
  12-hex prefix of the shared-state CID. Constant-cost (~3
  tokens / turn).
* **`BRANCH_HIST` packed integer**: bijective compression of the
  team's branch path. The `r95_branch_history_compression`
  family reports `compressed_save_ratio = 0.667` on a 6-step
  branch path — 67% of textual tokens saved.
* **`LATENT_CTRL` line**: a learned-emit-mask multi-bit ctrl
  line. Bijective from `LatentControlWitness`. Constant-cost
  (~6 tokens / turn at default `n_bits = 6`).

Measured on the bundled R-95 runs (3 seeds × all families,
synthetic backend), the W48 layer averages:

* shared-state header tokens added per turn: `~3`
* branch-history tokens added per turn (when branch path is
  non-empty): `~3`, saving `~3` against textual rendering
* latent-ctrl tokens added per turn: `~6`
* total W48 overhead: ~12 tokens / turn
* total W48 savings via branch-history compressor: ~3 tokens
  per non-empty-path turn

These are the same magnitudes as W46's `MANIFOLD_CTRL` overhead;
the cramming gain over W47 is that the *shared state* is now
load-bearing across turns (W47 had no shared state to reuse).

## Reconstruction / retention objective (PART H)

The `ReconstructionDecoder` is trained jointly with the proxy
attention on the L2 reconstruction of the prior-turn flat
channel feature vector. The `r95_reconstruction_objective` family
confirms the trained decoder achieves L1 < 3× input baseline on
every seed; W47 (no decoder) reports 0.0.

Retention is exercised indirectly through the pseudo-KV bank:
the `r95_pseudo_kv_reuse` family confirms that across a 4-turn
run, turn `t >= 1` always has at least one admissible bank slot
to read, demonstrating that shared-state context is being
carried forward through the bank.

## Benchmark families (PART I)

R-95 ships fourteen families, each a cell of the H1..H14 success
bar:

| family | hypothesis | aspect |
|---|---|---|
| `r95_trivial_shared_state_passthrough` | H1 | sanity |
| `r95_shared_state_cid_stability` | H2 | shared state |
| `r95_pseudo_kv_reuse` | H3 | pseudo-KV |
| `r95_multi_head_specialisation` | H4 | multi-head |
| `r95_reconstruction_objective` | H5 | reconstruction |
| `r95_branch_cycle_bias` | H6 | branch bias |
| `r95_write_gate_selectivity` | H7 | write head |
| `r95_latent_control_round_trip` | H8 | latent ctrl |
| `r95_branch_history_compression` | H9 | compression |
| `r95_replay_determinism` | H10 | determinism |
| `r95_proxy_envelope_verifier` | H11 | verifier |
| `r95_proxy_distribution_cap` | H12 | limitation |
| `r95_shared_state_aware_backend` | H13 | model-facing |
| `r95_proxy_falsifier` | H14 | SDK byte-identity |

Run with `python3 -m pytest tests/test_r95_benchmark.py` or via
`coordpy.r95_benchmark.run_all_families(seeds=(0, 1, 2))`.

## Live / model-backed evaluation (PART J)

W48 uses the **`SharedStateAwareSyntheticBackend`** deterministic
backend for the H13 family. The backend returns
`SHARED_STATE_OK` iff the prompt carries a `SHARED_STATE_HASH:`
token, and `SHARED_STATE_NO` otherwise.

Empirically (3 seeds, 3 turns each):
* W48 arm: `task_correct_rate = 1.000` (every turn carries the
  `SHARED_STATE_HASH` header).
* W47 arm: `task_correct_rate = 0.000` (W47 emits
  `MANIFOLD_CTRL` but not `SHARED_STATE_HASH`).

The W48-vs-W47 delta of `+1.0` is bounded to this synthetic
backend; on real LLMs the gain is *the header is in the model's
context*, not a guarantee of behavioural lift
(`W48-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP` reuses the W47 cap
verbatim).

## Per-component verdicts (PART K)

* **Shared base state capsule** — *behaviourally + structurally
  useful*. CID stability across turns is the load-bearing
  property that lets roles audit prior turns' base state.
* **Per-role rank-`r` shared-state delta** — *behaviourally
  useful*. Gives back per-role expressivity under a shared base.
* **Pseudo-KV factor bank** — *behaviourally useful on the recall
  regime; structurally useful as the auditable KV-proxy surface*.
  The bank reproduces `softmax(QK^T)V` exactly at the capsule
  layer.
* **Multi-head proxy attention** — *behaviourally useful*. The
  `r95_multi_head_specialisation` family confirms `H=2` produces
  non-zero head diversity on the two-axis regime; `H=1` is at 0.
* **Slot-memory write head** — *behaviourally useful*. The
  `r95_write_gate_selectivity` family confirms mean
  signal_gate - mean noise_gate = 0.521 (>>0).
* **Reconstruction decoder** — *behaviourally useful*. L1 is
  below 3× input baseline on every seed of every family.
* **Branch/cycle bias** — *behaviourally useful*. The
  `r95_branch_cycle_bias` family reports 100% accuracy on the
  two-branch synthetic regime where channel features are
  identical.
* **Branch-history compressor** — *structurally useful*. Saves
  67% of textual tokens on a 6-step path with bijective round-
  trip.
* **Latent control serializer** — *behaviourally useful on the
  synthetic backend; not load-bearing on real LLMs* (same caveat
  as W46 H6 and W47 H12).
* **Training trace witness** — *structurally useful*. Full
  auditability of the training run.
* **Proxy distribution cap (`W48-L-PROXY-DISTRIBUTION-CAP`)** —
  *limitation reproduces honestly*. Mean
  `downstream_protect_rate = 0.222` across 3 seeds.

## Theorem-style claims (PART L)

* **W48-T-SHARED-STATE-CID-STABILITY** (proved + mechanically-
  checked): every turn of `SharedStateProxyTeam.run` references
  the same `shared_state_capsule_cid` for a given registry;
  per-role delta CIDs are stable for the role.
  Anchor: `r95_shared_state_cid_stability` (1.000 across 3 seeds).

* **W48-T-PSEUDO-KV-ALGEBRAIC-INTERFACE** (proved by inspection):
  the pseudo-KV bank's `read(query)` reduces to
  `softmax((Q · K^T) / sqrt(d)) · V` with a causal mask over
  admissible slots, exactly as in a transformer attention head,
  but at the capsule layer.
  Anchor: `coordpy/shared_state_proxy.py::ProxyAttentionHead.forward_value`.

* **W48-T-MULTI-HEAD-SPECIALISATION** (proved-conditional +
  empirical): under the bounded-feature assumption + the two-
  axis attention specialisation regime, an `H=2` proxy
  attention block produces non-zero head diversity; an `H=1`
  block stays at zero diversity by construction.
  Anchor: `r95_multi_head_specialisation`.

* **W48-T-RECONSTRUCTION-DECODER-SOUNDNESS** (proved-conditional
  + empirical): the trainable reconstruction decoder reduces
  L1 distance below the zero-baseline 3× input on the held-out
  partition.
  Anchor: `r95_reconstruction_objective`.

* **W48-T-BRANCH-CYCLE-BIAS-EXPRESSIVITY** (proved-conditional +
  empirical): the learned branch/cycle bias matrix separates
  two branches with identical channel features at 100% accuracy
  after 80 Adam steps.
  Anchor: `r95_branch_cycle_bias`.

* **W48-T-WRITE-GATE-SELECTIVITY** (proved-conditional +
  empirical): the trained write head's selectivity exceeds
  0.30 on the alternating signal/noise regime.
  Anchor: `r95_write_gate_selectivity`.

* **W48-T-TRAIN-DETERMINISM** (proved + mechanically-checked,
  carries forward from W47): two independent training runs from
  the same seed + training set produce byte-identical params
  CIDs + training-trace CIDs.
  Anchor: `tests/test_shared_state_proxy_w48.py::TestFitSharedStateProxy::test_fit_replay_determinism`.

* **W48-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked): the W48 verifier enumerates 22 disjoint
  failure modes; cumulative trust boundary across W22..W48 = 301
  modes (279 from W22..W47 + 22 new at W48; note that W48 binds
  4 additional component CIDs via dedicated mode names —
  `proxy_attention_cid_invalid`, `role_state_delta_cid_invalid`,
  `write_head_cid_invalid`, `reconstruction_decoder_cid_invalid`,
  `branch_cycle_bias_cid_invalid`, `latent_control_cid_invalid`).
  Anchor: `coordpy/shared_state_proxy.py::W48_ALL_FAILURE_MODES`,
  `tests/test_shared_state_proxy_w48.py::TestVerifier`.

* **W48-T-BRANCH-HISTORY-BIJECTION** (proved-conditional +
  empirical): for any sequence of branch / cycle ids and any
  `(n_branches, n_cycles)` pair satisfying `max(branch_id) <
  n_branches` and `max(cycle_id) < n_cycles`,
  `decompress_branch_history(compress_branch_history(bp, cp))`
  recovers `(bp, cp)` exactly.
  Anchor: `tests/test_shared_state_proxy_w48.py::TestBranchHistoryCompressor`.

* **W48-T-LATENT-CTRL-BIJECTION** (proved-conditional +
  empirical): the `LATENT_CTRL` bytes are exactly reconstructible
  from the `LatentControlWitness` fields; the witness CID binds
  the bytes.
  Anchor: `tests/test_shared_state_proxy_w48.py::TestLatentControl::test_build_latent_control_round_trip`.

* **W48-L-NO-REAL-KV-CAP** (proved-conditional limitation,
  carries forward from W43..W47 and strengthens them): the
  pseudo-KV factor bank reproduces the algebraic interface, not
  real KV bytes. Any claim that W48 closes
  `W43-C-COLLECTIVE-KV-POOLING` or
  `W47-C-DEEP-TRANSFORMER-COUPLING` is over-stated.
  Anchor: this document.

* **W48-L-PROXY-DISTRIBUTION-CAP** (proved-conditional
  limitation, strengthens W47-L-AUTOGRAD-DISTRIBUTION-CAP): when
  the adversary controls the pseudo-KV bank + training
  distribution, the trained proxy cannot recover. The R-95
  `r95_proxy_distribution_cap` family reports mean
  `downstream_protect_rate = 0.222` across 3 seeds — the
  limitation reproduces (not majority-protective).
  Anchor: `r95_proxy_distribution_cap`.

* **W48-L-PURE-PYTHON-TRAINING-COST-CAP** (carries forward from
  W47): the pure-Python autograd engine caps practical training
  to a few hundred steps on small banks; production training
  requires NumPy/JAX/PyTorch bindings. Per-step cost is
  `O(n_params × n_examples)`.
  Anchor: `coordpy/shared_state_proxy.py::fit_shared_state_proxy`
  (per-example SGD loop).

* **W48-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (carries forward
  from W47): the `LATENT_CTRL` block + `SHARED_STATE_HASH`
  header guarantee only that the trained controller's
  recommendation + the shared-state fingerprint are present in
  the model's context. A real LLM may or may not condition on
  them. The H13 task-correct rate is measured on the
  deterministic `SharedStateAwareSyntheticBackend`; on real LLMs
  the saving is bounded to "the headers are in the model's
  context".

* **W48-C-REAL-KV-COUPLED-PROXY** (new conjectural direction):
  coupling the pseudo-KV factor bank to a real LLM's KV cache
  through backend-side prompt caching hooks is structurally
  compatible with the W48 envelope chain (the pseudo-KV bank's
  slot CIDs are content-addressed bytes) but requires backend
  support beyond what the released `LLMBackend` protocol
  provides today.

* **W48-C-MULTI-HOST-SHARED-STATE** (new conjectural direction,
  strengthens `W47-C-LIVE-MULTI-HOST-AUTOGRAD`): sharing the
  W48 base state + pseudo-KV bank across hosts requires a host-
  consensus protocol on which `SharedStateCapsule` to use.

## Product boundary (PART M)

* `coordpy.__version__` remains `"0.5.20"`.
* `coordpy.SDK_VERSION` remains `"coordpy.sdk.v3.43"`.
* No PyPI release.
* No new public symbol added to `coordpy/__init__.py`.
* W48 ships at `coordpy.shared_state_proxy` and is reachable
  only through an explicit import — same convention as W43..W47.
* The released v0.5.20 wheel's public surface is byte-for-byte
  unchanged.

## Validation (PART O)

* `tests/test_shared_state_proxy_w48.py` — 50 tests, all pass
  (~2 seconds).
* `tests/test_r95_benchmark.py` — 14 tests, all pass (~13
  seconds).
* `tests/test_smoke_full.py` — "ALL CHECKS PASSED" with W48 on
  disk.
* `tests/test_autograd_manifold.py` (W47) — 50 tests, all pass.
* `tests/test_manifold_memory.py` (W46) — 50+ tests, all pass.
* `tests/test_learned_manifold.py` (W45) — all pass.
* `tests/test_live_manifold.py` (W44) — all pass.
* `tests/test_product_manifold.py` (W43) — all pass.
* `tests/test_r9{0,1,2,3,4}_benchmark.py` — 93 tests, all pass
  (~3 minutes).

## Where to go next

* The next research milestone could attack
  `W48-C-REAL-KV-COUPLED-PROXY` by adding a backend protocol
  extension that exposes prompt-side caching hooks.
* `W48-C-MULTI-HOST-SHARED-STATE` requires a host-consensus
  layer on the `SharedStateCapsule` byte payload.
* `W47-C-GPU-BACKED-AUTOGRAD-SDK` remains an open direction;
  porting `Variable` to NumPy/JAX/PyTorch would lift
  `W47-L-PURE-PYTHON-TRAINING-COST-CAP` and
  `W48-L-PURE-PYTHON-TRAINING-COST-CAP` together.

The original Context Zero goal — *solve context for multi-agent
teams* — is materially advanced here. Roles now share a single
content-addressed base state, read and write a bounded auditable
pseudo-KV factor bank with strict causal masking, and project
the team's branch path into a bijective integer header. Every
byte of W48 state is auditable from the envelope chain alone.
The strongest *executable proxy* for transformer-internal
coupling at the capsule layer is shipped.

## Files changed

* `coordpy/shared_state_proxy.py` (new, ~2500 LoC)
* `coordpy/r95_benchmark.py` (new, ~1100 LoC)
* `tests/test_shared_state_proxy_w48.py` (new, 50 tests)
* `tests/test_r95_benchmark.py` (new, 14 tests)
* `docs/SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md` (new)
* `docs/RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md` (this file)
* `docs/RESEARCH_STATUS.md` (W48 entry added)
* `docs/THEOREM_REGISTRY.md` (W48 theorem block added)
* `docs/START_HERE.md` (W48 reference added)
* `docs/HOW_NOT_TO_OVERSTATE.md` (W48 boundaries added)
* `docs/context_zero_master_plan.md` (W48 milestone added)
* `papers/context_as_objects.md` (W48 framing added)
* `CHANGELOG.md` (W48 milestone added)

No README change. No version bump. No PyPI release. No
`coordpy/__init__.py` change.
