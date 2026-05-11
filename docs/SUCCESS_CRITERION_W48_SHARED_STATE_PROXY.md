# Pre-committed success criterion — W48 Shared-State Transformer-Proxy

> Programme step: post-W47. Mints axis 45 of the Context Zero
> programme. Strictly additive on top of W47 AMS, W46 MMC, W45
> LMC, W44 LMCC, W43 PMC, and the released v3.43 line. Honest
> scope: the W48 mechanism is the first capsule-native CoordPy
> layer where the controller maintains a **shared state across
> turns and roles** that is read and written via a **multi-head,
> causally-masked, autograd-trained proxy attention head** with
> a **trainable pseudo-KV low-rank factor bank** standing in for
> the parts of a real transformer KV cache that are not exposed
> at the capsule layer. It does NOT touch transformer-internal
> hidden state, real KV cache bytes, attention weights, or
> embeddings — those remain substrate-blocked. W48 is the
> strongest honest proxy line we can write today; it is *not*
> a closure of `W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`, or
> `W47-C-DEEP-TRANSFORMER-COUPLING`.

## Mechanism

W48 introduces the **Shared-State Transformer-Proxy (SSTP)** —
the first capsule-native CoordPy layer where a single
**team-shared state vector**, a **trainable pseudo-KV factor
bank**, and a **multi-head proxy attention block** are read /
written by every role on every turn and bound under the
existing W47 envelope chain. All of it is pure-Python /
stdlib; no NumPy, no PyTorch, no JAX dependency. The pure-
Python reverse-mode autograd engine from W47 (`Variable` +
`AdamOptimizer`) is reused. The released SDK v3.43 contract
remains byte-for-byte unchanged.

The eleven trainable, content-addressed components of W48 are:

* **Shared base state capsule.** A single, deterministically-
  initialised hidden-state vector `s_0 ∈ R^{d_state}` that lives
  on the team and is reused turn-after-turn across roles. Bound
  by a content-addressed `SharedStateCapsule` (CID seals
  shape + values + init seed) that is the first reachable W48
  capsule per run.

* **Per-role rank-`r` shared-state delta.** A trainable LoRA-
  style low-rank correction `Δs_role = U_role · V_role^T` applied
  to `s_0` on each turn for that role's read; **roles never share
  the delta**, but they share the base `s_0` and the read/write
  policy. Reduces the "every role re-bootstraps" problem
  observed at W44..W47.

* **Trainable pseudo-KV factor bank.** A bounded, content-
  addressed ring buffer of **low-rank factor tuples** `(K_i, V_i)`
  with `K_i ∈ R^{n_slots × d_factor}`, `V_i ∈ R^{n_slots ×
  d_factor}`. The bank is updated by a learned **write head**
  (read attention with a write gate) on every ratified turn and
  read by a learned **read head** on every turn. The bank is
  **NOT** a real KV cache; it is a *pseudo-KV factor bank* that
  reproduces the algebraic interface (low-rank `softmax(QK^T)V`)
  at the capsule layer with byte-deterministic provenance.

* **Multi-head proxy attention block.** An `H`-head, depth-`L_p`
  trainable attention block over a stacked input
  `[s_t || flat_channels_t || pseudo_kv_read_t]`. Each head
  has its own trainable `(W_Q, W_K, W_V)`. Heads run in
  parallel; outputs are concatenated and projected. A
  trainable residual stream + LayerNorm-style scaling makes the
  block behave like a single transformer block at the capsule
  layer.

* **Causal mask + branch/cycle-aware bias.** The proxy
  attention block uses a causal mask over the pseudo-KV bank
  (only entries with `turn_index < t` are admissible) and an
  additive **branch/cycle-aware bias matrix** that learns a
  scalar correction for each `(branch_id, cycle_id)` pair —
  the first capsule-layer attention bias that explicitly
  encodes branch/cycle structure.

* **Slot-memory write head with read/write gate.** A trainable
  sigmoid write-gate decides at every turn whether the new
  observation should be appended to the pseudo-KV bank, modify
  an existing slot, or be discarded. Read/write gates are
  bound under a `SlotMemoryHeadWitness` CID.

* **Reconstruction objective.** A trainable decoder head
  reconstructs the previous turn's **flat channel feature
  vector** from the current shared state + the read pseudo-KV
  read. Trained jointly with the gate via L2 reconstruction.
  Per-turn reconstruction L1 / L2 distances are bound under the
  envelope; reconstruction quality is a load-bearing metric in
  the W48 success bar.

* **Branch-history compressor.** Branch-aware packed bytes
  derived from the team's branch path so far, with bijective
  decode. Two metrics fall out: how many visible tokens the
  branch-history compressor saves vs the textual rendering,
  and how well the reconstruction objective recovers the
  history.

* **Latent control serializer.** A learned, *visible-token-
  budget-bounded* serializer over the proxy attention block's
  output. Emits a single line `LATENT_CTRL: ...` whose bits
  carry quantised slot indices + write-gate bits + branch-bias
  bits. Bijective from the witness CID.

* **Training trace witness.** Identical envelope role to W47's
  `TrainingTraceWitness`. Records seed, n_steps, optimiser
  config, loss history, gradient norm history, final params
  CID, divergence flag. Bound under the W48 envelope.

* **Autograd training pipeline.** A `fit_shared_state_proxy`
  trainer that:
  * trains the per-head proxy attention by Adam SGD on the
    binary cross-entropy gate loss;
  * trains the read/write head on a synthetic memory-
    completion loss;
  * trains the reconstruction decoder on flat-feature L2 loss;
  * trains the branch/cycle bias on a synthetic branching
    regime;
  * trains the latent control serializer on a learned-bits-
    per-target-mask loss.
  All under the shared `AdamOptimizer` from W47.

Honest scope (do-not-overstate)
-------------------------------

W48 does NOT touch transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. Every
parameter of the shared-state proxy operates over W43
capsule-layer encodings, the W47 trainable channel features,
and the pseudo-KV factor bank's *capsule-layer* slots. The
**pseudo-KV bank reproduces the algebraic interface** of a
KV cache (low-rank `Q K^T V` with a causal mask and a
write head) at the capsule layer; it does **not** transplant
real KV state from a transformer's attention layers. The W43
substrate-blocked conjectures
(`W43-C-MIXED-CURVATURE-LATENT`,
`W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) and the W47 carry-forward
`W47-C-DEEP-TRANSFORMER-COUPLING` are unchanged.

W48 does NOT claim full-precision parity with a real
transformer block. The trainable proxy block uses tanh
nonlinearities, a single-head-depth-equivalent capsule-layer
attention pool, and trains on synthetic regimes. It is the
strongest honest *executable proxy* we can write at the
capsule layer.

W48 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine reused from W47 is correct but slow; W48 trains
on bounded synthetic banks (~32–64 examples for ~120–200 Adam
steps).

W48 does NOT claim adversarial robustness under
training-distribution forgery — `W48-L-PROXY-DISTRIBUTION-CAP`
(strengthens `W47-L-AUTOGRAD-DISTRIBUTION-CAP`).

W48 is strictly additive. When configured trivially, the
`SharedStateProxyTeam` orchestrator reduces to
`AutogradManifoldTeam.run` byte-for-byte — the
`W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH` falsifier.

This module lives at `coordpy.shared_state_proxy` and is NOT
exported through `coordpy.__experimental__` at this milestone;
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W48 surface through an
explicit `from coordpy.shared_state_proxy import ...` import.

## H1..H14 success bar

Fourteen pre-committed hypotheses on the R-95 benchmark
family; each is exercised by a per-family test in
`tests/test_r95_benchmark.py` plus per-component unit coverage
in `tests/test_shared_state_proxy_w48.py`.

### H1 — Trivial shared-state passthrough

A trivially-configured `SharedStateProxyRegistry`
(`proxy_enabled=False`, `pseudo_kv_enabled=False`,
W47-trivial inner) reduces to `AutogradManifoldTeam.run`
byte-for-byte. The `r95_trivial_shared_state_passthrough`
family reports `passthrough_ok = 1.0` across all six W43..W47
arms and the new `w48_shared_state` arm across all three
seeds.

### H2 — Shared-state CID stability across turns

For any deterministic team config and any task, every turn of
a `SharedStateProxyTeam.run` references the *same*
`shared_state_capsule_cid` (the base state is reused
unchanged). Per-role delta CIDs are stable for the role
(same role on different turns → same delta CID). The
`r95_shared_state_cid_stability` family;
`shared_state_cid_stable = 1.0` AND
`per_role_delta_cid_stable = 1.0` across all seeds.

### H3 — Pseudo-KV factor bank: reuse beats rebuild

On a multi-turn fact-recall regime where turn `t >= 2` needs
information emitted at turn 0, the W48 pseudo-KV read achieves
mean cosine similarity ≥ 0.85 between the recovered slot and
the true emission, while the W47 W46-cosine pool drops below
0.50 because there is no shared-state context. The
`r95_pseudo_kv_reuse` family; `proxy_recall_cosine ≥ 0.80`
AND `proxy_recall_cosine - w47_recall_cosine ≥ 0.20` across
all seeds.

### H4 — Multi-head proxy attention beats single-head

On a synthetic two-feature task where head A should attend to
the spherical channel and head B to the causal channel
(reproduced from R-92 H3 attention specialisation), an
`H=2` multi-head proxy block achieves task accuracy ≥ 0.90
while an `H=1` single-head block stays ≤ 0.65. The
`r95_multi_head_specialisation` family;
`multi_head_acc - single_head_acc ≥ 0.20` AND
`multi_head_acc ≥ 0.85` minimum.

### H5 — Reconstruction objective recovers prior-turn channels

The reconstruction decoder achieves mean L1 reconstruction
distance ≤ 0.30 on a held-out partition of the prior-turn
feature vector. The `r95_reconstruction_objective` family;
`reconstruction_l1 ≤ 0.30` AND
`reconstruction_l1 < 0.5 * baseline_l1` across all seeds.

### H6 — Branch/cycle bias separates two branches

On a synthetic branching regime where branch A's gold gate is
+1 and branch B's gold gate is -1 (with identical channel
features), the learned branch/cycle bias recovers both signs
with accuracy ≥ 0.90. The `r95_branch_cycle_bias` family;
`branch_a_acc + branch_b_acc ≥ 1.80` (i.e. ≥ 0.90 mean) AND
neither branch ≤ 0.85.

### H7 — Slot-memory write head: write gate is selective

On a synthetic high-noise regime where 50% of turns are pure
noise, the trained write head writes to the pseudo-KV bank on
the signal turns and *skips* the noise turns with mean
selectivity ≥ 0.80. The `r95_write_gate_selectivity` family;
`signal_write_rate ≥ 0.80` AND `noise_write_rate ≤ 0.30`.

### H8 — Latent control serializer is bijective and bounded

The `LATENT_CTRL` bytes round-trip through the
`LatentControlWitness` CID; total visible-token cost stays
within `[W46_compact_ctrl_tokens, 2 * W46_compact_ctrl_tokens]`.
The `r95_latent_control_round_trip` family;
`latent_ctrl_round_trip_ok = 1.0` AND
`mean_latent_ctrl_tokens ≤ 16`.

### H9 — Branch-history compressor saves visible tokens

The branch-history compressor saves at least 50% of the
visible tokens vs the textual rendering of the branch path,
while the bijection through the compressor CID is exact. The
`r95_branch_history_compression` family;
`compressed_tokens_saved >= 0.5 * textual_tokens` AND
`compressor_round_trip_ok = 1.0`.

### H10 — Shared-state proxy attention determinism

Two independent runs of `SharedStateProxyTeam.run` with the
same training set, seed, registry, and observation builder
produce byte-identical `final_output`, root CID, every
`proxy_outer_cid`, every `pseudo_kv_bank_head_cid`, the
trained-params CID, training-trace CID, and the
`shared_state_capsule_cid`. The `r95_replay_determinism`
family; `replay_determinism_ok = 1.0` across all seeds.

### H11 — W48 envelope verifier soundness

The W48 verifier rejects 22+ disjoint forged envelopes
(schema mismatch, shared-state CID mismatch, pseudo-KV
mismatch, multi-head proxy mismatch, reconstruction witness
mismatch, branch-bias mismatch, slot-memory mismatch,
latent-ctrl mismatch, outer-CID mismatch, etc.). Cumulative
trust boundary across W22..W48 = **301 enumerated failure
modes** (279 from W22..W47 + 22 new at W48).

### H12 — Proxy distribution cap reproduces

Adversarial all-channel forgery + a forged pseudo-KV bank +
a forged training distribution (so the trained controller
learns the adversary's distribution): the W48 mechanism cannot
recover. The `r95_proxy_distribution_cap` family;
`downstream_protect_rate ≤ 0.3` across all seeds — proved-
conditional limitation `W48-L-PROXY-DISTRIBUTION-CAP`
(strengthens `W47-L-AUTOGRAD-DISTRIBUTION-CAP`).

### H13 — Shared-state-aware backend gain on synthetic task

Using a `SharedStateAwareSyntheticBackend` that conditions
its answer on the presence of `SHARED_STATE_HASH:` in the
prompt, the W48 arm achieves task-correct rate 1.0 while the
W47 arm stays at 0.0 (W47 emits `MANIFOLD_CTRL` but not
`SHARED_STATE_HASH`). The `r95_shared_state_aware_backend`
family; `task_correct_rate ≥ 0.9` AND
`task_correct_rate - w47_task_correct_rate ≥ 0.9` across all
seeds.

### H14 — Released SDK byte-identity preserved

`tests/test_smoke_full.py` reports "ALL CHECKS PASSED" with
the W48 module on disk; `coordpy.__version__` is still
`"0.5.20"`; `coordpy.SDK_VERSION` is still
`"coordpy.sdk.v3.43"`; the released wheel surface is
byte-for-byte unchanged. The W48 module ships in the source
tree at `coordpy.shared_state_proxy` and is reachable only
through an explicit import.

## Falsifiers

* **W48-L-TRIVIAL-SHARED-STATE-PASSTHROUGH** — a trivially-
  configured `SharedStateProxyRegistry`
  (`proxy_enabled=False`, `pseudo_kv_enabled=False`, W47-
  trivial inner) reduces to `AutogradManifoldTeam.run` byte-
  for-byte; if `H1` fails, the trivial-passthrough property is
  falsified.

* **W48-L-PROXY-DISTRIBUTION-CAP** — adversarial all-channel
  forgery + forged pseudo-KV bank + forged training set: the
  trained proxy controller cannot recover because it learned
  the adversary's distribution.

* **W48-L-NO-REAL-KV-CAP** — the W48 stack still does not
  touch transformer-internal KV bytes; it reproduces the
  algebraic interface at the capsule layer with auditable
  byte-deterministic provenance, no more. The R-95 results
  use the `SharedStateAwareSyntheticBackend` and the
  `CtrlAwareAutogradBackend`; real-LLM realism anchors are
  bounded.

## Per-component verdicts (preview)

* **Shared base state capsule** — *structurally + behaviourally
  useful* (CID stability across turns means roles can audit
  prior turns' base state).
* **Per-role rank-`r` delta** — *behaviourally useful* (gives
  back the per-role expressivity that was carried in W46's
  multi-rank role adapter, now under a *shared* base).
* **Pseudo-KV factor bank** — *behaviourally useful on the
  recall regime; structurally useful as the auditable
  KV-proxy surface*.
* **Multi-head proxy attention** — *behaviourally useful*
  (head specialisation reproduces R-92 H3).
* **Reconstruction objective** — *behaviourally useful*
  (recovers prior-turn flat channels with L1 ≤ 0.30 on the
  synthetic regime).
* **Branch/cycle bias** — *behaviourally useful* (two branches
  separable from identical channel features).
* **Slot-memory write head** — *behaviourally useful*
  (selectively writes signal turns and skips noise turns).
* **Latent control serializer** — *behaviourally useful on the
  synthetic backend; not load-bearing on real LLMs* (same
  caveat as W46 H6 and W47 H12).
* **Branch-history compressor** — *structurally useful*
  (bijection + visible-token savings on long branch paths).
* **Training trace witness** — *structurally useful* (full
  auditability of the training run).
* **Proxy distribution cap** — *limitation reproduces honestly*.

## Architecture triage

| Frontier candidate                                  | W48 bucket                                              | Verdict |
|---|---|---|
| Shared base state capsule                           | **trainable now (capsule-layer surrogate)**             | shipped |
| Per-role rank-`r` shared-state delta                | **trainable now**                                       | shipped |
| Pseudo-KV factor bank                               | **transformer-proxy now** (low-rank `QK^T V` proxy)     | shipped |
| Trainable slot memory + write/read gates            | **trainable now**                                       | shipped |
| Multi-head proxy attention                          | **transformer-proxy now**                               | shipped |
| Reconstruction / retention objective                | **trainable now**                                       | shipped |
| Branch/cycle-aware bias                             | **trainable now**                                       | shipped |
| Branch-history compressor                           | **approximable now**                                    | shipped |
| Trainable latent control serializer                 | **trainable now**                                       | shipped |
| Real KV-cache pooling across turns                  | **substrate-blocked**                                   | unchanged |
| Transformer-internal mixed-curvature attention      | **substrate-blocked**                                   | unchanged |
| Continuous Grassmannian homotopy                    | **substrate-blocked**                                   | unchanged |
| Hidden-state-aware time attention                   | **substrate-blocked**                                   | unchanged |
| GPU/CUDA-backed autograd                            | **substrate-blocked (deliberately deferred)**           | unchanged |

## What W48 explicitly does NOT do

* W48 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`. The
  multi-head proxy attention is the strongest *capsule-layer
  proxy* we can write today; closing the conjecture still
  requires architectural access to the transformer's
  attention computation and KV cache.
* W48 does NOT transplant real KV-cache bytes. The pseudo-KV
  factor bank is a capsule-layer surrogate that reproduces
  the algebraic interface; it never reads or writes a real
  transformer KV cache.
* W48 does NOT claim multi-host coupling. The shared state +
  pseudo-KV bank runs on a single process / single backend.
* W48 does NOT claim training-data-free generalisation. The
  shared-state proxy is trained on a hermetic synthetic bank
  pre-committed in the R-95 sources.
* W48 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD` or
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`.
* W48 does NOT ship CUDA / GPU support.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`. The
  W48 module ships at `coordpy.shared_state_proxy` and is
  reachable only through an explicit import — same convention
  as W43..W47.

## New theorem-style claims (preview)

* **W48-T-SHARED-STATE-CID-STABILITY** (proved + mechanically-
  checked) — every turn of `SharedStateProxyTeam.run`
  references the same `shared_state_capsule_cid` for a given
  registry; per-role delta CIDs are stable for the role.
* **W48-T-PSEUDO-KV-ALGEBRAIC-INTERFACE** (proved by
  inspection) — the pseudo-KV bank's `read(query)` reduces to
  `softmax((Q · K^T) / sqrt(d)) · V` with a causal mask over
  admissible slots, exactly as in a transformer attention
  head, but at the capsule layer.
* **W48-T-MULTI-HEAD-SPECIALISATION** (proved-conditional +
  empirical) — under the bounded-feature assumption + the
  two-axis attention specialisation regime, an `H=2`
  proxy attention block strictly separates the two axes; an
  `H=1` block stays at ~chance.
* **W48-T-RECONSTRUCTION-DECODER-SOUNDNESS** (proved + empirical)
  — the trainable reconstruction decoder achieves L1 ≤ 0.30
  on the held-out prior-turn flat feature vector.
* **W48-T-BRANCH-CYCLE-BIAS-EXPRESSIVITY** (proved-conditional
  + empirical) — the learned branch/cycle bias matrix
  separates two branches that share identical channel
  features with ≥ 0.90 accuracy.
* **W48-T-WRITE-GATE-SELECTIVITY** (proved-conditional +
  empirical) — the trained write head's selectivity exceeds
  0.80 on the high-noise regime.
* **W48-T-TRAIN-DETERMINISM** (proved + mechanically-checked,
  carries forward from W47) — two independent training runs
  produce byte-identical parameter CIDs.
* **W48-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked) — the W48 verifier enumerates 22
  disjoint failure modes; cumulative trust boundary across
  W22..W48 = 301 modes.
* **W48-L-NO-REAL-KV-CAP** (proved-conditional limitation,
  carries forward from W47 and strengthens it) — the
  pseudo-KV factor bank reproduces the algebraic interface,
  not real KV bytes.
* **W48-L-PROXY-DISTRIBUTION-CAP** (proved-conditional
  limitation, strengthens W47-L-AUTOGRAD-DISTRIBUTION-CAP) —
  when the adversary controls the pseudo-KV bank + training
  distribution, the trained proxy cannot recover.
* **W48-L-PURE-PYTHON-TRAINING-COST-CAP** (carries forward from
  W47) — the pure-Python autograd engine caps practical
  training to a few hundred steps on small banks; production
  training requires NumPy/JAX/PyTorch bindings.

## What this enables for the programme

* **Strengthens** the W46/W47 carry-forward
  `W47-C-DEEP-TRANSFORMER-COUPLING` by adding a *multi-head
  proxy attention block + shared base state + pseudo-KV
  factor bank* — the closest executable capsule-layer
  reconstruction of a transformer block we can write today.
* **Strengthens** `W47-L-AUTOGRAD-DISTRIBUTION-CAP` to include
  pseudo-KV bank forgery (`W48-L-PROXY-DISTRIBUTION-CAP`).
* **Preserves** all of W43..W47's deterministic-audit
  properties — the W48 module is strictly additive.
* **Does not close** the substrate-blocked W43 conjectures or
  `W47-C-DEEP-TRANSFORMER-COUPLING`. The honest summary:
  W48 is the strongest *executable proxy* available without
  substrate access.
* **Mints** new W48 conjectures:
  `W48-C-REAL-KV-COUPLED-PROXY` (couple the pseudo-KV bank
  to a real LLM's KV cache through prompt-side caching hooks
  — requires backend support), and
  `W48-C-MULTI-HOST-SHARED-STATE` (share the W48 base state +
  pseudo-KV bank across hosts; needs a host-consensus
  protocol — strengthens W47-C-LIVE-MULTI-HOST-AUTOGRAD).

## Done = the following commits land

1. `coordpy/shared_state_proxy.py` ≈ 2500 LoC, pure Python /
   stdlib only. Reuses the W47 `Variable` + `AdamOptimizer`
   autograd engine via explicit import.
2. `coordpy/r95_benchmark.py` ≈ 1100 LoC, dependency-free.
3. `tests/test_shared_state_proxy_w48.py` — ≥ 30 tests covering
   every component + the trivial-passthrough falsifier +
   shared-state CID stability + pseudo-KV reuse + reconstruction
   + branch/cycle bias + multi-head specialisation + verifier
   soundness.
4. `tests/test_r95_benchmark.py` — ≥ 14 tests covering H1..H14.
5. `docs/RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md` and this
   success-criterion file.
6. Updates to `docs/RESEARCH_STATUS.md`,
   `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
   `docs/HOW_NOT_TO_OVERSTATE.md`,
   `docs/context_zero_master_plan.md`,
   `papers/context_as_objects.md`, and `CHANGELOG.md`.

No README change. No version bump. No PyPI release.
