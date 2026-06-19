# Pre-committed success criterion — W50 Cross-Backend Latent Coordination (XBLC)

> Programme step: post-W49. Mints axis 47 of the Context Zero
> programme. Strictly additive on top of W49 MBCC, W48 SSTP, W47
> AMS, W46 MMC, W45 LMC, W44 LMCC, W43 PMC, and the released
> v3.43 line. Honest scope: W50 stacks **four+ orthogonal trainable
> advances** on top of W49 — a deeper proxy stack (`L=4`),
> cross-backend latent projection, adaptive emit-mask compression
> with a K=16 dictionary, role-pair pseudo-KV transfer with an
> age/retention/transfer-aware eviction policy V2, and a
> reconstruction-aware shared-latent carrier V2 with explicit
> chain-walking. It does NOT touch transformer-internal hidden
> state, real KV cache bytes, attention weights, or embeddings.
> W43..W49's substrate-blocked conjectures
> (`W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`,
> `W47-C-DEEP-TRANSFORMER-COUPLING`,
> `W48-C-REAL-KV-COUPLED-PROXY`,
> `W48-C-MULTI-HOST-SHARED-STATE`,
> `W49-C-DEEP-TRANSFORMER-COUPLING`,
> `W49-C-CROSS-MODEL-LATENT-TRANSFER`) carry forward unchanged.
> W50 is the strongest *executable proxy* line we can write
> today; it is *not* a closure of those.

## Mechanism

W50 introduces the **Cross-Backend Latent Coordination (XBLC)**
layer — five orthogonal capsule-native advances:

1. **`CrossBackendAlignmentLayer`** (M1). Per-backend trainable
   encoder + per-backend trainable decoder routed through a
   shared "lingua franca" code of dimension `D_xb`. Trains a
   behavioral-fidelity score that measures whether the decoded
   shared-latent capsule chain from backend A induces equivalent
   downstream behavior on backend B. Carries a
   `CrossBackendAlignmentWitness` with `(source_model_tag,
   target_model_tag, projection_params_cid, fidelity_score,
   anchor_status)`. **Primary anchor**: two synthetic
   backends with distinct hint-aware policies. **Best-effort
   anchor**: `OllamaBackend` against `SyntheticLLMClient` when
   `COORDPY_W50_OLLAMA_REACHABLE=1`; otherwise `anchor_status =
   "synthetic_only"`.

2. **`DeepProxyStack`** (M2). `L=4` stacked `ProxyTransformerBlock`s
   (vs W49's `L_p=2`), each with its own trainable
   `residual_scale` and a new per-layer **learned mask gate**
   that suppresses non-load-bearing block outputs. Reuses W49
   `ProxyTransformerBlock` weights as the initialiser for the
   first two blocks. Witnesses per-block activation norms and
   per-layer residual scales in `DeepProxyStackForwardWitness`.

3. **`AdaptiveCompressionCodebook`** (M3). Trainable K=16
   prototype codebook (vs W49's K=8) plus a learned
   **per-bit emit-mask gate** that suppresses bits below a
   trained importance threshold. Goal: ≥ 8 bits per visible-
   token at retention cosine ≥ 0.90. Carries an
   `AdaptiveCompressionWitness` binding
   `(codebook_cid, target_rate, achieved_bits_per_token,
   retention_floor, emit_mask_cid)`.

4. **`CrossBankTransferLayer`** + **`AdaptiveEvictionPolicyV2`**
   (M4). Trainable role-pair-conditioned transfer matrix that
   moves slots between role banks (a slot written by role A
   can be projected into role B's bank under a trained
   `(role_a, role_b)` transformation). Adaptive eviction policy
   V2: extends W49's sigmoid scorer with `(slot_age,
   retention_probability, transfer_signal)` inputs. Carries a
   `CrossBankTransferWitness` and replaces W49's plain
   `EvictionPolicy`.

5. **`SharedLatentCarrierV2`** + **`ReconstructionV2Head`** (M5).
   Trainable, chain-walkable, reconstruction-aware shared-
   latent carrier with explicit cross-role reuse map and a
   trained reconstruction head that recovers turn `t-k` state
   from current turn's carrier (for `k ≤ 3`). Carries a
   `SharedLatentCarrierWitness` and a `ReconstructionV2Witness`.

All of it is pure-Python / stdlib. No NumPy, no PyTorch, no JAX
dependency. The pure-Python reverse-mode autograd engine from W47
(`Variable` + `AdamOptimizer`) is reused unchanged. The released
SDK v3.43 contract remains byte-for-byte unchanged.

Honest scope (do-not-overstate)
-------------------------------

W50 does NOT touch transformer-internal hidden state, KV cache
bytes, attention weights, or embeddings. Every parameter of every
W50 module operates over W43..W49 capsule-layer encodings, the
W47 trainable channel features, the W49 multi-bank pseudo-KV
factor slots, and the W49 shared-latent capsule chain. The
cross-backend alignment is **not** cross-tokenizer alignment;
it operates over the capsule-layer carrier exclusively.

W50 does NOT claim "real cross-model latent transfer". The
primary anchor is two synthetic backends with distinct hint-aware
behaviors. When the Ollama anchor is reachable, W50 records a
single real-LLM realism probe that **bounds**, not closes, the
cross-model conjecture. The
`W49-C-CROSS-MODEL-LATENT-TRANSFER` conjecture carries forward
sharpened as `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`.

W50 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
`W48-C-DEEP-TRANSFORMER-COUPLING`, or
`W49-C-DEEP-TRANSFORMER-COUPLING`. The L=4 deep stack is a
deeper *capsule-layer proxy*, not a deeper transformer.

W50 does NOT claim adversarial robustness under training-
distribution forgery — `W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`
(strengthens `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP`).

W50 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine reused from W47 is correct but slow; W50 trains
on bounded synthetic banks (~16–48 examples for ~8–12 Adam
steps per module, stage-wise; optional 16-example × 3-epoch e2e
pass).

W50 is strictly additive. When configured trivially
(`cross_backend_enabled=False`, `deep_stack_enabled=False`,
`adaptive_compression_enabled=False`,
`cross_bank_transfer_enabled=False`,
`shared_latent_carrier_v2_enabled=False`, W49-trivial inner),
the `W50Team` orchestrator reduces to `MultiBlockProxyTeam.run`
byte-for-byte — the `W50-L-TRIVIAL-W50-PASSTHROUGH` falsifier.

This module ships at `coordpy.cross_backend_alignment`,
`coordpy.deep_proxy_stack`,
`coordpy.adaptive_compression`,
`coordpy.cross_bank_transfer`,
`coordpy.shared_latent_carrier`, and the composition module
at `coordpy.w50_team`. It is NOT exported through
`coordpy.__init__`'s public surface at this milestone; the
stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W50 surface through explicit
`from coordpy.<module> import ...` imports — same convention as
W43..W49.

## H1..H16 success bar

Sixteen pre-committed hypotheses across **two benchmark families**
(R-98 + R-99); each is exercised by a per-family test in
`tests/test_r98_benchmark.py` / `tests/test_r99_benchmark.py`
plus per-component unit coverage in
`tests/test_cross_backend_alignment_w50.py`,
`tests/test_deep_proxy_stack_w50.py`,
`tests/test_adaptive_compression_w50.py`,
`tests/test_cross_bank_transfer_w50.py`,
`tests/test_shared_latent_carrier_w50.py`,
`tests/test_w50_trivial_passthrough_byte_identical.py`, and
`tests/test_w50_team_envelope_chain.py`.

### H1 — Trivial W50 passthrough (R-98)

A trivially-configured `W50Registry`
(`cross_backend_enabled=False`, `deep_stack_enabled=False`,
`adaptive_compression_enabled=False`,
`cross_bank_transfer_enabled=False`,
`shared_latent_carrier_v2_enabled=False`, W49-trivial inner)
reduces to `MultiBlockProxyTeam.run` byte-for-byte. The
`family_trivial_w50_passthrough` family reports
`passthrough_ok = 1.0` across all baseline + new arms across all
three seeds.

### H2 — Deep stack L=4 strict retention gain (R-98)

On a synthetic four-step product-amplified composition regime
where the target gate depends on a composed nonlinear function
of four channels (W49's `L=2` stack cannot fit it cleanly), an
`L=4` deep stack achieves classifier accuracy strictly greater
than W49's `L=2` baseline. The
`family_deep_stack_depth_strict_gain` family;
`acc(L=4) - acc(L=2) ≥ 0.05` on the mean across seeds — honest
bar under the W47 pure-Python autograd training cost cap
(`W50-L-PURE-PYTHON-TRAINING-COST-CAP`). The structural-utility
floor is `acc(L=4) ≥ 0.65` (well above chance) regardless of
the relative gain.

### H3 — Cross-backend alignment fidelity (synthetic primary) (R-98)

On the synthetic primary anchor (two synthetic backends with
distinct hint-aware policies), the trained
`CrossBackendAlignmentLayer` decodes a shared-latent carrier
emitted on backend A into a carrier whose induced downstream
behavior on backend B has fidelity ≥ 0.95 against the native
backend-B carrier baseline. The
`family_cross_backend_alignment_synthetic` family;
`mean(fidelity_score) ≥ 0.95` across three seeds.

### H4 — Cross-bank role-pair transfer gain (R-98)

On a 4-role × 4-role role-pair grid where role-A's bank has a
target fact and role-B reads, the trained role-pair transfer
matrix recovers the target fact with mean cosine ≥ 0.75 — the
W49 no-transfer baseline saturates at ≤ 0.60. The
`family_cross_bank_transfer_role_pair_gain` family;
`(transfer_recall - no_transfer_recall) ≥ 0.15` on mean across
seeds.

### H5 — Adaptive eviction V2 gain (R-98)

On a regime where the first slot carries a signal fact and the
last few slots carry noise, the adaptive eviction policy V2
keeps the signal slot strictly more reliably than W49's
sigmoid scorer at tight bank capacity. The
`family_adaptive_eviction_v2_vs_v1` family;
`(recall_v2 - recall_v1) ≥ 0.10` on mean across seeds.

### H6 — 8-turn long-horizon retention (R-99)

On a length-8 branch path with a target fact emitted at turn 0,
the W50 retention head + multi-bank read + carrier V2 jointly
recover the fact with cosine ≥ 0.90 at turn 7. W49 baseline
drops below 0.80 on the same regime. The
`family_long_horizon_retention_8turn` family;
`w50_cosine_at_turn_7 ≥ 0.90` AND
`(w50_cosine - w49_cosine) ≥ 0.10` on mean across seeds.

### H7 — 12-turn long-horizon retention stretch (R-99)

On a length-12 branch path (deliberately a stretch — we do NOT
expect 1.0), W50 cosine ≥ 0.70 at turn 11; W49 drops below
0.50. The `family_long_horizon_retention_12turn_stretch`
family; `w50_cosine_at_turn_11 ≥ 0.70` (no saturation
expected — honest about drop-off).

### H8 — Reconstruction V2 recovers turn t-3 (R-99)

The trained `ReconstructionV2Head` recovers the flat-feature
vector of turn `t-k` from the carrier at turn `t` with MSE
≤ 0.25 for `k ≤ 3` — well below the random-prediction floor
of 0.33 (uniform target in [-1, 1]). The
`family_reconstruction_v2_recovers_prior_turn` family;
`mse_at_k3 ≤ 0.25` on mean across seeds — honest bar under the
W47 pure-Python autograd training cost cap
(`W50-L-PURE-PYTHON-TRAINING-COST-CAP`). Lower MSE is
achievable with NumPy / GPU acceleration but out of W50 scope.

### H9 — 8 bits/visible-token cramming (R-99)

W50 achieves a structured-bits-per-visible-token ratio ≥ 8.0
on a fixed-length task while retention cosine stays ≥ 0.90.
W49 baseline on the same regime hits ≤ 5.0. The
`family_adaptive_compression_8bits` family;
`w50_bits_per_token ≥ 8.0` AND `retention_cosine ≥ 0.90` on
mean across seeds.

### H10 — W50 outer-CID chain verifier soundness (R-98)

The W50 verifier rejects 20+ disjoint forged envelopes (schema
mismatch, cross-backend witness CID mismatch, deep-stack
forward witness CID mismatch, adaptive-compression witness CID
mismatch, cross-bank transfer witness CID mismatch, shared-
latent carrier V2 witness CID mismatch, reconstruction V2
witness CID mismatch, outer-CID mismatch, etc.). Cumulative
trust boundary across W22..W50 = **343 enumerated failure
modes** (323 from W22..W49 + 20 new at W50).

### H11 — Cross-backend realism anchor (best-effort, R-98)

When `COORDPY_W50_OLLAMA_REACHABLE=1` is set in the
environment, the cross-backend realism probe runs the W50
alignment layer across `OllamaBackend(model="qwen2.5:0.5b")`
and a synthetic ctrl-aware backend; fidelity ≥ 0.80 on n=10
turns. When the env var is absent, the family records
`anchor_status = "synthetic_only"` and `skipped_ok = 1.0` —
the skip path is itself a falsifiable claim. The
`family_cross_backend_alignment_realism_probe` family.

### H12 — Cross-bank compromise cap reproduces (R-98)

Adversarial all-channel forgery + forged role banks + forged
role-pair transfer training set: the W50 cross-bank transfer
cannot recover. The
`family_cross_bank_transfer_compromise_cap` family;
`downstream_protect_rate ≤ 0.3` across all seeds — proved-
conditional limitation `W50-L-CROSS-BANK-COMPROMISE-CAP`.

### H13 — Deep-stack residual pathology falsifier (R-98)

Setting every block's `residual_scale` to 0.0 disables
residual signal flow; the L=4 stack collapses to noise. The
`family_deep_stack_residual_pathology_falsifier` family;
`accuracy_at_zero_residual ≤ 0.55` (no better than chance on
the H2 regime). Witnesses
`W50-L-DEEP-STACK-RESIDUAL-PATHOLOGY-CAP`.

### H14 — Adaptive compression rate-floor falsifier (R-99)

Setting the adaptive compression target to 16 bits/visible-
token (beyond the K=16 codebook's information capacity) is
infeasible: either the bits/token target is missed or the
retention floor breaks. The
`family_adaptive_compression_rate_falsifier` family;
`achieved_bits_per_token < 16.0` OR
`retention_cosine < 0.50`. Witnesses
`W50-L-RATE-FLOOR-CAP`.

### H15 — W50 distribution cap (carry-forward sharper) (R-99)

Adversarial all-channel forgery + forged role banks + forged
shared-latent carrier + forged cross-backend alignment training
set: the W50 stack cannot recover. The
`family_w50_distribution_cap` family;
`downstream_protect_rate ≤ 0.3` across all seeds — proved-
conditional limitation `W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`
(strengthens `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP`).

### H16 — W50 replay determinism (R-98)

Two independent runs of `W50Team.run` with the same training
set, seed, registry, and observation builder produce byte-
identical `final_output`, root CID, every `w50_outer_cid`,
every per-module witness CID, the shared-latent carrier V2
chain CID list, and the cross-backend alignment witness CID.
The `family_w50_replay_determinism` family;
`replay_determinism_ok = 1.0` across all seeds.

## Falsifiers

* **W50-L-TRIVIAL-W50-PASSTHROUGH** — a trivially-configured
  `W50Registry` reduces to `MultiBlockProxyTeam.run` byte-for-
  byte; if H1 fails, the trivial-passthrough property is
  falsified.

* **W50-L-CROSS-BACKEND-TOKENIZER-CAP** — when the realism
  probe is skipped (Ollama unreachable), the cross-tokenizer
  conjecture remains carried forward, witnessed by H11's skip
  path.

* **W50-L-CROSS-BANK-COMPROMISE-CAP** — adversarial all-channel
  forgery + forged role banks + forged role-pair transfer
  training: the trained cross-bank transfer cannot recover.
  Reproduces honestly in the R-98 family.

* **W50-L-DEEP-STACK-RESIDUAL-PATHOLOGY-CAP** — `residual_scale =
  0` collapses the L=4 stack; bounds the depth claim.

* **W50-L-RATE-FLOOR-CAP** — target rate 16 bits/token exceeds
  the K=16 codebook's information capacity; either rate is
  missed or retention floor breaks.

* **W50-L-MULTI-BLOCK-DISTRIBUTION-CAP** — adversarial all-
  channel forgery + forged role banks + forged shared-latent
  carrier + forged cross-backend alignment training: the
  trained W50 stack cannot recover. Strengthens
  `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP`.

* **W50-L-NO-REAL-KV-CAP** — the W50 stack still does not
  touch transformer-internal KV bytes; multi-bank transfer
  + adaptive eviction does not change this.

* **W50-L-PURE-PYTHON-TRAINING-COST-CAP** — the pure-Python
  autograd engine carries forward from W47; W50 stage-wise
  training cost is approximately the sum of per-module costs.

* **W50-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** — carries forward
  from W48/W49 — real LLMs may or may not condition on the
  W50 carrier. H3 evidence is anchored to synthetic backends;
  H11 is a best-effort anchor on Ollama.

## Per-component verdicts (preview)

* **Cross-backend alignment layer (M1)** — *behaviourally useful*
  if H3 passes; *structurally useful* always (auditable
  alignment witness CID).
* **Deep proxy stack L=4 (M2)** — *behaviourally useful* if H2
  passes; *structurally useful* always (per-block CIDs).
* **Adaptive compression K=16 (M3)** — *behaviourally useful*
  on H9 (8 bits/token); *structurally useful*
  (round-trips through codebook CID).
* **Cross-bank transfer + eviction V2 (M4)** — *behaviourally
  useful* on H4 + H5; *structurally useful* (transfer witness
  CID).
* **Shared-latent carrier V2 + reconstruction V2 (M5)** —
  *behaviourally useful* on H6 + H7 + H8; *structurally useful*
  always (chain-walkable carrier).
* **Deep stack residual pathology cap** — *limitation
  reproduces honestly*.
* **Rate-floor cap** — *limitation reproduces honestly*.
* **Cross-bank compromise cap** — *limitation reproduces
  honestly*.
* **Multi-block distribution cap** — *limitation reproduces
  honestly*.

## Architecture triage

| Frontier candidate                                  | W50 bucket                                              | Verdict |
|---|---|---|
| Cross-backend latent projector (synthetic primary)  | **trainable now (cross-backend, synthetic anchor)**     | shipped |
| Cross-tokenizer behavioral transfer                  | **substrate-blocked**                                   | carry-forward |
| Deep proxy stack (L=4)                              | **transformer-proxy now (depth)**                       | shipped |
| Role-pair pseudo-KV transfer                         | **trainable now (transfer)**                            | shipped |
| Adaptive eviction policy V2                          | **trainable now (eviction)**                            | shipped |
| Adaptive compression K=16 + emit-mask gate           | **trainable now (compression)**                         | shipped |
| Shared-latent carrier V2 + reconstruction V2         | **trainable now (carrier)**                             | shipped |
| Cross-backend realism anchor (Ollama best-effort)   | **anchor probe (best-effort)**                          | shipped (conditional on env) |
| Real KV-cache pooling across turns                   | **substrate-blocked**                                   | unchanged |
| Transformer-internal mixed-curvature attention       | **substrate-blocked**                                   | unchanged |
| True hidden-state / KV sharing                       | **substrate-blocked**                                   | unchanged |
| Multi-host shared-state transfer                     | **substrate-blocked**                                   | unchanged |
| GPU/CUDA-backed autograd                             | **substrate-blocked (deliberately deferred)**           | unchanged |

## What W50 explicitly does NOT do

* W50 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
  `W48-C-DEEP-TRANSFORMER-COUPLING`,
  `W49-C-DEEP-TRANSFORMER-COUPLING`, or any other W43..W49
  substrate-blocked direction.
* W50 does NOT transplant real KV-cache bytes. The role-pair
  transfer operates on capsule-layer pseudo-KV slots only.
* W50 does NOT close `W49-C-CROSS-MODEL-LATENT-TRANSFER`. The
  realism probe **bounds**, not closes, the conjecture; the
  conjecture is sharpened forward as
  `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER`.
* W50 does NOT claim multi-host coupling.
* W50 does NOT claim training-data-free generalisation. Each
  W50 module is trained on hermetic synthetic banks pre-
  committed in the R-98 / R-99 sources.
* W50 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`,
  `W48-C-REAL-KV-COUPLED-PROXY`, or
  `W48-C-MULTI-HOST-SHARED-STATE`.
* W50 does NOT ship CUDA / GPU support.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`. The
  W50 modules ship at `coordpy.cross_backend_alignment`,
  `coordpy.deep_proxy_stack`,
  `coordpy.adaptive_compression`,
  `coordpy.cross_bank_transfer`,
  `coordpy.shared_latent_carrier`, and `coordpy.w50_team` —
  reachable only through explicit imports — same convention
  as W43..W49.

## New theorem-style claims (preview)

* **W50-T-CROSS-BACKEND-PROJECTOR-SOUNDNESS** (proved by
  inspection + mechanically-checked) — the cross-backend
  encode→decode roundtrip is a deterministic fixed point at
  trivial parameters.
* **W50-T-DEEP-STACK-L4-EXPRESSIVITY** (proved-conditional +
  empirical) — under the bounded-feature assumption + the
  four-step composition regime, an L=4 deep stack strictly
  separates a composition that L=2 cannot.
* **W50-T-CROSS-BANK-TRANSFER-INTERFACE** (proved by
  inspection + mechanically-checked) — the role-pair transfer
  preserves the bank algebraic interface (slot keys/values
  remain factor-dim vectors; transfer is a learned linear
  projection).
* **W50-T-ADAPTIVE-EVICTION-V2-MONOTONICITY** (proved by
  inspection + mechanically-checked) — the V2 keep-score is
  monotone non-decreasing in retention probability and write
  gate, non-increasing in age.
* **W50-T-ADAPTIVE-COMPRESSION-RATE-BOUND** (proved-conditional
  + empirical) — under the K=16 codebook + adaptive emit mask
  + retention floor ≥ 0.90, the bits-per-visible-token ratio
  ≥ 8.0.
* **W50-T-RECONSTRUCTION-V2-CORRECTNESS** (proved-conditional
  + empirical) — the trained reconstruction V2 head recovers
  prior-turn flat features at MSE ≤ 0.05 for k ≤ 3.
* **W50-T-W50-OUTER-CID-CHAIN** (proved by inspection +
  mechanically-checked) — `w47_outer → w48_proxy_outer →
  w49_multi_block_outer → w50_outer` chain verifies; tamper
  detected at every link.
* **W50-T-TRIVIAL-PASSTHROUGH-BYTE-IDENTICAL** (proved by
  inspection + empirical) — `W50Team` configured with
  `W50Params.build_trivial()` produces a W50 envelope whose
  internal `w49_multi_block_outer_cid` field equals the W49
  outer CID byte-for-byte.
* **W50-T-CRAMMING-WITNESS-V2-SOUNDNESS** (proved by
  inspection + mechanically-checked) — the per-turn W50
  cramming witness's `structured_bits` field equals the sum
  of (K=16 dictionary code bits) + (adaptive emit mask bits) +
  (bits payload bits) exactly.
* **W50-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked) — the W50 verifier enumerates 20
  disjoint failure modes; cumulative trust boundary across
  W22..W50 = 343 modes.
* **W50-T-LONG-HORIZON-RETENTION-8TURN** (proved-conditional +
  empirical) — on a length-8 branch path, W50 cosine ≥ 0.90.
* **W50-T-LONG-HORIZON-RETENTION-12TURN-STRETCH** (proved-
  conditional + empirical) — on a length-12 branch path, W50
  cosine ≥ 0.70 (honest about drop-off).
* **W50-T-CROSS-BACKEND-FIDELITY-SYNTHETIC** (proved-conditional
  + empirical) — synthetic primary anchor fidelity ≥ 0.95.

* **W50-L-TRIVIAL-W50-PASSTHROUGH** (proved by inspection +
  empirical) — trivial W50 = W49 byte-for-byte.
* **W50-L-CROSS-BACKEND-TOKENIZER-CAP** (carries forward,
  sharpened from `W49-C-CROSS-MODEL-LATENT-TRANSFER`) — when
  the Ollama realism probe is skipped, the cross-tokenizer
  conjecture remains carried forward.
* **W50-L-CROSS-BANK-COMPROMISE-CAP** (proved-conditional
  limitation) — adversarial role-bank forgery cannot be
  recovered.
* **W50-L-DEEP-STACK-RESIDUAL-PATHOLOGY-CAP** (proved by
  inspection + empirical) — `residual_scale = 0` collapses
  the L=4 stack.
* **W50-L-RATE-FLOOR-CAP** (proved-conditional limitation) —
  16-bit target exceeds K=16 codebook capacity.
* **W50-L-MULTI-BLOCK-DISTRIBUTION-CAP** (proved-conditional
  limitation, strengthens W49) — adversarial all-channel
  forgery + W50-specific forgery: cannot recover.
* **W50-L-NO-REAL-KV-CAP** (carries forward, strengthens W49)
  — role-pair transfer does not transplant real KV bytes.
* **W50-L-PURE-PYTHON-TRAINING-COST-CAP** (carries forward) —
  stage-wise W50 training cost is the sum of per-module costs.
* **W50-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (carries forward
  from W48/W49).

* **W50-C-CROSS-TOKENIZER-LATENT-TRANSFER** (sharper than
  `W49-C-CROSS-MODEL-LATENT-TRANSFER`) — the conjecture is now
  blocked only at tokenizer divergence; the projector layer
  + carrier chain is itself trained and auditable, but
  behavioral transfer across genuinely different tokenizers
  requires backend-side adapters out of W50 scope.
* **W50-C-DEEP-TRANSFORMER-COUPLING** (carries forward,
  bounds W47/W48/W49 further) — full transformer-internal
  hidden-state + KV-cache coupling remains substrate-blocked.
* **W50-C-REAL-KV-COUPLED-PROXY** (carries forward W48-C
  unchanged).
* **W50-C-MULTI-HOST-SHARED-STATE** (carries forward W48-C
  unchanged).

## What this enables for the programme

* **Strengthens** every W47..W49 carry-forward by adding a
  deeper proxy stack + cross-backend latent projector +
  adaptive compression + role-pair transfer + reconstruction-
  aware carrier — five orthogonal advances.
* **Strengthens** `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP` to
  include W50-specific forgery
  (`W50-L-MULTI-BLOCK-DISTRIBUTION-CAP`).
* **Sharpens** `W49-C-CROSS-MODEL-LATENT-TRANSFER` to
  `W50-C-CROSS-TOKENIZER-LATENT-TRANSFER` — the projector +
  carrier are now trained and auditable; only tokenizer-level
  divergence remains carried forward.
* **Preserves** all of W43..W49's deterministic-audit
  properties — the W50 modules are strictly additive.
* **Does not close** the substrate-blocked W43..W49
  conjectures. The honest summary: W50 is the strongest
  *executable proxy* available without substrate access, with
  one real-LLM realism anchor when reachable.

## Done = the following commits land

1. `coordpy/cross_backend_alignment.py` — pure Python / stdlib.
2. `coordpy/deep_proxy_stack.py` — pure Python / stdlib.
3. `coordpy/adaptive_compression.py` — pure Python / stdlib.
4. `coordpy/cross_bank_transfer.py` — pure Python / stdlib.
5. `coordpy/shared_latent_carrier.py` — pure Python / stdlib.
6. `coordpy/w50_team.py` — pure Python / stdlib (composition).
7. `coordpy/r98_benchmark.py` — dependency-free benchmark
   family (10 families).
8. `coordpy/r99_benchmark.py` — dependency-free benchmark
   family (7 families).
9. `tests/test_cross_backend_alignment_w50.py`
10. `tests/test_deep_proxy_stack_w50.py`
11. `tests/test_adaptive_compression_w50.py`
12. `tests/test_cross_bank_transfer_w50.py`
13. `tests/test_shared_latent_carrier_w50.py`
14. `tests/test_w50_trivial_passthrough_byte_identical.py`
15. `tests/test_w50_team_envelope_chain.py`
16. `tests/test_r98_benchmark.py`
17. `tests/test_r99_benchmark.py`
18. `examples/w50_smoke_driver.py`
19. `examples/w50_replay_live.py` (Ollama best-effort anchor)
20. `docs/RESULTS_W50_CROSS_BACKEND_LATENT_COORDINATION.md`
    and this success-criterion file.
21. Updates to `docs/RESEARCH_STATUS.md`,
    `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
    `docs/HOW_NOT_TO_OVERSTATE.md`,
    `docs/context_zero_master_plan.md`,
    `papers/context_as_objects.md`, and `CHANGELOG.md`.
