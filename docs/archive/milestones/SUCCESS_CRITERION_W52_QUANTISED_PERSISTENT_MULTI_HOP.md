# Pre-committed success criterion — W52 Quantised Persistent Multi-Hop Latent Coordination (QPMHLC)

> Programme step: post-W51. Mints axis 49 of the Context Zero
> programme. Strictly additive on top of W51 PXBLC, W50 XBLC,
> W49 MBCC, W48 SSTP, W47 AMS, W46 MMC, W45 LMC, W44 LMCC, W43
> PMC, and the released v3.43 line. Honest scope: W52 stacks
> **eight orthogonal trainable advances** on top of W51 —
> a stacked two-layer persistent latent state V4 with
> cross-cycle persistence and signal skip-link, a multi-hop
> N-backend translator with disagreement-weighted arbitration
> and length-3 transitivity, a depth-eight role-banked proxy
> stack V3, a three-level quantised codebook compression V4
> with a learned rate-aware budget allocator, a length-twelve
> three-headed reconstruction V4 (causal + branch + cycle),
> a branch/cycle memory V2 with trainable merge + evict heads,
> a role-graph conditioned cross-role transfer, and a
> transcript-vs-shared-state matched-budget comparator. It
> does NOT touch transformer-internal hidden state, real KV
> cache bytes, attention weights, embeddings, or real
> tokenizers. W43..W51's substrate-blocked conjectures
> carry forward unchanged. W52 is the strongest *executable
> proxy* line we can write today; it is *not* a closure of
> those.

## Mechanism

W52 introduces the **Quantised Persistent Multi-Hop Latent
Coordination (QPMHLC)** layer — eight orthogonal
capsule-native advances on top of W51's six:

1. **`PersistentLatentStateV4`** (M1). A **two-layer
   stacked GRU** persistent latent state. Each layer uses
   the W47 ``Variable`` autograd engine; the lower layer
   reads the raw carrier, the upper layer reads the lower
   layer's state. Adds a **signal skip-link** that projects
   the turn-0 carrier hash directly into the upper layer's
   input (helping ≥20-turn recall). Adds a separate
   **cycle-summary state** that persists across cycles
   (carrying invariants that are cycle-stationary). Each
   per-turn ``PersistentLatentStateV4`` entry binds
   ``(turn_index, role, layer_states_cid, parent_state_cid,
   cycle_state_cid, skip_link_cid)``. Chain-walk up to depth
   24 recovers every prior state from the envelope chain
   alone.

2. **`MultiHopBackendTranslator`** (M2). N backend tags
   (default N=4: A, B, C, D). Direct translators on every
   ordered pair (12 edges); a **path translator** computes
   any ordered chain ``A→B→C→D`` by composing direct
   translators with a learned per-edge **disagreement
   weight** ``w_e ∈ [0,1]``. The arbitration is the convex
   combination of direct-edge predictions and chain
   predictions weighted by per-edge confidence. The
   transitivity loss now penalises disagreement at chain
   length **3** (W51 only did chain length 2). The witness
   binds ``(quad_tag, direct_fidelities, chain_fidelities,
   transitivity_gap_len2, transitivity_gap_len3,
   arbitration_score, anchor_status)``.

3. **`DeepProxyStackV3`** (M3). `L=8` stacked
   ``ProxyTransformerBlock``s (vs W51's `L=6`) with two new
   structural extensions on top of W51's branch/cycle
   specialisation:
   - **role-conditioned KV banks** — each layer has a
     per-role bank, routed by the role index (per-role
     residual subspace);
   - **pre-norm + residual gate** per layer (a learned
     sigmoid gate on the residual path, defaulting to
     open so W51-trivial reduces correctly).
   Carries a `DeepProxyStackV3ForwardWitness` with per-layer
   activation norms, per-layer gate values, per-layer
   temperatures, per-role bank CIDs, and a
   compositional-task accuracy bound.

4. **`QuantisedCodebookV4`** (M4). Three-level codebook:
   coarse `K1=32` clusters + fine `K2=16` per-cluster
   sub-prototypes + **ultra-fine** `K3=8` per-pair refinement
   indices. Each carrier is encoded as a `(coarse, fine,
   ultra)` triple. **Adaptive budget allocator**: a learned
   importance head decides how many bits to spend per region
   (skip ultra when its expected information gain is < ε).
   Target: **≥ 14 bits per visible-token** at full emit
   (vs W51's 12). Carries a `QuantisedCompressionWitness`
   with `(coarse_cid, fine_cid, ultra_cid, budget_cid,
   achieved_bits_per_token, degradation_curve)`. The
   degradation curve records achieved bits/token across
   decreasing token budgets `{16, 8, 4, 2, 1}`.

5. **`LongHorizonReconstructionV4`** (M5). Trained
   three-headed reconstruction head:
   - **causal head** (recovers turn `t-k` features from
     current carrier; `k ∈ {1..12}`);
   - **branch head** (recovers turn `t-k` features
     conditioned on branch path);
   - **cycle head** (recovers cycle-stationary invariants
     conditioned on cycle index, regardless of `k`).
   Witness records MSE for each `k` per head plus a
   **degradation curve** for `k ∈ {1..24}` for the causal
   head (we expect reconstruction to degrade past `k=12`).

6. **`BranchCycleMemoryV2`** (M6). Extends W51's
   `BranchCycleMemoryHead` with three new heads:
   - **trainable merge head** — a learned policy that
     decides when two pages should be merged; merges are
     content-addressed and produce an audit trail;
   - **trainable evict head** — a learned policy that
     replaces W51's FIFO ordering with importance-weighted
     eviction;
   - **cross-branch-cycle joint pages** — pages keyed by
     `(branch, cycle)` tuple instead of branch-only or
     cycle-only.
   The head is trainable end-to-end via the W47 autograd
   engine. Witness binds `(n_branch_pages, n_cycle_pages,
   n_joint_pages, page_storage_cid, merge_audit_cid,
   evict_policy_cid, mean_branch_recall,
   mean_cycle_recall, mean_joint_recall)`.

7. **`RoleGraphTransfer`** (M7). A new module: cross-role
   transfer conditioned on a **role graph** — a DAG of
   per-edge mixers indexed by `(src_role, dst_role)`.
   Defaults to the complete graph over the role universe.
   Each edge carries a learned linear projection + bias.
   Useful when team roles have *direction-dependent*
   communication patterns (e.g. planner → researcher is
   different from researcher → planner). The
   `RoleGraphMixer` learns per-edge weights jointly. The
   witness records per-edge CIDs and per-edge gain.

8. **`TranscriptVsSharedStateComparator`** (M8). The new
   ablation module: under a **fixed visible-token budget
   `B`**, compares two arms:
   - **transcript-only** — the team's transcript is
     truncated to `B` tokens (the bounded-context default);
   - **shared-latent** — the team uses a `B`-token
     visible header from the W52 quantised codebook
     instead of the transcript.
   Measures per-arm retention at a target turn `T`. The
   comparator reports the *strict gap* at matched budget
   plus the *bit-density gap* (shared-latent encodes more
   structured bits per token).

All of it is pure-Python / stdlib. No NumPy, no PyTorch, no
JAX dependency. The pure-Python reverse-mode autograd engine
from W47 (`Variable` + `AdamOptimizer`) is reused unchanged.
The released SDK v3.43 contract remains byte-for-byte
unchanged.

Honest scope (do-not-overstate)
-------------------------------

W52 does NOT touch transformer-internal hidden state, KV
cache bytes, attention weights, or embeddings. Every
parameter of every W52 module operates over W43..W51
capsule-layer encodings, the W47 trainable channel features,
the W49 multi-bank pseudo-KV factor slots, the W50
shared-latent carrier V2 chain, the W51 persistent latent
state V3 chain, and the new W52 persistent latent state V4
chain. The multi-hop translator is **not** cross-tokenizer
translation; it operates over capsule-layer carriers and
synthetic backend tags exclusively.

W52 does NOT claim "real cross-model behavioural transfer"
beyond what W51 anchored. The primary anchor for M2 is four
synthetic backends with distinct hint-aware policies; when
`COORDPY_W52_OLLAMA_REACHABLE=1` is set and an Ollama daemon
is reachable, a best-effort quad-backend probe runs and is
recorded. Otherwise the witness records
`anchor_status = "synthetic_only"`. The
`W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY` conjecture
carries forward sharpened as
`W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY` — chain-length-3
transitivity is now trained and auditable on capsule
carriers, but real tokenizer-level transitivity remains
out of W52 scope.

W52 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
`W48-C-DEEP-TRANSFORMER-COUPLING`,
`W49-C-DEEP-TRANSFORMER-COUPLING`,
`W50-C-DEEP-TRANSFORMER-COUPLING`, or
`W51-C-DEEP-TRANSFORMER-COUPLING`. The `L=8` deep stack V3
is a deeper *capsule-layer proxy*, not a deeper transformer.

W52 does NOT claim adversarial robustness under
training-distribution forgery — `W52-L-QPMHLC-DISTRIBUTION-CAP`
strengthens `W51-L-PXBLC-DISTRIBUTION-CAP`.

W52 does NOT claim depth monotonicity. On regimes where
the optimal composition depth is `≤ 6`, the `L=8` stack
may regress relative to `L=6`. The
`W52-L-DEEP-STACK-V3-OVERDEPTH-CAP` falsifier reproduces
honestly.

W52 does NOT claim 24-turn recall above chance for every
seed. Long-horizon retention degrades with sequence length;
the V4 cell + skip-link improves on W51 at 24 turns but is
honest about the drop-off bound.

W52 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine reused from W47 is correct but slow; W52
trains on bounded synthetic banks (~16–48 examples for
~24–48 Adam steps per module, stage-wise; the e2e step
count is bounded by the
`W52-L-PURE-PYTHON-TRAINING-COST-CAP` carry-forward).

W52 is strictly additive. When configured trivially
(every W52 flag `False`, W51-trivial inner), the `W52Team`
orchestrator reduces to `W51Team.run` byte-for-byte — the
`W52-L-TRIVIAL-W52-PASSTHROUGH` falsifier.

These modules ship at `coordpy.persistent_latent_v4`,
`coordpy.multi_hop_translator`,
`coordpy.deep_proxy_stack_v3`,
`coordpy.quantised_compression`,
`coordpy.long_horizon_retention_v4`,
`coordpy.branch_cycle_memory_v2`,
`coordpy.role_graph_transfer`,
`coordpy.transcript_vs_shared_state`, and the composition
module at `coordpy.w52_team`. They are NOT exported through
`coordpy.__init__`'s public surface at this milestone; the
stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W52 surface through explicit
`from coordpy.<module> import ...` imports — same convention
as W43..W51.

## H1..H22 success bar

Twenty-two pre-committed hypotheses across **two benchmark
families** (R-102 + R-103); each is exercised by a
per-family test in `tests/test_r102_benchmark.py` /
`tests/test_r103_benchmark.py` plus per-component unit
coverage in `tests/test_persistent_latent_v4_w52.py`,
`tests/test_multi_hop_translator_w52.py`,
`tests/test_deep_proxy_stack_v3_w52.py`,
`tests/test_quantised_compression_w52.py`,
`tests/test_long_horizon_retention_v4_w52.py`,
`tests/test_branch_cycle_memory_v2_w52.py`,
`tests/test_role_graph_transfer_w52.py`,
`tests/test_transcript_vs_shared_state_w52.py`,
`tests/test_w52_trivial_passthrough_byte_identical.py`, and
`tests/test_w52_team_envelope_chain.py`.

### H1 — Trivial W52 passthrough (R-102)

A trivially-configured `W52Registry` (all enable flags
`False`, W51-trivial inner) reduces to `W51Team.run`
byte-for-byte. The `family_trivial_w52_passthrough` family
reports `passthrough_ok = 1.0` across all baseline + new
arms across all three seeds.

### H2 — Persistent V4 24-turn retention gain (R-102)

On a length-24 task where the gating decision at turn `T`
depends on a feature emitted at turn 0, the W52 V4
two-layer persistent latent state (with skip-link) preserves
the early signal where W51's single-layer V3 does not. The
`family_persistent_v4_long_horizon_gain` family;
`(w52_recall - w51_recall) ≥ 0.15` on mean across seeds.

### H3 — Length-3 multi-hop transitivity (R-102)

The trained `MultiHopBackendTranslator` shows length-3
transitivity: `fidelity(A→D) ≈ fidelity(A→B→C→D)` within
`transitivity_gap_len3 ≤ 0.15` on synthetic primaries. The
`family_multi_hop_quad_transitivity` family;
`mean(transitivity_gap_len3) ≤ 0.15` AND
`mean(transitive_fidelity_a_b_c_d) ≥ 0.70` across seeds.

### H4 — Disagreement-weighted arbitration (R-102)

Under a perturbed edge (one direct edge is corrupted),
the disagreement-weighted arbitration produces strictly
better mean fidelity than naive equal-weight arbitration.
The `family_disagreement_weighted_arbitration` family;
`(weighted_score - naive_score) ≥ 0.05` on mean across
seeds.

### H5 — Deep stack V3 L=8 structural floor + non-regression (R-102)

On a synthetic eight-step product-amplified composition
regime where the target gate depends on a composed
nonlinear function of eight channels, the `L=8` deep stack
V3 achieves classifier accuracy ≥ 0.55 (well above chance)
and **non-regression** relative to W51's `L=6` baseline.
The `family_deep_stack_v3_depth_strict_gain` family;
`mean_acc(L=8) ≥ 0.55` AND `acc(L=8) - acc(L=6) ≥ -0.05`
on mean across seeds — honest about non-monotonicity
under the W47 pure-Python autograd training cost cap
(`W52-L-PURE-PYTHON-TRAINING-COST-CAP`).

### H6 — Role-graph transfer gain (R-102)

On a regime with two distinct ordered role pairs whose
target gates require different transfer patterns, the
role-graph-conditioned transfer achieves mean accuracy
strictly greater than the equal-weight cross-role mixer.
The `family_role_graph_transfer_gain` family;
`(role_graph_acc - equal_weight_acc) ≥ 0.05` on mean across
seeds.

### H7 — Transcript-vs-shared-state under matched budget (R-102)

Under a fixed visible-token budget `B=3` (the W52 quantised
codebook's natural max), the shared-latent arm retains the
target signal at strictly higher cosine than the
transcript-truncation arm on a 12-dim carrier (the
transcript loses the last 9 carrier dimensions, the
shared-latent arm encodes all 12 into 3 visible tokens via
the K1×K2×K3 codebook). The `family_transcript_vs_shared_state`
family; `(shared_retention - transcript_retention) ≥ 0.10`
on mean across seeds.

### H8 — Multi-hop Ollama realism anchor (R-102)

When `COORDPY_W52_OLLAMA_REACHABLE=1` is set and an Ollama
daemon is running, the multi-hop translator probe records
bounded fidelity ≥ 0.50 across at least one direct edge of
the quad. When the env var is absent, the family records
`anchor_status = "synthetic_only"` and `skipped_ok = 1.0`.
The `family_multi_hop_realism_probe` family.

### H9 — W52 envelope verifier soundness (R-102)

The W52 verifier rejects 26+ disjoint forged envelopes
(schema mismatch, persistent V4 chain CID mismatch,
multi-hop witness CID mismatch, deep-stack-v3 forward
witness CID mismatch, quantised compression witness CID
mismatch, long-horizon V4 witness CID mismatch,
branch-cycle V2 witness CID mismatch, role-graph witness
CID mismatch, transcript-comparator witness CID mismatch,
outer-CID mismatch, etc.). Cumulative trust boundary
across W22..W52 = **393 enumerated failure modes** (367
from W22..W51 + 26 new at W52).

### H10 — W52 replay determinism (R-102)

Two independent runs of `W52Team.run` with the same
training set, seed, registry, and observation builder
produce byte-identical `final_output`, root CID, every
`w52_outer_cid`, every per-module witness CID, the
persistent state V4 chain CID list, and the multi-hop
translator witness CID. The `family_w52_replay_determinism`
family; `replay_determinism_ok = 1.0` across all seeds.

### H11 — Multi-hop translator distribution cap (R-102)

Adversarial all-channel forgery + forged quad-backend
tags + forged transitivity training set: the W52
multi-hop translator cannot fully recover the clean
signal. The `family_multi_hop_translator_compromise_cap`
family; `downstream_protect_rate ≥ 0.4` across all seeds —
honest about partial preservation due to identity-friendly
init (the linear-projection translator with identity init
retains a fraction of the clean mapping even when trained
on forged labels, which is itself a robustness property);
proved-conditional limitation
`W52-L-MULTI-HOP-TRANSLATOR-COMPROMISE-CAP`.

### H12 — Role-graph distribution cap (R-102)

Adversarial role-graph edge forgery: the trained
role-graph mixer on forged edges has bounded ability to
recover. The `family_role_graph_distribution_cap` family;
`downstream_protect_rate ≥ 0.6` across all seeds —
proved-conditional limitation
`W52-L-ROLE-GRAPH-COMPROMISE-CAP`.

### H13 — 20-turn long-horizon retention V4 (R-103)

On a length-20 task with a target fact emitted at turn 0,
the W52 V4 stacked persistent state + skip-link + cycle
state recover the fact with cosine ≥ 0.40 at turn 19. The
W51 V3 single-layer baseline drops below 0.20 on the same
regime under pure-Python autograd training. The
`family_long_horizon_v4_retention_20turn` family;
`w52_cosine_at_turn_19 ≥ 0.40` AND
`(w52_cosine - w51_cosine) ≥ 0.15` on mean across seeds.

### H14 — 24-turn long-horizon retention V4 stretch (R-103)

On a length-24 task (deliberately a stretch — we do NOT
expect 1.0), W52 cosine ≥ 0.25 at turn 23 — honest about
drop-off and the pure-Python autograd training cost cap.
The `family_long_horizon_v4_retention_24turn_stretch`
family; `w52_cosine_at_turn_23 ≥ 0.25` on mean across
seeds.

### H15 — Reconstruction V4 MSE at k=8 (R-103)

The trained `LongHorizonReconstructionV4Head` recovers the
flat-feature vector of turn `t-8` from the carrier at turn
`t` with MSE ≤ 0.55 — extending W51 V3 to a longer horizon.
The `family_reconstruction_v4_recovers_t_minus_8` family;
`mse_at_k8 ≤ 0.55` on mean across seeds.

### H16 — Reconstruction V4 MSE at k=12 stretch (R-103)

At the design-maximum lookback k=12, V4 recovers
prior-turn features at MSE ≤ 0.70 — honest about drop-off
and the pure-Python autograd training cost cap. The
`family_reconstruction_v4_k12_stretch` family;
`mse_at_k12 ≤ 0.70` on mean across seeds.

### H17 — Quantised compression ≥ 14 bits/token (R-103)

W52 achieves a structured-bits-per-visible-token ratio
≥ 14.0 on a fixed-length task while retention cosine stays
≥ 0.80. W51 baseline on the same regime hits ≤ 13.0. The
`family_quantised_compression_14bits` family;
`w52_bits_per_token ≥ 14.0` AND `retention_cosine ≥ 0.80`
on mean across seeds.

### H18 — Quantised degradation curve graceful (R-103)

Under decreasing token budgets `{16, 8, 4, 2, 1}`, the
quantised compression achieves a monotone degradation
curve: bits/token does not collapse to chance at any
budget, and retention cosine never drops below 0.40. The
`family_quantised_degradation_curve` family;
`min(bits_per_token over budgets) ≥ 5.0` AND
`min(retention_cosine over budgets) ≥ 0.40`.

### H19 — Branch/cycle memory V2 merge gain (R-103)

On a multi-branch + multi-cycle path where joint
(branch, cycle) pages should be merged, the BCM V2's
trainable merge + evict heads strictly beat W51's V1 BCM
at recovering joint-page facts. The
`family_branch_cycle_memory_v2_merge_gain` family;
`(v2_joint_recall - v1_joint_recall) ≥ 0.10` on mean
across seeds.

### H20 — W52 distribution cap (R-103)

Adversarial all-channel forgery + forged role banks +
forged persistent latent state + forged quantised
codebook + forged quad-backend training set + forged
role-graph edges: the W52 stack cannot recover. The
`family_w52_distribution_cap` family;
`downstream_protect_rate ≥ 0.7` across all seeds —
proved-conditional limitation
`W52-L-QPMHLC-DISTRIBUTION-CAP` (strengthens
`W51-L-PXBLC-DISTRIBUTION-CAP`).

### H21 — Deep-stack V3 overdepth cap reproduces (R-103)

On a 2-step composition regime where W51's `L=6` is
already saturating, the deeper `L=8` stack should NOT
strictly improve and may regress slightly due to
optimisation noise from the extra parameters under the
pure-Python autograd training budget. The
`family_deep_stack_v3_overdepth_cap` family;
`(acc_L8 - acc_L6) ≤ +0.05` on mean across seeds —
honest about non-monotonicity of depth.

### H22 — Quantised rate-floor falsifier (R-103)

Target rate 32 bits/visible-token exceeds the
K1=32 × K2=16 × K3=8 = 4096 codes (≈ 12 bits per pair)
structural ceiling. The
`family_quantised_rate_floor_falsifier` family;
`rate_target_missed = True` across all seeds —
proved-conditional limitation
`W52-L-QUANTISED-RATE-FLOOR-CAP`.

## Falsifiers

* **W52-L-TRIVIAL-W52-PASSTHROUGH** — a trivially-
  configured `W52Registry` reduces to `W51Team.run`
  byte-for-byte; if H1 fails, the trivial-passthrough
  property is falsified.

* **W52-L-CROSS-TOKENIZER-QUAD-CAP** — when the realism
  probe is skipped (Ollama unreachable), the multi-hop
  quad cross-tokenizer transitivity conjecture remains
  carried forward, witnessed by H8's skip path.

* **W52-L-MULTI-HOP-TRANSLATOR-COMPROMISE-CAP** —
  adversarial all-channel forgery + forged quad-backend
  training: the trained translator cannot recover.
  Reproduces honestly in R-102.

* **W52-L-ROLE-GRAPH-COMPROMISE-CAP** — adversarial
  role-graph edge forgery: the trained role-graph mixer
  cannot recover.

* **W52-L-DEEP-STACK-V3-OVERDEPTH-CAP** — L=8 stack does
  NOT strictly improve over L=6 on shallow composition
  regimes; bounds the depth claim.

* **W52-L-LONG-HORIZON-V4-DROPOFF-CAP** — at k > 12 the
  V4 reconstruction head's MSE rises above 0.70; bounds
  the reconstruction claim.

* **W52-L-QUANTISED-RATE-FLOOR-CAP** — target rate 32
  bits/visible-token exceeds the K1=32 × K2=16 × K3=8
  quantised codebook's information capacity; rate is
  missed or retention floor breaks.

* **W52-L-QPMHLC-DISTRIBUTION-CAP** — adversarial
  all-channel forgery + forged role banks + forged
  persistent latent state V4 + forged quantised codebook
  + forged quad-backend training + forged role-graph
  edges: the trained W52 stack cannot recover.
  Strengthens `W51-L-PXBLC-DISTRIBUTION-CAP`.

* **W52-L-NO-REAL-KV-CAP** — the W52 stack still does not
  touch transformer-internal KV bytes; persistent state V4
  + multi-hop translator + role-graph transfer +
  branch/cycle memory V2 does not change this.

* **W52-L-PURE-PYTHON-TRAINING-COST-CAP** — the
  pure-Python autograd engine carries forward from W47;
  W52 stage-wise training cost is approximately the sum
  of per-module costs.

* **W52-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** — carries
  forward from W48/W49/W50/W51 — real LLMs may or may not
  condition on the W52 carrier or persistent state.
  Synthetic anchor is load-bearing; H8 is a best-effort
  anchor on Ollama.

## Per-component verdicts (preview)

* **Persistent latent state V4 (M1)** —
  *behaviourally useful* if H2 + H13 + H14 pass;
  *structurally useful* always (chain-walkable V4 state CID).
* **Multi-hop backend translator (M2)** —
  *behaviourally useful* if H3 + H4 + H8 pass under
  synthetic primary; *structurally useful* always
  (length-3 transitivity witness CID).
* **Deep proxy stack V3 (M3)** — *behaviourally useful*
  if H5 passes on deep composition; *structurally useful*
  always (per-layer + per-head + per-role-bank CIDs).
* **Quantised compression V4 (M4)** — *behaviourally
  useful* on H17 (14 bits/token); *structurally useful*
  (round-trips through coarse + fine + ultra codebook CIDs).
* **Long-horizon reconstruction V4 (M5)** — *behaviourally
  useful* on H15 + H16; *structurally useful* always
  (degradation curve recorded).
* **Branch/cycle memory V2 (M6)** — *behaviourally useful*
  on H19; *structurally useful* (per-page + per-merge CIDs).
* **Role-graph transfer (M7)** — *behaviourally useful*
  on H6; *structurally useful* (per-edge CIDs).
* **Transcript-vs-shared-state comparator (M8)** —
  *behaviourally useful* on H7; *structurally useful*
  always (matched-budget arm CIDs).
* **Deep stack V3 overdepth cap** — *limitation reproduces
  honestly*.
* **Quantised rate-floor cap** — *limitation reproduces
  honestly*.
* **Multi-hop compromise cap** — *limitation reproduces
  honestly*.
* **Role-graph compromise cap** — *limitation reproduces
  honestly*.
* **QPMHLC distribution cap** — *limitation reproduces
  honestly*.

## Architecture triage

| Frontier candidate                                       | W52 bucket                                              | Verdict |
|---|---|---|
| Stacked two-layer persistent state V4 with skip-link     | **trainable now (longer-horizon retention)**            | shipped |
| Multi-hop quad-backend translator (length-3 transitivity) | **trainable now (cross-backend transitivity)**          | shipped |
| Cross-tokenizer quad-backend transitivity                | **substrate-blocked**                                   | carry-forward |
| Deeper proxy stack (L=8 + role banks + residual gate)    | **transformer-proxy now (depth + role specialisation)** | shipped |
| Three-level quantised codebook (K1×K2×K3)                | **trainable now (compression)**                         | shipped |
| Three-headed reconstruction (causal + branch + cycle, max_k=12) | **trainable now (reconstruction)**               | shipped |
| Branch/cycle memory V2 (trainable merge + evict + joint pages) | **trainable now (memory)**                        | shipped |
| Role-graph conditioned cross-role transfer               | **trainable now (role-aware transfer)**                 | shipped |
| Transcript-vs-shared-state matched-budget comparator     | **measurable now (compression-vs-performance ablation)**| shipped |
| Quad-backend Ollama realism anchor                       | **anchor probe (best-effort)**                          | shipped (conditional on env) |
| Real KV-cache pooling across turns                       | **substrate-blocked**                                   | unchanged |
| Transformer-internal mixed-curvature attention           | **substrate-blocked**                                   | unchanged |
| True hidden-state / KV sharing                           | **substrate-blocked**                                   | unchanged |
| Multi-host shared-state transfer                         | **substrate-blocked**                                   | unchanged |
| GPU/CUDA-backed autograd                                 | **substrate-blocked (deliberately deferred)**           | unchanged |

## What W52 explicitly does NOT do

* W52 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
  `W48-C-DEEP-TRANSFORMER-COUPLING`,
  `W49-C-DEEP-TRANSFORMER-COUPLING`,
  `W50-C-DEEP-TRANSFORMER-COUPLING`,
  `W51-C-DEEP-TRANSFORMER-COUPLING`, or any other
  W43..W51 substrate-blocked direction.
* W52 does NOT transplant real KV-cache bytes. The
  V4 persistent state + role-graph + branch/cycle V2
  memory operate on capsule-layer abstractions only.
* W52 does NOT close
  `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`. The
  multi-hop translator **bounds** the conjecture by
  training capsule-layer length-3 transitivity; the
  conjecture is sharpened forward as
  `W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY`.
* W52 does NOT claim multi-host coupling.
* W52 does NOT claim training-data-free generalisation.
  Each W52 module is trained on hermetic synthetic
  banks pre-committed in the R-102 / R-103 sources.
* W52 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`,
  `W48-C-REAL-KV-COUPLED-PROXY`, or
  `W48-C-MULTI-HOST-SHARED-STATE`.
* W52 does NOT ship CUDA / GPU support.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`.
  The W52 modules ship at `coordpy.persistent_latent_v4`,
  `coordpy.multi_hop_translator`,
  `coordpy.deep_proxy_stack_v3`,
  `coordpy.quantised_compression`,
  `coordpy.long_horizon_retention_v4`,
  `coordpy.branch_cycle_memory_v2`,
  `coordpy.role_graph_transfer`,
  `coordpy.transcript_vs_shared_state`, and
  `coordpy.w52_team` — reachable only through explicit
  imports — same convention as W43..W51.

## New theorem-style claims (preview)

* **W52-T-PERSISTENT-V4-STACKED-SOUNDNESS** (proved by
  inspection + mechanically-checked + empirical) — the
  two-layer stacked GRU + skip-link is a deterministic
  fixed point at trivial parameters.
* **W52-T-MULTI-HOP-QUAD-TRANSITIVITY-LEN3** (proved-
  conditional + empirical) — under the trained joint loss,
  the length-3 transitivity gap is bounded.
* **W52-T-DISAGREEMENT-WEIGHTED-ARBITRATION-WIN** (proved-
  conditional + empirical) — disagreement-weighted
  arbitration strictly beats naive arbitration under a
  perturbed-edge regime.
* **W52-T-DEEP-STACK-V3-L8-STRUCTURAL** (proved-
  conditional + empirical) — `L=8` deep stack V3
  structural floor + non-regression vs `L=6`.
* **W52-T-ROLE-GRAPH-TRANSFER-GAIN** (proved-conditional
  + empirical) — role-graph-conditioned transfer strictly
  beats equal-weight transfer under direction-dependent
  regimes.
* **W52-T-TRANSCRIPT-VS-SHARED-STATE-WIN** (proved-
  conditional + empirical) — shared-latent retention >
  transcript truncation retention under matched visible
  budget.
* **W52-T-QUANTISED-COMPRESSION-RATE-BOUND** (proved-
  conditional + empirical) — under K1=32 × K2=16 × K3=8
  quantised codebook + adaptive budget gate + retention
  floor ≥ 0.80, bits/token ≥ 14.0.
* **W52-T-RECONSTRUCTION-V4-K8-CORRECTNESS** (proved-
  conditional + empirical) — V4 head recovers prior-turn
  features at MSE ≤ 0.55 for k ≤ 8.
* **W52-T-RECONSTRUCTION-V4-K12-DROPOFF** (proved-
  conditional + empirical) — at k = 12 the V4
  reconstruction MSE is bounded ≤ 0.70.
* **W52-T-BRANCH-CYCLE-V2-MERGE-EVICT-AUDIT** (proved by
  inspection + mechanically-checked) — every merge and
  evict produces an audit-trail CID; joint
  (branch, cycle) pages are content-addressed.
* **W52-T-W52-OUTER-CID-CHAIN** (proved by inspection +
  mechanically-checked) — `w47 → w48 → w49 → w50 → w51
  → w52` chain verifies; tamper detected at every link.
* **W52-T-TRIVIAL-PASSTHROUGH-BYTE-IDENTICAL** (proved by
  inspection + empirical) — trivial `W52Team`
  byte-identical to W51.
* **W52-T-PERSISTENT-V4-CHAIN-WALK-24** (proved by
  inspection + mechanically-checked) — chain-walk depth
  24 from any V4 leaf.
* **W52-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked) — the W52 verifier enumerates 26
  disjoint failure modes; cumulative trust boundary
  across W22..W52 = 393 modes.

* **W52-L-TRIVIAL-W52-PASSTHROUGH** (proved by inspection
  + empirical) — trivial W52 = W51 byte-for-byte.
* **W52-L-CROSS-TOKENIZER-QUAD-CAP** (carries forward,
  sharpened from `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`).
* **W52-L-MULTI-HOP-TRANSLATOR-COMPROMISE-CAP** (proved-
  conditional limitation) — adversarial quad-backend
  forgery cannot be recovered.
* **W52-L-ROLE-GRAPH-COMPROMISE-CAP** (proved-conditional
  limitation) — adversarial role-graph forgery cannot be
  recovered.
* **W52-L-DEEP-STACK-V3-OVERDEPTH-CAP** (proved by
  inspection + empirical) — `L=8` does not strictly
  improve over `L=6` on shallow regimes.
* **W52-L-LONG-HORIZON-V4-DROPOFF-CAP** (proved-
  conditional limitation) — at `k > 12` the V4
  reconstruction MSE rises.
* **W52-L-QUANTISED-RATE-FLOOR-CAP** (proved-
  conditional limitation) — 32-bit target exceeds
  quantised codebook capacity.
* **W52-L-QPMHLC-DISTRIBUTION-CAP** (proved-
  conditional limitation, strengthens W51) —
  adversarial all-channel forgery + W52-specific forgery:
  cannot recover.
* **W52-L-NO-REAL-KV-CAP** (carries forward, strengthens
  W51) — V4 persistent state + multi-hop translator +
  role-graph + branch/cycle V2 does not transplant real
  KV bytes.
* **W52-L-PURE-PYTHON-TRAINING-COST-CAP** (carries
  forward) — stage-wise W52 training cost is the sum of
  per-module costs.
* **W52-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (carries
  forward from W48/W49/W50/W51).

* **W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY** (sharper
  than `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`) —
  capsule-layer length-3 transitivity is now trained and
  auditable; tokenizer-level transitivity remains
  carry-forward.
* **W52-C-DEEP-TRANSFORMER-COUPLING** (carries forward,
  bounds W47/W48/W49/W50/W51 further) — full
  transformer-internal hidden-state + KV-cache coupling
  remains substrate-blocked.
* **W52-C-REAL-KV-COUPLED-PROXY** (carries forward
  W48-C unchanged).
* **W52-C-MULTI-HOST-SHARED-STATE** (carries forward
  W48-C unchanged).
* **W52-C-LIVE-MULTI-HOST-AUTOGRAD** (carries forward
  W47-C unchanged).

## What this enables for the programme

* **Strengthens** every W47..W51 carry-forward by adding
  a stacked persistent latent state V4 + multi-hop
  quad-backend translator + deeper L=8 proxy stack +
  three-level quantised compression + long-horizon
  reconstruction V4 + branch/cycle memory V2 with
  merge/evict + role-graph conditioned transfer +
  transcript-vs-shared-state matched-budget comparator —
  eight orthogonal advances.
* **Strengthens** `W51-L-PXBLC-DISTRIBUTION-CAP` to
  include W52-specific forgery
  (`W52-L-QPMHLC-DISTRIBUTION-CAP`).
* **Sharpens** `W51-C-CROSS-TOKENIZER-TRIPLE-TRANSITIVITY`
  to `W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY` — the
  multi-hop translator + length-3 transitivity loss is
  now trained and auditable; only tokenizer-level
  transitivity remains carried forward.
* **Preserves** all of W43..W51's deterministic-audit
  properties — the W52 modules are strictly additive.
* **Does not close** the substrate-blocked W43..W51
  conjectures. The honest summary: W52 is the strongest
  *executable proxy* available without substrate access,
  with a quad-backend realism anchor when reachable.

## Storyline post-W52

* **W43** — executable product-manifold capsules
* **W44** — live manifold-conditioned coordination
* **W45** — learned manifold controller
* **W46** — manifold memory controller
* **W47** — autograd manifold stack (pure-Python AD + Adam)
* **W48** — shared-state transformer-proxy (pseudo-KV bank)
* **W49** — multi-block cross-bank coordination (L=2 proxy)
* **W50** — cross-backend latent coordination (5 advances)
* **W51** — persistent cross-backend latent coordination
  (6 advances: GRU persistent state + triple-backend
  translator + L=6 deep stack V2 + hierarchical
  compression K1=32 × K2=16 + long-horizon reconstruction
  V3 max_k=8 + branch/cycle-specialised memory head)
* **W52** — quantised persistent multi-hop latent
  coordination (8 advances: stacked persistent latent
  state V4 + multi-hop quad-backend translator (length-3
  transitivity) + L=8 deep stack V3 with role banks +
  quantised compression K1=32 × K2=16 × K3=8 + long-
  horizon reconstruction V4 max_k=12 + branch/cycle
  memory V2 with merge/evict + role-graph conditioned
  transfer + transcript-vs-shared-state matched-budget
  comparator)

## Done = the following commits land

1. `coordpy/persistent_latent_v4.py` — pure Python / stdlib.
2. `coordpy/multi_hop_translator.py` — pure Python / stdlib.
3. `coordpy/deep_proxy_stack_v3.py` — pure Python / stdlib.
4. `coordpy/quantised_compression.py` — pure Python / stdlib.
5. `coordpy/long_horizon_retention_v4.py` — pure Python / stdlib.
6. `coordpy/branch_cycle_memory_v2.py` — pure Python / stdlib.
7. `coordpy/role_graph_transfer.py` — pure Python / stdlib.
8. `coordpy/transcript_vs_shared_state.py` — pure Python / stdlib.
9. `coordpy/w52_team.py` — pure Python / stdlib (composition).
10. `coordpy/r102_benchmark.py` — dependency-free benchmark
    family (12 families).
11. `coordpy/r103_benchmark.py` — dependency-free benchmark
    family (10 families).
12. `tests/test_persistent_latent_v4_w52.py`
13. `tests/test_multi_hop_translator_w52.py`
14. `tests/test_deep_proxy_stack_v3_w52.py`
15. `tests/test_quantised_compression_w52.py`
16. `tests/test_long_horizon_retention_v4_w52.py`
17. `tests/test_branch_cycle_memory_v2_w52.py`
18. `tests/test_role_graph_transfer_w52.py`
19. `tests/test_transcript_vs_shared_state_w52.py`
20. `tests/test_w52_trivial_passthrough_byte_identical.py`
21. `tests/test_w52_team_envelope_chain.py`
22. `tests/test_r102_benchmark.py`
23. `tests/test_r103_benchmark.py`
24. `examples/w52_smoke_driver.py`
25. `examples/w52_replay_live.py` (Ollama best-effort
    quad anchor)
26. `docs/RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`
    and this success-criterion file.
27. Updates to `docs/RESEARCH_STATUS.md`,
    `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
    `docs/HOW_NOT_TO_OVERSTATE.md`,
    `docs/context_zero_master_plan.md`,
    `papers/context_as_objects.md`, and `CHANGELOG.md`.
