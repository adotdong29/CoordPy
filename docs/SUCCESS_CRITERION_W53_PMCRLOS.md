# Success criterion — W53 Persistent Mergeable Corruption-Robust Latent Operating System

> Pre-committed, falsifiable bar for the post-W52 W53 milestone
> (PMCRLOS — Persistent Mergeable Corruption-Robust Latent
> Operating System). 2026-05-12.
>
> If any H-bar fails on the released-bench at 3 seeds, that bar
> is **falsified** and the corresponding ``W53-L-…`` cap is
> recorded in `docs/THEOREM_REGISTRY.md`.

## TL;DR

W53 stacks **ten orthogonal mechanism advances** on top of W52
into a capsule-native latent operating system that is
**persistent, mergeable, corruption-robust, budget-aware,
cross-backend, auditable, and behaviourally useful** at the
capsule layer. It does NOT touch transformer-internal state.

The ten advances:

* **M1 Persistent Latent State V5** — 3-layer GRU stack with
  *persistent* skip-link (applied at every step) plus a
  state-merge head; chain walks past 32 turns
* **M2 Multi-Hop Translator V3** — 5-backend (A,B,C,D,E)
  translator with chain-length-4 transitivity and
  *uncertainty-aware arbitration* that returns
  per-dim 1-sigma confidence intervals
* **M3 Mergeable Latent State Capsule (MLSC)** — the
  load-bearing new abstraction: content-addressed mergeable
  capsules with an explicit ``MergeOperator`` + content-
  addressed ``MergeAuditTrail``; supports K-of-N consensus
  quorum with abstain semantics
* **M4 Deep Proxy Stack V4** — depth-10 with merge-aware head
  + corruption-aware head (per-layer L2-pathology detection)
* **M5 ECC Codebook V5** — K1×K2×K3×K4 = 16384 codes with
  *XOR parity* per segment; ≥ 14.5 bits/visible-token target
* **M6 Long-Horizon Reconstruction V5** — 4 heads (causal +
  branch + cycle + merged-branch); max_k = 16; degradation
  curve probe to k=32
* **M7 Branch Merge Memory V3** — consensus pages with K-of-N
  quorum + content-addressed consensus audit + abstain when
  quorum unreached
* **M8 Corruption-Robust Carrier** — composes ECC parity +
  3-of-3 majority repetition over the bits payload; reports
  detect / partial-correct / abstain / silent-failure rates
* **M9 Transcript-vs-Shared Arbiter V2** — explicit per-turn
  policy over {transcript, shared, abstain} with confidence-
  triggered fallback; 3-arm comparison with oracle-correctness
* **M10 Uncertainty / Confidence Layer** — composite confidence
  scalar over all 5 component signals + a *calibration check*
  (high-conf strictly more accurate than low-conf)

W53 is the strongest *executable proxy* available at the
capsule layer with one **best-effort** real-LLM realism anchor
(quint-backend). It does NOT touch real KV bytes, hidden states,
attention weights, embeddings, or real tokenizers.

## H-bars (34 total)

H-bars are split into three benchmark families: R-104 (H1-H12),
R-105 (H13-H22), R-106 (H23-H34).

### R-104 — Persistent / Multi-Hop / Mergeable family (12 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H1 | family_trivial_w53_passthrough | trivial W53 reduces to W52 byte-for-byte |
| H2 | family_persistent_v5_long_horizon_gain | V5 24-turn cosine recall ≥ 0.5 on corrupted regime |
| H3 | family_quint_chain_len4_transitivity | 5-backend chain-length-4 fidelity ≥ 0.7 |
| H4 | family_uncertainty_arbitration_gain | uncertainty-aware ≥ naive arbitration under perturbed edge |
| H5 | family_mlsc_consensus_quorum | consensus reached on consistent branches AND abstained on random |
| H6 | family_deep_stack_v4_corruption_aware | corruption flag fires on pathological input only |
| H7 | family_w53_envelope_verifier | W53 envelope verifier rejects forged envelopes |
| H8 | family_w53_replay_determinism | W53 replay byte-identical across two runs |
| H9 | family_quint_translator_compromise_cap | forged quint backend → translator cannot recover |
| H10 | family_uncertainty_layer_calibration | calibration_gap ≥ 0.10 on separable confidence/accuracy |
| H11 | family_mlsc_audit_trail_integrity | merge audit walks back to root CIDs without orphans |
| H12 | family_quint_realism_probe | quint anchor synthetic-only when Ollama unreachable |

### R-105 — Long-Horizon Retention / Reconstruction / Cramming family (10 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H13 | family_persistent_v5_28turn | 28-turn V5 cosine recall ≥ 0.5 |
| H14 | family_persistent_v5_32turn_stretch | 32-turn V5 stretch cosine ≥ 0.25 |
| H15 | family_lhr_v5_recovers_t_minus_12 | V5 MSE at k=12 ≤ 0.55 |
| H16 | family_lhr_v5_k16_stretch | V5 MSE at k=16 ≤ 0.80 stretch |
| H17 | family_ecc_compression_14p5_bits | ECC ≥ 14.5 bits/visible-token at full emit |
| H18 | family_lhr_v5_degradation_curve | min MSE in well-trained range (k ≤ 12) ≤ 1.0 |
| H19 | family_w53_distribution_cap | combined V5 forge → protect_rate ≥ 0.50 mean |
| H20 | family_deep_v4_overdepth_cap | L=10 V4 doesn't strictly improve over L=8 V3 on shallow regime (cap reproduces) |
| H21 | family_ecc_rate_floor_falsifier | 40-bit target structurally missed by codebook |
| H22 | family_arbiter_strict_dominance | TVS arbiter V2 oracle-correctness rate ≥ 0.5 |

### R-106 — Corruption / Hostile-Channel / Consensus-Merge family (12 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H23 | family_single_bit_detect_rate | single-bit corruption detect rate ≥ 0.80 |
| H24 | family_single_bit_correction_rate | single-bit partial correction rate ≥ 0.30 |
| H25 | family_two_bit_graceful_degrade | 2-bit corruption → abstain ≥ 0.50 AND silent_failure ≤ 0.30 |
| H26 | family_consensus_recall_kof2 | BMM V3 consensus recall ≥ 0.70 with K=2-of-N |
| H27 | family_consensus_abstain_when_disagreed | BMM V3 abstains when quorum K too high |
| H28 | family_mlsc_merge_replay_determinism | same parents → same merged CID across two runs |
| H29 | family_perturbed_edge_uncertainty_report | perturbed translator edge → arbitration uncertainty rises |
| H30 | family_compromise_v5_persistent_state | forged V5 train → protect_rate ≥ 0.50 mean |
| H31 | family_corruption_robust_carrier_safety | silent_failure_rate ≤ 0.10 across single-bit flips |
| H32 | family_uncertainty_calibration_under_noise | calibration holds under per-component noise |
| H33 | family_persistent_v5_chain_walk_depth | V5 chain walks back ≥ 16 turns |
| H34 | family_w53_integration_envelope | W53 envelope binds all required CIDs |

## Falsifiers and limitation reproductions (the W53 caps)

* **W53-L-TRIVIAL-W53-PASSTHROUGH** (H1) — any divergence in
  W53 trivial passthrough reproduction is a fatal regression.
* **W53-L-OVERDEPTH-V4** (H20) — L=10 V4 must NOT strictly
  improve over L=8 V3 on a shallow 2-step regime; a positive
  delta reproduces the depth cap.
* **W53-L-ECC-RATE-FLOOR** (H21) — the 4-level codebook has
  capacity ~14 bits/triple; a 40-bit target is structurally
  unachievable.
* **W53-L-CRC-TWO-BIT-PATHOLOGY** (H25) — XOR parity detects
  single-bit flips but cannot detect or correct all 2-bit
  flips; silent failure is bounded but nonzero.
* **W53-L-MULTI-HOP-V3-COMPROMISE-CAP** (H9) — a forged
  quint training set means the translator cannot recover
  clean-direction fidelity; protect rate is honestly bounded.
* **W53-L-V5-DISTRIBUTION-CAP** (H19, H30) — adversarial V5
  training scrambles the persistent state; protect rate is
  honestly bounded ≤ 1.0.
* **W53-L-NO-REAL-KV-CAP** (extends W52-L-NO-REAL-KV-CAP) —
  W53 does not touch real transformer-internal hidden states,
  KV cache bytes, attention weights, embeddings, or real
  tokenizers. The persistent V5 state is a capsule-layer
  carrier evolving across turns.
* **W53-L-PURE-PYTHON-TRAINING-COST-CAP** (extends W52) —
  pure-Python autograd; per-module training cost grows as
  ``O(n_params × n_examples)``. Long-horizon V5 fits use
  truncated BPTT.
* **W53-C-CROSS-TOKENIZER-QUINT-CAP** — sharpens
  W52-C-CROSS-TOKENIZER-QUAD-TRANSITIVITY to 5 backends;
  capsule-layer transitivity holds; behavioural transitivity
  across genuinely different tokenizers still requires
  backend-side adapters.

## Stable-boundary preservation

* `coordpy.__version__` remains `0.5.20`. **No bump.**
* `SDK_VERSION` unchanged.
* No PyPI release.
* W53 modules ship at explicit-import paths only:
  `coordpy.persistent_latent_v5`, `coordpy.multi_hop_translator_v3`,
  `coordpy.mergeable_latent_capsule`, `coordpy.deep_proxy_stack_v4`,
  `coordpy.ecc_codebook_v5`, `coordpy.long_horizon_retention_v5`,
  `coordpy.branch_merge_memory_v3`, `coordpy.corruption_robust_carrier`,
  `coordpy.transcript_vs_shared_arbiter_v2`, `coordpy.uncertainty_layer`,
  `coordpy.w53_team`.
* SDK contract is byte-for-byte unchanged.

## Strong success / partial success / failure

* **Strong success**: 32+/34 H-bars pass at 3 seeds.
* **Partial success**: 24+/34 H-bars pass at 3 seeds with
  identified, reproducible caps for the failures.
* **Failure**: < 24 H-bars pass; the milestone is rolled back.

## Anchor status

The H12 quint realism probe reuses the W52 quad anchor scaffold
plus a synthetic 5th backend tag E. When
`COORDPY_W53_OLLAMA_REACHABLE=1` and Ollama is reachable, the
inner W52 quad anchor runs; W53 inherits the anchor result and
records `multi_hop_anchor_status` from the inner W52 envelope.
When Ollama is unreachable, the W53 envelope still seals
byte-identically with `anchor_status: synthetic_only`.
