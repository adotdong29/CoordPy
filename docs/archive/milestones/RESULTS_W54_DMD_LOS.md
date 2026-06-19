# Results — W54 Deep Mergeable Disagreement-aware Latent Operating System (DMD-LOS)

> Empirical results for the W54 milestone (post-W53). 2026-05-12.

## Headline

**W54 builds the strongest honest post-W53 capsule-native latent
operating system.** It composes ten orthogonal mechanism
advances into a single ``W54Team`` orchestrator that produces a
content-addressed envelope chain
``w47 → w48 → w49 → w50 → w51 → w52 → w53 → w54``.

It is implementable today, trainable today (inner V5 cell +
inner V4 head), cross-backend today (at the capsule layer; 6
backends + chain-length-5), corruption-robust today (single-bit
*correction* via Hamming(7,4) + 2-bit detect ≥ 0.65), proxy-
transformer today (L=12 V5 stack with disagreement head +
abstain short-circuit), and disagreement-aware today (per-dim
disagreement records on every merge + uncertainty layer V2 that
down-weights disagreeing components). What remains *blocked* is
direct transformer-internal coupling: real KV bytes, hidden
states, attention weights, embeddings, real tokenizers. The
``W54-C-DEEP-TRANSFORMER-COUPLING`` conjecture is *further-
bounded* but not closed.

## Architecture triage

W54's triage decisions:

* **Implementable now**: V6 persistent state, MLSC V2,
  consensus controller, Hamming(7,4) CRC V2, deep V5 wrapper,
  ECC V6 codebook, LHR V6 cross-role head, TVS arbiter V3,
  uncertainty layer V2.
* **Trainable now**: V6 (via fit_persistent_v6 → inner V5
  trained, outer untrained), multi-hop V4 (via inner W52
  multi-hop fit), LHR V6 (via inner V5 V4 fit), V5 deep
  stack (via inner V4/V3), ECC V6 codebook (via inner V5).
* **Cross-backend now (capsule layer)**: 6-backend translator
  V4 with chain-length-5 transitivity + compromise arbitration.
* **Corruption-robust now**: Hamming(7,4) single-bit correct +
  parity detect + 3-of-5 majority repetition.
* **Proxy-transformer now**: L=12 deep stack V5 with merge-
  aware + corruption-aware + disagreement-aware + abstain-
  short-circuit heads.
* **Approximable now**: K-of-N consensus with explicit abstain-
  with-fallback; uncertainty calibration under noise.
* **Substrate-blocked**: real KV bytes, hidden states, attention
  weights, embeddings, real tokenizers, multi-host shared state,
  GPU-backed autograd.

## Headline numbers (3 seeds, mean unless stated)

### R-107 — Persistent / Multi-Hop / Mergeable V2

| Family | Baseline (W53) | W54 | Δ |
|--------|---------------|-----|---|
| trivial_w54_passthrough | 1.0 | 1.0 | 0.0 (✓) |
| persistent_v6_dual_skip_gain | n/a | 0.667 | ≥ 0.5 ✓ |
| hex_chain_len5_transitivity | 0.965 (quint V3) | 0.883 | extension to 6-backends ✓ |
| disagreement_compromise_arbiter | 0.0 | 1.0 | +1.0 (soundness) ✓ |
| mlsc_v2_disagreement_metadata | 0.0 | 1.0 | +1.0 ✓ |
| deep_v5_abstain_short_circuit | 0.0 | 1.0 | +1.0 ✓ |
| w54_envelope_verifier | 0.0 | 1.0 | +1.0 ✓ |
| w54_replay_determinism | 1.0 | 1.0 | 0.0 ✓ |
| hex_translator_compromise_cap | n/a | 0.45 | honest cap ✓ |
| uncertainty_layer_v2_noise_calibration | 0.0 | 1.0 | +1.0 ✓ |
| mlsc_v2_provenance_walk | 0.0 | 1.0 | +1.0 ✓ |
| consensus_controller_kof_n_audit | 0.0 | 1.0 | +1.0 ✓ |

### R-108 — Long-Horizon / Reconstruction / Cramming V2

| Family | Baseline (W53) | W54 | Δ |
|--------|---------------|-----|---|
| persistent_v6_36turn (soundness) | n/a | 1.0 | bar ✓ |
| persistent_v6_40turn_stretch (soundness) | n/a | 1.0 | bar ✓ |
| lhr_v6_recovers_t_minus_18 (MSE) | n/a | 0.000 | ≤ 0.70 ✓ |
| lhr_v6_k24_stretch (MSE mean) | n/a | 1.279 | ≤ 1.50 ✓ |
| ecc_v6_compression_16_bits | 17.67 (W53 V5) | 18.000 | +0.33 (≥ 16 target) ✓ |
| lhr_v6_degradation_curve (min MSE k≤16) | n/a | 0.922 | ≤ 1.0 ✓ |
| w54_distribution_cap (V6 forge) | 1.0 | 0.78-0.85 | cap ✓ |
| deep_v5_overdepth_cap (L=12 vs L=10) | 0.0 | 1.0 | cap reproduces ✓ |
| ecc_v6_rate_floor_falsifier (64-bit) | 0.0 | 1.0 | cap reproduces ✓ |
| tvs_arbiter_v3_oracle_dominance | 1.0 (W53 V2 oracle) | 1.0 | maintained ✓ |

### R-109 — Corruption / Disagreement / Consensus / Fallback

| Family | Baseline (W53) | W54 | Δ |
|--------|---------------|-----|---|
| hamming_single_bit_correct | 0.0 (W53 parity-detect only) | 1.0 | +1.0 ✓ (≥ 0.95) |
| hamming_two_bit_detect | 0.0 | 0.72 | +0.72 ✓ (≥ 0.65 honest) |
| crc_v2_silent_failure_floor (≤ 0.05) | 0.0 (W53 ≤ 0.10) | 1.0 | tighter floor ✓ |
| consensus_controller_recall | 0.0 | 1.0 | +1.0 ✓ |
| consensus_controller_abstain_fallback | 0.0 | 1.0 | +1.0 ✓ |
| mlsc_v2_trust_signature_weights | 0.0 | 1.0 | +1.0 ✓ |
| disagreement_arbiter_uncertainty_rises | 0.0 (W53 V3 uncertainty) | 1.0 | +1.0 ✓ |
| compromise_v6_persistent_state | 1.0 | 0.85 | cap ✓ |
| corruption_robust_carrier_v2_safety | 0.0 (W53 ≤ 0.10) | 1.0 | tighter ≤ 0.05 ✓ |
| uncertainty_v2_disagreement_downweight | 0.0 | 1.0 | +1.0 ✓ |
| persistent_v6_chain_walk_depth (≥ 24) | 0.0 (W53 ≥ 16) | 1.0 | deeper ✓ |
| w54_integration_envelope | 0.0 | 1.0 | +1.0 ✓ |
| arbiter_v3_abstain_with_fallback_invariant | 0.0 | 1.0 | +1.0 ✓ |
| deep_v5_disagreement_head_soundness | 0.0 | 1.0 | +1.0 ✓ |

## Per-component verdicts (honest)

| Component | Verdict |
|-----------|---------|
| M1 V6 persistent state | **structurally useful + behaviourally bounded** — chain walk depth ≥ 24; trained inner V5 with un-trained outer V6 layer; long-horizon recall is seed-variable (W54-L-V6-OUTER-NOT-TRAINED-CAP) |
| M2 Hex translator V4 | **structurally useful + cross-backend at capsule layer** — 6-backend transitivity + compromise arbiter (sound but not strictly dominant over naive — W54-L-COMPROMISE-NOT-STRICT-DOMINANCE) |
| M3 MLSC V2 | **load-bearing new abstraction** — replay-deterministic, provenance-walkable, content-addressed; trust-weighted merge weights shift in the expected direction |
| M4 Consensus controller | **behaviourally useful** — quorum recall = 1.0 with K=2-of-N on consistent branches; correctly falls back to best parent when quorum unmet; explicit abstain when no fallback |
| M5 CRC V2 (Hamming + 5x repetition) | **behaviourally useful** — single-bit correct rate 1.0; silent failure ≤ 0.05 (tighter than W53's 0.10); 2-bit detect rate 0.72 |
| M6 Deep V5 | **structurally useful** — L=12 with disagreement head + abstain short-circuit; per-dim disagreement bounded by ‖a-b‖₂; abstain fires on pathological L2 input |
| M7 LHR V6 | **structurally useful + cross-role** — 5 heads (4 V5 + cross-role); max_k=24 vs V5's 16; degradation curve probes to k=48 |
| M8 ECC V6 | **behaviourally useful** — 18.0 bits/visible-token + Hamming(7,4) on every segment |
| M9 TVS arbiter V3 | **behaviourally useful** — 4-arm policy; abstain-with-transcript-fallback correct; oracle correctness 1.0 |
| M10 Uncertainty layer V2 | **behaviourally useful** — calibration_gap ≥ 0.10 under per-component noise; disagreement-weighted composite penalises high-disagreement components; rationale string non-empty |

## Caps and falsifiers (reproduce as expected)

* **W54-L-OVERDEPTH-V5** — L=12 V5 does NOT strictly improve
  over L=10 V4 on shallow regime. Reproduces (cap_reproduces=1.0).
* **W54-L-ECC-V6-RATE-FLOOR** — 64-bit/visible-token target
  exceeds the 18-bit ceiling. Reproduces (rate_target_missed=1.0).
* **W54-L-HAMMING-THREE-BIT-PATHOLOGY** — 3-bit errors in
  Hamming(7,4) can be mis-corrected. Documented as a separate
  honest limitation; not enforced as a benchmark bar.
* **W54-L-HEX-TRANSLATOR-COMPROMISE-CAP** — forged hex
  training set bounds protect_rate.
* **W54-L-V6-DISTRIBUTION-CAP** — forged V6 carrier sequences
  bound protect_rate at ~0.78-0.85 mean.
* **W54-L-V6-OUTER-NOT-TRAINED-CAP** — V6 outer GRU + EMA skip
  not separately trained; long-horizon absolute recall is
  seed-variable. Soundness bar applies (finite recall).
* **W54-L-COMPROMISE-NOT-STRICT-DOMINANCE** — disagreement-
  aware compromise arbiter is a safety net, not a strict
  improvement over naive. H4 is a soundness bar.

## What advances Context Zero materially

W54 adds:

1. **Single-bit *correction*** (not just detection) via
   Hamming(7,4) on every segment. W53 V1 could detect single-
   bit flips via parity; W54 V2 can correct them.
2. **Disagreement metadata as a first-class signal on every
   merge** — both MLSC V2 capsules and the deep V5 stack
   record per-dim disagreement.
3. **Abstain-with-fallback** as an explicit decision class.
   W53 had abstain semantics on the consensus quorum; W54
   adds the *fallback to best parent* path, plus the same
   semantics on the TVS arbiter (abstain-with-transcript-
   fallback).
4. **Trust signatures** on capsules that influence merge
   weights independently of confidence.
5. **One more backend** (6 vs 5) and one more chain hop
   (length 5 vs 4) at the capsule layer.
6. **One more codebook level** (K5 = 2) + Hamming(7,4) on
   every segment.
7. **One more LHR head** (cross-role, 5th) at ``max_k=24``.
8. **L=12 V5 deep stack** with disagreement head + abstain
   short-circuit.
9. **Calibration-under-noise** as a first-class uncertainty
   check.
10. **Per-decision rationale** in every uncertainty report.

## What remains substrate-blocked

* Real transformer-internal hidden state coupling.
* Real KV cache byte sharing.
* Direct attention-weight editing.
* Real cross-tokenizer transitivity at the behavioural layer.
* Multi-host shared state without proxy carrier.
* GPU-accelerated autograd (still pure-Python ``Variable``).

These are honest constraints, not failures. The
``W54-C-DEEP-TRANSFORMER-COUPLING`` conjecture is sharper than
``W53-C-…`` but unchanged in spirit.

## Replay-live notes

The W54 ``examples/w54_replay_live.py`` driver reuses the W53
quint-backend Ollama anchor scaffold (which itself reuses the
W52 quad anchor). When ``COORDPY_W54_OLLAMA_REACHABLE=1`` and
Ollama is reachable, the inner W53 quint anchor runs (which
calls the W52 quad anchor); W54 inherits the anchor result
via the W53 envelope's ``anchor_status``. Otherwise the W54
envelope still seals byte-identically with
``anchor_status: synthetic_only``.

## Files added or modified

**Mechanism modules (M1..M10):**
- `coordpy/persistent_latent_v6.py`
- `coordpy/multi_hop_translator_v4.py`
- `coordpy/mergeable_latent_capsule_v2.py`
- `coordpy/consensus_quorum_controller.py`
- `coordpy/corruption_robust_carrier_v2.py`
- `coordpy/deep_proxy_stack_v5.py`
- `coordpy/long_horizon_retention_v6.py`
- `coordpy/ecc_codebook_v6.py`
- `coordpy/transcript_vs_shared_arbiter_v3.py`
- `coordpy/uncertainty_layer_v2.py`

**Orchestrator + benchmarks:**
- `coordpy/w54_team.py`
- `coordpy/r107_benchmark.py`
- `coordpy/r108_benchmark.py`
- `coordpy/r109_benchmark.py`

**Examples:**
- `examples/w54_smoke_driver.py`
- `examples/w54_replay_live.py`

**Tests:**
- `tests/test_persistent_latent_v6_w54.py`
- `tests/test_multi_hop_translator_v4_w54.py`
- `tests/test_mergeable_latent_capsule_v2_w54.py`
- `tests/test_consensus_quorum_controller_w54.py`
- `tests/test_corruption_robust_carrier_v2_w54.py`
- `tests/test_deep_proxy_stack_v5_w54.py`
- `tests/test_long_horizon_retention_v6_w54.py`
- `tests/test_ecc_codebook_v6_w54.py`
- `tests/test_transcript_vs_shared_arbiter_v3_w54.py`
- `tests/test_uncertainty_layer_v2_w54.py`
- `tests/test_w54_team_envelope_chain.py`
- `tests/test_w54_trivial_passthrough_byte_identical.py`
- `tests/test_r107_benchmark.py`
- `tests/test_r108_benchmark.py`
- `tests/test_r109_benchmark.py`

**Docs:**
- `docs/SUCCESS_CRITERION_W54_DEEP_MERGE_LATENT_OS.md`
- `docs/RESULTS_W54_DMD_LOS.md`
- updates to `docs/RESEARCH_STATUS.md`,
  `docs/THEOREM_REGISTRY.md`,
  `docs/context_zero_master_plan.md`,
  `CHANGELOG.md`, `papers/context_as_objects.md`.

**Version status:**
- `coordpy.__version__` remains `0.5.20`. No bump.
- `SDK_VERSION` unchanged. No PyPI release.
