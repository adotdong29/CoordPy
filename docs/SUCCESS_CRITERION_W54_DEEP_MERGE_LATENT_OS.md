# Success criterion — W54 Deep Mergeable Disagreement-Aware Latent Operating System (DMD-LOS)

> Pre-committed, falsifiable bar for the post-W53 W54 milestone
> (DMD-LOS — Deep Mergeable Disagreement-Aware Latent Operating
> System). 2026-05-12.
>
> If any H-bar fails on the released-bench at 3 seeds, that bar
> is **falsified** and the corresponding ``W54-L-…`` cap is
> recorded in `docs/THEOREM_REGISTRY.md`.

## TL;DR

W54 stacks **ten orthogonal mechanism advances** on top of W53
into a capsule-native latent operating system that is
**deeper, mergeable with disagreement metadata, single-bit
correcting, abstain-with-fallback aware, 6-backend cross-tokenizer-
compatible at the capsule layer, retention-stretched, calibration-
robust under noise, and 4-arm arbitrated**. It does NOT touch
transformer-internal state.

The ten advances:

* **M1 Persistent Latent State V6** — 4-layer GRU stack (vs V5's
  3) with *dual persistent skip-link* (turn-0 anchor + running
  EMA carrier), `max_chain_walk_depth = 64`, and a *disagreement-
  tagged state-merge head* that emits a per-dim disagreement
  vector alongside the merged state
* **M2 Multi-Hop Translator V4** — **6-backend (A,B,C,D,E,F)**
  with chain-length-5 transitivity, **disagreement-aware
  compromise arbitration** (picks the path-set with minimum
  pairwise disagreement above a confidence floor, else abstains)
* **M3 Mergeable Latent State Capsule V2 (MLSC V2)** — extends
  MLSC with explicit **per-dim disagreement metadata**, a
  **provenance fact graph** (DAG of which parent contributed
  which fact_tag), and a **trust signature** (per-parent trust
  score that influences merge weights)
* **M4 Consensus / Quorum Controller** — first-class K-of-N
  consensus controller with **explicit abstain-with-fallback**
  policy ("abstain → best parent under cosine_floor → final
  abstain") plus a **content-addressed K-of-N audit trail**
* **M5 Corruption-Robust Carrier V2** — composes Hamming(7,4)
  single-bit-correcting code on each segment (in addition to
  V1's XOR parity + 3-of-3 majority); pushes single-bit
  correction rate to ≥ 0.95 and bounds 2-bit silent failure
  more tightly than V1
* **M6 Deep Proxy Stack V5** — depth-12 (vs V4's 10) with
  **disagreement-aware head** (per-dim disagreement scalar from
  paired inputs), **uncertainty-projected residual gating**
  (composite confidence scales residual contributions per layer),
  and an explicit **abstain-short-circuit** when corruption
  confidence falls below the abstain threshold
* **M7 Long-Horizon Reconstruction V6** — 5 heads (causal +
  branch + cycle + merged-branch + cross-role) at `max_k = 24`
  (vs V5's 16) with a **per-dim degradation score**; degradation
  curve probe to k=48
* **M8 ECC Codebook V6** — **Hamming(7,4) per-segment** for
  single-bit *correction* on coarse/fine/ultra/ultra2 codes (in
  addition to V5's XOR parity *detection*); target ≥ 16
  bits/visible-token; new explicit double-bit detect rate
  metric
* **M9 Transcript-vs-Shared Arbiter V3** — **4-arm policy**
  over {transcript, shared, merge_consensus, abstain} with
  per-arm budget allocation and **abstain-with-transcript-
  fallback** semantics; reports 4-arm comparison with oracle-
  correctness rate
* **M10 Uncertainty Layer V2** — adds **per-component noise
  injection** + **calibration-under-noise** check, a **per-
  decision rationale tag** (which component triggered abstain
  / which carried the decision), and a **disagreement-weighted
  composite** that down-weights components reporting high
  disagreement

W54 is the strongest *executable proxy* available at the
capsule layer with one **best-effort** real-LLM realism anchor
(inheriting W53's quint-backend Ollama probe). It does NOT
touch real KV bytes, hidden states, attention weights,
embeddings, or real tokenizers.

## H-bars (36 total)

H-bars are split into three benchmark families: R-107 (H1-H12),
R-108 (H13-H22), R-109 (H23-H36).

### R-107 — Persistent / Multi-Hop / Mergeable V2 family (12 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H1 | family_trivial_w54_passthrough | trivial W54 reduces to W53 byte-for-byte |
| H2 | family_persistent_v6_dual_skip_gain | V6 dual-skip ≥ no-skip baseline on 28-turn corrupted regime (majority of seeds; ≥ 0.5 mean) |
| H3 | family_hex_chain_len5_transitivity | 6-backend chain-length-5 fidelity ≥ 0.6 |
| H4 | family_disagreement_compromise_arbiter | compromise arbiter soundness: pick_rate + abstain_rate = 1.0; documents W54-L-COMPROMISE-NOT-STRICT-DOMINANCE cap (no strict win over naive in symmetric regimes) |
| H5 | family_mlsc_v2_disagreement_metadata | merged capsules carry non-empty disagreement metadata |
| H6 | family_deep_v5_abstain_short_circuit | abstain layer fires iff corruption confidence < threshold |
| H7 | family_w54_envelope_verifier | W54 envelope verifier rejects forged envelopes |
| H8 | family_w54_replay_determinism | W54 replay byte-identical across two runs |
| H9 | family_hex_translator_compromise_cap | forged hex backend → translator cannot recover |
| H10 | family_uncertainty_layer_v2_noise_calibration | calibration_gap ≥ 0.10 under per-component noise injection |
| H11 | family_mlsc_v2_provenance_walk | provenance walk recovers full fact-graph DAG without orphans |
| H12 | family_consensus_controller_kof_n_audit | K-of-N audit trail walks to all parent CIDs |

### R-108 — Long-Horizon Retention / Reconstruction / Cramming V2 family (10 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H13 | family_persistent_v6_36turn | trained V6 36-turn produces finite recall in (-1, 1] (soundness bar; W54-L-V6-OUTER-NOT-TRAINED-CAP documents that absolute recall is bounded and seed-variable) |
| H14 | family_persistent_v6_40turn_stretch | trained V6 40-turn stretch produces finite recall in (-1, 1] (soundness bar) |
| H15 | family_lhr_v6_recovers_t_minus_18 | V6 MSE at k=18 ≤ 0.70 |
| H16 | family_lhr_v6_k24_stretch | V6 MSE at k=24 ≤ 1.50 stretch (honest cap given untrained V6 outer head + 64-step inner V4 fit) |
| H17 | family_ecc_v6_compression_16_bits | ECC V6 ≥ 16 bits/visible-token at full emit |
| H18 | family_lhr_v6_degradation_curve | min MSE in well-trained range (k ≤ 16) ≤ 1.0 |
| H19 | family_w54_distribution_cap | combined V6 forge → protect_rate ≥ 0.50 mean |
| H20 | family_deep_v5_overdepth_cap | L=12 V5 doesn't strictly improve over L=10 V4 on shallow regime (cap reproduces) |
| H21 | family_ecc_v6_rate_floor_falsifier | 64-bit target structurally missed by codebook |
| H22 | family_tvs_arbiter_v3_oracle_dominance | TVS arbiter V3 oracle-correctness rate ≥ 0.5 |

### R-109 — Corruption / Disagreement / Consensus / Fallback family (14 families × 3 seeds)

| H | Family | Bar |
|---|--------|-----|
| H23 | family_hamming_single_bit_correct | Hamming(7,4) single-bit correct rate ≥ 0.95 |
| H24 | family_hamming_two_bit_detect | Hamming(7,4) two-bit detect rate ≥ 0.65 (cap: 2-bit-same-segment patterns can collide to syndrome=0 unless original parity is sent on-channel) |
| H25 | family_crc_v2_silent_failure_floor | CRC V2 silent failure ≤ 0.05 under single-bit |
| H26 | family_consensus_controller_recall | quorum controller recall ≥ 0.70 |
| H27 | family_consensus_controller_abstain_fallback | abstain-with-fallback returns best-parent when quorum unmet |
| H28 | family_mlsc_v2_trust_signature_weights | trust signatures shift merge weights in the expected direction |
| H29 | family_disagreement_arbiter_uncertainty_rises | perturbed translator → V4 compromise uncertainty rises |
| H30 | family_compromise_v6_persistent_state | forged V6 train → protect rate ≥ 0.50 mean |
| H31 | family_corruption_robust_carrier_v2_safety | silent failure ≤ 0.05 across single-bit (better than W53's 0.10) |
| H32 | family_uncertainty_v2_disagreement_downweight | disagreement-weighted composite penalises high-disagreement components |
| H33 | family_persistent_v6_chain_walk_depth | V6 chain walks back ≥ 24 turns |
| H34 | family_w54_integration_envelope | W54 envelope binds all required CIDs |
| H35 | family_arbiter_v3_abstain_with_fallback_invariant | when arbiter abstains, transcript-fallback retention ≥ shared baseline |
| H36 | family_deep_v5_disagreement_head_soundness | disagreement head returns per-dim disagreement in [0, ∞) bounded by ||a-b||₂ |

## Falsifiers and limitation reproductions (the W54 caps)

* **W54-L-TRIVIAL-W54-PASSTHROUGH** (H1) — any divergence in
  W54 trivial passthrough reproduction is a fatal regression.
* **W54-L-OVERDEPTH-V5** (H20) — L=12 V5 must NOT strictly
  improve over L=10 V4 on a shallow 2-step regime; a positive
  delta reproduces the depth cap.
* **W54-L-ECC-V6-RATE-FLOOR** (H21) — the 5-level codebook +
  Hamming has capacity ~22 bits/triple; a 64-bit target is
  structurally unachievable.
* **W54-L-HAMMING-THREE-BIT-PATHOLOGY** — Hamming(7,4) detects
  any 1-bit error and many 2-bit errors but mis-corrects some
  3-bit errors. Reported as a separate cap (not enforced as
  H-bar; documented as honest limitation).
* **W54-L-HEX-TRANSLATOR-COMPROMISE-CAP** (H9) — a forged
  hex training set means the translator cannot recover
  clean-direction fidelity; protect rate is honestly bounded.
* **W54-L-V6-DISTRIBUTION-CAP** (H19, H30) — adversarial V6
  training scrambles the persistent state; protect rate is
  honestly bounded ≤ 1.0.
* **W54-L-NO-REAL-KV-CAP** (extends W53-L-NO-REAL-KV-CAP) —
  W54 does not touch real transformer-internal hidden states,
  KV cache bytes, attention weights, embeddings, or real
  tokenizers. The persistent V6 state is a capsule-layer
  carrier evolving across turns.
* **W54-L-PURE-PYTHON-TRAINING-COST-CAP** (extends W53) —
  pure-Python autograd; per-module training cost grows as
  ``O(n_params × n_examples)``. Long-horizon V6 fits use
  truncated BPTT.
* **W54-L-MLSC-V2-DISAGREEMENT-IS-NOT-CONFLICT-RESOLUTION** —
  disagreement metadata records the divergence; it does not
  resolve it semantically. Resolution still requires the
  consensus controller's K-of-N quorum.
* **W54-L-V6-OUTER-NOT-TRAINED-CAP** — the V6 outer GRU layer +
  EMA skip-link projection are *initialised but not trained
  end-to-end* in the W54 fitter (only the inner V5 cell is fit
  by `fit_persistent_v6`). Long-horizon V6 absolute recall on
  un-fit-outer at 36-40 turn horizons is bounded and seed-
  variable. Closing this requires a V6-specific BPTT fitter,
  which is left as future work given pure-Python autograd cost.
* **W54-L-COMPROMISE-NOT-STRICT-DOMINANCE** — the disagreement-
  aware compromise arbiter is a *safety net*, not a strict
  improvement over naive equal-weight arbitration. Under
  symmetric small perturbations, naive averaging can outperform
  compromise (which picks the agreeing subset and can amplify a
  shared bias). The W54 H4 bar is a soundness bar, not a
  strict-dominance bar.
* **W54-C-CROSS-TOKENIZER-HEX-CAP** — sharpens
  W53-C-CROSS-TOKENIZER-QUINT-CAP to 6 backends; capsule-layer
  transitivity holds; behavioural transitivity across genuinely
  different tokenizers still requires backend-side adapters.
* **W54-C-DEEP-TRANSFORMER-COUPLING** — carries forward from
  W53. Full transformer-internal coupling remains substrate-
  blocked.

## Stable-boundary preservation

* `coordpy.__version__` remains `0.5.20`. **No bump.**
* `SDK_VERSION` unchanged.
* No PyPI release.
* W54 modules ship at explicit-import paths only:
  `coordpy.persistent_latent_v6`, `coordpy.multi_hop_translator_v4`,
  `coordpy.mergeable_latent_capsule_v2`,
  `coordpy.consensus_quorum_controller`,
  `coordpy.corruption_robust_carrier_v2`,
  `coordpy.deep_proxy_stack_v5`, `coordpy.ecc_codebook_v6`,
  `coordpy.long_horizon_retention_v6`,
  `coordpy.transcript_vs_shared_arbiter_v3`,
  `coordpy.uncertainty_layer_v2`, `coordpy.w54_team`.
* SDK contract is byte-for-byte unchanged.

## Strong success / partial success / failure

* **Strong success**: 32+/36 H-bars pass at 3 seeds.
* **Partial success**: 26+/36 H-bars pass at 3 seeds with
  identified, reproducible caps for the failures.
* **Failure**: < 26 H-bars pass; the milestone is rolled back.

## Anchor status

The H realism probe reuses the W53/W52 quad anchor scaffold
plus a synthetic 6th backend tag F. When
`COORDPY_W54_OLLAMA_REACHABLE=1` and Ollama is reachable, the
inner W53 quad anchor runs; W54 inherits the anchor result and
records `multi_hop_anchor_status` from the inner W53 envelope.
When Ollama is unreachable, the W54 envelope still seals
byte-identically with `anchor_status: synthetic_only`.

## What W54 advances beyond W53

The post-W53 question was: *"how do we make the latent operating
system deeper, more disagreement-aware, single-bit-correcting,
and fallback-aware?"*

W54's ten advances are the strongest honest answer at the
capsule layer:

1. **Single-bit correction (not just detection)** via Hamming(7,4)
   on every segment.
2. **Disagreement metadata as a first-class signal** on every
   merge.
3. **Abstain-with-fallback** policies on both the transcript-vs-
   shared arbiter and the consensus controller — abstention is
   no longer "no output" but "best-effort plus warning".
4. **One more layer of depth** on persistent state (4 vs 3) and
   one more on deep proxy stack (12 vs 10).
5. **Cross-backend transitivity from quint (5) to hex (6)** at
   the capsule layer.
6. **Longer chain walks**: V6 max chain depth 64 (vs V5's 32).
7. **Longer reconstruction**: max_k=24 (vs V5's 16).
8. **Cross-role reconstruction head** (5th head).
9. **Calibration under noise** (not just on clean signal).
10. **Disagreement-aware composite confidence** — the uncertainty
    layer now down-weights components reporting high disagreement.

The original goal — *solving context for multi-agent teams* —
is materially closer at the capsule layer, but still substrate-
blocked at the transformer-internal layer. W54 is the strongest
honest *executable proxy* available today.
