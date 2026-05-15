# Success criterion — W62 Trainable Replay-Dominance Hidden-vs-KV Substrate-Coupled Latent OS

> Pre-committed, falsifiable bar for the post-W61 W62 milestone
> (the seventh substrate-attack milestone, with the first
> programme-wide **trained per-head replay-dominance head**,
> **trainable hidden-vs-KV regime classifier**, **per-(layer,
> head, slot) cache-write ledger axis**, **prefix-state-reuse
> drift-curve fitted predictor**, and **two-stage attention-
> steering V6** with closed-form clamp). 2026-05-15.
>
> If any H-bar fails on the released-bench at 3 seeds, that bar is
> **falsified** and the corresponding ``W62-L-…`` cap is recorded
> in `docs/THEOREM_REGISTRY.md`.

## TL;DR

W62 is the **seventh substrate-attack milestone** in the Context
Zero programme. W56 cracked the in-repo substrate open. W57
deepened the breach. W58 made cache reuse load-bearing. W59
made closed-form ridge fits substrate-load-bearing in three
places. W60 made substrate-side trainable control multi-layer,
multi-head, multi-direction, replay-conditional, AND corruption-
aware. W61 added the first **content-addressable cache-key
axis**, **bilinear retrieval head**, **trained replay-threshold
head**, **4-D attention-budget tensor**, **multi-target stacked
HSB fit**, **attention-pattern-target KV fit**, and a **six-way
bidirectional substrate hybrid loop**. W62 pushes the next wall:

* Make replay control **decisively dominate** transcript replay
  and proxy-only baselines on a *named regime* (R-131 H163).
* Make **hidden-state injection** measurably beat KV-only
  injection on a *named regime* (R-131 H165), using a fitted
  hidden-vs-KV regime classifier (controller V5 + replay
  controller V3 + retrieval head V2).
* Make cache **retrieval, eviction, retention, AND repair**
  load-bearing through the **two-objective ridge fit** in cache
  controller V5 (drop oracle + retrieval relevance) and the
  **trained corruption-repair head** that fits a per-slot
  repair correction rather than only a floor.
* Make **prefix-state chain reuse** decisively faster than full
  recompute through the **fitted three-feature drift-curve
  predictor** (prefix bridge V6).
* Make **long-horizon retention** survive **2048+ turns** with
  **bounded reconstruction L2** under the V14 retention head
  family.
* Make **mergeable capsule V10** preserve per-(layer, head, slot)
  trust at merge time and surface a **disagreement Wasserstein-1
  distance** that the consensus controller V8 reads.
* Make **corruption-robust carrier V10** survive **2× the W61
  hostile envelope** with a **1024-bucket** wrap-around fingerprint
  and a **17-bit** adversarial burst family.
* Make **deep substrate hybrid V7** a **seven-way** loop with
  attention-pattern correction folded into the closed-form ridge
  fit.

## H-bars

12 mechanism advances, 3 benchmark families, ≥ 70 H-bars across
H163..H200+ at 3 seeds. The numbers below are pre-committed
floors; the released-bench result is the comparison.

### R-131 — Real-substrate / latent-bridge / replay-dominance / hidden-vs-KV

* **H163  replay_v3_dominance_vs_transcript** — Replay controller
  V3 with the fitted per-regime head outperforms transcript-only
  fallback on the **synthetic-corruption** regime: chosen-drift
  ≤ transcript-fallback-drift by ≥ 0.05 L2 averaged over 16
  candidates, AND chosen-flop ≤ transcript-fallback-flop. Falsifier:
  if the trained head can't beat transcript on the synthetic-
  corruption regime, the bar fails.
* **H163b replay_v3_dominance_vs_recompute** — Replay V3 chooses
  REUSE strictly more often than the V2 baseline on the
  **CRC-passed-low-drift** regime (R3.dominance_count ≥ V2.count).
* **H163c replay_v3_per_regime_head_fits** — Per-regime ridge fit
  converges (post-fit residual ≤ pre-fit residual + 1e-9) across
  4 regimes {synthetic-corruption, CRC-passed-low-drift,
  hidden-write-heavy, transcript-only}.
* **H164  cache_v5_two_objective_ridge** — Cache controller V5
  closed-form ridge fit with stacked drop-oracle and retrieval-
  relevance targets converges on both columns (worst-residual
  reduction ≥ 1e-6).
* **H164b cache_v5_trained_repair_head** — Trained corruption-
  repair head reduces L2 of corrupted-slot recovery vs the V4
  trained-floor (mean-residual reduction ≥ 1.5×).
* **H164c cache_v5_composite_five_head_mixture_v5** — composite_v5
  ridge mixture over 6 heads (V4's 5 + repair head) converges
  (post ≤ pre + 1e-9).
* **H165  hidden_vs_kv_v6_regime_classifier** — Trained hidden-
  vs-KV regime classifier (closed-form ridge over a 5-dim
  regime feature against a 3-class label) achieves training
  accuracy ≥ 0.8 on synthetic supervision.
* **H165b hidden_v6_three_layer_target_fit** — HSB V6 three-
  target stacked ridge fit converges (post ≤ pre + 1e-9) on
  all three target columns simultaneously.
* **H165c hidden_v6_into_v7_substrate** — HSB V6 writes
  propagate into the V7 substrate's per-(layer, head, slot)
  cache-write ledger with positive cumulative L2.
* **H166  prefix_v6_drift_curve_predictor** — Prefix bridge V6
  fitted three-feature drift-curve predictor (closed-form ridge
  over [reuse_len, recompute_len, drop_len] against a stacked
  drift-curve target) converges (post ≤ pre + 1e-9).
* **H166b prefix_v6_flop_saving_vs_recompute** — V6 prefix chain
  reuse on the standard {reuse, recompute, drop} split saves
  ≥ 25% flops vs full recompute.
* **H167  attention_v6_two_stage_clamp** — Attention-steering V6
  two-stage clamp (coarse L1-mass clamp + fine per-(L,H,Q,K)
  KL clamp) keeps max-KL ≤ budget + 1e-3 across all (L,H,Q,K)
  while preserving the coarse L1-mass shift.
* **H167b attention_v6_signed_falsifier_v2** — Signed-coefficient
  falsifier with independent ±1 per (L,H,Q,K) AND per-coarse-
  bucket produces a non-zero (post-pre) attention shift signed
  correlation with the per-bucket coefficient mean.

### R-132 — Long-horizon retention / reconstruction / aggressive-compression

* **H168  persistent_v14_chain_walk_depth_2048** —
  ``max_chain_walk_depth ≥ 2048`` honoured by the V14 chain.
* **H168b persistent_v14_decuple_skip** — Persistent V14 carries
  10 skip carriers (V13's 9 + a new replay-dominance-EMA carrier).
* **H168c persistent_v14_distractor_rank_8** — Distractor basis
  rank ≥ 8 (V13 was 6, V12 was 4).
* **H169  lhr_v14_thirteen_way_head** — 13-head reconstruction
  (V13's 12 + replay-dominance-conditioned head) runs without
  crashing on standard inputs.
* **H169b lhr_v14_replay_dominance_head_nontrivial** — Replay-
  dominance-conditioned head produces a non-trivial output (L2
  diff > 1e-6 between zero and non-zero indicator).
* **H169c lhr_v14_four_layer_scorer** — Closed-form ridge fit on
  the post-tanh-2 feature (random + ReLU → random + tanh →
  random + tanh-2 → ridge) converges (post ≤ pre + 1e-9).
* **H170  ecc_v14_bits_per_token** — ≥ 25.0 bits/visible-token
  at full emit. Falsifier: if K1..K13 product < 2^23.
* **H170b ecc_v14_total_codes** — Total codebook size = 2^23 =
  8 388 608.
* **H170c ecc_v14_rate_floor_falsifier** — 4096-bit/token target
  reproduces the structural ceiling (above log2(2^23) = 23).
* **H171  multi_hop_v12_chain_length_17** — 7-axis composite
  trust at chain-length 17 over 20 backends and 380 directed
  edges.
* **H171b multi_hop_v12_seven_axis** — 7-axis composite
  (substrate × hidden × attention × retrieval × replay ×
  attention_pattern × replay_dominance) used.
* **H171c multi_hop_v12_compromise_threshold** — Compromise
  threshold in [1, 7] (adversary needs to drive at least 1
  axis to zero on dominant path).

### R-133 — Corruption / disagreement / consensus / fallback / abstention

* **H172  crc_v10_kv1024_detect** — 1024-bucket fingerprint detect
  rate for single-byte flips ≥ 0.95.
* **H172b crc_v10_17bit_burst** — 17-bit adversarial burst detect
  rate ≥ 0.95.
* **H172c crc_v10_post_repair_jaccard** — Post-repair top-K
  Jaccard floor ≥ 0.5.
* **H173  consensus_v8_twelve_stages** — Consensus chain has
  exactly 12 disjoint stages (V7's 11 + ``trained_repair``).
* **H173b consensus_v8_repair_stage_fires** — ``trained_repair``
  stage fires when CRC V10 detects but the V5 repair head
  succeeds.
* **H174  uncertainty_v10_nine_axis** — 9-axis weighted
  composite returns a value in [0, 1] (V9's 8 + replay-
  dominance fidelity).
* **H174b uncertainty_v10_replay_dominance_aware** —
  ``replay_dominance_aware`` flips True when fidelity < 1.0.
* **H175  disagreement_algebra_v8_wasserstein_identity** —
  Wasserstein-1-equivalence identity holds iff argmax preserved
  AND Wasserstein-1 ≤ floor.
* **H175b disagreement_algebra_v8_wasserstein_falsifier** —
  Falsifier triggers when Wasserstein-1 > floor.
* **H176  tvs_v11_twelve_arms_sum_to_one** — Pick-rates over 12
  arms sum to 1.0 within 1e-9.
* **H176b tvs_v11_replay_dominance_arm_fires** — Replay-
  dominance arm fires when replay_dominance_fidelity is strict
  highest score.
* **H176c tvs_v11_reduces_to_v10** — When
  replay_dominance_fidelity = 0, V11 reduces to V10 (no replay-
  dominance arm).
* **H177  mlsc_v10_replay_dominance_chain** — Replay-dominance
  witness chain inherits as union of parent chains plus merge
  addition.
* **H177b mlsc_v10_disagreement_wasserstein_distance** —
  Disagreement Wasserstein-1 distance per merge is computed and
  surfaced.

### Substrate axes / coupling (cross-family)

* **H178  substrate_v7_per_layer_head_slot_ledger** —
  Per-(layer, head, slot) cache-write ledger has shape (L, H, T)
  and survives evictions.
* **H178b substrate_v7_logit_lens_probe** — Per-layer logit-
  lens probe (linear projection from layer hidden state to
  vocab) returns shape (L, V).
* **H178c substrate_v7_attention_receive_delta** — Per-(layer,
  head, position) attention-receive *delta* (forward-to-forward
  difference) is recorded.
* **H178d substrate_v7_replay_trust_ledger** — Per-(layer,
  head) replay-trust ledger (EMA of replay decisions) recorded.
* **H179  deep_hybrid_v7_seven_way** — Seven-way bidirectional
  hybrid bridge sets ``seven_way=True`` when all seven axes
  fire.
* **H180  substrate_v7_adapter_tier** — ``substrate_v7_full``
  tier satisfied only by the V7 in-repo runtime (not by wrapped
  V6 / V5 or synthetic).

## Strong / partial / failure thresholds

* **Strong success**: ≥ 95% H-bars pass at 3 seeds AND every
  per-mechanism advance is exercised end-to-end (≥ 0 mechanisms
  inactive on the released bench).
* **Partial success**: ≥ 80% H-bars pass at 3 seeds.
* **Failure**: < 80% H-bars pass at 3 seeds OR ≥ 1 falsifier
  triggers in a way that materially invalidates a mechanism.

## Substrate triage (pre-committed)

Truly accessible now on real local backends:
* logits / logprobs at the HTTP edge.

Already real in the in-repo substrate (W61 + carried forward):
* per-(layer, head, position) cumulative attention-receive,
  per-(layer, head) Jacobian table, multi-segment partial-prefix
  reuse, corruption flag channel, content-addressable cache-key
  axis, hidden-write trace, replay-age channel, forward counter,
  cross-layer coupling.

New in W62 (in-repo V7 substrate; honest):
* per-(layer, head, slot) cache-write ledger (axis tracking which
  slots were written, by which bridge, at which forward),
* per-layer logit-lens probe,
* per-(layer, head, position) attention-receive *delta*,
* per-(layer, head) replay-trust ledger (EMA of REUSE / RECOMPUTE
  / FALLBACK / ABSTAIN decisions),
* per-(layer, head) **head-specific hidden-vs-KV regime label**
  produced by the trained classifier.

Still substrate-blocked (carries forward; ruthlessly honest):
* third-party hosted-model substrate (Ollama, OpenAI-compatible)
  remains text-only at the HTTP surface. W62 does NOT change
  this. ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` is the
  carried-forward cap.

## Honest scope statements that carry forward and that W62 adds

* ``W62-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``: hosted backends
  remain text-only at the HTTP surface. W62 makes no claim of
  third-party transformer-internal access.
* ``W62-L-V7-NO-AUTOGRAD-CAP``: W62 adds **five new closed-form
  ridge fits** on top of W61's seven: (a) cache controller V5
  stacked drop-oracle + retrieval-relevance two-objective fit;
  (b) cache controller V5 trained-repair head; (c) replay
  controller V3 per-regime head; (d) hidden-vs-KV V6 regime
  classifier; (e) prefix bridge V6 drift-curve predictor. Total
  **twelve closed-form ridge solves**. No SGD, no autograd, no
  GPU.
* ``W62-L-NUMPY-CPU-V7-SUBSTRATE-CAP``: the V7 substrate is 9
  layers / d_model=64 / byte-vocab / max_len=128 / untrained
  NumPy on CPU. NOT a frontier model.
* ``W62-L-V14-OUTER-NOT-TRAINED-CAP``: the V14 outer wrapper adds
  one carrier (replay-dominance EMA); does NOT train the V13 outer
  GRU end-to-end.
* ``W62-L-PERSISTENT-V14-DECUPLE-SKIP-PROXY-CAP``: the new replay-
  dominance carrier is propagated by EMA, not by a learned gate.
* ``W62-L-ECC-V14-RATE-FLOOR-CAP``: structural rate ceiling
  log2(2^23) = 23 raw data bits per segment-tuple.
* ``W62-L-V6-CACHE-CONTROLLER-NO-AUTOGRAD-CAP``: cache controller
  V5 fits three additional heads (two-objective ridge,
  trained-repair, composite_v5 mixture) by closed-form ridge.
* ``W62-L-V3-REPLAY-NO-AUTOGRAD-CAP``: replay controller V3
  fits a 5×4 per-regime head and a 5-dim regime gate by closed-
  form ridge.
* ``W62-L-V6-HSB-NO-AUTOGRAD-CAP``: HSB V6 three-target stack
  fit delegates the per-target solve to V5 with target stacking.
* ``W62-L-V6-PREFIX-DRIFT-CURVE-LINEAR-CAP``: drift-curve
  predictor is a linear ridge on a 3-d configuration feature; it
  does NOT model token-content-conditional drift.
* ``W62-L-V6-ATTN-NO-AUTOGRAD-CAP``: V6 attention-steering is a
  two-stage clamp; no autograd.
* ``W62-L-MULTI-HOP-V12-SYNTHETIC-BACKENDS-CAP``: V12 backends are
  named, not executed.
* ``W62-L-CRC-V10-FINGERPRINT-SYNTHETIC-CAP``: 1024-bucket
  fingerprint is computed over the in-repo substrate cache, not
  third-party hosted cache state.
* ``W62-L-CONSENSUS-V8-REPAIR-STAGE-SYNTHETIC-CAP``: trained-
  repair stage runs the V5 repair head on the in-repo cache; it
  is not a real third-party model recovery path.

## Boundary preservation

* `coordpy.__version__` remains `0.5.20`.
* `coordpy.SDK_VERSION` remains `coordpy.sdk.v3.43`.
* No PyPI release.
* Smoke driver passes.
* W62 ships at explicit-import paths only.
* W62 envelope chain: ``w61_outer_cid`` carries the W61 envelope
  CID byte-for-byte; verifier ≥ 65 disjoint failure modes.
