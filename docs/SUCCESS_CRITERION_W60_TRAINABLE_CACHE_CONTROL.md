# Success criterion — W60 Trainable Cache-Control Substrate-Coupled Latent OS

> Pre-committed, falsifiable bar for the post-W59 W60 milestone
> (the fifth substrate-attack milestone, with first-class
> ReplayController and per-(layer, head) trained heads). 2026-05-14.
>
> If any H-bar fails on the released-bench at 3 seeds, that bar is
> **falsified** and the corresponding ``W60-L-…`` cap is recorded
> in `docs/THEOREM_REGISTRY.md`.

## TL;DR

W60 is the **fifth substrate-attack milestone** in the Context
Zero programme. W56 cracked the in-repo substrate open. W57
deepened the breach. W58 made cache reuse load-bearing. W59 made
closed-form ridge fits substrate-load-bearing in three places.
W60 makes substrate-side trainable control **multi-layer, multi-
head, multi-direction, replay-conditional, AND corruption-aware**.

The previous milestones (W43..W59) built an increasingly strong
capsule + tiny-substrate-coupling stack. W60 retains all of that
and **upgrades the in-repo substrate runtime** from V4 to V5
(seven layers, per-(layer, head, position) cumulative attention-
receive matrix, per-(layer, head) linearised logit Jacobian
table, per-(layer, position) corruption flag channel, multi-
segment partial reuse). The W60 controllers fit *several* heads
by closed-form ridge across multiple substrate-side features
simultaneously, and the new **W60 ReplayController** is the
programme's first first-class state-reuse-vs-recompute-vs-
fallback-vs-abstain policy.

Eighteen orthogonal mechanism advances on top of W59:

* **M1 Tiny Transformer Runtime V5** — ``coordpy.tiny_substrate_v5``:
  7 layers (vs V4's 6), GQA (8q/4kv), RMSNorm + SwiGLU,
  per-(layer, head, position) cumulative EMA *attention-receive*
  matrix that survives evictions, per-(layer, head) linearised
  *logit Jacobian table* under the linearised-head approximation,
  per-(layer, position) **corruption flag channel** (set by CRC
  V8, read by cache controller V3), multi-segment partial-prefix
  reuse (segments=[(start, end, kind), ...] where kind ∈
  {reuse, recompute, drop}) with per-segment flop split, and a
  V5 write log tag every cache write.
* **M2 KV Bridge V5** — ``coordpy.kv_bridge_v5``: multi-direction
  closed-form ridge fit ``α ∈ R^d`` over ``d`` orthogonal
  correction directions; logit-direction fit (vs V4's scalar L2
  fit); two correction layers (``layer_a`` + ``layer_b``)
  composed at inject time; all-bank fingerprint (4 banks × 128
  buckets) for cross-bank corruption discrimination; **reverse-
  extract** carrier estimate from a cache slot via least-squares
  against the projection matrix.
* **M3 Hidden-State Bridge V4** — ``coordpy.hidden_state_bridge_v4``:
  **per-(layer, head) closed-form ridge fit** of the inject-scale
  tensor (``δ ∈ R^{L*H}``); recovery path that fits a counter-
  perturbation against an adversarial per-(layer, head) attack;
  **KV-vs-Hidden head-to-head harness** that runs the same
  carrier through both the hidden V4 fit and a KV V4 fit on the
  same target logit direction and reports per-arm L2 residual,
  cosine alignment, and ``argmax_preserved``. The W60 R-125
  H-bar tests the falsifiable
  ``hidden_beats_kv ∨ kv_beats_hidden ∨ tie`` claim.
* **M4 Prefix-State Bridge V4** — ``coordpy.prefix_state_bridge_v4``:
  multi-segment partial reuse on the V5 substrate (vs V3's
  single-split); chain forward over a list of follow-up steps
  with per-step drift L2 and cumulative drift envelope.
* **M5 Attention-Steering Bridge V4** —
  ``coordpy.attention_steering_bridge_v4``: per-(layer, head,
  query) **3-D budget tensor** (vs V3's 2-D); measurable
  attention-map L1 mass shift alongside KL; **negative-budget
  falsifier** (budget set to all-zero ⇒ post-KL must be < 1e-6
  and attention pattern must NOT shift).
* **M6 Cache Controller V3** — ``coordpy.cache_controller_v3``:
  **four** new policies on top of V2's three —
  ``learned_attention_receive`` (closed-form ridge over the
  per-(layer, head) cumulative attention-receive feature),
  ``learned_corruption_aware`` (V2 + hard ``-inf`` floor on
  flagged slots), ``trained_eviction`` (closed-form ridge over a
  ``[hidden, importance, attention_receive_l1, retrieval]``
  feature against the V1 leave-one-out drop oracle), and
  ``composite_v3`` (4-feature ridge mixture over the four V3
  heads). The composite mixture weights are themselves fit by
  closed-form ridge against the drop oracle.
* **M7 Replay Controller** — ``coordpy.replay_controller``:
  the programme's first first-class **state-reuse-vs-recompute-
  vs-fallback-vs-abstain** policy. Decision rule (in order):
  REUSE if CRC passed AND saving above floor AND drift below
  ceiling; RECOMPUTE if recompute under flop ceiling AND reuse
  drift over ceiling OR CRC failed; FALLBACK if transcript
  available; ABSTAIN otherwise. Emits an audit log + ``flop-vs-
  drift`` trade-off curve.
* **M8 Persistent Latent V12** — ``coordpy.persistent_latent_v12``:
  10 layers (vs V11's 9), **octuple persistent skip-link**
  (V11's septuple plus a *replay-controller decision EMA*),
  ``max_chain_walk_depth = 1024``, **distractor-resistant**
  replay projection (random orthonormal distractor basis fit
  by Gram-Schmidt at init; replay skip projected orthogonal to
  the basis before injection).
* **M9 Multi-Hop Translator V10** —
  ``coordpy.multi_hop_translator_v10``: 16 backends (A..P) over
  240 directed edges with chain-length-15, **five-axis trust
  composite** (substrate × hidden × attention × retrieval ×
  replay), and a **compromise-of-N detector** that estimates the
  minimum number of axes an adversary must drive to zero on the
  dominant path to flip the arbitration outcome.
* **M10 Mergeable Latent State Capsule V8 (MLSC V8)** —
  ``coordpy.mergeable_latent_capsule_v8``: ``replay_witness_chain``
  + ``substrate_witness_chain`` + ``provenance_trust_table``
  (per-backend trust scalar) all union-inherited from parents;
  two new algebra signatures (``replay_choice``,
  ``substrate_state_inject``).
* **M11 Consensus Fallback Controller V6** —
  ``coordpy.consensus_fallback_controller_v6``: **10-stage chain**
  {K-of-N → trust-weighted → substrate → logit_lens →
  cache_reuse → retrieval_replay → **replay_controller** →
  best_parent → transcript → abstain}. The new
  ``replay_controller`` stage picks the parent whose
  ReplayController decision aligns with the substrate's CRC +
  flop-saving signal.
* **M12 Corruption-Robust Carrier V8** —
  ``coordpy.corruption_robust_carrier_v8``: **256-bucket** Reed-
  Solomon-style fingerprint (vs V7's 128); ``recover_v8_kv_cache``
  operator that writes per-(layer, position) corruption flags
  back into the V5 cache; adversarial **11-bit burst** family;
  **post-replay top-K agreement** floor (must be ≥ pre-replay).
* **M13 Deep Substrate Hybrid V5** —
  ``coordpy.deep_substrate_hybrid_v5``: the programme's first
  **five-way bridge** — V6 ↔ substrate V5 ↔ cache controller
  V3 (composite) ↔ replay controller ↔ retrieval head, with
  ``five_way=True`` flag set when REUSE or RECOMPUTE fires.
* **M14 Long-Horizon Retention V12** —
  ``coordpy.long_horizon_retention_v12``: 11-head V12 (V11's 10
  + **replay-conditioned**) at ``max_k=96``; **two-layer**
  retention scorer (random projection + frozen ReLU + closed-
  form ridge over the post-ReLU features).
* **M15 ECC Codebook V12** — ``coordpy.ecc_codebook_v12``:
  K1..K11 = 2 097 152 codes (= 2^21); **23.333 bits/visible-
  token** at full emit (≥ 23.0 target, exceeds V11's 22.333).
* **M16 Uncertainty Layer V8** — ``coordpy.uncertainty_layer_v8``:
  **7th weighting axis** ``replay_fidelity`` on top of V7's
  (confidence, trust, substrate, hidden, cache, retrieval).
* **M17 Disagreement Algebra V6** —
  ``coordpy.disagreement_algebra_v6``: adds **replay-controller
  equivalence identity** with argmax + L2-budget check (V5
  retrieval identity + V4 cache-reuse identity all carry
  forward).
* **M18 TVS Arbiter V9 (Ten-Arm)** —
  ``coordpy.transcript_vs_shared_arbiter_v9``: tenth arm
  ``replay_controller_choice``, preferred when the replay-
  controller fidelity is the strict highest score.

W60 is the **fifth executable substrate-coupling milestone**;
it is NOT a claim of third-party transformer-internal access.
``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` documents that
Ollama / OpenAI-compatible / hosted backends remain text-only on
the HTTP surface (carries forward unchanged from W57/W58/W59).
``W60-L-V5-NO-AUTOGRAD-CAP`` documents the ridge-only training
boundary: W60 fits **only** (a) the KV bridge V5 multi-direction
α along ``n_directions`` random directions, (b) the KV bridge V5
logit-direction α, (c) the HSB V4 per-(layer, head) δ tensor,
(d) the cache controller V3 ``learned_attention_receive`` linear
head, (e) the cache controller V3 ``trained_eviction`` linear
head, (f) the cache controller V3 ``composite_v3`` mixture
weights, (g) the LHR V12 two-layer scorer (random first layer,
ridge second layer) — every one a single closed-form linear
ridge solve, no end-to-end backprop, no autograd, no GPU.
``W60-L-V12-OUTER-NOT-TRAINED-CAP`` carries forward V11's outer-
untrained cap.
``W60-L-ECC-V12-RATE-FLOOR-CAP`` documents the new structural
ceiling (log2(2 097 152) = 21 bits/segment); the 2048-bit/token
falsifier reproduces honestly as the new H136c bar.
``W60-L-LHR-V12-SCORER-FIT-CAP`` documents that the LHR V12
two-layer scorer's first layer is *random + frozen* (ridge only
on layer 2).
``W60-L-CORRUPTION-FLAG-CHANNEL-CAP`` documents that the V5
corruption flag channel is a *channel*, not a detector — flags
are written by external CRC V8 and read by cache controller V3.

## H-bars (45 total)

H-bars enumerate the empirical content of W60. Pre-committed;
if 3-seed mean falls below the bar, the corresponding limitation
theorem fires.

### R-125 — real-substrate / latent-bridge / cache-reuse / trained-controller / hidden-vs-KV (20 H-bars)

| H | Subject | Bar |
| --- | --- | --- |
| H125  | substrate V5 forward determinism | identical params + token_ids → byte-identical V5 trace CID |
| H125b | substrate V5 attention-receive propagates | shape (n_heads, n_tokens) and non-zero |
| H125c | substrate V5 logit Jacobian table shape | (n_layers, n_heads) per the V5 config |
| H125d | substrate V5 corruption flag channel | flags written by CRC V8, read by controller |
| H126  | KV bridge V5 multi-direction ridge fit | n_directions=3, post ≤ pre + 1e-9 |
| H126b | KV bridge V5 logit-direction fit | converges, fit_kind == "logit_direction" |
| H126c | KV bridge V5 reverse-extract | residual L2 < 1e-3 |
| H127  | HSB V4 per-(layer, head) ridge fit | ``L*H = 16`` decision variables, post ≤ pre + 1e-9 |
| H127b | HSB V4 recovery from adversarial | post ≤ pre + 1e-9 after counter-fit |
| H128  | attention V4 per-(layer, head, query) budget | every (l,h,q) ≤ budget + 1e-3 |
| H128b | attention V4 negative-budget falsifier | post-KL < 1e-6 AND no shift |
| H129  | prefix V4 multi-segment flop saving | flop_saved > 0 over full recompute |
| H129b | prefix V4 chain drift bounded | per-step drifts and cumulative drift recorded |
| H130  | cache controller V3 trained_eviction | post ≤ pre + 1e-9 (closed-form ridge) |
| H130b | cache controller V3 composite weights | 4-dim weight vector, post ≤ pre + 1e-9 |
| H131  | replay controller chooses REUSE | CRC passed + saving above floor → REUSE |
| H131b | replay controller chooses RECOMPUTE | CRC failed → RECOMPUTE |
| H132  | hidden-vs-KV head-to-head | hidden_beats_kv ∨ kv_beats_hidden ∨ tie (falsifiable) |
| H132b | deep substrate hybrid V5 five-way flag | true under default config |
| H133  | substrate adapter V5 substrate_v5_full tier | only the V5 in-repo runtime |

### R-126 — long-horizon retention / reconstruction / aggressive cramming (13 H-bars)

| H | Subject | Bar |
| --- | --- | --- |
| H134  | persistent V12 replay-EMA propagates | through 64 turns |
| H134b | persistent V12 chain walk depth | ≥ 64 |
| H134c | persistent V12 distractor basis | rank == 4 |
| H135  | LHR V12 six-way runs | proxy / substrate / hidden / attention / retrieval / replay MSEs reported |
| H135b | LHR V12 two-layer scorer fit | post ≤ pre + 1e-9 (ridge over post-ReLU features) |
| H135c | LHR V12 max_k | == 96 |
| H136  | ECC V12 bits/visible-token | ≥ 23.0 |
| H136b | ECC V12 total codes | == 2^21 = 2 097 152 |
| H136c | ECC V12 2048-bit/token falsifier | reproduces structural ceiling |
| H137  | multi-hop V10 chain length | == 15 over 16 backends, n_edges == 240 |
| H137b | multi-hop V10 replay axis | replay_axis_used + compromise_threshold ≥ 1 |
| H138  | TVS V9 ten arms sum to one | n_arms == 10 |
| H138b | TVS V9 replay arm dominates | when rp = 0.95 strictly above other axes (= 0.4) |

### R-127 — corruption / disagreement / consensus / fallback (12 H-bars)

| H | Subject | Bar |
| --- | --- | --- |
| H139  | CRC V8 KV-256 detect rate | ≥ 0.99 (single-byte flips) |
| H139b | CRC V8 post-replay top-K agreement | ≥ pre-replay top-K agreement |
| H139c | CRC V8 adversarial 11-bit burst detect | ≥ 0.95 |
| H140  | consensus V6 10-stage chain enumerated | n_stages == 10 |
| H140b | consensus V6 replay_controller stage fires | when only the replay oracle resolves the tie |
| H140c | consensus V6 abstains | when all paths below floor and no oracles |
| H141  | uncertainty V8 brackets | pessimistic ≤ weighted ≤ optimistic |
| H141b | uncertainty V8 replay_aware | true when replay fidelities are distinct |
| H142  | disagreement algebra V6 replay identity | ok under valid oracle |
| H142b | disagreement algebra V6 replay falsifier | not-ok under flipped oracle |
| H143  | MLSC V8 replay chain inheritance | union of parents + merge addition |
| H143b | MLSC V8 substrate chain + provenance trust inheritance | union of parents + merge addition |

## Strong success / partial success / failure

* **Strong success (claimed)** — 45/45 H-bars pass at 3 seeds
  on the released bench AND the W60 envelope verifier rejects
  every named failure mode (≥ 50) AND the trivial passthrough is
  preserved. Multiple substrate-facing mechanisms (V5 substrate +
  KV V5 + HSB V4 + cache V3 + replay controller) are
  *behaviourally load-bearing on the in-repo substrate*.
* **Partial success** — at least one of the substrate-coupling
  mechanisms (M1, M2, M3, M5, M6, M7, M13, M18) is real and
  behaviourally distinct from the V4 baseline, but several
  others remain structurally-only useful.
* **Failure** — most of the substrate-coupling mechanisms reduce
  to the V4 baseline byte-for-byte, OR the trivial passthrough
  is broken.

W60 reports: **strong success**. 45/45 H-bars pass at 3 seeds.
135/135 cells pass.

## Stable boundary

W60 ships at ``coordpy.tiny_substrate_v5``, ``coordpy.kv_bridge_v5``,
``coordpy.hidden_state_bridge_v4``,
``coordpy.prefix_state_bridge_v4``,
``coordpy.attention_steering_bridge_v4``,
``coordpy.cache_controller_v3``, ``coordpy.replay_controller``,
``coordpy.persistent_latent_v12``,
``coordpy.multi_hop_translator_v10``,
``coordpy.mergeable_latent_capsule_v8``,
``coordpy.consensus_fallback_controller_v6``,
``coordpy.corruption_robust_carrier_v8``,
``coordpy.long_horizon_retention_v12``,
``coordpy.ecc_codebook_v12``,
``coordpy.transcript_vs_shared_arbiter_v9``,
``coordpy.uncertainty_layer_v8``,
``coordpy.disagreement_algebra_v6``,
``coordpy.deep_substrate_hybrid_v5``,
``coordpy.substrate_adapter_v5``, ``coordpy.w60_team``,
``coordpy.r125_benchmark``, ``coordpy.r126_benchmark``,
``coordpy.r127_benchmark`` — reachable only through explicit
imports. ``coordpy.__version__`` remains ``0.5.20``; SDK contract
is byte-for-byte unchanged. **No PyPI release**.
