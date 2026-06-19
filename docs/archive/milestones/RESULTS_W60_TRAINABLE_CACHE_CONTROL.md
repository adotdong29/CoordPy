# Results — W60 Trainable Cache-Control Substrate-Coupled Latent OS

> Empirical results for the W60 milestone (post-W59,
> 2026-05-14). Status vocabulary defined in
> `docs/HOW_NOT_TO_OVERSTATE.md`. Bench gates the corresponding
> H-bars in `docs/SUCCESS_CRITERION_W60_TRAINABLE_CACHE_CONTROL.md`.

## TL;DR

W60 is the **fifth substrate-attack milestone** in the Context
Zero programme. **45 of 45 H-bars pass at 3 seeds** on
R-125 + R-126 + R-127 (135/135 cells). The W60 envelope verifier
enumerates **52 disjoint failure modes** and rejects every
named one on the trivial passthrough. No H-bar required a code-
side hint to pass; every H-bar runs against the V5 substrate or
its bridges with the same default config the team orchestrator
uses.

## Headline numbers (3 seeds, mean)

| Subject | W60 result | Comparison |
| --- | --- | --- |
| substrate V5 forward determinism | 3/3 byte-identical CIDs | unchanged from V4 |
| KV bridge V5 multi-direction ridge fit | converges (post ≤ pre + 1e-9) at n_directions=3 | new vs V4's single α |
| KV bridge V5 logit-direction fit | converges (post ≤ pre + 1e-9) | new vs V4 |
| KV bridge V5 reverse-extract residual L2 | ≤ 2.4e-7 on uncorrupted bank | new |
| HSB V4 per-(layer, head) ridge fit | converges, 16-dim feature solve | new vs V3's single α |
| HSB V4 recovery from adversarial | post ≤ pre + 1e-9 after counter-fit | new |
| attention V4 per-(layer, head, query) budget | every (l,h,q) ≤ 0.4 + 1e-3 | new vs V3's per-head |
| attention V4 negative-budget falsifier | post-KL < 1e-6, no shift | new |
| prefix V4 multi-segment flop saving | ~46% of full recompute | new vs V3's single split |
| cache controller V3 trained_eviction | residual 28.4 → 0.39 (~73x) | new vs V2 |
| cache controller V3 composite_v3 weights | 4-dim ridge mixture, post ≤ pre + 1e-9 | new |
| replay controller chooses REUSE on CRC-passed | 3/3 | new (first first-class policy) |
| replay controller chooses RECOMPUTE on CRC-failed | 3/3 | new |
| hidden-vs-KV head-to-head | hidden_beats_kv on tiny V5 default | new (falsifiable) |
| deep substrate hybrid V5 five_way | True under default config | new vs V4's four_way |
| persistent V12 chain walk depth | ≥ 64 (vs V11's ≥ 32 in r123) | extended |
| persistent V12 distractor basis rank | 4 | new |
| LHR V12 max_k | 96 (vs V11's 80) | extended |
| LHR V12 two-layer scorer fit | post ≤ pre + 1e-9 | new vs V11's single linear |
| ECC V12 bits/visible-token | 23.333 (vs V11's 22.333) | +1 bit/token at full emit |
| ECC V12 total codes | 2 097 152 (= 2^21) | +1 bit |
| multi-hop V10 chain length | 15 over 16 backends, 240 edges | +2 chain, +2 backends |
| multi-hop V10 compromise threshold | ≥ 1 axis (5-axis trust composite) | new five-axis |
| TVS V9 ten arms | sum to 1.0 within 1e-9 | extended (10 vs 9) |
| TVS V9 replay arm dominance @ rp=0.95 | 1.0 pick rate | new arm |
| CRC V8 256-bucket detect rate | 1.0 single-byte flips | extended (256 vs 128) |
| CRC V8 post-replay top-K agreement | ≥ pre-replay agreement | new |
| CRC V8 adversarial 11-bit burst | ≥ 0.95 detect rate | extended |
| consensus V6 stages | 10 (vs V5's 9) | extended |
| consensus V6 replay_controller stage fires | 3/3 under replay-only oracle | new |
| uncertainty V8 axes | 7 (vs V7's 6); replay_aware live | extended |
| disagreement algebra V6 replay identity | ok under valid oracle | new |
| MLSC V8 replay+substrate chain inheritance | union from parents + merge | extended |

## Substrate-coupling verdict

W60 is the strongest honest substrate-coupling milestone the
programme has shipped. The substrate-side mechanisms that are
*behaviourally load-bearing* on the in-repo V5 substrate
runtime:

* **KV Bridge V5** multi-direction ridge fit on real substrate-
  side Jacobians.
* **HSB V4** per-(layer, head) ridge fit (16-dim feature) on
  real per-head hidden-state taps.
* **Cache Controller V3** ``trained_eviction`` ridge fit on
  real ``[hidden, importance, attention_receive_l1, retrieval]``
  features with the V1 leave-one-out drop oracle as supervision.
* **Replay Controller** decisions wired through to the V5
  substrate's actual flop accounting and CRC channel.
* **Deep Substrate Hybrid V5** five-way bridge runs the
  composed V6 ↔ V5 ↔ controller V3 ↔ replay ↔ retrieval loop on
  a real V5 forward.

Substrate-blocked surfaces (carries forward from W57+):

* **Hosted backends** (Ollama / OpenAI-compatible / closed-API)
  remain text-only at the HTTP surface. ``W60-L-NO-THIRD-PARTY-
  SUBSTRATE-COUPLING-CAP`` documents this honestly.
* **Autograd / SGD** through the controllers is NOT done. All
  W60 fits are single closed-form linear ridge solves.
  ``W60-L-V5-NO-AUTOGRAD-CAP`` documents this.

## R-125 — real-substrate / latent-bridge / cache-reuse / trained-controller / hidden-vs-KV (20 H-bars, 3 seeds)

20/20 H-bars pass at 3 seeds; 60/60 cells. Wall time per seed:
~3.6 s.

Selected highlights:

* **H125** substrate V5 forward determinism: 3/3 byte-identical.
* **H126** KV bridge V5 multi-direction ridge fit at n_directions=3
  converges on every seed.
* **H126c** reverse-extract residual L2 ≤ 2.4e-7 on the
  default seed; carrier is recoverable from the cache slot to
  near-machine precision when banks are uncorrupted.
* **H127** HSB V4 per-(layer, head) ridge fit (16-dim solve)
  converges; the new richer fit reduces residual against the
  target logit-shift direction below the V3 single-α bound.
* **H128b** negative-budget falsifier: setting the per-(layer,
  head, query) budget to all-zero forces post-KL < 1e-6 AND no
  attention shift; the attention V4 controller correctly
  collapses to a no-op rather than hallucinating a steered
  pattern.
* **H129** multi-segment partial reuse on the V5 substrate
  saves ~46% of the full-recompute flops on the standard
  {reuse, recompute, drop} split with bounded drift.
* **H130** trained-eviction ridge fit reduces residual from
  28.4 → 0.39 (over 70x improvement) against the leave-one-out
  drop oracle on the default config.
* **H131 / H131b** the W60 ReplayController correctly chooses
  REUSE on CRC-passed candidates and RECOMPUTE on CRC-failed
  candidates.
* **H132** hidden-vs-KV head-to-head: on the tiny V5 default,
  the hidden-state V4 per-(layer, head) ridge fit gets a smaller
  L2 residual to the target logit-shift direction than the
  KV V4 single-α correction. This is the first time the
  programme has a falsifiable head-to-head.
* **H132b** the deep substrate hybrid V5 five-way bridge fires
  ``five_way=True`` end-to-end on the default config.

## R-126 — long-horizon retention / reconstruction / aggressive cramming (13 H-bars, 3 seeds)

13/13 H-bars pass at 3 seeds; 39/39 cells. Wall time per seed:
~0.8 s.

Selected highlights:

* **H134** persistent V12 replay-EMA propagates through 64 turns
  with measurable replay carrier L2.
* **H134c** distractor basis at init has rank == 4 (Gram-Schmidt
  over random vectors); the replay skip is projected orthogonal
  to the basis before injection.
* **H135b** LHR V12 two-layer scorer (random projection +
  frozen ReLU + closed-form ridge over the post-ReLU features)
  reduces residual on the synthetic supervised set; fit is a
  single linear solve on top of frozen features.
* **H135c** LHR V12 max_k = 96 (extends V11's 80).
* **H136** ECC V12 hits 23.333 bits/visible-token at full emit
  (target ≥ 23.0).
* **H136c** ECC V12 falsifier: a 2048 bits/token target lies
  above the V12 structural ceiling (log2(2 097 152) = 21);
  reproduces honestly as the W60 falsifier.
* **H137b** multi-hop V10 reports compromise_threshold ≥ 1 (the
  five-axis composite is robust against single-axis attack on
  the dominant path).

## R-127 — corruption / disagreement / consensus / fallback (12 H-bars, 3 seeds)

12/12 H-bars pass at 3 seeds; 36/36 cells. Wall time per seed:
~0.06 s.

Selected highlights:

* **H139** CRC V8 256-bucket fingerprint detects 100% of
  single-byte flips (well above the 0.99 floor).
* **H139b** post-replay top-K agreement is ≥ pre-replay top-K
  agreement, validating that the ReplayController's RECOMPUTE
  decision restores corrupted slot scores.
* **H140b** consensus V6 ``replay_controller_choice`` stage
  fires correctly when only the replay oracle resolves the
  tie.
* **H140c** consensus V6 abstains when all paths below floor
  and no oracles are available.
* **H141b** uncertainty V8 reports ``replay_aware = True`` when
  replay fidelities are distinct.
* **H142b** disagreement algebra V6 replay falsifier produces
  ``not-ok`` when the oracle disagrees on argmax — the
  identity is genuinely falsifiable, not asserted.

## Envelope chain

End-to-end live verification:

```
W59 envelope ok: True
W60 envelope ok: True
W59→W60 chain CID match: True
W60 weighted mean: 0.804
W60 five_way_used: True
W60 substrate_v5_used: True
```

The W60 envelope's ``w59_outer_cid`` carries the W59 envelope CID
byte-for-byte under default config; the trivial W60 envelope
correctly fails verification with 19+ missing-witness failure
modes.

## Test wall time

* `tests/test_w60_modules.py`: 23 tests, ~10 s.
* `tests/test_r125_r126_r127_w60.py`: 3 tests (one per family
  at 1 seed each), ~5 s.
* W55-W60 module + benchmark + smoke regression: 211 tests,
  ~9 minutes total.

## What W60 does NOT claim

* W60 does NOT claim third-party hosted-model substrate access.
  ``W60-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` carries
  forward unchanged from W57+.
* W60 does NOT claim end-to-end backprop. All fits are single
  closed-form ridge solves; ``W60-L-V5-NO-AUTOGRAD-CAP`` is
  load-bearing.
* The H132 hidden-vs-KV claim is a *constructive observation on
  the tiny V5 config*, NOT a general claim about hidden vs KV
  injection at frontier scale; ``W60-C-HIDDEN-DOMINATES-KV-AT-
  SCALE`` is open.
* The 16 "backends" of multi-hop V10 are *named*, not
  *executed*; this is a graph + trust arbiter, not a multi-
  machine harness. ``W60-L-MULTI-HOP-V10-SYNTHETIC-BACKENDS-
  CAP`` documents this.
* The V5 corruption flag channel is a *channel*, not a
  detector. Flags must be written by external CRC V8 to be
  observable. ``W60-L-CORRUPTION-FLAG-CHANNEL-CAP`` documents
  this.
