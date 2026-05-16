# W63 — Stronger Replay-Dominance / Hidden-Wins / 4096-Turn Substrate-Coupled Latent Operating System

> Post-W62 research milestone. 2026-05-15. Strong success per the
> W63 success criterion.

## TL;DR

W63 is the **eighth substrate-attack milestone**. It extends W62
on five fronts simultaneously:

1. **Richer in-repo substrate** — `coordpy.tiny_substrate_v8` adds
   five new internal axes on top of W62's V7 (now 10 layers, GQA,
   per-(layer, head, slot) hidden-vs-KV contention tensor,
   per-layer hidden-state confidence probe, per-(layer, head,
   slot) replay-determinism channel, per-(layer, head) prefix-
   reuse trust ledger, per-(L, H, L, H) cross-layer-head coupling
   matrix).
2. **Stronger trainable controllers** — six-regime replay
   controller V4 with three-way bridge classifier; three-objective
   cache controller V6 with trained retrieval-repair head.
3. **Stronger latent bridges** — four-target KV V8, four-target
   HSB V7 with explicit hidden-wins target, token-content-
   conditional prefix V7, three-stage JS+L1+KL attention V7.
4. **Bigger capsules / retention** — V15 persistent (14 layers,
   12 skip carriers, 4096-turn walk, distractor rank 10),
   V15 LHR (14-head, max_k=160, five-layer scorer), V15 ECC
   (2^24 codes, 26.333 bits/token).
5. **Stronger eight-way hybrid + 49 H-bars across 3 benchmark
   families** — R-134 (17), R-135 (16), R-136 (16) all pass 3/3
   seeds (147/147 cells).

## Success criterion (pre-committed, falsifiable)

The W63 success bar requires:

* ≥ 12 mechanism advances ✓ 19
* ≥ 3 benchmark families ✓ 3 (R-134, R-135, R-136)
* ≥ 1 substrate-coupling bar ✓ H181..H181e, H187, H200
* ≥ 1 latent-to-KV bar ✓ H186, H186b
* ≥ 1 latent-to-hidden-state bar ✓ H183b, H183c
* ≥ 1 cache-reuse-vs-recompute bar ✓ H181b
* ≥ 1 trainable substrate-controller bar ✓ H181c, H182, H182b, H183
* ≥ 1 cache retrieval/eviction policy bar ✓ H182b, H182c
* ≥ 1 hidden-vs-KV comparison bar ✓ H183c, H186b, H202b
* ≥ 1 replay-controller dominance bar ✓ H181, H181b
* ≥ 1 long-horizon retention / reconstruction bar ✓ H188, H189
* ≥ 1 merge / consensus / disagreement bar ✓ H198, H198b, H198c, H199
* ≥ 1 hostile-channel / corruption bar ✓ H196, H196b, H196c
* ≥ 1 transcript-vs-shared head-to-head ✓ H193, H193b
* ≥ 1 abstain / fallback bar ✓ H197b (hidden_wins_arbiter)
* ≥ 1 falsifier ✓ H186b, H190c, H199b, H202b
* ≥ 1 limitation reproduction ✓ W63-L-* family (15 caps)
* ≥ 1 cramming / robustness-vs-rate tradeoff ✓ H190, H190b, H190c

All 49 H-bars pass 3/3 seeds. **Strong success**.

## Failure-mode coverage

W63 envelope verifier enumerates **72 disjoint failure modes**
(`coordpy.w63_team.W63_ENVELOPE_VERIFIER_FAILURE_MODES`). The
cumulative trust boundary across W22..W63 is **916 enumerated
failure modes**.

## Honest scope

* W63 makes the **in-repo V8 substrate** richer, not the third-
  party hosted-model substrate. `W63-L-NO-THIRD-PARTY-SUBSTRATE-
  COUPLING-CAP` carries forward W56..W62 unchanged.
* All five new fits are **closed-form linear ridge solves**.
  `W63-L-V8-NO-AUTOGRAD-CAP` documents.
* The V8 substrate is `10 layers / d_model=64 / byte-vocab /
  max_len=128 / untrained` NumPy on CPU. `W63-L-NUMPY-CPU-V8-
  SUBSTRATE-CAP`. Not a frontier model.
* The three-way bridge classifier is trained on synthetic
  supervision. It does NOT prove hidden bridges beat KV bridges
  in the wild. `W63-L-CONSENSUS-V9-HIDDEN-WINS-STAGE-SYNTHETIC-CAP`.
* The hidden-wins target in the V8 KV bridge four-target stack
  is *constructed*. `W63-L-KV-BRIDGE-V8-HIDDEN-WINS-TARGET-
  CONSTRUCTED-CAP`.

## Hard verdict on substrate breach

W63 deepens the in-repo substrate breach. Five new internal axes,
five new closed-form ridge solves, an eight-way bidirectional
hybrid loop, and a hidden-wins falsifier. The **third-party
hosted-model substrate remains blocked**. Closing that gap
requires hooks for per-(layer, head, slot) cache-write ledgers,
per-layer logit-lens probes, per-(L, H, L, H) coupling matrices,
and per-(layer, head) replay-trust EMAs in the hosted runtime —
which the HTTP surface does NOT expose.

## Files (new in W63)

```
coordpy/tiny_substrate_v8.py
coordpy/kv_bridge_v8.py
coordpy/hidden_state_bridge_v7.py
coordpy/prefix_state_bridge_v7.py
coordpy/attention_steering_bridge_v7.py
coordpy/cache_controller_v6.py
coordpy/replay_controller_v4.py
coordpy/deep_substrate_hybrid_v8.py
coordpy/persistent_latent_v15.py
coordpy/multi_hop_translator_v13.py
coordpy/mergeable_latent_capsule_v11.py
coordpy/consensus_fallback_controller_v9.py
coordpy/corruption_robust_carrier_v11.py
coordpy/long_horizon_retention_v15.py
coordpy/ecc_codebook_v15.py
coordpy/uncertainty_layer_v11.py
coordpy/disagreement_algebra_v9.py
coordpy/transcript_vs_shared_arbiter_v12.py
coordpy/substrate_adapter_v8.py
coordpy/w63_team.py
coordpy/r134_benchmark.py
coordpy/r135_benchmark.py
coordpy/r136_benchmark.py
tests/test_w63_modules.py
tests/test_w63_team_envelope_chain.py
tests/test_w63_trivial_passthrough_byte_identical.py
tests/test_r134_r135_r136_w63.py
```

## Empirical results (seed 199)

```
R-134: 17/17 cells passed
R-135: 16/16 cells passed
R-136: 16/16 cells passed
TOTAL: 49/49 H-bars passed at seed 199.

Across (199, 299, 399): 147/147 cells passed.
```

## Notable single-shot numbers

* **H187**: in-repo V8 substrate adapter returns
  `substrate_v8_full` tier; hosted backends do NOT.
* **H190**: ECC V15 emits **26.333 bits/visible-token** at full
  emit (vs V14's 25.333 — +1 bit/token).
* **H190b**: ECC V15 total codes = `2^24 = 16 777 216`.
* **H191b**: multi-hop V13 has **24 backends, 552 edges**.
* **H193**: TVS V12 has **13 arms** summing to 1.0 within 1e-6.
* **H197**: consensus V9 has **13 disjoint stages** (V8's 12 +
  hidden_wins_arbiter).
* **H183**: three-way bridge classifier reaches **96 % training
  accuracy** on cleanly-separable synthetic supervision.
* **H188c**: persistent V15 distractor rank = **10** (V14 = 8).
* **H202**: envelope verifier knows **72 failure modes**.
* **H202b**: hidden-wins falsifier returns **0** under inversion
  on all three regimes (tie / hidden-wins / kv-wins).
