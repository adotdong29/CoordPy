# W64 — Replay-Dominance-Primary / Hidden-Wins-Primary / 6144-Turn / Nine-Way Substrate-Coupled Latent OS

> Post-W63 research milestone, 2026-05-15. Ninth substrate-attack
> milestone in the programme. No version bump; no PyPI release.

## What ships

Nineteen orthogonal substrate-coupling and capsule-native advances
on top of W63:

| Module | Path | Headline |
| ------ | ---- | -------- |
| M1 | `coordpy.tiny_substrate_v9` | 11 layers; five new V9 axes — hidden-wins-primary tensor, replay-dominance-witness channel, attention-entropy probe, cache-similarity matrix, hidden-state-trust ledger |
| M2 | `coordpy.kv_bridge_v9` | Five-target stacked ridge fit (4 V8 + 1 replay-dominance-primary); per-bucket hidden-wins-primary falsifier; KV-fingerprint perturbation control |
| M3 | `coordpy.hidden_state_bridge_v8` | Five-target stacked ridge with hidden-wins-primary target; recovery audit V4 (three-stage); V9 hidden-state-trust coupling |
| M4 | `coordpy.prefix_state_bridge_v8` | Token+role-conditional 12-feature stacked drift-curve predictor over up to K=32; three-way prefix/hidden/replay comparator |
| M5 | `coordpy.attention_steering_bridge_v8` | Four-stage clamp (Hellinger + JS + coarse L1 + fine KL); per-bucket entropy-amplified falsifier; attention-map delta L2 |
| M6 | `coordpy.cache_controller_v7` | Four-objective stacked ridge + similarity-aware eviction (6-dim) + composite_v7 (8-head mixture) |
| M7 | `coordpy.replay_controller_v5` | Per-regime 10×4 ridge × 7 regimes; 9-dim regime gate; 4×9 four-way bridge classifier; 4×9 replay-dominance-primary head; replay-dominance-primary REUSE bonus |
| M8 | `coordpy.deep_substrate_hybrid_v9` | Nine-way bidirectional loop |
| M9 | `coordpy.substrate_adapter_v9` | 5 new capability axes; `substrate_v9_full` tier |
| M10 | `coordpy.persistent_latent_v16` | 15 layers; thirteenth skip carrier; `max_chain_walk_depth=6144`; distractor rank 12 |
| M11 | `coordpy.multi_hop_translator_v14` | 27 backends; 702 directed edges; chain-length-21; 9-axis composite |
| M12 | `coordpy.mergeable_latent_capsule_v12` | replay-dominance-primary chain + hidden-state-trust chain + TV distance |
| M13 | `coordpy.consensus_fallback_controller_v10` | 14-stage chain inserting `replay_dominance_primary_arbiter` |
| M14 | `coordpy.corruption_robust_carrier_v12` | 4096-bucket fingerprint; 23-bit adversarial burst; replay-dominance recovery probe; substrate-corruption blast-radius probe |
| M15 | `coordpy.long_horizon_retention_v16` | 15 heads; max_k=192; six-layer scorer (random+silu fifth) |
| M16 | `coordpy.ecc_codebook_v16` | 2^25 = 33 554 432 codes; 27.333 bits/visible-token at full emit (≥ 27.0) |
| M17 | `coordpy.uncertainty_layer_v12` | 11-axis composite (adds `replay_dominance_primary_fidelity`) |
| M18 | `coordpy.disagreement_algebra_v10` | TV-equivalence identity + falsifier |
| M19 | `coordpy.transcript_vs_shared_arbiter_v13` | 14-arm comparator (adds `replay_dominance_primary` arm) |

## Benchmark results

R-137 + R-138 + R-139 over 3 seeds:
* R-137 (V9 substrate / replay-dominance / hidden-wins-primary / four-way bridge / nine-way hybrid): **20/20** H-bars × 3 seeds = 60/60 cells
* R-138 (long-horizon retention / persistent V16 / multi-hop V14 / LHR V16 / ECC V16): **12/12** H-bars × 3 seeds = 36/36 cells
* R-139 (corruption / disagreement / consensus / fallback / replay-dominance-primary / hostile-channel): **19/19** H-bars × 3 seeds = 57/57 cells

**Total: 51/51 H-bars × 3 seeds = 153/153 cells passing (strong success).**

## Honest scope

* The W64 substrate is the in-repo V9 NumPy runtime. Hosted backends remain text-only at the HTTP surface (`W64-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`).
* All "training" is closed-form linear ridge: 23 closed-form ridge solves total across W61..W64 (twelve from W61+W62, five from W63, six from W64). No SGD / autograd / GPU. (`W64-L-V9-NO-AUTOGRAD-CAP`)
* The V9 substrate is 11 layers / d_model=64 / byte-vocab / max_len=128 / untrained NumPy on CPU. NOT a frontier model. (`W64-L-NUMPY-CPU-V9-SUBSTRATE-CAP`)
* The fifth (replay-dominance-primary) target in the V9 KV bridge is *constructed* to be unreachable by KV alone (`W64-L-KV-BRIDGE-V9-REPLAY-DOMINANCE-PRIMARY-TARGET-CONSTRUCTED-CAP`).
* The 27 multi-hop backends are NAMED, not EXECUTED (`W64-L-MULTI-HOP-V14-SYNTHETIC-BACKENDS-CAP`).
* The four-way bridge classifier is fit on synthetic supervision; it does NOT prove that replay bridges beat hidden / prefix / KV bridges in general (`W64-L-CONSENSUS-V10-REPLAY-DOMINANCE-PRIMARY-STAGE-SYNTHETIC-CAP`).
* The 4096-bucket fingerprint is wrap-around XOR over the in-repo substrate cache, not third-party hosted cache state (`W64-L-CRC-V12-FINGERPRINT-SYNTHETIC-CAP`).

## Envelope chain

* `W63 envelope CID == W64.w63_outer_cid` (verified by `test_w64_team_envelope_chain`).
* Trivial passthrough preserves byte-for-byte (verified by `test_w64_trivial_passthrough_byte_identical`).
* W64 envelope verifier enumerates **86 disjoint failure modes** (≥ 85 target met).

## Cumulative trust boundary across W22..W64

**1002 enumerated failure modes** (916 from W22..W63 + 86 new W64 envelope verifier modes).
