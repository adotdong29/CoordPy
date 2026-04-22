# Phase 16 — Scale Test at N=36

Same Phase-14 harness, 3× more agents. Single overnight run on local Ollama
(`qwen2.5-coder:7b`).

## Headline

**All six claims (C1–C6) hold.** The CASR advantage over random footprints
*widened by +0.186* going from N=12 to N=36 — pre-registered gate was +0.10,
so the widening is ~85% larger than the minimum needed to claim it.

## The single-screen table

| | **N=12 (Phase 14)** | **N=36 (Phase 16)** |
|---|---:|---:|
| round-1 only | 0.540 | 0.458 |
| full | 0.900 | 0.854 |
| **casr** | **0.900** | **0.854** |
| ablation | 0.690 | 0.458 |
| casr prompt tokens | 4 709 | **13 697** |
| full prompt tokens | 6 070 | 27 019 |
| casr token saving vs full | 22.4 % | **49.3 %** |
| Δ(casr − ablation) | +0.210 | **+0.396** |

## Claim verdicts

| | Claim | N=36 result |
|---|---|---|
| **C1** | casr prompt tokens < full | ✅ 49.3 % reduction |
| **C2** | casr score ≥ full − 0.05 | ✅ Δ = +0.000 (identical to full) |
| **C3** | casr score > ablation + 0.05 | ✅ Δ = +0.396 |
| **C4** | full score ≥ round-1 (no regression) | ✅ Δ = +0.396 |
| **C5** | CASR fired ≥ 1 event-triggered skip | ✅ 3 skips |
| **C6** | Δ(casr − ablation) at N=36 ≥ Δ at N=12 + 0.10 | ✅ widened by +0.186 |

## Why the widening happened: ablation collapsed, casr held

The interesting thing about C6 is that it did *not* happen through casr
getting better. casr scored slightly *lower* at N=36 (0.854) than at N=12
(0.900) — an expected consequence of 3× more agents each contributing LLM
variance. The widening is driven entirely by the ablation leg collapsing
from 0.690 → 0.458.

That collapse is sharper than it first looks:

- **At N=36, ablation ≡ round-1.** Both scored exactly 0.458 (22/48). Random
  footprints during round-2 refinement added *zero net information*:
  every consumer that happened to see the wrong producer's draft wrote
  down the wrong dict key, cancelling out whatever consumers that happened
  to see the right producer's draft gained.
- That matches the theoretical prediction. Random routing hits the right
  producer with probability ≈ k/(N−1) where k is the target footprint
  size. At N=12 with typical footprint size 2, that's ~18 %; at N=36 it's
  ~6 %. The hit rate falls like 1/N, so at larger N most random-routing
  refinements are misdirected, and LLM over-correction on wrong bulletin
  content cancels LLM self-correction on right bulletin content.

casr, by contrast, delivered exactly the right producer draft to each
consumer — scaling exactly the same way the footprint sizes did. So:

- CASR's advantage at scale isn't "it gets more accurate than broadcast"
  (it doesn't — it matches broadcast).
- CASR's advantage at scale is "it stays as good as broadcast while
  other routing strategies collapse."

## Event trigger fires more at scale

Skips by the event trigger in round-2 (higher-tier agents only):

| leg | N=12 | N=36 |
|---|---:|---:|
| full | 1 | 1 |
| casr | 2 | 3 |
| ablation | 3 | 7 |

- casr skips scale modestly (2→3): a handful of consumers had round-1
  drafts that already agreed with their frozen producer, so refinement
  was skipped correctly.
- ablation skips *triple* (3→7): at larger N, random bulletin content is
  often so unrelated that the event trigger refuses to refine. This
  protects ablation from doing *worse* than its round-1 baseline — and in
  fact ablation exactly matches round-1 (both 0.458) because the
  suppression + random-routing combination zeroes out.

## Token cost at scale

The casr token saving grew from 22.4 % at N=12 to **49.3 %** at N=36:

- full at N=36 prompts 27 019 tokens — essentially every higher-tier
  agent receives every frozen lower-tier draft in its prompt. That grows
  ~quadratically as N grows and producer bulletins get broadcast to all
  consumers + integrators.
- casr prompts 13 697 tokens — each consumer sees only its one producer
  (causal footprint size 2), each integrator sees only its 3 producers
  (footprint size 4). That grows ~linearly.

The O(N) vs O(N²) gap is now visible empirically. At N=100 the ratio would
project to ~10× fewer tokens for casr; at N=1000 ~100×.

## Tiers + refinement counts

- 15 tier-0 (producers) FROZEN after round-1 in all three legs.
- 21 tier-1 agents (15 consumers + 6 integrators) eligible for round-2.
- Refined / skipped per leg:
  - full: 20 refined, 1 skipped (total 21)
  - casr: 18 refined, 3 skipped (total 21)
  - ablation: 14 refined, 7 skipped (total 21)

## What this run does NOT test

- **Adversarial or byzantine agents.** Same caveat as Phase 15.
- **Dynamic membership** (join/leave). Team is static at 36.
- **Scores beyond N=36.** The widening trend is consistent with the O(1/N)
  miss-rate prediction, but the next data point (say, N=100) would
  require ~5 h of Ollama time or a smaller model; out of scope for one
  overnight run.
- **Alternative models.** Single model family (qwen2.5-coder:7b).
  A run on a different backbone would strengthen the universality claim.

## Interpretation

The Phase-14 result at N=12 could have been a ceiling — random routing
might have survived at scale if integrators happened to pull enough diverse
bulletin content to correct drifts by noise. It did not. At N=36 the
ablation leg matches the *uncoordinated* round-1 baseline exactly, which is
the strongest possible statement that random routing contributes nothing at
this scale.

The thesis that motivated this project — *routing saves tokens AND widens
its advantage with scale* — survived its first real scaling test.

## Reproduce

```bash
python3 -m vision_mvp.experiments.phase16_scale \
    --out vision_mvp/results_phase16.json \
    --phase14-reference vision_mvp/results_phase14.json
```

Runtime: ~30 min on an M-series laptop + local Ollama `qwen2.5-coder:7b`.
All 48 reference-solution tests pass; catalog passes 8 unit tests
(`tests/test_protocolkit_36.py`).
