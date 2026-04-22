# Vision MVP — Phase 4 Results

**Date:** 2026-04-16 (same session).
**What's new in Phase 4:**
1. **Vectorized predictor bank** — all N neural predictors run in a single `einsum` pass, collapsing the Python loop. Enables N ≥ 10⁴.
2. **Holographic boundary protocol** (VISION_MILLIONS Idea 5). ~N^{2/3} boundary agents are the only eligible writers; interior agents read-only. Models Ryu-Takayanagi on an agent team.
3. **Massive-N validation** — hierarchical protocol pushed up to N = 100 000 agents.

---

## Experiment 1 — Massive-N scaling (hierarchical)

| N | Steps | Workspace | Peak ctx | Writes/round | Steady err | Basis alignment | Wall time | Total tokens |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 000 | 60 | **13** | **13** | 13.00 | 0.108 | 0.90 | 5.0 s | 3.9 M |
| 20 000 | 60 | **15** | **15** | 15.00 | 0.452 | 0.97 | 23.4 s | 18 M |
| 50 000 | 40 | **16** | **16** | 16.00 | 0.125 | 0.98 | 38.2 s | 32 M |
| **100 000** | 40 | **17** | **17** | **17.00** | **0.151** | **0.99** | **81.4 s** | **68 M** |

At every scale, **workspace = peak context = writes per round = ⌈log₂ N⌉**. The three metrics track each other exactly all the way to N = 100 000. Basis alignment climbs from 0.90 to 0.99 as more observations arrive per round.

The N = 20 000 case has a higher steady error (0.45) than neighbors — the drift magnitude accumulated enough over 60 steps to cross a basis-reorientation event. This is real wobble in the adaptive system, not noise.

Wall time grows roughly linearly with N (5 000→5s, 100 000→81s), as expected when per-agent work is O(1) per round.

---

## Experiment 2 — Hierarchical vs Holographic head-to-head

Same drifting-consensus task, 200 steps, drift σ=0.05, at three scales.

| N | Protocol | Peak ctx | Writes/round | Steady err | Tokens | Wall | Boundary size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 000 | hierarchical | 10 | 10.0 | 0.145 | 607 090 | 1.3 s | — |
| 1 000 | holographic | 10 | **7.0** | 0.187 | 604 543 | 2.4 s | 100 |
| 10 000 | hierarchical | 14 | 14.0 | 0.094 | 8 413 230 | 11.4 s | — |
| 10 000 | holographic | 14 | **9.0** | 0.102 | 8 407 965 | 19.8 s | 465 |
| 50 000 | hierarchical | 16 | 16.0 | 0.114 | 32 011 248 | 39.0 s | — |
| 50 000 | holographic | 16 | **11.0** | 0.123 | 32 007 293 | 66.0 s | 1 358 |

Holographic trade-offs:
- **30–36% fewer writes per round** at every N (workspace draws from the smaller N^{2/3} boundary set).
- Total tokens essentially identical — reads dominate, and both protocols broadcast to all N agents.
- Slight accuracy loss (0.1–0.3 σ).
- Wall time ~2× slower (the holographic bus still broadcasts-to-all per round).

**When holographic is worth it:** when writes are genuinely the bottleneck (bandwidth, not compute). For systems with expensive write semantics (e.g., persistent storage, distributed consensus), the ~30% reduction compounds.

**When hierarchical is worth it:** for in-memory compute-bound systems where all agents can read cheaply but writes don't dominate.

Both protocols preserve the O(log N) peak-context bound — holographic does not break it.

---

## The Final Scaling Table

Across Phases 1–4, the key metric — **peak per-agent context** — tracks ⌈log₂ N⌉ exactly at every scale:

| N | log₂ N | Phase 1 (full-stack) | Phase 3 (hierarchical) | Phase 4 (hierarchical) |
|---:|---:|---:|---:|---:|
| 10 | 3.3 | 4 | — | — |
| 50 | 5.6 | 6 | 6 | — |
| 200 | 7.6 | 8 | 8 | — |
| 1 000 | 10.0 | 10 | 10 | 10 |
| 5 000 | 12.3 | 13 | 13 | 13 |
| 10 000 | 13.3 | 14 | 14 | 14 |
| 20 000 | 14.3 | — | — | 15 |
| 50 000 | 15.6 | — | — | 16 |
| **100 000** | **16.6** | — | — | **17** |

Four different implementations of increasing sophistication (Phase 1 on a static task, Phase 3 on a drifting task with learned basis and neural predictor, Phase 4 on massive N) — all return the same scaling.

That is the empirical confirmation of CASR's O(log N) claim across the regimes we built.

---

## Ideas Implemented So Far (of 10 in VISION_MILLIONS)

| Idea | Name | Status |
|:-:|---|:-:|
| 1 | Shared Latent Manifold | ✓ |
| 2 | Generative Agent Networks | ✓ (neural predictor) |
| 3 | Stigmergic Environment | ✓ |
| 4 | Global Workspace | ✓ |
| 5 | Holographic Boundary | ✓ |
| 6 | Swarm Physics | — |
| 7 | Language-as-Protocol | next (Phase 5) |
| 8 | Market-Cleared Routing | — |
| 9 | Pre-Shared Randomness | — |
| 10 | Continuous Scale | — |

**Five of ten ideas empirically demonstrated.** Phase 5 plugs in real LLM agents for idea 7.
