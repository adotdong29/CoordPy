# vision_mvp — a built-out demo of the million-agent vision

Four phases in one session. Concrete, running code that validates the core
claims of `VISION_MILLIONS.md` without any LLM: per-agent context scales as
O(log N) even at N = 100 000, writes per round match the workspace size
exactly, and tracking error reaches oracle level at steady state.

## Quick start

```bash
# Smallest quick check (≈5 seconds)
python3 -m vision_mvp.experiments.smoke

# Phase 1 scaling (≈30s; out to N=10k)
python3 -m vision_mvp.experiments.scaling --n-values 10 50 200 1000 5000 10000

# Phase 2 — learned basis, drifting task
python3 -m vision_mvp.experiments.phase2

# Phase 3 — global workspace, neural predictor, role hierarchy
python3 -m vision_mvp.experiments.phase3

# Phase 4 — massive-N (up to 100k) and holographic boundary
python3 -m vision_mvp.experiments.phase4
```

## What each phase added

| Phase | New machinery | Paradigm from `VISION_MILLIONS` |
|:-:|---|---|
| 1 | Shared latent manifold + stigmergic CRDT + surprise-filtered writes | Ideas 1, 3 |
| 2 | Streaming PCA for learned basis + non-stationary drift + linear AR predictor | Idea 2 (linear) |
| 3 | Neural-net predictor (basis-invariant) + Global Workspace + role hierarchy | Ideas 2, 4 |
| 4 | Vectorized predictor bank + holographic boundary + N = 10⁵ validation | Idea 5 |

## Layout

```
vision_mvp/
├── core/
│   ├── agent.py                # Bayesian-updating agent state
│   ├── bus.py                  # Communication accounting
│   ├── manifold.py             # SLM with given or random basis (Phase 1)
│   ├── learned_manifold.py     # Streaming PCA (Phase 2+)
│   ├── stigmergy.py            # Binned CRDT environment
│   ├── predictor.py            # Linear AR predictor (Phase 2)
│   ├── neural_predictor.py     # Per-agent MLP (Phase 3)
│   ├── vectorized_predictor.py # Batched bank of MLPs (Phase 4)
│   ├── workspace.py            # Top-k salience selector (Phase 3)
│   └── hierarchy.py            # Orchestrator/worker roles (Phase 3)
├── tasks/
│   ├── consensus.py            # Static consensus (Phase 1)
│   └── drifting_consensus.py   # Non-stationary trajectory (Phase 2+)
├── protocols/
│   ├── naive.py                # All-to-all baseline
│   ├── gossip.py               # Pairwise averaging
│   ├── manifold_only.py        # SLM alone
│   ├── full_stack.py           # SLM + CRDT + surprise filter (Phase 1)
│   ├── adaptive.py             # Phase 2 stack
│   ├── hierarchical.py         # Phase 3 stack
│   └── holographic.py          # Phase 4 boundary-encoding protocol
├── experiments/
│   ├── smoke.py                # One run of each P1 protocol
│   ├── scaling.py              # P1 N-sweep
│   ├── phase2.py               # Drift, shock, basis learning
│   ├── phase3.py               # Long-horizon, workspace scaling, shock, phase compare
│   └── phase4.py               # Massive N + holographic comparison
├── RESULTS.md                  # Phase 1 results
├── RESULTS_PHASE2.md           # Phase 2 results
├── RESULTS_PHASE3.md           # Phase 3 results
├── RESULTS_PHASE4.md           # Phase 4 results
└── FINAL_RESULTS.md            # Consolidated scaling table across all phases
```

## The one-number summary

At N = 100 000 agents on a drifting consensus task:
- Peak context per agent: **≈17 tokens** (≈ log₂ 100 000).
- Writes per round: **≈17** (= workspace size).
- The naive baseline would require **6.5 million context tokens per agent**
  and ~6.5 × 10¹¹ total — a ratio of about **382 000×** less context.

See `FINAL_RESULTS.md` for the full numbers.
