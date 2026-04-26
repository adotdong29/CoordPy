# Benchmark Reproduction Guide

Everything reported in `vision_mvp/RESULTS*.md` is reproducible by running
the commands below. Numbers may vary ±5% due to random seeds and hardware
quirks, but scaling trends are deterministic.

## Prerequisites

```bash
git clone <this-repo>
cd context-zero
pip install -e .
```

NumPy is the only hard dependency. For the LLM benchmarks, install Ollama
and pull two models:

```bash
ollama pull qwen2.5:0.5b           # 397 MB, for Phase 5 / 6 consensus
ollama pull qwen2.5-coder:7b       # 4.7 GB, for Phase 7 code review
ollama serve &
```

## Unit + integration tests (≤ 1 s)

```bash
python -m vision_mvp.experiments.scaling --n-values 10 50 200 1000 5000
```

Expected: 113+ passing tests, no failures. Verifies that peak context
stays at ⌈log₂ N⌉ exactly across protocols.

## Phase 1 — pure-numpy scaling sweep (≤ 5 min)

```bash
python -m vision_mvp.experiments.scaling --n-values 10 50 200 1000 5000 10000
```

Expected output: a table where `peak_agent_context` column = ⌈log₂ N⌉ for
the `full` and `manifold` protocols at every N.

## Phase 2/3 — drifting consensus with learned basis (≤ 5 min)

```bash
python -m vision_mvp.experiments.phase2
python -m vision_mvp.experiments.phase3
```

Expected: the hierarchical protocol achieves O(log N) writes/round and
oracle-level tracking accuracy after warm-up.

## Phase 4 — massive-N validation (≤ 10 min)

```bash
python -m vision_mvp.experiments.phase4
```

Expected: peak context per agent = ⌈log₂ N⌉ up to N=100 000.

## Phase 5 — small LLM validation (~2 min)

```bash
python -m vision_mvp.experiments.phase5_llm --n-values 5 10 --question-idx 0 2
```

Expected: ~25 % token savings vs naive on N=10 questions.

## Phase 6 — 1 000–5 000 LLM agents (~1–2 min each)

```bash
python -m vision_mvp.experiments.phase6_llm_1000 --n 1000
python -m vision_mvp.experiments.phase6_llm_1000 --n 5000
```

Expected for N=5000: 46 s wall, 7 513 LLM tokens, 100 % accuracy, naive/vision
ratio ≈ 76 800×.

## Phase 7 — actual-task code review (~7 min per task)

```bash
python -m vision_mvp.experiments.phase7_code_review --task sql --n 100
python -m vision_mvp.experiments.phase7_code_review --task race --n 50
python -m vision_mvp.experiments.phase7_code_review --task memory --n 50
```

Expected: the synthesis step outputs a 3-section report with the correct
critical issue named. For SQL: 100 % sample accuracy, synthesis contains
"SQL injection".

## Full-suite quick reproduction

```bash
python -m vision_mvp demo --n 500        # ~5 s
python examples/01_basic_consensus.py    # ~5 s
python examples/02_drift_tracking.py     # ~10 s
python examples/03_scaling_demo.py       # ~5 s
# Examples 04 and 05 need Ollama + specific models
```

## Hardware notes

All numbers were produced on an Apple M3 Pro with 18 GB unified memory.
Pure-numpy phases use a single CPU core well; phase 4–5 scale up to ~4
cores. Ollama runs on GPU (via Metal) at 15–25 tok/s for the 0.5 B and
7 B models respectively.

If you are on a smaller machine, reduce `--n` accordingly. The scaling
law is independent of hardware — at N=100 you should always see peak
context = 7.
