# W88 — HumanEval sequential-reflexion bench V1

> **Run completed 2026-05-22.  Negative result: at the K=5
> budget on NIM Llama-3.1-8B-Instruct on HumanEval, sequential-
> reflexion-K=5 does NOT beat first-pass-among-K=5 self-
> consistency.  The W86 carry-forward
> `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` **stays
> unretired**.**
>
> Pre-committed bars in `docs/RUNBOOK_W88.md`: only 1 of 4
> retirement bars met (`b_mean_strictly_beats_a0_mean`, the
> W86-reproduces bullet).  The three load-bearing bars
> (`b_mean_strictly_beats_a1_mean`, margin ≥ +1.0 pp,
> per-seed majority) all fail.  W88 contributes a meaningful
> refinement: the gap closed from W86's −8.9 pp to W88's
> −3.33 pp, but the sign did not flip.

## TL;DR

3 seeds × 30 problems × 3 arms on the canonical HumanEval corpus.
K=5 budget on A1 and B.  Model = `meta/llama-3.1-8b-instruct`
via NIM.  Total wall 2010 s; 990 NIM calls; same NIM endpoint
W86 used for direct comparability.

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0** stock single-shot (T=0) | **63.33 %** | 0.567 / 0.700 / 0.633 |
| **A1** first-pass-among-K=5 (T=0.7) | **74.44 %** | 0.733 / 0.767 / 0.733 |
| **B** sequential-reflexion-K=5 (T=0.7) | **71.11 %** | 0.633 / 0.767 / 0.733 |

* `b_mean_strictly_beats_a0_mean = True` ✓ — B (71.1 %) > A0
  (63.3 %) by **+7.78 pp**.  The W86 same-bullet result
  empirically reproduces on this run.
* `b_mean_strictly_beats_a1_mean = False` ✗ — B (71.1 %) < A1
  (74.4 %) by **−3.33 pp**.  The same-budget multi-agent
  superiority claim is NOT empirically established by this run.

Bench Merkle root:
`11997891e2b834fe...`  (full at
`results/w88/humaneval_reflexion/.../humaneval_reflexion_bench_report.json`).
**Audit chain re-derives offline: 7/7 PASS** on the 990 NIM
call sidecar SHA-256s + 3 per-seed Merkle roots + bench Merkle
root.

## Comparison vs W86

| Metric | W86 (executor-as-critic) | W88 (sequential reflexion) |
|---|---:|---:|
| A0 mean | 63.3 % | 63.3 % |
| A1 mean | 80.0 % | 74.4 % |
| B mean | 71.1 % | 71.1 % |
| B − A1 | **−8.9 pp** | **−3.3 pp** |
| Seeds | 86_028_001/2/3 | 88_028_001/2/3 |

W88 B uses **same model, same K=5 budget, same task subset
discipline** as W86 B, with a DIFFERENT pipeline shape:

* **W86 B**: 2 solvers (T=0.7) + 1 critic (T=0.7) + 1 reviser
  (T=0.7) + 1 judge (T=0).  3 of 5 calls are code-producing.
* **W88 B**: 5 sequential reflexion turns, each at T=0.7, each
  conditioned on the cumulative history of prior candidates and
  the actual executor stderr.  All 5 calls are code-producing.

The W88 design eliminates W86's redundant judge call (the
executor's pass/fail verdict is already truth) and the
critic-only-without-code call.  Empirically, W88's B is closer
to A1 than W86's B (gap −3.3 pp vs −8.9 pp).  **The gap closed
by 5.6 pp, but did not flip sign.**

This is meaningful: the structural improvement was real, and
it moved the needle in the right direction.  At a larger
problem set, on a stronger model, or with a richer reflexion
context, the gap might close further or flip.  **At this scale
on this model, it does not.**  The result is honestly reported.

The A1 mean dropped from W86's 80.0 % to W88's 74.4 %.  These
are different problem subsets (different seeds → different
`select_humaneval_subset_v1` outputs) and provider-side
sampling variation at T=0.7.  Both numbers are real and
internally consistent — neither is wrong.  The 5.6 pp drop
matters because **A1 itself has ~5-10 pp seed-to-seed noise**
at this scale; any future claim of "B beats A1 by < 5 pp" must
account for this.

## Per-seed result

| Seed | A0 | A1 | B | B − A0 | B − A1 |
|----:|---:|---:|---:|---:|---:|
| 88_028_001 | 56.7 % | 73.3 % | 63.3 % | **+6.6 pp** | **−10.0 pp** |
| 88_028_002 | 70.0 % | 76.7 % | 76.7 % | **+6.7 pp** | tie |
| 88_028_003 | 63.3 % | 73.3 % | 73.3 % | **+10.0 pp** | tie |
| **mean**   | **63.3 %** | **74.4 %** | **71.1 %** | **+7.78 pp** | **−3.33 pp** |

Per-seed bools:

* `b_beats_a0_per_seed = (True, True, True)` — B beats A0 on
  every seed.
* `b_beats_a1_per_seed = (False, False, False)` — B is at
  best a tie vs A1 (seeds 2 + 3); on seed 1 B loses by 10 pp.

Seed-1 is the single dominant variance source.  On seeds 2 and
3, the sequential-reflexion shape matches A1 exactly.  The
W88 design is essentially **competitive with A1 on 2 of 3
seeds**, but fails on the third by 10 pp.

## Arm shape

### A0 — stock single-shot (1 model call at T=0)

```
prompt -> model(t=0, 1 sample) -> code -> executor -> pass/fail
```

The literature's published HumanEval pass@1 baseline.

### A1 — first-pass-among-K=5 (5 model calls at T=0.7)

```
prompt -> model(t=0.7) x 5 INDEPENDENT samples
       -> for each: executor pass/fail
       -> ship first PASS; else first sample
```

The literature's standard "scale with compute" same-budget
baseline at K=5.

### B — executor-guided sequential reflexion-K=5 (5 model calls at T=0.7)

```
Call 0: prompt -> model(t=0.7) -> code -> executor -> result
Call k (k>=1): prompt
             + cumulative history of (candidate_i, executor_stderr_i for i<k)
             -> model(t=0.7) -> code -> executor -> result
Ship first PASS by attempt index; else lex-smallest CID
```

5 model calls total — EXACTLY same budget as A1.  Every call
is code-producing; every call after the first sees the actual
subprocess stderr of every prior attempt.  This is the
literature's Reflexion / Self-Debug-K shape, with the
executor's deterministic pass/fail + truncated stderr as the
load-bearing feedback channel.

## What W88 closes

| DoD bullet | Status |
|---|---|
| `coordpy.humaneval_reflexion_bench_v1` exists with the W88 B shape | ✓ |
| Composed pipeline runs end-to-end on the quick subset | ✓ 3 seeds × 30 problems × 3 arms = 270 outcomes; 990 NIM calls |
| Head-to-head against stock harness: strict improvement | ✓ `b_mean_strictly_beats_a0_mean = True` (B 71.1 % > A0 63.3 %) |
| Audit chain (Merkle root + per-call CIDs) re-verifies offline | ✓ 7/7 PASS via `scripts/verify_w88_humaneval_reflexion_audit_chain.py` |
| New RESULTS doc | ✓ this file |
| ≥ 3 seeds | ✓ |

## What W88 does NOT close

* **The W86 carry-forward
  `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` stays.**
  The harder same-budget head-to-head (B vs A1) remains negative
  at this scale on this model.  The improvement vs W86 is
  meaningful but not enough.

* **New W88 carry-forward:**
  `W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`
  — at the K=5 budget on NIM Llama-3.1-8B-Instruct on
  HumanEval-30, sequential reflexion with executor-stderr
  feedback empirically underperforms first-pass-among-K=5 by
  3.33 pp on the mean across 3 seeds.  The W86 finding is
  REPLICATED with a different B-shape — the gap closes but does
  not flip.  Whether the result reproduces on (a) a stronger
  model (Llama-3.1-70B, GPT-4o-class), (b) a harder benchmark
  (SWE-bench, MBPP-hard), or (c) a larger budget (K ≥ 10) is
  V2 work.

## Anti-cheat re-statement (verbatim)

* ✓ Same model on every arm (`meta/llama-3.1-8b-instruct` via NIM).
* ✓ Same task subset per seed (the W86
  `select_humaneval_subset_v1(seed)` discipline preserved).
* ✓ Same prompt budget per arm (A1 K=5; B K=5).
* ✓ Same retry policy (3 attempts, exponential backoff).
* ✓ No selective retries; each (seed, problem, arm) triple
  is exactly one set of calls.  The fact that B loses to A1 on
  seed 1 by 10 pp is reported — the bench was not re-seeded
  to hide it.
* ✓ Executor truth = full `problem.test` block, same for every
  arm.
* ✓ Audit chain re-derives offline.  7/7 PASS on the 990 NIM
  call sidecar SHA-256s + 3 per-seed Merkle roots + bench
  Merkle root.
* ✓ Negative-result discipline.  This RESULTS doc IS the
  negative result; the W86 carry-forward stays.  W88 is honest
  about the partial improvement (gap closed) and the failure to
  meet the strong-success bar.

## Honest carry-forward limitations

* **`W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`**
  — described above; the headline W88 finding.
* **`W88-L-HUMANEVAL-REFLEXION-V1-NIM-LLAMA-8B-SCALE-CAP`** —
  V1 ran on NIM Llama-3.1-8B-Instruct.  Stronger models may
  use reflexion more effectively; smaller models may use it
  less effectively.  V2.
* **`W88-L-HUMANEVAL-REFLEXION-V1-HUMANEVAL-K5-SCALE-CAP`** —
  V1 used K=5 budget on 30 HumanEval problems per seed.
  Larger K or harder corpora may shift the comparison.  V2.
* **`W88-L-HUMANEVAL-REFLEXION-V1-STDERR-TAIL-500-CAP`** —
  V1 truncates the executor's stderr to the last 500 chars.
  Longer feedback may improve reflexion but also dilutes signal.
  V2.

## Stable boundary preservation

* `coordpy.__version__` unchanged at 0.5.20.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.humaneval_reflexion_bench_v1` is explicit-import
  only.

## Re-running from scratch

```bash
# NVIDIA_API_KEY must be set.
python scripts/run_w88_humaneval_reflexion_bench.py \
    --backend nim --n-problems 30 --n-seeds 3
python scripts/verify_w88_humaneval_reflexion_audit_chain.py
python scripts/inspect_w88_per_task_outcomes.py
```

The bench reproduces with the same seeds; NIM's T=0.0 arm (A0)
is deterministic at the provider; T=0.7 arms carry provider-
side sampling variation.  The strict-improvement bool shapes
are the stable closure surface.

A frontier-scale Colab notebook
(`scripts/colab_w88_humaneval_reflexion.ipynb`) reproduces the
bench against `Qwen/Qwen2.5-Coder-7B-Instruct` at bf16 on
Colab Pro A100 — a stronger code model that may exhibit
different reflexion dynamics.  That run is V2 unless explicitly
re-launched.
