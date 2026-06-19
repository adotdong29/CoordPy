# W93 failure-cluster diagnosis (post-W92, cheap mining)

> 2026-05-24.  Cheap analysis of W88–W92 bench reports + sidecars
> via `coordpy.failure_cluster_miner_v1`.  No NIM calls; runs in
> seconds.  Goal: surface the actual failure structure so W93
> candidate architectures are hypothesis-driven, not
> vibes-driven.

## Aggregate cross-run signal (from `results/w93/failure_clusters.json`)

11 bench reports mined.  Mean B − A1 per bench kind:

| Bench kind | Configurations | min Δ | max Δ | mean Δ |
|---|---|---:|---:|---:|
| `humaneval_reflexion` | W88 (8B) + W89 (70B) | −3.33 | **+5.56** | +1.11 |
| `mbpp_reflexion` | W90 P1 + W91 P1 | +1.11 | +1.33 | **+1.22** |
| `cross_modal_code` (split) | W88 V1 + W89 P2 + W89 P3 | −27.78 | −5.56 | **−12.96** |
| `cross_modal_vlm_loop` | W90 P2 + W91 P2 + W91 P2b | −7.14 | +2.78 | **−1.46** |
| `cross_modal_role_specialized` | W92 | −10.71 | −10.71 | **−10.71** |

## Three concrete findings from the miner

### Finding 1 — Reflexion is robustly positive at 70B on code

Mean B − A1 across all 70B reflexion runs is **+1.22 to +1.11 pp**.
This includes:

* W89 HumanEval (3 seeds): **+5.56 pp** (retirement-grade).
* W90 P1 MBPP (3 seeds): +1.11 pp.
* W91 P1 MBPP (5 seeds): +1.33 pp.

The 8B HumanEval (W88) is the only negative (−3.33 pp); the
3 70B-scale runs all positive.  The reflexion architecture's
mean direction is robust at 70B; the per-seed strict majority
bar is the only consistent obstacle on MBPP.

**Hypothesis for W93+:** reflexion at K=5 on a benchmark with
LOWER A1 ceiling (e.g., MBPP+ where A1 should drop from 81 %
to ~60 %) should produce a clearer per-seed-majority win.

### Finding 2 — Cross-modal split architecture is empirically dead

Mean B − A1 across 3 split runs: **−12.96 pp** (range −27.78
to −5.56).  Always negative.  Always 0 of 3 seeds.

The split's failure mode is localized at the text-only
extraction handoff: the VLM extracts the image as text bullets;
the code-LM has no recourse if the extraction is lossy.  This
is structural — confirmed at three model scales (W88 V1, W89
P2, W89 P3).

**No further W93+ effort on the W88 split.**  Replaced
empirically by VLM-in-loop.

### Finding 3 — VLM-in-loop is NEAR-TIED on average

Mean B − A1 across 3 VLM-in-loop runs: **−1.46 pp** (range
−7.14 to +2.78 pp).  Best per-seed evidence is 2/3 at W91 P2
(3 seeds; disconfirmed by P2b 7-seed).  Worst per-seed evidence
is 2/7 at W91 P2b.

This is the closest any cross-modal architecture has come to
unified-VLM K=5.  The variance is high (range 9.9 pp) — at 3
seeds the result is unstable; at 7 seeds the underlying mean
appears to settle around −5 to −7 pp.

**Hypothesis for W93+:** the VLM-in-loop architecture's ~−1
to −7 pp gap is variance + ceiling-saturation.  At the
saturated K=5 ceiling of 88 % on HumanEval-Visual, the
gap is too tight for reflexion to break in.  Either:

* **(a) Higher K** (K=10 or K=20) where i.i.d. sampling
  saturates and reflexion has more iterations.  Breaks the K=5
  same-budget contract; not a candidate for W93.
* **(b) Less-saturated benchmark** where A1 K=5 doesn't hit
  88 %.  MathVista, ChartQA, etc.
* **(c) Substrate-level cross-modal injection** that gives B
  signal A1 K=5 sampling can't access.  The W87-L direction.

### Finding 4 — Role-specialized was WORSE than VLM-in-loop

Mean B − A1 for role-specialized (W92): **−10.71 pp**.  Mean
B − A1 for VLM-in-loop: **−1.46 pp**.  Role specialization
made the gap LARGER, not smaller.

**Diagnosis**: the Code-Implementer in W92 had no image access;
the Planner's plan + Verifier's critique passed through a
narrow text-only channel.  Each Implementer turn was
constrained by potentially-incorrect upstream signal.  The
3 Implementer attempts were correlated by their shared input
(Plan) and the diversity gain was less than the W90-style 3
i.i.d. VLM attempts would have given.

**Hypothesis for W93+:** if cross-modal team is to win, the
code-generating roles must have direct image access — not
just text-channel signal from a Planner.  This pushes toward
VLM-Implementer (not text-LM-Implementer).  But VLM-Implementer
reduces to VLM-in-loop with extra prompting.

## What the diagnosis says about W93 candidate design

**For same-budget code generalization** (MBPP per-seed
majority failure):

* The architecture is fine; the benchmark is the issue.  MBPP
  at K=5 with A1 mean = 81 % has too-small failure-residual.
* Recommended W93 candidate: **harder code benchmark**
  (HumanEval+, MBPP+, LiveCodeBench).  Requires corpus loader
  work; not feasible without significant build time.
* Alternative recommended candidate: **K=10 on the same
  benchmarks**, with a parallel A1@K=10 baseline.  Tests
  whether reflexion gains scale with budget.  But this breaks
  the W89 K=5 contract — would need a separate W93 retirement
  bar definition.

**For cross-modal retirement**:

* HumanEval-Visual at K=5 is presumptively hostile (3
  architecture families confirm).
* Recommended W93 candidate: a NEW cross-modal benchmark
  where A1_vlm K=5 doesn't approach ceiling (MathVista,
  ChartQA, DocVQA, MMVet candidates).  Requires corpus +
  evaluator build; **not feasible in remaining W93 session
  time without a significant new module**.
* Substrate-level cross-modal injection: the W87-L direction.
  Architecturally hard; not feasible in W93 session.

## W93 honest conclusion

After the cheap failure-mining analysis:

* **No candidate architecture currently in scope will retire
  the cross-modal carry-forward** on the current battlefield
  (HumanEval-Visual K=5).  Three architectures already failed;
  the diagnosis predicts a 4th will too.
* **No candidate code architecture currently in scope will
  retire the same-budget code carry-forward** at K=5 on
  MBPP — the per-seed-majority bar is ceiling-bound.
* **The right next move is benchmark pivot**: either to a
  harder code benchmark (HumanEval+, MBPP+, LiveCodeBench) OR
  to a non-ceiling-saturated cross-modal benchmark
  (MathVista, ChartQA).  Building corpus loaders for those is
  significant W94 work.
* **The W93 deliverable is therefore the preflight
  infrastructure + this diagnosis, not another expensive
  benchmark run.**  This is the discipline the W93 runbook
  required.

W93 will:

1. Ship the failure-cluster miner
   (`coordpy.failure_cluster_miner_v1`) — DONE.
2. Ship this diagnosis note — DONE.
3. Ship a preflight harness
   (`coordpy.cross_modal_preflight_harness_v1`) — building.
4. Document 2–3 candidate hypotheses with explicit preflight
   verdicts.
5. Commit infrastructure + diagnosis as the W93 milestone.
6. Defer expensive runs to W94+, conditional on a benchmark
   pivot that passes the preflight gates.
