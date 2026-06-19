# W93 — Preflight-First Empirical Superiority Wave V6

> **2026-05-24 — DISCIPLINE-FIRST MILESTONE.  No expensive NIM
> bench was launched in W93.  All 3 candidate architectures
> were killed in cheap preflight against W88–W92 evidence.  The
> deliverable is the preflight infrastructure
> (`coordpy.failure_cluster_miner_v1` +
> `coordpy.cross_modal_preflight_harness_v1`), the failure-mode
> diagnosis (`docs/W93_FAILURE_DIAGNOSIS.md`), and 3
> documented candidate kills.  This is exactly the discipline
> the W93 runbook required: stop paying full benchmark price
> for weak ideas.**
>
> **Carry-forward retirements**: NONE.  The W89 70B-HumanEval
> retirement remains; all other carry-forwards remain
> unaffected by W93.  W93 contributes the empirical
> infrastructure for cheap candidate evaluation going forward.

## TL;DR

W93 was a deliberate execution-discipline pivot.  W88 → W92
spent ~30 hours of NIM compute and produced 1 retirement +
6 negative results.  The W92 5-hour run cost full price to
discover an architecture (role-specialized) is worse than the
simpler VLM-in-loop — evidence the cheap miner could have
predicted.  W93's deliverable is the iteration infrastructure
to prevent this pattern.

**What W93 ships:**

1. **`coordpy.failure_cluster_miner_v1`** — analyses W88–W92
   bench reports + sidecars locally; no NIM calls; runs in
   seconds.  Discovers 11 bench runs, summarises
   per-bench-kind B − A1 deltas, identifies architecture
   families.
2. **`coordpy.cross_modal_preflight_harness_v1`** — 5
   pre-committed preflight gates (hypothesis written / cheap
   sidecar evidence / adversarial ablation / budget accounting
   / benchmark justification).  A candidate must pass ALL 5
   gates before earning the expensive bench run.
3. **`docs/W93_FAILURE_DIAGNOSIS.md`** — written diagnosis of
   what W88–W92 evidence actually says.
4. **`scripts/run_w93_candidate_preflight.py`** — 3 candidate
   architectures with explicit hypotheses, each run through
   the harness.
5. **`results/w93/{failure_clusters.json,candidate_preflight_verdicts.json}`**
   — full preflight evidence committed alongside the
   infrastructure.
6. **16 new CI tests** (`tests/test_w93_preflight_v1.py`); all
   green.

**What W93 does NOT ship:** any expensive bench run.  No
candidate passed preflight.  Per W93 runbook contract: the
discipline is the milestone.

## Failure-miner summary (no NIM calls)

| Bench kind | n runs | B − A1 mean | Best per-seed | Status |
|---|---:|---:|---|---|
| `humaneval_reflexion` | 2 | +1.11 pp (range −3.33 to +5.56) | W89 70B: 2/3 | **RETIRED at 70B (W89)** |
| `mbpp_reflexion` | 2 | +1.22 pp (range +1.11 to +1.33) | best 2/5 | Stays: per-seed bar fails |
| `cross_modal_code` (split) | 3 | **−12.96 pp** | 0/3 always | Stays: empirically dead |
| `cross_modal_vlm_loop` | 3 | −1.46 pp (range −7.14 to +2.78) | W91 P2 2/3 (3-seed; disconfirmed) | Stays: near-tied; W91 P2b disconfirms |
| `cross_modal_role_specialized` | 1 | **−10.71 pp** | 0/7 | Stays: 3rd dead architecture |

**Cumulative cross-modal pattern**: 7 configurations across 3
architectures all FAIL to beat unified-VLM K=5 on HumanEval-
Visual.  Image-load-bearing PROVEN at 7/7 (B − A0_text always
> +5 pp).  Team-organisation FALSIFIED at 7/7.

## Three W93 candidate kills

All 3 candidates were defined with explicit hypotheses, then
evaluated against W88–W92 sidecars via the 5-gate harness.

### W93-A — Self-Verifying VLM-in-loop (KILLED)

**Hypothesis**: same architecture as W90 P2 / W91 P2b, plus a
structured self-verification token emitted by the VLM on each
turn.  Use self-verification to break ties between candidates
that all pass the executor, preferring the most-confident.

**Gates passed:** G1 (hypothesis), G4 (budget), G5
(benchmark).

**Gates failed:**
- **G2** (sidecar evidence): W91 P2b shows VLM-in-loop at 7
  seeds is −7.14 pp below A1_vlm on the SAME benchmark + same
  config.  No evidence in existing sidecars that
  self-verification token would close this gap.
- **G3** (adversarial ablation): W92 evidence directly shows
  that adding verifier-style turns to VLM-in-loop made it
  WORSE (−10.71 pp), not better.  The "verification" feature
  is empirically harmful at this budget.

**Verdict**: KILLED.  Adding cosmetic verification to a known-
failing architecture is unlikely to flip the sign.

### W93-B — Heterogeneous Pool (KILLED)

**Hypothesis**: 3 VLM samples + 2 code-LM samples (text-only,
conditioned on VLM extraction).  Pool all 5; ship first PASS.
Heterogeneity adds diversity beyond i.i.d. VLM sampling.

**Gates passed:** G1, G4, G5.

**Gates failed:**
- **G2** (sidecar evidence): the W88 split architecture used
  code-LM downstream of VLM extraction.  All 3 W88/W89 split
  runs showed B − A1_vlm = negative (−5.56 to −27.78 pp).
  The marginal code-LM-downstream-of-VLM sample contribution
  is empirically NEGATIVE.  Adding such samples to a pool
  cannot help.
- **G3** (adversarial ablation): same as W93-A — W92 shows
  mixing modalities at K=5 doesn't help.

**Verdict**: KILLED.  The code-LM-downstream-of-VLM sample is
falsified at the per-sample level by the W88 split evidence;
no heterogeneous-pool variant on this benchmark family can
recover.

### W93-C — Reflexion at K=10 budget (KILLED on G2 only)

**Hypothesis**: same architecture as W88/W89 sequential
reflexion, but at K=10 budget on both A1 and B.  At K=10,
i.i.d. sampling saturates (diminishing returns past K=5);
reflexion has 10 iterations to apply executor-feedback-driven
refinement.

**Gates passed:** G1 (hypothesis), G3 (ablation — K=5 reflexion
already wins at 70B on HumanEval), G4 (budget), G5 (benchmark).

**Gates failed:**
- **G2** (sidecar evidence): no K=10 evidence in any W88–W92
  sidecar.  ALL prior runs are K=5.  The hypothesis cannot
  pass cheap preflight without a small NIM pilot run at K=10
  to gather first evidence.

**Verdict**: KILLED in W93 preflight, BUT this is the most
promising next direction.  The architecture is empirically
validated at K=5 (W89 retirement); the ablation says the
"extra K" is load-bearing.  The only thing missing is K=10
pilot evidence.

**Recommended W94 next step**: a SMALL K=10 pilot (e.g.,
1 seed × 10 problems) on HumanEval-70B to gather preflight
evidence for K=10.  Cheap (~1 hour).  If pilot shows
B − A1 > +5 pp at K=10, then full K=10 bench earns the
expensive run.  If pilot shows ≤ +1 pp, the K=10 hypothesis
dies cheaply.

## Anti-cheat

* The failure miner reads ONLY committed W88–W92 evidence.
  No re-running of NIM.
* The preflight harness uses synthetic discriminators
  pre-committed in the harness module.  No "discriminators
  tuned to make my candidate look good".
* All 3 candidates were defined with explicit hypotheses
  BEFORE the gates were evaluated.  The candidate definitions
  are committed in
  `scripts/run_w93_candidate_preflight.py`.
* The preflight verdicts are content-addressed JSON; verdicts
  can be re-derived offline.
* Negative evidence is preserved: 3 candidates documented as
  killed, with the specific gate that killed each.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.failure_cluster_miner_v1` and
  `coordpy.cross_modal_preflight_harness_v1` are
  explicit-import only.

## What W93 retires

Nothing.  No expensive run was launched; no candidate earned it.

## What W93 contributes

* **Discipline shift**: from "build candidate; run full bench;
  discover failure" to "build candidate; preflight against
  cheap discriminators; kill or earn the expensive run".
* **Infrastructure**: 2 new modules + 16 CI tests + verifiable
  preflight verdicts.
* **Diagnosis**: written analysis of what 11 W88–W92 bench
  reports collectively say.
* **3 candidate kills, documented**: each kill cites the exact
  gate that failed and the W88–W92 evidence that supports
  the kill.
* **W94 recommendation**: pilot K=10 reflexion on HumanEval-70B
  (small + cheap), use as preflight evidence for a full K=10
  bench.

## The honest claim W93 earns

**W93 produces the cheapest possible cross-architecture
empirical analysis of W88–W92 evidence + a 5-gate preflight
discipline that prevents future expensive runs on weak
candidates.  Three candidate architectures (self-verifying
VLM-in-loop, heterogeneous pool, K=10 reflexion) were killed
in seconds-long preflight against committed W88–W92 sidecars.
The cumulative cross-modal evidence (now 7 configurations) is
decisively negative; HumanEval-Visual K=5 is empirically the
wrong battlefield.  W94+ work must pivot benchmarks or move
to substrate-level cross-modal injection.  W93's preflight
infrastructure is the new minimum bar for any future expensive
bench run.**

## Recommended W94 plan (NOT executed in W93)

1. Run a small K=10 reflexion pilot on HumanEval-70B
   (1 seed × 10–15 problems, ~30 min wall).  Gathers preflight
   evidence for the K=10 hypothesis.
2. If pilot shows B − A1 ≥ +5 pp at K=10:
   * The K=10 hypothesis is empirically supported.
   * W93-C candidate then passes G2 (sidecar evidence).
   * Launch full K=10 HumanEval bench (3 seeds × 30 problems,
     same as W89 shape).  Retirement decision per W89's bars.
3. If pilot shows B − A1 < +1 pp at K=10:
   * The K=10 hypothesis dies cheaply.
   * Pivot to a new benchmark (MathVista / MBPP+ /
     LiveCodeBench) or substrate-level cross-modal injection.
