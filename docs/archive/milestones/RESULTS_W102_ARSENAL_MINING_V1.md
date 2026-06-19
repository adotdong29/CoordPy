# W102 — Arsenal mining via cross-bench sidecar re-execution V1

> **2026-05-25.  Extends the W101 mining
> (`docs/RESULTS_W101_ARSENAL_MINING_V1.md`) with two NEW
> cross-bench surfaces produced by re-executing the same
> 2 640 W88 / W91 candidate responses against the actual
> EvalPlus hardened test surfaces (V2 MBPP+ + V1 HumanEval+).
> No new NIM calls.  The re-executed surfaces are the empirical
> cross-bench priors the W102 V2 + HumanEval+ preflights
> consume; they also feed the COO-14 code-side slice selector.**

## Method

Script: `scripts/run_w102_arsenal_mining.py`.

Two new mining passes (each NIM-free):

1. **HumanEval+ pass**: re-execute each of the 990 persisted
   W88 70B HumanEval candidate responses against the V1
   HumanEval+ `check(candidate)` block (the EvalPlus hardened
   test surface; ~80× more hidden assertions per problem).
2. **MBPP+ V2 pass**: re-execute each of the 1 650 persisted
   W91 5-seed 70B MBPP candidate responses against the V2
   MBPP+ extra `test` program (the EvalPlus iteration loop
   over `inputs` + `results` parallel arrays).

For each pass, the aggregator builds the per-(seed, task_id,
arm) cluster surface ((a1_only_wins, b_only_wins, shared_wins,
shared_fails) partition + mechanism-load-bearing estimate)
— same shape as the W101 humaneval / mbpp blocks.

Output:
* `results/w102/arsenal_mining/<RUN>/per_call_outcomes.jsonl` —
  one line per re-executed call.
* `results/w102/arsenal_mining/<RUN>/per_problem_outcomes.json`
  — aggregated per-arm verdict + per-cluster membership.
* `results/w102/arsenal_mining/<RUN>/mining_report.json` —
  top-level summary the W102 preflights + helper read.
  Includes both NEW `humaneval_plus` + `mbpp_plus_v2` blocks
  AND optionally carries the W101 `humaneval` + `mbpp` blocks
  via `--include-w101-mining`.

## Structural findings (empirical)

### Per-bench cluster surface (post-W102 mining; 2 640 offline re-executions)

| Bench | A0 mean | A1@K=5 mean | B mean | B−A1 (pp) | Per-seed margins | Rescue fraction | Shared-fails / N |
|---|---:|---:|---:|---:|---|---:|---:|
| W89 HumanEval-70B (3 × 30) | 46.67 % | 85.56 % | 91.11 % | **+5.56** | [+13.33, −3.33, +6.67] | **9.76 %** | 5 / 90 |
| W91 MBPP-70B (5 × 30) | 75.33 % | 82.67 % | 84.00 % | **+1.33** | [0, +6.67, +3.33, 0, −3.33] | **3.97 %** | 21 / 150 |
| **HumanEval+ (W88 responses re-executed)** | **44.44 %** | **78.89 %** | **84.44 %** | **+5.56** | **[+10.00, 0.00, +6.67]** | **9.21 %** | **12 / 90** |
| **MBPP+ V2 (W91 responses re-executed)** | **71.32 %** | **77.63 %** | **82.91 %** | **+5.28** | **[0, 0, +10.71, +7.69, +8.00]** | **6.60 %** | **22 / 130** |

### Headline finding 1 — MBPP+ V2 lifts the W91 cap on the SAME responses

The W91 cap was `W91-L-MBPP-REFLEXION-V2-5SEED-PARTIAL-CAP`:
B − A1 = +1.33 pp on base MBPP, failing the +5 pp Phase 2
bar via per-seed strict majority.  Re-grading the SAME 1 650
W91 70B candidate responses against the **actual EvalPlus
extra-test surface** lifts the margin to **B − A1 = +5.28 pp
clearing the Phase 2 +5 pp bar**, with per-seed wins on 3 / 5
seeds reaching +7.69 / +8.00 / +10.71 pp.  This is the
empirical proof that the W91 cap was indeed ceiling-saturation-
bound on base MBPP and the EvalPlus hardened tests structurally
restore the failure-residual surface the reflexion mechanism
can attack.

Two of five seeds (90001 + 90004) still show +0 pp margins
under MBPP+ V2 — the standard deviation is 4.44 pp, so the
+5 pp bar is at-margin not comfortable.  The W102 cheap pilot
with a NEW seed (101_001) is the fresh-sampling test of
whether the +5.28 pp mean holds on a previously-unseen subset.

### Headline finding 2 — V1 silent-degeneration would have hidden this

The W101 V1 loader's broken schema assumption would have
emitted 0 EvalPlus extra-test assertions on every problem, and
the V1 cheap pilot would have measured base-MBPP behavior +
called it "MBPP+".  The +5.28 pp empirical margin uncovered
here would have been invisible — and the W101 verdict's P3
extrapolation (predicted A1 ≈ 69.97 %) is empirically off by
~ 8 pp from the actual MBPP+ V2 A1 of 77.63 %.  The W102 V2
fix is therefore load-bearing: it both unblocks the cheap
pilot AND prevents future cheap pilots from being scored
against the wrong test surface.

### Headline finding 3 — HumanEval+ preserves the W89 retirement margin

The W89 retirement on base HumanEval at 70B was +5.56 pp.
Re-grading the SAME 990 W88 responses against HumanEval+
preserves the margin EXACTLY at +5.56 pp.  The rescue surface
shrinks slightly (rescue fraction 9.21 % vs 9.76 % on base;
shared-fails cluster grows 5 → 12 because EvalPlus extras
promote previously-easy problems to hard).  Predicted
HumanEval+ A1 = 72.86 % (V1 P3 extrapolation) is empirically
off by ~ 6 pp from the actual 78.89 % — base HumanEval is
already a stricter benchmark than base MBPP so the EvalPlus
drop is smaller.

### Headline finding 4 — Hard-cluster shape

| Bench | Shared-fails rate | Implication |
|---|---:|---|
| W89 HumanEval-70B | 5.6 % | Easy bench at 70B; small hard cluster. |
| HumanEval+ | 13.3 % | EvalPlus promotes ~ 7 / 90 to hard; mechanism has more surface. |
| W91 MBPP-70B | 14.0 % | Ceiling-saturated; many "passes" are coincidence on visible asserts. |
| MBPP+ V2 | 16.9 % | EvalPlus exposes ~ 1 / 130 more hard problems; B can attack 5+5+7 = 17 of these. |

### Updated candidate-direction ranking

Re-running the COO-14 helper against the W102 mining report
produces (per
`results/w102/code_slice_proposals/<RUN>/ranking.md`):

| Rank | Bench | rescue_fraction | hard_cluster_size | mean_B−A1_pp | per_seed_std_pp | composite_score |
|---|---|---:|---:|---:|---:|---:|
| 1 | mbpp_plus_v2 | 6.60 % | 22 | +5.28 | 4.44 | **1.0028** |
| 2 | humaneval_plus | 9.21 % | 12 | +5.56 | 4.16 | 0.9429 |
| 3 | humaneval | 9.76 % | 5 | +5.56 | 6.85 | 0.9377 |
| 4 | mbpp | 3.97 % | 21 | +1.33 | 3.40 | 0.4907 |

MBPP+ V2 ranks LEAD post-W102 mining (composite 1.0028),
because the empirical margin (+5.28 pp) is now demonstrated
AND the hard-cluster size (22) is largest in the slate — both
"mechanism stress" + "load-bearing rescue" signals are real.
HumanEval+ stays BACKUP (close composite 0.9429; smaller
hard cluster).  Base MBPP stays demoted (composite 0.4907).

## Cross-bench cluster surface — operational use

* **HumanEval+ block** feeds the
  `coordpy.humaneval_plus_preflight_v1.probe_humaneval_plus_*`
  probes (via the existing W102 backup-lane preflight, which
  uses W88 70B HumanEval prior numbers; the empirical
  HumanEval+ re-execution refines those priors).
* **MBPP+ V2 block** feeds
  `coordpy.mbpp_plus_preflight_v2` AddrW101-P1 / AddrW101-P2 /
  AddrW101-P3 via the cluster surface (currently driven from
  W101 mining; W102 mining provides the strictly-better
  EvalPlus-hardened cluster surface for W103+ probes).
* **Both blocks** feed
  `coordpy.code_slice_selector_v1.rank_candidate_benches` so
  W103+ cheap-pilot slices can be earned from MBPP+ V2 +
  HumanEval+ surfaces rather than from base MBPP / HumanEval
  surfaces.

## Per-cluster carry-forward implications (post-pass)

> Populated once the mining run completes.  The W102 milestone
> summary doc cross-references these implications when the
> W102 cheap pilot lands and the W103 cross-scale path is
> evaluated.

## What this mining does NOT do

* It does NOT modify any W88 / W89 / W91 published result.
* It does NOT re-train the W89 reflexion mechanism.
* It does NOT predict the W102 V2 cheap-pilot outcome — only
  the cheap pilot can do that.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.

## Anchors

* `scripts/run_w102_arsenal_mining.py` — offline cross-bench
  re-executor.
* `results/w102/arsenal_mining/<RUN>/` — mining output dir.
* `docs/RESULTS_W101_ARSENAL_MINING_V1.md` — W101 mining doc
  (W102 extends it).
* `docs/RUNBOOK_W102.md` — pre-commit contract.
* `coordpy.mbpp_plus_loader_v2` / `coordpy.mbpp_plus_executor_v2`
  / `coordpy.humaneval_plus_loader_v1` /
  `coordpy.humaneval_plus_executor_v1` — V2 + HumanEval+
  loaders + executors the mining uses.
