# W102 — Code-side slice-selection + candidate-ranking helper V1 (COO-14)

> **2026-05-25.  COO-14 Definition of Done delivered in W102.
> The helper takes an arsenal-mining report (per the W101 /
> W102 cross-bench sidecar re-execution) and produces:**
>
> 1. **A ranked candidate-direction table** ordered by
>    composite score over (rescue_fraction, hard_cluster_size,
>    mean_B−A1_pp, per-seed variance).
> 2. **A cheap-pilot slice proposal** per bench, prioritised
>    by (b_only_wins > shared_fails > a1_only_wins >
>    shared_wins), subject to a cheap-pilot budget (default
>    30 problems × 11 calls/problem = 330 NIM calls).
> 3. **Runbook-ready Markdown** for both the ranking and each
>    slice proposal.
> 4. **An anti-pattern guard** that refuses to propose a slice
>    for any bench module whose name contains a forbidden
>    token (`bounded_window`, `compaction`, etc.).
>
> Stand-alone module: no NIM, no expensive bench, no model
> loading.  Unit-tested (15 tests).

## COO-14 Definition of Done — line-by-line

| COO-14 DoD | W102 delivery |
|---|---|
| 1. Use committed W88–W93 evidence to rank candidate directions cheaply. | `rank_candidate_benches` consumes the W101 arsenal-mining report (which itself re-executes W88 70B HumanEval + W91 5-seed 70B MBPP sidecars offline against the canonical executors); the W102 mining extension adds HumanEval+ and MBPP+ V2 cross-bench surfaces.  Output: ranked list of `BenchCandidateRanking` rows. |
| 2. Build helpers for selecting failure-cluster slices for pilots. | `propose_cheap_pilot_slice` reads the per-(seed, task_id) cluster surface and returns the (seed, task_id) tuples ordered by cluster priority. |
| 3. Make it easy to ask: what exact problems should the next cheap pilot attack? | `propose_cheap_pilot_slice(bench, n_problems)` is the explicit API; the returned `SliceProposal` has a per-problem `justification` string ("unique-B-rescue: reflexion mechanism rescued this in the historical bench", etc.). |
| 4. Feed that output back into runbooks before expensive runs are approved. | `format_slice_proposal_markdown(...)` serialises to a runbook-ready Markdown table that W103+ runbooks can `include` verbatim.  `scripts/run_w102_code_slice_proposal.py` produces both `ranking.md` + `slice_<bench>.md` + a combined `proposals.json` for the W103 runbook to cite by CID. |

## Worked example (W101 arsenal-mining report)

Driver: `python scripts/run_w102_code_slice_proposal.py`.
Input: `results/w101/arsenal_mining/w101_arsenal_20260525T231104Z/mining_report.json`.

Output (excerpt):

```
# Code-side candidate-direction ranking

| Rank | Bench    | rescue_fraction | hard_cluster_size | mean_B-A1_pp | per_seed_std_pp | composite_score |
|---   |---       |---              |---                |---           |---              |---              |
| 1    | humaneval| 9.76%           | 5                 | +5.56        | 6.85            | 0.9377          |
| 2    | mbpp     | 3.97%           | 21                | +1.33        | 3.40            | 0.4907          |
```

The ranking is intuitive: humaneval ranks first because (a) its
B-only rescue surface is 2.5× richer than mbpp's (the W101
arsenal-mining structural finding), AND (b) its mean B−A1
margin is already positive at +5.56 pp.  This matches the W101
battlefield-selection LEAD intuition that the EvalPlus-
hardened version of either base bench is worth attacking
(MBPP+ chosen LEAD because the ABSOLUTE residual surface is
larger; HumanEval+ chosen BACKUP because the mean margin is
already at retirement-grade).

### Slice proposal — humaneval (30 problems)

| # | Cluster | Count |
|---|---|---|
| 1-8 | b_only_wins | 8 (all available) |
| 9-13 | shared_fails | 5 (all available) |
| 14-16 | a1_only_wins | 3 (all available) |
| 17-30 | shared_wins | 14 (top-up) |

Total NIM budget: 30 × 11 = 330 calls.

The cluster distribution is exactly the empirical W89 surface
(8 + 3 + 74 + 5 = 90 problem-seeds; helper picks the small
clusters first because they're the highest-information
samples).

### Slice proposal — mbpp (30 problems)

| # | Cluster | Count |
|---|---|---|
| 1-5 | b_only_wins | 5 (all available) |
| 6-26 | shared_fails | 21 (all available) |
| 27-29 | a1_only_wins | 3 (all available) |
| 30 | shared_wins | 1 (top-up) |

Total NIM budget: 30 × 11 = 330 calls.

The mbpp slice is heavily dominated by `shared_fails` (21/30) —
the structural mbpp ceiling-saturation problem that MBPP+
relieves.  This matches the W101 finding that base-MBPP's
hard cluster is 2.5× larger than HumanEval's.  If a future
cheap pilot tries to attack base MBPP again, this slice
proposal would already warn that the mechanism's rescue
surface is small (only 5 b_only_wins to attack on).

## Composite-score formula (locked 2026-05-25)

```python
composite_score = (
    2.0 * rescue_fraction
    + 0.10 * mean_b_minus_a1_pp
    + 0.30 * (hard_cluster_size / n_problems_per_seed)
    + 0.20 * min(per_seed_margin_std_pp / 10.0, 1.0))
```

Rationale:

* **`rescue_fraction`** (weight 2.0): the mechanism-load-
  bearing prior.  This is the single most important signal
  per the W96-C / W100 / W101 MLB sub-gate discipline.
* **`mean_b_minus_a1_pp`** (weight 0.10): rewards already-
  positive margins; small coefficient so it doesn't dominate.
* **`hard_cluster_size / n_problems_per_seed`** (weight 0.30):
  rewards benchmarks with non-trivial hard surface (MBPP-like
  ceiling-saturated benches get penalised here — most of the
  pass is on the easy cluster).
* **`per_seed_margin_std_pp / 10.0`** saturating at 1 (weight
  0.20): rewards SOME per-seed variance (mechanism has
  something to attack) but does not reward extreme variance
  (which would suggest seed-luck rather than mechanism load-
  bearingness).

The composite is documented as ONE possible ordering — the
helper exposes the four input fields explicitly so downstream
consumers can implement custom orderings.  The W102 RUNBOOK
locks this formula; future milestones may extend it but should
add a V2 helper rather than silently change the V1 behavior.

## Anti-pattern guard

`propose_cheap_pilot_slice` accepts an optional
`bench_module_name` parameter; if it contains any token in
`FORBIDDEN_TOKENS = ('bounded_window', 'compaction',
'context_compaction', 'prose_summary', 'context_pruning',
'summarizer')`, the helper raises `ValueError`.  Unit-tested
(`test_propose_cheap_pilot_slice_refuses_anti_pattern`).

This is a structural defense — even if a future caller
accidentally points the helper at a bounded-window baseline
module, the slice proposal will not be produced (and thus no
NIM budget can be earned for that direction).

## What this helper does NOT do

* It does NOT itself re-execute candidate responses.  That
  lives in `scripts/run_w101_arsenal_mining.py` and
  `scripts/run_w102_arsenal_mining.py`.
* It does NOT call NIM.
* It does NOT load any model.
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.
* It does NOT modify `coordpy/__init__.py`.

## Anchors

* `coordpy/code_slice_selector_v1.py` — module.
* `scripts/run_w102_code_slice_proposal.py` — driver.
* `tests/test_w102_code_slice_selector_v1.py` — 15 unit tests.
* `docs/RUNBOOK_W102.md` — pre-commit contract.
* `results/w102/code_slice_proposals/w102_slice_proposals_<RUN>/`
  — first proposal artifact set (ranking.md + slice_<bench>.md +
  proposals.json).
