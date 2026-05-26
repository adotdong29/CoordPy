# W103 — Helper-consumption attestation V1 (COO-14 downstream)

> **2026-05-25.  First real downstream consumption of
> `coordpy.code_slice_selector_v1` in a NIM-spending milestone.
> The W103 HumanEval+ cheap-pilot slice is helper-driven, not
> a parallel deterministic seed shuffle.  This doc is the
> attestation the pilot driver and result docs reference.**

## Why helper consumption matters

The W102 helper lane delivered `coordpy.code_slice_selector_v1`
with a worked example on the W101 arsenal-mining report
(humaneval composite 0.9377; mbpp composite 0.4907).  W102 did
NOT consume the helper as a load-bearing pilot input — the
W102 MBPP+ V2 cheap pilot used the same deterministic
`seed=101_001` shuffle as W101 V1.  W103 closes that loop:
the HumanEval+ cheap pilot's 30-problem slice is constructed
BY the helper, not in parallel to it.

This is the COO-14 Definition-of-Done item 4 ("Feed that output
back into runbooks before expensive runs are approved")
realised as a NIM-pilot earning constraint.

## Inputs

| Field | Value |
|---|---|
| Helper module | `coordpy.code_slice_selector_v1` (schema `coordpy.code_slice_selector_v1.v1`) |
| Driver | `scripts/run_w102_code_slice_proposal.py` |
| Mining report path | `results/w102/arsenal_mining/w102_arsenal_20260526T000910Z/mining_report.json` |
| Mining report bench blocks | `humaneval`, `mbpp`, `humaneval_plus`, `mbpp_plus_v2` |
| Slice-proposal output dir | `results/w103/code_slice_proposals/w102_slice_proposals_20260526T015420Z/` |
| HumanEval+ proposal CID (primary) | `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327` |
| HumanEval (base) proposal CID (top-up source) | `b7325b9646009a4a3fd71442cc55d3fd7c72a44690f6b6878ee5fb6d9ffcf607` |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| Final W103 slice CID | **`c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2`** |

## Helper-anchored slice rule (pre-locked in `docs/RUNBOOK_W103.md`)

1. Take the helper's `humaneval_plus` proposal entries in
   priority order (b_only_wins → shared_fails → a1_only_wins
   → shared_wins).
2. De-duplicate on `task_id` (some task_ids appear in multiple
   historical seeds; the same HumanEval+ task_id is the same
   problem regardless of which historical seed surfaced it).
3. If fewer than 30 unique task_ids result, top up from the
   helper's BASE `humaneval` proposal (same composite-priority
   order) using task_ids NOT already in the set.
4. Cap at 30 unique task_ids.
5. Verify each is present in the SHA-pinned HumanEval+ corpus.
6. Compute slice CID =
   `SHA-256(",".join(final_task_ids).encode("utf-8"))`.

## Final 30-problem slice (W103 lead lane)

| # | task_id | Source cluster |
|---|---|---|
| 1 | HumanEval/118 | humaneval_plus : b_only_wins |
| 2 | HumanEval/16 | humaneval_plus : b_only_wins |
| 3 | HumanEval/160 | humaneval_plus : b_only_wins |
| 4 | HumanEval/163 | humaneval_plus : b_only_wins |
| 5 | HumanEval/121 | humaneval_plus : b_only_wins |
| 6 | HumanEval/125 | humaneval_plus : b_only_wins |
| 7 | HumanEval/84 | humaneval_plus : b_only_wins |
| 8 | HumanEval/129 | humaneval_plus : shared_fails |
| 9 | HumanEval/76 | humaneval_plus : shared_fails |
| 10 | HumanEval/91 | humaneval_plus : shared_fails |
| 11 | HumanEval/132 | humaneval_plus : shared_fails |
| 12 | HumanEval/137 | humaneval_plus : shared_fails |
| 13 | HumanEval/140 | humaneval_plus : shared_fails |
| 14 | HumanEval/154 | humaneval_plus : shared_fails |
| 15 | HumanEval/32 | humaneval_plus : shared_fails |
| 16 | HumanEval/55 | humaneval_plus : shared_fails |
| 17 | HumanEval/83 | humaneval_plus : shared_fails |
| 18 | HumanEval/17 | humaneval_plus : a1_only_wins |
| 19 | HumanEval/122 | humaneval_plus : a1_only_wins |
| 20 | HumanEval/100 | humaneval_plus : shared_wins |
| 21 | HumanEval/101 | humaneval_plus : shared_wins |
| 22 | HumanEval/104 | humaneval_plus : shared_wins |
| 23 | HumanEval/111 | humaneval_plus : shared_wins |
| 24 | HumanEval/113 | humaneval_plus : shared_wins |
| 25 | HumanEval/119 | humaneval_plus : shared_wins |
| 26 | HumanEval/14 | humaneval_plus : shared_wins |
| 27 | HumanEval/35 | humaneval_plus : shared_wins |
| 28 | HumanEval/44 | humaneval_plus : shared_wins |
| 29 | HumanEval/49 | humaneval (top-up) : shared_wins |
| 30 | HumanEval/61 | humaneval (top-up) : shared_wins |

## Cluster mix

| Cluster source | Count | % of slice |
|---|---:|---:|
| humaneval_plus : b_only_wins | 7 | 23.3 % |
| humaneval_plus : shared_fails | 10 | 33.3 % |
| humaneval_plus : a1_only_wins | 2 | 6.7 % |
| humaneval_plus : shared_wins | 9 | 30.0 % |
| humaneval (top-up) : shared_wins | 2 | 6.7 % |
| **Total** | **30** | **100.0 %** |

19 of 30 problems (63.3 %) are NOT `shared_wins` — i.e., the
slice is concentrated on the historical rescue / stress
surface, not on the easy cluster.  A parallel deterministic
shuffle (e.g., `seed=103_001` over the 164-row corpus) would
land roughly 27 of 30 on the easy cluster at the predicted
W88 70B HumanEval A1 = 85.56 % saturation rate, and the
HumanEval+ A1 = 72.86 % residual on the full corpus would
leave only ~ 8 / 30 in the unique-failure clusters — half as
much rescue surface to test.

## Ranking context

| Rank | Bench | rescue_fraction | hard_cluster_size | mean_B − A1 pp | per_seed_std_pp | composite_score |
|---|---|---|---:|---:|---:|---:|
| 1 | mbpp_plus_v2 | 6.60 % | 22 | +5.28 | 4.44 | 1.0028 |
| 2 | **humaneval_plus** | **9.21 %** | **12** | **+5.56** | **4.16** | **0.9429** |
| 3 | humaneval | 9.76 % | 5 | +5.56 | 6.85 | 0.9377 |
| 4 | mbpp | 3.97 % | 21 | +1.33 | 3.40 | 0.4907 |

mbpp_plus_v2 has the highest composite score because the
historical hard-cluster-size component dominates; W102's
fresh-K=5 cheap pilot at seed 101_001 demonstrated that this
composite is an UPPER BOUND on what the mechanism can produce
on the new surface (W102 fresh-K=5 swing was -11.95 pp below
the +5.28 pp arsenal-mining prior).  humaneval_plus ranks #2
on composite score but #2 also on absolute fresh-K=5 risk:
the W89 retirement on the SAME bench family (base HumanEval)
at +5.56 pp is the closest empirical precedent.  HumanEval+
is the right next attack per the W101 + W102 + W103 chain.

## Anti-cheat / honest-framing notes

* The slice is computed BEFORE any NIM call.  The slice CID
  is recorded in the runbook + this doc + the pilot driver's
  output JSON.
* The slice is NOT chosen to maximise expected B − A1.  It is
  chosen to maximise the per-problem RESCUE-SURFACE coverage
  per the helper's locked composite-priority.  This is a
  faithful realisation of the COO-14 charter.
* The slice contains 19 / 30 historically-hard problems and
  11 / 30 easy-cluster top-ups.  The easy-cluster top-ups are
  necessary to (a) prevent A1@K=5 from collapsing too far
  below the saturation gate and (b) keep the 30-problem
  per-arm count fixed (W89 / W91 / W101 / W102 cheap-pilot
  convention).
* The slice is helper-anchored, NOT random-seed-anchored.  The
  W101 / W102 deterministic `seed=101_001` shuffle is NOT
  used.  The W103 pilot uses `--seed 103001` for the candidate
  sampling RNG only (not for slice selection); this preserves
  audit-chain isolation from the W88 / W89 / W102 namespaces.

## Anti-pattern guard cross-check

* Helper module name does NOT contain any forbidden token
  (`bounded_window`, `compaction`, `context_compaction`,
  `prose_summary`, `context_pruning`, `summarizer`).
* `coordpy.humaneval_plus_reflexion_bench_v1` (the bench
  module the W103 pilot consumes) does NOT contain any
  forbidden token (confirmed by W102 preflight probe
  `AddrW102-Hplus-AntiPattern`).

## Anchors

* `docs/RUNBOOK_W103.md` — pre-commit contract.
* `coordpy/code_slice_selector_v1.py` — helper module.
* `scripts/run_w102_code_slice_proposal.py` — driver.
* `scripts/run_w103_humaneval_plus_pilot.py` — pilot driver
  consumer.
* `results/w103/code_slice_proposals/w102_slice_proposals_20260526T015420Z/proposals.json` —
  raw helper output.
* `results/w102/arsenal_mining/w102_arsenal_20260526T000910Z/mining_report.json` —
  arsenal-mining input.
