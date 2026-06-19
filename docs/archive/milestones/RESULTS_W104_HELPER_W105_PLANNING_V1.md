# W104 — W105 planning artifact V1 (helper + fallback)

> **2026-05-26.  Pre-builds the W105 next step under BOTH the
> W104 PASS branch (Branch A — Phase 3 retirement bench) and
> the W104 FAIL branch (Branch C — code-line ranking refresh
> with explicit triage).  Built BEFORE the W104 cross-scale
> pilot launches so that EITHER outcome leaves W105 as
> execution, not paperwork.**

## Why pre-build W105 now

The W93 – W103 discipline thread says: pre-commit milestone
boundaries BEFORE empirical evidence arrives.  W103 PASSed
strongly (+20 pp at 70B) but a cross-scale Phase 2 PASS is the
hardest empirical claim the programme has yet attempted on a
code benchmark, and cross-modal cross-scale precedents at
W96-A / W96-C / W100 showed -5 to -10 pp margin shifts.  The
W104 cross-scale Phase 2 pilot could come back PASS, PASS-
non-mechanism, or FAIL.

If W105 setup is left until AFTER the W104 verdict, momentum
between milestones drops and the boundary drifts on outcome.
This artifact eliminates that drift:

* If W104 PASSes Branch A: launch W105 Phase 3 retirement
  bench (3 seeds × 100 problems × K=5 × 2 scales = 6 600 NIM
  calls) using the **slice pack** below — execution-ready.
* If W104 fails Branch B or C: apply the **fallback ranking
  refresh** table below — execution-ready.

This is the COO-14 helper at its second real downstream
consumption.  W103 made `coordpy.code_slice_selector_v1` load-
bearing in a NIM-spending cheap pilot; W104 makes it load-
bearing in the W105 Phase 3 slice-pack construction.

## Branch A — W105 Phase 3 retirement-bench slice pack (pre-built)

The Phase 3 slice pack is constructed deterministically and
pre-committed BEFORE the W104 pilot launches.

### Artifact

* Generator: `scripts/run_w104_w105_phase3_slice_pack.py`.
* Helper consumed: `coordpy.code_slice_selector_v1` (second
  load-bearing downstream consumption; first was W103).
* Output:
  `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/`.
* `slice_pack.json` and `slice_pack.md` are the runbook-
  ready forms.

### Slice rule (deterministic; locked)

1. Inner kernel = W103 helper-anchored 30-problem slice
   (helper-priority order; BYTE-FOR-BYTE from
   `results/w103/humaneval_plus_pilot/<RUN>/provenance.json`).
2. Mid-shell extension = helper proposal of 140 humaneval_plus
   entries (oversize ask); de-duplicated on `task_id` against
   the inner kernel.
3. Outer top-up = helper proposal of 140 base humaneval
   entries; de-duplicated against everything seen so far.
4. Corpus-fill = walk the SHA-pinned HumanEval+ corpus in
   natural-integer-suffix order on `HumanEval/n` task_ids;
   add unseen task_ids until n = 100.
5. The final 100-tuple is the W105 Phase 3 slice (helper-
   priority order); slice CID = SHA-256(",".join(task_ids)).

### W105 Phase 3 slice pack — locked values

| Field | Value |
|---|---|
| Schema | `coordpy.w104_w105_phase3_slice_pack.v1` |
| Pack CID | `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314` |
| n_problems | 100 |
| Phase 3 seeds | (105 001, 105 002, 105 003) |
| Scales locked | `meta/llama-3.3-70b-instruct`, `meta/llama-3.1-405b-instruct` |
| Per-scale per-seed NIM budget | 1 100 calls (100 problems × 11 calls/problem) |
| Per-scale total Phase 3 budget | 3 300 calls (× 3 seeds) |
| Total Phase 3 budget (two scales) | 6 600 calls |
| K_multi_sample | 5 |
| Inner kernel CID (W103 helper-priority) | `c35155956ece605c0169b0cf35a6b69267bee04f5f68cf5a5de466dcc01dd8d2` |
| Mid-shell helper proposal CID | `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327` (humaneval_plus; W103 reused) |
| Outer top-up helper proposal CID | `b7325b9646009a4a3fd71442cc55d3fd7c72a44690f6b6878ee5fb6d9ffcf607` (humaneval; W103 reused) |

### Cluster mix (W105 Phase 3 slice)

| Source | Count | % of slice |
|---|---:|---:|
| `humaneval_plus:b_only_wins` (inner kernel) | 7 | 7.0 % |
| `humaneval_plus:shared_fails` (inner kernel) | 10 | 10.0 % |
| `humaneval_plus:a1_only_wins` (inner kernel) | 2 | 2.0 % |
| `humaneval_plus:shared_wins` (inner kernel) | 9 | 9.0 % |
| `humaneval(top-up):shared_wins` (inner kernel) | 2 | 2.0 % |
| `humaneval_plus:shared_wins:mid_shell` | 45 | 45.0 % |
| `humaneval_plus_corpus:corpus_fill` | 25 | 25.0 % |
| **Total** | **100** | **100.0 %** |

The Phase 3 slice is **30 % rescue-surface concentrated**
(inner kernel 30/100 problems carrying the b_only_wins +
shared_fails + a1_only_wins clusters from the arsenal-mining
report) **+ 25 % broad corpus coverage** (corpus-fill ensures
the slice is not arbitrarily skewed toward the helper's view
of the failure surface).  The remaining 45 % is helper-
selected shared_wins extension; this provides per-problem
margin coverage at the "easy" cluster (necessary for A1 not
to drop too far below the saturation gate).

### W105 Phase 3 retirement bars (locked carry-forward from W88 / W89 / W95)

1. **Per-problem majority bar**: B ≥ A1 on ≥ 16/30 problems
   averaged across seeds × scales.
2. **Per-seed majority bar**: B > A1 on ≥ 2/3 seeds × scales.
3. **Margin bar**: mean B − A1 ≥ +5 pp across seeds × scales.
4. **A1 not saturated**: A1@K=5 < 90 % on each (seed, scale)
   cell.
5. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline at ≥ 13/14 PASS
   (mirrors the W95 Phase 3 audit-coverage bar).
6. **Executor stays clean**: canonical-solution self-test
   passes at 100 % on each of the 100 Phase 3 problems.

### W105 cross-scale retirement margin

Phase 3 retirement is entitled IFF **all six bars PASS on
BOTH scales** AND the cross-scale mean B − A1 difference is
within ± 5 pp (no cross-scale collapse).  This is stricter
than W89's single-scale retirement bar.  The bar shape is
locked here so future expectations don't drift.

## Branch B — W105 mechanism-variation slate (pre-committed)

If W104 lands `PASS_NON_MECHANISM_DRIVEN` (9/9 Phase 2 gates
PASS but MLB-2 rescue rate < 33 % at 405B), W105 explores
mechanism variations on HumanEval+ at 405B BEFORE Phase 3
becomes entitled.

* **B1 — enforced-reflexion-on-attempt-0**: no early-pass
  short-circuit; reflexion chain runs the full K = 5
  unconditionally.
* **B2 — test-aware decomposition reader + solver** on the
  EvalPlus extra-test surface.
* **B3 — sidecar-driven failure-cluster targeting at 405B**:
  the reflexion prompt receives the executor stderr ranked
  by failure-cluster shape from the historical sidecar.

Cheap-pilot earning rule: at least one B-variant must lift
MLB-2 ≥ 33 % AT 405B AND keep margin ≥ +5 pp.  This branch's
slice pack is the SAME W103 helper-anchored 30-problem slice
(cheap-pilot budget; 330 NIM calls × N variants).

## Branch C — W105 fallback code-line ranking refresh (pre-committed)

If W104 FAILs (margin / G2 / per-problem-majority FAIL), the
W105 dispatch is determined by the FAIL mode per the table
below.  The dispatch logic is pre-committed here so the next
move is execution-ready under any FAIL shape.

### Branch C triage table

| FAIL mode | W105 lead step | NIM-spend ceiling | Justification |
|---|---|---|---|
| Margin < 0 pp AND MLB-2 < 33 % at 405B (mechanism-distribution shift; close to W102 MBPP+ V2 pattern) | LiveCodeBench preflight (NIM-free) | $0 | The W101 matrix ranked LiveCodeBench second-best; its lower per-problem ceiling at 70B + 405B makes it the natural next attack on the cross-bench mechanism. NIM-free preflight first; no expensive bench until P1-P4 + AddrW105-Pn pass. |
| G2 saturation (A1 ≥ 90 % on the slice; ceiling-pressure FAIL) | APPS preflight (NIM-free) | $0 | The W101 matrix ranked APPS C-grade on infra cost but its larger problem ceiling (10 K problems) defuses any A1 saturation; APPS preflight is the right move if 405B saturates HumanEval+. |
| Margin < +5 pp but ≥ 0 pp AND MLB-2 ≥ 33 % (per-seed sampling variance) | HumanEval+ multi-seed cheap confirmation at 405B (3 seeds × 30 problems × K=5; same W103 slice) | ~990 NIM calls × 405B rate | If reflexion is load-bearing but the per-seed margin missed by < 5 pp, the next cheap step is multi-seed sampling at 405B on the same slice rather than a new bench. |
| Branch-C-quad-bottom (margin < 0 AND MLB-2 ≥ 33 % AND G2 < 90 % at 405B; close to W96-A 11B→90B Phase 3 pattern) | Cross-scale-collapse audit + 3-seed Phase 3-shape confirmation at 70B ONLY | ~3 300 NIM calls at 70B | If the mechanism IS load-bearing at 405B but the margin reversed, the next move is to test whether the W103 70B PASS replicates at multi-seed and Phase 3 size at the SAME scale, isolating cross-scale collapse from per-seed sampling. |
| SWE-bench-lite | OUT OF SCOPE | — | The W89 reflexion mechanism does not have the structural shape to attack SWE-bench-lite's repo-level failure surface; the W101 matrix locked this out. |

### Branch C dispatch JSON (machine-readable; for the W105 driver)

```json
{
  "schema": "coordpy.w104_branch_c_dispatch.v1",
  "rules": [
    {
      "condition": {
        "b_minus_a1_pp_lt": 0.0,
        "mlb2_lt": 0.33
      },
      "next_step": "livecodebench_preflight_nim_free",
      "estimated_nim_calls": 0
    },
    {
      "condition": {
        "a1_pct_ge": 90.0
      },
      "next_step": "apps_preflight_nim_free",
      "estimated_nim_calls": 0
    },
    {
      "condition": {
        "b_minus_a1_pp_lt": 5.0,
        "b_minus_a1_pp_ge": 0.0,
        "mlb2_ge": 0.33
      },
      "next_step": "humaneval_plus_multi_seed_at_405b",
      "estimated_nim_calls": 990
    },
    {
      "condition": {
        "b_minus_a1_pp_lt": 0.0,
        "mlb2_ge": 0.33,
        "a1_pct_lt": 90.0
      },
      "next_step": "cross_scale_collapse_audit_plus_70b_phase3",
      "estimated_nim_calls": 3300
    }
  ],
  "fallback": "code_line_ranking_refresh_by_w101_matrix"
}
```

The driver that consumes this dispatch is built in W105 IF
W104 FAILs; pre-committing the dispatch JSON here ensures the
W105 driver is execution-ready under any FAIL shape.

## Honest framing

* This artifact is **planning**, not evidence.  No claim
  earned here.  Both branches' artifacts are pre-built so
  the W104 verdict triggers execution, not paperwork.
* The W105 Phase 3 slice pack (Branch A) is the **largest
  same-budget multi-agent superiority retirement attempt
  the programme has yet pre-committed** if W104 PASSes
  Branch A.  6 600 NIM calls × cross-scale × multi-seed.
* The Branch C dispatch logic ensures the next code-line
  move is honest under any FAIL shape; SWE-bench-lite stays
  unconditionally out of scope; all triage paths PRE-FLIGHT
  before any NIM spend except the per-seed multi-seed step.

## Anchors

* `docs/RUNBOOK_W104.md` — pre-commit contract.
* `scripts/run_w104_w105_phase3_slice_pack.py` — slice-pack
  pre-builder.
* `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.json` —
  W105 Phase 3 slice pack (locked; pack CID
  `8be55f3bf1650df3...`).
* `data/w105/phase3_slice_pack/w105_phase3_slice_pack_20260526T215647Z/slice_pack.md` —
  W105 Phase 3 slice pack runbook-ready markdown.
* `coordpy/code_slice_selector_v1.py` — helper module
  (second load-bearing downstream consumption).
* `results/w102/arsenal_mining/w102_arsenal_20260526T000910Z/mining_report.json` —
  arsenal-mining input.
* `results/w103/humaneval_plus_pilot/<RUN>/provenance.json` —
  W103 inner kernel source.
