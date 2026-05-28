## W105 Phase 3 retirement verdict (schema `coordpy.phase3_retirement_evaluator_v1.v1`)

* slice_pack_cid: `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
* corpus_sha256: `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`

### Class `meta/llama-3.1-70b-instruct` — verdict **FAIL_MARGIN** (5/6 bars PASS)

| # | Bar | Value | Threshold | PASS |
|---|---|---|---|---|
| 1 | mean B − A1 (pp) | +2.33 | ≥ +5.0 | NO |
| 2 | per-seed majority count | 3 | ≥ 2 | YES |
| 3 | per-problem majority count | 294 | ≥ 159 | YES |
| 4 | A1 < 90 % on each cell | 85.00, 87.00, 87.00 | all < 90 | YES |
| 5 | audit chain re-derives | 3/3 | ≥ 2 | YES |
| 6 | canonical-solution pass rate | 100.00 % | = 100 % | YES |

* mean MLB-1 invocation: 23.00 %
* mean MLB-2 rescue: 50.54 % (load-bearing)

Per-cell summary:

| seed | A0 % | A1 % | B % | B − A1 pp | MLB-1 % | MLB-2 % | bench Merkle |
|---|---:|---:|---:|---:|---:|---:|---|
| 105001 | 79.00 | 85.00 | 90.00 | +5.00 | 24.00 | 58.33 | `acf956e8d56affd2...` |
| 105002 | 79.00 | 87.00 | 88.00 | +1.00 | 22.00 | 45.45 | `3e3c0301dbb37787...` |
| 105003 | 79.00 | 87.00 | 88.00 | +1.00 | 23.00 | 47.83 | `1011a848aca8cdb8...` |

