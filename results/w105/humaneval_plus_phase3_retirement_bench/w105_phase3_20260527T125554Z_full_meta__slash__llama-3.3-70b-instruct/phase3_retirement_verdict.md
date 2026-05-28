## W105 Phase 3 retirement verdict (schema `coordpy.phase3_retirement_evaluator_v1.v1`)

* slice_pack_cid: `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
* corpus_sha256: `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`

### Class `meta/llama-3.3-70b-instruct` — verdict **RETIRED** (6/6 bars PASS)

| # | Bar | Value | Threshold | PASS |
|---|---|---|---|---|
| 1 | mean B − A1 (pp) | +7.00 | ≥ +5.0 | YES |
| 2 | per-seed majority count | 3 | ≥ 2 | YES |
| 3 | per-problem majority count | 295 | ≥ 159 | YES |
| 4 | A1 < 90 % on each cell | 84.00, 82.00, 82.00 | all < 90 | YES |
| 5 | audit chain re-derives | 3/3 | ≥ 2 | YES |
| 6 | canonical-solution pass rate | 100.00 % | = 100 % | YES |

* mean MLB-1 invocation: 23.33 %
* mean MLB-2 rescue: 55.62 % (load-bearing)

Per-cell summary:

| seed | A0 % | A1 % | B % | B − A1 pp | MLB-1 % | MLB-2 % | bench Merkle |
|---|---:|---:|---:|---:|---:|---:|---|
| 105001 | 78.00 | 84.00 | 89.00 | +5.00 | 22.00 | 50.00 | `82ed16f0d3c131c9...` |
| 105002 | 78.00 | 82.00 | 91.00 | +9.00 | 23.00 | 60.87 | `ea4c791c25d43b2a...` |
| 105003 | 78.00 | 82.00 | 89.00 | +7.00 | 25.00 | 56.00 | `d00737d0cb80ea3d...` |

