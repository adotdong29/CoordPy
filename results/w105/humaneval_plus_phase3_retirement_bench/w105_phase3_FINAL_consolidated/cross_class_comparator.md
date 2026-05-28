## Cross-class comparator (schema `coordpy.cross_class_comparator_v1.v1`)

* class A: `meta/llama-3.1-70b-instruct`
* class B: `meta/llama-3.3-70b-instruct`
* slice_pack_cid: `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
* corpus_sha256: `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`

### Aggregate (mean across seeds)

* cross-class shift on B − A1: +4.67 pp
* cross-class shift on MLB-2: +5.09 pp
* aggregate cluster shifts: {"stayed": 242, "improved": 27, "regressed": 28, "flipped": 3}

### Per-seed deltas

| seed | A B-A1 pp | B B-A1 pp | shift on B-A1 pp | A MLB-2 % | B MLB-2 % | shift on MLB-2 pp | stayed | improved | regressed | flipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 105001 | +5.00 | +5.00 | +0.00 | 58.33 | 50.00 | -8.33 | 81 | 9 | 10 | 0 |
| 105002 | +1.00 | +9.00 | +8.00 | 45.45 | 60.87 | +15.42 | 80 | 9 | 9 | 2 |
| 105003 | +1.00 | +7.00 | +6.00 | 47.83 | 56.00 | +8.17 | 81 | 9 | 9 | 1 |
