| Field | Scale A | Scale B |
|---|---|---|
| Model | `meta/llama-3.3-70b-instruct` | `meta/llama-3.1-70b-instruct` |
| Bench Merkle | `68f4a9669f1bd03e...` | `1a3d93aa5a211933...` |
| MLB-1 invocation | 56.67% | 56.67% |
| MLB-2 rescue | 47.06% | 35.29% |
| B − A1 (pp) | +20.00 | +10.00 |

**Cross-scale shift on B − A1**: -10.00 pp
**Cross-scale shift on MLB-2**: -11.77 pp

### Cluster-shift aggregate

| Shift | Count |
|---|---:|
| `stayed` | 8 |
| `improved` | 11 |
| `regressed` | 11 |
| `flipped` | 0 |

### Per-problem cross-scale rows

| idx | task_id | A0 A→B | A1 A→B | B A→B | bidx A→B | shift |
|---|---|---|---|---|---|---|
| 1 | HumanEval/129 | 0→1 | 0→1 | 0→1 | -1→1 | `improved` |
| 2 | HumanEval/44 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 3 | HumanEval/111 | 1→1 | 1→1 | 1→1 | 0→0 | `stayed` |
| 4 | HumanEval/16 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 5 | HumanEval/17 | 1→1 | 1→1 | 1→1 | 1→0 | `stayed` |
| 6 | HumanEval/76 | 0→1 | 0→1 | 0→1 | -1→0 | `improved` |
| 7 | HumanEval/55 | 0→1 | 0→1 | 1→1 | 2→0 | `improved` |
| 8 | HumanEval/163 | 0→0 | 0→1 | 1→1 | 1→1 | `improved` |
| 9 | HumanEval/132 | 0→1 | 0→1 | 0→1 | -1→1 | `improved` |
| 10 | HumanEval/160 | 0→1 | 0→1 | 1→1 | 2→0 | `improved` |
| 11 | HumanEval/83 | 0→1 | 0→1 | 0→1 | -1→0 | `improved` |
| 12 | HumanEval/61 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 13 | HumanEval/101 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 14 | HumanEval/35 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 15 | HumanEval/118 | 1→0 | 0→0 | 1→0 | 1→-1 | `regressed` |
| 16 | HumanEval/140 | 0→1 | 0→1 | 0→1 | -1→0 | `improved` |
| 17 | HumanEval/119 | 1→1 | 1→1 | 1→1 | 0→0 | `stayed` |
| 18 | HumanEval/154 | 0→1 | 1→1 | 0→1 | -1→0 | `improved` |
| 19 | HumanEval/49 | 1→0 | 1→0 | 1→1 | 0→4 | `regressed` |
| 20 | HumanEval/32 | 0→0 | 0→0 | 0→0 | -1→-1 | `stayed` |
| 21 | HumanEval/121 | 0→0 | 0→0 | 1→1 | 1→2 | `stayed` |
| 22 | HumanEval/113 | 1→1 | 1→1 | 1→1 | 0→0 | `stayed` |
| 23 | HumanEval/91 | 0→1 | 0→1 | 0→1 | -1→0 | `improved` |
| 24 | HumanEval/137 | 0→0 | 0→0 | 1→0 | 2→-1 | `regressed` |
| 25 | HumanEval/100 | 1→0 | 1→0 | 1→1 | 0→2 | `regressed` |
| 26 | HumanEval/122 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 27 | HumanEval/125 | 0→0 | 0→1 | 1→1 | 2→0 | `improved` |
| 28 | HumanEval/104 | 1→0 | 1→0 | 1→0 | 0→-1 | `regressed` |
| 29 | HumanEval/14 | 1→1 | 1→1 | 1→1 | 0→0 | `stayed` |
| 30 | HumanEval/84 | 0→0 | 0→0 | 0→0 | -1→-1 | `stayed` |
