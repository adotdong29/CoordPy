# W104 — Cross-scale comparator V1

> **2026-05-26.  Cross-scale comparison of the W103 70B HumanEval+ cheap-pilot result vs the W104 cross-scale pilot at `meta/llama-3.1-70b-instruct` on the BYTE-EQUAL 30-problem helper-anchored slice.**

* Cross-scale form achieved: cross-generation (Llama-3.1 vs Llama-3.3 at 70B).
* Slice CID (bench-iteration order; identical at both scales): `d5364a2f5a6ab3d6febe69b99d8424f75a54ad6f1dbde9e5e8e2d7e62c9e3052`.
* HumanEval+ corpus SHA-256 (identical at both scales): `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492`.
* Comparator schema: `coordpy.cross_scale_comparator_v1.v1`.

## Aggregate cross-scale numbers

| Field | Scale A (W103 70B Llama-3.3) | Scale B (W104 meta/llama-3.1-70b-instruct) |
|---|---|---|
| Model | `meta/llama-3.3-70b-instruct` | `meta/llama-3.1-70b-instruct` |
| Bench Merkle | `68f4a9669f1bd03e6b3cb393a436e4f04aca034a0bad9c4b5ea8a002faabfd6d` | `1a3d93aa5a2119338a9dca94e4d23e9275d921631b9b712c4f13f3e1e99d0171` |
| A0 mean pass-rate | +20.00 pp (see below) | see verdict doc |
| MLB-1 invocation | 56.67% | 56.67% |
| MLB-2 rescue | 47.06% | 35.29% |
| B − A1 (pp) | +20.00 | +10.00 |

**Cross-scale shift on B − A1**: -10.00 pp
**Cross-scale shift on MLB-2 rescue rate**: -11.77 pp

## Aggregate arm deltas (B − A)

* A0: -3.33 pp
* A1: +3.33 pp
* B: -6.67 pp

## Cluster-shift aggregate

| Shift | Count |
|---|---:|
| `stayed` | 8 |
| `improved` | 11 |
| `regressed` | 11 |
| `flipped` | 0 |

## Per-problem rows

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

## Honest framing

* The cross-scale form achieved is cross-generation (Llama-3.1 vs Llama-3.3 at 70B).  Pre-locked primary `meta/llama-3.1-405b-instruct` was unreachable; pre-locked backup `meta/llama-3.1-70b-instruct` was applied per the W104 RUNBOOK § Target-model selection rule.
* This comparator output reflects cross-GENERATION at the same parameter scale, NOT cross-scale-UP.  Future W104.x or W105 milestones must use the proper cross-scale-UP target if 405B becomes reachable on NIM.
* The slice + corpus + schema are byte-equal at both scales (the comparator REFUSES to run otherwise per the W104 hardening lane).  The per-problem rows are apples-to-apples.

## Anchors

* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/cross_scale_comparator_report.json` — comparator JSON.
* `results/w104/humaneval_plus_cross_scale_pilot/w104_humaneval_plus_cross_scale_pilot_meta_llama-3.1-70b-instruct_20260526T215829Z/cross_scale_comparator_report.md` — comparator markdown emitted by the driver.
* `docs/RESULTS_W104_HUMANEVAL_PLUS_PHASE2_405B_V1.md` — W104 verdict.
* `docs/RUNBOOK_W104.md` — pre-commit contract.