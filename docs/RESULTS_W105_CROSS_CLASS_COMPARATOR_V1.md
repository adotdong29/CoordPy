# W105 — Cross-class comparator + entitlement verdict (V1)

> **2026-05-28.  Per-seed-aligned cross-class comparison of the
> two earned model classes on the byte-equal W105 Phase 3 slice
> pack (CID `8be55f3bf1650df3...`), 3 matched seeds × 100
> problems × K = 5.  Computed by `coordpy.cross_class_comparator_v1`
> (per-(matched seed) iteration alignment — the W105 fix for the
> W104 V1 row-misalignment lesson).**
>
> **Cross-class retirement entitlement: NOT ENTITLED.**  Only
> one class (`meta/llama-3.3-70b-instruct`) cleared all 6
> retirement bars; the other (`meta/llama-3.1-70b-instruct`)
> FAILed the margin bar.  The bounded claim is therefore
> single-class: HumanEval+ Phase 3 retirement on
> `meta/llama-3.3-70b-instruct` ONLY.

## Per-class verdicts (evaluated independently FIRST)

| Class | Verdict | mean B − A1 | per-seed maj | A1 (per cell) | MLB-2 | bars |
|---|---|---:|---:|---|---:|---:|
| `meta/llama-3.3-70b-instruct` | **RETIRED** | +7.00 pp | 3/3 | 84/82/82 % | 55.62 % | 6/6 |
| `meta/llama-3.1-70b-instruct` | **FAIL_MARGIN** | +2.33 pp | 3/3 | 85/87/87 % | 50.54 % | 5/6 |

## Cross-class entitlement rule (W105 RUNBOOK; locked BEFORE the run)

> Cross-class retirement claim entitled IFF:
> (i) both per-class verdicts are `RETIRED`;
> AND
> (ii) cross-class mean B − A1 difference within ± 5 pp.

* Condition (i): **FAILS** — Llama-3.1-70B is `FAIL_MARGIN`,
  not `RETIRED`.
* Condition (ii): the cross-class mean B − A1 difference is
  **4.67 pp** (|+7.00 − +2.33|), which IS within the ± 5 pp
  envelope — but (i) is not met, so the stronger claim is not
  earned regardless.
* **Entitlement: NOT ENTITLED.**  Claim label:
  `CLASS_B_RETIRED_CLASS_A_FAIL_BOUNDED_CLAIM` (the comparator's
  internal class_a/class_b labels are assigned by dict-iteration
  order: class_a = Llama-3.1 (FAIL), class_b = Llama-3.3
  (RETIRED)).

## Per-seed cross-class deltas (per-seed-aligned)

The comparator iterates each matched seed with the IDENTICAL
bench-internal shuffle on both sides (the W105 alignment
guarantee).  "class A" = Llama-3.1-70B; "class B" =
Llama-3.3-70B.

| seed | Llama-3.1 B−A1 | Llama-3.3 B−A1 | shift (3.1→3.3) | Llama-3.1 MLB-2 | Llama-3.3 MLB-2 | shift | stayed | improved | regressed | flipped |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 105001 | +5.00 | +5.00 | +0.00 | 58.33 % | 50.00 % | −8.33 | 81 | 9 | 10 | 0 |
| 105002 | +1.00 | +9.00 | +8.00 | 45.45 % | 60.87 % | +15.42 | 80 | 9 | 9 | 2 |
| 105003 | +1.00 | +7.00 | +6.00 | 47.83 % | 56.00 % | +8.17 | 81 | 9 | 9 | 1 |
| **mean** | **+2.33** | **+7.00** | **+4.67** | **50.54 %** | **55.62 %** | **+5.09** | — | — | — | — |

## Aggregate cluster transitions (300 problem-seed cells)

| Shift | Count |
|---|---:|
| `stayed` | 242 |
| `improved` (Llama-3.1 → Llama-3.3 gains) | 27 |
| `regressed` (Llama-3.1 → Llama-3.3 loses) | 28 |
| `flipped` (ambiguous) | 3 |

Only **3 of 300** transitions are ambiguous (`flipped`) — the
per-seed alignment is structurally clean (contrast with the
W104 V1 comparator, whose per-problem labels were arithmetically
mis-aligned because the two scales used different seeds; W105
fixes this by matching seeds).  The near-symmetric improved (27)
vs regressed (28) count on seed-aligned problems is consistent
with model-class sampling variation, NOT a systematic per-problem
advantage of one class.

## Interpretation

* Seed 105001 is the ONLY seed where the two classes agree at
  +5.00 pp.  On seeds 105002 + 105003, Llama-3.3 holds +9 / +7
  pp while Llama-3.1 collapses to +1 / +1 pp.  The collapse is
  driven by Llama-3.1's A1 rising to 87 % on those seeds (vs
  85 % on seed 105001), compressing the reflexion headroom.
* Both classes keep MLB-2 above the 33 % load-bearing floor on
  every seed.  The mechanism is real on BOTH classes; the
  DIFFERENCE is margin magnitude on the broad Phase 3 slice.
* The honest reading: the W89 sequential-reflexion mechanism
  retires on HumanEval+ at Phase 3 on Llama-3.3-70B, and is
  load-bearing-but-margin-capped on Llama-3.1-70B.  The
  cross-class "across two model classes" claim is NOT earned.

## Honest scope

* This comparator does NOT claim cross-class retirement.
* It does NOT average the two classes' margins into a combined
  number — the per-class verdicts are the load-bearing surface.
* It does NOT bear on cross-scale-UP (405B unreachable) or
  cross-modal (frozen at 11B) or MBPP-family (W102 cap).

## Anchors

* `coordpy/cross_class_comparator_v1.py` — per-seed-aligned
  comparator module.
* `coordpy/phase3_retirement_evaluator_v1.py` — per-class +
  cross-class entitlement evaluator.
* `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_FINAL_consolidated/cross_class_comparator.json` —
  machine-readable comparator report.
* `results/w105/humaneval_plus_phase3_retirement_bench/w105_phase3_FINAL_consolidated/phase3_retirement_verdict.json` —
  unified per-class + cross-class verdict.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA33_V1.md` —
  Llama-3.3 RETIRED verdict.
* `docs/RESULTS_W105_HUMANEVAL_PLUS_PHASE3_LLAMA31_V1.md` —
  Llama-3.1 FAIL_MARGIN verdict.
