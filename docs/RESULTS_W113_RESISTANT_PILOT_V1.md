# W113 — Maverick × RESISTANT-for-Llama-4 LiveCodeBench cheap pilot (clean FAIL → exposure confirmed)

**One line:** on a benchmark VERIFIABLY contamination-resistant for Llama-4-Maverick
(date-filtered LiveCodeBench, every problem 2025-01..02, months strictly after the
Aug-2024 cutoff), the W89 same-budget reflexion mechanism gets **B − A1 = +0.00 pp
(FAIL)** at Maverick scale — collapsing exactly as 70B did (W108, −3.33 pp). The
W112 **+10.00 pp on EXPOSED BigCodeBench was contamination EXPOSURE, not a
capability reopening of resistant superiority.** W89 + W105 STAND; W113 adds none.

---

## The result

Run `w113_resistant_pilot_meta_llama-4-maverick-17b-128e-instruct_20260529T225839Z`
(1 seed × 30 × K=5 = 330 NIM calls; ~55.5 min wall; `meta/llama-4-maverick-17b-128e-instruct`):

| metric | value |
|---|---|
| A0 (single-shot, T=0) | **30.00 %** (9/30) |
| A1 (first-pass-among-K=5, T=0.7) | **50.00 %** (15/30) |
| B (sequential-reflexion-K=5) | **50.00 %** (15/30) |
| **B − A1** | **+0.00 pp** |
| B − A0 | +20.00 pp |
| Phase-2 gates | **7/9** (G3 `B>A1` FAIL, G4 `margin≥5pp` FAIL) |
| MLB-1 invocation | **63.33 %** (19/30) → PASS |
| MLB-2 rescue | **21.05 %** (4/19) → **FAIL** (< 33 % floor) |
| **Verdict** | **`FAIL`** |
| Cross-scale outcome | **`EXPOSURE_CONFIRMED`** (clean_reopening = False) |

Mechanism detail: reflexion was **genuinely invoked** (19/30 problems had a failing
attempt-0, MLB-1 PASS — a *higher* invocation rate than 70B's 53 %). It rescued 4
problems (`#7,#8,#20,#25`) but **regressed 2** (`#14,#24`, where A1's self-consistency
sample passed and B's reflexion path did not) ⇒ **net zero** (+2 B>A1 wins − 2
regressions). The rescue rate (21 %) is the SAME resistant-code collapse seen at 70B
(W108 25 %, W110 25 %) — the mechanism is exercised and still does not beat
same-budget self-consistency on contamination-resistant code.

Provenance / audit:
* corpus SHA `bb4c364f…` (`release_v6` test6.jsonl); resistant partition **63/63**
  (boundary 2024-08-31 KNOWN; 0 excluded — 0 missing / 0 unparseable / 0 in-August).
* resistant slice CID `2afc318c…` == the **W108 slice CID** (clean cross-scale: the
  ONLY variable vs W108's 70B run is the model).
* preflight verdict CID `6f30990c…`; cross-scale interpretation CID `aa324208…`.
* `results/w113/resistant_pilot/w113_resistant_pilot_meta_llama-4-maverick-17b-128e-instruct_20260529T225839Z/`.

---

## The clean 2×2 (the disambiguation W113 was built for)

Rows = the W89 mechanism's margin; columns = the slice's contamination status **for
the tested model** (model-cutoff-relative, `W112-T-…`):

|                       |  70B (Llama-3.3)        |  Maverick (Llama-4)        |
|-----------------------|-------------------------|----------------------------|
| **EXPOSED**  (BigCodeBench 2024-06) | +0.00 pp  (W110) | **+10.00 pp** (W112)  |
| **RESISTANT** (LiveCodeBench 2025)  | **−3.33 pp** (W108, FAIL) | **+0.00 pp** (W113, FAIL) |

* The resistant column is now **0/2 across both scales** (70B −3.33; Maverick +0.00).
* **Within the SAME model + SAME mechanism, the margin flips +10.00 pp (exposed
  BigCodeBench) → +0.00 pp (resistant LiveCodeBench) purely on slice resistance** —
  the sharpest contamination dissociation in the programme (W112 was the first
  *within-benchmark* flip across models; W113 is the first *within-model* flip across
  slices).
* Corroborating capability check: on the resistant slice, Maverick's raw A1 (**50 %**)
  is *below* 70B's A1 (**63.33 %**) — the "stronger model" did not even deliver higher
  raw accuracy on genuinely-unseen 2025 problems, undercutting a capability
  explanation for the W112 +10 pp. (Single-seed; not asserted as a capability ranking,
  but it removes the "Maverick is just better at code" alternative for the exposed
  margin.)

---

## What W113 does and does NOT establish

* **DOES** confirm the W112 +10 pp was **EXPOSURE, not a clean resistant reopening**:
  the identical-shape mechanism on a verifiably-resistant slice gives a flat FAIL at
  the same scale. `W113-T-STRONGER-MODEL-RESISTANT-CODE-EXPOSURE-CONFIRMED`.
* **DOES** strengthen the contamination-confound via the **within-model exposed→resistant
  dissociation** (cleaner than W112's within-benchmark / cross-model one). Still **NOT
  proof** — single-seed each; orthogonal difficulty between BigCodeBench-2024 and
  LiveCodeBench-2025 is not fully excluded as a co-driver (though the within-slice
  A1 comparison weakens that alternative). `W113-T-CONTAMINATION-CONFOUND-STRENGTHENED-WITHIN-MODEL`.
* **DOES** harden the bounded claim: contamination-resistant same-budget code
  superiority is now **0 clean demonstrations across BOTH 70B and Maverick** (reflexion
  0/2 at 70B + Maverick resistant FAIL; M3 sub-floor at 70B).
  `W113-L-STRONGER-MODEL-RESISTANT-SUPERIORITY-FAIL-CAP`.
* **Does NOT** add a retirement (W113 adds none; W89 + W105 STAND, contamination-EXPOSED
  HumanEval-family at 70B). **Does NOT** weaken W89/W105. **Does NOT** prove the
  confound. Still NOT cross-class / MBPP-family / cross-modal / "context solved".

## W114 (pre-committed by the FAIL verdict label, RUNBOOK_W113 § 8)

`EXPOSURE_CONFIRMED` → **W114 = accept the bounded contamination-EXPOSED-HumanEval-
family-at-70B claim as the honest code ceiling and pursue a GENUINELY DIFFERENT axis**
— not another exposed rerun, not another same-scale resistant reflexion pilot. A tier-2
follow-up is BLOCKED on a missing instrument (no tier-2 model has a certifiably-resistant
slice on the pinned corpus; § 6 / Lane β). `COO-9` stays lead.

Anchors: `docs/RESULTS_W113_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W113_V1.md`, `docs/RUNBOOK_W113.md`,
`results/w113/resistant_slice_preflight/preflight_verdict.json`,
`results/w113/tier2_readiness/tier2_readiness.json`.
