# Contamination-control framing — W113 (the resistant column is filled; W112 +10 pp was exposure)

W113 completes the 2×2 that W112 set up. This note records exactly what the clean
resistant-for-Llama-4 result does and does not license, so it cannot be over- or
under-stated.

## The completed 2×2 (model scale × slice resistance, same W89 mechanism)

Columns = the slice's contamination status **for the tested model**
(model-cutoff-relative). Cells = B − A1.

| | contamination-EXPOSED (for the tested model) | contamination-RESISTANT (for the tested model) |
|---|---|---|
| **70B (Llama-3.3, cutoff ~2024-01)** | W89 HumanEval +5.56 (retire) · W105 HumanEval+ +7.00 (retire) · W109 APPS +16.67 (PASS_NON_MECH) · W110 BigCodeBench-2024 +0.00¹ | W108 LCB-2025 **−3.33 (FAIL)** · W110 BigCodeBench-2024 **+0.00 (FAIL)** |
| **Maverick (Llama-4, cutoff Aug-2024)** | **W112 BigCodeBench-2024 +10.00 (PASS_NON_MECH)** | **W113 LCB-2025 +0.00 (FAIL)** |

¹ BigCodeBench-2024 is RESISTANT for 70B (cutoff < release), so W110 sits in the 70B
resistant column; it is EXPOSED for Maverick, so W112 sits in the Maverick exposed
column — the same slice, two columns, by cutoff (the W112 model-cutoff-relativity
finding).

* **Resistant column: 0 clean wins across BOTH scales** (70B −3.33 / +0.00; Maverick
  +0.00). Reflexion is 0/3 on resistant code by margin; M3 sub-floor at 70B.
* **Exposed column: every margin** (2 retirements + 2 PASS_NON_MECHANISM_DRIVEN).
* **Two dissociations now:** W112 = first WITHIN-benchmark flip (BigCodeBench +0.00→+10.00
  across 70B→Maverick, i.e. as the cutoff crosses the release). W113 = first
  **WITHIN-MODEL** flip (Maverick +10.00 exposed BigCodeBench → +0.00 resistant
  LiveCodeBench, same model + same mechanism, only the slice's resistance differs) —
  the **cleanest contamination dissociation in the programme**.

## What the W113 +0.00 pp does and does NOT mean

* **DOES** confirm the W112 +10 pp was **contamination EXPOSURE**: the same mechanism on
  a verifiably-resistant slice (all 2025 dates ≫ Aug-2024) gives a flat FAIL at the same
  Maverick scale. The reflexion loop was genuinely invoked (MLB-1 63 %, *higher* than
  70B) but rescued only 21 % (< 33 % floor) and netted 0 (4 rescues − 2 regressions) —
  the SAME resistant-code collapse seen at 70B.
* **DOES** remove the "Maverick is just a stronger coder" alternative for the exposed
  margin: on the resistant slice Maverick's raw A1 (50 %) is BELOW 70B's (63.33 %), so
  the +10 pp tracked recall of seen 2024-06 problems, not capability on unseen code.
  (Single-seed; stated as alternative-removal, not a capability ranking.)
* **Does NOT** prove the contamination-confound — single-seed each; BigCodeBench-2024
  and LiveCodeBench-2025 differ in construction/difficulty, not only vintage (the
  within-slice A1 comparison weakens but does not eliminate this).
* **Does NOT** add a retirement or weaken W89/W105 (they STAND, contamination-EXPOSED
  HumanEval-family at 70B).

## Honesty rules (W113)

1. Report the W113 +0.00 pp as a **clean resistant FAIL at Maverick scale**, and use it
   to re-label the W112 +10 pp as **EXPOSURE** (never as a stronger-model resistant win).
2. Place W113 in the Maverick RESISTANT column; place W112 in the Maverick EXPOSED column.
3. The contamination-confound is **STRENGTHENED** (fourth, cleanest = within-model
   dissociation), **NOT proven**.
4. Resistant same-budget code superiority is **0 clean across BOTH scales** — say this,
   not "needs more seeds".
5. Re-qualify every "resistant benchmark" claim by the cutoff of the model tested
   (model-cutoff-relativity).
6. The next move (W114) is a GENUINELY DIFFERENT axis, not another exposed/resistant
   reflexion pilot; tier-2 is blocked on a missing certifiably-resistant instrument.

Anchors: `docs/RESULTS_W113_RESISTANT_PILOT_V1.md`,
`docs/RESULTS_W113_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W112_V1.md`, `docs/RUNBOOK_W113.md`.
