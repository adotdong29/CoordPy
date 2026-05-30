# Contamination-control framing — W115 (the supply blocker, LIVE-re-verified + operationalised)

W115 does not run a contamination experiment — it RE-VERIFIES, LIVE from primary
sources, the certification-supply boundary W114 established, and OPERATIONALISES it
into a durable pipeline. This note records exactly what W115 does and does not
license.

## The registered position (unchanged from W114; W115 re-confirms it LIVE)

The 2×2 (model scale × slice resistance, same W89 mechanism) is complete and the
RESISTANT column is **0 clean across BOTH scales**:

| | contamination-EXPOSED (for the tested model) | contamination-RESISTANT (for the tested model) |
|---|---|---|
| **70B (Llama-3.3, cutoff ~2024-01)** | W89 HumanEval +5.56 (retire) · W105 HumanEval+ +7.00 (retire) · W109 APPS +16.67 (PASS_NON_MECH) | W108 LCB-2025 **−3.33 (FAIL)** · W110 BigCodeBench-2024 **+0.00 (FAIL)** |
| **Maverick (Llama-4, cutoff Aug-2024)** | W112 BigCodeBench-2024 +10.00 (PASS_NON_MECH) | W113 LCB-2025 **+0.00 (FAIL)** |

* Resistant column: **0 clean wins** across both scales. The two confirmed
  retirements (W89, W105) are BOTH in the EXPOSED column at 70B.
* This bounded position is the honest code-superiority ceiling, registered at W114
  and re-confirmed current at W115.

## What W115 adds: LIVE re-verification + the future-fire pipeline

The forward move was instrument/cutoff SUPPLY, not "run another seed". W115:

* **Re-verified the supply blocker LIVE** (primary sources, 2026-05-29): LCB
  `release_v6` is still the latest official release (no `test7`+; frontier
  2025-04-05), and no reachable stronger-than-Maverick model has a KNOWN cutoff
  ≤ ~Jan-2025. The one external change since W114 — the DeepSeek V4 official model
  card now EXISTS (published 2026-04-27) — discloses NO cutoff, so it does not move
  the verdict. The blocker is real and current
  (`W115-L-EXTERNAL-FRONTIER-UNCHANGED-NO-CERTIFIABLE-SLICE-REVERIFIED-CAP`).
* **Operationalised the supply chain** into `coordpy.frontier_certification_
  pipeline_v1`: a latest-release detector + frontier-date summary + per-model
  go/no-go matrix + disclosure-consistency guard + structured W116 fire condition,
  driven by a `FrontierSnapshotV1` (external state as data). The next clean shot is
  push-button (`W115-T-FUTURE-FIRE-CERTIFICATION-PIPELINE-SHIPS`).

So: `NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM — a contamination-control SUPPLY
finding (which models can even be cleanly tested, against which instrument), not a
contamination experiment.

## Honesty rules (W115)

1. The contamination-confound status is **UNCHANGED** from W113/W114 (STRENGTHENED,
   NOT proven). W115 does NOT run a contamination test, so it neither strengthens
   nor weakens the confound — say this explicitly.
2. Report the bounded ceiling as the **registered honest floor, re-confirmed
   current**, not surrender.
3. The blocker is **model-cutoff-relativity made into a supply gate, now LIVE-
   re-verified**: the newest dated FUNCTIONAL problems available (Apr-2025) still
   predate/coincide with the reachable frontier models' (undisclosed) cutoffs.
4. Never say a stronger model "failed" a resistant test in W115 — none was tested;
   none is certifiable. The honest statement is "uncertifiable on the latest real
   data, re-verified live".
5. Never say "the DeepSeek V4 card proves a cutoff" — it discloses NONE; the card's
   *existence* (2026-04-27) is the only change, and it leaves the model UNKNOWN.
6. Never say LiveCodeBench is "exhausted" or "contaminated" — it remains a clean
   instrument; its newest FUNCTIONAL problems (Apr-2025) simply no longer post-date
   the reachable frontier models' cutoffs.
7. The next move (W116) requires a NEW certified post-cutoff instrument OR a newly-
   disclosed KNOWN cutoff; the pipeline makes that re-evaluation push-button. It is
   NOT another exposed/resistant reflexion pilot on the current instrument.

Anchors: `docs/RESULTS_W115_FRONTIER_CERTIFICATION_V1.md`,
`docs/RESULTS_W115_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W114_V1.md`, `docs/RUNBOOK_W115.md`.
