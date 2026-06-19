# Contamination-control framing — W114 (the bounded ceiling is registered; the resistant-instrument frontier has aged out)

W114 does not run a contamination experiment — it REGISTERS the contamination
boundary established through W113 and establishes WHY no further clean
contamination-resistant test of a stronger model is buildable from the latest
real data. This note records exactly what W114 does and does not license.

## The registered position (after W113, now the W114 floor)

The 2×2 (model scale × slice resistance, same W89 mechanism) is complete and the
RESISTANT column is **0 clean across BOTH scales**:

| | contamination-EXPOSED (for the tested model) | contamination-RESISTANT (for the tested model) |
|---|---|---|
| **70B (Llama-3.3, cutoff ~2024-01)** | W89 HumanEval +5.56 (retire) · W105 HumanEval+ +7.00 (retire) · W109 APPS +16.67 (PASS_NON_MECH) | W108 LCB-2025 **−3.33 (FAIL)** · W110 BigCodeBench-2024 **+0.00 (FAIL)** |
| **Maverick (Llama-4, cutoff Aug-2024)** | W112 BigCodeBench-2024 +10.00 (PASS_NON_MECH) | W113 LCB-2025 **+0.00 (FAIL)** |

* Resistant column: **0 clean wins** across both scales. Exposed column: every
  margin (2 retirements + 2 non-mechanism-driven).
* The two confirmed retirements (W89, W105) are BOTH in the EXPOSED column at 70B.
* This bounded position is the honest code-superiority ceiling W114 registers.

## What W114 adds: the certification-supply (instrument-frontier) analysis

The forward move from W113 was to build a NEW instrument certifiably resistant
for a model STRONGER than Maverick. W114 verified (primary sources, 2026-05-29)
that this is NOT buildable from the latest real data, for a precise, dated
reason — the **resistant-instrument frontier has aged out relative to the
reachable model frontier**:

* The latest LiveCodeBench release is `release_v6`; its FUNCTIONAL subset (the
  part the W89 mechanism attacks) is 63 problems dated 2025-01-11..2025-04-05.
  A ≥30 functional resistant slice requires a KNOWN cutoff ≤ ~Jan-2025.
* The reachable stronger-than-Maverick frontier models — Qwen3-Coder-480B
  (2025-07), DeepSeek-V4-pro (2025+), Mistral-Small-4 (2026-03) — have
  OFFICIALLY UNDISCLOSED cutoffs, so they cannot be certified resistant against
  ANY instrument (the W113 KNOWN-cutoff-only rule); and where estimable, their
  cutoffs meet/post-date the Apr-2025 frontier (so even disclosure would leave
  release_v6 exposed for them).
* Maverick (Aug-2024 KNOWN) is the only reachable model with a KNOWN cutoff and
  is already SETTLED here (W113 resistant FAIL ⇒ a second pilot is redundant).

So: `NO_CERTIFIABLE_STRONGER_MODEL`, $0 NIM. This is a contamination-control
SUPPLY finding (which models can even be cleanly tested, and against which
instrument), not a contamination experiment.

## Honesty rules (W114)

1. The contamination-confound status is **UNCHANGED** from W113 (STRENGTHENED a
   fourth time, NOT proven). W114 does NOT run a contamination test, so it neither
   strengthens nor weakens the confound — say this explicitly.
2. Report the bounded ceiling as the **registered honest floor**, not surrender.
3. The blocker is **model-cutoff-relativity made into a supply gate**: a
   benchmark resistant for an older model can be exposed for a newer one, and the
   newest dated FUNCTIONAL problems available (Apr-2025) now predate/coincide with
   the reachable frontier models' (undisclosed) cutoffs.
4. Never say a stronger model "failed" a resistant test in W114 — none was
   tested; none is certifiable. The honest statement is "uncertifiable on the
   latest real data".
5. Re-qualify every "resistant benchmark" claim by BOTH the tested model's cutoff
   AND the instrument's newest problem date.
6. The next move (W115) requires a NEW certified post-cutoff instrument; it is
   NOT another exposed/resistant reflexion pilot on the current instrument.

Anchors: `docs/RESULTS_W114_STRONGER_MODEL_CERTIFICATION_V1.md`,
`docs/RESULTS_W114_MILESTONE_SUMMARY_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W113_V1.md`, `docs/RUNBOOK_W114.md`.
