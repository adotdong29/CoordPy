# Contamination-control framing — W112 (model-cutoff-relativity)

W112's stronger-model pilot surfaced a sharpening of the contamination-confound
story that this note records explicitly so it cannot be over- or under-stated.

## The new principle: contamination-resistance is MODEL-CUTOFF-RELATIVE

A benchmark is "contamination-resistant" only RELATIVE TO A MODEL whose training
cutoff PRE-DATES the benchmark's public release. The property does NOT transfer
to a newer model:

| Benchmark | Release | Resistant for Llama-3.3-70B (cutoff ~2024-01)? | Resistant for Llama-4-Maverick (cutoff **Aug-2024**)? |
|---|---|---|---|
| HumanEval / HumanEval+ / APPS | 2021 | NO (exposed) | NO (exposed) |
| LiveCodeBench `release_v6` | 2025 | YES | YES (contest dates 2025 > Aug-2024) |
| **BigCodeBench v0.1.4** | **2024-06** | **YES** | **NO — EXPOSED** (Aug-2024 cutoff > 2024-06 release) |

So the SAME BigCodeBench slice is a resistant probe for the 70B model and an
EXPOSED probe for Llama-4. Any cross-model comparison on it must account for this.

## The exposed-vs-resistant 2×2, updated through W112

Rows = the W89 same-budget mechanism's verdict; columns = the benchmark's
contamination status **for the model tested**.

| | contamination-EXPOSED (for the tested model) | contamination-RESISTANT (for the tested model) |
|---|---|---|
| **margin / PASS** | W89 HumanEval +5.56 (retire) · W105 HumanEval+ +7.00 (retire) · W109 APPS +16.67 (PASS_NON_MECH) · **W112 Maverick·BigCodeBench +10.00 (PASS_NON_MECH)** | — (none) |
| **flat / FAIL** | — | W108 LCB-2025·3.3 −3.33 (FAIL) · W110 BigCodeBench-2024·3.3 +0.00 (FAIL) · W111 M3 sub-reflexion (no fair pilot) |

* **Exposed column: 4 margins** (2 retirements + 2 PASS_NON_MECHANISM_DRIVEN),
  now including the for-Llama-4-exposed BigCodeBench point.
* **Resistant column: 0 clean wins** (reflexion 0/2; M3 sub-floor).
* **W112 is the first WITHIN-benchmark dissociation**: BigCodeBench's verdict
  flips +0.00 pp → +10.00 pp purely because the model cutoff crossed the
  benchmark release date — the cleanest contamination signal in the programme.

## What the W112 +10 pp does and does NOT mean

* **Does NOT** demonstrate that scale reopens contamination-RESISTANT superiority
  — the benchmark is exposed (not verified resistant) for Llama-4, the verdict is
  PASS_NON_MECHANISM_DRIVEN (MLB-1 sub-floor), it is single-seed, and the
  capability-vs-recall ambiguity is unresolved.
* **Does** STRENGTHEN the contamination-confound (third dissociation point; first
  within-benchmark flip). The structural twin to W109 APPS (A0 = 73.33 %
  identical) is strong corroboration that the margin tracks EXPOSURE, not a clean
  resistant mechanism win.
* **Does NOT** prove the confound (still single-seed each; a stronger model could
  also genuinely fix more from the same stderr — the 3 rescued problems were
  fair-unreachable at 70B; capability is not excluded).

## Honesty rules (W112)

1. Report the W112 +10 pp ALWAYS with the Aug-2024-cutoff caveat — never as a
   "resistant-code win".
2. Place W112 in the EXPOSED column, alongside W109 APPS, by structural signature.
3. Do NOT claim the contamination-confound is proven; it is STRENGTHENED.
4. Do NOT weaken W89/W105; W112 adds no retirement.
5. Re-qualify every "resistant benchmark" claim by the cutoff of the model tested.
6. The disambiguator (W113) is a benchmark verifiably resistant FOR Llama-4
   (dates > Aug-2024), not merely another model on BigCodeBench.

Anchors: `docs/RESULTS_W112_STRONGER_MODEL_BIGCODEBENCH_PILOT_V1.md`,
`docs/RESULTS_W112_MILESTONE_SUMMARY_V1.md`, `docs/CONTAMINATION_CONTROL_FRAMING_W110_V1.md`,
`docs/CONTAMINATION_CONTROL_FRAMING_W111_V1.md`.
