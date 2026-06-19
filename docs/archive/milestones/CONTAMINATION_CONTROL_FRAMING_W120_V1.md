# Contamination-control framing — W120 (official-ICPC ≥30 resistant pilot)

## What W120 tested

The contamination confound asks: *do the W89/W105 same-budget multi-agent reflexion
retirements (+5.56 / +7.00 pp) reflect a real capability gain, or do they ride on
benchmark contamination (the model has seen HumanEval-family solutions)?* W114–W119 could
not test the resistant column directly because no certifiable ≥30 grader-clean post-cutoff
instrument existed. **W120 built one and ran it.**

## The clean 2×N table (after W120)

| setting | model | benchmark | resistance | B − A1 |
|---|---|---|---|---|
| W89 | Llama-3.3-70B | base HumanEval | EXPOSED | **+5.56** (retire) |
| W105 | Llama-3.3-70B | HumanEval+ | EXPOSED | **+7.00** (retire) |
| W109 | Llama-3.3-70B | APPS (exposed control) | EXPOSED | +16.67 (non-mechanism) |
| W112 | Maverick | BigCodeBench (for-it exposed) | EXPOSED | +10.00 (non-mechanism) |
| W108 | Llama-3.3-70B | LiveCodeBench 2025 | RESISTANT | −3.33 |
| W110 | Llama-3.3-70B | BigCodeBench 2024-06 | RESISTANT | +0.00 |
| W113 | Maverick | LiveCodeBench (date-filtered) | RESISTANT | +0.00 |
| **W120** | **Maverick** | **official ICPC ≥30 (RMRC+ECNA)** | **RESISTANT** | **+0.00** |

**Every EXPOSED cell shows a margin; every RESISTANT cell shows ≈0 or negative — now 4/4
resistant nulls, across both model scales and four distinct instruments, the newest a
genuinely-new official grader-clean ≥30 battlefield.** The dissociation by exposure is now
very robust.

## What this DOES license

* The mechanism's measured advantage is **contamination-EXPOSED-HumanEval-family-at-70B
  specific** as a registered, bounded claim — and that boundary is now *tested* on the
  resistant side at a certified scale, not merely asserted for want of an instrument.
* The "we cannot test resistant code at a certified scale" escape (W114–W119) is **closed**.

## What this does NOT license (the discipline)

* **NOT** "contamination is proven." This is observational, not interventional. Confounds
  remain: (a) **single seed** (120001) on W120; (b) **difficulty** — ICPC-regional and
  LiveCodeBench problems are harder than HumanEval, so resistant ≈0 could be a
  difficulty-ceiling effect rather than (or in addition to) contamination; (c) a
  **Python-under-time-limit floor** on ICPC depresses absolute pass rates (binds all arms
  equally — conservative, cannot manufacture +0.00 — but lowers power). A clean
  interventional test would hold difficulty fixed while flipping only exposure.
* **NOT** "the mechanism does not work." It demonstrably retired two exposed benchmarks and
  was genuinely invoked on 25/30 W120 problems; it simply did not produce net rescues on
  resistant ICPC at this scale/seed.
* **NOT** a third retirement, a resistant win, cross-class, or "context solved."

## Status

Contamination-confound: **STRENGTHENED** (4th and cleanest resistant null) — **NOT proven**.
W89 + W105 STAND. The bounded ceiling holds and is now empirically tested on resistant code.
