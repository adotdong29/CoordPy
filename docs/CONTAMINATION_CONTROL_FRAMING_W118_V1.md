# Contamination-control framing — W118

**Purpose.** Keep the contamination-confound claim honest after W118. W118 attacks the
INSTRUMENT-CONSTRUCTION supply (can a clean post-v6 functional instrument be built from
official sources?) and the model-cutoff supply — it does **NOT** test the
contamination-confound itself. The confound status is **UNCHANGED** from W110–W117.

## What W118 did and did not touch

| | Status after W118 |
|---|---|
| Confirmed retirements | **TWO** — W89 (base HumanEval, +5.56 pp) + W105 (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B, contamination-EXPOSED HumanEval-family. **UNCHANGED.** |
| Resistant same-budget superiority | **0 clean across both scales** (70B −3.33 / +0.00; Maverick +0.00). **UNCHANGED** (W118 ran no pilot). |
| Contamination-confound | **STRENGTHENED-not-proven** (W110/W113 reading). **UNCHANGED** — W118 tests construction + grader supply, not the confound. |
| What W118 added | A live CoordPy-OWNED post-v6 functional-IDENTITY manifest (894 official problems) + the proof that the executable GRADER is absent family-wide + the durable constructor pipeline. |

## Why W118 does not move the confound

The contamination-confound is the hypothesis that the W89 mechanism's same-budget
superiority appears on contamination-EXPOSED HumanEval-family problems but not on
contamination-RESISTANT code. Testing it requires a *runnable* resistant pilot
(generated code GRADED against hidden tests). W118 establishes that a resistant
*instrument identity* is constructible (894 official post-v6 problems, all resistant for
Maverick's KNOWN cutoff) but that **no official source publishes the executable grader**
needed to run the pilot. So W118 cannot — and does not — add a resistant data point. It
**sharpens the blocker** (from "no instrument" to "no official grader") without changing
the confound's evidential status.

## The honest reading

* W118 is a **supply-side construction result**, not a confound result.
* The post-v6 functional-identity axis is now demonstrably **officially constructible**
  (894 problems) — a real advance over W117's "cannot be inherited."
* The ONLY thing between the programme and a verdict-changing resistant Maverick pilot is
  a **reproducible official executable test suite** (O7). Maverick is already
  identity-certifiable on the constructed instrument.
* A **sample-only** grader (Codeforces statement examples) would make B−A1
  uninterpretable (hidden-test false-pass) and is refused; an **operator-synthesised**
  grader is operator curation and is refused. So $0 NIM is the disciplined outcome.

## What would change the reading (W119 triggers)

A primary-KNOWN ≤-frontier cutoff for a reachable stronger model, OR an official
executable grader for ≥30 post-v6 problems, OR a packaged `release_v7`+ /
LCB-published construction provenance. Until one holds, the bounded
contamination-EXPOSED-HumanEval-family-at-70B ceiling STANDS and resistant-code NIM is
BLOCKED on the missing official grader.
