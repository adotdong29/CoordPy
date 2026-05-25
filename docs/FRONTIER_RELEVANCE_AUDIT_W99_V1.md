# W99 — Frontier relevance audit V1 (supplement to W97 + W98 V1)

> 2026-05-25.  Third supplement, extending the W97 +
> W98 frontier-relevance audits with the W99 candidate slate
> (B2, B4, B5) post the W98 B1 Phase 2 11B FAIL and the
> empirically-mined "extract-then-text-reason" family cap at
> B − A1 ≈ −6.67 pp at 11B through TWO distinct mechanisms.
> The W97 + W98 audit classifications all remain in force;
> this supplement only ADDS to the active-frontier-arsenal
> column (B2 + B4 + B5) and re-asserts the anti-pattern column
> verbatim.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.

## Why a supplement, not a rewrite

W98 V1 documented the classification *as of the W98 B1 cheap-
pilot FAIL moment*.  W98's empirical evidence sharpened two
prior classifications:

1. The W95-B0-derived "extract-then-text-reason" family is
   now FOUR-attempts-empirically-capped at −6.67 pp on
   RealWorldQA at 11B (W97 D2-B0 + W98 B1; the W99 milestone
   tests B4 + B5 + B2 against the same cap).
2. The W98 B1 typed schema's *yes/no recovery surface* (4 / 5
   W97 unique-A1-rescues recovered) IS a load-bearing
   mechanism — but is offset by a previously-unidentified
   reader-hint-anchoring failure on multi-choice.

This supplement formalises:

* the addition of B2 / B4 / B5 to the active-frontier-arsenal
  column;
* the explicit *baseline-only ceiling* status of B5 (a
  switch baseline that bounds routing-ceiling, NOT a frontier
  mechanism);
* the re-statement of the W97 + W98 anti-pattern column
  *verbatim* — bounded / compaction / summary remain explicit
  anti-patterns;
* the explicit acknowledgement that if all three W99
  candidates FAIL Phase 2, the W95-B0 family is empirically
  capped on RealWorldQA at 11B and ``COO-9`` is the next lead
  path.

## Additions to the active frontier arsenal (W99)

| Mechanism | Module(s) | Why still frontier (with W98 evidence) |
|---|---|---|
| **B2 — Direct-vision final-turn answerer** | ``coordpy/realworldqa_bench_v3.py`` (W98) | Structural frontier mechanism: keep image alive at the decision boundary on the failure cluster.  Distinct from W96-C C1 verifier (committed answerer, not binary).  NIM-free upper bound from W97 confusion table predicts realistic +6.67 pp.  The user's W99 brief explicitly names this as the **structural lead**. |
| **B4 — Typed schema WITHOUT direct_answer_hint** | NEW ``coordpy/realworldqa_bench_v4.py`` (W99) | Minimal repair of W98 B1.  Preserves the 4 / 5 yes/no recovery surface (which IS load-bearing per W98 evidence) AND removes the proximate cause of W98 B1's 5 multi-choice regressions (reader-hint anchoring). |
| **B5 — Question-type router (switch baseline)** | NEW ``coordpy/realworldqa_bench_v5.py`` (W99) | NIM-free ORACLE on W97 slice = 100 % (30 / 30); B5 − A1 = +10 pp.  Routing-ceiling reference: bounds how much team superiority is achievable by routing alone.  Classified **baseline-only** (NOT frontier mechanism) even if it clears Phase 2. |
| **W99 multi-candidate cheap-discriminator rule** | NEW (this milestone): ``docs/RUNBOOK_W99.md`` | "Multiple cheap tries allowed when multiple candidates earn it."  Three pilots × 1 seed × 30 problems × K=5 = ~ 990 NIM calls; cheaper than one Phase 3 retirement bench. |
| **W98 sidecar mining (extends W97 failure-cluster miner)** | ``coordpy.failure_cluster_miner_v1`` + W98 sidecars + W99 oracle simulation | Validated load-bearing tool: the W99 candidate slate is mined directly from the W98 B1 sidecars' per-problem disagreement vs W97 D2-B0.  The B5 oracle prediction is an exact NIM-free computation from these sidecars. |

## Useful baselines (W99 changes from W97 / W98)

| Mechanism | Module(s) | Classification | Why baseline-only |
|---|---|---|---|
| ``bounded_window_baseline_v{1,2,3}`` | ``coordpy/bounded_window_baseline_v*.py`` | UNCHANGED from W97 / W98 — useful falsifier targets the substrate-coupled methods must beat. | Same. |
| **B5 question-type router (NEW W99 baseline)** | ``coordpy/realworldqa_bench_v5.py`` | **Baseline-only ceiling / floor reference.**  Even if Phase 2 PASSes, B5 is NOT promoted to frontier mechanism. | B5 is a *switch*; it does not introduce a new mechanism for cross-modal team coordination.  Its PASS only proves the per-question ceiling is high enough; it does NOT prove structural team superiority. |
| A0 / A1 baselines | unchanged | unchanged | unchanged |

## Historical artifacts (unchanged from W97 / W98 V1)

W90 / W92 / W88 / W81 / W83 / W84 unchanged.  Kept for
regression / audit; not active path.

## Dead directions (refuted by evidence; do not entertain as core strategy)

The W97 + W98 list remains in force verbatim.  W99 evidence
will *either*:

* **Extend** the dead-direction list with the W95-B0 family
  itself (if all three W99 candidates FAIL Phase 2 at 11B,
  the family is empirically capped through FOUR preflight-
  earned attempts on RealWorldQA at 11B); OR
* **Earn** B2 / B4 back into the active frontier (if any
  candidate PASSes), with B5 staying baseline-only ceiling.

| Mechanism | Evidence against | NEW W99 status |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C: 0/11 at 11B; 1/7 at 90B | UNCHANGED — refuted.  B2's final-VLM is mechanistically distinct (committed answerer, not binary). |
| **W95-B0 free-text bullet extraction as sufficient on vision-bound benches** | W97 D2-B0 11B: B − A1 = −6.67 pp | UNCHANGED — refuted on RealWorldQA yes/no perception. |
| **B1 typed-schema-with-direct_answer_hint as the sufficient typed fix** | W98 B1 11B: B − A1 = −6.67 pp via 5 multi-choice regressions | UNCHANGED — refuted; B4 is the explicit hint-removal repair. |
| **W95-B0-derived family as a whole (CONDITIONAL pending W99)** | W97 + W98 cap at −6.67 pp via two mechanisms; W99 tests B4 + B5 + B2 | TO BE RESOLVED IN W99: if all three FAIL, the family is refuted across FOUR attempts and the dead-direction list is extended. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed)

**The W97 + W98 anti-pattern list remains in force VERBATIM.**

| Anti-pattern | W99 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only. |
| Compaction / generic prose summarization as memory mechanism | UNCHANGED — anti-pattern; W97 + W98 evidence reinforces. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED. |
| Selective retries | UNCHANGED. |
| Single-seed pilots as retirement evidence | UNCHANGED — W99 cheap pilots are DISCRIMINATORS, not retirement evidence. |
| Architecture refinement by vibe | UNCHANGED — W99 B2 / B4 / B5 are arsenal-driven from per-problem failure-cluster diagnosis + W98 sidecar mining. |

## What W99 B2 / B4 / B5 are NOT

To pre-empt drift back toward commodity-LLM tricks under a new
name:

### B2 is NOT

* **Not a verifier.**  W96-C C1 was a binary agree/disagree
  mechanism that was empirically refuted.  B2's final turn is
  a **committed answerer** with full visual access; the
  decision surface is the question itself.
* **Not a generic "retry with more context" hack.**  The
  final turn invokes only on the failure cluster (where text-
  solver turns all FAILed); it cannot regress preserved
  short-circuit wins.
* **Not bounded / compaction / summary.**

### B4 is NOT

* **Not a verifier or final-turn mechanism.**  B4 has no
  final VLM call; it is byte-identical to W98 B1 modulo the
  removal of one schema field.
* **Not a new mechanism.**  B4 is a *minimal repair* of B1.
  Its hypothesis is empirically narrow: the
  ``direct_answer_hint`` was the proximate cause of B1's
  multi-choice regressions; removing it should restore
  reflexion-cycling discipline.
* **Not bounded / compaction / summary.**

### B5 is NOT

* **Not a frontier mechanism.**  B5 is a *switch baseline*
  that does not introduce a new mechanism for cross-modal
  team coordination.  It is explicitly **baseline-only**
  even if it clears Phase 2.
* **Not an oracle.**  The question-type classifier uses only
  surface regex features; it does NOT consult gold answers.
* **Not a verifier.**  B5 commits to the routed arm's output
  verbatim.
* **Not bounded / compaction / summary.**

## Honest classification of a W99 PASS by candidate

| Candidate PASSes | What we earn | What we DO NOT claim |
|---|---|---|
| Only **B5** | Per-question routing ceiling is high enough that the W95-B0-family cap is a routing problem.  B5 is a ceiling reference. | Multi-agent context superiority; structural team mechanism. |
| Only **B4** | The typed-schema-sans-hint mechanism IS load-bearing on RealWorldQA at 11B.  Schema repair earns the family back. | Multi-agent context superiority before Phase 3. |
| Only **B2** | The image-at-decision-boundary mechanism IS load-bearing on RealWorldQA at 11B.  Structural frontier earns the family back. | Multi-agent context superiority before Phase 3. |
| **B5 + (B2 or B4)** | Both a routing ceiling AND a structural mechanism; structural mechanism remains the load-bearing claim. | Multi-agent context superiority before Phase 3. |
| **All three** | Best case: the family has multiple repair paths.  B2 / B4 are structural frontier; B5 is baseline-only ceiling. | Multi-agent context superiority before Phase 3. |
| **None** | The W95-B0 family is empirically capped across FOUR preflight-earned attempts on RealWorldQA at 11B.  ``COO-9`` (second code benchmark) is promoted to lead. | Anything about the W95-B0 family being repairable on RealWorldQA without changing benchmark. |

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim any W99 candidate will beat A1 on
  RealWorldQA — no empirical evidence exists yet.
* It does NOT retire any prior carry-forward.
* It does NOT propose any new bench code beyond the
  documented B2 (re-used) + B4 + B5 modules.
* It does NOT bump ``coordpy.__version__`` or
  ``SDK_VERSION``.
* It does NOT publish to PyPI.

## Honest scope

This is a *classification supplement*, not a benchmark
result.  The W99 candidate slate (B2 + B4 + B5) is mined
from the W97 + W98 failure-cluster diagnoses; the empirical
verdict is delivered by the W99 cheap NIM pilots, which are
separate deliverables under ``docs/RUNBOOK_W99.md``.
