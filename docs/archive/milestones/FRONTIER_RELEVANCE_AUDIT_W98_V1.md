# W98 — Frontier relevance audit V1 (supplement to W97 V1)

> 2026-05-25.  Supplement to
> `docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md`.  Extends the W97
> classification with the W98 candidates **B1 (typed scene-graph
> extraction + question-typed solver)** and **B2 (direct-vision
> final-turn answerer)** based on the W97 cheap pilot's
> empirical failure-cluster diagnosis
> (`docs/RESULTS_W98_ARSENAL_MINING_V1.md`).  The W97 audit's
> classifications all remain in force; this supplement only
> ADDS to the active-frontier-arsenal column and re-asserts the
> anti-pattern column verbatim.
>
> No code is removed by this audit.  No version bump.  No PyPI
> publish.

## Why a supplement, not a rewrite

W97 V1 documented the classification *as of the W97 D2-B0
preflight-earned moment*.  The W97 D2-B0 11B Phase 2 cheap
pilot has since FAILed (B − A1 = −6.67 pp; gates 2/3/4 FAIL)
and the per-problem failure-cluster mining has identified two
distinct root-cause classes:

1. **Output-format mismatch** on yes/no questions (3 / 5
   unique-A1-rescues).
2. **Lossy extraction** on state primitives + depth ordering
   (2 / 5 unique-A1-rescues).

The W97 audit correctly predicted that "the W95-B0 shape
discards the image after the reader call" would be the
structural risk.  The W97 pilot bore this out; the W98
candidates B1 and B2 are the structurally-motivated
responses.

This supplement formalises:

* the addition of B1 and B2 to the active-frontier-arsenal
  column;
* the re-statement of the W97 V1 anti-pattern column
  *verbatim* — bounded / compaction / summary remain explicit
  anti-patterns, **and the failure of D2-B0's free-text
  bullet extraction is direct evidence that prose-summary as
  a memory mechanism is empirically refuted on RealWorldQA
  yes/no perception**;
* the explicit demotion of the "verifier as load-bearing
  rescue" idea (W96-C C1) to refuted-status across both
  MathVista AND RealWorldQA;
* the explicit reaffirmation that no W98 candidate is
  bounded-window / compaction / summary or any disguised
  variant thereof.

## Additions to the active frontier arsenal (W98)

| Mechanism | Module(s) | Why still frontier (with W97 evidence) |
|---|---|---|
| **B1 — Typed scene-graph extraction + question-typed solver** | NEW `coordpy/realworldqa_bench_v2.py` (W98) | Addresses *all 5* W97 unique-A1-rescue root causes: question-typed solver prompt fixes the 3 output-format-mismatch failures (extractions already contain the answer); typed schema with `state` / `orientation` / `depth` primitives fixes the 2 lossy-extraction failures.  Preserves D2-B0's 22 / 30 + 3 / 30 wins via first-PASS short-circuit on text-solver turns. |
| **B2 — Direct-vision final-turn answerer** | NEW `coordpy/realworldqa_bench_v3.py` (W98) | Structurally distinct mechanism: keeps the image alive at the decision boundary on the failure cluster.  Mechanistically different from W96-C C1 verifier (committed answerer, not binary agree/disagree).  Short-circuits on text-solver PASS so multi-choice wins are preserved.  Direct visual grounding across turns — exactly the "what we DO want" category. |
| **W98 cross-candidate cheap-discriminator rule** | New (this milestone): `docs/RUNBOOK_W98.md` | Two-phase discipline: NIM-free preflight + addressability probes (AddrP1–AddrP5) → promote at most ONE winner to a NIM pilot.  Prevents the W98 milestone from scattering across multiple half-funded candidates. |
| **W97 failure-cluster diagnosis** | `coordpy.failure_cluster_miner_v1` + W97 sidecars | Validated as a load-bearing tool: the W98 candidate slate is mined directly from the W97 sidecars, not from the W97 doc's executive summary alone.  This is the discipline pattern the user's brief explicitly demanded ("Do NOT leave useful prior work unused"). |

## Useful baselines (unchanged from W97 V1)

The W97 V1 classification of `bounded_window_baseline_v{1,2,3}`,
A0 / A1 baselines, and generic chat-completions text driver as
*useful baseline-only* remains in force verbatim.

## Historical artifacts (unchanged from W97 V1)

The W97 V1 classification of W90 / W92 / W88 cross-modal benches,
attention-steering V1–V13, W81 adversarial repair, W83 long-
horizon reconstruction substrate, and W84 tool-call substrate as
*historical artifacts (kept for regression / audit; not active
path)* remains in force verbatim.

## Dead directions (refuted by evidence; do not entertain as core strategy)

The W97 V1 list remains in force verbatim, with one
clarification added by W97 evidence:

| Mechanism | Evidence against | NEW W98 clarification |
|---|---|---|
| **VLM-Verifier-Final-Turn as load-bearing rescue** | W96-C: rescue 0/11 at 11B; 1/7 at 90B = not load-bearing on MathVista | W97 failure cluster is yes/no perception + lossy state primitives, *not* a "rescue the failing answer" failure mode.  Re-porting C4 (W96-C C1) to RealWorldQA would address the wrong failure mode.  **Re-affirmed refuted; W98 B2 is NOT a verifier — it is a committed answerer with image access** (see below). |
| **W95-B0 free-text bullet extraction as a sufficient extraction shape on vision-bound benches** | W97 D2-B0 Phase 2 11B: B − A1 = −6.67 pp; 2 / 5 unique-A1-rescues caused by lossy free-text extraction (state primitives + depth ordering missing) | **NEW W98 carry-forward implication**: free-text bullet extraction is empirically insufficient on RealWorldQA's vision-bound yes/no cluster.  Schema-constrained extraction (W98 B1) is the structurally-motivated response. |
| **B-solver system prompt that biases yes/no toward numeric output** | W97 D2-B0 11B: 3 / 5 unique-A1-rescues are output-format mismatches — extractions contain the right answer in prose but the solver returns numbers | **NEW W98 carry-forward implication**: typed solver prompts that include question_type are required on benches where the answer space spans yes/no AND numeric AND multi-choice-letter.  W98 B1 implements this. |

## Anti-patterns (NEVER promote as core strategy; baseline-only allowed)

**The W97 V1 anti-pattern list remains in force VERBATIM.**
Re-stated here for emphasis because the W97 D2-B0 cheap pilot's
free-text extraction is itself a *prose-summary mechanism* and it
FAILED on yes/no perception — direct empirical evidence that
prose summary as a memory/extraction mechanism is not the
frontier path on this battlefield.

| Anti-pattern | W98 status |
|---|---|
| Bounded context window as product thesis | UNCHANGED — anti-pattern; baseline-only. |
| Compaction / generic prose summarization as memory mechanism | **STRENGTHENED** — W97 D2-B0's free-text bullet extraction (a prose-summary) FAILED at recovering yes/no state primitives.  Empirical evidence reinforces the W97 V1 anti-pattern classification. |
| Shallow token compression without structural reason | UNCHANGED. |
| Context-pruning theater | UNCHANGED. |
| "Cram less / truncate better" as frontier memory system | UNCHANGED. |
| LLM-as-judge in executor chain | UNCHANGED. |
| Selective retries | UNCHANGED. |
| Single-seed pilots as retirement evidence | UNCHANGED. |
| Architecture refinement by vibe | UNCHANGED — W98 B1 + B2 are arsenal-driven from per-problem failure-cluster diagnosis. |

## What W98 B1 and B2 are NOT

To pre-empt any drift back toward commodity-LLM tricks under a
new name, the audit specifies what B1 and B2 are categorically
NOT:

* **B1 is NOT a compaction / token-compression mechanism.**  Its
  goal is not to reduce token count; its goal is to *force
  lossless extraction* of the spatial primitives that the W97
  failure cluster requires.  The JSON schema is allowed to be
  *longer* in tokens than the W97 free-text bullet list when
  the schema captures more required primitives.
* **B1 is NOT a generic prose-summary "memory".**  It is a
  typed, schema-constrained snapshot of the image's spatial
  primitives, with explicit existence / state / orientation /
  depth fields.  Schema is content-addressed; not a free-form
  paragraph.
* **B1 is NOT a bounded-context window.**  No tokens are
  pruned from the input; the bench's K=5 budget is unchanged.
* **B2 is NOT a verifier.**  W96-C C1 was a binary agree/
  disagree mechanism on a prior candidate.  W98 B2's final
  turn is a *committed answerer* with full visual access; the
  decision surface is the original question, not a meta-
  decision about a prior answer.
* **B2 is NOT a generic "retry with more context" hack.**  The
  final turn invokes only on the failure cluster (where all
  text-solver turns have FAILed); it cannot regress the
  preserved short-circuit wins.

## Specific Linear / repo mirror updates

* `COO-6` (parent backlog) — needs a new W98 verdict comment
  after the W98 milestone closes; this audit is referenced by
  that comment.
* `COO-21` (W97-A) — closed Done in W97; W98 is a separate
  child issue.
* **NEW** `COO-22` (W98) — to be created in this milestone;
  scoped as "W98 RealWorldQA multi-candidate slate (B1 + B2
  preflight + at most one cheap NIM pilot)".
* `COO-12` (substrate-level cross-modal injection) — UNCHANGED;
  still the *hard alternative* (Low priority).
* `COO-9` (second code benchmark) — UNCHANGED; still High but
  blocked behind W98 cross-modal lead.  If both W98 candidates
  die in preflight, `COO-9` is the pivot per Part H of the
  W98 brief.

## What this supplement DOES NOT do

* It does NOT claim multi-agent context is solved.
* It does NOT claim B1 or B2 will beat A1 on RealWorldQA — no
  empirical evidence exists yet.
* It does NOT retire any prior carry-forward.
* It does NOT propose any new bench code beyond the documented
  B1 + B2 modules (separate deliverables in the same
  milestone).
* It does NOT bump `coordpy.__version__` or `SDK_VERSION`.
* It does NOT publish to PyPI.

## Honest scope

This is a *classification supplement*, not a benchmark result.
The W98 candidate slate (B1 + B2) is mined from the W97
failure-cluster diagnosis; it does not yet have any empirical
evidence beyond the W97 V1 audit's already-documented active-
frontier-arsenal status.  The W98 milestone's deliverable is
to RUN the preflight + addressability probes + (at most one)
cheap NIM pilot — those are separate deliverables under
`docs/RUNBOOK_W98.md`.
