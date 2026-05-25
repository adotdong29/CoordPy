# W99 — RealWorldQA candidate tournament (B2 + B4 + B5) — runbook

> **Pre-commit contract for W99, locked 2026-05-25 BEFORE any
> NIM call for any W99 candidate.**
>
> W98 B1 Phase 2 at 11B FAILed by **B − A1 = −6.67 pp**
> (gates 2 / 3 / 4 FAIL) with structurally informative per-
> problem disagreement: B1 recovered 4 / 5 W97 unique-A1-
> rescues but regressed 5 D2-B0 multi-choice / numeric wins —
> identical −6.67 pp margin as W97 D2-B0 via a different per-
> problem distribution.  The W95-B0-derived "extract-then-text-
> reason" architecture family is now empirically capped on
> RealWorldQA at B − A1 ≈ −6.67 pp at 11B through TWO distinct
> mechanisms (D2-B0 free-text + D2-B1 typed schema).
>
> W99 is the **multi-candidate tournament** that decides
> whether the cap is structural at the family level or
> repairable by:
>
>   * **B2** — direct-vision final-turn answerer (image at
>     decision boundary; W98 preflight already earned this).
>   * **B4** — typed schema WITHOUT ``direct_answer_hint``
>     (strip the field that drove W98 B1's 5 multi-choice
>     regressions; keep the schema fix that recovered the 4
>     yes/no rescues).
>   * **B5** — question-type router / switch baseline
>     (route multi-choice → D2-B0; else → A1 K=5; NIM-free
>     oracle prediction = +10.00 pp on the W97 slice).
>
> **The W99 deliverable in this milestone is the NIM-free
> preflight + addressability probes for B2 / B4 / B5 + up to
> THREE cheap NIM pilots (one per surviving candidate) at 11B
> + the per-pilot Phase 2 verdict.  No Phase 3 retirement
> bench is in scope unless any candidate PASSes Phase 2 at
> BOTH 11B AND 90B per the W96-C cross-scale rule.**
>
> No version bump.  No PyPI publish.  ``coordpy.__version__``
> stays ``0.5.20``; ``coordpy.SDK_VERSION`` stays
> ``coordpy.sdk.v3.43``.

## Linear

* New issue **``COO-23``** (W99): RealWorldQA candidate
  tournament (B2 + B4 + B5; preflight + up to 3 cheap NIM
  pilots at Llama-3.2-11B-Vision; cross-scale 90B conditional
  on 11B PASS for each survivor).  Parent: ``COO-6``.  High
  priority.
* Related: ``COO-22`` (W98; closed Done).  ``COO-9`` (second
  code benchmark; promoted iff all three W99 candidates FAIL
  Phase 2).

## Arsenal-mining anchor

``docs/RESULTS_W99_ARSENAL_MINING_V1.md`` extends the W97 + W98
arsenal-mining passes with the W98 B1 sidecar-mined diagnosis:
B1's typed-solver was anchored to the reader's
``direct_answer_hint`` and stopped the K=4 reflexion-cycling
discipline.  The W99 candidate slate is:

* **B2** — ``coordpy/realworldqa_bench_v3.py`` (built in W98,
  preflight-earned in W98).  Distinct mechanism: keep image
  alive at decision boundary.
* **B4** — ``coordpy/realworldqa_bench_v4.py`` (new W99).
  Minimal repair of B1: strip ``direct_answer_hint``.
* **B5** — ``coordpy/realworldqa_bench_v5.py`` (new W99).
  Switch baseline; NIM-free oracle prediction +10.00 pp on the
  W97 slice.

## Frontier-relevance audit anchor

``docs/FRONTIER_RELEVANCE_AUDIT_W99_V1.md`` extends the W98
supplement.  All W97 + W98 classifications remain in force.
W99 additions:

* **Active frontier arsenal**: B2 / B4 / B5 (preflight-
  earned).  B2 is the **structural frontier lead** per the
  user's W99 brief ("the lead because it removes reader-hint
  dependency and keeps the image alive at the decision
  boundary").
* **Useful baseline-only**: B5 stays as a *switch ceiling /
  floor reference*.  Bounded-window baselines unchanged.
* **Anti-patterns**: bounded / compaction / prose-summary
  remain explicit anti-patterns.  No W99 candidate is any
  of those.

## Hypotheses (locked 2026-05-25)

### B2 (direct-vision final-turn answerer)

W97 unique-A1-rescues (5 / 30) are the cluster where unified
VLM K=5 wins by re-seeing the image.  B2's final turn gives
that cluster the same image access A1 has, plus structured
extraction context.  Short-circuit on text-solver PASS
preserves D2-B0's 22 / 30 + 3 / 30 wins.

NIM-free upper bound (W97 confusion table on 30 problems):
* Both-pass (22) → B2 PASS via short-circuit.
* Unique-B-rescues (3) → B2 PASS via text-solver in K=3 turns
  (W97 sidecars confirm first-PASS-at-≤2 on all 3).
* Unique-A1-rescues (5) → final-VLM invoked with full image
  access.  Expected rescue rate ~ 80–100 % (≈ A1).
* Neither (0).

Realistic prediction: 22 + 3 + 4 = 29 / 30 = 96.67 %; B2 − A1
= +6.67 pp ≥ +5 pp ✓ (best 100 %; conservative 93.33 %).

### B4 (typed schema WITHOUT direct_answer_hint)

W98 B1 demonstrated the schema primitives (state /
orientation / depth / text_in_object) recovered 4 / 5 of the
W97 unique-A1-rescues.  The proximate cause of the 5 multi-
choice regressions was the typed-solver anchoring on the
reader's ``direct_answer_hint``.  B4 removes the hint.

Reasoning prediction (NIM-free): the 4 yes/no rescues survive
(they depend on schema primitives, not the hint).  The 5
multi-choice regressions are addressed because the solver
relies on JSON primitives + question text instead of a
possibly-wrong hint, restoring reflexion-cycling discipline.
Subjective probability of clearing +5 pp: ~ 35–50 % (higher
than B1).

### B5 (switch baseline, question-type router)

NIM-free ORACLE on the W97 slice: route multi-choice (18
problems) → W97 D2-B0 (which PASSed 18 / 18 multi-choice);
route yes/no + numeric + short_text (12 problems) → W97 A1
K=5 (which PASSed 12 / 12).  **Predicted B5 pass rate = 30 /
30 = 100.00 %; B5 − A1 = +10.00 pp.**

Live NIM variance assumption: B5 ≈ 90–100 % at 11B.

### Pre-pilot prediction (locked 2026-05-25 BEFORE NIM)

> "Subjective priors over the W99 cheap pilots:
>
> * Probability B5 clears +5 pp Phase 2 at 11B: ~ 70–85 %
>   (highest NIM-free oracle prediction; only risk is NIM
>   sampling variance and parser mis-classification of a single
>   short-text question — 29 / 30 = 96.7 % parser accuracy in
>   the W98 AddrP6 probe).
> * Probability B2 clears +5 pp Phase 2 at 11B: ~ 35–50 %
>   (NIM-free realistic 96.67 %, but A1 may not saturate and
>   final-VLM rescue prior is structurally sound but
>   empirically uncertain).
> * Probability B4 clears +5 pp Phase 2 at 11B: ~ 25–40 %
>   (no NIM-free oracle; reasoning-only prediction).
> * Expected milestone outcome:
>     - At least one candidate clears +5 pp at 11B with
>       probability ~ 80–90 % (mostly from B5).
>     - If B5 PASSes by +10 pp but B2 and B4 fail: this
>       proves the per-question ceiling is high enough that
>       the W95-B0-family cap is a *routing* problem — NOT
>       evidence of structural team superiority.  B5 is a
>       baseline-only ceiling reference, not a frontier
>       retirement.
>     - If B2 also PASSes (a frontier mechanism win): that
>       earns the W95-B0 family back into the active arsenal
>       with a structural fix.  Cross-scale 90B Phase 2 would
>       then be entitled.
>     - If only B4 PASSes: that earns the *typed schema sans
>       hint* mechanism as the simplest frontier fix.
>     - If ALL THREE fail Phase 2: the W95-B0 family is
>       empirically capped through FOUR attempts on RealWorldQA
>       at 11B and COO-9 (second code benchmark) is promoted
>       as the next lead path."

## Baselines (locked 2026-05-25)

Identical W95-shape on RealWorldQA:

* **A0** — text-only mode (image=None) of the VLM at T=0.0, K=1.
* **A1** — unified VLM mode at T=0.7, K=5.
* **B2** — ``coordpy.realworldqa_bench_v3.run_realworldqa_bench_v3``:
  scene reader (T=0.0, 1 VLM call) + text solver (T=0.7, 3
  text calls / reflexion) + final-VLM answerer (T=0.0; runs
  only on text-solver FAIL).  K=5 byte-exact (padding on
  short-circuit).
* **B4** — ``coordpy.realworldqa_bench_v4.run_realworldqa_bench_v4``:
  typed scene-graph reader (T=0.0, 1 VLM call; schema sans
  ``direct_answer_hint``) + question-typed text solver (T=0.7,
  4 text calls / reflexion).  K=5 byte-exact.
* **B5** — ``coordpy.realworldqa_bench_v5.run_realworldqa_bench_v5``:
  deterministic route by ``detect_question_type_v2``; route
  multi-choice → W97 D2-B0 (K=5); else → A1 K=5.  K=5 byte-
  exact on either route.

Same VLM model on A0 / A1 / every B-route.  Same K=5 budget
on A1 / every B.  Same executor
(``evaluate_realworldqa_answer_v1``) on every arm.

## W99 NIM-free preflight + addressability probes

Layered ON TOP of the W93 / W96-D composite (which already
PASSED for D2 in W96-D at both 11B and 90B):

| Probe | Candidate | Threshold | What it does |
|---|---|---|---|
| **AddrW99-B2-P1 — NIM-free upper bound from W97 confusion table** | B2 | realistic predicted PASS rate ≥ A1 + 5 pp | Mines W97 per_problem.jsonl; counts both-pass / unique-B / unique-A1 / neither cells; predicts B2 PASS rate under best / realistic (80 % final-VLM rescue) / conservative (50 %) cases. |
| **AddrW99-B2-P2 — short-circuit static** | B2 | static code audit | Confirms ``_run_b_direct_vision_final`` uses first-PASS short-circuit + padding + final-VLM. |
| **AddrW99-B2-P3 — final-VLM rescue prior** | B2 | ≥ 3 unique-A1-rescues exist | Definitionally: A1 wins all unique-A1-rescues; B2 final-VLM has equivalent visual access. |
| **AddrW99-B2-P4 — budget exact** | B2 | K=5 | Static config audit. |
| **AddrW99-B4-P1 — schema primitives retained** | B4 | state + orientation + depth + text_in_object all present | Static schema audit of the reader prompt. |
| **AddrW99-B4-P2 — hint field removed** | B4 | ``direct_answer_hint`` not referenced by the solver template; reader prompt mentions it ≤ 1× (only the explicit removal admonition) | Static audit. |
| **AddrW99-B4-P3 — budget exact** | B4 | K=5 | Static config audit. |
| **AddrW99-B5-P1 — ORACLE simulation on W97 sidecars** | B5 | predicted B5 PASS rate − A1 rate ≥ +5 pp | Computes exact predicted outcome by routing each W97 slice problem and reading the corresponding W97 D2-B0 / A1 per-problem outcome. |
| **AddrW99-B5-P2 — parser correctness** | B5 | ≥ 90 % on W97 slice | Re-runs the W98 AddrP6 parser correctness probe. |
| **AddrW99-B5-P3 — budget exact** | B5 | K=5 on either route | Static config audit. |

## Pre-committed Phase 2 pilot gates (only if cheap probes earn a NIM pilot)

If a W99 candidate survives all NIM-free probes, the W99
milestone runs its 1-seed × 30-problem × K=5 cheap NIM pilot
at 11B under THE SAME 9 pre-committed Phase 2 gates as W97 /
W98 (byte-identical gate texts; only the candidate arm name
and slice seed change):

1. **Slice pre-committed**: 30 problems by deterministic
   slice with **seed 96_504_002** BEFORE any NIM call.  Slice
   SHA recorded.  (Same slice as W97 / W98 — anti-cheat
   parity.)
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice
   must stay below 90 %.  (Will likely FAIL again on
   96_504_002; honest acknowledgement below.)
3. **B > A1**: ``b_pass_rate > a1_pass_rate``.
4. **Margin ≥ +5 pp**: ``b_pass_rate − a1_pass_rate ≥ 5 pp``.
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice
   problems at end-of-run → 100 % pass.

### Honest acknowledgement of the slice-saturation risk

The 96_504_002 slice saturated A1@K=5 at exactly 90.00 % in
W97; W98 measured it at 86.67 % under fresh sampling.  Gate 2
may FAIL on each W99 candidate.  Per ``docs/RUNBOOK_W98.md``
**Option A** (which carries forward to W99 verbatim): re-run
on the same slice for direct cross-candidate comparison;
treat ``(B − A1)`` as the architecturally-relevant
discriminator regardless of whether A1 is at saturation.  If
gate 2 fails AND B − A1 > +5 pp AND per-problem majority ≥
16 / 30, the verdict is "**STRUCTURALLY POSITIVE despite
slice-saturation artefact**" and licenses Option B (new
slice) as a W100 follow-up.

## Cross-scale rule (W96-C carry-over, locked 2026-05-25)

Identical to W97 / W98:

* **Cross-scale 90B Phase 2 entitled** IFF 11B Phase 2 PASS
  (all 9 gates) — OR with written justification if 11B FAILS
  by a narrow margin (gate 2 alone on a saturated slice but
  B − A1 > +5 pp).
* **Phase 3 entitled** IFF Phase 2 PASS at BOTH 11B AND 90B,
  OR Phase 2 PASS at 90B alone with written justification.
* A Phase 2 PASS at 11B alone is NOT sufficient for Phase 3.
* A Phase 2 FAIL at 11B does NOT auto-launch a 90B Phase 2.

## Phase 3 — full bench (NOT in this milestone)

No Phase 3 retirement bench is launched in W99.  Phase 3
would require its own runbook section locked BEFORE any
Phase 3 NIM call, mirroring W96-A.

## Cross-candidate decision logic (PRE-LOCKED)

**Phase 0 — NIM-free preflight + addressability probes.**
Run the W96-D D2 composite + AddrW99-Bx-* probes for all
three candidates.  Any candidate failing any probe is
**KILLED** and documented.

**Phase 1 — Cheap NIM pilots (one per survivor).**  Up to
three cheap NIM pilots may be launched; each pilot is a
1-seed × 30-problem × K=5 run at 11B.  Promotion order is
by NIM-free expected lift (descending): B5 → B2 → B4.

If at least one candidate PASSes Phase 2 at 11B: that
candidate is entitled to a cross-scale 90B Phase 2 in a
follow-up milestone.

If NO candidate PASSes Phase 2 at 11B: per Part G of the W99
brief, document the kills, sync Linear, promote ``COO-9``
(second code benchmark) as the next lead path, and explicitly
note the W95-B0-derived family is now empirically capped
across FOUR preflight-earned attempts (D2-B0 / D2-B1 / B4 /
B5) at 11B on RealWorldQA.

### What B5 PASSING means honestly

B5 is a SWITCH baseline.  If B5 PASSes Phase 2:

* This is NOT evidence of structural team superiority.
* It IS evidence that the per-question ceiling is high
  enough that the W95-B0-family cap is a *routing* problem
  at the per-question level.
* B5 stays classified as **baseline-only ceiling / floor
  reference** in the W99 frontier audit — NOT promoted to
  active frontier arsenal.
* B5 PASS does NOT retire any prior carry-forward.

### What B2 PASSING means honestly

B2 is a **structural frontier mechanism** (image at decision
boundary; no reader-hint dependency).  If B2 PASSes Phase 2:

* This earns the W95-B0 family back into the active
  arsenal with a structural fix.
* Cross-scale 90B Phase 2 is entitled.
* If 90B also PASSes, the next milestone considers Phase 3
  full bench (3 seeds × 100 problems).
* B2 PASS would license a frontier-claim about direct
  visual grounding at the decision boundary — but multi-
  agent context superiority remains UN-CLAIMED unless the
  Phase 3 bar is also cleared.

### What B4 PASSING means honestly

B4 is a **minimal repair** of W98 B1.  If B4 PASSes:

* This earns the *typed-schema-sans-hint* mechanism as the
  simplest frontier fix.
* Cross-scale 90B entitled.
* Same Phase 3 entitlement rule.

## NIM smoke test (mandatory before any pilot)

Each candidate's first pilot launch is preceded by a 1-token
POST returning HTTP 200 + a non-empty completion.  No separate
smoke run dir is required (the W98 smoke at 11B already
validated the corpus + NIM HTTPS path).

## Anti-cheat (carry-forward from W88–W98)

All W88–W98 anti-cheat clauses carry forward verbatim:

* Slice = deterministic ``select_realworldqa_subset_v1`` with
  seed ``96_504_002``; SHA-anchored at run start.
* Same VLM model on every arm.
* Same K=5 byte-exact budget on A1 / every B.
* Executor = ``evaluate_realworldqa_answer_v1``.  No LLM
  judge.
* Parquet shards SHA-anchored at pilot start; mismatches
  refuse to run.
* No selective retries.
* Per-call sidecars (``text_calls.jsonl``,
  ``vlm_calls.jsonl``, ``per_problem.jsonl``) + per-seed
  Merkle + bench Merkle.

## Stable boundary preservation

* ``coordpy.__version__`` unchanged at ``0.5.20``.
* ``coordpy.SDK_VERSION`` unchanged at ``coordpy.sdk.v3.43``.
* No PyPI publish.
* ``coordpy/__init__.py`` untouched.
* New modules ``coordpy.realworldqa_bench_v4`` and
  ``coordpy.realworldqa_bench_v5`` are **explicit-import
  only**.
* No pre-existing W95 / W96 / W97 / W98 module is modified.

## Operational plan

1. **(W99 arsenal mining + frontier-relevance audit
   supplement)** — DONE; see
   ``docs/RESULTS_W99_ARSENAL_MINING_V1.md`` +
   ``docs/FRONTIER_RELEVANCE_AUDIT_W99_V1.md``.
2. **(W99 B4 + B5 bench modules + tests)** —
   ``coordpy/realworldqa_bench_v4.py`` +
   ``coordpy/realworldqa_bench_v5.py`` +
   ``tests/test_w99_realworldqa_bench_v4_v5.py``.  B2 already
   exists at ``coordpy/realworldqa_bench_v3.py`` (built in
   W98).
3. **(W99 NIM-free preflight + addressability probes)** —
   ``scripts/run_w99_realworldqa_preflight.py``.  Output:
   ``results/w99/realworldqa_preflight_b2_b4_b5/<RUN_ID>/``.
4. **(W99 chosen-winners Phase 2 at 11B — conditional)** —
   ``scripts/run_w99_realworldqa_pilot.py --candidate {B2,B4,B5}``.
   Output: ``results/w99/realworldqa_pilot/<RUN_ID>/``.  One
   subdirectory per candidate.
5. **(W99 cross-scale Phase 2 at 90B — conditional)** —
   only for candidates that PASS 11B Phase 2.
6. **(W99 Phase 3 — OUT OF SCOPE)**
7. **(Linear ↔ GitHub sync)**
   a. Create ``COO-23`` (W99) under ``COO-6``.
   b. Update ``COO-6`` summary with W99 verdict.
   c. Append a ``W99`` entry to
      ``linear_github_mapping.json`` and run
      ``scripts/sync_linear_github_v1.py validate``.
   d. If ALL three candidates FAIL Phase 2, promote
      ``COO-9`` (second code benchmark) to lead and update
      its priority.

## Honest framing

W99's job is to **earn or kill the W95-B0 family on
RealWorldQA by running THREE arsenal-driven cheap
discriminators**.  No expensive Phase 3 retirement bench is
in scope.  No bounded / compaction / summary mechanism
appears in any candidate.  Either Phase 2 earns at least one
W95-B0-family candidate or kills all three — either way the
W93–W98 discipline is preserved as the 9th consecutive
validation.

If all three candidates FAIL Phase 2 at 11B, the milestone
verdict is "RealWorldQA empirically capped across FOUR
preflight-earned attempts on the W95-B0 family; COO-9
promoted to lead".  That is itself an honest research result.
