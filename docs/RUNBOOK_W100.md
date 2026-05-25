# W100 — RealWorldQA cross-scale 90B confirmation (B2 frontier lead + B5 ceiling reference) — runbook

> **Pre-commit contract for W100, locked 2026-05-25 BEFORE any
> W100 NIM call.**
>
> W99 closed with TWO `STRUCTURALLY_POSITIVE_SLICE_SATURATION_CAP`
> verdicts at 11B on the W96-D-earned 96_504_002 / 30-problem
> slice of `lmms-lab/RealWorldQA` test:
>
>   * **B2 (direct-vision final-turn answerer)** — A0 = 36.67 %,
>     A1 @ K=5 = 93.33 %, **B2 = 100.00 %**, B2 − A1 = **+6.67 pp**,
>     8 / 9 gates PASS (gate 2 only fail; per-problem 30 / 30;
>     final-VLM 3 / 3 rescue).  **Structural frontier lead.**
>   * **B5 (deterministic NIM-free question-type router /
>     switch baseline)** — A0 = 36.67 %, A1 @ K=5 = 93.33 %,
>     **B5 = 100.00 %**, B5 − A1 = **+6.67 pp**, 8 / 9 gates
>     PASS (gate 2 only fail; per-problem 30 / 30; oracle 30 / 30
>     match).  **Baseline-only ceiling reference.**
>   * B4 (typed schema sans `direct_answer_hint`) — FAIL by
>     −16.67 pp; refuted the hint-removal hypothesis; the
>     typed-extract-then-text-reason sub-family is empirically
>     capped through THREE distinct mechanisms (D2-B0 + B1 + B4).
>     **B4 is DEAD.  Do not re-open in W100.**
>
> W99's strongest structural truth is that the W95-B0 family is
> repairable IF the image stays alive at the decision boundary
> (B2 mechanism).  The empirical question W100 must answer is:
>
> **Does B2 survive at 90B when A1 gets stronger, and does the
> image-at-decision-boundary mechanism remain load-bearing when
> the unified VLM baseline rises from 93.33 % to (estimated)
> ~ 79.49 % A1 @ K=5 with a much higher per-call ceiling?**
>
> W100 is a **cross-scale confirmation**, not a new candidate
> tournament.  The candidate slate is FROZEN by W99:
>
>   * **B2** = frontier lead.  90B Phase 2 entitled by W99 with
>     written justification (Option C of the cross-scale rule:
>     B − A1 = +6.67 pp at 11B with per-problem 30 / 30 satisfies
>     the structural-positive ladder).
>   * **B5** = ceiling reference.  Same 90B Phase 2 entitled
>     under Option C.  B5 stays classified baseline-only in the
>     frontier audit.  A B5 PASS at 90B does NOT substitute for
>     the B2 question.
>   * **B4** = dead.  Not in scope.
>   * **typed-extract-then-text-reason sub-family** = dead.
>     Not in scope.
>   * **Any new candidate** = NOT in scope.  The tournament is
>     over; W99 selected the winner.
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* New issue **`COO-24`** (W100): RealWorldQA cross-scale 90B
  Phase 2 confirmation (B2 frontier lead + B5 ceiling reference;
  Phase 3 conditional on B2 90B PASS).  Parent: `COO-6`.  High
  priority.
* Related: `COO-23` (W99; closed Done).  `COO-9` (second code
  benchmark; promoted to lead path IFF B2 collapses at 90B).

## What is NOT in scope (anti-drift contract)

This milestone explicitly does NOT:

1. Re-open B4 or any typed-extract-then-text-reason variant.
2. Invent new W100-only candidates.  No B6, B7, B8 etc.
3. Pivot away from B2 unless B2 collapses at 90B (in which
   case the W100 verdict promotes `COO-9` immediately per Part H
   of the user's W100 brief).
4. Treat a B5 PASS as substitute evidence for the B2 question.
5. Launch Phase 3 on momentum.  Phase 3 entitlement is governed
   by the rule below; if entitled, W100 either pre-commits the
   Phase 3 runbook explicitly OR explicitly defers Phase 3 to
   W101.
6. Re-introduce any anti-pattern under a prettier name (bounded
   windowing; compaction; generic prose summarization; shallow
   token compression; context-pruning theater; "cram less /
   truncate better").  W97 + W98 + W99 frontier-relevance audits
   remain in force verbatim; W100 carries them forward.

## Arsenal-mining anchor

The W99 arsenal mining (`docs/RESULTS_W99_ARSENAL_MINING_V1.md`)
remains the canonical mining record.  W100 does NOT mine new
arsenal at 11B — the W99 mining produced the slate, W99 selected
the winner, W100 confirms the winner at 90B.  If post-W100
mining is warranted (only if B2 90B FAILs and `COO-9` needs an
expanded surface), it lands in W101 under its own runbook.

## Frontier-relevance audit anchor

`docs/FRONTIER_RELEVANCE_AUDIT_W100_V1.md` is the third
supplement (after W97 / W98 / W99).  All W97 / W98 / W99
classifications remain in force.  W100-specific additions:

* **B2 (direct-vision final-turn answerer)** stays *active
  frontier lead*; this milestone is the cross-scale confirmation
  test, not a re-classification.
* **B5 (question-type router)** stays *baseline-only ceiling
  reference*.  A B5 PASS at 90B EXPANDS the routing-ceiling
  evidence; it does NOT promote B5 to frontier.
* **B4 + the typed-extract-then-text-reason sub-family** remain
  *dead*.  Do not entertain.
* **W99 multi-candidate tournament discipline** remains an
  active frontier *operating-system* piece.
* **All W97 / W98 / W99 anti-patterns** remain anti-patterns
  verbatim.

## Hypotheses (locked 2026-05-25)

### B2 at 90B (lead)

**Claim**: the image-at-decision-boundary mechanism remains
load-bearing at 90B.  Specifically, on the same 96_504_002 /
30-problem slice, with `meta/llama-3.2-90b-vision-instruct` as
the candidate VLM:

* **A0 (text-only mode of the 90B VLM, T=0.0, K=1)** is
  expected at roughly the same floor as 11B (~ 35-40 %), since
  the text-only floor is dominated by the modality-bound
  problems where text-only must guess.
* **A1 @ K=5** is expected near **79.49 %** per the W96-D
  preflight residual estimate at 90B (residual = 20.51 pp).
  Almost certainly not saturated (gate 2 should PASS cleanly).
* **B2** is expected to outperform A1 by **at least +5 pp**
  if the mechanism is load-bearing at 90B.  Conservatively,
  B2 ≈ 90-100 % depending on how many text-solver chains
  short-circuit before the final VLM rescue is needed.
* If B2 − A1 < +5 pp at 90B, the mechanism does NOT generalize
  cross-scale and the W95-B0-family repair claim is restricted
  to 11B only.

**Cross-scale-collapse precedent**: W96-C C1 verifier showed a
+13 pp single-seed win at 90B that was variance-driven (only
1 / 7 verifier rescues).  W100 must guard against the same
artefact: B2's PASS at 90B is only frontier-relevant if the
**final-VLM rescue rate stays load-bearing** (≥ 1 / 3 rescues
when invoked, mirroring the 11B 3 / 3 rate within sampling
variance, and final-VLM invocation rate ≤ 30 % of problems).

### B5 at 90B (ceiling reference)

**Claim**: the per-question routing ceiling continues to
support a near-perfect pass rate when each routed arm gets
stronger.  Specifically:

* **Route distribution** is identical to W99 (parser is
  deterministic and NIM-free): 18 multi-choice → D2-B0 at 90B;
  12 yes_no + numeric + short_text → A1 K=5 at 90B.
* **Per-route pass rates** are expected to rise relative to 11B
  because both D2-B0 and A1 are stronger at 90B (W96-D 90B
  preflight residual 20.51 pp vs 26.56 pp at 11B).
* **B5** ≈ 90-100 %, B5 − A1 ≈ +5 to +15 pp at 90B.
* If B5 − A1 < +5 pp at 90B with A1 not saturated, the
  routing ceiling itself is regime-bound and B5's 11B PASS is
  slice-specific.

**Honesty constraint**: even if B5 PASSes structurally at 90B,
B5 stays baseline-only.  This milestone does NOT re-classify
B5.  A B5 PASS is a useful *upper-bound* on what routing alone
can achieve; it is NOT a frontier substantive claim.

### Pre-pilot prediction (locked 2026-05-25 BEFORE any W100 NIM call)

> "Subjective priors over the W100 cheap pilots at 90B:
>
> * Probability B2 clears +5 pp Phase 2 at 90B: ~ 55-70 %
>   (NIM-free realistic +6.67 pp from W99; A1 likely stronger
>   so the slice-saturation cap may not apply, which is GOOD for
>   gate 2 and AMBIGUOUS for the B − A1 margin — A1 could rise
>   to the point where text-solver short-circuit covers more of
>   the slice and the final-VLM gets fewer rescue opportunities,
>   but the unique-A1-rescue cluster shrinks too).
> * Probability B5 clears +5 pp Phase 2 at 90B: ~ 65-80 %
>   (deterministic routing; both arms stronger; oracle
>   prediction remains structurally sound but A1's per-question
>   ceiling is the load-bearing assumption — if A1's yes_no
>   pass rate rises above the 12 / 12 floor at 90B, B5 has no
>   slack vs A1 K=5 directly).
> * Probability BOTH B2 and B5 clear +5 pp at 90B: ~ 40-55 %.
> * Probability NEITHER clears at 90B: ~ 10-20 %.
> * Expected milestone outcome:
>     - If B2 clears at 90B: the W95-B0-family-via-B2 frontier
>       claim survives cross-scale.  Phase 3 entitlement decision
>       is then explicit (either pre-commit Phase 3 in W100 OR
>       defer to W101).
>     - If only B5 clears at 90B: the routing-ceiling result is
>       confirmed cross-scale, but the structural frontier claim
>       does NOT generalize.  W100 carries forward a more limited
>       B2 carry-forward (11B-only) and `COO-9` is promoted
>       because the cross-modal arc is then structurally
>       restricted to the 11B regime.
>     - If NEITHER clears at 90B: the W95-B0 family is
>       empirically capped across BOTH scales; B2's 11B PASS is
>       slice-saturation artefact; `COO-9` is the lead path.
>     - If BOTH clear at 90B: pre-commit Phase 3 for B2 or defer
>       to W101 (explicit choice, not inertia)."

## Baselines (locked 2026-05-25)

Identical W95-shape on RealWorldQA, same slice, same K, same
sampling, same executor:

* **A0** — text-only mode of the 90B VLM at T=0.0, K=1.
* **A1** — unified 90B VLM at T=0.7, K=5.
* **B2** — `coordpy.realworldqa_bench_v3.run_realworldqa_bench_v3`
  (built W98; UNCHANGED): scene reader (T=0.0, 1 VLM call) +
  text solver (T=0.7, 3 text calls / reflexion on the same VLM
  in text-only mode) + final-VLM answerer (T=0.0; runs only
  on text-solver FAIL).  K=5 byte-exact (padding on
  short-circuit).
* **B5** — `coordpy.realworldqa_bench_v5.run_realworldqa_bench_v5`
  (built W99; UNCHANGED): deterministic route by
  `detect_question_type_v2`; route multi-choice → W97 D2-B0
  (K=5); else → A1 K=5.  K=5 byte-exact on either route.

Same VLM model on A0 / A1 / B2 / B5.  Same K=5 budget on A1 /
B2 / B5.  Same executor (`evaluate_realworldqa_answer_v1`).
No new bench module is built for W100.  No B4 module is invoked.

## W100 NIM-free preflight + addressability probes

**The W99 90B preflight already PASSed for B2 + B5 at $0 NIM
spend** (verdict cid
`0bacd989850008b5416eaa6c0f4b1bacd88968a03f53d9853f5c2c454af6e907`):

| Probe | B2 90B | B5 90B |
|---|---|---|
| P1 corpus integrity | PASS | PASS |
| P2 executor self-test on gold | PASS (765 / 765 = 100 %) | PASS |
| P3 A1@K=5 failure residual | PASS (residual 20.51 pp) | PASS |
| P4 decomposition argument + multimodal completeness | PASS (1698 chars; 100 %) | PASS (1234 chars; 100 %) |
| W99 addressability probes (Bx-specific) | PASS | PASS |

W100 adds **one** new NIM-free addressability probe to guard
against cross-scale-collapse before any 90B NIM call:

| Probe | Candidate | Threshold | What it does |
|---|---|---|---|
| **AddrW100-B2-P5 — cross-scale rescue-prior stability** | B2 | rescue-prior at 90B ≥ rescue-prior at 11B − 20 pp absolute | Mines the W99 11B B2 per-problem outcomes; reads the 5 W97 unique-A1-rescues; computes the W99-11B empirical rescue rate (3 / 3 = 100 %); accepts if the W96-D 90B preflight residual (20.51 pp) leaves enough A1-only rescue room (at least 3 unique-A1-rescues are still expected at 90B by Hoeffding lower bound on residual). |
| **AddrW100-B5-P4 — cross-scale route-mass stability** | B5 | per-route mass at 90B = per-route mass at 11B (deterministic parser; check is structural) | Re-runs the W99 question-type parser on the same slice; confirms 18 multi-choice + 12 yes_no/numeric/short_text route distribution is byte-identical at 90B (parser is NIM-free). |

These two probes are NIM-free and run before any 90B NIM call.

## Pre-committed Phase 2 pilot gates (W95 9-gate shape; locked 2026-05-25)

For B2 + B5 90B Phase 2 pilots, the gates are byte-identical to
the W95 / W96-A / W96-C / W97 / W98 / W99 gates; only the arm
name, scale tag, and slice seed appear:

1. **Slice pre-committed**: 30 problems by deterministic slice
   with **seed 96_504_002** BEFORE any NIM call.  Slice SHA
   recorded.  (Same slice as W97 / W98 / W99 — anti-cheat
   parity.)
2. **A1 < 90 %**: A1 @ K=5 pass rate on the 30-problem slice
   must stay below 90 %.  Expected to PASS cleanly at 90B (the
   W96-D residual estimate of 20.51 pp implies A1 @ K=5 ≈ 79.49 %).
   If this gate FAILs at 90B as well, Option A is honoured.
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5 pp`.
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice
   problems at end-of-run → 100 % pass.

### Honest acknowledgement of cross-scale risks

Three risks specifically for the 90B run:

1. **A1 may NOT saturate**.  This is GOOD for gate 2 but means
   the 11B Option-A escape hatch does not apply; the B − A1
   margin must clear +5 pp on its own merits.
2. **A1 may rise enough to absorb the unique-A1-rescue
   cluster**.  At 11B, the unique-A1-rescues were 5 / 30 = 16.7 %
   of the slice.  At 90B with A1 likely ~ 79.49 %, the
   unique-A1-rescue surface may shrink (A1 may PASS more of the
   problems without needing the final-VLM rescue), which COULD
   reduce B2's headroom even if the mechanism stays load-bearing.
3. **B2's mechanism-load-bearingness must be VERIFIED by
   rescue-rate, not just margin**.  Per the W96-C C1 lesson, a
   PASS at 90B with low rescue rate (< 1 / 3 rescues when
   invoked) is NOT a frontier-relevant PASS.  W100 records and
   gates on the rescue rate explicitly.

### Mechanism-load-bearingness sub-gates (B2 only)

In addition to the 9 W95 gates, B2 has TWO mechanism sub-gates
that the 90B run must clear:

* **MLB-1 — Final-VLM invocation rate** ≤ 50 % of problems
  (i.e., at most 15 / 30 invocations).  At 11B this was 10.00 %
  (3 / 30).  A 50 % ceiling at 90B leaves room for A1 to be
  weaker than 11B on some slice cells while still requiring B2
  to be a *rescue* mechanism, not a generic re-answer loop.
* **MLB-2 — Final-VLM rescue rate** ≥ 33 % of invocations
  (i.e., at least 1 in 3 invocations PASSes).  At 11B this was
  100.00 % (3 / 3).  A 33 % floor at 90B catches the W96-C
  C1-style variance-driven PASS where the mechanism is not
  actually doing the work.

These are reported alongside the 9 Phase 2 gates and the
structural verdict.  A B2 PASS with MLB-2 < 33 % is downgraded
to `PASS_NON_MECHANISM_DRIVEN` (echoing W96-C terminology)
and does NOT entitle Phase 3.

## Cross-scale rule (W96-C carry-over; locked 2026-05-25)

Identical to W97 / W98 / W99:

* **W100 90B Phase 2 entitled** for B2 + B5 by the W99
  Option-A structural verdict.  This is the realisation of that
  entitlement.
* **Phase 3 entitled** for B2 IFF B2 Phase 2 PASS at BOTH 11B
  AND 90B with mechanism-load-bearingness sub-gates clearing.
  W99 cleared the 11B side (structurally; gate 2 saturation
  excused under Option A).  W100 must clear the 90B side cleanly
  (Option A is NOT auto-applicable at 90B because A1 is not
  expected to saturate; if A1 unexpectedly saturates AND B2 − A1
  > +5 pp AND per-problem majority ≥ 16 / 30, Option A applies
  again, but mechanism sub-gates MLB-1 + MLB-2 are still
  required).
* A Phase 2 PASS at 90B alone is NOT sufficient for Phase 3 (the
  11B side is already PASSed via Option A; the conjunction is
  the entitlement).
* A Phase 2 FAIL at 90B does NOT auto-launch retries; it is the
  empirical verdict.

## Phase 3 decision logic (PRE-LOCKED)

After the W100 90B B2 + B5 pilots land:

1. **B2 90B Phase 2 FAIL** → Phase 3 NOT launched.  Carry
   forward `W100-L-REALWORLDQA-B2-DIRECT-VISION-FINAL-TURN-PHASE2-90B-CAP`.
   Promote `COO-9` (second code benchmark) to lead path; W100
   either documents the kill and stops, or begins `COO-9`
   selection/build in the same milestone if the runbook for that
   surface is also pre-committable.
2. **B2 90B Phase 2 PASS with MLB-1 + MLB-2 clearing** →
   Phase 3 ENTITLED.  Make an explicit choice:
   * **Choice A**: pre-commit the Phase 3 runbook IN THIS
     MILESTONE (W100 becomes W100-A + W100-B); launch the 3-seed
     × 100-problem × K=5 retirement bench under the W95 / W96-A
     pre-committed retirement-margin rule.  Approximate budget:
     ~3,300 NIM calls at 90B; ~6-10 h wall.
   * **Choice B**: defer Phase 3 to W101 with a written
     justification (e.g., to allow a second 90B seed at the
     Phase 2 size first; to revisit slice fairness; to mine the
     90B B2 per-problem outcomes for the multi-seed slate).
   * Choose A or B *explicitly*, by reading the empirical
     verdict, NOT by inertia or feel.
3. **B2 90B Phase 2 PASS without MLB-1 or MLB-2 clearing**
   (`PASS_NON_MECHANISM_DRIVEN`) → Phase 3 NOT launched per
   W96-C precedent; B2's 11B mechanism IS load-bearing but the
   90B PASS is variance-driven; W100 closes with a more limited
   carry-forward and Phase 3 deferred indefinitely on this slice.
4. **B5 90B Phase 2 PASS or FAIL** → B5 stays baseline-only
   regardless.  A B5 PASS records the cross-scale routing-
   ceiling result; a B5 FAIL records the routing-ceiling cap.
   Neither affects Phase 3 entitlement (which is governed by B2
   alone).

## Code-pivot contingency (locked 2026-05-25)

If B2 90B Phase 2 FAILs:

1. Document the failure cleanly in
   `docs/RESULTS_W100_REALWORLDQA_B2_PHASE2_90B_V1.md` with
   per-problem mining vs W99 11B B2 outcomes.
2. Sync Linear immediately: add the W100 verdict comment to
   `COO-6` and `COO-24`; **promote `COO-9` priority to High
   (lead)** and post the lead-path comment; demote any
   RealWorldQA follow-up issues that exist.
3. If the second-code-benchmark selection/build surface is
   pre-committable in the same milestone (one of MBPP+, HumanEval+,
   APPS, LiveCodeBench, SWE-bench-lite per `COO-9`), begin the
   selection/runbook work for `COO-9` in W100 itself; do NOT
   wait for W101.
4. If the second-code-benchmark selection is not yet runbook-
   committable in W100, leave it as the explicit W101 charter.

The cross-modal RealWorldQA arc is then *frozen at 11B* in the
W100 frontier audit; the user's brief Part H requires that we
do NOT pretend the arc is one tweak away.

## NIM smoke test (mandatory before any 90B pilot)

Each B2 + B5 90B pilot launch is preceded by a 1-token POST to
the NIM `chat/completions` endpoint with model
`meta/llama-3.2-90b-vision-instruct`, returning HTTP 200 + a
non-empty completion.  The W99 90B preflight already confirmed
the corpus + parquet-shard SHAs; W100 only confirms the 90B NIM
HTTPS path responds.

## Anti-cheat (carry-forward from W88–W99)

All W88–W99 anti-cheat clauses carry forward verbatim:

* Slice = deterministic `select_realworldqa_subset_v1` with
  seed `96_504_002`; SHA-anchored at run start.
* Same VLM model on every arm.
* Same K=5 byte-exact budget on A1 / B2 / B5.
* Executor = `evaluate_realworldqa_answer_v1`.  No LLM
  judge.
* Parquet shards SHA-anchored at pilot start; mismatches
  refuse to run.
* No selective retries.
* Per-call sidecars (`text_calls.jsonl`,
  `vlm_calls.jsonl`, `per_problem.jsonl`) + per-seed
  Merkle + bench Merkle.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new `coordpy.*` module is introduced for W100.  The
  bench modules `coordpy.realworldqa_bench_v3` (B2) and
  `coordpy.realworldqa_bench_v5` (B5) are reused unchanged.
* Optionally, a thin W100 driver
  `scripts/run_w100_realworldqa_pilot.py` may be added that
  reuses the W99 pilot internals and adds W99-11B
  per-problem-outcome cross-scale comparison.  This is a
  *script*, not a library module; it does NOT enter the
  public API.

## Operational plan

1. **(W100 frontier-relevance audit supplement)** —
   `docs/FRONTIER_RELEVANCE_AUDIT_W100_V1.md`.
2. **(W100 NIM-free addressability probes AddrW100-B2-P5 +
   AddrW100-B5-P4)** — re-use the W99 preflight verdicts on
   disk; add the two cross-scale probes as a thin standalone
   addendum or as part of the W100 pilot driver pre-flight
   block.  No NIM call required.
3. **(W100 NIM smoke at 90B)** — 1 token POST with
   `meta/llama-3.2-90b-vision-instruct`; HTTP 200 + non-empty
   completion required before any pilot.
4. **(W100 B2 90B Phase 2 cheap pilot)** —
   `scripts/run_w100_realworldqa_pilot.py --candidate B2
   --vlm-model meta/llama-3.2-90b-vision-instruct`.  Output:
   `results/w100/realworldqa_pilot/<RUN_ID>/`.  ~330 NIM calls;
   ~20-40 min wall (90B is ~2× slower than 11B per the W96-A
   wall numbers).
5. **(W100 B5 90B Phase 2 cheap pilot)** — same script with
   `--candidate B5`.  ~330 NIM calls; ~30-60 min wall.
6. **(W100 Phase 3 — conditional)** — only if B2 90B PASS with
   MLB sub-gates clearing.  Decision tree above.
7. **(Linear ↔ GitHub sync)**
   a. Create `COO-24` (W100) under `COO-6`.
   b. Post W100 verdict to `COO-6` and `COO-24`.
   c. Append a `W100` entry to `linear_github_mapping.json` and
      run `scripts/sync_linear_github_v1.py validate`.
   d. If B2 90B FAILs: promote `COO-9` to lead and post the
      lead-path comment.
8. **(W100 code-pivot contingency)** — only if B2 90B FAILs;
   begin `COO-9` runbook work in W100 if pre-committable.

## Honest framing

W100's job is to **confirm or refute B2 at 90B**.  No new
candidates, no new architectures, no new mining surfaces, no
re-opening of B4 or the typed-extract sub-family.  The
cross-scale confirmation is the experiment; the verdict is the
deliverable.

If B2 90B PASSes cleanly with MLB sub-gates clearing, the
programme is entitled to a *stronger* claim than W99: the
image-at-decision-boundary mechanism is load-bearing at BOTH
scales on RealWorldQA, and the W95-B0-family-via-B2 repair
generalizes cross-scale.  Phase 3 retirement evidence remains
the bar for the *full* "multi-agent context superiority on
RealWorldQA" claim.

If B2 90B FAILs, the programme is restricted to the 11B
structural claim only; the cross-scale generalisation is not
earned; `COO-9` becomes the lead path; the W100 frontier audit
documents the 90B cap.

Either way, the W93 / W94 / W95 / W96-A / W96-C / W96-D / W97 /
W98 / W99 / **W100** preflight-first + cross-scale + multi-
candidate-tournament-then-confirm discipline is preserved as the
10th consecutive validation.
