# W98 — RealWorldQA multi-candidate assault (B1 + B2) — runbook

> **Pre-commit contract for W98, locked 2026-05-25 BEFORE any
> NIM call for either W98 candidate.**
>
> W97 D2-B0 Phase 2 11B FAILed (B − A1 = −6.67 pp; gates 2/3/4
> FAIL) with structurally informative per-problem disagreement:
> 22 / 30 both-pass; 5 / 30 unique-A1-rescues on vision-bound
> yes/no perception; 3 / 30 unique-B-rescues on multi-choice
> spatial; 0 / 30 neither.  Carry-forward
> `W97-L-REALWORLDQA-D2-B0-PHASE2-11B-CAP`.  W98 is the
> multi-candidate follow-up: build 2 serious arsenal-driven
> candidates (B1 typed scene-graph extraction + question-typed
> solver; B2 direct-vision final-turn answerer); run cheap NIM-
> free preflight + W98-specific addressability probes on both;
> promote AT MOST ONE to a 1-seed × 30-problem cheap NIM pilot
> at 11B.
>
> **The W98 deliverable in this milestone is the cheap
> preflight verdicts for B1 + B2 + (if at most one is
> entitled) one cheap NIM pilot.  No Phase 3 retirement bench
> is in scope unless the pilot PASSes Phase 2 at BOTH 11B AND
> 90B per the W96-C cross-scale rule.**
>
> No version bump.  No PyPI publish.  `coordpy.__version__`
> stays `0.5.20`; `coordpy.SDK_VERSION` stays
> `coordpy.sdk.v3.43`.

## Linear

* New issue **`COO-22`** (W98): RealWorldQA multi-candidate
  slate (B1 + B2 cheap preflight + at most one cheap NIM pilot
  at Llama-3.2-11B-Vision; cross-scale 90B conditional on 11B
  PASS).  Parent: `COO-6`.  High priority.
* Related: `COO-21` (W97-A; closed Done).

## Arsenal-mining anchor

`docs/RESULTS_W98_ARSENAL_MINING_V1.md` records the per-failure
diagnosis of the W97 D2-B0 unique-A1-rescue cluster (5 / 30)
and unique-B-rescue cluster (3 / 30) by mining the W97
sidecars directly.  The W98 candidate slate is:

* **B1** — `coordpy/realworldqa_bench_v2.py`: typed scene-graph
  extraction + question-typed solver.  Addresses all 5 W97
  unique-A1-rescue root causes (3 output-format-mismatch
  failures via the typed solver; 2 lossy-extraction failures
  via the schema).
* **B2** — `coordpy/realworldqa_bench_v3.py`: direct-vision
  final-turn answerer.  Distinct mechanism: keeps the image
  alive at the decision boundary on the failure cluster
  (text-solver FAIL → final VLM call with image + extraction
  + prior candidates).  Different from W96-C C1 verifier
  (committed answerer, not binary agree/disagree).

A **third candidate B3 (question-type oracle router)** is
sketched in the arsenal-mining doc but explicitly NOT
implemented — implementing it would scatter the milestone into
a non-substrate-coupled switch baseline.

## Frontier-relevance audit anchor

`docs/FRONTIER_RELEVANCE_AUDIT_W97_V1.md` and its W98
supplement `docs/FRONTIER_RELEVANCE_AUDIT_W98_V1.md` classify
every in-repo mechanism as active-frontier / baseline-only /
historical-artifact / dead-direction / anti-pattern.  B1 and B2
are both **active frontier arsenal** additions in W98.  No
bounded / compaction / summary mechanism is allowed in either
candidate's lead path.  W96-C C1 verifier (refuted on MathVista)
is explicitly NOT revived; B2 is mechanistically distinct from
a verifier (committed answerer).

## Hypotheses (locked 2026-05-25)

### B1 (typed scene-graph + question-typed solver)

W97 D2-B0's free-text bullet extraction lost the discriminating
signal on the failure cluster: 3 failures had the answer in
prose but the solver returned numbers (output-format mismatch);
2 failures had the reader degenerate or omit the required state
/ depth primitive.  B1's typed schema (`objects[].state`,
`objects[].orientation`, `objects[].depth`,
`spatial_relations[].depth_relation`) + question-typed solver
prompts directly address all 5.  Expected B1 − A1 distribution:
centred around +0 to +10 pp; non-trivial chance of clearing the
+5 pp Phase 2 bar.

### B2 (direct-vision final-turn answerer)

W97 unique-A1-rescues are exactly the cluster where unified
VLM K=5 wins by re-seeing the image.  B2's final turn gives
the failure cluster the same image access A1 has, plus the
structured extraction context.  Short-circuit on text-solver
PASS protects D2-B0's existing wins.  Expected B2 − A1
distribution: centred around +0 to +10 pp; comparable to B1
but via a different mechanism.

### Pre-pilot prediction (locked 2026-05-25 BEFORE NIM)

> "On the W98 cheap preflight + addressability probes,
> subjective priors:
>
> * Probability B1 passes all preflight + AddrP1–AddrP5 probes:
>   ~ 70–80 % (the question-type parser is rule-based +
>   deterministic; the typed solver prompt is a documented
>   format; AddrP1 / AddrP2 are NIM-free).
> * Probability B2 passes all preflight + AddrP3–AddrP5 probes:
>   ~ 80–90 % (B2's mechanism is a direct repurposing of A1's
>   image access on the failure cluster; AddrP3 is satisfied by
>   construction).
> * If both pass: the cheap-discriminator ranking picks the
>   higher addressability score and promotes ONLY THAT ONE.
> * The cheap NIM pilot (if launched) costs ~ 330 NIM calls,
>   ~ 25 min wall at 11B.  The pilot's pre-committed +5 pp
>   bar is the discriminator; subjective probability of
>   clearing +5 pp: ~ 25–35 % for whichever wins the ranking.
> * Expected milestone outcome: at most one cheap NIM pilot;
>   most likely outcome is a documented kill or narrow
>   verdict per the W93/W94/W95/W96-A/W96-C/W96-D/W97
>   discipline."

This prediction is anchored in the run record (the preflight
verdict cid; written BEFORE the pilot launches if any).

## Baselines (locked 2026-05-25)

Identical W95-shape on RealWorldQA, used for both B1 and B2:

* **A0** — text-only mode (image=None) of the VLM at T=0.0,
  K=1.
* **A1** — unified VLM mode at T=0.7, K=5.
* **B1** — `coordpy.realworldqa_bench_v2.run_realworldqa_bench_v2`:
  typed scene-graph reader (T=0.0, 1 VLM call) + question-typed
  text solver (T=0.7, 4 text calls / reflexion).  K=5 byte-
  exact.
* **B2** — `coordpy.realworldqa_bench_v3.run_realworldqa_bench_v3`:
  scene reader (T=0.0, 1 VLM call) + text solver (T=0.7, 3
  text calls / reflexion) + final-VLM answerer (T=0.0; sees
  image; runs only on text-solver FAIL).  Short-circuit on
  first text-solver PASS pads with text-solver retries to keep
  K=5 byte-exact.

Same VLM model on A0 / A1 / B reader / B2 final-VLM.  Same K=5
budget on A1, B1, B2.  Same executor
(`evaluate_realworldqa_answer_v1`) on every arm.

## W98-specific NIM-free preflight + addressability probes

Layered ON TOP of the W93 / W96-D composite (which already
PASSED for D2 in W96-D at both 11B and 90B):

| Probe | Candidate | Threshold | Implementation |
|---|---|---|---|
| **AddrP1 — typed-prompt yes/no recovery rate** | B1 | ≥ 3 / 5 W97 unique-A1-rescues have the answer present in the W97 reader extraction in prose form | Re-read W97 sidecars; count failure-cluster extractions containing the gold answer in prose. |
| **AddrP2 — schema coverage of W97 failure cluster** | B1 | Schema contains `objects[].state`, `objects[].orientation`, `objects[].depth`, `objects[].text_in_object` | Static schema audit. |
| **AddrP3 — direct-vision rescue prior** | B2 | A1 K=5 rescues 5 / 5 of failure cluster; B2 final turn has equivalent visual access | Definitionally satisfied (5 unique-A1-rescues = A1 PASS on those). |
| **AddrP4 — short-circuit preserves D2-B0 both-pass + unique-B** | B1 + B2 | First-PASS short-circuit logic in both new benches; same ship_first_pass semantics on text-solver turns as W97 D2-B0 | Static code audit of `_run_b_typed_vlm_team` (B1) and `_run_b_direct_vision_final` (B2). |
| **AddrP5 — budget exact (K=5; same VLM on every arm)** | B1 + B2 | Total calls per problem = 1 + K + K = 11 at K=5 (1 A0 + 5 A1 + 5 B) | Static code audit. |
| **AddrP6 — question-type parser correctness on W97 slice** | B1 | Parser correctly classifies ≥ 90 % of the W97 30-problem slice by question type | Run `detect_question_type_v2` on every W97 slice question; verify against manual annotation. |
| **AddrP7 — B2 final-VLM short-circuit count plausibility** | B2 | On a simulation of the W97 slice (using W97 D2-B0 per-problem outcomes as a stand-in for the text-solver chain), the final-VLM invocation count is ≤ 30 % of slice size | Static probe: count W97 D2-B0 FAILs on the W97 slice as upper-bound estimate. |

## Pre-committed Phase 2 pilot gates (only if cheap probes earn a NIM pilot)

If exactly one W98 candidate survives all NIM-free probes, the
W98 milestone promotes it to a 1-seed × 30-problem cheap NIM
pilot at 11B under THE SAME 9 pre-committed Phase 2 gates as
W97 (byte-identical gate texts; only the candidate arm name
and slice seed change):

1. **Slice pre-committed**: 30 problems by the W96-D
   RealWorldQA loader's deterministic slice with **seed
   96_504_002** BEFORE any NIM call.  Slice SHA recorded.
   (Same slice as W97 — anti-cheat parity.)
2. **A1 < 90 %**: A1@K=5 pass rate on the 30-problem slice
   must stay below 90 %.  (Will likely FAIL again on the
   96_504_002 slice — A1 was 90.00 % in W97.  See "Cross-
   scale rule" below for the structural implication.)
3. **B > A1**: `b_pass_rate > a1_pass_rate`.
4. **Margin ≥ +5 pp**: `b_pass_rate − a1_pass_rate ≥ 5.0 pp`.
5. **B > A0 by ≥ +5 pp**: image is load-bearing in B.
6. **Per-problem majority**: B ≥ A1 on ≥ 16 of 30 problems.
7. **Budget accounting exact**: 1 + 5 + 5 = 11 calls per
   problem.
8. **Audit chain re-derives**: per-call sidecars + per-seed
   Merkle + bench Merkle re-derive offline.
9. **Executor stays clean**: P2 re-run on the 30 slice problems
   at end-of-run → 100 % pass.

### Honest acknowledgement of the slice-saturation risk

The 96_504_002 slice saturated A1@K=5 at exactly 90.00 % in
W97; gate 2 will likely FAIL again at 11B on the same slice
even if the W98 candidate is structurally better.  Three
honest options:

* **Option A (default)**: re-run on the same 96_504_002 slice
  for direct cross-candidate comparison; accept gate 2 FAIL
  and treat the (B − A1) delta vs the W97 D2-B0's −6.67 pp as
  the discriminator.  Document this as `W98-L-REALWORLDQA-
  SLICE-96504002-A1-SATURATION-CAP` if gate 2 FAILs.
* **Option B**: introduce a NEW slice seed (e.g. `96_504_003`)
  that avoids A1 saturation.  Risk: loses direct comparability
  to W97 D2-B0; introduces a second-slice-shopping anti-pattern
  unless pre-committed.
* **Option C**: skip 11B and go directly to 90B on the same
  slice (90B A1@K=5 ≈ 79.49 % per W96-D preflight; expected
  ~ 25 pp residual on the slice).  Risk: violates the cross-
  scale rule's preferred 11B-first ordering.

**This runbook pre-commits Option A** as the default: re-run
on `96_504_002` for direct comparability; document the
slice-saturation cap if gate 2 FAILs; treat the (B − A1) delta
as the architecturally-relevant discriminator regardless of
whether A1 is at saturation.  If gate 2 fails AND B − A1 > 0
AND B − A1 > +5 pp AND per-problem majority ≥ 16/30, the
verdict is "**STRUCTURALLY POSITIVE despite slice-saturation
artefact**" and licenses Option B (new slice) as a W99
follow-up — not as an in-milestone slice change.

## Cross-scale rule (W96-C carry-over, locked 2026-05-25)

Identical to W97:

* **Cross-scale 90B Phase 2 entitled** IFF 11B Phase 2 PASS
  (all 9 gates) — OR with written justification if 11B FAILS
  by a narrow margin (e.g. gate 2 alone FAILing on a known-
  saturated slice but B − A1 > +5 pp).
* **Phase 3 entitled** IFF Phase 2 PASS at BOTH 11B AND 90B,
  OR Phase 2 PASS at 90B alone with written justification.
* A Phase 2 PASS at 11B alone is NOT sufficient for Phase 3.
* A Phase 2 FAIL at 11B does NOT auto-launch a 90B Phase 2.

## Phase 3 — full bench (NOT in this milestone)

If both Phase 2 stages clear, Phase 3 will require its own
runbook section locked BEFORE any Phase 3 NIM call, mirroring
W96-A Phase 3.

**No Phase 3 will be launched in this milestone.**

## Cross-candidate decision logic (PRE-LOCKED)

**Phase 0 — NIM-free preflight + addressability probes.**
Run the W96-D D2 composite + W98 AddrP1–AddrP7 probes for both
candidates.  Any candidate failing any probe is **KILLED** and
documented.

**Phase 1 — Cheap NIM pilot (only if entitled).**  If exactly
one candidate survives, run its 1-seed × 30-problem × K=5
pilot at 11B.

If both candidates survive: rank by addressability score:
* B1 addressability score = (count of W97 failure-cluster
  diagnoses addressed by typed schema + question-typed solver)
  / 5  → expected ~ 5/5 = 1.00.
* B2 addressability score = (count of failure-cluster diagnoses
  addressed by direct-vision final-turn) / 5  → expected ~
  5/5 = 1.00 (definitionally, since A1 wins all 5).
* TIE-BREAKER: prefer B1 (lower expected NIM cost — text
  solver dominates the budget; B2 invokes the costlier VLM
  on every text-fail).

If neither candidate survives preflight: document both kills,
pivot to `COO-9` (second code benchmark) per Part H of the
W98 brief.

## NIM smoke test (mandatory before any pilot)

1. One 1-token POST against the candidate model returns HTTP
   200 with a non-empty completion.
2. One 1-problem dry-run on the W98 candidate's wiring at 11B.
3. Smoke run dir: `results/w98/realworldqa_smoke_11b/<RUN_ID>/`.

## Anti-cheat (carry-forward from W88–W97)

All W88–W97 anti-cheat clauses carry forward verbatim:

* Slice is deterministic by `select_realworldqa_subset_v1`
  with seed `96_504_002`; SHA-anchored at run start.
* Same VLM model on every arm.
* Same K=5 byte-exact budget on A1 / B (1 A0).
* Executor = `evaluate_realworldqa_answer_v1`.  No LLM judge.
* Parquet shards SHA-anchored at pilot start; mismatches
  refuse to run.
* No selective retries.
* Per-call sidecars (`text_calls.jsonl`, `vlm_calls.jsonl`,
  `per_problem.jsonl`) + per-seed Merkle + bench Merkle.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.realworldqa_bench_v2` and
  `coordpy.realworldqa_bench_v3` are **explicit-import only**.
* New scripts are independent of W95 / W96 / W97 scripts;
  nothing pre-existing is modified.

## Operational plan

1. **(W98 arsenal mining + frontier-relevance audit
   supplement)** — DONE; see
   `docs/RESULTS_W98_ARSENAL_MINING_V1.md` +
   `docs/FRONTIER_RELEVANCE_AUDIT_W98_V1.md`.
2. **(W98 B1 + B2 bench modules + tests)** —
   `coordpy/realworldqa_bench_v2.py` +
   `coordpy/realworldqa_bench_v3.py` +
   `tests/test_w98_realworldqa_bench_v2_v3.py`.
3. **(W98 NIM-free preflight + addressability probes)**
   a. Run `scripts/run_w98_realworldqa_preflight.py
      --candidate-model meta/llama-3.2-11b-vision-instruct`
      and record the verdict to
      `results/w98/realworldqa_preflight_b1_b2/<RUN_ID>/`.
   b. Both candidates' addressability probes are run as part
      of the same preflight script.
4. **(W98 NIM smoke test at 11B — conditional)**
   Only if both candidates PASS preflight AND the chosen
   winner is promoted to a NIM pilot.
5. **(W98 chosen-winner Phase 2 at 11B — conditional)**
   Apply the 9 Phase 2 gates.  Output:
   `results/w98/realworldqa_pilot/<RUN_ID>/`.
6. **(W98 cross-scale Phase 2 at 90B — conditional on 11B
   PASS or written-justification narrow miss)**
7. **(W98 Phase 3 — OUT OF SCOPE)**
8. **(Linear ↔ GitHub sync)**
   a. Create `COO-22` (W98) under `COO-6`.
   b. Update `COO-6` summary with W98 verdict.
   c. Append a `W98` entry to `linear_github_mapping.json` and
      run `scripts/sync_linear_github_v1.py validate`.

## Honest framing

W98's job in this milestone is to **earn or kill the next
expensive run on RealWorldQA by cheap evidence on a
multi-candidate slate**.  The milestone is a discriminator,
not a retirement run.  No bounded / compaction / summary
mechanism appears in either candidate.  No vibes-driven
architectural choice — the slate is mined from the W97
per-problem failure cluster.  Either preflight earns one
candidate or kills both; either way the W93/W94/W95/W96-A/
W96-C/W96-D/W97 discipline is preserved as the 8th consecutive
validation.
