# W98 — RealWorldQA B1 + B2 NIM-free preflight V1 (BOTH PASS; winner = B1)

> **2026-05-25 — Both W98 candidates B1 (typed scene-graph
> extraction + question-typed solver) and B2 (direct-vision
> final-turn answerer) PASS all preflight gates at BOTH 11B
> AND 90B: W96-D D2 composite (P1..P4) + W98 addressability
> probes AddrP1..AddrP7.  Per the pre-committed cross-candidate
> decision logic in `docs/RUNBOOK_W98.md` (tie-break = lower
> expected NIM cost), winner = B1.  B2 is deferred to W99 only
> if B1 pilot PASSes Phase 2 at both scales and B2's distinct
> mechanism remains plausibly load-bearing.  No NIM was spent
> in this milestone-step.**

## Configuration

| Field | Value |
|---|---|
| Battlefield | RealWorldQA test (`lmms-lab/RealWorldQA`) |
| Parquet shard SHA-256 (shard 0) | `0ed8b555...` |
| Parquet shard SHA-256 (shard 1) | `7dcb3ac3...` |
| Corpus Merkle root | `dc629acbf88d929d95c4a47c8e553c050eb829aa48ba32ad53ca6283090618ab` |
| W97 slice | `seed=96_504_002`, n=30 |
| W97 unique-A1-rescues mined | 5 / 30 |
| W97 unique-B-rescues mined | 3 / 30 |
| B1 candidate module | `coordpy/realworldqa_bench_v2.py` |
| B2 candidate module | `coordpy/realworldqa_bench_v3.py` |
| Preflight runner | `scripts/run_w98_realworldqa_preflight.py` |

## Per-candidate composite preflight (W96-D P1..P4)

Both B1 and B2 PASS every probe at both scales:

| Probe | B1 11B | B1 90B | B2 11B | B2 90B |
|---|---|---|---|---|
| P1 corpus integrity | PASS | PASS | PASS | PASS |
| P2 executor self-test (gold) | PASS (765/765 = 100%) | PASS | PASS | PASS |
| P3 A1@K=5 failure residual | PASS (residual 26.56 pp) | PASS (residual 20.51 pp) | PASS | PASS |
| P4 decomposition argument + multimodal completeness | PASS (1607 chars) | PASS | PASS (1354 chars) | PASS |

## W98 addressability probes (NIM-free)

| Probe | Candidate | Verdict | Detail |
|---|---|---|---|
| AddrP1 — typed-prompt yes/no recovery rate | B1 | **PASS** | 3 / 5 W97 unique-A1-rescues have the gold answer present in the W97 reader's prose extraction.  A typed solver prompt that knows the question type would plausibly recover at least these 3 even without any schema change.  Pids: `rwqa_test_000135` (Yes; reader said "1 stop sign"), `rwqa_test_000555` (No; reader said "are not facing left"), `rwqa_test_000615` (No; reader said "currently red"). |
| AddrP2 — schema coverage of failure cluster | B1 | **PASS** | B1's typed schema explicitly includes `objects[].state`, `objects[].orientation`, `objects[].depth`, `objects[].text_in_object` — every primitive required for the 5 W97 failures. |
| AddrP3 — direct-vision rescue prior | B2 | **PASS** | A1 K=5 rescues 5 / 5 W97 unique-A1-rescues (definitionally — those are the unique-A1-rescues).  B2's final-turn VLM has equivalent visual access on the failure cluster. |
| AddrP4 — short-circuit preserves D2-B0 wins | B1+B2 | **PASS** | Both new benches' B-arm runners use first-PASS short-circuit on text-solver turns; D2-B0's 22 / 30 both-pass + 3 / 30 unique-B-rescues are mechanistically protected. |
| AddrP5 — budget exact (K=5; same VLM) | B1+B2 | **PASS** | V2: 1 reader + 4 text-solver = 5 = K(5).  V3: 1 reader + 3 text-solver + 1 final-VLM-or-pad = 5 = K(5). |
| AddrP6 — question-type parser correctness | B1 | **PASS** | Parser correctly classifies 29 / 30 = 96.7 % of the W97 slice (threshold ≥ 90 %).  Misclassification on a single short-text question that opens with "Where" (not classified as yes/no since "Where" is not in the verb prefix set; correctly falls through to short_text — i.e. the parser was correct; the manual gold heuristic was overly permissive here).  No yes/no question is misclassified. |
| AddrP7 — B2 final-VLM invocation share | B2 | **PASS** | W97 D2-B0 FAILed on 5 / 30 = 16.7 % of slice (threshold ≤ 30 %).  Upper bound on B2 final-VLM invocation share on the same slice; well below the budget-burn ceiling. |

## Cross-candidate decision (locked in `RUNBOOK_W98.md`)

Both candidates survived preflight at both 11B and 90B.  Per
the pre-committed cross-candidate decision logic:

* **B1 addressability score = 5 / 5** (typed schema +
  question-typed solver addresses all 5 W97 failures).
* **B2 addressability score = 5 / 5** (direct-vision final
  turn has visual access equivalent to A1 K=5 which rescues
  5 / 5 by definition).
* **Tie-break = lower expected NIM cost.**

B1's expected NIM cost is text-dominated (1 VLM reader + 4
text solver calls per problem; text-only NIM throughput is
higher).  B2 invokes a second VLM call on text-solver failures
(estimated ~ 16.7 % of the slice from AddrP7), adding modest
VLM cost on the failure cluster.

**Winner = B1.**  B2 is deferred to W99 only if B1's cheap
pilot PASSes Phase 2 at both scales AND B2's distinct
mechanism remains plausibly load-bearing (a positive B1 pilot
that does not exhaust the failure-cluster signal would license
a B2 follow-up).

## Honest framing

This is a *preflight* — the candidate is *entitled* to a NIM
pilot, not certified as a win.  Per the W97 prediction in the
runbook, the subjective probability that B1's cheap NIM pilot
clears the +5 pp Phase 2 bar is ~ 25–35 %; the most likely
outcome is still a narrow verdict (either narrow PASS or
narrow FAIL).  Pre-committed slice-saturation risk on
`96_504_002` is acknowledged and documented:

* A1@K=5 saturated at exactly 90.00 % on this slice in W97;
  gate 2 will likely FAIL again at 11B even if B1 is
  structurally better.
* Per the runbook, **Option A** is pre-committed: re-run on
  the same slice for direct cross-candidate comparison; if
  gate 2 FAILs but B − A1 > +5 pp AND per-problem majority
  ≥ 16 / 30, the verdict is "STRUCTURALLY POSITIVE despite
  slice-saturation artefact".

## Anti-cheat (carry-forward from W88–W97)

All preflight steps are NIM-free; no model calls were issued.
The W97 sidecars used by AddrP1 / AddrP3 / AddrP6 / AddrP7 are
read from the on-disk run dir
(`results/w97/realworldqa_pilot/...`); no re-derivation of W97
truth is asserted.

## Carry-forwards

### Added (this milestone-step)

**None.**  Preflight by definition does not add carry-forwards;
it only earns (or kills) the next NIM call.

### Retired

**None.**  W89 70B-HumanEval K=5 remains the only confirmed
multi-seed same-budget multi-agent superiority retirement.

## Discipline status

Preflight-first + cross-scale discipline now validated EIGHT
consecutive times (W93 / W94 / W95 / W96-A / W96-C / W96-D /
W97 / W98).  W98's distinguishing addition is the
*multi-candidate slate discriminator* — the milestone runs
preflight on TWO arsenal-driven candidates and promotes AT
MOST ONE to a NIM pilot, preventing the milestone from
scattering across half-funded variants.

## Next move

**Promote B1 to a 1-seed × 30-problem × K=5 cheap NIM pilot
at 11B** under the pre-committed 9 Phase 2 gates in
`docs/RUNBOOK_W98.md`.  Same slice (`96_504_002`).  Same
budget (~ 330 NIM calls; ~ 20–30 min wall).  Pilot driver
(to be created): `scripts/run_w98_realworldqa_pilot.py`.

If `NVIDIA_API_KEY` is not set in the current operator
environment, the pilot launch is deferred to the next
operator-driven session.  All pre-committed contracts hold
across sessions because the slice + corpus + gates are
content-addressed.

## Re-running

```bash
.venv/bin/python scripts/run_w98_realworldqa_preflight.py \
    --candidate-model meta/llama-3.2-11b-vision-instruct

.venv/bin/python scripts/run_w98_realworldqa_preflight.py \
    --candidate-model meta/llama-3.2-90b-vision-instruct
```

Outputs land under
`results/w98/realworldqa_preflight_b1_b2/`.  Canonical 11B
verdict cid: `a5d958506249292c3d88623c73001d819fefdae174fe43c3583d936c67c7bc9e`.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New modules `coordpy.realworldqa_bench_v2` and
  `coordpy.realworldqa_bench_v3` are explicit-import only.

## The honest claim W98 preflight (B1 + B2) earns

**On `lmms-lab/RealWorldQA` test with the W97 D2-B0 cheap
pilot's per-problem failure-cluster diagnosis as the cheap-
probe surface, both W98 candidates B1 (typed scene-graph +
question-typed solver) and B2 (direct-vision final-turn
answerer) PASS the W96-D D2 composite preflight (P1..P4) AND
all 7 W98 addressability probes (AddrP1..AddrP7) at both 11B
and 90B.  Per the pre-committed cross-candidate decision logic
(tie-break = lower expected NIM cost), B1 is entitled to a
1-seed × 30-problem × K=5 cheap NIM pilot at 11B; B2 is
deferred to W99 only if B1 PASSes Phase 2 at both scales.  No
retirement.  No carry-forward retired.  No NIM spent in this
preflight step.  Discipline validation #8.**
