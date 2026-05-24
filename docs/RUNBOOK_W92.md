# W92 — Post-W91 Empirical Superiority Wave V5 (runbook)

> **Pre-commit contract for the W92 wave.  W91 made it
> empirically clear that parameter-only retries (more seeds /
> larger N) cannot retire the cross-modal carry-forward at K=5
> on HumanEval-Visual + VLM-in-loop — the W91 P2 → P2b
> sequence cleanly disconfirmed the marginal 3-seed positive.
> W92 makes a hard ARCHITECTURAL pivot: replace VLM-in-loop
> single-model multi-turn with a TRUE multi-agent
> role-specialized team where each turn uses the strongest
> model for its role.**
>
> Locked 2026-05-23 BEFORE any W92 bench run.  Same retirement
> bars as W88/W89/W90/W91 (no relaxation); only the architecture
> + statistical scope move.

## What W91 actually showed (the inherited diagnosis)

**Cross-modal (W90 P2 + W91 P2 + W91 P2b cumulative):** the
VLM-in-loop architecture (single-VLM multi-turn with image in
context every turn) is the BEST shape so far but does NOT
strictly beat unified VLM at fair K=5 budget across model
scales and corpus regimes:

* W90 P2 doctest_only: 91.7 % = 91.7 % (tie, +0.00 pp).
* W91 P2 all_docstring 3-seed: 86.1 % > 83.3 % (+2.78 pp, but
  variance-driven; 5/6 bars met, +5 pp margin failed).
* W91 P2b all_docstring 7-seed: 77.4 % < 84.5 % (−7.14 pp;
  disconfirms P2; 3/6 bars).

**Structural reason**: the VLM-in-loop architecture uses the
SAME MODEL on every turn.  At K=5, the model has 5 chances at
the same task with slightly different context — but the
context shifts are not architecturally meaningful (image
re-encoded; stderr added).  The unified VLM at K=5 INDEPENDENT
samples gets 5 i.i.d. attempts at the same task; the
diversity gain ≥ the reflexion gain.

**MBPP P1 (W90 + W91 cumulative):** the W88 sequential
reflexion architecture's directional cross-benchmark
generalisation to MBPP at 70B is now confirmed on the mean
(+1.11 → +1.33 pp at 3 → 5 seeds) — but the per-seed strict
majority bar continues to fail (1/3 → 2/5) due to ceiling
effects on 2 of 5 seeds (A1 hits 90 %).  Pure scaling of
seed count cannot retire this bar.

## What W92 changes (and why)

### Prong 1 (single prong) — Cross-modal **role-specialized** at all_docstring + 90B-V + 70B-code, 7 seeds × 12

The W92 architecture is a TRUE multi-agent cross-modal team
with 3 distinct roles, each using the strongest available
model:

**B_role_spec architecture (5 model calls total = same K=5
as A1_vlm):**

```
Turn 0: VLM-Planner (90B-Vision)
        Inputs: image + stripped_prompt
        Output: structured plan (function signature,
                key behaviours, edge cases, sample I/O)

Turn 1: Code-Implementer-v1 (70B Llama-3.3-Instruct, no image)
        Inputs: stripped_prompt + Plan
        Output: first code attempt

Turn 2 (executor; not a model call):
        Run code v1 → pass/fail + stderr

Turn 3: VLM-Verifier (90B-Vision)
        Inputs: image + stripped_prompt + code_v1 + stderr
        Output: structured critique (is it correct against
                the image? what bug class? what to fix?)

Turn 4: Code-Implementer-v2 (70B Llama-3.3-Instruct)
        Inputs: stripped_prompt + Plan + code_v1 + stderr
                + Critique
        Output: corrected code v2

Turn 5: Code-Implementer-v3 (70B Llama-3.3-Instruct)
        Inputs: stripped_prompt + Plan + history of all
                prior code attempts + executor stderr +
                VLM Critique
        Output: final corrected code v3

Ship first PASS by attempt index; else lex-smallest CID.
```

**Total model calls**: 2 VLM (Planner, Verifier) + 3 code-LM
(Implementer ×3) = 5 model calls.  Exactly matches A1_vlm's
K=5 unified-VLM budget.  Note A1_vlm uses 5× 90B-Vision; B
uses 2× 90B-Vision + 3× 70B (less total compute cost — if B
wins, the win is stronger).

Why this is the right architectural pivot:

* **Role specialization is the key W91 evidence-driven
  hypothesis.**  W91 P2b's −7.14 pp shows that calling the
  SAME model 5 times (VLM-in-loop) doesn't add value beyond
  i.i.d. sampling.  Specialized roles using DIFFERENT models
  for different sub-tasks is structurally different.
* **Code generation is the bottleneck.**  The VLM is good at
  vision but not specialist at code; a 70B text-LM is a
  code specialist.  Separating "plan from image" (VLM) from
  "implement code" (code-LM) plays to each model's strength.
* **Verifier in the loop.**  The VLM-Verifier reads the image
  AND the failing code AND stderr; it can identify whether
  the code's bug is a vision-extraction issue (planner got
  the I/O wrong) or a code-implementation issue (implementer
  misunderstood the plan).  This routing signal is unavailable
  to the VLM-in-loop architecture.
* **Image stays in context for vision-relevant turns**
  (Planner, Verifier) but NOT for code-implementation turns
  — the code-LM doesn't need vision; pulling out vision-only
  context reduces noise.

### Sample size — 7 seeds × 12

The W91 P2b 7-seed scale is the new minimum honest
statistical bar.  W91 showed 3-seed positives are unreliable
under +5 pp margin.  7 seeds × 12 problems = 84 outcomes per
arm is comparable scope to W91 P2b.  Same seed range
(90_046_001 through 90_046_007).

### Corpus — all_docstring

Same strip mode as W91 P2 / P2b.  The image is strictly
necessary; no prose description in the prompt.  This is the
hardest regime tested.

## Pre-committed success criteria

**No relaxation from W88/W89/W90/W91.**  Same 6 bars:

1. `b_role_spec_mean_strictly_beats_a0_text_mean = True`
2. `b_role_spec_mean_strictly_beats_a1_vlm_mean = True`
3. B − A0_text margin ≥ +5.0 pp
4. **B − A1_vlm margin ≥ +5.0 pp**
5. B beats A0_text on more than half the seeds.
6. **B beats A1_vlm on more than half the seeds.**

If ALL 6 met: `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
RETIRES.  The W88 SPLIT (VLM-extract + code-LM-generate
naive) failure stands; W90/W91 VLM-in-loop tie/disconfirmation
stand.  W92 retirement claim: **at K=5 budget on
Llama-3.2-90B-Vision + Llama-3.3-70B-Instruct on
HumanEval-Visual all_docstring × 7 seeds × 12 problems, the
role-specialized multi-agent cross-modal team strictly beats
the same-budget unified-VLM K=5 baseline by ≥ +5 pp with per-
seed majority.**

If fewer bars met: carry-forward stays.  New
`W92-L-CROSS-MODAL-ROLE-SPEC-V1-*-CAP` carry-forward records
the additional negative evidence.  Decisive evidence that even
role specialization at K=5 on this benchmark is insufficient
would force the next wave to a different benchmark family.

`W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` is NOT
addressed by this prong (it's about substrate-level injection,
not benchmark superiority); stays.

## Anti-cheat clauses

All W88/W89/W90/W91 anti-cheat clauses carry forward.
W92-specific:

* Same VLM (90B-Vision) on A1_vlm AND on B's Planner +
  Verifier turns.
* Same code-LM (70B Llama-3.3-Instruct) on A0_text floor +
  B's Implementer turns.
* Same K=5 model-call budget per arm (A1_vlm 5× VLM;
  B 2× VLM + 3× code-LM).
* Same task subset per seed across arms (deterministic
  `select_cross_modal_subset_v1(seed)`).
* Same retry policy (6 attempts, 429-aware backoff).
* No selective retries.
* New seeds 90_046_001 through 90_046_007 (same as W91 P2b
  for direct comparability).

## What W92 does NOT do

* Does NOT change the benchmark (still HumanEval-Visual
  all_docstring).
* Does NOT relax the +5 pp margin or per-seed majority bars.
* Does NOT introduce any version-bump or PyPI changes.
* Does NOT attempt MBPP+/LiveCodeBench/SWE-bench (deferred to
  W93 — MBPP+ corpus loader requires differential-testing
  infrastructure for the plus_input set; not feasible in
  this session's time budget).

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New module `coordpy/cross_modal_role_specialized_bench_v1.py`
  ships as explicit-import only.

## Operational plan

1. Build the role-specialized bench module (~1.5h):
   `coordpy/cross_modal_role_specialized_bench_v1.py`.
2. Build the driver script (~30 min):
   `scripts/run_w92_cross_modal_role_specialized_bench.py`.
3. Build the verifier (~15 min):
   `scripts/verify_w92_cross_modal_role_specialized_audit_chain.py`.
4. Build CI tests (~30 min):
   `tests/test_w92_cross_modal_role_specialized_v1.py`.
5. Launch the bench in background (7 seeds × 12 × K=5 ≈ 84
   text + ~168 VLM = ~252 model calls; estimated wall ~3 hours
   on NIM at current rate).
6. Verify audit chain; produce RESULTS doc; update honesty
   surfaces; commit; ask for push approval.
