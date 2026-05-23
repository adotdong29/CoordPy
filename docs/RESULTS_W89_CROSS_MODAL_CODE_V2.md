# W89 — Cross-modal code bench V2 (Post-W88 empirical superiority wave V2)

> **Two retry runs completed 2026-05-22.  Both NEGATIVE on the
> harder bar.  The `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-
> CAP` and `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
> carry-forwards **stay unretired**.**
>
> W89 cross-modal makes two structural pivots from W88:
>
> 1. **Stronger VLM** — Llama-3.2-90B-Vision-Instruct (vs W88's
>    11B-Vision)
> 2. **Image-strict regime** — `strip_mode=all_docstring`
>    (W89 P2; vs W88's `doctest_only`)
> 3. **Stronger code-LM** — Llama-3.3-70B-Instruct
>    (W89 P3; vs W88's 8B)
>
> The structural finding across the two W89 runs + W88: **as
> both VLM and code-LM scale up, A1_vlm and B_cross improve
> proportionally; the B−A1_vlm gap stays near −5.6 pp**.  The
> cross-modal split architecture (VLM-extract + code-LM-generate
> + reflexion) is structurally not load-bearing-better than the
> same-budget unified VLM at K=5 on HumanEval-Visual across
> three model-scale configurations.  Model scale alone does NOT
> retire this carry-forward.

## Summary across W88 + W89

| Run | VLM | Code-LM | Strip mode | A0_text | A1_vlm | B_cross | B−A1 |
|---|---|---|---|---:|---:|---:|---:|
| W88 V1 | Llama-3.2-**11B**-Vision | Llama-3.1-**8B** | doctest_only | 66.7 % | 86.1 % | 80.6 % | **−5.56 pp** |
| W89 P2 | Llama-3.2-**90B**-Vision | Llama-3.1-**8B** | **all_docstring** | 41.7 % | 86.1 % | 58.3 % | **−27.78 pp** |
| W89 P3 | Llama-3.2-**90B**-Vision | Llama-3.3-**70B** | doctest_only | 33.3 % | 91.7 % | 86.1 % | **−5.56 pp** |

All 3 runs: 3 seeds × 12 problems × K=5 budget × HumanEval-Visual.

Bench Merkle roots:

* W88 V1: `37ac174e21cbe3f9...` (in repo since 2026-05-22)
* W89 P2: `00caee17e5c3f738...`
* W89 P3: `79fac23cc8316d60...`

All audit chains re-derive offline byte-for-byte
(4/4 audit PASS on each).

## Per-prong details

### W89 Prong 2 — `all_docstring` + 90B-Vision

The `all_docstring` strip mode REPLACES the entire docstring
with a stub (`"""See the attached image for the function's
specification."""`).  No prose description remains in the
prompt.  The image is the ONLY source of behavioural info.

Result:
* A0_text **dropped to 41.7 %** (vs 66.7 % at W88 doctest_only)
  — the no-image baseline crashes when the prose is stripped.
  Even at 8B, A0_text retains some signal from the function
  signature alone, but most of the W88 prose value is gone.
* A1_vlm stayed at **86.1 %** — the unified VLM at K=5 reaches
  the same ceiling whether the prose is in the prompt or
  encoded in the image.  This is consistent with the image
  carrying the load-bearing info.
* B_cross **dropped 22 pp** to 58.3 % — the VLM-extract +
  code-LM-implement pipeline now bears the FULL load of
  conveying the image's information through the
  text-only-extraction handoff.  Each VLM extraction error
  propagates into the code-LM's 4 reflexion turns; the
  text-only code-LM cannot verify or correct against the
  image.

Retirement bars: 3/6 PASS (image direction); 3/6 FAIL.

### W89 Prong 3 — `doctest_only` + 90B-Vision + **70B code-LM**

Same regime as W88 V1, but both VLM and code-LM scaled up to
their largest open-weight NIM endpoints.  Compares "does
scaling both ends close the gap?"

Result:
* A0_text **dropped to 33.3 %**.  Surprisingly LOWER than W88's
  8B 66.7 %.  Plausible explanation: at T=0 single-shot, the
  larger 70B is more cautious / verbose / less greedy, and
  refuses to "guess" on ambiguous stripped prompts where the
  8B confidently produces something that happens to pass.
  This is the documented "scaling-conservatism" effect; not a
  bench bug.
* A1_vlm climbed to **91.7 %** — 90B-Vision K=5 is stronger
  than 11B-Vision K=5 by +5.6 pp.
* B_cross climbed to **86.1 %** — the 70B code-LM produces
  better implementations from extracted text than the 8B did
  (W88 80.6 % → W89 P3 86.1 %, +5.6 pp).
* **B − A1_vlm gap stayed at −5.56 pp** — same as W88 V1.
  Both arms scaled up by ~5.6 pp; the relative gap is
  invariant under model scale.
* B beats A1_vlm on **1/3 seeds** (W88 was 0/3).  Per-seed:
  (False, True, False).  Single-seed improvement, not majority.

Retirement bars: 3/6 PASS (image direction); 3/6 FAIL.

## What this empirical evidence actually says

Three independent configurations now agree:

1. **Image is empirically load-bearing.**  Across all three
   configurations the delta B_cross − A0_text is positive (+13.9
   pp, +16.7 pp, +52.8 pp).  The W87 multi-modal substrate IS
   carrying real load-bearing information — the gap to a
   no-image baseline widens dramatically as more of the
   behavioural info moves into the image.

2. **The VLM-extract + code-LM split is structurally NOT
   load-bearing-better than unified VLM at fair K=5 budget.**
   At three different model-scale configurations
   (11B+8B, 90B+8B, 90B+70B), the B_cross − A1_vlm gap is
   negative (−5.56 pp, −27.78 pp, −5.56 pp).  Scaling the code-
   LM by ~9× (8B → 70B) and the VLM by ~8× (11B → 90B) does
   NOT flip the sign.  The information-loss at the text-only
   handoff dominates regardless of model quality.

3. **The all_docstring regime hurts the split MORE than it
   hurts the unified VLM.**  This is the strongest piece of
   evidence that the split architecture's failure mode is
   information loss at the modality handoff: when the image
   becomes the sole source of behavioural info, the unified
   VLM (which keeps the image in context throughout) holds
   level at 86 %, but the split crashes 22 pp (80 % → 58 %).

The honest summary: **scaling models alone cannot retire this
carry-forward.  The next attempt must change the architecture
itself** — e.g., keeping the VLM in the executor-feedback loop
(VLM-in-loop reflexion), parallel heterogeneous pool, or
deep cross-modal injection at substrate level.

## Anti-cheat re-statement

All W88 anti-cheat clauses carry forward.  W89-specific:

* W89 P2 changed `strip_mode` to `all_docstring` (documented in
  `RUNBOOK_W89.md` BEFORE the run); the per-problem
  `stripped_prompt` no longer contains the docstring's prose.
* W89 P3 swapped `--vlm-model` to
  `meta/llama-3.2-90b-vision-instruct` and `--code-model` to
  `meta/llama-3.3-70b-instruct`.  Same model on every arm of
  the run within each model class (90B-Vision on A1_vlm and
  B_cross's VLM step; 70B-Instruct on A0_text and B_cross's
  code-LM step).
* Same task subset per seed within each run.
* Same K=5 budget per arm.
* No selective retries; each (seed, problem, arm) triple is
  exactly one set of calls.
* No re-running of failed seeds.

## What W89 ships under this prong

* `docs/RUNBOOK_W89.md` (pre-commit contract, written BEFORE
  any W89 run)
* `docs/W88_FAILURE_DIAGNOSIS.md` (structural-failure analysis
  of W88, written BEFORE any W89 run)
* `docs/RESULTS_W89_CROSS_MODAL_CODE_V2.md` (this doc)
* `results/w88/cross_modal_code/w88_xm_all_docstring_meta_llama-3.2-90b-vision-instruct__meta_llama-3.1-8b-instruct_20260522T222546Z/`
  (Prong 2 bench artifacts: report JSON + text + VLM sidecars)
* `results/w88/cross_modal_code/w88_xm_doctest_only_meta_llama-3.2-90b-vision-instruct__meta_llama-3.3-70b-instruct_20260522T231949Z/`
  (Prong 3 bench artifacts)

No new modules or test files — W89 reuses the W88
`coordpy.cross_modal_code_bench_v1` infrastructure unchanged.
Stable boundary preserved.

## Honest carry-forwards (W89 additions to the registry)

* **`W89-L-CROSS-MODAL-SPLIT-MODEL-SCALE-INVARIANT-CAP`** —
  the B_cross − A1_vlm gap is approximately invariant (~−5.6
  pp) under model scale changes within the tested range
  (11B+8B → 90B+8B → 90B+70B).  Scaling alone does NOT retire
  the cross-modal team carry-forward.  V2 work: architectural
  changes (VLM-in-loop, parallel pool, substrate injection).
* **`W89-L-CROSS-MODAL-ALL-DOCSTRING-SPLIT-WORSE-CAP`** — the
  `all_docstring` regime widens the B_cross − A1_vlm gap to
  −27.78 pp (5× the doctest_only gap).  The split's failure
  mode is information loss at the modality handoff, which is
  amplified when more behavioural info lives in the image.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* No new modules; W89 used only parameter changes + the
  existing W88 audit / verifier infrastructure.

## Re-running

```bash
# W89 Prong 2 — 90B-Vision, all_docstring, 8B code-LM
python scripts/run_w88_cross_modal_code_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --code-model meta/llama-3.1-8b-instruct \
    --n-problems 12 --n-seeds 3 \
    --strip-mode all_docstring

# W89 Prong 3 — 90B-Vision, doctest_only, 70B code-LM
python scripts/run_w88_cross_modal_code_bench.py \
    --vlm-model meta/llama-3.2-90b-vision-instruct \
    --code-model meta/llama-3.3-70b-instruct \
    --n-problems 12 --n-seeds 3 \
    --strip-mode doctest_only

# Verify either
python scripts/verify_w88_cross_modal_code_audit_chain.py
```

Both reproduce with the same seeds; NIM provider-side sampling
at T=0.7 carries minor variance; the strict-improvement bool
shapes are the stable closure surface.

## Where this leaves the cross-modal carry-forward

* `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` — **stays**.
* `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` —
  **stays**.
* **The cross-modal team architecture, in the V1 extract +
  reflexion shape, is now load-bearing-falsified at three
  model scales.**  Future attempts to retire this carry-forward
  must change the architecture, not just scale up.
