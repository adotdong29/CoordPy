# W89 — Post-W88 Empirical Superiority Wave V2 (runbook)

> **Pre-commit contract for the W89 retry wave.  W88's V1 wave
> closed two gaps structurally (image is load-bearing; reflexion
> gap closed from −8.9 pp to −3.33 pp) but did NOT retire either
> canonical carry-forward.  W89 attacks the empirical bar
> aggressively: stronger model on the HumanEval prong, stronger
> VLM + harder image-load-bearing regime on the cross-modal
> prong.**
>
> Locked 2026-05-22 BEFORE any W89 bench run.  Retirement bars
> are the same shape as `docs/RUNBOOK_W88.md` (no relaxation);
> only the model + corpus shape change.

## What W88 actually showed (the diagnosis)

**HumanEval prong (W88 negative):** B sequential-reflexion-K=5
on NIM Llama-3.1-8B-Instruct beat A0 by +7.78 pp but lost A1 by
−3.33 pp.  The gap to A1 closed from W86's −8.9 pp by 5.6 pp.
Per-seed B beats A1 on 0/3 (tied on 2/3, lost 10 pp on 1).

**Structural reason for the loss:** at the 8B scale on a
moderate-difficulty benchmark, independent K=5 sampling is
near-optimal — there's not enough headroom for reflexion to
add value.  Specifically:

* A1 K=5 mean was 74.4 % — the "ceiling" leaves only ~26 pp of
  fail-rate for B's reflexion to recover from.
* The 8B model's ability to use stderr-to-fix mapping is
  marginal: reflexion's per-turn delta is comparable to
  independent-sample diversity gain.
* The structural overhead of running sequential conditioning
  (longer prompts, narrower attention) erodes the value of
  feedback at this scale.

This is consistent with the Reflexion / Self-Debug literature:
clear wins reported at GPT-3.5+ / GPT-4 scale, marginal at
smaller-scale instruction-tuned models.

**Cross-modal prong (W88 partial):** Image is load-bearing
(+13.9 pp B_cross over A0_text), but B_cross lost to single-
agent A1_vlm by −5.56 pp.

**Structural reason for the loss:** Llama-3.2-11B-Vision is
already strong enough at code that the VLM-extract +
code-LM-generate split introduces extraction-handoff loss
faster than the multi-agent organisation gains back.  In
particular:

* The W88 corpus used `strip_mode = doctest_only` — the
  docstring's prose description stays in the prompt.  Both
  A1_vlm and B_cross's code-LM see the prose; the image only
  adds the I/O examples.  The image's marginal information
  content is low.
* At K=5, A1_vlm has 5 independent shots at the unified task;
  B_cross has 1 extract + 4 code-LM turns where the code-LM
  CAN'T verify its own image understanding.
* If the VLM extracts perfectly, the code-LM does fine; if the
  VLM mis-extracts, the code-LM has no way to recover.

## What W89 changes (and why)

### Prong 1 — HumanEval reflexion at **Llama-3.3-70B-Instruct** scale

The same `coordpy.humaneval_reflexion_bench_v1` infrastructure
ships unchanged; only the model swaps from
`meta/llama-3.1-8b-instruct` to `meta/llama-3.3-70b-instruct`.

Why this is the right pivot:

* The Reflexion / Self-Debug literature shows clear same-budget
  wins at 70B-class and above.  W88's 8B-scale loss is the
  documented "small-model reflexion struggles" phenomenon.
  Testing at 70B is the direct empirical test of "does
  reflexion beat self-consistency at the scale where the
  literature says it does."
* Same model on both arms preserves same-budget fairness — A1
  also runs at 70B.
* Same task subset per seed.
* If B beats A1 at 70B, the W86 / W88 carry-forwards retire
  with the caveat "demonstrated at 70B scale; 8B-scale
  carry-forwards persist as their own
  `*-SMALL-MODEL-CAP` notes".

### Prong 2 — Cross-modal at `all_docstring` + **Llama-3.2-90B-Vision**

The same `coordpy.cross_modal_code_bench_v1` infrastructure
ships unchanged; the V1 `all_docstring` strip mode (added in
W88 but not run live) is invoked; the VLM swaps from 11B to
**`meta/llama-3.2-90b-vision-instruct`**.  The code-LM stays
at Llama-3.1-8B for budget asymmetry honesty.

Why this is the right pivot:

* `all_docstring` removes the prose description AND the
  doctest examples from the text prompt; the image is the ONLY
  source of behavioural information.  This pushes A0_text near
  zero (no info) and forces the comparison to be entirely
  about image utility.
* At 90B-Vision, the unified VLM is even stronger at code than
  at 11B; B_cross's split MUST add genuine value to win.
* This is a HARDER test for B_cross than W88's regime — if
  B_cross beats A1_vlm at 90B + all_docstring, the cross-modal
  team is unambiguously load-bearing.
* If B_cross loses, the result is honestly negative at the
  STRONGER VLM scale, and the W87 carry-forward stays.

## Pre-committed success criteria

**Retirement bars are the same as W88 (no relaxation).**

### Prong 1 — Retires `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` and `W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP` iff ALL 4 bars met:

1. `b_mean_strictly_beats_a1_mean = True`
2. `b_mean − a1_mean ≥ +1.0 pp`
3. `b_mean_strictly_beats_a0_mean = True`
4. B beats A1 on more than half the seeds.

If 4 of 4: BOTH carry-forwards retire with the W89 evidence
appended.  The retired carry-forwards' anti-cheat clauses are
preserved verbatim in the new claim.

If 3 of 4 or fewer: carry-forwards stay; a new
`W89-L-HUMANEVAL-REFLEXION-70B-*` carry-forward is added
documenting the additional negative evidence.

### Prong 2 — Retires `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` and `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP` iff ALL 6 bars met:

1. `b_cross_mean_strictly_beats_a0_text_mean = True`
2. `b_cross_mean_strictly_beats_a1_vlm_mean = True`
3. B − A0_text margin ≥ +5.0 pp
4. B − A1_vlm margin ≥ +5.0 pp
5. B beats A0_text on more than half the seeds.
6. B beats A1_vlm on more than half the seeds.

If 6 of 6: BOTH carry-forwards retire with W89 evidence.

If fewer: carry-forwards stay; new `W89-L-CROSS-MODAL-*`
carry-forward documents the additional negative evidence at
the harder regime.

## Anti-cheat clauses (verbatim from `RUNBOOK_W88`)

All W88 anti-cheat clauses carry forward unchanged.  The new
W89 clauses:

* The MODEL change is documented EXPLICITLY: W89 Prong 1 uses
  Llama-3.3-70B-Instruct (≠ W88's Llama-3.1-8B-Instruct).
  Both A0/A1/B in the same Prong 1 run use the same 70B model.
* W89 Prong 2 uses Llama-3.2-90B-Vision-Instruct as VLM (≠
  W88's 11B); both A1_vlm and B_cross's vision step use 90B.
  Both A0_text and B_cross's code-LM step use Llama-3.1-8B
  (≠ Llama-3.3-70B; honest asymmetry — code-LM stays at 8B so
  the split is FAIR vs A1_vlm's 90B VLM).
* Strip mode change is documented EXPLICITLY: W89 Prong 2 uses
  `strip_mode = all_docstring`.  Per-problem prompt no longer
  carries any behavioural prose — only signature + "See image"
  stub.  This is the HARDER regime; A0_text expected near
  zero.
* No selective retries.  No re-seeding.  Same-budget,
  same-task-subset discipline preserved.

## What W89 does NOT do

* W89 does NOT change the B-pipeline architecture.  The W88
  designs (sequential reflexion for HumanEval; VLM-extract +
  code-LM-reflexion for cross-modal) ship unchanged.  The only
  W89 changes are the model and corpus shape.  This isolates
  the "model scale matters" hypothesis cleanly.
* W89 does NOT relax retirement bars.  If neither prong
  retires, the carry-forwards stay.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* All W89 work is parameter changes + new RESULTS docs + new
  registry entries; no new explicit-import modules unless an
  unforeseen architectural change is required.

## Operational plan

1. Diagnosis note documented (this runbook + the W88 RESULTS
   docs are the load-bearing diagnosis surfaces).
2. Launch Prong 1 (70B HumanEval) in background.  Wall ~60-90
   min.
3. Launch Prong 2 (90B-Vision cross-modal, `all_docstring`)
   in parallel.  Wall ~20-30 min.
4. Verify audit chains; produce RESULTS docs; update honesty
   surfaces; commit; ask for push approval.

The W88 automation stack (NIM stdlib HTTPS driver, JSONL
sidecars, content-addressed per-seed + bench Merkle, portable
`latest_run.txt` pointer, offline verifier, per-task inspector)
is reused unchanged — only the run command's model / mode args
flip.
