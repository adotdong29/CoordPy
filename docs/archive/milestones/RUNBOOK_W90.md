# W90 — Post-W89 Empirical Superiority Wave V3 (runbook)

> **Pre-commit contract for the W90 wave.  W89's V2 wave RETIRED
> two HumanEval carry-forwards at Llama-3.3-70B-Instruct scale
> but left three real frontiers open: (1) generalize the
> same-budget multi-agent superiority claim to a SECOND
> published benchmark; (2) replace the W88/W89 falsified
> cross-modal split with an architecture that actually beats
> the unified VLM at same budget; (3) ideally test whether the
> W89 win extends to GSM8K at 70B.**
>
> Locked 2026-05-22 BEFORE any W90 bench run.  Retirement bars
> are the same shape as `RUNBOOK_W88.md` / `RUNBOOK_W89.md`
> (no relaxation).

## What W89 actually showed (the inherited diagnosis)

* **HumanEval at 70B** — W89 P1 retired both `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN`
  and `W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`
  by running the W88 sequential-reflexion B-pipeline on
  Llama-3.3-70B-Instruct: B 91.1 % > A1 85.6 % by +5.56 pp, with
  B beating A1 on 2/3 seeds.
* **Cross-modal at three scales** — W88 V1 (11B-V + 8B doctest_only),
  W89 P2 (90B-V + 8B all_docstring), W89 P3 (90B-V + 70B doctest_only)
  all FAILED the team-organisation bar: B − A1_vlm = −5.56 / −27.78 /
  −5.56 pp.  Image-load-bearing direction PROVEN at all three.
* **8B HumanEval gap persists** — `W89-L-HUMANEVAL-REFLEXION-V2-8B-CAP`
  records that smaller models still lose; W89 retirement is
  scoped to 70B.
* **GSM8K (W85) untested at 70B** — the W85 negative was at
  Llama-3.1-8B-Instruct only.  Whether the model-scale lever
  also retires the GSM8K carry-forward is open.

## What W90 changes (and why)

### Prong 1 — MBPP+ at Llama-3.3-70B-Instruct (NEW BENCHMARK)

A SECOND published code benchmark (MBPP-sanitized,
`google-research/mbpp/sanitized-mbpp.json`).  Same architecture
as W88/W89's sequential-reflexion B; same K=5 budget; same
70B model.  Anti-cheat: corpus SHA-256-anchored against the
canonical upstream.

Why this is the right pivot:

* **Generalization is the question.**  W89's HumanEval-only
  retirement is one bench; the next claim must extend.  MBPP
  is the most-cited code-LM eval after HumanEval; if the W88
  sequential-reflexion architecture works there too, the
  generalization claim is genuine.
* **Same architecture, same model, different benchmark** —
  isolates the benchmark axis cleanly.  No new B-pipeline.
* **Public + recognized** — MBPP appears in every code-LM
  paper; reviewers and external auditors can compare numbers
  directly.

If `b_mean_strictly_beats_a1_mean` on MBPP at 70B → second
published-benchmark retirement.

### Prong 2 — Cross-modal VLM-in-loop reflexion (NEW ARCHITECTURE)

The W88/W89 cross-modal V1 / V2 used a "VLM-extract + code-LM-
generate + reflexion" split — falsified at 3 model scales.
Prong 2 is a **structural pivot**: drop the split entirely; keep
the SAME VLM in the loop for all 5 turns, conditioning each
reflexion turn on (prior_candidate, executor_stderr) with the
IMAGE in context every turn (no extraction-handoff).

Why this is the right pivot:

* **The W88/W89 failure mode is information loss at the modality
  handoff.**  VLM-in-loop never extracts to text; the image is
  re-encoded on every turn.
* **Symmetric to W89 Prong 1.**  W89 P1 proved sequential-
  reflexion beats independent K=5 sampling on text at 70B.
  Prong 2 tests whether the same multi-turn-with-feedback shape
  wins on multimodal at 90B-Vision.
* **Direct fair budget**: A1_vlm = K=5 independent VLM samples;
  B_vlm_loop = K=5 sequential VLM samples with stderr-history
  conditioning.  Same model, same total VLM calls.

If `b_vlm_loop_mean_strictly_beats_a1_vlm_mean` → retirement
of `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
(specifically the *split-vs-unified* claim — the W88 split's
failure stands, but the broader "multi-agent / multi-turn at
multimodal substrate beats first-pass-K" claim is established).
The `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP`
(substrate-injection) is NOT directly addressed by this prong
(it's about mid-LLM-forward injection at the hidden-state
layer, not benchmark superiority); it stays.

### Prong 3 — GSM8K retry at 70B (STRETCH)

Re-run the W85 `coordpy.gsm8k_real_bench_v1` infrastructure on
NIM `meta/llama-3.3-70b-instruct` instead of `meta/llama-3.1-8b-instruct`.

Why this is the right stretch:

* **Cheapest possible additional retirement test.**  The W85
  bench shipped; just swap the model env var.
* **Tests whether W89's model-scale lever generalizes beyond
  HumanEval.**  If 70B retires GSM8K too, the
  `W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-BEAT-SELF-CONSISTENCY-CAP`
  carry-forward also retires.
* **Honest caveat**: W85's B-pipeline is the W86 shape
  (solver + alt-solver + critic + reviser + judge), NOT the W88
  sequential-reflexion shape.  If W85-B-at-70B loses, that's
  evidence the OLD architecture is the limiter, not just model
  scale.  Either outcome is informative.

## Pre-committed success criteria

**No relaxation from W88/W89.**

### Prong 1 — MBPP retirement bars

Retires `W89-L-HUMANEVAL-REFLEXION-V2-HUMANEVAL-K5-SCALE-CAP` (the
"only tested on HumanEval" cap) iff ALL 4 bars:

1. `b_mean_strictly_beats_a1_mean = True`
2. `b_mean − a1_mean ≥ +1.0 pp`
3. `b_mean_strictly_beats_a0_mean = True`
4. B beats A1 on more than half the seeds.

If 4 of 4: the W89 retirement EXTENDS to a second published
benchmark — strong generalization claim.

If 3 of 4 or fewer: new carry-forward
`W90-L-MBPP-REFLEXION-V1-NOT-BEATEN-BAR-FAIL` records the
benchmark-specific limitation.

### Prong 2 — Cross-modal VLM-in-loop retirement bars

Retires `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`
iff ALL 6 bars (same shape as W88):

1. `b_vlm_loop_mean_strictly_beats_a0_text_mean = True`
2. `b_vlm_loop_mean_strictly_beats_a1_vlm_mean = True`
3. B − A0_text margin ≥ +5.0 pp
4. B − A1_vlm margin ≥ +5.0 pp
5. B beats A0_text on more than half the seeds.
6. B beats A1_vlm on more than half the seeds.

Note: this prong specifically RETIRES the W88 split's failure
by providing a different architecture that DOES beat A1_vlm.
The original W88-L description identifies "VLM-extract +
code-LM-generate" as the falsified shape; W90 P2 replaces it
with VLM-in-loop and tests whether THAT shape wins.

`W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` stays — it's
about substrate-level injection, not benchmark superiority.

### Prong 3 — GSM8K retirement bars

Retires `W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-BEAT-SELF-CONSISTENCY-CAP`
iff ALL 4 bars same as Prong 1.

## Anti-cheat clauses

All W88/W89 anti-cheat clauses carry forward.  W90-specific:

* **Prong 1**: MBPP-sanitized corpus SHA-256 anchored against
  the canonical upstream (SHA `ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`
  on `google-research/google-research` master HEAD as of
  2026-05-22).  Same task subset per seed across arms.  K=5 on
  A1 and B; same model on all three arms.
* **Prong 2**: VLM-in-loop is single-model-multi-turn; the
  "multi-agent" label refers to multiple ROLES across turns
  (initial solver, reflexion-driven refiner, reflexion-driven
  repairer).  All turns see the image; no text-only extraction
  handoff.  A1_vlm uses the SAME VLM at the same K=5 budget.
* **Prong 3**: GSM8K-test corpus SHA-256 unchanged from W85.
  Same task subset per seed across arms.  Same K=5 on A1 and B.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* W90 new modules (`mbpp_reflexion_bench_v1`,
  `cross_modal_vlm_loop_bench_v1`) are explicit-import only;
  not re-exported through `coordpy.__init__` or
  `coordpy.__experimental__`.

## Operational plan

1. Write the W90 diagnosis pointer (this runbook + the
   W88_FAILURE_DIAGNOSIS still apply).
2. Launch GSM8K-at-70B (P3) in background — cheapest, fastest.
3. Build MBPP module + driver + tests; launch in background.
4. Build VLM-in-loop module + driver + tests; launch in
   background.
5. Verify audit chains; produce RESULTS docs; update honesty
   surfaces; commit; ask for push approval.
