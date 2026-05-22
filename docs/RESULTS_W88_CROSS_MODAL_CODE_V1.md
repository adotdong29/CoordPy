# W88 — Cross-modal code bench V1 (post-W87 empirical wave)

> **Run completed 2026-05-22.  Partial result: image is strongly
> load-bearing; multi-agent cross-modal split is NOT load-bearing
> against a same-budget single-agent VLM.  The W87 carry-forward
> `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` **does NOT
> retire**.**
>
> Pre-committed bars in `docs/RUNBOOK_W88.md`: 3 of 6 retirement
> bars MET (image-load-bearing direction); 3 of 6 NOT MET
> (multi-agent-team-better-than-single-agent-VLM direction).
> The two strict-superiority claims required for retirement
> need BOTH directions; we have only one.  Honest report below.

## TL;DR

3 seeds × 12 problems × 3 arms on the HumanEval-Visual corpus
(strip_mode = `doctest_only`).  K=5 budget on A1_vlm and B_cross.
VLM = `meta/llama-3.2-11b-vision-instruct`; code-LM =
`meta/llama-3.1-8b-instruct`; total wall 964 s; 180 text-LM
calls + 216 VLM calls.

| Arm | Mean pass@1 | Per-seed |
|----|---:|---|
| **A0_text** (text-only, no image, single-shot) | **66.67 %** | 0.667 / 0.667 / 0.667 |
| **A1_vlm** (single-agent VLM, K=5 first-pass) | **86.11 %** | 0.833 / 0.917 / 0.833 |
| **B_cross** (VLM-extract → code-LM-reflexion-K=4) | **80.56 %** | 0.833 / 0.750 / 0.833 |

* `b_cross_mean_strictly_beats_a0_text_mean = True` ✓ —
  **image is strongly load-bearing** (+13.89 pp over text-only).
* `b_cross_mean_strictly_beats_a1_vlm_mean = False` ✗ — at
  the same K=5 budget, the **single-agent VLM beats the cross-
  modal team by 5.56 pp**.

Bench Merkle root:
`37ac174e21cbe3f9...`  (full at
`results/w88/cross_modal_code/.../cross_modal_code_bench_report.json`).
**Audit chain re-derives offline: 4/4 PASS** on the 180 text + 216
VLM SHA-256 sidecar entries + 3 per-seed Merkle roots + bench
Merkle root.

## What this empirical result actually says

Two structural claims live inside the W87 carry-forward
`W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP`:

1. *"The image modality carries information that a text-only
   agent cannot recover."* — **MET.**  A0_text without image
   loses to A1_vlm with image by **+19.4 pp** (66.67 % →
   86.11 %).  On the same N=36 corpus, B_cross with VLM-extracted
   image content beats A0_text by **+13.9 pp**.  Both deltas are
   well above the +5.0 pp pre-committed margin.  The W87
   multi-modal substrate IS doing real work — it's not a
   decorative wrapper.

2. *"Organising the multi-modal team across distinct agents
   (vision agent extracts → code agent generates) beats letting
   a single multi-modal agent do the whole job at the same
   compute budget."* — **NOT MET.**  A single VLM with K=5
   independent first-pass-among-K samples (86.11 %) beats the
   cross-modal team (80.56 %) by **−5.56 pp**.  Per-seed, B_cross
   loses to A1_vlm on **0 of 3 seeds**.  The cross-modal split
   is not load-bearing on this benchmark at this scale.

**Honest interpretation:** at the 11B-VLM + 8B-code-LM scale on
docstring-stripped HumanEval, the unified VLM is strong enough at
code that splitting "extract from image" and "generate code from
extraction" introduces more information loss at the handoff than
the multi-agent diversity gains back.  The structural claim
"vision + code split is better than unified VLM" empirically
**fails** at this scale on this corpus.

## Per-seed result

| Seed | A0_text | A1_vlm | B_cross | B−A0_text | B−A1_vlm |
|----:|---:|---:|---:|---:|---:|
| 88_046_001 | 0.6667 | 0.8333 | 0.8333 | **+16.7 pp** | tie |
| 88_046_002 | 0.6667 | 0.9167 | 0.7500 | tie | **−16.7 pp** |
| 88_046_003 | 0.6667 | 0.8333 | 0.8333 | **+16.7 pp** | tie |
| **mean**   | **0.6667** | **0.8611** | **0.8056** | **+13.9 pp** | **−5.6 pp** |

The seed-2 effect (B_cross loses to A1_vlm by 16.7 pp on a 12-
problem subset) is the largest single-seed driver.  Even on the
two seeds where B_cross matches A1_vlm, B_cross does not
strictly beat — the per-seed B > A1_vlm count is **0/3**.

## Arm shape

### A0_text — text-only, no image (1 model call)

```
stripped_prompt -> text-LM(T=0.0) -> code -> executor -> pass/fail
```

The "no-image floor".  Sees only the stripped prompt (docstring
prose stays; the `>>>` example lines are removed and rendered as
the corpus image).  Establishes how much of HumanEval is
solvable from prose description alone — about **66.7 %** at this
scale on Llama-3.1-8B-Instruct.

### A1_vlm — single-agent VLM, K=5 first-pass (5 model calls)

```
stripped_prompt + image -> VLM(T=0.7, K=5 independent samples)
                       -> for each: executor pass/fail
                       -> ship first PASS; else first sample
```

The "strongest same-budget single-agent multi-modal baseline".
A literal pass@K-with-visible-test-filter on the VLM.  At K=5
this reaches **86.1 %**, exceeding A0_text by +19.4 pp — the
image's load-bearing-ness measured by the strongest single-agent
arm.

### B_cross — VLM-extract → code-LM-reflexion-K=4 (5 model calls)

```
Call 0 (VLM, T=0):
    image -> VLM extracts ">>> input \n expected_output" lines
          -> text bullets

Calls 1..4 (code-LM, T=0.7):
    call k=0:  stripped_prompt + VLM_extraction -> code -> executor
    call k>=1: stripped_prompt + VLM_extraction
             + history of (prior_candidates, executor_stderr)
             -> code -> executor
    ship first PASS; else lex-smallest CID
```

The literal "cross-modal multi-agent team" construction.
1 VLM call extracts the image's I/O examples as text; 4 code-LM
calls generate code with sequential reflexion over the extraction
plus prior failures.  Total: 5 model calls, EXACTLY matching
A1_vlm's budget.  On this corpus this **loses** to A1_vlm by
5.56 pp.

## Anti-cheat re-statement (verbatim from `RUNBOOK_W88`)

* ✓ "Same model on every arm."  VLM = `meta/llama-3.2-11b-vision-
  instruct` on A1_vlm AND on B_cross's VLM-extract step; code-LM
  = `meta/llama-3.1-8b-instruct` on A0_text AND on B_cross's
  code-generation step.
* ✓ "Same task subset per seed."  Both A0_text, A1_vlm, and
  B_cross use the SAME `select_cross_modal_subset_v1(seed)`
  output.
* ✓ "Same prompt budget."  A1_vlm spends exactly K=5 model calls;
  B_cross spends exactly K=5 (1 VLM extract + 4 code-LM
  generation/reflexion).
* ✓ "Same retry policy."  3 attempts, exponential backoff, then
  `[ERR: ...]` text stored verbatim, identical across arms.
* ✓ "No selective retries."  Each (seed, problem, arm) triple
  is exactly one set of calls.  The fact that B_cross loses on
  seed 2 by 16.7 pp is reported; the bench was not re-seeded to
  hide it.
* ✓ "Audit chain re-derives offline."  4/4 PASS via
  `scripts/verify_w88_cross_modal_code_audit_chain.py` against
  the 180 text-LM and 216 VLM sidecar SHA-256s + per-seed +
  bench Merkle roots.
* ✓ "Negative result discipline."  This RESULTS doc IS the
  negative result for the harder bar; the W87 carry-forward
  stays.  No carry-forward retirement is claimed beyond what the
  empirical bars actually support.

## What W88 cross-modal V1 closes

| DoD bullet | Status |
|---|---|
| `coordpy.cross_modal_code_bench_v1` exists for the HumanEval-Visual corpus | ✓ |
| Image is load-bearing (A1_vlm > A0_text and B_cross > A0_text) | ✓ `B_cross − A0_text = +13.89 pp`; `A1_vlm − A0_text = +19.44 pp` |
| Multi-agent cross-modal team beats single-agent VLM at same budget | ✗ `B_cross − A1_vlm = −5.56 pp` — fails |
| Audit chain (Merkle root + per-call CIDs) re-verifies offline | ✓ 4/4 PASS |
| RESULTS doc | ✓ this file |
| ≥ 3 seeds | ✓ (88_046_001 / 88_046_002 / 88_046_003) |

## What W88 cross-modal V1 does NOT close

* `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` **stays**.
  The team-organisation half of the retirement bar is not met.
  V1's specific split (VLM-extract → code-LM-generate) is not
  load-bearing-better than single-agent VLM at this scale.
  Other splits (e.g. cross-modal INJECTION at the substrate
  layer — swapping image embedding mid-LLM-forward to influence
  code generation hidden state) remain V2 work.
* The W88 cross-modal bench is a NEW honest carry-forward:
  **`W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`**:
  at the 11B-VLM + 8B-code-LM scale on HumanEval-Visual K=5,
  the VLM-extract + code-LM-generate split is empirically
  worse than the same-budget single-agent VLM baseline by
  5.56 pp.  Whether a different cross-modal architecture
  (deep injection, parallel hybrid pool, multi-modal Reflexion)
  beats single-agent VLM is V2 work.

## Honest carry-forward limitations

* **`W88-L-CROSS-MODAL-CODE-V1-DOCTEST-ONLY-CAP`** — V1 uses the
  `doctest_only` strip mode (prose description preserved; only
  the `>>>` example lines moved into the image).  The
  `all_docstring` mode (full docstring → "See image" stub) was
  added to the synthesiser but has not been run yet; the
  prediction is that it would push A0_text near 0 % (image
  becomes strictly necessary) but the B_cross vs A1_vlm
  comparison would remain similar to V1.
* **`W88-L-CROSS-MODAL-CODE-V1-NIM-VLM-CAP`** — V1 drives the
  bench through the NIM `meta/llama-3.2-11b-vision-instruct`
  endpoint.  Other VLMs (LLaVA-1.5-7B, Qwen2-VL-7B, Idefics-3-
  8B) are V2.  Provider determinism beyond temperature=0 is not
  assumed.
* **`W88-L-CROSS-MODAL-CODE-V1-SUBSET-CAP`** — V1 uses N=12
  problems × 3 seeds = 36 outcomes per arm.  The HumanEval-
  Visual corpus has 58 problems meeting the ≥ 2-doctest-line
  threshold; larger sweeps are V2.
* **`W88-L-CROSS-MODAL-CODE-V1-DOCTEST-IMAGE-PIL-CAP`** — V1
  renders the doctest text as a PIL image with the default
  monospaced font; the load-bearing fact is whether the VLM can
  READ the examples from the image, not the byte-exact image.
  The audit chain records the image bytes' SHA-256.

## Stable boundary preservation

* `coordpy.__version__` unchanged at 0.5.20.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* `coordpy.cross_modal_code_bench_v1` is explicit-import only.

## Re-running from scratch

```bash
# NVIDIA_API_KEY must be set in env.
python scripts/run_w88_cross_modal_code_bench.py \
    --vlm-model meta/llama-3.2-11b-vision-instruct \
    --code-model meta/llama-3.1-8b-instruct \
    --n-problems 12 --n-seeds 3 \
    --strip-mode doctest_only
python scripts/verify_w88_cross_modal_code_audit_chain.py
python scripts/inspect_w88_per_task_outcomes.py
```

The bench reproduces with the same seeds; NIM's temperature=0.0
arm (B_cross's VLM-extract step + A0_text) is deterministic at
the provider; the temperature=0.7 arms carry provider-side
sampling variation, so the pass@1 numbers may drift by a
problem or two on a fresh re-run.  The strict-improvement bool
shapes are the stable closure surface.
