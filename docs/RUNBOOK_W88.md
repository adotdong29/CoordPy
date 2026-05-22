# W88 — Post-W87 Empirical Superiority Wave V1 (runbook)

> **The blocker era is over.  The next era is empirical
> superiority.**  W88 is the first programme milestone explicitly
> oriented around the empirical bars in
> `docs/HONEST_FRAMING_POST_W87.md` §"What would actually constitute
> 'solving' it", **not** another closure cycle.
>
> Pre-committed by `docs/RUNBOOK_W88.md` 2026-05-22 BEFORE any
> bench run.  This file is the falsifiable contract; the result
> docs (`docs/RESULTS_W88_*.md`) report what actually happened
> against it, including negative results.

## What W88 targets

Two canonical carry-forwards from `HONEST_FRAMING_POST_W87`:

| Carry-forward | What W88 attempts | Retirement bar |
|---|---|---|
| `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` | Build a stronger same-budget B-pipeline (executor-guided sequential reflexion) that empirically beats A1 (first-pass-among-K=5) on the **mean pass@1** across ≥3 seeds × ≥30 problems on the same NIM Llama-3.1-8B-Instruct corpus as W86. | `b_mean_strictly_beats_a1_mean = True` |
| `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` | Build a cross-modal team where a VLM agent extracts image-borne information into structured text, a code-LM agent generates code conditioned on that extraction, and an executor verifies — beating both the single-modal text-only baseline AND the same-budget single-agent VLM baseline on an image-conditioned coding task corpus. | `b_cross_modal_mean_strictly_beats_a1_vlm_mean = True` |

The W85 GSM8K negative result
(`W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-BEAT-SELF-CONSISTENCY-CAP`)
is **out of scope for W88** — the natural attack vector for that
carry-forward is tool-augmented reasoning (Python interpreter for
arithmetic), which is a separate research direction and merits its
own milestone.  W88 stays focused on the two carry-forwards above.

## Pre-committed success criterion

**Primary head-to-head**: `coordpy.humaneval_reflexion_bench_v1`

Bench shape: ≥3 seeds × ≥30 problems × 3 arms × NIM
Llama-3.1-8B-Instruct.  Same model, same task subset (deterministic
per seed via `select_humaneval_subset_v1`), same total model-call
budget per arm (`K=5` for A1 and B).

* **Arms**:
  * **A0** — stock single-shot at `T=0.0`. 1 model call.
  * **A1** — first-pass-among-K=5 self-consistency at `T=0.7`. 5
    independent samples; executor verdict on each; ship first
    PASS, fall back to first sample. **The W86 baseline that
    beat B; this is the strongest published-published same-budget
    baseline.**
  * **B** — `executor_guided_sequential_reflexion_v1`. 5
    sequential model calls, each conditioned on the cumulative
    history of prior candidates + their stderr tails.  Ship the
    first PASS; if no PASS, ship the candidate with the longest
    matching-prefix to the prompt (a content-addressed
    deterministic tie-breaker).

* **Pre-committed strong-success bars** (BEFORE the run):
  1. `b_mean_strictly_beats_a1_mean = True` — B's mean pass@1
     strictly exceeds A1's mean pass@1 across the seed set.
  2. B's mean pass@1 minus A1's mean pass@1 ≥ **+1.0 pp** — not
     just a tiebreak; a measurable structural advantage.
  3. `b_mean_strictly_beats_a0_mean = True` — B also beats the
     stock baseline.  (The W86 closure already shipped this; W88
     reproduces.)
  4. Per-seed: B beats A1 on ≥ 2 / 3 seeds (so the win is not a
     single-seed artefact).

* **Pre-committed partial-success bars**:
  1. `b_mean_strictly_beats_a1_mean = True` but with margin <
     +1.0 pp.  Reported honestly; carry-forward NOT retired
     because the effect size is too small to claim structural
     superiority on this scale.
  2. Per-seed: B beats A1 on exactly 1 / 3 seeds.  Reported as
     "B and A1 are statistical ties at this scale; W88 does NOT
     retire the carry-forward".

* **Pre-committed failure**:
  * `b_mean_strictly_beats_a1_mean = False`.  Reported as the W88
    HumanEval negative result; the W86 carry-forward stays.

**Cross-modal head-to-head**: `coordpy.cross_modal_code_bench_v1`

Bench shape: ≥3 seeds × ≥10 problems × 3 arms.  Image-conditioned
code corpus (deterministically synthesised from a public HumanEval
subset; per-problem docstring examples and/or full docstring are
re-rendered as a PIL image; the code-relevant information is then
ONLY in the image).  Same total model-call budget per arm.

The synthesis recipe has two ``strip_mode`` settings:

* ``doctest_only`` (default V1 attempt) — strips only the
  ``>>>``-prefixed lines from the docstring; the prose
  description stays in the prompt.  The weakest "image-is-
  load-bearing" mode.  If V1 wins under this mode, the result is
  the strongest possible (image illustrating I/O examples adds
  measurable value on top of a full prose spec).
* ``all_docstring`` — replaces the entire docstring with a "See
  image" stub.  The strongest "image-is-load-bearing" mode —
  no behavioural info in the text at all.  Used as the fall-
  back if ``doctest_only`` produces a tie / negative result.

* **Arms**:
  * **A0_text** — text-only LLM single-shot; **no image
    access**.  Establishes the floor: how often does a strong
    code-LM solve the problem from a stripped prompt alone?
  * **A1_vlm** — single-agent VLM, K=5 same-budget self-consistency
    on (image + stripped text); executor first-PASS.  The
    strongest single-agent multimodal baseline.
  * **B_cross** — cross-modal team:
    1. VLM agent extracts I/O examples from the image into a
       structured text description.
    2. Code-LM agent generates code conditioned on (stripped text
       + VLM extraction).
    3. Executor verifies; if FAIL, the code-LM does one
       executor-conditioned revision.
    Total: 1 VLM call + ≤ 4 code-LM calls = 5 model calls
    (matching A1_vlm's K=5).

* **Pre-committed strong-success bars**:
  1. `b_cross_modal_mean_strictly_beats_a0_text_mean = True` —
     cross-modal team beats the no-image-access baseline (the
     image is load-bearing).
  2. `b_cross_modal_mean_strictly_beats_a1_vlm_mean = True` —
     cross-modal team beats the same-budget single-agent VLM
     baseline (the multi-agent split is load-bearing).
  3. Margin ≥ +5 pp on both vs A1_vlm.  Cross-modal benchmarks
     have higher variance than HumanEval; the bar is higher to
     guard against single-seed luck.

* **Pre-committed partial / failure bars**: as above for the
  HumanEval case, with the carry-forward stay/retire decisions
  inverted.

* **`W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP` retirement
  policy**: ONLY retired if both strong-success bars 1 and 2
  above are met.  "Cross-modal beats single-modal" alone is NOT
  sufficient — that would just demonstrate that "vision helps",
  not that "multi-agent cross-modal organisation helps".  The
  load-bearing claim is the harder one (B > A1_vlm).

## Anti-cheat clauses (carried verbatim from W85 / W86 + new)

1. **Same model on every arm** of every head-to-head.  NIM Llama-3.1-8B
   on the HumanEval primary; same VLM+code-LM on both A1_vlm and
   B_cross's VLM/code agents on the cross-modal bench.
2. **No selective retries**.  Each (seed, problem, arm) triple is
   exactly one set of calls.  No "re-run only the failing seeds".
3. **Same task subset per seed** across arms.  The W86
   `select_humaneval_subset_v1` discipline is preserved
   verbatim.
4. **Same prompt budget**.  A1 and B both use `K=5` model calls.
5. **Same retry policy** on transient provider errors (3 attempts,
   exponential backoff, then `[ERR: ...]` text stored verbatim).
6. **Executor truth = full test block**.  The HumanEval executor
   runs the canonical `problem.test` (which already encodes the
   hidden tests).  No arm has access to a different oracle.
7. **Cross-modal anti-cheat**: A0_text does NOT see the image
   (this is the no-modality baseline by design — the W88 claim
   is "the image is load-bearing AND the cross-modal team
   organisation is load-bearing").
8. **Cross-modal anti-cheat**: The VLM in B_cross is the SAME
   model as in A1_vlm.  The code-LM in B_cross is a SEPARATE
   model from the VLM (this is the "split is load-bearing"
   claim).  No model swap between arms.
9. **Cross-modal anti-cheat**: The image corpus is synthesised
   deterministically from the public HumanEval corpus (no
   private data); the synthesis recipe is SHA-256-anchored.
10. **Audit chain re-verifies offline**.  Every per-call CID +
    per-seed Merkle + bench Merkle re-derive byte-for-byte from
    the persisted JSONL sidecar.
11. **Negative result discipline**.  If either head-to-head fails
    its strong-success bar, the result IS published with the
    same audit chain; the carry-forward stays; the docs do not
    overstate.

## Module inventory

To be built in this milestone:

* `coordpy/humaneval_reflexion_bench_v1.py` — new bench module
  with the B = sequential-reflexion-K=5 arm.  Mirrors
  `humaneval_real_bench_v1.py`'s capsule + Merkle discipline.
* `coordpy/cross_modal_code_bench_v1.py` — new bench module with
  the B_cross arm (VLM-extract + code-LM-generate + executor) and
  the image-corpus synthesiser.
* `scripts/run_w88_humaneval_reflexion_bench.py` — driver, NIM
  + Ollama paths.
* `scripts/run_w88_cross_modal_code_bench.py` — driver.
* `scripts/verify_w88_humaneval_reflexion_audit_chain.py` —
  offline verifier.
* `scripts/verify_w88_cross_modal_code_audit_chain.py` — offline
  verifier.
* `tests/test_w88_humaneval_reflexion_v1.py` — CI tests.
* `tests/test_w88_cross_modal_code_v1.py` — CI tests.
* `scripts/colab_w88_humaneval_reflexion.ipynb` — frontier-scale
  Colab notebook (Qwen2.5-Coder-7B-Instruct on A100 + execution
  on a stronger code model as a cross-model robustness check).
* `docs/RESULTS_W88_HUMANEVAL_REFLEXION_V1.md` — results doc
  (only written after the run produces actual numbers).
* `docs/RESULTS_W88_CROSS_MODAL_CODE_V1.md` — results doc.

## Stable boundary preservation

* `coordpy.__version__` stays at `0.5.20`.
* `coordpy.SDK_VERSION` stays at `coordpy.sdk.v3.43`.
* **NO PyPI publish**.
* `coordpy/__init__.py` untouched.
* All W88 modules are explicit-import only.
* No co-author trailers / generated-by markers on any commit.

## Honest framing rules

If W88 wins on either head-to-head:

* Update `docs/HONEST_FRAMING_POST_W87.md` to reflect EXACTLY
  which carry-forward was retired and EXACTLY how.  Do not
  overstate.  Keep the unretired carry-forwards.
* Add the retirement entry to `docs/THEOREM_REGISTRY.md`.
* Add the new RESULTS doc to `docs/HOW_NOT_TO_OVERSTATE.md`
  with explicit forbidden phrasing.
* Update `docs/RESEARCH_STATUS.md` with the result.
* Update `CHANGELOG.md`.

If W88 loses on either head-to-head:

* Add a NEW `W88-L-*-CAP` carry-forward to
  `docs/THEOREM_REGISTRY.md` documenting the loss.
* Update `docs/HONEST_FRAMING_POST_W87.md` ONLY if a stronger
  honest framing of the carry-forward emerges from the W88
  evidence.
* Update `CHANGELOG.md` with the negative result.

Reading order for any future result claim: this runbook FIRST,
then the matching `RESULTS_W88_*.md`, then `HONEST_FRAMING_POST_W87`.
If they disagree about whether a carry-forward retired, this
file's pre-commit + the RESULTS doc's bench-derived bools win;
HONEST_FRAMING_POST_W87 is updated to match, not vice versa.
