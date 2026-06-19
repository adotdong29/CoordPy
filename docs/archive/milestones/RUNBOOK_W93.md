# W93 — Preflight-First Empirical Superiority Wave V6 (runbook)

> **Pre-commit contract for the W93 wave.  W88 → W92 spent
> roughly 30+ hours of NIM compute and produced 1 retirement
> (W89 HumanEval-70B) and 6 distinct negative results.  The
> last 5-hour W92 run cost full benchmark price to discover the
> role-specialized architecture is structurally worse than
> VLM-in-loop on HumanEval-Visual K=5 — evidence that could
> have been guessed cheaply.  W93 introduces preflight-first
> discipline: NO expensive bench run until cheap preflight
> gates pass.**
>
> Locked 2026-05-24 BEFORE any W93 candidate is built.

## What W92 actually showed (the inherited diagnosis)

Three architecturally-distinct cross-modal team approaches on
HumanEval-Visual K=5 with Llama-3.2-{11B, 90B}-Vision all LOST
to unified-VLM K=5:

* Split (VLM-extract + code-LM-generate): −5.6 to −27.8 pp.
* VLM-in-loop (single VLM multi-turn): +0.0 (3-seed) to
  −7.14 pp (7-seed; disconfirmed).
* Role-specialized (VLM-Planner + Code-Implementer-×3 +
  VLM-Verifier): **−10.71 pp** at 7 seeds.  W92 was the most
  sophisticated architecture attempted; it had the WORST mean.

**W92-L-CROSS-MODAL-HUMANEVAL-VISUAL-WRONG-BATTLEFIELD-CAP**
is canonical: HumanEval-Visual at K=5 vs unified-VLM K=5 is
empirically the wrong battlefield for cross-modal team
retirement.

Two W92 lessons:

1. **Architecture sophistication alone is not enough.**  Going
   from VLM-in-loop (simpler) to role-specialized (more
   sophisticated) made the mean delta WORSE.
2. **The K=5 unified-VLM baseline is too strong on this
   corpus** — 88 % at 7-seed means only ~12 % failure-residual
   for B to rescue, and i.i.d. sampling diversity > the
   reflexion/verifier gain at this scale.

## What W93 changes (and why)

### The expensive-benchmark gate

W93 commits to a discipline: **no expensive long benchmark run
unless cheap preflight gates pass FIRST**.

Required preflight gates (must all pass before a long run):

1. **Hypothesis written**: the candidate architecture has a
   written, falsifiable hypothesis for why it should beat A1.
2. **Cheap preflight evidence**: synthetic discriminators
   (no NIM calls) show the hypothesized advantage is
   structurally present in the architecture.
3. **Adversarial ablation**: at least one ablation (drop a
   role / drop the verifier / drop the planner) reduces the
   candidate's advantage, confirming the architectural feature
   is load-bearing internally.
4. **Budget accounting validated locally**: the candidate's
   model-call budget matches A1 exactly under all branches of
   the pipeline.
5. **Benchmark justification**: the chosen benchmark is
   either (a) a new battlefield with stronger justification
   than the W88/W90/W91/W92 choices, or (b) the same
   battlefield with a radically different hypothesis.

A candidate that fails any of these 5 gates is killed BEFORE
the expensive run.  This is non-negotiable.

### W93 deliverables

1. **`coordpy/failure_cluster_miner_v1.py`** — analyses
   existing W88–W92 bench reports + sidecars to identify
   common B failure modes.  Produces a structured failure-
   cluster JSON.  Cheap (no NIM calls); runs in seconds.
2. **`coordpy/cross_modal_preflight_harness_v1.py`** — runs
   cheap synthetic discriminators against a candidate
   architecture.  No NIM calls.  Tests: role-dropout, budget
   accounting, per-role contribution.  Returns a structured
   preflight verdict.
3. **Failure-mode diagnosis note** — concrete writeup of what
   the W88–W92 evidence says about B's failure clusters.
4. **2–3 candidate architectures** — each with explicit
   hypothesis and preflight evaluation.  Killed early if
   they fail preflight.
5. **Documented preflight outcomes** — which candidates
   survived; which were killed; why.

### What W93 does NOT do (unless preflight earns it)

W93 does NOT launch another expensive HumanEval-Visual
benchmark run.  W93 may launch ONE expensive benchmark run
ONLY if:

* A surviving candidate architecture passes all 5 preflight
  gates.
* The candidate is for a NEW benchmark (not HumanEval-Visual
  at K=5) OR has a radically different cross-modal
  architecture from W88–W92.
* The benchmark is justified per gate #5.

If no candidate survives preflight, W93's deliverable is the
infrastructure + failure analysis itself.  The discipline
shift is the milestone.

## Pre-committed retirement bars (unchanged from W88–W92)

If an expensive run does land in W93, the retirement bars
are unchanged:

* Same-budget code benchmark: 4 bars (B>A1 mean / margin ≥
  +1 pp / B>A0 / per-seed strict majority).
* Cross-modal: 6 bars (image-load-bearing × 3 +
  team-load-bearing × 3 with +5 pp margin and per-seed
  majority).

W93 retirement decision is the SAME shape.  Only the
discipline of WHEN to spend the expensive budget changes.

## Anti-cheat (carry-forward)

All W88–W92 anti-cheat clauses carry forward.  W93-specific:

* The failure miner reads ONLY committed W88–W92 evidence.
  No re-running of NIM; no cherry-picking.
* The preflight harness uses synthetic discriminators that
  are pre-committed in code.  No "discriminators tuned to
  make my candidate look good".
* If a candidate fails preflight, it's documented as failed.
  Preflight failures count as honest negative evidence.
* If no candidate survives preflight, NO expensive run.  The
  discipline is the deliverable.

## Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* New W93 modules (`failure_cluster_miner_v1`,
  `cross_modal_preflight_harness_v1`) are explicit-import only.

## Operational plan

1. Build `coordpy/failure_cluster_miner_v1.py` (~1h).
2. Run miner against W88–W92 artifacts; produce diagnosis
   writeup (~30 min).
3. Build `coordpy/cross_modal_preflight_harness_v1.py` (~1h).
4. Design 2–3 candidate architectures with explicit
   hypotheses (~30 min).
5. Preflight each candidate (~30 min total).
6. Document survival / kills.
7. IF any candidate survives preflight: write expensive-run
   runbook addendum; launch.  IF NOT: commit discipline +
   infrastructure as W93's deliverable.
