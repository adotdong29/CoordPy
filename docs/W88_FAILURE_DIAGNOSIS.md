# W88 failure diagnosis — what went wrong, what to attack next

> Internal-facing diagnosis written 2026-05-22, BEFORE the W89
> retry runs.  Documents the structural reasons W88's two
> negative / partial results landed, and the specific levers
> W89 pulls in response.  Pair with `docs/RUNBOOK_W89.md` for
> the pre-commit contract of the retry.

## W88 HumanEval reflexion — why B lost A1 by 3.33 pp

**Setup**: 3 seeds × 30 HumanEval problems × NIM
Llama-3.1-8B-Instruct.  B = K=5 sequential reflexion (each turn
conditioned on cumulative (candidate, executor_stderr) history).
A1 = K=5 first-pass-among-K independent samples.  Same model,
same task subset per seed, same K=5 budget.

**Result**: A0 63.3 % / A1 74.4 % / B 71.1 %.  B − A1 = −3.33 pp.
Per-seed (88028001 / 88028002 / 88028003) B−A1 = (−10 / 0 / 0)
pp.  B beats A1 on 0/3 seeds (ties on 2/3, loses 10 pp on
seed 1).

**Structural reasons for the loss**:

1. **Ceiling is too close at this scale.**  A1 at K=5 already
   captures 74.4 % of solvable problems.  Only ~25.6 % of
   problem-attempts are available for B to "rescue" via
   reflexion.  Even a 30 % rescue rate on that subset would
   yield only ~8 pp added — less than the gap to A1.

2. **Reflexion at 8B is marginal.**  The Reflexion / Self-Debug
   literature reports clear same-budget wins at GPT-3.5 (175B)
   and GPT-4 scale, but the effect attenuates significantly
   at instruction-tuned 7B / 8B models.  Llama-3.1-8B-Instruct
   can parse stderr but its derived-from-stderr fixes are
   often the same diversity an independent T=0.7 sample would
   have produced.

3. **Sequential conditioning hurts attention quality.**  Each
   reflexion turn embeds the full history of prior candidates
   + their stderr — prompts grow to 3-4 KB by turn 5.  Within
   the 8B context window this is fine in tokens but the
   attention-weighted focus shifts away from the task
   description toward the history.  This is plausibly why
   B's seed-1 result (−10 pp) is anomalously bad — when early
   attempts have misleading bugs, the history makes the
   model fixate on the wrong correction.

4. **A1's "first PASS" oracle is essentially pass@K.**  Because
   the W86 executor uses the FULL `problem.test` block (not
   a filtered visible-tests-only subset), A1's "first sample
   that passes the executor" is equivalent to "any of K
   samples solves the problem" — i.e., A1 K=5 ≈ pass@5
   reported as pass@1.  This is a STRONG baseline that B
   must exceed by either finding solutions A1's K=5 random
   samples cannot find, OR by improving an A1 K=5 PASS to a
   "better PASS" (which is not meaningful since PASS is
   binary in HumanEval).  The only way for B to win is to
   find solutions OUTSIDE A1's K=5 pool — which is
   structurally hard at 8B scale.

**What W89 changes for Prong 1**: swap to
**Llama-3.3-70B-Instruct**.  At 70B, the literature
(Shinn et al. 2023; Chen et al. 2023) is consistent that
reflexion adds real value over independent sampling at
fixed budget — the model is competent enough to map stderr
→ fix.  Same K=5, same task subset discipline.

**Risk**: even at 70B, A1 may jump so much (say 88 %+) that
the ceiling closes again and B can't beat A1.  Honest report
will tell us.

## W88 cross-modal code — why B_cross lost A1_vlm by 5.56 pp

**Setup**: 3 seeds × 12 HumanEval-Visual problems × NIM
Llama-3.2-11B-Vision-Instruct + Llama-3.1-8B-Instruct.  Corpus
synthesised in `doctest_only` strip mode (docstring prose
stays; only `>>>` lines moved to image).  Same model on each
arm's task; K=5 budget.

**Result**: A0_text 66.7 % / A1_vlm 86.1 % / B_cross 80.6 %.
B_cross − A1_vlm = −5.56 pp.  Per-seed B beats A1 on 0/3.

**Structural reasons for the loss**:

1. **A0_text is too strong because prose stays.**  At
   `doctest_only`, the docstring's prose description (e.g.,
   "Check if any two numbers are closer than threshold") is
   preserved in the prompt.  This gives A0_text a real
   chance (66.7 %) — much of HumanEval is solvable from
   prose alone.

2. **A1_vlm is too strong because it sees both image AND
   prose.**  At K=5 with `doctest_only`, the unified VLM has
   access to the FULL task description (prose + image) on
   every call.  Reaching 86.1 % is consistent with a strong
   VLM doing well at code given complete context.

3. **B_cross's extraction-handoff is information-lossy.**  The
   VLM extracts the doctest examples as text bullets (1 model
   call), then 4 code-LM calls implement code from
   (stripped_prompt + extraction).  If the VLM extraction has
   any noise (typo in a number, missed example, ambiguous
   format), the code-LM CANNOT see the image to verify or
   correct.  Each VLM extraction error becomes a propagated
   error through all 4 code-LM turns.

4. **Code-LM diversity is restricted vs A1_vlm.**  A1_vlm has
   5 independent samples from the unified-model distribution;
   B_cross has 1 VLM extract + 4 code-LM samples conditioned
   on the SAME extraction.  The diversity per sample is
   lower for B_cross.

**What W89 changes for Prong 2**:

* **Strip mode → `all_docstring`**: removes the prose
  description entirely; the prompt has ONLY the signature +
  "See image" stub.  This pushes A0_text near 0 % (no
  behavioural info) and forces A1_vlm and B_cross to rely
  EXCLUSIVELY on the image.  In this regime, the marginal
  value of correct image interpretation is much higher, and
  the code-LM's text-only failure becomes more visible.
* **VLM scale → Llama-3.2-90B-Vision-Instruct**: 90B is a
  larger VLM.  Both A1_vlm and B_cross's vision step use it.
  At 90B, the VLM's extraction quality should be higher,
  reducing handoff loss for B_cross.  If B_cross still
  loses, it's evidence that the SPLIT itself is the problem
  — not the VLM quality.

**Risk**: the unified 90B VLM may be SO strong at code given
image that A1_vlm scores 95 %+ and B_cross can't compete.
The `all_docstring` regime may also make A1_vlm WEAKER if
the VLM can't always read the doctest accurately — which
would be a B-favorable shift.  Honest empirical run will
disambiguate.

## What W88 did NOT show (carry-forwards W88 did NOT add)

* Multi-agent IS worse than independent sampling at all scales —
  W88 only showed this at 8B HumanEval-30 K=5.  Other scales,
  benchmarks, or budgets may differ.
* Cross-modal SPLITS are always worse than unified VLMs — W88
  only showed this at 11B-VLM + 8B-code-LM on HumanEval-Visual
  K=5 doctest_only.  Different decomposition (VLM-in-loop,
  parallel pool, deep cross-modal injection) or different
  benchmarks may differ.

W88's negative evidence is SCOPED to the specific configurations
tested.  W89 explicitly tests both axes the literature
suggests should move the needle (model scale; harder image-
load-bearing regime).

## What W89 explicitly does NOT do

* Does NOT change the B-pipeline shape (sequential reflexion
  for HumanEval; VLM-extract + code-LM-reflexion for
  cross-modal).  This isolates the model-scale / regime-shape
  hypothesis.
* Does NOT relax retirement bars.  Same bars as W88; same
  fair-budget discipline.
* Does NOT add a "stretch" V3 budget (K=10).  Budget stays at
  K=5 to maintain comparability with W86 / W88.

If both W89 prongs fail to retire, the carry-forwards stay and
the next wave (W90) must change something more fundamental —
likely the B architecture itself or the benchmark choice.
