# Success criterion — solving multi-agent context (SDK v3.9 bar)

> Pre-committed, falsifiable bar for what counts as a *real* advance
> on "solving multi-agent context" in the Context Zero / Wevra
> programme. This document is the **referee** for SDK v3.9 (and
> later milestones). Any milestone note that claims an advance must
> name the bar it cleared and cite the code-anchored evidence.
> Last touched: SDK v3.9, 2026-04-26.
>
> The history of this programme is full of moves where a partial
> result was written up too strongly and later had to be sharpened
> or retracted (W3-C7 strict, W4-C1 OOD, W6-C1, W6-C2). This bar
> is the dam against more of the same.

## TL;DR

A milestone *advances* the multi-agent-context thesis only if it
clears one of three pre-committed bars below. A milestone *does
not* advance the thesis if it produces a single cherry-picked
cell, regresses on a previously-shipped regime, or breaks the
lifecycle audit on any cell. Honest negative or null results are
acceptable — but they must be *labelled* as such, not framed as
advances.

The **named regimes** the bar refers to (anchored in code):

* **R-53** — Phase-53 real-LLM regime (low-surplus, single-service
  per scenario). FIFO unbeatable by W7-1.
  Anchor: `vision_mvp/experiments/phase53_scale_vs_structure.py`.
* **R-54** — Phase-54 deterministic gold-plurality regime
  (foreign-service decoys, gold tag in strict raw plurality).
  W7-2 win.
  Anchor:
  `vision_mvp/experiments/phase54_cross_role_coherence.py`.
* **R-55** — the *new harder regime* this milestone introduces.
  Must satisfy the four properties in § 3.1.

## 1. Three pre-committed bars

### 1.1 Strong success bar (a "real" advance)

A milestone *strongly advances* the thesis iff **all six** hold:

1. **Code anchor.** A new admission/decoder/coordination method
   ships in `vision_mvp/wevra/team_coord.py` (or sibling SDK
   module), exported from the SDK's public surface
   (`vision_mvp.wevra.__init__`), with a docstring that names its
   theorem family and falsifier.
2. **Strict gain on a hard regime.** The new method strictly
   improves `accuracy_full` over substrate FIFO **and** over the
   previous SDK's strongest method (e.g. SDK v3.8 buffered cohort)
   on **R-55**, by `≥ 0.20` pooled, on `n_eval ≥ 10` scenarios.
3. **Cross-bank stability.** The same gap `≥ 0.20` holds across
   `≥ 3` distinct `bank_seed` values, with no parameter retuning.
   Pre-commit the seed set in test code before running.
4. **No regression elsewhere.** On **R-53** and **R-54**, the new
   method does not regress `accuracy_full` by more than `0.05`
   pooled relative to the SDK v3.8 baseline.
5. **Audit + lifecycle preserved.** Every capsule strategy passes
   the team-lifecycle audit (T-1..T-7) on every cell of every
   regime. `audit_ok_grid[strategy] = True` for all capsule
   strategies on R-53 / R-54 / R-55.
6. **Named bench property + falsifier regime.** R-55's structural
   property is named in code (mechanically verified by a test in
   `test_wevra_*.py`) and at least **one** *falsifier regime* is
   explicitly identified — a regime where the bench property does
   not hold and the new method does *not* beat FIFO (i.e. the
   conditionality is sharp).

If a milestone clears 1.1, the canonical phrasing is:

> *"On R-55 (named bench property), method M strictly improves
> accuracy_full by ≥ 0.20 over both substrate FIFO and SDK v3.8
> buffered cohort (W7-2), stable across ≥ 3 bank seeds, with no
> regression on R-53 / R-54 and audit_ok preserved on every cell.
> The win does not transfer to the falsifier regime F (named)."*

### 1.2 Partial success bar (a real but narrower advance)

A milestone *partially advances* the thesis iff **all four** hold:

1. **Code anchor** (same as 1.1).
2. **Either-or gain.** Either:
   * a strict gain `≥ 0.20` on **R-55** but a regression `> 0.05`
     on R-53 or R-54 (a *trade-off* result), OR
   * a strict gain `≥ 0.10` on **two** of {R-53, R-54, R-55} with
     no regression `> 0.05` on the third.
3. **Audit preserved.** As in 1.1.5.
4. **One named theorem-empirical claim** with stated falsifier.

The canonical phrasing for partial success names the trade-off
explicitly: *"...wins R-55 by +X but trades R-54 by −Y; net
mean across the three regimes is +Z..."*. No "broad win" framing.

### 1.3 Falsifying failure bar (the milestone is null)

A milestone is a *falsifying failure* iff **any** holds:

1. The new method does not strictly improve any single regime by
   `≥ 0.10` and the cross-regime mean gain is `≤ 0.0`.
2. The new method introduces a regression `> 0.10` on R-53 or
   R-54 *and* the gain on R-55 does not exceed the regression
   magnitude.
3. The lifecycle audit breaks on any cell.
4. The W7-2 contract test
   (`test_buffered_cohort_strictly_beats_fifo_at_K4`) fails.

A null milestone is *not* a sin. The sin is calling a null
milestone an advance.

## 2. Required ingredients of R-55

The "harder fair regime" must combine:

* **Cross-role complementary evidence.** Each scenario's gold answer
  requires aggregating signals from `≥ 3` distinct producer roles.
  No single role is sufficient.
* **Budget pressure.** `|candidates_to_auditor| > K_auditor` on
  every scenario (mechanically tested).
* **Locally-misleading distractors.** For each role, the role's
  local plurality of `service=<tag>` mentions is a **decoy** — i.e.
  if the auditor naively picked the role-local plurality, it would
  pick the wrong service. The cross-role aggregate must point to
  the gold service. (This is the "cleanliness is misleading"
  property the SDK v3.9 brief asks for.)
* **Foreign-service decoys + decoy raw plurality.** The auditor
  sees `≥ 2` distinct decoy services. **Some decoy service must
  have strictly more raw mentions than the gold service** — so
  W7-2 buffered cohort (single-tag plurality) picks the *decoy*
  and ties FIFO at 0.000. This is the **strict separation** from
  the W7-2 win.
* **Cross-role corroboration of gold.** The gold service is
  mentioned by **strictly more distinct producer roles** than any
  decoy service. This is the signal the new method must be able to
  exploit.
* **Realistic claim_kinds.** The gold claims carry the *causal*
  claim_kinds (those that the priority decoder maps to the right
  root_cause); decoy claims carry plausible-but-not-causal
  claim_kinds.

A regime missing any of these is *not* R-55 — it does not test
the harder hypothesis.

## 3. What we are explicitly NOT testing

* **Not** "does cohort coherence ever beat FIFO?" — that's W7-2,
  already shipped and conditional.
* **Not** "does the learned policy generalise OOD?" — the W6-C2
  falsification in SDK v3.7 closed that question; learning is not
  the right tool here.
* **Not** "does scaling the LLM solve coordination?" — W6-C1
  closed that.
* **Not** "is the runtime fully capsule-native?" — see
  `HOW_NOT_TO_OVERSTATE.md`.

## 4. How to use this document

* Before a milestone starts: declare which bar (1.1 / 1.2 / 1.3)
  the milestone is targeting. If no bar is declared, the default
  is 1.1.
* During the milestone: do not relax the bar after seeing partial
  results. If the milestone ends up at 1.2 or 1.3, label it
  honestly.
* After the milestone: the milestone note must cite this document
  and state which bar was cleared. Reviewers reject milestone
  notes that overstate.

## Cross-references

* Theorem registry: `docs/THEOREM_REGISTRY.md`
* Research status: `docs/RESEARCH_STATUS.md`
* Overstatement guard: `docs/HOW_NOT_TO_OVERSTATE.md`
* Master plan: `docs/context_zero_master_plan.md` § 4.26 (SDK v3.9)
